#!/usr/bin/env python3
"""Evaluate full-trajectory q0 contact and orientation gates for A4-m-C8.

Only the sixteen mapped vinyl carbon coordinates are extracted from each XTC.
All 32,004 frames are then evaluated, avoiding conclusions from a small set of
hand-picked snapshots.  The output decomposes the per-pair near-attack mass as

    NAC mass = contact accessibility x conditional orientation fraction.

This remains a catalyst-free configurational proxy, not a metathesis barrier.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import analyze_explicit_dcm_q0_geometry_gate as gate


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_vinyl_index(
    path: Path, mapping: Mapping[str, Sequence[int]]
) -> tuple[list[int], dict[str, list[int]]]:
    serials = sorted({int(serial) for values in mapping.values() for serial in values})
    lookup = {serial: index + 1 for index, serial in enumerate(serials)}
    local_mapping = {
        name: [lookup[int(serial)] for serial in values] for name, values in mapping.items()
    }
    path.write_text(
        "[ VinylAtoms ]\n" + " ".join(str(value) for value in serials) + "\n",
        encoding="utf-8",
    )
    return serials, local_mapping


def extract_vinyl_trajectory(
    gmx: Path, run_root: Path, replica: str, index: Path, output: Path
) -> None:
    command = [
        str(gmx), "trjconv",
        "-f", str(run_root / replica / "production_2ns.xtc"),
        "-s", str(run_root / replica / "production_2ns.tpr"),
        "-n", str(index),
        "-pbc", "whole",
        "-o", str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"GROMACS extraction failed for {replica}:\n{completed.stdout}\n{completed.stderr}"
        )


def parse_pdb_models(path: Path, atoms_per_frame: int) -> np.ndarray:
    frames: list[np.ndarray] = []
    current: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            current.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
        elif line.startswith("ENDMDL"):
            if len(current) != atoms_per_frame:
                raise ValueError(
                    f"{path} frame has {len(current)} atoms, expected {atoms_per_frame}"
                )
            frames.append(np.asarray(current, dtype=float))
            current = []
    if current:
        raise ValueError(f"unterminated PDB model in {path}")
    result = np.asarray(frames, dtype=float)
    if result.ndim != 3 or result.shape[1:] != (atoms_per_frame, 3):
        raise ValueError(f"invalid parsed trajectory shape: {result.shape}")
    return result


def read_distance_reference(path: Path, columns: Sequence[str]) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([[float(row[column]) for column in columns] for row in rows])


def extract_geometry(
    positions: np.ndarray,
    mapping: Mapping[str, Sequence[int]],
    pair_schema: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    n_frames = len(positions)
    n_pairs = len(pair_schema)
    distances = np.empty((n_frames, n_pairs), dtype=float)
    axes = np.empty_like(distances)
    sides = np.empty_like(distances)
    minimum_bond = math.inf
    maximum_bond = -math.inf
    for frame, coordinates in enumerate(positions):
        d, a, s, lengths = gate.pair_geometry(coordinates, mapping, pair_schema)
        distances[frame] = d
        axes[frame] = a
        sides[frame] = s
        minimum_bond = min(minimum_bond, float(np.min(lengths)))
        maximum_bond = max(maximum_bond, float(np.max(lengths)))
    return distances, axes, sides, minimum_bond, maximum_bond


def frame_components(
    distances: np.ndarray,
    axes: np.ndarray,
    sides: np.ndarray,
    same_mask: np.ndarray,
    cross_mask: np.ndarray,
    distance_center_A: float,
    distance_width_A: float,
    axis_max_deg: float,
    side_min_deg: float,
    angle_width_deg: float,
) -> np.ndarray:
    contact = gate.soft_upper(distances, distance_center_A, distance_width_A)
    orientation = gate.soft_upper(
        axes, axis_max_deg, angle_width_deg
    ) * gate.soft_lower(sides, side_min_deg, angle_width_deg)
    nac = contact * orientation
    return np.column_stack(
        (
            np.mean(contact[:, same_mask], axis=1),
            np.mean(contact[:, cross_mask], axis=1),
            np.mean(nac[:, same_mask], axis=1),
            np.mean(nac[:, cross_mask], axis=1),
        )
    )


def decompose(components: np.ndarray) -> dict[str, float]:
    same_contact, cross_contact, same_nac, cross_nac = np.asarray(components, dtype=float)
    same_orientation = same_nac / same_contact
    cross_orientation = cross_nac / cross_contact
    return {
        "same_contact": float(same_contact),
        "cross_contact": float(cross_contact),
        "contact_cross_over_same": float(cross_contact / same_contact),
        "same_conditional_orientation": float(same_orientation),
        "cross_conditional_orientation": float(cross_orientation),
        "orientation_cross_over_same": float(cross_orientation / same_orientation),
        "same_nac_mass": float(same_nac),
        "cross_nac_mass": float(cross_nac),
        "nac_cross_over_same": float(cross_nac / same_nac),
    }


def full_decomposition(replica_components: Sequence[np.ndarray]) -> dict[str, float]:
    replica_means = np.stack([np.mean(values, axis=0) for values in replica_components])
    return decompose(np.mean(replica_means, axis=0))


def hierarchical_bootstrap(
    replica_components: Sequence[np.ndarray],
    block_frames: int,
    samples: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    block_arrays = []
    for values in replica_components:
        count = len(values) // block_frames
        trimmed = values[: count * block_frames]
        block_arrays.append(trimmed.reshape(count, block_frames, 4).mean(axis=1))
    rng = np.random.default_rng(seed)
    keys = ("contact_cross_over_same", "orientation_cross_over_same", "nac_cross_over_same")
    results = {key: np.empty(samples, dtype=float) for key in keys}
    for sample in range(samples):
        selected_replicas = rng.integers(0, len(block_arrays), size=len(block_arrays))
        replica_means = []
        for index in selected_replicas:
            blocks = block_arrays[int(index)]
            selected_blocks = rng.integers(0, len(blocks), size=len(blocks))
            replica_means.append(np.mean(blocks[selected_blocks], axis=0))
        decomposition = decompose(np.mean(replica_means, axis=0))
        for key in keys:
            results[key][sample] = decomposition[key]
    return {
        key: tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))
        for key, values in results.items()
    }


def make_figure(
    output: Path,
    overall: Mapping[str, float],
    intervals: Mapping[str, Sequence[float]],
    replica_rows: Sequence[Mapping[str, object]],
    sensitivity: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.5), constrained_layout=True)
    keys = ("contact_cross_over_same", "orientation_cross_over_same", "nac_cross_over_same")
    labels = ("Contact", "Orientation | contact", "Combined NAC")
    values = np.asarray([overall[key] for key in keys])
    lower = values - np.asarray([intervals[key][0] for key in keys])
    upper = np.asarray([intervals[key][1] for key in keys]) - values
    axes[0].bar(labels, values, color=("#176B87", "#DAA520", "#C4473A"))
    axes[0].errorbar(labels, values, yerr=np.vstack((lower, upper)), fmt="none", ecolor="0.2", capsize=3)
    axes[0].axhline(1.0, color="0.4", lw=1, ls="--")
    axes[0].set_yscale("log")
    axes[0].set(ylabel="Cross/same per-pair ratio (log scale)", title="Configurational decomposition")
    axes[0].tick_params(axis="x", rotation=18)

    names = [str(row["replica"]).replace("replica_", "R") for row in replica_rows]
    x = np.arange(len(names))
    axes[1].bar(x - 0.18, [float(row["contact_cross_over_same"]) for row in replica_rows], width=0.36, label="Contact", color="#176B87")
    axes[1].bar(x + 0.18, [float(row["orientation_cross_over_same"]) for row in replica_rows], width=0.36, label="Orientation", color="#DAA520")
    axes[1].axhline(1.0, color="0.4", lw=1, ls="--")
    axes[1].set_yscale("log")
    axes[1].set(xticks=x, xticklabels=names, ylabel="Cross/same (log scale)", title="Independent replicas")
    axes[1].legend(frameon=False, fontsize=8)

    axes_values = sorted({float(row["axis_max_deg"]) for row in sensitivity})
    side_values = sorted({float(row["side_min_deg"]) for row in sensitivity})
    matrix = np.empty((len(side_values), len(axes_values)))
    for row in sensitivity:
        i = side_values.index(float(row["side_min_deg"]))
        j = axes_values.index(float(row["axis_max_deg"]))
        matrix[i, j] = float(row["nac_cross_over_same"])
    image = axes[2].imshow(matrix, origin="lower", cmap="RdYlBu_r", aspect="auto")
    axes[2].set(xticks=range(len(axes_values)), xticklabels=[f"{v:g}" for v in axes_values],
                yticks=range(len(side_values)), yticklabels=[f"{v:g}" for v in side_values],
                xlabel="Maximum axis angle (°)", ylabel="Minimum side angle (°)",
                title="Combined NAC sensitivity")
    for i in range(len(side_values)):
        for j in range(len(axes_values)):
            axes[2].text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axes[2], shrink=0.75, label="Cross/same")
    fig.savefig(output / "full_geometry_decomposition.png", dpi=240)
    fig.savefig(output / "full_geometry_decomposition.pdf")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=root / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs/A4_m_C8_q0",
    )
    parser.add_argument("--output", type=Path, default=root / "results/a4_m_c8_contact_origin")
    parser.add_argument("--gmx", type=Path, default=Path("/opt/homebrew/bin/gmx"))
    parser.add_argument("--block-ps", type=float, default=200.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--temperature-K", type=float, default=313.15)
    parser.add_argument("--refresh-cache", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    cache = output / "vinyl_coordinate_cache"
    cache.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run_root / "setup/metadata.json").read_text(encoding="utf-8"))
    mapping = metadata["vinyl_mapping_gromacs_1based"]
    pair_schema = list(metadata["pair_schema"])
    pair_columns = [str(row["column"]) for row in pair_schema]
    same_mask = np.asarray([row["topology"] == "same_arm" for row in pair_schema])
    cross_mask = ~same_mask
    serials, local_mapping = make_vinyl_index(cache / "vinyl_atoms.ndx", mapping)
    geometry_by_replica = {}
    quality_rows = []
    for index in range(1, 5):
        replica = f"replica_{index:02d}"
        pdb = cache / f"{replica}_vinyl_atoms.pdb"
        if args.refresh_cache or not pdb.is_file():
            extract_vinyl_trajectory(args.gmx.resolve(), run_root, replica, cache / "vinyl_atoms.ndx", pdb)
        positions = parse_pdb_models(pdb, len(serials))
        distances, axes, sides, minimum_bond, maximum_bond = extract_geometry(
            positions, local_mapping, pair_schema
        )
        reference = read_distance_reference(run_root / replica / "vinyl_distances.csv", pair_columns)
        if reference.shape != distances.shape:
            raise ValueError(f"frame/pair shape mismatch for {replica}: {reference.shape} vs {distances.shape}")
        max_error = float(np.max(np.abs(reference - distances)))
        if max_error > 0.03:
            raise ValueError(f"PDB/reference distance mismatch for {replica}: {max_error:.4f} Å")
        geometry_by_replica[replica] = (distances, axes, sides)
        quality_rows.append(
            {"replica": replica, "frames": len(positions), "maximum_distance_error_A": max_error,
             "minimum_vinyl_bond_A": minimum_bond, "maximum_vinyl_bond_A": maximum_bond}
        )
        print(f"{replica}: {len(positions)} frames [OK]", flush=True)

    primary_components = []
    replica_rows = []
    for replica, (distances, axes, sides) in geometry_by_replica.items():
        values = frame_components(
            distances, axes, sides, same_mask, cross_mask,
            5.0, 0.5, 45.0, 45.0, 7.5,
        )
        primary_components.append(values)
        replica_rows.append({"replica": replica, **decompose(np.mean(values, axis=0))})
    overall = full_decomposition(primary_components)
    block_frames = int(round(args.block_ps / 0.5))
    intervals = hierarchical_bootstrap(
        primary_components, block_frames, args.bootstrap_samples, args.seed
    )

    sensitivity = []
    for axis_max in (30.0, 45.0, 60.0):
        for side_min in (30.0, 45.0, 60.0):
            components = [
                frame_components(d, a, s, same_mask, cross_mask, 5.0, 0.5, axis_max, side_min, 7.5)
                for d, a, s in geometry_by_replica.values()
            ]
            sensitivity.append(
                {"axis_max_deg": axis_max, "side_min_deg": side_min, **full_decomposition(components)}
            )

    decomposition_fields = [
        "replica", "same_contact", "cross_contact", "contact_cross_over_same",
        "same_conditional_orientation", "cross_conditional_orientation",
        "orientation_cross_over_same", "same_nac_mass", "cross_nac_mass", "nac_cross_over_same",
    ]
    write_csv(output / "full_geometry_replica_decomposition.csv", replica_rows, decomposition_fields)
    write_csv(
        output / "full_geometry_gate_sensitivity.csv", sensitivity,
        ["axis_max_deg", "side_min_deg", *decomposition_fields[1:]],
    )
    write_csv(
        output / "full_geometry_quality_control.csv", quality_rows,
        ["replica", "frames", "maximum_distance_error_A", "minimum_vinyl_bond_A", "maximum_vinyl_bond_A"],
    )
    make_figure(output, overall, intervals, replica_rows, sensitivity)

    rt = gate.GAS_CONSTANT_KCAL_MOL_K * args.temperature_K
    summary = {
        "analysis": "full-trajectory A4-m-C8 q0 configurational decomposition",
        "frames": sum(len(values[0]) for values in geometry_by_replica.values()),
        "replicas": len(geometry_by_replica),
        "primary_gate": {"distance_center_A": 5.0, "distance_width_A": 0.5,
                         "axis_max_deg": 45.0, "side_min_deg": 45.0, "angle_width_deg": 7.5},
        "overall": overall,
        "ratio_95pct_CI": intervals,
        "free_energy_proxies_kcal_mol": {
            key.replace("_cross_over_same", "_ddG_cross_minus_same"):
                -rt * math.log(overall[key])
            for key in ("contact_cross_over_same", "orientation_cross_over_same", "nac_cross_over_same")
        },
        "bootstrap": {"block_ps": args.block_ps, "samples": args.bootstrap_samples, "seed": args.seed},
        "interpretation_boundary": "catalyst-free q0 contact/orientation proxy; not chemical activation barrier",
    }
    (output / "full_geometry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = [
        "# A4-m-C8 全轨迹接触—取向分解",
        "",
        f"四条显式 DCM 轨迹的全部 {summary['frames']:,} 帧均已纳入。主参数为软接触中心 5 Å、宽度 0.5 Å，"
        "取向门控为 C=C 轴夹角不高于 45°且侧向接近角不低于 45°。",
        "",
        "| 分量 | 跨臂/同臂 | 95% CI | ΔΔG(cross−same) / kcal mol⁻¹ |",
        "|---|---:|---:|---:|",
    ]
    for label, key in (
        ("接触可及性", "contact_cross_over_same"),
        ("给定接触后的取向", "orientation_cross_over_same"),
        ("接触×取向的总 NAC 质量", "nac_cross_over_same"),
    ):
        low, high = intervals[key]
        ddg = -rt * math.log(overall[key])
        report.append(f"| {label} | {overall[key]:.3f} | {low:.3f}–{high:.3f} | {ddg:.3f} |")
    report.extend(
        [
            "",
            "结果将来源拆开了：跨臂在进入接触区之前具有明显优势；进入相同接触区后，取向门控没有显示稳定的额外跨臂优势。"
            "二者相乘后得到总体近反应构象质量。",
            "",
            "取向项的区间若跨过 1，应解释为当前采样无法分辨同臂与跨臂的接触后取向差异，而不是证明二者严格相等。",
            "",
            "这里没有催化剂，也没有发生断键/成键，因此所有 ΔΔG 都是构象自由能代理量，而非烯烃复分解活化能垒。",
        ]
    )
    (output / "full_geometry_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
