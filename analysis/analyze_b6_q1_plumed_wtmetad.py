#!/usr/bin/env python3
"""Audit B6 q1 Rg WT-MetaD as basin discovery, not selectivity evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
N_SOLUTE = 438
VINYLS = {vinyl: np.array([183 + 2 * vinyl, 184 + 2 * vinyl]) - 1 for vinyl in range(3, 13)}
TRI = np.triu_indices(10, 1)
FEATURES = (
    "time_ps", "rg_nm", "kappa2", "distance_nm", "axis_deg", "side_deg",
    "soft_contact", "soft_nac", "max_bond_nm",
)
RUNS = (
    "b6_q1_full_switch_d2_rep1",
    "b6_q1_full_switch_d2_rep3",
    "b6_q1_one_arm_reuse_rep1",
    "b6_q1_one_arm_reuse_rep3",
)
UMBRELLA_JOBS = (20752763, 20752764, 20752765, 20752766, 20752767, 20752768, 20752770, 20752771)


def numeric(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments="#", ndmin=2)


def topology(path: Path) -> tuple[list[tuple[int, int]], list[str]]:
    section = ""
    molecule = 0
    bonds: list[tuple[int, int]] = []
    names: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            section = line.strip("[] ").strip()
            if section == "moleculetype":
                molecule += 1
            continue
        if molecule != 1:
            continue
        fields = line.split()
        if section == "atoms":
            names.append(fields[4])
        elif section in ("bonds", "constraints"):
            bonds.append((int(fields[0]) - 1, int(fields[1]) - 1))
    if len(names) != N_SOLUTE:
        raise ValueError(f"expected {N_SOLUTE} solute atoms, got {len(names)}")
    return sorted(set(bonds)), names


def soft_upper(values: np.ndarray | float, center: float, width: float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(np.clip((values - center) / width, -700, 700)))


def extract(
    xtc: Path,
    gro: Path,
    bonds: list[tuple[int, int]],
    names: list[str],
    pair: tuple[int, int],
    cache: Path,
    stride: int = 1,
    begin_ps: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    token = f"{xtc.resolve()}|{xtc.stat().st_size}|{xtc.stat().st_mtime_ns}|{pair}|{stride}|{begin_ps}"
    key = hashlib.sha256(token.encode()).hexdigest()[:24]
    destination = cache / f"{key}.npz"
    if destination.is_file():
        with np.load(destination) as source:
            return source["data"], source["fingerprint"]
    original = mda.Universe(str(gro))
    if original.atoms[:N_SOLUTE].names.tolist() != names:
        raise ValueError(f"solute order mismatch: {gro}")
    with XTCReader(str(xtc)) as reader:
        atom_count = reader.n_atoms
    if atom_count != N_SOLUTE:
        raise ValueError(f"expected solute-only XTC with {N_SOLUTE} atoms: {xtc}, got {atom_count}")
    universe = mda.Merge(original.atoms[:N_SOLUTE])
    universe.add_TopologyAttr("bonds", bonds)
    universe.load_new(str(xtc))
    if len(universe.atoms.fragments) != 1:
        raise ValueError("solute bond graph is not one connected fragment")
    bi, bj = np.asarray(bonds).T
    records: list[list[float]] = []
    fingerprints: list[np.ndarray] = []
    left_index, right_index = pair
    for frame_index, ts in enumerate(universe.trajectory):
        if frame_index % stride or ts.time < begin_ps - 1.0e-4:
            continue
        positions = universe.atoms.unwrap(compound="fragments", reference=None, inplace=False).astype(float) / 10.0
        centered = positions - positions.mean(axis=0)
        eigen = np.linalg.eigvalsh(centered.T @ centered / N_SOLUTE)
        rg = math.sqrt(float(eigen.sum()))
        kappa2 = 1.0 - 3.0 * float(eigen[0] * eigen[1] + eigen[0] * eigen[2] + eigen[1] * eigen[2]) / float(eigen.sum() ** 2)
        ends = np.asarray([positions[VINYLS[vinyl]] for vinyl in range(3, 13)])
        midpoints = ends.mean(axis=1)
        axes = ends[:, 1] - ends[:, 0]
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        left = left_index - 3
        right = right_index - 3
        connector = midpoints[right] - midpoints[left]
        distance = float(np.linalg.norm(connector))
        axis_angle = math.degrees(math.acos(np.clip(abs(float(axes[left] @ axes[right])), 0, 1)))
        side_angle = min(
            math.degrees(math.acos(np.clip(abs(float(axes[index] @ connector)) / distance, 0, 1)))
            for index in (left, right)
        )
        contact = float(soft_upper(distance, 0.50, 0.05))
        orientation = float(soft_upper(axis_angle, 45.0, 7.5)) * (1.0 - float(soft_upper(side_angle, 45.0, 7.5)))
        max_bond = float(np.max(np.linalg.norm(positions[bi] - positions[bj], axis=1)))
        records.append([ts.time, rg, kappa2, distance, axis_angle, side_angle, contact, contact * orientation, max_bond])
        fingerprints.append(np.linalg.norm(midpoints[TRI[0]] - midpoints[TRI[1]], axis=1))
    data = np.asarray(records)
    fingerprint = np.asarray(fingerprints)
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, data=data, fingerprint=fingerprint)
    return data, fingerprint


def visits(time: np.ndarray, selector: np.ndarray) -> list[dict[str, float | int]]:
    blocks: list[dict[str, float | int]] = []
    start: int | None = None
    for index, active in enumerate(np.r_[selector, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            blocks.append({
                "start_ps": float(time[start]),
                "end_ps": float(time[index - 1]),
                "duration_ps": float(time[index - 1] - time[start]),
                "frames": index - start,
            })
            start = None
    return sorted(blocks, key=lambda row: float(row["duration_ps"]), reverse=True)


def synopsis(data: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "frames": len(data), "start_ps": float(data[0, 0]), "end_ps": float(data[-1, 0]),
    }
    for column, name in ((1, "rg_nm"), (2, "kappa2"), (3, "distance_nm")):
        result[name] = dict(zip(
            ("min", "q01", "median", "q99", "max"),
            map(float, np.quantile(data[:, column], (0, 0.01, 0.5, 0.99, 1))),
        ))
    result.update({
        "mean_soft_contact_geometry": float(data[:, 6].mean()),
        "mean_soft_nac_geometry": float(data[:, 7].mean()),
        "hard_contact_frames": int((data[:, 3] <= 0.5).sum()),
        "hard_nac_frames": int(((data[:, 3] <= 0.5) & (data[:, 4] <= 45) & (data[:, 5] >= 45)).sum()),
        "max_bond_nm": float(data[:, 8].max()),
    })
    return result


def histogram_overlap(left: np.ndarray, right: np.ndarray, edges: np.ndarray) -> float:
    a = np.histogram(left, edges)[0].astype(float)
    b = np.histogram(right, edges)[0].astype(float)
    a /= a.sum()
    b /= b.sum()
    return float(np.minimum(a, b).sum())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(output: Path, meta: dict[str, np.ndarray], reference: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    colors = {RUNS[0]: "#4477AA", RUNS[1]: "#66CCEE", RUNS[2]: "#CC6677", RUNS[3]: "#AA3377"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), layout="constrained")
    for name, data in meta.items():
        axes[0, 0].plot(data[:, 0] / 1000, data[:, 1], lw=0.55, color=colors[name], label=name.replace("b6_q1_", ""))
    axes[0, 0].set(xlabel="Time (ns)", ylabel="Rg (nm)", title="a  Biased global-CV exploration")
    axes[0, 0].legend(fontsize=5.5, ncol=2)
    axes[0, 1].scatter(reference[::4, 1], reference[::4, 2], s=1, c="0.75", rasterized=True, label="MD + umbrella reference")
    for name, data in meta.items():
        axes[0, 1].scatter(data[:, 1], data[:, 2], s=2, alpha=0.35, color=colors[name], rasterized=True)
    axes[0, 1].set(xlabel="Rg (nm)", ylabel="Relative shape anisotropy", title="b  Global-shape coverage")
    axes[0, 1].legend(fontsize=6)
    for name, data in meta.items():
        axes[1, 0].scatter(data[:, 1], data[:, 3], s=2, alpha=0.35, color=colors[name], rasterized=True)
    axes[1, 0].axhline(0.5, color="0.2", ls="--", lw=0.8)
    axes[1, 0].set(xlabel="Rg (nm)", ylabel="Monitored pair distance (nm)", title="c  Pair distance was not biased")
    for name, data in meta.items():
        axes[1, 1].hist(data[:, 3], bins=np.arange(0.3, 3.71, 0.05), density=True, histtype="step", lw=1.0, color=colors[name], label=name.replace("b6_q1_", ""))
    axes[1, 1].axvline(0.5, color="0.2", ls="--", lw=0.8)
    axes[1, 1].set(xlabel="Monitored pair distance (nm)", ylabel="Biased density", title="d  Replica dependence remains")
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"b6_q1_wtmetad_sampling_diagnostics.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/b6_q1_wtmetad_analysis_20260919")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    raw = workspace / "results/b6_q1_wtmetad_results_20260919/raw/b6_q1_targeted_wtmetad_server_20260918"
    bonds, names = topology(raw / RUNS[0] / "system.top")

    meta: dict[str, np.ndarray] = {}
    fingerprints: dict[str, np.ndarray] = {}
    audits: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for name in RUNS:
        run = raw / name
        metadata = json.loads((run / "pathway.json").read_text(encoding="utf-8"))
        pair = (int(metadata["left"]), int(metadata["right"]))
        data, fp = extract(run / "wtmeta.xtc", run / "start.gro", bonds, names, pair, cache)
        cv = numeric(run / "COLVAR")
        indices = np.searchsorted(cv[:, 0], data[:, 0])
        if np.any(indices >= len(cv)) or not np.allclose(cv[indices, 0], data[:, 0], atol=1.0e-4):
            raise ValueError(f"COLVAR/XTC time mismatch: {name}")
        hills = numeric(run / "HILLS")
        log = (run / "wtmeta.log").read_text(encoding="utf-8", errors="replace")
        hard_visits = visits(cv[:, 0], cv[:, 2] <= 0.5)
        audit = {
            "orbit": metadata["route_family"], "replica": metadata["replica"], "pair": f"V{pair[0]}-V{pair[1]}",
            "finished": bool(re.search(r"Finished mdrun on rank 0", log)), "lincs_warning_count": log.count("LINCS WARNING"),
            "colvar_frames": len(cv), "trajectory_frames": len(data), "duration_ns": float(cv[-1, 0] / 1000),
            "hills": len(hills), "first_100_hill_height_kJ_mol": float(hills[:100, 3].mean()),
            "last_1000_hill_height_kJ_mol": float(hills[-1000:, 3].mean()),
            "hill_height_ratio_late_early": float(hills[-1000:, 3].mean() / hills[:100, 3].mean()),
            "rg_colvar_rms_error_nm": float(np.sqrt(np.mean((data[:, 1] - cv[indices, 1]) ** 2))),
            "pair_colvar_rms_error_nm": float(np.sqrt(np.mean((data[:, 3] - cv[indices, 2]) ** 2))),
            "colvar_hard_contact_fraction": float(np.mean(cv[:, 2] <= 0.5)),
            "colvar_soft_contact_mean": float(np.mean(soft_upper(cv[:, 2], 0.5, 0.05))),
            "hard_contact_visits": hard_visits[:10], "hard_contact_visits_total": len(hard_visits),
            "hard_contact_visits_ge_10ps": sum(float(row["duration_ps"]) >= 10 for row in hard_visits),
            "full": synopsis(data), "post_2ns": synopsis(data[data[:, 0] >= 2000]),
        }
        meta[name], fingerprints[name], audits[name] = data, fp, audit
        rows.append({
            "run": name, "orbit": audit["orbit"], "replica": audit["replica"], "pair": audit["pair"],
            "duration_ns": audit["duration_ns"], "hills": audit["hills"],
            "rg_min_nm": float(data[:, 1].min()), "rg_median_nm": float(np.median(data[:, 1])), "rg_max_nm": float(data[:, 1].max()),
            "pair_min_nm": float(cv[:, 2].min()), "pair_median_nm": float(np.median(cv[:, 2])), "pair_max_nm": float(cv[:, 2].max()),
            "hard_contact_fraction": audit["colvar_hard_contact_fraction"], "soft_contact_mean": audit["colvar_soft_contact_mean"],
            "hill_height_ratio_late_early": audit["hill_height_ratio_late_early"],
        })
        print(f"MetaD {name}: {len(data)} trajectory frames, {len(hills)} hills", flush=True)

    reference_data: list[np.ndarray] = []
    reference_fp: list[np.ndarray] = []
    md_fp: dict[int, np.ndarray] = {}
    md_root = workspace / "bohrium_download/b6_neighbor_md_extensions_2to10ns_20260918/20750428/extracted/workspace/gromacs_dcm/runs/B6_p_C6_q1_V1_V2_cis"
    for replica in (1, 2, 3):
        rep = md_root / f"replica_{replica:02d}"
        data, fp = extract(rep / "production_2ns.xtc", rep / "production_2ns.gro", bonds, names, (6, 8), cache, stride=10)
        reference_data.append(data)
        reference_fp.append(fp)
        md_fp[replica] = fp
        print(f"Reference MD rep{replica}: {len(data)} frames", flush=True)
    umbrella_root = workspace / "bohrium_download/b6_neighbor_q1_umbrella_20260918"
    for job in UMBRELLA_JOBS:
        pathway = umbrella_root / str(job) / "pathway"
        metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
        pair = (int(metadata["left"]), int(metadata["right"]))
        for window in sorted((pathway / "windows").glob("window_*")):
            data, fp = extract(window / "production.xtc", pathway / "start.gro", bonds, names, pair, cache, stride=10, begin_ps=100)
            reference_data.append(data)
            reference_fp.append(fp)
        print(f"Reference umbrella job {job}: 25 windows", flush=True)
    reference = np.concatenate(reference_data)
    reference_fingerprint = np.concatenate(reference_fp)

    calibration: list[float] = []
    for replica in (1, 2, 3):
        held = md_fp[replica]
        train = np.concatenate([fp for key, fp in md_fp.items() if key != replica])
        calibration.extend((cKDTree(train[::2]).query(held[::5], workers=2)[0] / math.sqrt(45)).tolist())
    novelty_threshold = float(np.quantile(calibration, 0.99))
    tree = cKDTree(reference_fingerprint[::2])
    novel_masks: dict[str, np.ndarray] = {}
    for name in RUNS:
        distances = tree.query(fingerprints[name], workers=2)[0] / math.sqrt(45)
        novel = distances > novelty_threshold
        novel_masks[name] = novel
        blocks = visits(meta[name][:, 0], novel)
        audits[name]["fingerprint_novelty"] = {
            "heldout_md_q99_threshold_nm": novelty_threshold,
            "nn_rms_nm_median": float(np.median(distances)), "nn_rms_nm_q99": float(np.quantile(distances, 0.99)),
            "nn_rms_nm_max": float(distances.max()), "novel_frame_fraction": float(novel.mean()),
            "longest_novel_visit_ps": float(blocks[0]["duration_ps"]) if blocks else 0.0,
            "visits_ge_100ps": sum(float(row["duration_ps"]) >= 100 for row in blocks),
        }
        np.savez_compressed(output / f"{name}_source_data.npz", data=meta[name], fingerprint=fingerprints[name], features=np.asarray(FEATURES))

    novelty_overlap: dict[str, dict[str, dict[str, float]]] = {}
    for source in RUNS:
        novelty_overlap[source] = {}
        source_fp = fingerprints[source][novel_masks[source]]
        for target in RUNS:
            if source == target:
                continue
            target_fp = fingerprints[target][novel_masks[target]]
            if not len(source_fp) or not len(target_fp):
                median = math.inf
                fraction = 0.0
            else:
                distances = cKDTree(target_fp).query(source_fp, workers=2)[0] / math.sqrt(45)
                median = float(np.median(distances))
                fraction = float(np.mean(distances <= novelty_threshold))
            novelty_overlap[source][target] = {
                "median_nearest_novel_fingerprint_rms_nm": median,
                "fraction_within_heldout_md_q99_threshold": fraction,
            }

    rg_edges = np.arange(0.80, 1.31, 0.01)
    distance_edges = np.arange(0.30, 3.71, 0.05)
    comparisons: dict[str, dict[str, float]] = {}
    for family, first, second in (
        ("full_switch_d2", RUNS[0], RUNS[1]),
        ("one_arm_reuse", RUNS[2], RUNS[3]),
    ):
        left, right = meta[first], meta[second]
        comparisons[family] = {
            "rg_histogram_overlap": histogram_overlap(left[:, 1], right[:, 1], rg_edges),
            "distance_histogram_overlap": histogram_overlap(left[:, 3], right[:, 3], distance_edges),
            "rep1_rg_distance_pearson": float(np.corrcoef(left[:, 1], left[:, 3])[0, 1]),
            "rep3_rg_distance_pearson": float(np.corrcoef(right[:, 1], right[:, 3])[0, 1]),
            "soft_contact_ratio_max_over_min": float(max(audits[first]["colvar_soft_contact_mean"], audits[second]["colvar_soft_contact_mean"]) / max(min(audits[first]["colvar_soft_contact_mean"], audits[second]["colvar_soft_contact_mean"]), np.finfo(float).tiny)),
        }

    # Conservative Rg/kappa2 coverage test against all MD and umbrella frames.
    rg2_edges = np.arange(0.78, 1.321, 0.015)
    k2_edges = np.arange(0.0, 0.501, 0.015)
    reference_hist = np.histogram2d(reference[:, 1], reference[:, 2], bins=(rg2_edges, k2_edges))[0] > 0
    buffered = binary_dilation(reference_hist, structure=np.ones((3, 3)))
    for name, data in meta.items():
        i = np.searchsorted(rg2_edges, data[:, 1], side="right") - 1
        j = np.searchsorted(k2_edges, data[:, 2], side="right") - 1
        valid = (i >= 0) & (j >= 0) & (i < buffered.shape[0]) & (j < buffered.shape[1])
        novel = np.ones(len(data), dtype=bool)
        novel[valid] = ~buffered[i[valid], j[valid]]
        blocks = visits(data[:, 0], novel)
        audits[name]["rg_kappa2_novelty"] = {
            "buffered_novel_frame_fraction": float(novel.mean()),
            "longest_novel_visit_ps": float(blocks[0]["duration_ps"]) if blocks else 0.0,
            "visits_ge_100ps": sum(float(row["duration_ps"]) >= 100 for row in blocks),
        }

    summary = {
        "system": "B6_p_C6_q1_V1_V2_cis", "scope": "supplemental basin-discovery audit only",
        "actual_bias_cv": "equal-atom-weight solute Rg", "not_biased": ["reactive-pair distance", "orientation", "kappa2"],
        "runs": audits, "same_orbit_replica_comparisons": comparisons,
        "novel_fingerprint_cross_run_overlap": novelty_overlap,
        "reference_frames": len(reference), "reference_fingerprint_frames": len(reference_fingerprint),
        "fingerprint_heldout_md_q99_nm": novelty_threshold,
        "primary_umbrella_result_not_modified": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "run_audit.csv", rows)
    np.savez_compressed(output / "reference_coverage.npz", data=reference, fingerprint=reference_fingerprint, features=np.asarray(FEATURES))
    plot(output, meta, reference)
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
