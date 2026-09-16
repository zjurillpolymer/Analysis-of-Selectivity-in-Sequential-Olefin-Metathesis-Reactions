#!/usr/bin/env python3
"""Extract distance-matched A4-m-C8 q0 frames and evaluate vinyl geometry.

Candidates are selected upstream using midpoint distance only.  This script
extracts the solute from the corresponding XTC frames with GROMACS, makes the
solute molecule whole, and evaluates a catalyst-free near-attack orientation
proxy: parallel C=C axes plus side-on midpoint approach.
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

import analyze_explicit_dcm_q0_geometry_gate as geometry


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_pdb_positions(path: Path) -> np.ndarray:
    positions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            positions.append(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"could not parse PDB coordinates: {path}")
    return array


def extract_frame(
    gmx: Path,
    run_root: Path,
    setup_index: Path,
    replica: str,
    time_ps: float,
    output: Path,
) -> None:
    replica_root = run_root / replica
    command = [
        str(gmx), "convert-trj",
        "-f", str(replica_root / "production_2ns.xtc"),
        "-s", str(replica_root / "production_2ns.tpr"),
        "-n", str(setup_index),
        "-b", f"{time_ps:.6f}",
        "-e", f"{time_ps:.6f}",
        "-select", 'group "Solute"',
        "-o", str(output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"GROMACS extraction failed for {replica} at {time_ps:g} ps:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )


def geometric_mean(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if np.any(array <= 0.0) or not np.all(np.isfinite(array)):
        raise ValueError("geometric mean requires finite positive values")
    return float(np.exp(np.mean(np.log(array))))


def paired_bootstrap(ratios: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    logs = np.log(np.asarray(ratios, dtype=float))
    indices = rng.integers(0, len(logs), size=(samples, len(logs)))
    estimates = np.exp(np.mean(logs[indices], axis=1))
    low, high = np.quantile(estimates, (0.025, 0.975))
    return float(low), float(high)


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=root / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs/A4_m_C8_q0",
    )
    parser.add_argument("--analysis-root", type=Path, default=root / "results/a4_m_c8_contact_origin")
    parser.add_argument("--gmx", type=Path, default=Path("/opt/homebrew/bin/gmx"))
    parser.add_argument("--distance-center-A", type=float, default=5.0)
    parser.add_argument("--distance-width-A", type=float, default=0.5)
    parser.add_argument("--axis-max-deg", type=float, default=45.0)
    parser.add_argument("--side-min-deg", type=float, default=45.0)
    parser.add_argument("--angle-width-deg", type=float, default=7.5)
    parser.add_argument("--temperature-K", type=float, default=313.15)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    analysis_root = args.analysis_root.resolve()
    structures = analysis_root / "matched_candidate_structures"
    structures.mkdir(parents=True, exist_ok=True)
    rows = read_csv(analysis_root / "matched_candidate_frames.csv")
    metadata = json.loads((run_root / "setup/metadata.json").read_text(encoding="utf-8"))
    mapping = metadata["vinyl_mapping_gromacs_1based"]
    pair_schema = list(metadata["pair_schema"])
    pair_index_by_name = {
        str(pair["column"]).rsplit("_", 1)[0]: index
        for index, pair in enumerate(pair_schema)
    }
    output_rows: list[dict[str, object]] = []
    for row in rows:
        pair_id = row["matched_pair_id"]
        topology = row["topology"]
        sample_id = f"{pair_id}_{topology}"
        pdb = structures / f"{sample_id}.pdb"
        extract_frame(
            args.gmx.resolve(), run_root, run_root / "setup/index.ndx",
            row["replica"], float(row["minimum_time_ps"]), pdb,
        )
        positions = parse_pdb_positions(pdb)
        if len(positions) != int(metadata["source_atom_count"]):
            raise ValueError(f"unexpected solute atom count in {pdb}: {len(positions)}")
        distances, axes, sides, lengths = geometry.pair_geometry(
            positions, mapping, pair_schema
        )
        pair_index = pair_index_by_name[row["minimum_pair"]]
        distance = float(distances[pair_index])
        if not math.isclose(distance, float(row["minimum_distance_A"]), abs_tol=0.003):
            raise ValueError(
                f"distance mismatch for {sample_id}: CSV={row['minimum_distance_A']} PDB={distance}"
            )
        contact = float(
            geometry.soft_upper(
                np.asarray([distance]), args.distance_center_A, args.distance_width_A
            )[0]
        )
        axis_gate = float(
            geometry.soft_upper(
                np.asarray([axes[pair_index]]), args.axis_max_deg, args.angle_width_deg
            )[0]
        )
        side_gate = float(
            geometry.soft_lower(
                np.asarray([sides[pair_index]]), args.side_min_deg, args.angle_width_deg
            )[0]
        )
        orientation = axis_gate * side_gate
        output_rows.append(
            {
                "sample_id": sample_id,
                "matched_pair_id": pair_id,
                "topology": topology,
                "replica": row["replica"],
                "time_ps": row["minimum_time_ps"],
                "pair": row["minimum_pair"],
                "midpoint_distance_A": distance,
                "axis_angle_deg": float(axes[pair_index]),
                "side_angle_deg": float(sides[pair_index]),
                "soft_contact": contact,
                "axis_gate": axis_gate,
                "side_gate": side_gate,
                "orientation_gate": orientation,
                "near_attack_score": contact * orientation,
                "minimum_vinyl_bond_A": float(np.min(lengths)),
                "maximum_vinyl_bond_A": float(np.max(lengths)),
                "structure_pdb": str(pdb.relative_to(analysis_root.parent.parent)),
            }
        )

    fields = [
        "sample_id", "matched_pair_id", "topology", "replica", "time_ps", "pair",
        "midpoint_distance_A", "axis_angle_deg", "side_angle_deg", "soft_contact",
        "axis_gate", "side_gate", "orientation_gate", "near_attack_score",
        "minimum_vinyl_bond_A", "maximum_vinyl_bond_A", "structure_pdb",
    ]
    write_csv(analysis_root / "matched_candidate_geometry.csv", output_rows, fields)

    grouped: dict[str, dict[str, Mapping[str, object]]] = {}
    for row in output_rows:
        grouped.setdefault(str(row["matched_pair_id"]), {})[str(row["topology"])] = row
    comparison_rows = []
    orientation_ratios = []
    score_ratios = []
    for pair_id in sorted(grouped):
        same = grouped[pair_id]["same_arm"]
        cross = grouped[pair_id]["cross_arm"]
        orientation_ratio = float(cross["orientation_gate"]) / float(same["orientation_gate"])
        score_ratio = float(cross["near_attack_score"]) / float(same["near_attack_score"])
        orientation_ratios.append(orientation_ratio)
        score_ratios.append(score_ratio)
        comparison_rows.append(
            {
                "matched_pair_id": pair_id,
                "distance_mismatch_A": abs(float(cross["midpoint_distance_A"]) - float(same["midpoint_distance_A"])),
                "same_axis_angle_deg": same["axis_angle_deg"],
                "cross_axis_angle_deg": cross["axis_angle_deg"],
                "same_side_angle_deg": same["side_angle_deg"],
                "cross_side_angle_deg": cross["side_angle_deg"],
                "same_orientation_gate": same["orientation_gate"],
                "cross_orientation_gate": cross["orientation_gate"],
                "cross_over_same_orientation": orientation_ratio,
                "same_near_attack_score": same["near_attack_score"],
                "cross_near_attack_score": cross["near_attack_score"],
                "cross_over_same_near_attack": score_ratio,
            }
        )
    comparison_fields = [
        "matched_pair_id", "distance_mismatch_A", "same_axis_angle_deg", "cross_axis_angle_deg",
        "same_side_angle_deg", "cross_side_angle_deg", "same_orientation_gate",
        "cross_orientation_gate", "cross_over_same_orientation", "same_near_attack_score",
        "cross_near_attack_score", "cross_over_same_near_attack",
    ]
    write_csv(analysis_root / "matched_geometry_comparison.csv", comparison_rows, comparison_fields)

    orientation_ratios_array = np.asarray(orientation_ratios)
    score_ratios_array = np.asarray(score_ratios)
    orientation_gm = geometric_mean(orientation_ratios_array)
    score_gm = geometric_mean(score_ratios_array)
    orientation_ci = paired_bootstrap(
        orientation_ratios_array, args.bootstrap_samples, args.seed
    )
    score_ci = paired_bootstrap(score_ratios_array, args.bootstrap_samples, args.seed + 1)
    rt = geometry.GAS_CONSTANT_KCAL_MOL_K * args.temperature_K
    summary = {
        "analysis": "distance-matched catalyst-free q0 orientation gate",
        "matched_pairs": len(grouped),
        "matching_variable": "vinyl midpoint distance only",
        "orientation_proxy": "parallel C=C axes times side-on midpoint approach",
        "cross_over_same_orientation_geometric_mean": orientation_gm,
        "orientation_ratio_95pct_CI": orientation_ci,
        "orientation_ddG_cross_minus_same_kcal_mol": -rt * math.log(orientation_gm),
        "cross_over_same_near_attack_geometric_mean": score_gm,
        "near_attack_ratio_95pct_CI": score_ci,
        "near_attack_ddG_cross_minus_same_kcal_mol": -rt * math.log(score_gm),
        "gate_settings": {
            "distance_center_A": args.distance_center_A,
            "distance_width_A": args.distance_width_A,
            "axis_max_deg": args.axis_max_deg,
            "side_min_deg": args.side_min_deg,
            "angle_width_deg": args.angle_width_deg,
        },
        "interpretation_boundary": "catalyst-free geometry proxy, not a metathesis activation barrier",
    }
    (analysis_root / "matched_geometry_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
