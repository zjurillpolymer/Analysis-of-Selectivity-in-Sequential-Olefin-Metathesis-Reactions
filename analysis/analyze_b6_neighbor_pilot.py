#!/usr/bin/env python3
"""Audit five downloaded B6 neighbour-only MD pilots and summarize encounters.

These are descriptive, correlated-trajectory encounter metrics, not reaction
probabilities or a substitute for the planned umbrella free energies.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from build_b6_neighbor_orbit_table import cycle_edges, load_arm_map


JOBS = {
    "B6_p_C6_q1_V1_V2_cis": 20748984,
    "B6_p_C6_q2_neighbor_full_reuse_cis": 20749009,
    "B6_p_C6_q2_neighbor_one_arm_reuse_cis": 20749047,
    "B6_p_C6_q2_neighbor_full_switch_d1_cis": 20749048,
    "B6_p_C6_q2_neighbor_full_switch_d2_cis": 20749051,
}
EXTENSION_JOBS = {
    "B6_p_C6_q1_V1_V2_cis": 20750428,
    "B6_p_C6_q2_neighbor_full_reuse_cis": 20750429,
    "B6_p_C6_q2_neighbor_one_arm_reuse_cis": 20750430,
    "B6_p_C6_q2_neighbor_full_switch_d1_cis": 20750431,
    "B6_p_C6_q2_neighbor_full_switch_d2_cis": 20750432,
}
CONTACT_A = 5.0


def soft_contact(distance: np.ndarray) -> np.ndarray:
    z = np.clip((distance - 5.0) / 0.5, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(z))


def read_manifest(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one manifest row: {path}")
    return rows[0]


def read_q1_orbits(path: Path) -> dict[tuple[int, int], str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        pair = tuple(int(label.removeprefix("V")) for label in row["q1_pair"].split("-"))
        result[pair] = row["category"]
    if len(result) != 17:
        raise ValueError(f"expected 17 q1 candidates, found {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--project", type=Path, default=project)
    parser.add_argument("--download", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extension", action="store_true",
                        help="analyze the 2 -> 10 ns checkpoint-extension archives")
    args = parser.parse_args()
    project = args.project.resolve()
    jobs = EXTENSION_JOBS if args.extension else JOBS
    download_default = project.parent / (
        "bohrium_download/b6_neighbor_md_extensions_2to10ns_20260918"
        if args.extension else "bohrium_download/b6_neighbor_md_pilot_20260917"
    )
    output_default = project / (
        "results/b6_neighbor_md_extension_20260918/neighbor_analysis"
        if args.extension else "results/b6_neighbor_md_pilot_20260917/neighbor_analysis"
    )
    download = (args.download or download_default).resolve()
    output = (args.output or output_default).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arm_of = load_arm_map(project)
    neighbors = cycle_edges()
    q1_orbits = read_q1_orbits(project / "results/b6_neighbor_orbit_table/q1_neighbor_orbits.csv")

    quality_rows = []
    pair_rows = []
    orbit_rows = []
    for state, job in jobs.items():
        workspace = download / str(job) / "extracted/workspace"
        manifest = read_manifest(workspace / "gromacs_dcm/state_manifest.csv")
        if manifest["state"] != state:
            raise ValueError(f"job {job}: manifest state {manifest['state']} != {state}")
        schema = json.loads(manifest["pair_schema_json"])
        permitted = {}
        for pair in schema:
            left, right = int(pair["left"]), int(pair["right"])
            edge = tuple(sorted((arm_of[left], arm_of[right])))
            if edge in neighbors:
                permitted[(left, right)] = pair
        if state.startswith("B6_p_C6_q1") and set(permitted) != set(q1_orbits):
            raise ValueError("q1 neighbour-filtered manifest differs from 17-candidate orbit table")
        if not permitted:
            raise ValueError(f"{state}: no permitted neighbour pairs")

        for replica in range(1, 4):
            repdir = workspace / manifest["output_dir"] / f"replica_{replica:02d}"
            for suffix in ("xtc", "gro", "tpr", "log"):
                path = repdir / f"production_2ns.{suffix}"
                if not path.is_file() or path.stat().st_size == 0:
                    raise ValueError(f"missing/empty production output: {path}")
            if "Finished mdrun" not in (repdir / "production_2ns.log").read_text(errors="replace"):
                raise ValueError(f"production log lacks normal completion: {repdir}")
            data = pd.read_csv(repdir / "vinyl_distances.csv")
            times = data["time_ps"].to_numpy(float)
            expected_duration = 10000.0 if args.extension else 2000.0
            if (len(times) < 2 or abs(times[0]) > 1e-5
                    or not np.allclose(np.diff(times), 0.5, atol=1e-5)
                    or abs(times[-1] - expected_duration) > 1e-5):
                raise ValueError(f"bad 0–{expected_duration:g} ps, 0.5 ps trajectory grid: {repdir}")
            all_columns = [item["column"] for item in schema]
            if list(data.columns[3:]) != all_columns:
                raise ValueError(f"distance CSV/manifest columns differ: {repdir}")
            values = data[all_columns].to_numpy(float)
            if not np.isfinite(values).all() or np.min(values) <= 0:
                raise ValueError(f"nonfinite/nonpositive distance: {repdir}")
            quality_rows.append({"state": state, "job_id": job, "replica": replica,
                                 "frames": len(data), "duration_ps": times[-1],
                                 "all_pair_count": len(schema), "neighbor_pair_count": len(permitted),
                                 "status": "pass"})

            by_orbit: dict[str, list[np.ndarray]] = defaultdict(list)
            for (left, right), item in permitted.items():
                distance = data[item["column"]].to_numpy(float)
                soft = soft_contact(distance)
                category = q1_orbits[(left, right)] if state.startswith("B6_p_C6_q1") else "q2_neighbor"
                by_orbit[category].append(distance)
                pair_rows.append({
                    "state": state, "replica": replica, "left": left, "right": right,
                    "pair": f"V{left}-V{right}", "arm_edge": "-".join(
                        f"A{value}" for value in sorted((arm_of[left], arm_of[right]))),
                    "category": category, "soft_q_mean": float(soft.mean()),
                    "hard_5A_frames": int(np.count_nonzero(distance < CONTACT_A)),
                    "hard_5A_fraction": float(np.mean(distance < CONTACT_A)),
                    "hard_6A_fraction": float(np.mean(distance < 6.0)),
                    "distance_min_A": float(distance.min()),
                    "distance_median_A": float(np.median(distance)),
                    "first_half_soft_q": float(soft[:2000].mean()),
                    "second_half_soft_q": float(soft[2001:].mean()),
                })
            for category, arrays in by_orbit.items():
                matrix = np.stack(arrays, axis=1)
                soft_matrix = soft_contact(matrix)
                orbit_rows.append({
                    "state": state, "replica": replica, "category": category,
                    "candidate_count": matrix.shape[1],
                    "sum_pair_soft_q": float(soft_matrix.mean(axis=0).sum()),
                    "mean_pair_soft_q": float(soft_matrix.mean()),
                    "any_pair_5A_frames": int(np.count_nonzero(np.min(matrix, axis=1) < CONTACT_A)),
                    "any_pair_5A_fraction": float(np.mean(np.min(matrix, axis=1) < CONTACT_A)),
                    "any_pair_6A_fraction": float(np.mean(np.min(matrix, axis=1) < 6.0)),
                    "minimum_distance_A": float(matrix.min()),
                    "first_half_sum_soft_q": float(soft_matrix[:2000].mean(axis=0).sum()),
                    "second_half_sum_soft_q": float(soft_matrix[2001:].mean(axis=0).sum()),
                })

    quality = pd.DataFrame(quality_rows)
    pairs = pd.DataFrame(pair_rows)
    orbits = pd.DataFrame(orbit_rows)
    quality.to_csv(output / "trajectory_quality.csv", index=False)
    pairs.to_csv(output / "neighbor_pair_by_replica.csv", index=False)
    orbits.to_csv(output / "neighbor_orbit_by_replica.csv", index=False)
    q1 = orbits[orbits["state"] == "B6_p_C6_q1_V1_V2_cis"]
    q1_summary = q1.groupby("category", as_index=False).agg(
        candidate_count=("candidate_count", "first"),
        sum_pair_soft_q_mean=("sum_pair_soft_q", "mean"),
        sum_pair_soft_q_min=("sum_pair_soft_q", "min"),
        sum_pair_soft_q_max=("sum_pair_soft_q", "max"),
        mean_pair_soft_q=("mean_pair_soft_q", "mean"),
        any_pair_5A_fraction_mean=("any_pair_5A_fraction", "mean"),
        any_pair_5A_frames_total=("any_pair_5A_frames", "sum"),
        first_half_sum_soft_q=("first_half_sum_soft_q", "mean"),
        second_half_sum_soft_q=("second_half_sum_soft_q", "mean"),
    )
    q1_summary["relative_soft_score_descriptive"] = (
        q1_summary["sum_pair_soft_q_mean"] / q1_summary["sum_pair_soft_q_mean"].sum()
    )
    q1_summary.to_csv(output / "q1_orbit_pilot_summary.csv", index=False)
    q2 = pairs[pairs["state"] != "B6_p_C6_q1_V1_V2_cis"]
    q2_summary = q2.groupby(["state", "pair", "arm_edge"], as_index=False).agg(
        soft_q_mean=("soft_q_mean", "mean"),
        soft_q_replica_min=("soft_q_mean", "min"),
        soft_q_replica_max=("soft_q_mean", "max"),
        hard_5A_frames_total=("hard_5A_frames", "sum"),
        distance_min_A=("distance_min_A", "min"),
    )
    q2_summary.sort_values(["state", "soft_q_mean"], ascending=[True, False]).to_csv(
        output / "q2_neighbor_pair_pilot_summary.csv", index=False)
    summary = {
        "status": "pilot_descriptive_only",
        "states": list(jobs), "job_ids": jobs,
        "trajectory_count": len(quality), "frames_per_trajectory": 4001,
        "duration_ns_per_trajectory": 2.0,
        "contact_threshold_A": CONTACT_A,
        "soft_contact_center_A": 5.0, "soft_contact_width_A": 0.5,
        "q1_orbit_candidate_counts": dict(zip(q1_summary.category, q1_summary.candidate_count.astype(int))),
        "cautions": [
            "Frames within a trajectory are correlated; three replicas are the uncertainty units.",
            "Orbit soft-score shares are geometric encounter descriptors, not reaction probabilities.",
            "q2 states are separately initialized conditional structures and cannot be pooled as if visited by q1 trajectories.",
            "Umbrella/PMF work remains necessary for interpretable free-energy-based selectivity.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(q1_summary.to_string(index=False))
    print(f"Audited {len(quality)} trajectories; output: {output}")


if __name__ == "__main__":
    main()
