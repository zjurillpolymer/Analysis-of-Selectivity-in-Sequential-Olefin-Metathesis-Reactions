#!/usr/bin/env python3
"""Audit completed B6 q1 umbrella paths and reweight distance/orientation.

The eight paths are independent endpoint-started replicas of four neighbour
orbits. This script reports per-path/replica sampling diagnostics; it does not
convert restrained equilibrium weights into reaction yields.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_a4_m_c6_c7_q1_umbrella_2d import prepare_topology
from analyze_a4_q2_umbrella_2d_reweight import analyze_pathway, write_csv


PROJECT = Path(__file__).resolve().parents[1]
DOWNLOAD = PROJECT.parent / "bohrium_download/b6_neighbor_q1_umbrella_20260918"
JOBS = (20752763, 20752764, 20752765, 20752766,
        20752767, 20752768, 20752770, 20752771)


def numeric_xvg(path: Path) -> np.ndarray:
    rows = [
        [float(value) for value in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and line.lstrip()[0] not in "#@&"
    ]
    return np.asarray(rows)


def wham_audit(pathway: Path) -> dict[str, float]:
    with (pathway / "window_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        windows = list(csv.DictReader(handle))
    if len(windows) != 25:
        raise ValueError(f"{pathway}: expected 25 windows, found {len(windows)}")
    if any(int(row["sample_count"]) != 801 for row in windows):
        raise ValueError(f"{pathway}: incomplete WHAM production samples")
    overlaps = [float(row["left_neighbor_gaussian_overlap"]) for row in windows[1:]]
    wham = pathway / "wham"
    full = numeric_xvg(wham / "pmf.xvg")
    first = numeric_xvg(wham / "pmf_first_half.xvg")
    second = numeric_xvg(wham / "pmf_second_half.xvg")
    if not (np.isfinite(full).all() and np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError(f"{pathway}: non-finite PMF values")
    if not (np.allclose(full[:, 0], first[:, 0]) and np.allclose(full[:, 0], second[:, 0])):
        raise ValueError(f"{pathway}: mismatched PMF grids")
    interior = (full[:, 0] >= 0.40) & (full[:, 0] <= 2.70)
    difference = first[interior, 1] - second[interior, 1]
    return {
        "minimum_adjacent_gaussian_overlap": min(overlaps),
        "pmf_half_rms_kcal_mol": float(np.sqrt(np.mean(difference**2))),
        "pmf_half_max_abs_kcal_mol": float(np.max(np.abs(difference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, default=DOWNLOAD)
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "results/b6_neighbor_q1_umbrella_20260918/analysis")
    parser.add_argument("--gmx", default="gmx")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology_root = output / "reactive_pair_topologies"
    rows: list[dict[str, object]] = []
    for job_id in JOBS:
        pathway = args.download.resolve() / str(job_id) / "pathway"
        metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
        audit = wham_audit(pathway)
        prepare_topology(pathway, topology_root)
        result = analyze_pathway(pathway, PROJECT, output, args.gmx, topology_root)
        row: dict[str, object] = {
            "bohr_id": job_id,
            "pathway": metadata["pathway"],
            "orbit": metadata["route_family"],
            "pair": f"V{metadata['left']}-V{metadata['right']}",
            "replica": metadata["replica"],
            "orbit_multiplicity": metadata["orbit_multiplicity"],
            "windows": len(result.centers_nm),
            "frames": len(result.distance_nm),
            "mbar_iterations": result.mbar_iterations,
            "mbar_residual": result.mbar_residual,
            "effective_sample_size": result.effective_sample_size,
            "geometry_pull_rms_A": result.geometry_pull_rms_A,
            "geometry_pull_max_A": result.geometry_pull_max_A,
            **audit,
            **result.metrics,
        }
        for half, metrics in result.half_metrics.items():
            row.update({f"{half}_{key}": value for key, value in metrics.items()})
        rows.append(row)
        write_csv(output / "pathway_audit.csv", rows)
        print(f"{job_id} {row['orbit']} rep{row['replica']} "
              f"NAC={row['soft_nac_probability']:.6g} "
              f"ESS={row['effective_sample_size']:.1f}", flush=True)
    (output / "pathway_audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
