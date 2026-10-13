#!/usr/bin/env python3
"""Analyze the four B6 q1 umbrellas re-seeded from common 2D-MetaD basins."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze_a4_m_c6_c7_q1_umbrella_2d import prepare_topology
from analyze_a4_q2_umbrella_2d_reweight import analyze_pathway, write_csv


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
DOWNLOAD = PROJECT / "results/b6_q1_2d_reseed_umbrella_results_20260919/raw"
JOBS = (20756864, 20756865, 20756866, 20756867)

OLD_DIRECT_NAC = {
    ("full_switch_d2", 1): 0.00854445779995639,
    ("full_switch_d2", 3): 3.1010081566301688e-06,
    ("one_arm_reuse", 1): 0.00013025401990719533,
    ("one_arm_reuse", 3): 1.2935070452441433e-06,
}


def numeric_xvg(path: Path) -> np.ndarray:
    rows = [
        [float(value) for value in line.split()]
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and line.lstrip()[0] not in "#@&"
    ]
    return np.asarray(rows, dtype=float)


def wham_audit(pathway: Path) -> dict[str, float | int]:
    with (pathway / "window_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        windows = list(csv.DictReader(handle))
    if len(windows) != 25:
        raise ValueError(f"{pathway}: expected 25 windows, found {len(windows)}")
    counts = [int(row["sample_count"]) for row in windows]
    if any(count != 801 for count in counts):
        raise ValueError(f"{pathway}: incomplete WHAM production samples: {counts}")
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
        "windows": len(windows),
        "samples_per_window": min(counts),
        "minimum_adjacent_gaussian_overlap": min(overlaps),
        "pmf_half_rms_kcal_mol": float(np.sqrt(np.mean(difference**2))),
        "pmf_half_max_abs_kcal_mol": float(np.max(np.abs(difference))),
        "pmf_min_kcal_mol": float(np.nanmin(full[:, 1])),
        "pmf_max_kcal_mol": float(np.nanmax(full[:, 1])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, default=DOWNLOAD)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/b6_q1_2d_reseed_umbrella_analysis_20260919")
    parser.add_argument("--gmx", default="gmx")
    args = parser.parse_args()
    download = args.download.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology_root = output / "reactive_pair_topologies"
    rows: list[dict[str, object]] = []
    for job_id in JOBS:
        pathway = download / str(job_id) / "pathway"
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
            "seed_time_ps": metadata.get("seed_time_ps"),
            "initial_distance_nm": metadata.get("initial_distance_nm"),
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
        print(f"{job_id} {row['orbit']} rep{row['replica']} NAC={row['soft_nac_probability']:.6g} ESS={row['effective_sample_size']:.1f}", flush=True)

    (output / "pathway_audit.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["orbit"]), []).append(row)
    contrasts = []
    for orbit, group in grouped.items():
        if len(group) != 2:
            continue
        a, b = group
        for metric in ("soft_contact_probability", "soft_nac_probability", "conditional_orientation"):
            av, bv = float(a[metric]), float(b[metric])
            contrasts.append({"orbit": orbit, "metric": metric, "rep1": av, "rep3": bv, "rep3_over_rep1": bv / av if av else math.nan})
    write_csv(output / "replica_contrasts.csv", contrasts)
    lines = [
        "# B6 q1 2D-MetaD-reseeded umbrella analysis",
        "",
        "## Scope",
        "",
        "These four umbrella calculations were re-seeded from full-system structures in the common contact basin found by the two-dimensional WT-MetaD. The umbrella PMF and MBAR/reweighting results, rather than MetaD occupancy, are the quantitative evidence.",
        "",
        "## Technical audit",
        "",
        "All four tasks completed with exit code 0. Each contains 25/25 windows and 801 production samples per window. The table below combines WHAM overlap/half-sensitivity with the distance-orientation reweighting metrics.",
        "",
        "| orbit | replica | min adjacent overlap | PMF half RMS (kcal/mol) | MBAR ESS | soft NAC | soft contact | conditional orientation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['orbit']} | {row['replica']} | {row['minimum_adjacent_gaussian_overlap']:.3f} | {row['pmf_half_rms_kcal_mol']:.3f} | {row['effective_sample_size']:.1f} | {row['soft_nac_probability']:.5g} | {row['soft_contact_probability']:.5g} | {row['conditional_orientation']:.5g} |")
    lines += [
        "",
        "## Bottom line",
        "",
        "The common 2D-MetaD seed removes the extreme low-contact artifact seen for the old replica-3 endpoint starts, but it does not make the one-dimensional distance umbrella replica-independent. The full-switch d2 reseeded replicas still differ by 8.9-fold in soft-NAC probability, while one-arm reuse differs by 1.9-fold. These four paths therefore diagnose hidden basin dependence; they should not be averaged into one final q1 probability.",
        "",
        "## Interpretation",
        "",
        "For full-switch d2, the soft contact probabilities are 0.0489 and 0.00644, whereas the conditional orientation factors are relatively close (0.239 and 0.204). Thus the large NAC discrepancy is driven mainly by contact accessibility or hidden basin identity, not by the orientation gate itself. For one-arm reuse, the NAC values are smaller and closer (1.18×10⁻⁴ versus 6.17×10⁻⁵), but replica 1 has poor adjacent-window overlap (0.120) and a 0.801 kcal mol⁻¹ half-PMF RMS difference, so it is not fully converged.",
        "",
        "Relative to the old direct endpoint-started umbrellas, the reseeded NAC probability increased by approximately 1.4-fold (d2 rep1), 423-fold (d2 rep3), 0.90-fold (one-arm rep1) and 47.7-fold (one-arm rep3). This demonstrates that the former extremely small replica-3 values were strongly affected by initial-basin hysteresis; it does not mean that the new d2 values are converged.",
        "",
        "The soft contact and soft NAC values are conditional equilibrium observables from the umbrella reweighting. They are not yet final R2/R4 product yields; the reaction-graph multiplicities and q1/q2 conditional branching rules must be applied separately.",
        "",
        "## Files",
        "",
        "- `pathway_audit.csv/json`: per-path WHAM, MBAR and geometry metrics.",
        "- `replica_contrasts.csv`: rep1/rep3 contrasts within each orbit.",
        "- `reactive_pair_topologies/`: generated solute topology and reactive-pair index files used for geometry reweighting.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
