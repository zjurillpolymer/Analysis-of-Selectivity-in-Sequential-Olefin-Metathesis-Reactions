#!/usr/bin/env python3
"""Analyze B6 q1 umbrella windows extended from 0.5 to 2.0 ns.

The primary interval is the newly generated 500--2000 ps segment.  Original
0--500 ps samples are deliberately excluded so that the result tests whether
the continued trajectories lose their initial-window memory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from analyze_a4_m_c6_c7_q1_umbrella_2d import prepare_topology
from analyze_a4_q2_umbrella_2d_reweight import (
    RT_KCAL,
    analyze_pathway,
    analyze_subset,
    write_csv,
)


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
RAW = PROJECT / "results/b6_q1_umbrella_extensions_2ns_results_20260920/raw"
OUTPUT = PROJECT / "results/b6_q1_umbrella_extensions_2ns_analysis_20260920"
RUNNER = (
    WORKSPACE
    / "bohrium_submit/b6_q1_2d_reseed_umbrella_20260919"
    / "b6_q1_2d_reseed_full_switch_d2_rep1/run_umbrella_path.py"
)

JOBS = {
    20759493: ("full_reuse", 1),
    20759495: ("full_reuse", 3),
    20759498: ("full_switch_d2", 1),
    20759494: ("full_switch_d2", 3),
    20759496: ("one_arm_reuse", 1),
    20759497: ("one_arm_reuse", 3),
}

OLD_NAC = {
    ("full_reuse", 1): 0.01007256727319321,
    ("full_reuse", 3): 0.00803863893624606,
    ("full_switch_d2", 1): 0.011675932317562634,
    ("full_switch_d2", 3): 0.0013112451977509345,
    ("one_arm_reuse", 1): 0.00011755705987265231,
    ("one_arm_reuse", 3): 0.00006169284118616867,
}

D1_NAC = {1: 1.6049203658057703e-06, 3: 3.456258201370154e-06}
MULTIPLICITY = {
    "full_reuse": 1,
    "full_switch_d2": 4,
    "one_arm_reuse": 4,
    "full_switch_d1": 8,
}


def numeric_xvg(path: Path) -> np.ndarray:
    rows = [
        [float(value) for value in line.split()]
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and line.lstrip()[0] not in "#@&"
    ]
    return np.asarray(rows, dtype=float)


def unique_glob(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"{root}: expected one {pattern}, found {len(matches)}")
    return matches[0]


def hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    os.link(source, target)


def prepare_pathway(raw: Path, prepared: Path) -> None:
    prepared.mkdir(parents=True, exist_ok=True)
    for name in ("system.top", "index.ndx", "windows.csv"):
        shutil.copy2(raw / name, prepared / name)
    metadata = json.loads((raw / "pathway.json").read_text(encoding="utf-8"))
    metadata["production_ps_per_window"] = 2000.0
    metadata["wham_begin_ps"] = 500.0
    metadata["wham_end_ps"] = 2000.0
    metadata["analysis_interval"] = "extension only: 500--2000 ps"
    (prepared / "pathway.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_windows = sorted((raw / "windows").glob("window_*"))
    if len(source_windows) != 25:
        raise ValueError(f"{raw}: expected 25 windows, found {len(source_windows)}")
    for source in source_windows:
        target = prepared / "windows" / source.name
        target.mkdir(parents=True, exist_ok=True)
        hardlink(source / "production_2ns.tpr", target / "production.tpr")
        hardlink(unique_glob(source, "production_ext.part*.xtc"), target / "production.xtc")
        hardlink(unique_glob(source, "production_ext.part*.gro"), target / "production.gro")
        hardlink(unique_glob(source, "production_ext_pullx.part*.xvg"), target / "production_pullx.xvg")
        hardlink(unique_glob(source, "production_ext_pullf.part*.xvg"), target / "production_pullf.xvg")


def run_wham(prepared: Path, gmx: str) -> None:
    pmf = prepared / "wham/pmf.xvg"
    diagnostics = prepared / "window_diagnostics.csv"
    if pmf.is_file() and diagnostics.is_file():
        return
    command = [
        sys.executable,
        str(RUNNER),
        "--pathway",
        str(prepared),
        "--gmx",
        gmx,
        "--stage",
        "wham",
        "--wham-begin-ps",
        "500",
    ]
    subprocess.run(command, check=True)


def pmf_audit(pathway: Path) -> dict[str, float | int]:
    with (pathway / "window_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        windows = list(csv.DictReader(handle))
    counts = [int(row["sample_count"]) for row in windows]
    if len(windows) != 25 or min(counts) != 3001 or max(counts) != 3001:
        raise ValueError(f"{pathway}: unexpected window/sample counts {len(windows)}, {counts}")
    overlap = [float(row["left_neighbor_gaussian_overlap"]) for row in windows[1:]]
    full = numeric_xvg(pathway / "wham/pmf.xvg")
    first = numeric_xvg(pathway / "wham/pmf_first_half.xvg")
    second = numeric_xvg(pathway / "wham/pmf_second_half.xvg")
    interior = (full[:, 0] >= 0.40) & (full[:, 0] <= 2.70)
    difference = first[interior, 1] - second[interior, 1]
    worst = int(np.argmin(overlap)) + 1
    return {
        "window_count": len(windows),
        "samples_per_window": min(counts),
        "minimum_adjacent_overlap": min(overlap),
        "minimum_overlap_right_window": windows[worst]["window_name"],
        "half_pmf_rms_kcal_mol": float(np.sqrt(np.mean(difference**2))),
        "half_pmf_max_abs_kcal_mol": float(np.max(np.abs(difference))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--gmx", default="gmx")
    args = parser.parse_args()
    raw_root = args.raw.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prepared_root = output / "prepared"
    topology_root = output / "reactive_pair_topologies"

    rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    for job_id, (orbit, replica) in JOBS.items():
        raw = raw_root / str(job_id) / "pathway"
        prepared = prepared_root / str(job_id) / "pathway"
        prepare_pathway(raw, prepared)
        run_wham(prepared, args.gmx)
        audit = pmf_audit(prepared)
        prepare_topology(prepared, topology_root)
        result = analyze_pathway(prepared, PROJECT, output, args.gmx, topology_root)
        blocks: dict[str, dict[str, float]] = {}
        for label, lower, upper in (
            ("block_500_1000", 500.0, 1000.0),
            ("block_1000_1500", 1000.0, 1500.0),
            ("block_1500_2000", 1500.0, 2000.0),
        ):
            select = (result.time_ps >= lower - 1e-6) & (result.time_ps <= upper + 1e-6)
            metrics, _ = analyze_subset(
                result.distance_nm[select], result.orientation_gate[select],
                result.centers_nm, result.window_index[select],
            )
            blocks[label] = metrics
            block_rows.append({
                "bohr_id": job_id, "orbit": orbit, "replica": replica,
                "block": label, "begin_ps": lower, "end_ps": upper, **metrics,
            })
        block_nac = np.asarray([value["soft_nac_probability"] for value in blocks.values()])
        old = OLD_NAC[(orbit, replica)]
        row: dict[str, object] = {
            "bohr_id": job_id,
            "pathway": result.metadata["pathway"],
            "orbit": orbit,
            "replica": replica,
            "orbit_multiplicity": result.metadata.get("orbit_multiplicity"),
            "frames": len(result.distance_nm),
            "effective_sample_size": result.effective_sample_size,
            "mbar_iterations": result.mbar_iterations,
            "mbar_residual": result.mbar_residual,
            "geometry_pull_rms_A": result.geometry_pull_rms_A,
            "geometry_pull_max_A": result.geometry_pull_max_A,
            **audit,
            **result.metrics,
            "old_0p5ns_soft_nac_probability": old,
            "new_over_old_nac": result.metrics["soft_nac_probability"] / old,
            "block_nac_min": float(np.min(block_nac)),
            "block_nac_max": float(np.max(block_nac)),
            "block_nac_max_over_min": float(np.max(block_nac) / np.min(block_nac)),
        }
        for label, metrics in blocks.items():
            row[f"{label}_soft_nac_probability"] = metrics["soft_nac_probability"]
            row[f"{label}_nac_free_energy_kcal_mol"] = metrics["nac_free_energy_kcal_mol"]
        rows.append(row)
        print(
            f"{job_id} {orbit} rep{replica}: NAC={row['soft_nac_probability']:.6g} "
            f"overlap={row['minimum_adjacent_overlap']:.3f} "
            f"block_ratio={row['block_nac_max_over_min']:.2f}",
            flush=True,
        )

    write_csv(output / "pathway_audit.csv", rows)
    write_csv(output / "block_metrics.csv", block_rows)
    (output / "pathway_audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    contrasts: list[dict[str, object]] = []
    for orbit in ("full_reuse", "full_switch_d2", "one_arm_reuse"):
        pair = sorted((row for row in rows if row["orbit"] == orbit), key=lambda row: int(row["replica"]))
        a, b = pair
        nac1 = float(a["soft_nac_probability"])
        nac3 = float(b["soft_nac_probability"])
        contact1 = float(a["soft_contact_probability"])
        contact3 = float(b["soft_contact_probability"])
        orientation1 = float(a["conditional_orientation"])
        orientation3 = float(b["conditional_orientation"])
        contrasts.append({
            "orbit": orbit,
            "rep1_nac": nac1,
            "rep3_nac": nac3,
            "larger_over_smaller_nac": max(nac1, nac3) / min(nac1, nac3),
            "absolute_ddG_nac_kcal_mol": RT_KCAL * abs(math.log(nac1 / nac3)),
            "rep1_contact": contact1,
            "rep3_contact": contact3,
            "larger_over_smaller_contact": max(contact1, contact3) / min(contact1, contact3),
            "rep1_orientation": orientation1,
            "rep3_orientation": orientation3,
            "larger_over_smaller_orientation": max(orientation1, orientation3) / min(orientation1, orientation3),
        })
    write_csv(output / "replica_contrasts.csv", contrasts)

    sensitivity: list[dict[str, object]] = []
    for replica in (1, 3):
        probabilities = {
            str(row["orbit"]): float(row["soft_nac_probability"])
            for row in rows if int(row["replica"]) == replica
        }
        probabilities["full_switch_d1"] = D1_NAC[replica]
        weights = {
            orbit: MULTIPLICITY[orbit] * probability
            for orbit, probability in probabilities.items()
        }
        total = sum(weights.values())
        sensitivity.append({
            "replica_scenario": replica,
            "normalization": total,
            **{f"{orbit}_weight": weights[orbit] for orbit in MULTIPLICITY},
            **{f"{orbit}_fraction": weights[orbit] / total for orbit in MULTIPLICITY},
            "reuse_fraction": (
                weights["full_reuse"] + weights["one_arm_reuse"]
            ) / total,
            "switch_fraction": (
                weights["full_switch_d1"] + weights["full_switch_d2"]
            ) / total,
        })
    write_csv(output / "provisional_branch_sensitivity.csv", sensitivity)

    lines = [
        "# B6 q1 umbrella extension to 2 ns/window",
        "",
        "## Scope",
        "",
        "All quantitative values below use only the newly generated 500--2000 ps segment. The original 0--500 ps segment is excluded from the primary estimator. Each path contains 25 windows and 3001 samples per window.",
        "",
        "## Per-path audit",
        "",
        "| orbit | rep | min overlap | half-PMF RMS | ESS | soft contact | orientation | soft NAC | three-block ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['orbit']} | {row['replica']} | {row['minimum_adjacent_overlap']:.3f} | "
            f"{row['half_pmf_rms_kcal_mol']:.3f} | {row['effective_sample_size']:.0f} | "
            f"{row['soft_contact_probability']:.5g} | {row['conditional_orientation']:.4f} | "
            f"{row['soft_nac_probability']:.5g} | {row['block_nac_max_over_min']:.2f} |"
        )
    lines += [
        "",
        "## Replica agreement",
        "",
        "| orbit | NAC ratio | |ΔΔG(NAC)| (kcal/mol) | contact ratio | orientation ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            f"| {row['orbit']} | {row['larger_over_smaller_nac']:.2f} | "
            f"{row['absolute_ddG_nac_kcal_mol']:.3f} | {row['larger_over_smaller_contact']:.2f} | "
            f"{row['larger_over_smaller_orientation']:.2f} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "`full_reuse` is the best-behaved class: replica NAC values agree within 1.12-fold (0.069 kcal/mol), both overlaps exceed 0.26, and both half-PMF RMS values are below 0.5 kcal/mol. Replica 1 still has a non-monotonic three-block spread of 2.66-fold, so this class is provisionally rather than absolutely converged.",
        "",
        "`full_switch_d2` remains unresolved. Its replica NAC ratio is 8.95-fold (1.364 kcal/mol), essentially unchanged from the 8.9-fold discrepancy in the 0.5 ns reseeded calculation. Both extended NAC estimates decreased by almost the same factor (~0.55), and the minimum window overlaps are now 0.313--0.323. Therefore additional one-dimensional distance sampling improved statistics but did not remove the basin identity. The missing coordinate is structural, not pair distance or the orientation gate alone.",
        "",
        "`one_arm_reuse` no longer has the former 0.120 overlap defect (the new minima are 0.325 and 0.384), but its replicas still differ by 2.96-fold (0.675 kcal/mol), and replica 3 drifts by 4.95-fold across the three 500 ps blocks. It is not quantitatively converged, although its multiplicity-weighted contribution remains close to 1% in both replica scenarios.",
        "",
        "## Provisional multiplicity sensitivity (not a final q1 estimate)",
        "",
        "Using `W_i = m_i P_i` with multiplicities 1, 4, 4 and 8 for full-reuse, full-switch-d2, one-arm-reuse and full-switch-d1, respectively, shows why the unresolved d2 basin cannot be averaged away:",
        "",
        "| scenario | full-reuse | full-switch d2 | one-arm reuse | full-switch d1 | reuse total | switch total |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sensitivity:
        lines.append(
            f"| rep{row['replica_scenario']} | {row['full_reuse_fraction']:.3%} | "
            f"{row['full_switch_d2_fraction']:.3%} | {row['one_arm_reuse_fraction']:.3%} | "
            f"{row['full_switch_d1_fraction']:.3%} | {row['reuse_fraction']:.3%} | "
            f"{row['switch_fraction']:.3%} |"
        )
    lines += [
        "",
        "The two internally consistent replica scenarios predict opposite q1 family preferences. Consequently these fractions are a sensitivity bound only; no final B6 q1 or product ratio should be reported from their average.",
        "",
        "## Next decision",
        "",
        "Do not extend the same one-dimensional d2 umbrella again. Compare matched-distance structures between d2 replicas, identify a discriminator from the remaining-arm contact map or core/arm torsions, and use that discriminator as the second coordinate of a targeted two-dimensional umbrella for full-switch d2. MetaD remains basin-discovery support; the two-dimensional umbrella supplies the quantitative branch weight.",
        "",
        "## Decision rules",
        "",
        "A path is treated as quantitatively converged only when adjacent-window overlap is >0.20, half-PMF RMS is <0.5 kcal/mol, three 500 ps NAC blocks show no material drift, and the two replicas differ by <0.5 kcal/mol. Failing paths remain diagnostic and are not averaged into final q1 weights.",
        "",
        "## Files",
        "",
        "- `pathway_audit.csv/json`: full 500--2000 ps path metrics and convergence diagnostics.",
        "- `block_metrics.csv`: independent 500 ps block estimates.",
        "- `replica_contrasts.csv`: rep1/rep3 comparisons.",
        "- `provisional_branch_sensitivity.csv`: multiplicity-weighted replica scenarios, explicitly not a final q1 estimate.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
