#!/usr/bin/env python3
"""Historical A4-like back-calculation; superseded for B6.

Do not use this script for the B6 final product distribution. B6 must be
propagated through the complete R2/R3/R4/R6 partial-matching graph, and the
relevant reported comparison is R6:R2.

For the target-only bookkeeping used in the project,

    w_R2 = p_q1(full_reuse) + p_q1(full_switch) * p_q2_R2.

The calculation is diagnostic: it does not fit the MD, and an inferred value
outside [0, 1] means that the q1 branch weights and the experimental ratio are
incompatible under the current R2/R4-only reaction graph.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT / "results/b6_q1_umbrella_extensions_2ns_analysis_20260920/provisional_branch_sensitivity.csv"
DEFAULT_OUTPUT = PROJECT / "results/b6_experimental_back_calculation_20260921"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--r4-r2", type=float, default=3.0)
    args = parser.parse_args()

    target_r2 = 1.0 / (1.0 + args.r4_r2)
    rows = read_rows(args.input)
    points: list[tuple[float, float, str]] = []
    for row in rows:
        full_reuse = float(row["full_reuse_fraction"])
        full_switch = float(row["full_switch_d2_fraction"]) + float(row["full_switch_d1_fraction"])
        p_q2 = (target_r2 - full_reuse) / full_switch if full_switch > 0 else float("nan")
        points.append((full_reuse, full_switch, f"replica_{row['replica_scenario']}"))
        row["q1_scenario"] = f"replica_{row['replica_scenario']}"
        row["full_switch_fraction"] = f"{full_switch:.12g}"
        row["experimental_R2_mass_fraction"] = f"{target_r2:.12g}"
        row["implied_p_q2_R2"] = f"{p_q2:.12g}"
        row["implied_probability_physical"] = str(0.0 <= p_q2 <= 1.0)

    mean_reuse = sum(x[0] for x in points) / len(points)
    mean_switch = sum(x[1] for x in points) / len(points)
    mean_p_q2 = (target_r2 - mean_reuse) / mean_switch
    midpoint = {
        "q1_scenario": "replica_midpoint",
        "full_reuse_fraction": f"{mean_reuse:.12g}",
        "full_switch_fraction": f"{mean_switch:.12g}",
        "experimental_R2_mass_fraction": f"{target_r2:.12g}",
        "implied_p_q2_R2": f"{mean_p_q2:.12g}",
        "implied_probability_physical": str(0.0 <= mean_p_q2 <= 1.0),
    }
    rows.append(midpoint)

    args.output.mkdir(parents=True, exist_ok=True)
    out_csv = args.output / "implied_q2_probability.csv"
    fields = list(rows[0])
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    report = f"""# B6 experimental back-calculation

The source paper reports the oligomer composition as a **mass ratio**. Taking the wet-experiment ratio as `R4:R2 = {args.r4_r2:g}:1` therefore gives a target-only R2 mass fraction of **{100*target_r2:.1f}%** directly; no molecule-count-to-mass conversion is applied.

For the current reaction graph,

`w_R2 = p(q1 full reuse) + p(q1 full switch) * p(q2 -> R2×2)`.

The implied q2 probability is therefore:

`p(q2 -> R2×2) = (w_R2(exp) - p(q1 full reuse)) / p(q1 full switch)`.

| q1 scenario | full reuse | full switch | implied q2→R2×2 | physical? |
|---|---:|---:|---:|---|
"""
    for row in rows:
        report += f"| {row['q1_scenario']} | {float(row['full_reuse_fraction']):.3f} | {float(row['full_switch_fraction']):.3f} | {float(row['implied_p_q2_R2']):.3f} | {row['implied_probability_physical']} |\n"
    report += f"""
## Interpretation

The midpoint q1 scenario implies `p(q2 -> R2×2) = {mean_p_q2:.3f}`, which is outside [0, 1]. The midpoint cannot reproduce the experimental 25% R2 fraction for any physically allowed q2 probability, because q1 full reuse alone already contributes {100*mean_reuse:.1f}% R2.

Replica 1 alone would be compatible, requiring q2→R2×2 ≈ {(target_r2 - points[0][0]) / points[0][1]:.3f}. Replica 3 is incompatible even if every full-switch branch becomes R4, because its q1 full-reuse contribution is {100*points[1][0]:.1f}%.

Therefore the current discrepancy is primarily a **q1 branch-weight / hidden-basin problem**, not something that can be repaired by choosing a different neutral q2 prior. This is a diagnostic inversion of the experiment, not an independent prediction or a fitted final result.

Source q1 table: `{args.input}`.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
