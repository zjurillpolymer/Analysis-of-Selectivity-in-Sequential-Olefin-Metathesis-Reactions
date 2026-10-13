#!/usr/bin/env python3
"""Historical A4-like bookkeeping test; superseded for B6.

Do not use this script for the B6 final product distribution. B6 must be
propagated through the complete R2/R3/R4/R6 partial-matching graph.

This is deliberately a bookkeeping model, not a new equilibrium estimator.
The q1 input is the existing multiplicity-weighted sensitivity table.  The
full-switch q2 branch is not yet a converged conditional probability, so it is
represented by an explicit effective p_q2_R2 sweep.  This keeps the unresolved
part visible instead of silently turning a geometric/q2 pilot descriptor into
a reaction probability.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]
DEFAULT_INPUT = PROJECT / "results/b6_q1_umbrella_extensions_2ns_analysis_20260920/provisional_branch_sensitivity.csv"
DEFAULT_OUTPUT = PROJECT / "results/b6_provisional_yield_prediction_20260921"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def q1_values(row: dict[str, str]) -> tuple[float, float, float]:
    full_reuse = float(row["full_reuse_fraction"])
    full_switch = float(row["full_switch_d2_fraction"]) + float(row["full_switch_d1_fraction"])
    one_arm_reuse = float(row["one_arm_reuse_fraction"])
    total = full_reuse + full_switch + one_arm_reuse
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"q1 target fractions do not sum to one: {total}")
    return full_reuse, full_switch, one_arm_reuse


def propagate(full_reuse: float, full_switch: float, one_arm_reuse: float, p_q2_r2: float) -> tuple[float, float, float, float]:
    """Return (R2 molecule count, R4 molecule count, R2 mass fraction, R4 mass fraction).

    full_reuse gives two R2 molecules.  one_arm_reuse gives one R4 molecule.
    A full-switch q1 branch gives two R2 molecules with probability p_q2_r2,
    otherwise one R4 molecule.  The denominator counts R4 twice because an R4
    product consumes two R2-equivalent reaction events in this target-only
    bookkeeping, matching the A4 propagation convention used in the project.
    """
    r2_count = 2.0 * full_reuse + 2.0 * full_switch * p_q2_r2
    r4_count = one_arm_reuse + full_switch * (1.0 - p_q2_r2)
    total = r2_count + 2.0 * r4_count
    return r2_count, r4_count, r2_count / total, (2.0 * r4_count) / total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = read_rows(args.input)
    scenarios: list[dict[str, object]] = []
    q1_points: list[tuple[float, float, float]] = []
    for row in rows:
        fr, fs, fo = q1_values(row)
        q1_points.append((fr, fs, fo))
        scenarios.append({
            "q1_scenario": f"replica_{row['replica_scenario']}",
            "full_reuse": fr,
            "full_switch": fs,
            "one_arm_reuse": fo,
        })

    midpoint = tuple(sum(point[index] for point in q1_points) / len(q1_points) for index in range(3))
    scenarios.append({
        "q1_scenario": "replica_midpoint_diagnostic",
        "full_reuse": midpoint[0],
        "full_switch": midpoint[1],
        "one_arm_reuse": midpoint[2],
    })

    prediction_rows: list[dict[str, object]] = []
    p_grid = (0.0, 0.25, 0.50, 0.75, 1.0)
    for scenario in scenarios:
        fr = float(scenario["full_reuse"])
        fs = float(scenario["full_switch"])
        fo = float(scenario["one_arm_reuse"])
        for p_q2_r2 in p_grid:
            r2, r4, p_r2, p_r4 = propagate(fr, fs, fo, p_q2_r2)
            prediction_rows.append({
                "q1_scenario": scenario["q1_scenario"],
                "p_q2_R2": p_q2_r2,
                "full_reuse_q1": fr,
                "full_switch_q1": fs,
                "one_arm_reuse_q1": fo,
                "R2_count": r2,
                "R4_count": r4,
                "R2_mass_fraction": p_r2,
                "R4_mass_fraction": p_r4,
            })

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "q1_input_scenarios.csv", scenarios)
    write_csv(args.output / "r2_r4_sensitivity.csv", prediction_rows)

    fr, fs, fo = midpoint
    r2, r4, p_r2, p_r4 = propagate(fr, fs, fo, 0.5)
    report = f"""# Provisional B6 R2/R4 yield prediction

## Working estimate

Using the midpoint of the two internally consistent q1 umbrella scenarios and a neutral effective q2 split (`p_q2_R2 = 0.50`), the target-only prediction is:

- **R2: {100*p_r2:.1f}% by mass**
- **R4: {100*p_r4:.1f}% by mass**

The corresponding bookkeeping counts are `{r2:.4f}` R2 molecules and `{r4:.4f}` R4 molecules per normalized q1 event. R2 is counted twice when q1 full-reuse or q2 R2 completion produces two R2 molecules.

## What is measured and what is assumed

The q1 input is the existing 2-ns/window multiplicity-weighted umbrella sensitivity table. The two replicas disagree strongly because the B6 full-switch-d2 basin remains seed-dependent; therefore their midpoint is a **working central value**, not an equilibrium estimator.

The q2 pilot does not yet provide a conditional R2-versus-R4 reaction probability. It provides geometry/encounter descriptors from separately initialized q2 structures. We therefore expose that missing quantity as `p_q2_R2` rather than converting it silently.

The target-only mapping is:

1. q1 full-reuse -> R2 x2;
2. q1 one-arm-reuse -> R4;
3. q1 full-switch -> q2, with probability `p_q2_R2` giving R2 x2 and the complementary probability giving R4.

## Sensitivity

For the midpoint q1 scenario, the predicted R2/R4 mass fractions are 42.4/57.6% at `p_q2_R2=0`, 70.7/29.3% at 0.5, and 99.1/0.9% at 1.0. The replica-specific 0.5 cases are 58.3/41.7% (replica 1) and 83.2/16.8% (replica 3).

These ranges are the honest current result. The 70.7/29.3% number is suitable as a provisional prediction for comparison with experiment, but should be labeled as such until full-switch q2 is sampled conditionally and the d2 hidden basin is resolved.

Source q1 table: `{args.input}`.
"""
    (args.output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
