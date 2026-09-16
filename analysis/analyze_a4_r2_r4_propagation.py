#!/usr/bin/env python3
"""Propagate the A4 q1/q2 NAC weights into an R2/R4 target-only model.

The bookkeeping follows the reaction graph rather than the old binary
reuse/switch labels:

* q1 full reuse (same reacted arm pair) -> two R2 products;
* q1 partial reuse (one reacted + one untouched arm) -> q2 adjacent;
* q1 full switch (two untouched arms) -> q2 opposite;
* q2 opposite has three final perfect matchings, with the same-arm-pair
  matching yielding two R2 products and the other two matchings yielding R4.

All weights are the primary-setting, per-pair soft-NAC values from the
existing explicit-DCM trajectories.  Same-arm/non-target leakage is reported
but excluded from the conditional R2/R4 prediction, as requested for the
paper's target-product comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY_SETTING = "d5_axis45_side45"
STATES = ("C6", "C7")
Q1_CATEGORIES = ("reuse_arm_pair", "one_reacted_arm", "untouched_arm_pair", "same_arm")
Q1_TARGET = Q1_CATEGORIES[:3]
PAIR_RE = re.compile(r"V(\d+)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_pair(name: str) -> tuple[int, int]:
    values = tuple(int(value) for value in PAIR_RE.findall(name))
    if len(values) != 2:
        raise ValueError(f"cannot parse vinyl pair: {name}")
    return values


def union_sizes(edges: Iterable[tuple[int, int]], n_arms: int = 4) -> tuple[int, ...]:
    parent = list(range(n_arms))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in edges:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left
    return tuple(sorted(Counter(find(index) for index in range(n_arms)).values()))


def q2_outcome(
    state: str,
    pair_name: str,
    q2_mapping: dict[str, dict[int, int]],
    active_by_state: dict[str, tuple[int, ...]],
    q2_qc: dict[str, dict[str, str]],
) -> str:
    mapping = q2_mapping[state]
    active = list(active_by_state[state])
    pair = parse_pair(pair_name)
    if not set(pair) <= set(active):
        raise ValueError(f"{state}: inactive q2 pair {pair_name}")
    remaining = tuple(value for value in active if value not in pair)
    if len(remaining) != 2:
        raise ValueError(f"{state}: q2 pair does not leave a forced pair: {pair_name}")
    first = parse_pair(q2_qc[state]["first_reacted_pair"])
    second = parse_pair(q2_qc[state]["second_reacted_pair"])
    sizes = union_sizes(
        [(mapping[a], mapping[b]) for a, b in (first, second, pair, remaining)]
    )
    if sizes == (2, 2):
        return "R2x2"
    if sizes == (4,):
        return "R4"
    return "non_target_R1_R3"


def q1_replica_weights(rows: Sequence[dict[str, str]], state: str) -> dict[str, dict[str, float]]:
    selected = [
        row for row in rows
        if row["state"] == state and row["setting_id"] == PRIMARY_SETTING
    ]
    by_rep: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in selected:
        by_rep[row["replica"]][row["history_class"]] += float(row["mean_soft_nac"])
    expected = set(Q1_CATEGORIES)
    for replica, values in by_rep.items():
        if set(values) != expected:
            raise ValueError(f"{state} q1 replica {replica}: missing categories {expected - set(values)}")
    return {replica: dict(values) for replica, values in by_rep.items()}


def q2_replica_weights(
    rows: Sequence[dict[str, str]],
    state: str,
    outcomes: dict[str, str],
) -> dict[str, dict[str, float]]:
    selected = [
        row for row in rows
        if row["state"] == state and row["setting_id"] == PRIMARY_SETTING
    ]
    by_rep: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in selected:
        outcome = outcomes[row["pair"]]
        by_rep[row["replica"]][outcome] += float(row["mean_soft_nac"])
    return {replica: dict(values) for replica, values in by_rep.items()}


def mean_weights(by_rep: dict[str, dict[str, float]], keys: Sequence[str]) -> dict[str, float]:
    return {
        key: float(np.mean([values.get(key, 0.0) for values in by_rep.values()]))
        for key in keys
    }


def conditional(values: dict[str, float], keys: Sequence[str]) -> dict[str, float]:
    total = sum(values.get(key, 0.0) for key in keys)
    if total <= 0.0:
        return {key: float("nan") for key in keys}
    return {key: values.get(key, 0.0) / total for key in keys}


def product_counts(
    q1: dict[str, float],
    adjacent_r4: float,
    opposite_r2: float,
    opposite_r4: float,
) -> tuple[float, float]:
    target = sum(q1[key] for key in Q1_TARGET)
    p_full_reuse = q1["reuse_arm_pair"] / target
    p_partial = q1["one_reacted_arm"] / target
    p_full_switch = q1["untouched_arm_pair"] / target
    n_r2 = 2.0 * p_full_reuse + 2.0 * p_full_switch * opposite_r2
    n_r4 = p_partial * adjacent_r4 + p_full_switch * opposite_r4
    return n_r2, n_r4


def experimental_targets(path: Path, system: str) -> tuple[float, float]:
    rows = [row for row in read_csv(path) if row["system"] == f"A4_m_{system}"]
    values = {int(row["ring_size"]): float(row["ring_wt_percent"]) for row in rows}
    total = values[2] + values[4]
    return values[2] / total, values[4] / total


def analyze(root: Path, output: Path, bootstrap: int, seed: int) -> None:
    q1_rows = read_csv(root / "results/a4_m_c6_c7_q1_10ns_analysis_all_pairs/pair_nac_statistics.csv")
    q2_rows = read_csv(root / "results/a4_m_c6_c7_pilot_analysis_all_pairs/pair_nac_statistics.csv")
    mapping_rows = read_csv(root / "results/a4_postreaction_nac_prepared_q2/results/react2/vinyl_mapping.csv")
    qc_rows = read_csv(root / "results/a4_postreaction_nac_prepared_q2/results/react2/structure_qc.csv")
    q2_mapping: dict[str, dict[int, int]] = defaultdict(dict)
    q2_active: dict[str, list[int]] = defaultdict(list)
    for row in mapping_rows:
        vinyl = int(row["vinyl"][1:])
        q2_mapping[row["state"]][vinyl] = int(row["arm_zero_based"])
        if row["vinyl_status"] == "active":
            q2_active[row["state"]].append(vinyl)
    q2_qc = {row["state"]: row for row in qc_rows}

    output.mkdir(parents=True, exist_ok=True)
    branch_rows: list[dict[str, object]] = []
    matching_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)

    for system in STATES:
        q1_state = f"A4_m_{system}_1_cis"
        q1_rep = q1_replica_weights(q1_rows, q1_state)
        q1_point = mean_weights(q1_rep, Q1_CATEGORIES)
        q1_target = sum(q1_point[key] for key in Q1_TARGET)
        q1_same_leak = q1_point["same_arm"] / sum(q1_point.values())
        p_q1 = {key: q1_point[key] / q1_target for key in Q1_TARGET}
        for category in Q1_CATEGORIES:
            branch_rows.append({
                "system": system,
                "stage": "q1",
                "category": category,
                "candidate_count": {"reuse_arm_pair": 1, "one_reacted_arm": 8, "untouched_arm_pair": 4, "same_arm": 2}[category],
                "mean_total_soft_nac": q1_point[category],
                "target_conditioned_probability": p_q1.get(category, ""),
                "q1_same_arm_leakage_unconditional": q1_same_leak,
            })

        branch_data: dict[str, tuple[dict[str, dict[str, float]], dict[str, str]]] = {}
        for branch in ("adjacent", "opposite"):
            state = f"A4_m_{system}_2_{branch}_cis"
            available_pairs = sorted({row["pair"] for row in q2_rows if row["state"] == state and row["setting_id"] == PRIMARY_SETTING})
            outcomes = {
                pair: q2_outcome(state, pair, q2_mapping, q2_active, q2_qc)
                for pair in available_pairs
            }
            weights = q2_replica_weights(q2_rows, state, outcomes)
            branch_data[branch] = (weights, outcomes)
            keys = ("R4", "non_target_R1_R3") if branch == "adjacent" else ("R2x2", "R4")
            point_weights = mean_weights(weights, keys)
            point_cond = conditional(point_weights, keys)
            for pair in available_pairs:
                matching_rows.append({
                    "system": system,
                    "state": state,
                    "branch": branch,
                    "pair": pair,
                    "outcome": outcomes[pair],
                    "mean_soft_nac": float(np.mean([
                        float(row["mean_soft_nac"])
                        for row in q2_rows
                        if row["state"] == state
                        and row["setting_id"] == PRIMARY_SETTING
                        and row["pair"] == pair
                    ])),
                })
            branch_rows.extend([
                {
                    "system": system,
                    "stage": f"q2_{branch}",
                    "category": key,
                    "candidate_count": sum(1 for outcome in outcomes.values() if outcome == key),
                    "mean_total_soft_nac": point_weights[key],
                    "target_conditioned_probability": point_cond[key],
                    "q1_same_arm_leakage_unconditional": "",
                }
                for key in keys
            ])

        adj_rep = branch_data["adjacent"][0]
        opp_rep = branch_data["opposite"][0]
        adj_point = conditional(mean_weights(adj_rep, ("R4", "non_target_R1_R3")), ("R4", "non_target_R1_R3"))
        opp_point = conditional(mean_weights(opp_rep, ("R2x2", "R4")), ("R2x2", "R4"))
        n2, n4 = product_counts(q1_point, adj_point["R4"], opp_point["R2x2"], opp_point["R4"])
        target_total = n2 + n4
        mol_r2 = n2 / target_total
        mol_r4 = n4 / target_total
        mass_r2 = n2 / (n2 + 2.0 * n4)
        exp_r2, exp_r4 = experimental_targets(root / "analysis/experimental_targets.csv", system)

        boot_mol: list[float] = []
        boot_mass: list[float] = []
        q1_reps = sorted(q1_rep)
        adj_reps = sorted(adj_rep)
        opp_reps = sorted(opp_rep)
        for _ in range(bootstrap):
            q1_draw = [q1_reps[index] for index in rng.integers(0, len(q1_reps), len(q1_reps))]
            adj_draw = [adj_reps[index] for index in rng.integers(0, len(adj_reps), len(adj_reps))]
            opp_draw = [opp_reps[index] for index in rng.integers(0, len(opp_reps), len(opp_reps))]
            q1_boot = {key: float(np.mean([q1_rep[rep].get(key, 0.0) for rep in q1_draw])) for key in Q1_CATEGORIES}
            adj_w = {key: float(np.mean([adj_rep[rep].get(key, 0.0) for rep in adj_draw])) for key in ("R4", "non_target_R1_R3")}
            opp_w = {key: float(np.mean([opp_rep[rep].get(key, 0.0) for rep in opp_draw])) for key in ("R2x2", "R4")}
            adj_c = conditional(adj_w, ("R4", "non_target_R1_R3"))
            opp_c = conditional(opp_w, ("R2x2", "R4"))
            b2, b4 = product_counts(q1_boot, adj_c["R4"], opp_c["R2x2"], opp_c["R4"])
            boot_mol.append(b2 / (b2 + b4))
            boot_mass.append(b2 / (b2 + 2.0 * b4))
        mol_ci = np.quantile(boot_mol, [0.025, 0.5, 0.975])
        mass_ci = np.quantile(boot_mass, [0.025, 0.5, 0.975])
        prediction_rows.append({
            "system": system,
            "q1_full_reuse_probability": p_q1["reuse_arm_pair"],
            "q1_partial_reuse_probability": p_q1["one_reacted_arm"],
            "q1_full_switch_probability": p_q1["untouched_arm_pair"],
            "q1_same_arm_leakage_unconditional": q1_same_leak,
            "q2_adjacent_R4_probability_target_conditioned": adj_point["R4"],
            "q2_adjacent_non_target_leakage": 1.0 - adj_point["R4"],
            "q2_opposite_R2x2_probability": opp_point["R2x2"],
            "q2_opposite_R4_probability": opp_point["R4"],
            "R2_count_proxy": n2,
            "R4_count_proxy": n4,
            "R2_molecule_fraction": mol_r2,
            "R4_molecule_fraction": mol_r4,
            "R2_ring_unit_mass_fraction_M4_over_M2_2": mass_r2,
            "R4_ring_unit_mass_fraction_M4_over_M2_2": 1.0 - mass_r2,
            "R2_molecule_fraction_bootstrap_ci95_low": mol_ci[0],
            "R2_molecule_fraction_bootstrap_median": mol_ci[1],
            "R2_molecule_fraction_bootstrap_ci95_high": mol_ci[2],
            "R2_mass_proxy_bootstrap_ci95_low": mass_ci[0],
            "R2_mass_proxy_bootstrap_median": mass_ci[1],
            "R2_mass_proxy_bootstrap_ci95_high": mass_ci[2],
            "experimental_R2_wt_fraction_target_normalized": exp_r2,
            "experimental_R4_wt_fraction_target_normalized": exp_r4,
        })

    write_csv(output / "q1_q2_branch_weights.csv", branch_rows)
    write_csv(output / "q2_matching_outcomes.csv", matching_rows)
    write_csv(output / "r2_r4_predictions.csv", prediction_rows)
    lines = [
        "# A4-m-C6/C7 target-only R2/R4 propagation",
        "",
        f"Primary setting: `{PRIMARY_SETTING}`. q0 is fixed to cross. q1 same-arm and q2 R1/R3 outcomes are reported as leakage but excluded from the conditional R2/R4 prediction.",
        "",
        "## Product bookkeeping",
        "",
        "q1 full reuse contributes 2 R2; q1 partial reuse enters q2 adjacent and contributes R4 on the target cross branch; q1 full switch enters q2 opposite, where R2x2 contributes 2 R2 and each R4 matching contributes 1 R4.",
        "",
        "| system | q1 full reuse | q1 partial reuse | q1 full switch | q2 opposite R2x2 | q2 opposite R4 | predicted R2/R4 (molecule) | predicted R2/R4 (M4/M2=2 mass proxy) | experiment R2/R4 (wt%, target-normalized) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in prediction_rows:
        lines.append(
            f"| {row['system']} | {float(row['q1_full_reuse_probability']):.3f} | {float(row['q1_partial_reuse_probability']):.3f} | {float(row['q1_full_switch_probability']):.3f} | {float(row['q2_opposite_R2x2_probability']):.3f} | {float(row['q2_opposite_R4_probability']):.3f} | {float(row['R2_molecule_fraction']):.3f}/{float(row['R4_molecule_fraction']):.3f} | {float(row['R2_ring_unit_mass_fraction_M4_over_M2_2']):.3f}/{float(row['R4_ring_unit_mass_fraction_M4_over_M2_2']):.3f} | {float(row['experimental_R2_wt_fraction_target_normalized']):.3f}/{float(row['experimental_R4_wt_fraction_target_normalized']):.3f} |"
        )
    lines.extend([
        "",
        "The mass-proxy column is only a diagnostic with M4/M2=2; the final comparison to SEC wt% should use the exact post-cleavage molecular formulas. The source tables retain candidate multiplicity explicitly, so the four partial-reuse q1 candidates are not collapsed to a single per-pair mean.",
        "",
        "Files: `q1_q2_branch_weights.csv`, `q2_matching_outcomes.csv`, and `r2_r4_predictions.csv`.",
    ])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "primary_setting": PRIMARY_SETTING,
        "q0_assumption": "cross fixed",
        "target_products": ["R2", "R4"],
        "systems": list(STATES),
        "bootstrap_replicates": bootstrap,
        "files": ["q1_q2_branch_weights.csv", "q2_matching_outcomes.csv", "r2_r4_predictions.csv", "REPORT.md"],
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "predictions": prediction_rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/a4_r2_r4_propagation")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    analyze(PROJECT, args.output.resolve(), args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
