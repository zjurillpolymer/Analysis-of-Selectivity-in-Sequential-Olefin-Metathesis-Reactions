#!/usr/bin/env python3
"""Map the measured four-bond B6 state onto the q4/q5 completion branch.

The input file is called ``q4`` in the computational package, but it contains
four reacted pairs.  Under the project's reaction-event convention (the first
reaction is q0), this is the q3 state.  The next branching event is q4; after
that event only one pair remains, so q5 is deterministic.  V1-V4, V1-V5,
V2-V4 and V2-V5 extend the path and lead to the six-arm closure (R6), whereas
V1-V2 and V4-V5 close the wrong components (R5+R1).  This script computes the
q3 -> q4 branch probability only; q4 -> q5 has probability one conditional on
the selected q4 branch.  It is not a full q0--q5 yield prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair-summary",
        type=Path,
        default=project / (
            "bohrium_download/b6_q4_md_pilot_20260922_fix1/20764906/results/"
            "workspace/results/b6_q4_2ns_analysis/pair_state_summary.csv"
        ),
    )
    parser.add_argument(
        "--orientation-summary",
        type=Path,
        default=project / "results/b6_q4_v1_v5_orientation_analysis/summary.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "results/b6_q4_completion_sensitivity",
    )
    args = parser.parse_args()
    rows = list(csv.DictReader(args.pair_summary.open(encoding="utf-8")))
    scores = {(int(r["left"]), int(r["right"])): float(r["soft_q_mean"]) for r in rows}
    orientation = json.loads(args.orientation_summary.read_text(encoding="utf-8"))
    kappa = float(orientation["full"]["conditional_orientation_gate"])
    r6_pairs = {(1, 4), (1, 5), (2, 4), (2, 5)}
    wrong_pairs = {(1, 2), (4, 5)}
    expected = r6_pairs | wrong_pairs
    if set(scores) != expected:
        raise ValueError(f"unexpected q4 candidate pairs: {sorted(scores)}")

    scenarios = {
        "distance_only": {pair: 1.0 for pair in expected},
        "common_orientation_factor": {pair: kappa for pair in expected},
        "conservative_V1_V5_only_orientation": {
            pair: (kappa if pair == (1, 5) else 1.0) for pair in expected
        },
    }
    result_rows = []
    for name, factors in scenarios.items():
        weighted = {pair: scores[pair] * factors[pair] for pair in expected}
        total = sum(weighted.values())
        r6 = sum(weighted[pair] for pair in r6_pairs) / total
        wrong = sum(weighted[pair] for pair in wrong_pairs) / total
        result_rows.append(
            {
                "scenario": name,
                "orientation_factor_V1_V5": factors[(1, 5)],
                "q4_to_R6_completion_probability": r6,
                "q4_to_wrong_R5_plus_R1_probability": wrong,
                "weighted_V1_V5_share": weighted[(1, 5)] / total,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "q4_completion_sensitivity.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    summary = {
        "state": rows[0]["state"],
        "orientation_factor_from_existing_umbrella": kappa,
        "event_indexing": {
            "internal_state_name": "q4 (four reacted pairs)",
            "user_reaction_event_name": "q3",
            "branching_event": "q4",
            "final_event": "q5 (unique remaining pair)",
        },
        "r6_pairs": ["V1-V4", "V1-V5", "V2-V4", "V2-V5"],
        "wrong_component_pairs": ["V1-V2", "V4-V5"],
        "pair_soft_contact_scores": {f"V{a}-V{b}": value for (a, b), value in scores.items()},
        "scenarios": result_rows,
        "interpretation": (
            "Within the measured four-bond (user q3) topology, the q4 branching event is already strongly R6-favouring. "
            "Even the conservative case that penalizes only the measured V1-V5 pair by its orientation factor "
            "leaves the R6 branch probability above 97%. Once that q4 branch is selected, q5 is the unique remaining closure. "
            "The unresolved yield uncertainty therefore lies upstream in q1/q2 branch weights, not in q5."
        ),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report = f"""# B6 q3→q4→q5 局部产物映射审计

## 图映射

当前文件名为 q4 的共价图实际包含 4 条已形成的键；按反应事件编号它是 q3 状态。它是一个五臂链加一个孤立臂。在只考虑邻居候选的模型中，第五次反应（q4）有两类结果：

- `V1–V4`, `V1–V5`, `V2–V4`, `V2–V5` 会把孤立臂接入主链，随后剩余两个端点闭合，得到 R6；
- `V1–V2` 或 `V4–V5` 会先闭合错误的组分，得到 R5+R1，而不是目标 R6。

V1–V5 的取向因子来自现有一维距离 umbrella 的后处理：`kappa = {kappa:.4f}`。第五次反应完成后只剩一对乙烯基，因此第六次反应（q5）是唯一位点，条件概率为 1。

## 条件概率

| 情景 | q4 分支→R6（随后 q5 唯一闭合） | q4 分支→R5+R1 | 说明 |
|---|---:|---:|---|
""" + "\n".join(
        f"| {row['scenario']} | {100*float(row['q4_to_R6_completion_probability']):.2f}% | "
        f"{100*float(row['q4_to_wrong_R5_plus_R1_probability']):.2f}% | "
        + ("所有竞争对共用同一取向因子" if row["scenario"] == "common_orientation_factor" else "距离权重；或仅对 V1–V5 施加取向修正")
        + " |"
        for row in result_rows
    ) + "\n\n## 结论\n\n"
    "在 q3 状态到 q4 的第五次反应分支中，R6 路径已经明显占优：即使采取最保守的处理，只对目前有 umbrella 证据的 V1–V5 施加取向惩罚，R6 分支概率仍高于 97%。选定 q4 分支后，q5 是唯一剩余位点。因此继续在 q5 增加 umbrella 没有意义；最终 R6:R2 的不确定性主要来自更早的 q1/q2 分支权重。\n"
    (args.output / "B6_q4_completion_mapping.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
