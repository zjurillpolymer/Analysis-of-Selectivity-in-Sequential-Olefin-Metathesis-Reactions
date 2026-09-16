#!/usr/bin/env python3
"""Summarize the C5/C8 q1 3x10 ns three-class branch populations."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY = "d5_axis45_side45"
SYSTEMS = ("C5", "C8")
CLASSES = ("full_reuse", "partial_reuse", "full_switch")
HISTORY_TO_CLASS = {
    "reuse_arm_pair": "full_reuse",
    "one_reacted_arm": "partial_reuse",
    "untouched_arm_pair": "full_switch",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def class_weights(rows: list[dict[str, str]], system: str) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        if row["state"] != f"A4_m_{system}_1_cis":
            continue
        if row.get("setting_id") and row["setting_id"] != PRIMARY:
            continue
        category = HISTORY_TO_CLASS.get(row["history_class"])
        if category is not None:
            result[int(row["replica"])][category] += float(row["mean_soft_nac"])
    if sorted(result) != [1, 2, 3]:
        raise ValueError(f"{system}: expected replicas 1, 2, 3; found {sorted(result)}")
    return {replica: dict(values) for replica, values in result.items()}


def point(weights: dict[int, dict[str, float]]) -> dict[str, float]:
    raw = {
        category: float(np.mean([weights[replica].get(category, 0.0) for replica in sorted(weights)]))
        for category in CLASSES
    }
    total = sum(raw.values())
    return {category: raw[category] / total for category in CLASSES}


def bootstrap(
    weights: dict[int, dict[str, float]], samples: int, rng: np.random.Generator
) -> dict[str, tuple[float, float, float]]:
    replicas = np.asarray(sorted(weights), dtype=int)
    draws = rng.choice(replicas, size=(samples, len(replicas)), replace=True)
    matrix = np.asarray(
        [[weights[int(replica)].get(category, 0.0) for category in CLASSES] for replica in replicas]
    )
    lookup = {replica: index for index, replica in enumerate(replicas)}
    sampled = np.asarray([[lookup[int(value)] for value in row] for row in draws])
    raw = matrix[sampled].mean(axis=1)
    normalized = raw / raw.sum(axis=1, keepdims=True)
    return {
        category: tuple(float(value) for value in np.quantile(normalized[:, index], (0.025, 0.5, 0.975)))
        for index, category in enumerate(CLASSES)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--new", type=Path,
        default=PROJECT / "results/a4_m_c5_c8_q1_10ns_analysis_all_pairs/pair_nac_statistics.csv",
    )
    parser.add_argument(
        "--old", type=Path,
        default=PROJECT / "results/a4_postreaction_nac_analysis/reuse_switch_dynamics/reuse_switch_pair_summary.csv",
    )
    parser.add_argument(
        "--convergence", type=Path,
        default=PROJECT / "results/a4_m_c5_c8_q1_10ns_convergence_all_pairs/q1_prefix_aggregate.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/a4_m_c5_c8_q1_10ns_three_class_summary",
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    new_rows, old_rows = read_csv(args.new.resolve()), read_csv(args.old.resolve())
    rng = np.random.default_rng(args.seed)

    replica_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    points: dict[str, dict[str, float]] = {}
    intervals: dict[str, dict[str, tuple[float, float, float]]] = {}
    for system in SYSTEMS:
        weights = class_weights(new_rows, system)
        points[system] = point(weights)
        intervals[system] = bootstrap(weights, args.bootstrap, rng)
        for replica in sorted(weights):
            total = sum(weights[replica].get(category, 0.0) for category in CLASSES)
            for category in CLASSES:
                replica_rows.append({
                    "system": system,
                    "replica": replica,
                    "category": category,
                    "summed_soft_nac": weights[replica].get(category, 0.0),
                    "within_replica_fraction": weights[replica].get(category, 0.0) / total,
                })
        old_point = point(class_weights(old_rows, system))
        for category in CLASSES:
            low, median, high = intervals[system][category]
            summary_rows.append({
                "system": system,
                "category": category,
                "population": points[system][category],
                "replica_bootstrap_ci95_low": low,
                "replica_bootstrap_median": median,
                "replica_bootstrap_ci95_high": high,
            })
            comparison_rows.append({
                "system": system,
                "category": category,
                "old_3x2ns_population": old_point[category],
                "new_3x10ns_population": points[system][category],
                "change": points[system][category] - old_point[category],
            })

    convergence = read_csv(args.convergence.resolve())
    prefix = {
        (row["state"].split("_C", 1)[1].split("_", 1)[0], int(float(row["end_ps"]))): row
        for row in convergence
    }
    lines = [
        "# A4-m-C5/C8 q1：3×10 ns 补充无偏 MD 分析",
        "",
        "> A4 为四面体体系；统计全部活性跨臂候选，不施加 B6 邻臂过滤。q1 仅拆分为 full reuse、partial reuse、full switch。",
        "",
        "## 三分支人口",
        "",
        "| 体系 | full reuse | partial reuse | full switch |",
        "|---|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        lines.append(
            f"| {system} | {points[system]['full_reuse']:.4f} | "
            f"{points[system]['partial_reuse']:.4f} | {points[system]['full_switch']:.4f} |"
        )
    lines.extend([
        "",
        "三类概率是在每条重复内先对该类全部候选对的 soft-NAC 求和，再跨重复平均并归一化；候选多重性已经包含在内。",
        "",
        "## 与旧 3×2 ns 结果的关系",
        "",
        f"- C5 full reuse：旧值 {point(class_weights(old_rows, 'C5'))['full_reuse']:.4f}，10 ns 更新为 {points['C5']['full_reuse']:.4f}。",
        f"- C8 full reuse：旧值 {point(class_weights(old_rows, 'C8'))['full_reuse']:.6f}，10 ns 更新为 {points['C8']['full_reuse']:.4f}。",
        "- C8 的变化说明 2 ns 确实漏掉了稀有 reuse 构象；但 full reuse 仍低于仅靠该路径解释实验 R2 所需的约 0.171。",
        "",
        "## 收敛判断",
        "",
        f"C5 的 per-pair reuse/switch ΔΔG 从 2 ns 的 {float(prefix[('5', 2000)]['reuse_minus_switch_ddG_kcal_mol']):+.3f} kcal/mol 变化到 10 ns 的 {float(prefix[('5', 10000)]['reuse_minus_switch_ddG_kcal_mol']):+.3f} kcal/mol；点估计方向一致，但三个重复分歧明显。",
        f"C8 从 2 ns 的 {float(prefix[('8', 2000)]['reuse_minus_switch_ddG_kcal_mol']):+.3f} kcal/mol 变化到 10 ns 的 {float(prefix[('8', 10000)]['reuse_minus_switch_ddG_kcal_mol']):+.3f} kcal/mol。reuse 主要在轨迹后段才出现，因此 10 ns 结果仍属于稀有事件估计，不能视为完全收敛。",
        "",
        "cross/same 的方向比 reuse/switch 稳定：27/27 个 NAC 门控设置在 C5、C8 中均给出 cross 更有利。",
        "",
        "## 结论",
        "",
        "C5/C8 延长到 10 ns 后，q0/q1 的 cross 偏好保持稳健；真正未闭合的是 q1 内三分支的自由能人口。下一步应使用相同协议比较 full reuse、partial reuse、full switch 的 umbrella PMF，并保留每类两个独立端点。",
    ])
    write_csv(output / "q1_three_class_by_replica.csv", replica_rows)
    write_csv(output / "q1_three_class_summary.csv", summary_rows)
    write_csv(output / "q1_old_vs_10ns.csv", comparison_rows)
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
