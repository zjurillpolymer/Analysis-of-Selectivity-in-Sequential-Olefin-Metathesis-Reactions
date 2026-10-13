#!/usr/bin/env python3
"""Exact R2/R6 propagation for the B6 neighbour-arm-only model.

The first event (q0) is fixed to V1--V12. Only cross-arm pairs on adjacent
edges of the six-arm cycle are allowed.  All perfect endpoint matchings are
enumerated exactly, then conditioned on the four measured q1 orbit classes.

No unmeasured transition is assigned a neutral probability.  The sole
remaining ambiguous q1 class is exposed as a symbolic probability ``s_d1``.
The reported interval spans its entire physical range [0, 1].
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_arm_map(root: Path) -> dict[int, int]:
    # Authoritative mapping: arms are (V1,V2), (V3,V4), ..., (V11,V12).
    # The legacy manifest used a different label ordering and is not used here.
    return {vinyl: (vinyl - 1) // 2 for vinyl in range(1, 13)}


def arm_edge(left: int, right: int) -> tuple[int, int]:
    return tuple(sorted((left, right)))


def allowed_arm_edges() -> set[tuple[int, int]]:
    return {arm_edge(index, (index + 1) % 6) for index in range(6)}


def allowed_endpoint_pairs(arm_of: dict[int, int]) -> set[tuple[int, int]]:
    allowed = allowed_arm_edges()
    return {
        (left, right)
        for left in range(1, 13)
        for right in range(left + 1, 13)
        if arm_of[left] != arm_of[right]
        and arm_edge(arm_of[left], arm_of[right]) in allowed
    }


def enumerate_matchings(
    unused: frozenset[int],
    allowed: set[tuple[int, int]],
) -> Iterable[tuple[tuple[int, int], ...]]:
    if not unused:
        yield ()
        return
    left = min(unused)
    for right in sorted(unused - {left}):
        edge = (left, right)
        if edge not in allowed:
            continue
        remaining = unused - {left, right}
        for suffix in enumerate_matchings(remaining, allowed):
            yield (edge,) + suffix


def product_partition(
    matching: tuple[tuple[int, int], ...], arm_of: dict[int, int]
) -> tuple[int, ...]:
    parent = list(range(6))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right in matching:
        union(arm_of[left], arm_of[right])
    counts = Counter(find(arm) for arm in range(6))
    return tuple(sorted(counts.values(), reverse=True))


def q1_category(q0_arm_edge: tuple[int, int], candidate: tuple[int, int]) -> str:
    overlap = len(set(q0_arm_edge) & set(candidate))
    if overlap == 2:
        return "full_reuse"
    if overlap == 1:
        return "one_arm_reuse"
    distances = []
    for left in q0_arm_edge:
        for right in candidate:
            delta = abs(left - right)
            distances.append(min(delta, 6 - delta))
    return f"full_switch_d{min(distances)}"


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument(
        "--q1-scenarios", type=Path, default=None,
        help="corrected q1 weights; omitted because legacy weights use an incompatible arm map",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root_default / "results/b6_neighbor_r2_r6_propagation_20260923",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    arm_of = load_arm_map(root)
    allowed = allowed_endpoint_pairs(arm_of)
    q0 = (1, 12)
    if q0 not in allowed:
        raise ValueError("fixed q0 V1-V2 is not an allowed neighbour edge")
    q0_arm = arm_edge(arm_of[q0[0]], arm_of[q0[1]])
    remaining = frozenset(range(1, 13)) - set(q0)
    complete = [(q0,) + suffix for suffix in enumerate_matchings(remaining, allowed)]
    if not complete:
        raise RuntimeError("no complete neighbour-only matchings")

    topology_counts = Counter(product_partition(matching, arm_of) for matching in complete)
    unexpected = set(topology_counts) - {(6,), (2, 2, 2)}
    if unexpected:
        raise ValueError(f"unexpected products in neighbour-only graph: {unexpected}")

    q1_rows: list[dict[str, object]] = []
    by_category: defaultdict[str, Counter[tuple[int, ...]]] = defaultdict(Counter)
    candidates = sorted(
        edge for edge in allowed
        if not (set(edge) & set(q0))
    )
    for candidate in candidates:
        category = q1_category(
            q0_arm, arm_edge(arm_of[candidate[0]], arm_of[candidate[1]])
        )
        compatible = [matching for matching in complete if candidate in matching]
        outcomes = Counter(product_partition(matching, arm_of) for matching in compatible)
        by_category[category].update(outcomes)
        q1_rows.append(
            {
                "q1_pair": f"V{candidate[0]}-V{candidate[1]}",
                "q1_arm_edge": f"A{arm_of[candidate[0]]}-A{arm_of[candidate[1]]}",
                "category": category,
                "complete_endpoint_matchings": len(compatible),
                "R6_completions": outcomes[(6,)],
                "three_R2_completions": outcomes[(2, 2, 2)],
                "terminal_outcome": (
                    "R6" if set(outcomes) == {(6,)}
                    else "3R2" if set(outcomes) == {(2, 2, 2)}
                    else "R6_or_3R2"
                ),
            }
        )
    write_csv(output / "q1_candidate_terminal_map.csv", q1_rows)

    category_rows = []
    for category in ("full_reuse", "one_arm_reuse", "full_switch_d1", "full_switch_d2"):
        counts = by_category[category]
        possible = []
        if counts[(6,)] > 0:
            possible.append("R6")
        if counts[(2, 2, 2)] > 0:
            possible.append("3R2")
        category_rows.append(
            {
                "q1_category": category,
                "candidate_count": sum(1 for row in q1_rows if row["category"] == category),
                "enumerated_completion_count": sum(counts.values()),
                "R6_completion_count": counts[(6,)],
                "three_R2_completion_count": counts[(2, 2, 2)],
                "possible_terminal_products": ";".join(possible),
                "product_deterministic_from_q1": len(possible) == 1,
            }
        )
    write_csv(output / "q1_category_terminal_map.csv", category_rows)

    # Audit whether the third event (q2) uniquely fixes the terminal product.
    # We retain every endpoint-labelled history; symmetry is summarized later.
    q2_rows: list[dict[str, object]] = []
    for q1 in candidates:
        q1_class = q1_category(
            q0_arm, arm_edge(arm_of[q1[0]], arm_of[q1[1]])
        )
        used = set(q0) | set(q1)
        for q2 in sorted(edge for edge in allowed if not (set(edge) & used)):
            compatible = [
                matching for matching in complete if q1 in matching and q2 in matching
            ]
            if not compatible:
                continue
            outcomes = Counter(product_partition(matching, arm_of) for matching in compatible)
            possible = []
            if outcomes[(6,)]:
                possible.append("R6")
            if outcomes[(2, 2, 2)]:
                possible.append("3R2")
            q2_rows.append({
                "q1_pair": f"V{q1[0]}-V{q1[1]}",
                "q1_category": q1_class,
                "q2_pair": f"V{q2[0]}-V{q2[1]}",
                "q2_arm_edge": f"A{arm_of[q2[0]] + 1}-A{arm_of[q2[1]] + 1}",
                "compatible_completions": len(compatible),
                "R6_completions": outcomes[(6,)],
                "three_R2_completions": outcomes[(2, 2, 2)],
                "possible_terminal_products": ";".join(possible),
                "product_determined_after_q2": len(possible) == 1,
            })
    write_csv(output / "q2_history_terminal_map.csv", q2_rows)
    unresolved_q2 = [row for row in q2_rows if not row["product_determined_after_q2"]]

    q3_history_count = 0
    q3_unresolved_count = 0
    for q1 in candidates:
        used_q1 = set(q0) | set(q1)
        for q2 in sorted(edge for edge in allowed if not (set(edge) & used_q1)):
            used_q2 = used_q1 | set(q2)
            for q3 in sorted(edge for edge in allowed if not (set(edge) & used_q2)):
                compatible = [
                    matching for matching in complete
                    if q1 in matching and q2 in matching and q3 in matching
                ]
                if not compatible:
                    continue
                q3_history_count += 1
                outcomes = {product_partition(matching, arm_of) for matching in compatible}
                q3_unresolved_count += len(outcomes) > 1

    propagation_rows: list[dict[str, object]] = []
    experimental_r6 = 0.75
    for row in read_csv(args.q1_scenarios) if args.q1_scenarios else []:
        p_fr = float(row["full_reuse_fraction"])
        p_oar = float(row["one_arm_reuse_fraction"])
        p_d1 = float(row["full_switch_d1_fraction"])
        p_d2 = float(row["full_switch_d2_fraction"])
        total = p_fr + p_oar + p_d1 + p_d2
        if not math.isclose(total, 1.0, abs_tol=2.0e-6):
            raise ValueError(f"q1 fractions do not normalize: {total}")
        r6_min = p_oar + p_d2
        r6_max = r6_min + p_d1
        implied = (experimental_r6 - r6_min) / p_d1 if p_d1 > 0 else math.nan
        propagation_rows.append(
            {
                "replica_scenario": row["replica_scenario"],
                "q1_full_reuse_to_3R2": p_fr,
                "q1_one_arm_reuse_to_R6": p_oar,
                "q1_full_switch_d2_to_R6": p_d2,
                "q1_full_switch_d1_ambiguous": p_d1,
                "R6_mass_fraction_min_s_d1_0": r6_min,
                "R6_mass_fraction_max_s_d1_1": r6_max,
                "R2_mass_fraction_max_s_d1_0": 1.0 - r6_min,
                "R2_mass_fraction_min_s_d1_1": 1.0 - r6_max,
                "experimental_R6_mass_fraction": experimental_r6,
                "implied_s_d1_for_experiment": implied,
                "experiment_reachable_by_d1_only": 0.0 <= implied <= 1.0,
            }
        )
    write_csv(output / "r2_r6_propagation_bounds.csv", propagation_rows)

    summary = {
        "model": "B6 neighbour-arm-only exact endpoint matching",
        "event_indexing": "q0 is the first reaction; q5 is the sixth and final reaction",
        "fixed_q0": "V1-V12",
        "allowed_rule": "cross-arm pairs on adjacent edges of the six-arm cycle only",
        "complete_endpoint_matching_count": len(complete),
        "terminal_topology_counts": {
            "R6": topology_counts[(6,)],
            "three_R2": topology_counts[(2, 2, 2)],
        },
        "q1_orbit_multiplicities": {
            "full_reuse": 1,
            "one_arm_reuse_mirror_normalized": 4,
            "full_switch_d1_mirror_pair": 8,
            "full_switch_d2_central": 4,
        },
        "q2_history_count": len(q2_rows),
        "q2_unresolved_history_count": len(unresolved_q2),
        "q3_history_count": q3_history_count,
        "q3_unresolved_history_count": q3_unresolved_count,
        "exact_yield_formula": {
            "R6_mass_fraction": "p(one_arm_reuse) + p(full_switch_d2) + p(full_switch_d1) * s_d1",
            "R2_mass_fraction": "p(full_reuse) + p(full_switch_d1) * (1 - s_d1)",
            "s_d1": "conditional probability that the q1 full-switch-d1 branch completes as R6",
        },
        "q1_scenario_bounds": propagation_rows,
        "excluded_data": (
            "The V3-V6/V11-V12/V8-V10/V7-V9 four-bond trajectory belongs to the legacy all-pair path, "
            "not to the corrected fixed-q0 V1-V12 neighbour-only graph, and is therefore not used here."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# B6 邻居模型：修正臂映射后的精确拓扑枚举",
        "",
        "臂定义为 A1=(V1,V2)、A2=(V3,V4)、…、A6=(V11,V12)，固定 q0=V1–V12（A1–A6）。只允许相邻臂 cross。",
        "",
        "## q1 对称归一化",
        "",
        "| 粗类别 | 内部轨道 | 端点候选简并度 | q1 后终态 |",
        "|---|---|---:|---|",
        "| reuse | A1–A6 | 1 | 3×R2 |",
        "| one-arm reuse | A1–A2 与 A6–A5 为镜像同类 | 2+2=4 | R6 |",
        "| switch | d1：A2–A3 与 A5–A4 为镜像同类 | 4+4=8 | R6 或 3×R2 |",
        "| switch | d2：A3–A4，自身对称 | 4 | R6 |",
        "",
        "因此 one-arm reuse 左右两侧不能当作两个独立机理概率；应先把简并度相加后再归一化。switch 在语言上仍可是一大类，但自由能加权时必须保留 d1/d2 两个内部轨道。",
        "",
        "## 三次反应是否足够",
        "",
        f"完整端点匹配共 {len(complete)} 个，只产生 {topology_counts[(6,)]} 个 R6 标记匹配与 {topology_counts[(2,2,2)]} 个 3×R2 标记匹配。因此三步以内已经能枚举出全部产物类型：只有 R6 与 3×R2。",
        "",
        f"但是在 {len(q2_rows)} 条允许的 q0→q1→q2 端点历史中，仍有 {len(unresolved_q2)} 条历史同时兼容 R6 与 3×R2。也就是说，三次反应不能保证给每条具体历史唯一贴上终产物标签；这些残余历史还需看 q3。枚举至 q3 后共有 {q3_history_count} 条有序端点历史，歧义数降为 {q3_unresolved_count}。",
        "",
        "## 数值概率状态",
        "",
        "旧 q1 umbrella 权重来自不兼容的旧臂标签，已从本报告停用。下一步只能对修正后的四个轨道代表（reuse、one-arm reuse、switch-d1、switch-d2）核对结构并重新计算/重标，之后才能与实验 R6:R2=3:1 比较。",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
