#!/usr/bin/env python3
"""Compare converged B6_p_C6 q0 and V3-V6-cyclized q1 contact ensembles."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


ARM_BY_VINYL = {
    1: "A", 3: "A", 2: "B", 12: "B", 4: "C", 5: "C",
    6: "D", 7: "D", 8: "E", 9: "E", 10: "F", 11: "F",
}
ARMS = tuple("ABCDEF")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def arm_pair(left: int, right: int) -> str:
    first, second = sorted((ARM_BY_VINYL[left], ARM_BY_VINYL[right]))
    return f"{first}-{second}"


def mean_sd(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=1))


def arm_tables(
    q0_replica: list[dict[str, str]], q1_replica: list[dict[str, str]]
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    per_replica: list[dict[str, object]] = []
    for state, rows in (("q0", q0_replica), ("q1_V3-V6", q1_replica)):
        grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            pair = arm_pair(int(row["left"]), int(row["right"]))
            if pair[0] == pair[2]:
                continue
            grouped[(int(row["replica"]), pair)].append(row)
        for (replica, pair), selected in sorted(grouped.items()):
            q = np.asarray([float(row["soft_q_mean"]) for row in selected])
            hard5 = np.asarray([float(row["hard_5A_fraction"]) for row in selected])
            per_replica.append(
                {
                    "state": state,
                    "arm_pair": pair,
                    "replica": replica,
                    "active_endpoint_pairs": len(selected),
                    "soft_q_sum": float(np.sum(q)),
                    "soft_q_mean_per_candidate": float(np.mean(q)),
                    "hard_5A_sum": float(np.sum(hard5)),
                    "hard_5A_mean_per_candidate": float(np.mean(hard5)),
                }
            )

    summary: list[dict[str, object]] = []
    by_state_pair: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in per_replica:
        by_state_pair[(str(row["state"]), str(row["arm_pair"]))].append(row)
    for (state, pair), rows in sorted(by_state_pair.items()):
        q_sum_mean, q_sum_sd = mean_sd(float(row["soft_q_sum"]) for row in rows)
        q_pair_mean, q_pair_sd = mean_sd(
            float(row["soft_q_mean_per_candidate"]) for row in rows
        )
        hard_mean, hard_sd = mean_sd(float(row["hard_5A_sum"]) for row in rows)
        summary.append(
            {
                "state": state,
                "arm_pair": pair,
                "replicas": len(rows),
                "active_endpoint_pairs": int(rows[0]["active_endpoint_pairs"]),
                "soft_q_sum_mean": q_sum_mean,
                "soft_q_sum_replica_sd": q_sum_sd,
                "soft_q_mean_per_candidate": q_pair_mean,
                "soft_q_mean_per_candidate_replica_sd": q_pair_sd,
                "hard_5A_sum_mean": hard_mean,
                "hard_5A_sum_replica_sd": hard_sd,
            }
        )

    summary_by_key = {
        (str(row["state"]), str(row["arm_pair"])): row for row in summary
    }
    comparison: list[dict[str, object]] = []
    for pair in sorted({str(row["arm_pair"]) for row in summary}):
        q0 = summary_by_key[("q0", pair)]
        q1 = summary_by_key[("q1_V3-V6", pair)]
        ratio_sum = float(q1["soft_q_sum_mean"]) / float(q0["soft_q_sum_mean"])
        ratio_pair = float(q1["soft_q_mean_per_candidate"]) / float(
            q0["soft_q_mean_per_candidate"]
        )
        comparison.append(
            {
                "arm_pair": pair,
                "q0_active_endpoint_pairs": q0["active_endpoint_pairs"],
                "q1_active_endpoint_pairs": q1["active_endpoint_pairs"],
                "q0_soft_q_sum_mean": q0["soft_q_sum_mean"],
                "q0_soft_q_sum_replica_sd": q0["soft_q_sum_replica_sd"],
                "q1_soft_q_sum_mean": q1["soft_q_sum_mean"],
                "q1_soft_q_sum_replica_sd": q1["soft_q_sum_replica_sd"],
                "q1_over_q0_sum_ratio": ratio_sum,
                "log2_q1_over_q0_sum": math.log2(ratio_sum),
                "q0_soft_q_mean_per_candidate": q0["soft_q_mean_per_candidate"],
                "q1_soft_q_mean_per_candidate": q1["soft_q_mean_per_candidate"],
                "q1_over_q0_per_candidate_ratio": ratio_pair,
            }
        )
    return per_replica, summary, comparison


def degree_table(per_replica: list[dict[str, object]]) -> list[dict[str, object]]:
    by_state_replica: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in per_replica:
        by_state_replica[(str(row["state"]), int(row["replica"]))].append(row)
    values: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for (state, _), rows in by_state_replica.items():
        degree = {arm: 0.0 for arm in ARMS}
        total = sum(float(row["soft_q_sum"]) for row in rows)
        for row in rows:
            left, right = str(row["arm_pair"]).split("-")
            score = float(row["soft_q_sum"])
            degree[left] += score
            degree[right] += score
        for arm in ARMS:
            values[(state, arm)].append((degree[arm], degree[arm] / (2.0 * total)))
    output: list[dict[str, object]] = []
    for (state, arm), rows in sorted(values.items()):
        degree_mean, degree_sd = mean_sd(row[0] for row in rows)
        fraction_mean, fraction_sd = mean_sd(row[1] for row in rows)
        output.append(
            {
                "state": state,
                "arm": arm,
                "replicas": len(rows),
                "weighted_degree_mean": degree_mean,
                "weighted_degree_replica_sd": degree_sd,
                "degree_fraction_mean": fraction_mean,
                "degree_fraction_replica_sd": fraction_sd,
            }
        )
    return output


def common_pair_table(
    q0_state: list[dict[str, str]], q1_state: list[dict[str, str]]
) -> tuple[list[dict[str, object]], dict[str, float]]:
    q0 = {row["pair"]: row for row in q0_state}
    q1 = {row["pair"]: row for row in q1_state}
    output: list[dict[str, object]] = []
    for pair in sorted(q0.keys() & q1.keys()):
        before = float(q0[pair]["soft_q_mean"])
        after = float(q1[pair]["soft_q_mean"])
        output.append(
            {
                "pair": pair,
                "arm_pair": arm_pair(int(q1[pair]["left"]), int(q1[pair]["right"])),
                "topology": q1[pair]["topology"],
                "q0_soft_q_mean": before,
                "q1_soft_q_mean": after,
                "absolute_change": after - before,
                "q1_over_q0_ratio": after / before,
                "q0_hard_5A_fraction": q0[pair]["hard_5A_fraction"],
                "q1_hard_5A_fraction": q1[pair]["hard_5A_fraction"],
            }
        )
    cross = [row for row in output if row["topology"] == "cross_arm"]
    before = np.asarray([float(row["q0_soft_q_mean"]) for row in cross])
    after = np.asarray([float(row["q1_soft_q_mean"]) for row in cross])
    metrics = {
        "common_cross_arm_pairs": float(len(cross)),
        "q0_common_cross_arm_soft_q_sum": float(np.sum(before)),
        "q1_common_cross_arm_soft_q_sum": float(np.sum(after)),
        "q1_over_q0_common_cross_arm_sum": float(np.sum(after) / np.sum(before)),
        "common_cross_arm_pearson_r": float(pearsonr(before, after).statistic),
        "common_cross_arm_spearman_rho": float(spearmanr(before, after).statistic),
    }
    return output, metrics


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "axes.linewidth": 0.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
        }
    )


def draw_network(
    axis: plt.Axes,
    values: dict[str, float],
    title: str,
    color: str,
    fixed_ad: bool = False,
) -> None:
    positions = {
        "A": (0.0, 1.0), "B": (0.87, 0.50), "C": (0.87, -0.50),
        "D": (0.0, -1.0), "E": (-0.87, -0.50), "F": (-0.87, 0.50),
    }
    maximum = max(values.values())
    for pair, score in sorted(values.items(), key=lambda item: item[1]):
        left, right = pair.split("-")
        x = [positions[left][0], positions[right][0]]
        y = [positions[left][1], positions[right][1]]
        axis.plot(
            x, y, color=color, alpha=0.16 + 0.74 * score / maximum,
            linewidth=0.35 + 5.2 * score / maximum, solid_capstyle="round", zorder=1,
        )
    if fixed_ad:
        axis.plot(
            [positions["A"][0], positions["D"][0]],
            [positions["A"][1], positions["D"][1]],
            color="#7A1F2B", linestyle=(0, (2, 2)), linewidth=1.3, zorder=2,
        )
        axis.text(0.04, 0.0, "fixed V3–V6 link", color="#7A1F2B", fontsize=5.8,
                  rotation=90, va="center")
    for arm, (x, y) in positions.items():
        face = "#E3B341" if arm == "C" else "white"
        axis.scatter(x, y, s=195, facecolor=face, edgecolor="#222222", linewidth=0.8, zorder=3)
        axis.text(x, y, arm, ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)
    axis.set_title(title, pad=1.5)
    axis.set_xlim(-1.18, 1.18)
    axis.set_ylim(-1.14, 1.14)
    axis.set_aspect("equal")
    axis.axis("off")


def make_figure(
    output: Path,
    arm_summary: list[dict[str, object]],
    arm_comparison: list[dict[str, object]],
    q1_state: list[dict[str, str]],
    q1_replica: list[dict[str, str]],
) -> None:
    configure_style()
    by_state = defaultdict(dict)
    for row in arm_summary:
        by_state[str(row["state"])][str(row["arm_pair"])] = float(row["soft_q_sum_mean"])

    fig = plt.figure(figsize=(7.08, 6.25), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.25), width_ratios=(1.0, 1.14))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    draw_network(ax_a, by_state["q0"], "q0: unreacted ensemble", "#4C78A8")
    draw_network(ax_b, by_state["q1_V3-V6"], "q1: after V3–V6 closure", "#E07A5F", fixed_ad=True)

    ordered = sorted(arm_comparison, key=lambda row: float(row["log2_q1_over_q0_sum"]))
    y = np.arange(len(ordered))
    values = np.asarray([float(row["log2_q1_over_q0_sum"]) for row in ordered])
    colors = [
        "#E3B341" if row["arm_pair"] == "C-F" else
        "#3A8D8C" if value > 0 else "#A7ADB4"
        for row, value in zip(ordered, values)
    ]
    ax_c.barh(y, values, color=colors, height=0.72)
    ax_c.axvline(0.0, color="#333333", linewidth=0.7)
    ax_c.set_yticks(y, [str(row["arm_pair"]) for row in ordered])
    ax_c.set_xlabel(r"arm-pair opportunity change, log$_2$(q1/q0)")
    ax_c.set_title("Reaction-history redistribution")
    ax_c.grid(axis="x", color="#E5E5E5", linewidth=0.5, zorder=0)
    ax_c.set_axisbelow(True)
    ax_c.text(
        float(next(row["log2_q1_over_q0_sum"] for row in ordered if row["arm_pair"] == "C-F")),
        next(index for index, row in enumerate(ordered) if row["arm_pair"] == "C-F"),
        "  12.2×", va="center", ha="left", fontsize=6.2, color="#8A6514",
    )

    q1_sorted = sorted(q1_state, key=lambda row: float(row["soft_q_mean"]), reverse=True)
    selected = q1_sorted[:8]
    for pair in ("V4_V10_跨臂", "V5_V10_跨臂"):
        candidate = next(row for row in q1_state if row["pair"] == pair)
        if candidate not in selected:
            selected.append(candidate)
    selected = list(reversed(selected))
    y = np.arange(len(selected))
    means = np.asarray([float(row["soft_q_mean"]) for row in selected])
    errors = np.asarray([float(row["soft_q_replica_sd"]) for row in selected])
    colors = ["#E3B341" if row["pair"] in {"V4_V10_跨臂", "V5_V10_跨臂"} else "#66788A" for row in selected]
    ax_d.barh(y, means, xerr=errors, color=colors, height=0.62, alpha=0.9,
              error_kw={"elinewidth": 0.7, "capsize": 2, "capthick": 0.7})
    replica_by_pair: dict[str, list[float]] = defaultdict(list)
    for row in q1_replica:
        replica_by_pair[row["pair"]].append(float(row["soft_q_mean"]))
    for index, row in enumerate(selected):
        points = replica_by_pair[row["pair"]]
        ax_d.scatter(points, [index - 0.10, index, index + 0.10], s=9, color="#222222",
                     edgecolor="white", linewidth=0.25, zorder=3)
        ax_d.text(
            0.0795,
            index,
            f"{100*float(row['hard_5A_fraction']):.1f}%",
            va="center", ha="right", fontsize=5.5, color="#444444",
        )
    labels = [row["pair"].split("_")[0] + "–" + row["pair"].split("_")[1] for row in selected]
    ax_d.set_yticks(y, labels)
    ax_d.set_xlabel("q1 soft-contact score (mean ± replica SD)")
    ax_d.set_xlim(-0.004, 0.082)
    ax_d.set_title("Leading second-reaction candidates")
    ax_d.grid(axis="x", color="#E5E5E5", linewidth=0.5, zorder=0)
    ax_d.set_axisbelow(True)
    ax_d.text(0.99, 0.02, "labels: occupancy <5 Å", transform=ax_d.transAxes,
              ha="right", va="bottom", fontsize=5.5, color="#555555")

    for label, axis in zip("abcd", (ax_a, ax_b, ax_c, ax_d)):
        axis.text(-0.10, 1.04, label, transform=axis.transAxes, fontsize=9,
                  fontweight="bold", va="bottom")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q0-audit", type=Path, required=True)
    parser.add_argument("--q1-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    source = output / "source_data"
    figures = output / "figures"

    q0_replica = read_csv(args.q0_audit / "pair_replica_summary.csv")
    q1_replica = read_csv(args.q1_audit / "pair_replica_summary.csv")
    q0_state = read_csv(args.q0_audit / "pair_state_summary.csv")
    q1_state = read_csv(args.q1_audit / "pair_state_summary.csv")
    per_replica, arm_summary, arm_comparison = arm_tables(q0_replica, q1_replica)
    degrees = degree_table(per_replica)
    common_pairs, metrics = common_pair_table(q0_state, q1_state)
    top_q1 = sorted(q1_state, key=lambda row: float(row["soft_q_mean"]), reverse=True)

    write_csv(source / "arm_pair_replica_q0_vs_q1.csv", per_replica)
    write_csv(source / "arm_pair_summary_q0_vs_q1.csv", arm_summary)
    write_csv(source / "arm_pair_change_q0_vs_q1.csv", arm_comparison)
    write_csv(source / "arm_degree_q0_vs_q1.csv", degrees)
    write_csv(source / "common_endpoint_pair_change.csv", common_pairs)
    write_csv(source / "q1_endpoint_pair_ranking.csv", top_q1)
    write_csv(source / "common_pair_metrics.csv", [metrics])
    make_figure(figures / "figure_b6_q0_q1_comparison", arm_summary, arm_comparison, q1_state, q1_replica)
    print(f"Wrote comparison bundle: {output}")


if __name__ == "__main__":
    main()
