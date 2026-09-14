#!/usr/bin/env python3
"""Build the final paper-guided B6_p_C6 50 ns source data and figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep all figure text editable in SVG/PDF.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42

ARM_BY_VINYL = {
    1: "A",
    3: "A",
    2: "B",
    12: "B",
    4: "C",
    5: "C",
    6: "D",
    7: "D",
    8: "E",
    9: "E",
    10: "F",
    11: "F",
}
ARMS = list("ABCDEF")
BLUE = "#0F4D92"
BLUE_LIGHT = "#8FB7D9"
TEAL = "#33B5A5"
VIOLET = "#7C6CCF"
RED = "#B64342"
GREY = "#767676"
LIGHT_GREY = "#D9D9D9"


def arm_pair(left: int, right: int) -> str:
    return "-".join(sorted((ARM_BY_VINYL[int(left)], ARM_BY_VINYL[int(right)])))


def read_replica_summary(audit: Path) -> pd.DataFrame:
    return pd.read_csv(audit / "pair_replica_summary.csv")


def aggregate_arm_pairs(replicas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cross = replicas[replicas["topology"] == "cross_arm"].copy()
    cross["arm_pair"] = [
        arm_pair(left, right) for left, right in zip(cross["left"], cross["right"])
    ]
    by_replica = (
        cross.groupby(["replica", "arm_pair"], as_index=False)
        .agg(
            soft_q_sum=("soft_q_mean", "sum"),
            hard_5A_sum=("hard_5A_fraction", "sum"),
        )
        .sort_values(["replica", "arm_pair"])
    )
    by_replica["mass_fraction"] = by_replica["soft_q_sum"] / by_replica.groupby(
        "replica"
    )["soft_q_sum"].transform("sum")
    summary = (
        by_replica.groupby("arm_pair", as_index=False)
        .agg(
            soft_q_mean=("soft_q_sum", "mean"),
            soft_q_replica_sd=("soft_q_sum", "std"),
            mass_fraction_mean=("mass_fraction", "mean"),
            mass_fraction_replica_sd=("mass_fraction", "std"),
            hard_5A_sum_mean=("hard_5A_sum", "mean"),
        )
        .sort_values("soft_q_mean", ascending=False)
    )
    return by_replica, summary


def aggregate_arm_degree(by_replica: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for replica, group in by_replica.groupby("replica"):
        degree = {arm: 0.0 for arm in ARMS}
        for row in group.itertuples():
            left, right = row.arm_pair.split("-")
            degree[left] += row.soft_q_sum
            degree[right] += row.soft_q_sum
        total = sum(degree.values())
        for arm, value in degree.items():
            rows.append(
                {
                    "replica": int(replica),
                    "arm": arm,
                    "weighted_degree": value,
                    "degree_fraction": value / total,
                }
            )
    replica_degree = pd.DataFrame(rows)
    return (
        replica_degree.groupby("arm", as_index=False)
        .agg(
            weighted_degree_mean=("weighted_degree", "mean"),
            weighted_degree_replica_sd=("weighted_degree", "std"),
            degree_fraction_mean=("degree_fraction", "mean"),
            degree_fraction_replica_sd=("degree_fraction", "std"),
        )
        .sort_values("weighted_degree_mean", ascending=False)
    )


def topology_summary(replicas: pd.DataFrame) -> pd.DataFrame:
    per_replica = (
        replicas.groupby(["replica", "topology"], as_index=False)
        .agg(
            soft_q_mean=("soft_q_mean", "mean"),
            hard_5A_fraction_mean=("hard_5A_fraction", "mean"),
            distance_mean_A=("distance_mean_A", "mean"),
        )
        .pivot(index="replica", columns="topology")
    )
    rows = []
    for replica in per_replica.index:
        cross = per_replica.loc[replica, ("soft_q_mean", "cross_arm")]
        same = per_replica.loc[replica, ("soft_q_mean", "same_arm")]
        rows.append(
            {
                "replica": int(replica),
                "cross_soft_q_mean": cross,
                "same_soft_q_mean": same,
                "cross_to_same_soft_ratio": cross / same,
                "cross_hard_5A_fraction_mean": per_replica.loc[
                    replica, ("hard_5A_fraction_mean", "cross_arm")
                ],
                "same_hard_5A_fraction_mean": per_replica.loc[
                    replica, ("hard_5A_fraction_mean", "same_arm")
                ],
                "cross_distance_mean_A": per_replica.loc[
                    replica, ("distance_mean_A", "cross_arm")
                ],
                "same_distance_mean_A": per_replica.loc[
                    replica, ("distance_mean_A", "same_arm")
                ],
            }
        )
    return pd.DataFrame(rows)


def windowed_contacts(
    bundle: Path, pairs: pd.DataFrame, window_ps: float = 5000.0
) -> pd.DataFrame:
    pair_topology = pairs.set_index("pair")["topology"].to_dict()
    cross_cols = [pair for pair, topology in pair_topology.items() if topology == "cross_arm"]
    same_cols = [pair for pair, topology in pair_topology.items() if topology == "same_arm"]
    usecols = ["time_ps", *cross_cols, *same_cols]
    rows: list[dict[str, float | int]] = []
    for replica in (1, 2, 3):
        path = (
            bundle
            / "gromacs_dcm/runs/B6_p_C6_q0"
            / f"replica_{replica:02d}"
            / "vinyl_distances.csv"
        )
        totals = {
            index: {"n": 0, "cross": 0.0, "same": 0.0} for index in range(10)
        }
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=10000):
            bins = np.minimum((chunk["time_ps"].to_numpy() // window_ps).astype(int), 9)
            cross = 1.0 / (
                1.0 + np.exp((chunk[cross_cols].to_numpy(dtype=float) - 5.0) / 0.5)
            )
            same = 1.0 / (
                1.0 + np.exp((chunk[same_cols].to_numpy(dtype=float) - 5.0) / 0.5)
            )
            cross_mean = cross.mean(axis=1)
            same_mean = same.mean(axis=1)
            for index in np.unique(bins):
                mask = bins == index
                totals[int(index)]["n"] += int(mask.sum())
                totals[int(index)]["cross"] += float(cross_mean[mask].sum())
                totals[int(index)]["same"] += float(same_mean[mask].sum())
        for index, values in totals.items():
            rows.append(
                {
                    "replica": replica,
                    "window_start_ns": index * window_ps / 1000.0,
                    "window_end_ns": (index + 1) * window_ps / 1000.0,
                    "window_mid_ns": (index + 0.5) * window_ps / 1000.0,
                    "frames": int(values["n"]),
                    "cross_soft_q_mean_per_pair": values["cross"] / values["n"],
                    "same_soft_q_mean_per_pair": values["same"] / values["n"],
                }
            )
    return pd.DataFrame(rows)


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )


def make_figure(
    output: Path,
    arm_summary: pd.DataFrame,
    arm_degree: pd.DataFrame,
    arm_20_vs_50: pd.DataFrame,
    convergence: pd.DataFrame,
    pair_replicas: pd.DataFrame,
    pair_summary: pd.DataFrame,
    windows: pd.DataFrame,
) -> None:
    mpl.rcParams.update(
        {
            "font.size": 6.5,
            "axes.labelsize": 6.5,
            "axes.titlesize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "legend.fontsize": 6,
        }
    )
    fig = plt.figure(figsize=(7.2, 7.1))
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.05, 0.82],
        width_ratios=[1.25, 0.90, 1.15],
        left=0.075,
        right=0.985,
        bottom=0.07,
        top=0.965,
        wspace=0.72,
        hspace=0.82,
    )

    # a: arm-pair mass heatmap (hero panel)
    ax = fig.add_subplot(gs[0:2, 0])
    matrix = np.full((6, 6), np.nan)
    for row in arm_summary.itertuples():
        left, right = row.arm_pair.split("-")
        i, j = ARMS.index(left), ARMS.index(right)
        matrix[i, j] = matrix[j, i] = 100.0 * row.mass_fraction_mean
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=np.nanmax(matrix))
    for i in range(6):
        for j in range(6):
            if i == j:
                ax.add_patch(
                    plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="#F2F2F2")
                )
            elif np.isfinite(matrix[i, j]):
                color = "white" if matrix[i, j] > 10 else "#272727"
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=color)
    ax.set_xticks(range(6), ARMS)
    ax.set_yticks(range(6), ARMS)
    ax.set_xlabel("Arm")
    ax.set_ylabel("Arm")
    ax.set_title("Cross-arm contact mass (%)")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.042, pad=0.025)
    cbar.set_label("% of total cross-arm soft contact")
    add_panel_label(ax, "a")

    # b: convergence decision at 20 and 50 ns.
    sub = gs[0, 1].subgridspec(1, 2, wspace=0.72)
    ax_rhat = fig.add_subplot(sub[0, 0])
    ax_ess = fig.add_subplot(sub[0, 1])
    x = np.arange(len(convergence))
    ax_rhat.plot(x, convergence["max_material_split_rhat"], "o-", color=VIOLET, lw=1.3)
    ax_rhat.axhline(1.05, color=RED, ls="--", lw=0.9)
    ax_rhat.set_xticks(x, ["20", "50"])
    ax_rhat.set_ylabel("Max split-Rhat")
    ax_rhat.set_ylim(1.0, 1.065)
    ax_ess.plot(x, convergence["min_material_combined_ess"], "o-", color=TEAL, lw=1.3)
    ax_ess.axhline(400, color=RED, ls="--", lw=0.9)
    ax_ess.set_xticks(x, ["20", "50"])
    ax_ess.set_ylabel("Min ESS")
    ax_ess.set_ylim(0, max(650, convergence["min_material_combined_ess"].max() * 1.12))
    add_panel_label(ax_rhat, "b")

    # c: stability of arm-pair ranking.
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(arm_20_vs_50["q20"], arm_20_vs_50["q50"], color=BLUE, s=18)
    limit = 1.08 * max(arm_20_vs_50[["q20", "q50"]].to_numpy().max(), 0.01)
    ax.plot([0, limit], [0, limit], color=GREY, ls="--", lw=0.9)
    for row in arm_20_vs_50.nlargest(5, "q50").itertuples():
        ax.text(row.q20, row.q50, f" {row.arm_pair}", fontsize=5.5, va="bottom")
    r = arm_20_vs_50[["q20", "q50"]].corr().iloc[0, 1]
    ax.text(0.04, 0.95, f"Pearson r = {r:.3f}", transform=ax.transAxes, va="top")
    ax.set(
        xlim=(0, limit),
        ylim=(0, limit),
        xlabel="20 ns contact",
        ylabel="50 ns contact",
    )
    ax.set_aspect("equal", adjustable="box")
    add_panel_label(ax, "c")

    # d: weighted arm degree.
    ax = fig.add_subplot(gs[1, 1])
    degree_plot = arm_degree.sort_values("degree_fraction_mean", ascending=False)
    colors = [RED if arm == "C" else BLUE_LIGHT for arm in degree_plot["arm"]]
    ax.bar(
        degree_plot["arm"],
        100 * degree_plot["degree_fraction_mean"],
        yerr=100 * degree_plot["degree_fraction_replica_sd"],
        color=colors,
        edgecolor="#272727",
        linewidth=0.6,
        capsize=2,
    )
    ax.set_ylabel("Weighted degree (%)")
    ax.set_xlabel("Arm")
    ax.set_title("Arm participation")
    add_panel_label(ax, "d")

    # e: endpoint-resolved top contacts, with independent replicas shown.
    ax = fig.add_subplot(gs[1, 2])
    top_pairs = pair_summary.nlargest(10, "soft_q_mean")["pair"].tolist()[::-1]
    y = np.arange(len(top_pairs))
    offsets = {1: -0.16, 2: 0.0, 3: 0.16}
    colors_rep = {1: BLUE, 2: TEAL, 3: VIOLET}
    for replica in (1, 2, 3):
        subset = pair_replicas[
            (pair_replicas["replica"] == replica) & pair_replicas["pair"].isin(top_pairs)
        ].set_index("pair")
        ax.scatter(
            [subset.loc[pair, "soft_q_mean"] for pair in top_pairs],
            y + offsets[replica],
            s=11,
            color=colors_rep[replica],
            label=f"rep {replica}",
        )
    means = pair_summary.set_index("pair")
    ax.scatter(
        [means.loc[pair, "soft_q_mean"] for pair in top_pairs],
        y,
        marker="|",
        s=70,
        linewidth=1.2,
        color="#272727",
        label="mean",
    )
    labels = [pair.replace("_跨臂", "").replace("_", "–") for pair in top_pairs]
    ax.set_yticks(y, labels)
    ax.set_xlabel("Mean sigmoid contact")
    ax.set_title("Leading endpoint pairs")
    ax.legend(
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        columnspacing=0.8,
        handletextpad=0.3,
    )
    add_panel_label(ax, "e")

    # f: slow contact fluctuations across 5 ns windows.
    ax = fig.add_subplot(gs[2, :])
    for replica, color in colors_rep.items():
        subset = windows[windows["replica"] == replica]
        ax.plot(
            subset["window_mid_ns"],
            subset["cross_soft_q_mean_per_pair"],
            marker="o",
            ms=3,
            lw=1.1,
            color=color,
            label=f"rep {replica}",
        )
    ax.axhline(
        windows["cross_soft_q_mean_per_pair"].mean(), color=GREY, lw=0.8, ls="--"
    )
    ax.set(xlabel="Time (ns; 5 ns windows)", ylabel="Cross-arm sigmoid contact per pair")
    ax.set_title("Independent replicas sample persistent, slowly exchanging contact states", pad=5)
    ax.legend(ncol=3, loc="upper center")
    add_panel_label(ax, "f")

    base = output / "figures/figure_b6_p_c6_50ns_summary"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-20ns", type=Path, required=True)
    parser.add_argument("--audit-50ns", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    source = output / "source_data"
    figures = output / "figures"
    source.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rep20 = read_replica_summary(args.audit_20ns)
    rep50 = read_replica_summary(args.audit_50ns)
    by_rep20, arm20 = aggregate_arm_pairs(rep20)
    by_rep50, arm50 = aggregate_arm_pairs(rep50)
    degree50 = aggregate_arm_degree(by_rep50)
    topology50 = topology_summary(rep50)
    pairs50 = pd.read_csv(args.audit_50ns / "pair_state_summary.csv")
    convergence = pd.concat(
        [
            pd.read_csv(args.audit_20ns / "state_convergence_summary.csv"),
            pd.read_csv(args.audit_50ns / "state_convergence_summary.csv"),
        ],
        ignore_index=True,
    )
    arm_compare = arm20[["arm_pair", "soft_q_mean"]].rename(
        columns={"soft_q_mean": "q20"}
    ).merge(
        arm50[["arm_pair", "soft_q_mean"]].rename(columns={"soft_q_mean": "q50"}),
        on="arm_pair",
    )
    arm_compare["fold_50_vs_20"] = arm_compare["q50"] / arm_compare["q20"]
    windows = windowed_contacts(args.bundle, pairs50)

    by_rep50.to_csv(source / "arm_pair_replica_50ns.csv", index=False)
    arm50.to_csv(source / "arm_pair_summary_50ns.csv", index=False)
    degree50.to_csv(source / "arm_degree_summary_50ns.csv", index=False)
    topology50.to_csv(source / "topology_replica_summary_50ns.csv", index=False)
    arm_compare.to_csv(source / "arm_pair_20_vs_50.csv", index=False)
    convergence.to_csv(source / "convergence_20_vs_50.csv", index=False)
    windows.to_csv(source / "windowed_contact_50ns.csv", index=False)
    pairs50.nlargest(20, "soft_q_mean").to_csv(
        source / "top_20_endpoint_pairs_50ns.csv", index=False
    )

    make_figure(
        output,
        arm50,
        degree50,
        arm_compare,
        convergence,
        rep50,
        pairs50,
        windows,
    )
    print(f"Final source data and figure written to {output}")


if __name__ == "__main__":
    main()
