#!/usr/bin/env python3
"""Plot the two umbrella families' biased coverage in the joint distance plane."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_b6_local_q1_joint_mbar import JOBS, OUTPUT, window_data


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 3.8), sharex=True, sharey=True,
                             constrained_layout=True)
    edges = np.linspace(0, 4.0, 81)
    centers = 0.5 * (edges[1:] + edges[:-1])
    for pair_id, ax in enumerate(axes, start=1):
        ax.axvspan(0, 0.50, color="#19698A", alpha=0.08)
        ax.axhspan(0, 0.50, color="#C35636", alpha=0.08)
        for family, job_id in JOBS[pair_id].items():
            states = window_data(pair_id, family, job_id)
            x = np.concatenate([state["x_nm"] for state in states])
            y = np.concatenate([state["y_nm"] for state in states])
            counts, _, _ = np.histogram2d(x, y, bins=(edges, edges))
            density = counts / counts.sum()
            levels = [0.0002, 0.0008]
            color = "#19698A" if family == "A6" else "#C35636"
            ax.contour(centers, centers, density.T, levels=levels, colors=[color],
                       linewidths=[1.3, 2.0], alpha=0.90)
        ax.set_title(f"Paired replica {pair_id}")
        ax.set_xlim(0.3, 3.1)
        ax.set_ylim(0.3, 3.1)
        ax.set_aspect("equal")
        ax.set_xlabel("A1→A6 distance (nm)")
        if pair_id == 1:
            ax.set_ylabel("A1→A2 distance (nm)")
    fig.text(0.5, 1.02, "Biased trajectory coverage: blue A6 umbrellas, orange A2 umbrellas",
             ha="center", va="bottom", fontsize=9)
    fig.savefig(OUTPUT / "joint_sampling_coverage.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT / "joint_sampling_coverage.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
