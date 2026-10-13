#!/usr/bin/env python3
"""Plot the six completed distance PMFs on their common WHAM reference scale."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DOWNLOAD = PROJECT.parent / "bohrium_download/b6_local_q1_paired_umbrella_complete_20261001"
OUTPUT = PROJECT / "results/b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001"
JOBS = (
    (1, 20806766, 20806768),
    (2, 20806769, 20806770),
    (3, 20806772, 20806773),
)


def read_pmf(job_id: int) -> np.ndarray:
    path = DOWNLOAD / str(job_id) / "extracted/pathway/wham/pmf.xvg"
    return np.loadtxt(path, comments=["#", "@"])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    figure, axes = plt.subplots(3, 1, figsize=(6.8, 7.1), sharex=True, sharey=True,
                               constrained_layout=True)
    for axis, (pair_id, a6_job, a2_job) in zip(axes, JOBS):
        axis.axvspan(0.35, 0.50, color="#B5D7C0", alpha=0.28)
        for job_id, label, color in ((a6_job, "A1→A6", "#19698A"),
                                     (a2_job, "A1→A2", "#C35636")):
            data = read_pmf(job_id)
            mask = (data[:, 0] >= 0.35) & (data[:, 0] <= 2.75)
            axis.plot(data[mask, 0], data[mask, 1], color=color, lw=1.7, label=label)
        axis.axhline(0, color="0.7", lw=0.6)
        axis.set_title(f"Paired 10 ns terminal replica {pair_id}", loc="left", fontsize=9)
        axis.set_ylabel("PMF (kcal mol$^{-1}$)")
        axis.set_xlim(0.35, 2.75)
        axis.set_ylim(-4.5, 10.0)
    axes[0].legend(frameon=False, ncol=2, loc="upper right")
    axes[-1].set_xlabel("Vinyl midpoint distance (nm)")
    figure.savefig(OUTPUT / "paired_distance_pmfs.png", dpi=220)
    figure.savefig(OUTPUT / "paired_distance_pmfs.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
