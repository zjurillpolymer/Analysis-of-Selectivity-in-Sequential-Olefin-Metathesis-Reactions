#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/b6_postq0_unbiased_30ns_20261011"
OUT = BASE / "timescale_analysis"
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8, "axes.spines.right": False, "axes.spines.top": False, "svg.fonttype": "none", "pdf.fonttype": 42})
colors = {"seed_05000": "#3B6EA8", "seed_12500": "#C7772F", "seed_19500": "#4A8B63"}

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    obs = {p.name.replace("_observables_2ps.csv.gz", ""): pd.read_csv(p) for p in OUT.glob("*_observables_2ps.csv.gz")}
    fig, axes = plt.subplots(4, 1, figsize=(7.0, 7.0), sharex=True)
    cols = [("V2-V11_nm", "V2–V11 (nm)"), ("V2-V3_nm", "V2–V3 (nm)"), ("V2-V4_nm", "V2–V4 (nm)"), ("Rg_nm", "Rg (nm)")]
    for ax, (col, label) in zip(axes, cols):
        for name, d in obs.items():
            # 2 ps raw output is retained in CSV; thin only the rendered line.
            ax.plot(d.time_ps / 1000, d[col], lw=0.45, alpha=0.75, color=colors[name], label=name)
        ax.set_ylabel(label)
        ax.grid(axis="y", color="#dddddd", lw=0.4)
    axes[0].legend(ncol=3, frameon=False, loc="upper right")
    axes[-1].set_xlabel("time (ns)")
    fig.suptitle("B6 post-q0 unbiased MD: reaction geometry and global size", y=0.995)
    save(fig, "b6_postq0_30ns_observables")

    blocks = pd.read_csv(OUT / "5ns_block_means.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    for ax, (col, label) in zip(axes.flat, cols):
        for name, d in blocks.groupby("replica"):
            ax.plot(d.block_ps.str.split("-").str[0].astype(float) / 1000, d[f"{col}_mean"], marker="o", ms=3, lw=1.1, color=colors[name], label=name)
        ax.set_title(label, loc="left", fontsize=8)
        ax.set_xlabel("block start (ns)")
        ax.grid(axis="y", color="#dddddd", lw=0.4)
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.suptitle("Five-nanosecond block means: residual drift and replica dependence", y=1.0)
    fig.tight_layout()
    save(fig, "b6_postq0_30ns_block_means")

if __name__ == "__main__":
    main()
