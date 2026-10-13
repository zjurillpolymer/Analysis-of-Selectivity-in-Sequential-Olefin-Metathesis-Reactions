from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

OUT = Path(__file__).resolve().parents[1] / "reports/phase_summary"
OUT.mkdir(parents=True, exist_ok=True)

def draw_panel(ax, reacted=False):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.05, 2.05)
    center = np.array([0.0, 0.0])
    ax.add_patch(Circle(center, 0.18, facecolor="#30343b", edgecolor="none", zorder=4))
    ax.text(0, 0, "B6", color="white", ha="center", va="center", fontsize=8, zorder=5)
    vinyl = {}
    for arm in range(6):
        theta = np.pi / 2 - arm * np.pi / 3
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-np.sin(theta), np.cos(theta)])
        base = 0.25 * u
        end = 1.30 * u
        ax.plot([base[0], end[0]], [base[1], end[1]], color="#4c566a", lw=2.2, zorder=1)
        pair_center = end + 0.25 * u
        p1 = pair_center + 0.18 * v
        p2 = pair_center - 0.18 * v
        n1, n2 = 2 * arm + 1, 2 * arm + 2
        vinyl[n1], vinyl[n2] = p1, p2
        for n, p in ((n1, p1), (n2, p2)):
            ax.add_patch(Circle(p, 0.105, facecolor="#f2c14e", edgecolor="#8a5a00", lw=1.2, zorder=3))
            ax.text(*(p + 0.16 * u), f"V{n}", fontsize=7, ha="center", va="center", color="#5a4300")
        label = end + 0.54 * u
        ax.text(*label, f"arm {arm + 1}", fontsize=8, ha="center", va="center", color="#30343b")
    if reacted:
        p, q = vinyl[1], vinyl[11]
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#d1495b", lw=3.0, zorder=2)
        ax.text(0, 1.88, "q0: arm1–arm6 cross", color="#b02a3c", fontsize=10,
                ha="center", va="center", fontweight="bold")
        ax.text(*(p + 0.12 * np.array([-0.35, 0.95])), "reacted", fontsize=7,
                color="#b02a3c", ha="center")
        for n in (2, 12):
            pos = vinyl[n]
            ax.add_patch(Circle(pos, 0.125, facecolor="#8ecae6", edgecolor="#126782", lw=1.5, zorder=4))
            ax.text(*pos, f"V{n}", fontsize=7, ha="center", va="center", color="#073b4c", zorder=5)
        ax.text(0, -1.93, "blue: remaining vinyl sites", fontsize=8,
                ha="center", color="#126782")
    else:
        ax.text(0, 1.88, "unreacted B6", color="#30343b", fontsize=10,
                ha="center", va="center", fontweight="bold")

fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=220)
draw_panel(axes[0], reacted=False)
draw_panel(axes[1], reacted=True)
fig.subplots_adjust(wspace=0.02)
fig.savefig(OUT / "B6_original_and_postq0_schematic.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "B6_original_and_postq0_schematic.svg", bbox_inches="tight")

