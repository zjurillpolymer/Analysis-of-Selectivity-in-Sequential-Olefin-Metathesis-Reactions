#!/usr/bin/env python3
"""Draw audit schematics for the corrected B6 q0/q1 local competitors."""
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/b6_local_q1_arm1_competition_20260923"
OUT.mkdir(parents=True, exist_ok=True)

arms = {1: (1, 2), 2: (3, 4), 3: (5, 6), 4: (7, 8), 5: (9, 10), 6: (11, 12)}
angles = {i: np.deg2rad(90 - (i - 1) * 60) for i in range(1, 7)}
centers = {i: np.array([np.cos(a), np.sin(a)]) for i, a in angles.items()}

def vinyl_pos(arm, which):
    c = centers[arm]
    radial = c / np.linalg.norm(c)
    tang = np.array([-radial[1], radial[0]])
    return c * 0.64 + radial * 0.34 + tang * (0.12 if which == 0 else -0.12)

def draw(ax, q1, title):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)
    # central six-arm scaffold
    poly = np.array([centers[i] * 0.64 for i in range(1, 7)] + [centers[1] * 0.64])
    ax.plot(poly[:, 0], poly[:, 1], color="#8c8c8c", lw=1.2, zorder=0)
    ax.scatter([0], [0], s=500, c="#eeeeee", edgecolors="#555555", zorder=1)
    ax.text(0, 0, "B6", ha="center", va="center", fontsize=11, weight="bold")
    # arm nodes and two vinyl endpoints
    for arm, labels in arms.items():
        c = centers[arm] * 0.64
        ax.plot([0, c[0]], [0, c[1]], color="#bdbdbd", lw=1.0, zorder=0)
        ax.scatter([c[0]], [c[1]], s=240, c="#d9edf7", edgecolors="#31708f", zorder=2)
        ax.text(c[0], c[1], f"A{arm}", ha="center", va="center", fontsize=9, weight="bold")
        for k, v in enumerate(labels):
            p = vinyl_pos(arm, k)
            ax.scatter([p[0]], [p[1]], s=150, c="white", edgecolors="#333333", zorder=3)
            ax.text(p[0], p[1], f"V{v}", ha="center", va="center", fontsize=7)
    # q0 and q1 reaction bonds
    def endpoint(v):
        arm = (v - 1) // 2 + 1
        which = (v - 1) % 2
        return vinyl_pos(arm, which)
    p, r = endpoint(1), endpoint(12)
    ax.plot([p[0], r[0]], [p[1], r[1]], color="#222222", lw=3.0, zorder=4, label="q0: V1–V12")
    p, r = endpoint(q1[0]), endpoint(q1[1])
    ax.plot([p[0], r[0]], [p[1], r[1]], color="#d62728", lw=3.0, zorder=5, label=f"q1: V{q1[0]}–V{q1[1]} (cis)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=8)
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.35, 1.35)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, q1, title in zip(
    axes,
    [(2, 11), (2, 3), (2, 4)],
    ["A1→A6: local reuse", "A1→A2: endpoint V3", "A1→A2: endpoint V4"],
):
    draw(ax, q1, title)
fig.suptitle("B6 corrected local q1 competition after fixed q0 = A1–A6", fontsize=15, y=0.98)
fig.tight_layout(rect=(0, 0.03, 1, 0.94))
fig.savefig(OUT / "b6_local_q1_competition_templates.png", dpi=220)
fig.savefig(OUT / "b6_local_q1_competition_templates.svg")
plt.close(fig)

(OUT / "template_manifest.txt").write_text(
    "Authoritative arms: A1=(V1,V2), A2=(V3,V4), A3=(V5,V6), "
    "A4=(V7,V8), A5=(V9,V10), A6=(V11,V12).\n"
    "Fixed q0: V1-V12 (A1-A6).\n"
    "q1 candidates: V2-V11, V2-V3, V2-V4. Red bond is the new cis alkene.\n",
    encoding="utf-8",
)
print(OUT / "b6_local_q1_competition_templates.png")
