#!/usr/bin/env python3
"""Create topology-only state schematics and q2 free-energy surfaces."""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch


SOURCE = ROOT / "results/a4_q2_umbrella_2d_analysis/final/q2_2d_surface_source_data.csv"
OUT = ROOT / "results/a4_c5_c8_selectivity_summary_zh/figures"

NAVY = "#183153"
TEAL = "#0E7490"
RED = "#C43C39"
BLUE = "#3568B8"
PALE = "#EAF2F5"
GREY = "#A8B0B7"


def configure() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def export(fig: mpl.figure.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.tiff", dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


ARM_ENDPOINTS = {
    0: np.array([0.0, 1.12]),
    1: np.array([1.12, 0.0]),
    2: np.array([0.0, -1.12]),
    3: np.array([-1.12, 0.0]),
}
SITE_POSITIONS = {
    0: (np.array([-0.28, 1.64]), np.array([0.28, 1.64])),
    1: (np.array([1.64, 0.28]), np.array([1.64, -0.28])),
    2: (np.array([0.28, -1.64]), np.array([-0.28, -1.64])),
    3: (np.array([-1.64, -0.28]), np.array([-1.64, 0.28])),
}


ARM_MAP = {
    "C5": {0: ("V1", "V8"), 1: ("V2", "V3"), 2: ("V4", "V5"), 3: ("V6", "V7")},
    "C8": {0: ("V1", "V2"), 1: ("V3", "V4"), 2: ("V5", "V6"), 3: ("V7", "V8")},
}


def site_lookup(system: str) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for arm, labels in ARM_MAP[system].items():
        for label, position in zip(labels, SITE_POSITIONS[arm]):
            positions[label] = position
    return positions


def draw_edge(ax: mpl.axes.Axes, p1: np.ndarray, p2: np.ndarray, color: str, rad: float) -> None:
    edge = FancyArrowPatch(
        p1,
        p2,
        arrowstyle="-",
        connectionstyle=f"arc3,rad={rad}",
        linewidth=2.8,
        color=color,
        alpha=0.95,
        zorder=1,
    )
    ax.add_patch(edge)


def draw_state(
    ax: mpl.axes.Axes,
    system: str,
    title: str,
    reacted: tuple[tuple[str, str], ...],
) -> None:
    lookup = site_lookup(system)
    consumed = {label for pair in reacted for label in pair}

    ax.add_patch(Circle((0, 0), 0.38, facecolor=NAVY, edgecolor="none", zorder=3))
    ax.text(0, 0, "A4", color="white", weight="bold", ha="center", va="center", fontsize=10, zorder=4)

    for arm, endpoint in ARM_ENDPOINTS.items():
        ax.plot([0, endpoint[0]], [0, endpoint[1]], color=NAVY, lw=5.0, solid_capstyle="round", zorder=2)
        for position in SITE_POSITIONS[arm]:
            ax.plot([endpoint[0], position[0]], [endpoint[1], position[1]], color=NAVY, lw=2.2, zorder=2)
        label_pos = endpoint * 0.74
        ax.text(label_pos[0], label_pos[1], f"arm {arm}", color=NAVY, ha="center", va="center", fontsize=7)

    for edge_index, pair in enumerate(reacted):
        p1, p2 = lookup[pair[0]], lookup[pair[1]]
        # Opposite-side edges need stronger curvature to avoid the central core.
        opposite = np.linalg.norm(p1 + p2) < 0.8
        rad = (0.32 if opposite else 0.18) * (1 if edge_index == 0 else -1)
        draw_edge(ax, p1, p2, RED if edge_index == 0 else BLUE, rad)

    for label, position in lookup.items():
        fill = GREY if label in consumed else "white"
        edge = "white" if label in consumed else TEAL
        ax.add_patch(Circle(position, 0.205, facecolor=fill, edgecolor=edge, linewidth=1.7, zorder=5))
        ax.text(position[0], position[1], label, ha="center", va="center", fontsize=7.5,
                color="white" if label in consumed else NAVY, weight="bold", zorder=6)

    history = "  +  ".join(f"{a}–{b}" for a, b in reacted)
    ax.set_title(title, loc="left", weight="bold", color=NAVY, pad=3)
    ax.text(0, -2.08, f"reacted: {history}", ha="center", va="top", fontsize=8, color="#394B59")
    ax.set_xlim(-2.15, 2.15)
    ax.set_ylim(-2.28, 2.10)
    ax.set_aspect("equal")
    ax.axis("off")


def topology_figure() -> None:
    states = [
        ("C5", "C5 q1", (("V7", "V8"),)),
        ("C8", "C8 q1", (("V6", "V8"),)),
        ("C5", "C5 q2 adjacent", (("V7", "V8"), ("V4", "V6"))),
        ("C8", "C8 q2 adjacent", (("V6", "V8"), ("V3", "V7"))),
        ("C5", "C5 q2 opposite", (("V7", "V8"), ("V3", "V5"))),
        ("C8", "C8 q2 opposite", (("V6", "V8"), ("V1", "V3"))),
        ("C5", "C5 q2 mixed", (("V7", "V8"), ("V2", "V3"))),
        ("C8", "C8 q2 mixed", (("V6", "V8"), ("V1", "V2"))),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(8.2, 13.2))
    for ax, state in zip(axes.flat, states):
        draw_state(ax, *state)
    fig.suptitle("Topology-only definitions of the simulated reaction-history states",
                 x=0.08, ha="left", fontsize=14, weight="bold", color=NAVY, y=0.995)
    fig.text(0.08, 0.975,
             "Four arms and eight vinyl sites; red = first recorded edge, blue = second recorded edge, grey = consumed site.",
             ha="left", va="top", fontsize=9, color="#52606D")
    fig.tight_layout(rect=(0.03, 0.02, 0.98, 0.965), h_pad=1.0, w_pad=0.8)
    export(fig, "a4_state_topology_schematics")


def load_surfaces() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    grouped: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["group"]].append((
                float(row["distance_nm"]),
                float(row["orientation_gate"]),
                float(row["smoothed_pmf_kcal_mol"]),
            ))
    result = {}
    for group, rows in grouped.items():
        distances = np.array(sorted({row[0] for row in rows}))
        orientations = np.array(sorted({row[1] for row in rows}))
        index_d = {value: i for i, value in enumerate(distances)}
        index_o = {value: i for i, value in enumerate(orientations)}
        z = np.full((len(orientations), len(distances)), np.nan)
        for distance, orientation, free_energy in rows:
            z[index_o[orientation], index_d[distance]] = free_energy
        result[group] = distances, orientations, z
    return result


def surface_figure(
    surfaces: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    stem: str,
    heading: str,
    panels: tuple[tuple[str, str], ...],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.6), sharex=True, sharey=True)
    levels = np.linspace(0.0, 6.0, 25)
    mappable = None
    for ax, (group, title) in zip(axes.flat, panels):
        d, orientation, z = surfaces[group]
        mappable = ax.contourf(d, orientation, np.clip(z, 0.0, 6.0), levels=levels,
                               cmap="viridis_r", extend="max")
        ax.contour(d, orientation, np.clip(z, 0.0, 6.0), levels=np.arange(1, 6),
                   colors="white", linewidths=0.45, alpha=0.7)
        ax.axvline(0.50, color="white", lw=1.1, ls="--")
        ax.axhline(0.50, color="white", lw=1.1, ls=":")
        ax.set_title(title, loc="left", weight="bold", color=NAVY)
        ax.set_xlim(0.32, 2.78)
        ax.set_ylim(0.0, 1.0)
        ax.grid(False)
    for ax in axes[-1, :]:
        ax.set_xlabel("Vinyl midpoint distance / nm")
    for ax in axes[:, 0]:
        ax.set_ylabel("Orientation gate score")
    assert mappable is not None
    colorbar = fig.colorbar(mappable, ax=axes, fraction=0.035, pad=0.025)
    colorbar.set_label("Relative free energy / kcal mol$^{-1}$")
    fig.suptitle(heading, x=0.08, ha="left", fontsize=14, weight="bold", color=NAVY, y=0.985)
    fig.text(0.08, 0.94,
             "Reweighted from existing distance-umbrella windows; dashed: 0.50 nm contact center; dotted: orientation score 0.50.",
             ha="left", fontsize=8.5, color="#52606D")
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.10, top=0.87, hspace=0.28, wspace=0.15)
    export(fig, stem)


def main() -> None:
    configure()
    topology_figure()
    surfaces = load_surfaces()
    surface_figure(
        surfaces,
        "q2_adjacent_free_energy_surfaces",
        "q2 adjacent states: same-arm versus cross-arm pathways",
        (
            ("C5_adjacent_same", "C5 same: V2–V3 (arm 1–1)"),
            ("C5_adjacent_cross", "C5 cross: V2–V1 (arm 1–0)"),
            ("C8_adjacent_same", "C8 same: V1–V2 (arm 0–0)"),
            ("C8_adjacent_cross", "C8 cross: V1–V4 (arm 0–1)"),
        ),
    )
    surface_figure(
        surfaces,
        "q2_opposite_free_energy_surfaces",
        "q2 opposite states: reuse versus switch pathways",
        (
            ("C5_opposite_reuse", "C5 reuse: equal mixture of two paths"),
            ("C5_opposite_switch", "C5 switch: V4–V6 (arm 2–3)"),
            ("C8_opposite_reuse", "C8 reuse: V5–V7 (arm 2–3)"),
            ("C8_opposite_switch", "C8 switch: V4–V5 (arm 1–2)"),
        ),
    )


if __name__ == "__main__":
    main()
