#!/usr/bin/env python3
"""Distance-orientation reweighting of the four completed C6/C7 q1 paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.analyze_a4_q2_umbrella_2d_reweight import (  # noqa: E402
    DISTANCE_EDGES_NM,
    ORIENTATION_EDGES,
    RT_KCAL,
    PathwayResult,
    analyze_pathway,
    surface_rows,
    write_csv,
)
from analysis.extract_a4_q2_umbrella_qm_candidates import (  # noqa: E402
    extract_solute_gro,
    parse_ndx,
)


PATHWAYS = {
    "c6_q1_reuse_v1_v7": "20726912",
    "c7_q1_reuse_v2_v5": "20726913",
    "c6_q1_switch_v3_v4": "20726914",
    "c7_q1_switch_v1_v8": "20726915",
}
COMPARISONS = (
    ("C6_q1", "c6_q1_reuse_v1_v7", "c6_q1_switch_v3_v4"),
    ("C7_q1", "c7_q1_reuse_v2_v5", "c7_q1_switch_v1_v8"),
)


def prepare_topology(pathway: Path, topology_root: Path) -> None:
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    groups = parse_ndx(pathway / "index.ndx")
    left = int(metadata["left"])
    right = int(metadata["right"])
    pair = groups[f"V{left}_midpoint"] + groups[f"V{right}_midpoint"]
    if len(pair) != 4:
        raise ValueError(f"{pathway}: expected four reactive atoms, got {pair}")
    closest = list((pathway / "windows").glob("window_000_*/production.gro"))
    if len(closest) != 1:
        raise FileNotFoundError(f"{pathway}: expected one closest-window production.gro")
    target = topology_root / str(metadata["pathway"])
    target.mkdir(parents=True, exist_ok=True)
    extract_solute_gro(closest[0], groups["Solute"], target / "solute_topology.gro")
    (target / "reactive_pair.ndx").write_text(
        "[ Reactive_pair ]\n" + " ".join(map(str, pair)) + "\n", encoding="utf-8"
    )


def compare(label: str, reuse: PathwayResult, switch: PathwayResult) -> dict[str, object]:
    def ddg(metric: str, half: str | None = None) -> float:
        left = reuse.metrics if half is None else reuse.half_metrics[half]
        right = switch.metrics if half is None else switch.half_metrics[half]
        return -RT_KCAL * math.log(float(left[metric]) / float(right[metric]))

    record: dict[str, object] = {
        "state": label,
        "contrast": "reuse_minus_switch",
    }
    for prefix, half in (("full", None), ("first_half", "first_half"), ("second_half", "second_half")):
        contact = ddg("soft_contact_probability", half)
        orientation = ddg("conditional_orientation", half)
        nac = ddg("soft_nac_probability", half)
        record[f"{prefix}_ddG_contact_kcal_mol"] = contact
        record[f"{prefix}_ddG_orientation_kcal_mol"] = orientation
        record[f"{prefix}_ddG_nac_kcal_mol"] = nac
        record[f"{prefix}_additivity_residual_kcal_mol"] = nac - contact - orientation
    record["reuse_over_switch_nac_odds"] = math.exp(
        -float(record["full_ddG_nac_kcal_mol"]) / RT_KCAL
    )
    record["alpha_2d_reweight"] = float(record["full_ddG_nac_kcal_mol"]) / RT_KCAL
    record["half_difference_ddG_nac_kcal_mol"] = (
        float(record["second_half_ddG_nac_kcal_mol"])
        - float(record["first_half_ddG_nac_kcal_mol"])
    )
    return record


def plot_summary(output: Path, results: dict[str, PathwayResult], contrasts: list[dict[str, object]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.2, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.1, 0.75))
    panels = (
        ("C6 q1", "c6_q1_reuse_v1_v7", "c6_q1_switch_v3_v4"),
        ("C7 q1", "c7_q1_reuse_v2_v5", "c7_q1_switch_v1_v8"),
    )
    image = None
    for column, (label, reuse_name, switch_name) in enumerate(panels):
        ax = fig.add_subplot(grid[0, column])
        # Display reuse minus switch by placing their smoothed surfaces side by
        # side in the same panel; quantitative values below remain unsmoothed.
        reuse = results[reuse_name].surface_pmf_kcal.T
        switch = results[switch_name].surface_pmf_kcal.T
        combined = np.concatenate((reuse, switch), axis=1)
        image = ax.imshow(
            np.clip(combined, 0, 6), origin="lower", aspect="auto", cmap="magma_r",
            extent=(0.30, 5.30, 0, 1), vmin=0, vmax=6, interpolation="nearest",
        )
        ax.axvline(2.80, color="white", lw=1.0)
        ax.axvline(0.50, color="#2B6CB0", lw=0.8, ls="--")
        ax.axvline(3.00, color="#2B6CB0", lw=0.8, ls="--")
        ax.set_xticks((0.5, 1.5, 2.5, 3.0, 4.0, 5.0), ("0.5", "1.5", "2.5", "0.5", "1.5", "2.5"))
        ax.text(0.25, 0.96, "reuse", transform=ax.transAxes, ha="center", va="top", fontweight="bold")
        ax.text(0.75, 0.96, "switch", transform=ax.transAxes, ha="center", va="top", fontweight="bold")
        ax.set_title(label)
        ax.set_xlabel("vinyl midpoint distance (nm)")
        if column == 0:
            ax.set_ylabel("orientation gate")
    if image is not None:
        bar = fig.colorbar(image, ax=fig.axes[:2], shrink=0.8, pad=0.02)
        bar.set_label("reweighted F(d, Ω) (kcal mol$^{-1}$)")

    ax = fig.add_subplot(grid[1, :])
    y = np.arange(2)
    labels = ("C6 q1", "C7 q1")
    contact = np.array([float(row["full_ddG_contact_kcal_mol"]) for row in contrasts])
    orient = np.array([float(row["full_ddG_orientation_kcal_mol"]) for row in contrasts])
    total = np.array([float(row["full_ddG_nac_kcal_mol"]) for row in contrasts])
    first = np.array([float(row["first_half_ddG_nac_kcal_mol"]) for row in contrasts])
    second = np.array([float(row["second_half_ddG_nac_kcal_mol"]) for row in contrasts])
    ax.barh(y, contact, color="#4C78A8", height=0.48, label="contact")
    ax.barh(y, orient, left=contact, color="#E69F00", height=0.48, label="orientation")
    ax.scatter(total, y, color="black", marker="D", s=24, zorder=4, label="total NAC")
    for i in range(2):
        ax.plot([first[i], second[i]], [i + 0.20, i + 0.20], color="0.45", lw=1)
        ax.scatter([first[i], second[i]], [i + 0.20, i + 0.20], c=["#56B4E9", "#D55E00"], s=15)
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("reuse − switch ΔΔG (kcal mol$^{-1}$); negative favors reuse")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    base = output / "q1_2d_reweight_summary"
    fig.savefig(base.with_suffix(".png"), dpi=350)
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=PROJECT)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_umbrella_2d_analysis",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology_root = output / "reactive_pair_topologies"
    download = workspace / "results/a4_m_c6_c7_q1_umbrella_v2_download"

    results: dict[str, PathwayResult] = {}
    pathway_rows: list[dict[str, object]] = []
    for name, job_id in PATHWAYS.items():
        pathway = download / job_id / "extracted/pathway"
        prepare_topology(pathway, topology_root)
        result = analyze_pathway(pathway, workspace, output, args.gmx, topology_root)
        results[name] = result
        pathway_rows.append({
            "pathway": name,
            "state": result.metadata["state"],
            "system": result.metadata["system"],
            "route_family": result.metadata["route_family"],
            "windows": len(result.centers_nm),
            "frames": len(result.distance_nm),
            "mbar_iterations": result.mbar_iterations,
            "mbar_residual": result.mbar_residual,
            "effective_sample_size": result.effective_sample_size,
            "geometry_pull_rms_A": result.geometry_pull_rms_A,
            "geometry_pull_max_A": result.geometry_pull_max_A,
            **result.metrics,
            **{
                f"{half}_{key}": value
                for half, values in result.half_metrics.items() for key, value in values.items()
            },
        })
        print(f"analyzed={name} windows={len(result.centers_nm)} frames={len(result.distance_nm)}")

    contrasts = [compare(label, results[reuse], results[switch]) for label, reuse, switch in COMPARISONS]
    groups = {
        name: {"probability": result.surface_probability, "pmf": result.surface_pmf_kcal}
        for name, result in results.items()
    }
    write_csv(output / "pathway_reweight_summary.csv", pathway_rows)
    write_csv(output / "q1_2d_reweighted_contrasts.csv", contrasts)
    write_csv(output / "q1_2d_surface_source_data.csv", surface_rows(groups))
    plot_summary(output, results, contrasts)
    summary = {
        "temperature_K": 313.15,
        "method": "0.002-nm binned WHAM/MBAR-equivalent reweighting with frame-resolved orientation",
        "scope": "four prescreened representative q1 pathways; quantitative surfaces are unsmoothed",
        "pathways": pathway_rows,
        "contrasts": contrasts,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(contrasts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
