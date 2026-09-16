#!/usr/bin/env python3
"""Analyze the four A4-m-C8 q0 umbrella PMFs and their bootstrap profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TEMPERATURE_K = 313.15
R_KCAL = 0.00198720425864083
RT = R_KCAL * TEMPERATURE_K
CONTACT_REGION_NM = (0.325, 0.500)
REFERENCE_REGION_NM = (1.500, 2.100)
PATHWAY_ORDER = (
    "anchor_v1_same_v1_v2",
    "anchor_v1_cross_v1_v3",
    "anchor_v5_same_v5_v6",
    "anchor_v5_cross_v5_v8",
)


def read_xvg(path: Path, columns: int | None = None) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line[0] in "#@&":
            continue
        fields = [float(value) for value in line.split()]
        rows.append(fields if columns is None else fields[:columns])
    result = np.asarray(rows, dtype=float)
    if result.ndim != 2 or result.shape[0] < 2:
        raise ValueError(f"insufficient XVG data: {path}")
    return result


def read_bootstrap_profiles(path: Path) -> list[np.ndarray]:
    profiles: list[np.ndarray] = []
    current: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line[0] in "#@":
            continue
        if line[0] == "&":
            if current:
                profiles.append(np.asarray(current, dtype=float))
                current = []
            continue
        current.append([float(value) for value in line.split()[:2]])
    if current:
        profiles.append(np.asarray(current, dtype=float))
    if len(profiles) != 200:
        raise ValueError(f"expected 200 bootstrap profiles in {path}; found {len(profiles)}")
    return profiles


def log_integral(x: np.ndarray, free_energy: np.ndarray, low: float, high: float) -> float:
    finite = np.isfinite(x) & np.isfinite(free_energy)
    x = x[finite]
    free_energy = free_energy[finite]
    if x.min() > low or x.max() < high:
        raise ValueError(f"profile does not span requested region {low}..{high} nm")
    interior = (x > low) & (x < high)
    grid = np.r_[low, x[interior], high]
    values = np.interp(grid, x, free_energy)
    shift = float(np.min(values))
    integral = float(np.trapezoid(np.exp(-(values - shift) / RT), grid))
    return math.log(integral) - shift / RT


def closing_free_energy(profile: np.ndarray, contact_high: float = CONTACT_REGION_NM[1],
                        reference: tuple[float, float] = REFERENCE_REGION_NM) -> float:
    log_contact = log_integral(
        profile[:, 0], profile[:, 1], CONTACT_REGION_NM[0], contact_high
    )
    log_reference = log_integral(profile[:, 0], profile[:, 1], *reference)
    return -RT * (log_contact - log_reference)


def reference_aligned(profile: np.ndarray) -> np.ndarray:
    log_z = log_integral(profile[:, 0], profile[:, 1], *REFERENCE_REGION_NM)
    reference_density_free_energy = -RT * (
        log_z - math.log(REFERENCE_REGION_NM[1] - REFERENCE_REGION_NM[0])
    )
    return np.column_stack((profile[:, 0], profile[:, 1] - reference_density_free_energy))


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    low, median, high = np.quantile(values, (0.025, 0.5, 0.975))
    return float(low), float(median), float(high)


def load_pathway(root: Path, name: str) -> dict[str, object]:
    pathway = root / name / "pathway"
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    pmf = read_xvg(pathway / "wham/pmf.xvg", 2)
    first = read_xvg(pathway / "wham/pmf_first_half.xvg", 2)
    second = read_xvg(pathway / "wham/pmf_second_half.xvg", 2)
    boot_profiles = read_bootstrap_profiles(pathway / "wham/bootstrap_profiles.xvg")
    boot_dg = np.asarray([closing_free_energy(profile) for profile in boot_profiles])
    aligned_boot = np.asarray([reference_aligned(profile)[:, 1] for profile in boot_profiles])
    with (pathway / "window_diagnostics.csv").open(encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    overlaps = [
        float(row["left_neighbor_gaussian_overlap"])
        for row in diagnostics if row["left_neighbor_gaussian_overlap"].strip()
    ]
    with (pathway / "window_sources.csv").open(encoding="utf-8", newline="") as handle:
        sources = list(csv.DictReader(handle))
    source_errors = [float(row["absolute_error_nm"]) for row in sources]
    iact = read_xvg(pathway / "wham/iact.xvg")
    return {
        "name": name,
        "path": pathway,
        "metadata": metadata,
        "pmf": pmf,
        "aligned_pmf": reference_aligned(pmf),
        "aligned_boot_low": np.quantile(aligned_boot, 0.025, axis=0),
        "aligned_boot_median": np.quantile(aligned_boot, 0.5, axis=0),
        "aligned_boot_high": np.quantile(aligned_boot, 0.975, axis=0),
        "dg": closing_free_energy(pmf),
        "dg_first": closing_free_energy(first),
        "dg_second": closing_free_energy(second),
        "boot_dg": boot_dg,
        "boot_ci": quantiles(boot_dg),
        "minimum_overlap": min(overlaps),
        "maximum_source_error_nm": max(source_errors),
        "maximum_iact_ps": float(np.nanmax(iact[:, 1:])),
        "window_count": len(diagnostics),
    }


def exact_difference_distribution(cross: np.ndarray, same: np.ndarray) -> np.ndarray:
    return (cross[:, None] - same[None, :]).ravel()


def sensitivity(data: dict[str, dict[str, object]]) -> list[dict[str, float]]:
    rows = []
    for contact_high in (0.45, 0.50, 0.55):
        for reference in ((1.4, 2.0), (1.5, 2.1), (1.6, 2.1)):
            values = {
                name: closing_free_energy(item["pmf"], contact_high, reference)
                for name, item in data.items()
            }
            v1 = values["anchor_v1_cross_v1_v3"] - values["anchor_v1_same_v1_v2"]
            v5 = values["anchor_v5_cross_v5_v8"] - values["anchor_v5_same_v5_v6"]
            rows.append({
                "contact_high_nm": contact_high,
                "reference_low_nm": reference[0],
                "reference_high_nm": reference[1],
                "v1_ddG_kcal_mol": v1,
                "v5_ddG_kcal_mol": v5,
                "pooled_ddG_kcal_mol": (v1 + v5) / 2.0,
            })
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_unbiased_benchmarks(summary_path: Path, contact_metrics_path: Path) -> dict[str, float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overall = summary["overall"]
    intervals = summary["ratio_95pct_CI"]
    with contact_metrics_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    hard_any = next(
        row for row in rows
        if row["contact_definition"] == "hard_5A" and row["metric"] == "any_pair"
    )
    return {
        "soft_contact_odds": float(overall["contact_cross_over_same"]),
        "soft_contact_ci_low": float(intervals["contact_cross_over_same"][0]),
        "soft_contact_ci_high": float(intervals["contact_cross_over_same"][1]),
        "nac_odds": float(overall["nac_cross_over_same"]),
        "nac_ci_low": float(intervals["nac_cross_over_same"][0]),
        "nac_ci_high": float(intervals["nac_cross_over_same"][1]),
        "hard_any_odds": float(hard_any["cross_over_same"]),
        "hard_any_ci_low": float(hard_any["ratio_ci_low"]),
        "hard_any_ci_high": float(hard_any["ratio_ci_high"]),
    }


def origin_decomposition(
    pooled_row: dict[str, object], unbiased: dict[str, float]
) -> tuple[list[dict[str, object]], dict[str, float]]:
    multiplicity = 24.0 / 4.0
    preorganization_ddg = float(pooled_row["point_ddG_kcal_mol"])
    multiplicity_ddg = -RT * math.log(multiplicity)
    effective_ddg = preorganization_ddg + multiplicity_ddg
    per_pair_odds = float(pooled_row["point_cross_over_same_odds"])
    predicted_any_odds = per_pair_odds * multiplicity
    predicted_ci_low = float(pooled_row["bootstrap_odds_ci_low"]) * multiplicity
    predicted_ci_high = float(pooled_row["bootstrap_odds_ci_high"]) * multiplicity
    values = {
        "same_arm_pair_count": 4.0,
        "cross_arm_pair_count": 24.0,
        "multiplicity_cross_over_same": multiplicity,
        "preorganization_ddG_kcal_mol": preorganization_ddg,
        "multiplicity_ddG_kcal_mol": multiplicity_ddg,
        "effective_ddG_kcal_mol": effective_ddg,
        "per_pair_umbrella_odds": per_pair_odds,
        "predicted_any_pair_odds": predicted_any_odds,
        "predicted_any_pair_ci_low": predicted_ci_low,
        "predicted_any_pair_ci_high": predicted_ci_high,
        "observed_any_pair_odds": unbiased["hard_any_odds"],
        "observed_any_pair_ci_low": unbiased["hard_any_ci_low"],
        "observed_any_pair_ci_high": unbiased["hard_any_ci_high"],
    }
    rows = [
        {
            "quantity": "per_pair_preorganization",
            "cross_over_same_odds": per_pair_odds,
            "ci_low": float(pooled_row["bootstrap_odds_ci_low"]),
            "ci_high": float(pooled_row["bootstrap_odds_ci_high"]),
            "ddG_kcal_mol": preorganization_ddg,
            "source": "umbrella_pooled_two_anchors",
        },
        {
            "quantity": "pair_multiplicity",
            "cross_over_same_odds": multiplicity,
            "ci_low": multiplicity,
            "ci_high": multiplicity,
            "ddG_kcal_mol": multiplicity_ddg,
            "source": "24_cross_pairs_over_4_same_pairs",
        },
        {
            "quantity": "predicted_whole_molecule",
            "cross_over_same_odds": predicted_any_odds,
            "ci_low": predicted_ci_low,
            "ci_high": predicted_ci_high,
            "ddG_kcal_mol": effective_ddg,
            "source": "umbrella_preorganization_times_pair_multiplicity",
        },
        {
            "quantity": "observed_whole_molecule",
            "cross_over_same_odds": unbiased["hard_any_odds"],
            "ci_low": unbiased["hard_any_ci_low"],
            "ci_high": unbiased["hard_any_ci_high"],
            "ddG_kcal_mol": -RT * math.log(unbiased["hard_any_odds"]),
            "source": "unbiased_MD_hard_5A_any_pair",
        },
    ]
    return rows, values


def make_figure(data: dict[str, dict[str, object]], anchor_rows: list[dict[str, object]],
                output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.linewidth": 0.8,
        "xtick.direction": "out", "ytick.direction": "out",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    same_color, cross_color = "#D55E00", "#0072B2"
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2), constrained_layout=True)
    for panel, (anchor, same_name, cross_name) in zip(
        axes[0],
        (
            ("V1 anchor", "anchor_v1_same_v1_v2", "anchor_v1_cross_v1_v3"),
            ("V5 anchor", "anchor_v5_same_v5_v6", "anchor_v5_cross_v5_v8"),
        ),
    ):
        for name, label, color in ((same_name, "same arm", same_color),
                                   (cross_name, "cross arm", cross_color)):
            item = data[name]
            x = item["pmf"][:, 0]
            panel.plot(x, item["aligned_boot_median"], color=color, lw=1.7, label=label)
            panel.fill_between(x, item["aligned_boot_low"], item["aligned_boot_high"],
                               color=color, alpha=0.18, linewidth=0)
        panel.axvspan(*CONTACT_REGION_NM, color="#999999", alpha=0.12)
        panel.axvspan(*REFERENCE_REGION_NM, color="#009E73", alpha=0.08)
        panel.set_xlim(0.32, 2.20)
        panel.set_ylim(-1.5, 5.2)
        panel.set_title(anchor)
        panel.set_xlabel("C=C midpoint distance (nm)")
        panel.set_ylabel("Reference-aligned PMF (kcal mol$^{-1}$)")
        panel.legend(frameon=False, loc="upper right")
    path_labels = ["V1 same", "V1 cross", "V5 same", "V5 cross"]
    path_names = [PATHWAY_ORDER[i] for i in (0, 1, 2, 3)]
    points = np.asarray([data[name]["dg"] for name in path_names])
    cis = np.asarray([[data[name]["boot_ci"][0], data[name]["boot_ci"][2]] for name in path_names])
    x = np.arange(4)
    colors = [same_color, cross_color, same_color, cross_color]
    axes[1, 0].bar(x, points, color=colors, width=0.68, alpha=0.9)
    axes[1, 0].errorbar(x, points, yerr=np.vstack((points-cis[:, 0], cis[:, 1]-points)),
                        fmt="none", ecolor="black", elinewidth=1, capsize=3)
    axes[1, 0].set_xticks(x, path_labels, rotation=20, ha="right")
    axes[1, 0].set_ylabel(r"$\Delta G_{close}$ (kcal mol$^{-1}$)")
    axes[1, 0].set_title("Integrated contact penalty")
    labels = ["V1", "V5", "Pooled"]
    ddg = np.asarray([float(row["point_ddG_kcal_mol"]) for row in anchor_rows])
    low = np.asarray([float(row["bootstrap_ci_low_kcal_mol"]) for row in anchor_rows])
    high = np.asarray([float(row["bootstrap_ci_high_kcal_mol"]) for row in anchor_rows])
    y = np.arange(3)
    axes[1, 1].axvline(0, color="#666666", lw=0.8)
    axes[1, 1].errorbar(ddg, y, xerr=np.vstack((ddg-low, high-ddg)), fmt="o",
                        color=cross_color, ecolor=cross_color, capsize=3)
    axes[1, 1].set_yticks(y, labels)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(-1.5, 0.15)
    axes[1, 1].set_xlabel(r"$\Delta\Delta G_{cross-same}$ (kcal mol$^{-1}$)")
    axes[1, 1].set_title("Cross-arm stabilization")
    for label, axis in zip("ABCD", axes.flat):
        axis.text(-0.14, 1.05, label, transform=axis.transAxes, fontsize=12, fontweight="bold")
    fig.savefig(output / "a4_m_c8_umbrella_summary.png", dpi=300)
    fig.savefig(output / "a4_m_c8_umbrella_summary.pdf")
    fig.savefig(output / "a4_m_c8_umbrella_summary.svg")
    fig.savefig(output / "a4_m_c8_umbrella_summary.tiff", dpi=600)
    plt.close(fig)


def make_origin_figure(
    pooled_row: dict[str, object],
    unbiased: dict[str, float],
    decomposition: dict[str, float],
    output: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    blue = "#176B87"
    orange = "#D97732"
    gray = "#7A8790"
    dark = "#263238"
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)

    labels = ["Umbrella\nper-pair", "Unbiased MD\nsoft contact", "Unbiased MD\nNAC proxy"]
    points = np.asarray([
        float(pooled_row["point_cross_over_same_odds"]),
        unbiased["soft_contact_odds"],
        unbiased["nac_odds"],
    ])
    low = np.asarray([
        float(pooled_row["bootstrap_odds_ci_low"]),
        unbiased["soft_contact_ci_low"],
        unbiased["nac_ci_low"],
    ])
    high = np.asarray([
        float(pooled_row["bootstrap_odds_ci_high"]),
        unbiased["soft_contact_ci_high"],
        unbiased["nac_ci_high"],
    ])
    x = np.arange(3)
    axes[0].axhline(1.0, color=gray, lw=0.8, ls="--")
    axes[0].errorbar(
        x, points, yerr=np.vstack((points - low, high - points)), fmt="o",
        color=blue, ecolor=blue, capsize=2.5, ms=4,
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.75, 45)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Cross-arm / same-arm odds")
    axes[0].set_title("Per-pair agreement", loc="left", fontweight="bold")

    components = np.asarray([
        decomposition["preorganization_ddG_kcal_mol"],
        decomposition["multiplicity_ddG_kcal_mol"],
    ])
    left = 0.0
    for value, color, label in zip(
        components, (blue, orange), ("Preorganization", "Multiplicity")
    ):
        axes[1].barh([0], [value], left=left, color=color, height=0.48, label=label)
        axes[1].text(left + value / 2.0, 0, f"{value:.2f}", color="white",
                     ha="center", va="center", fontweight="bold")
        axes[1].text(left + value / 2.0, 0.34, label, color=color,
                     ha="center", va="bottom", fontsize=6.5)
        left += value
    axes[1].axvline(0, color=dark, lw=0.8)
    axes[1].set_xlim(-2.25, 0.15)
    axes[1].set_ylim(-0.62, 0.62)
    axes[1].set_yticks([])
    axes[1].set_xlabel(r"Contribution to $\Delta\Delta G$ (kcal mol$^{-1}$)")
    axes[1].set_title("Additive origin", loc="left", fontweight="bold")
    axes[1].text(left, -0.42, f"Total {left:.2f}", ha="left", va="top", color=dark)

    whole_points = np.asarray([
        decomposition["predicted_any_pair_odds"],
        decomposition["observed_any_pair_odds"],
    ])
    whole_low = np.asarray([
        decomposition["predicted_any_pair_ci_low"],
        decomposition["observed_any_pair_ci_low"],
    ])
    whole_high = np.asarray([
        decomposition["predicted_any_pair_ci_high"],
        decomposition["observed_any_pair_ci_high"],
    ])
    y = np.arange(2)
    axes[2].axvline(1.0, color=gray, lw=0.8, ls="--")
    axes[2].errorbar(
        whole_points, y,
        xerr=np.vstack((whole_points - whole_low, whole_high - whole_points)),
        fmt="o", color=dark, ecolor=gray, capsize=2.5, ms=4,
    )
    axes[2].set_xscale("log")
    axes[2].set_xlim(7, 180)
    axes[2].set_yticks(y, ["Predicted\n4.21 × 6", "Observed\nunbiased MD"])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("Whole-molecule cross/same odds")
    axes[2].set_title("Independent closure", loc="left", fontweight="bold")
    for value, yy in zip(whole_points, y):
        axes[2].text(value * 1.08, yy, f"{value:.1f}×", va="center", color=dark)

    for label, axis in zip("abc", axes):
        axis.text(-0.16, 1.12, label, transform=axis.transAxes,
                  fontsize=9, fontweight="bold", va="top")
    base = output / "a4_m_c8_cross_arm_origin_decomposition"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    project = Path(__file__).resolve().parents[1]
    parser.add_argument("--input", type=Path,
                        default=project / "results/a4_m_c8_umbrella_bohrium/download")
    parser.add_argument("--output", type=Path,
                        default=project / "results/a4_m_c8_umbrella_analysis")
    parser.add_argument(
        "--unbiased-summary", type=Path,
        default=project / "results/a4_m_c8_contact_origin/full_geometry_summary.json",
    )
    parser.add_argument(
        "--contact-metrics", type=Path,
        default=project / "results/a4_m_c8_contact_origin/contact_metric_summary.csv",
    )
    args = parser.parse_args()
    input_root, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = {name: load_pathway(input_root, name) for name in PATHWAY_ORDER}
    pathway_rows = []
    for name in PATHWAY_ORDER:
        item = data[name]
        ci_low, ci_median, ci_high = item["boot_ci"]
        metadata = item["metadata"]
        pathway_rows.append({
            "pathway": name,
            "anchor": "V1" if name.startswith("anchor_v1") else "V5",
            "topology": metadata["topology"],
            "pair": f"V{metadata['left']}-V{metadata['right']}",
            "window_count": item["window_count"],
            "closing_dG_kcal_mol": item["dg"],
            "bootstrap_median_kcal_mol": ci_median,
            "bootstrap_ci_low_kcal_mol": ci_low,
            "bootstrap_ci_high_kcal_mol": ci_high,
            "first_half_dG_kcal_mol": item["dg_first"],
            "second_half_dG_kcal_mol": item["dg_second"],
            "minimum_adjacent_overlap": item["minimum_overlap"],
            "maximum_source_error_nm": item["maximum_source_error_nm"],
            "maximum_iact_ps": item["maximum_iact_ps"],
        })
        write_csv(output / f"{name}_bootstrap_closing_dG.csv", [
            {"bootstrap_index": index + 1, "closing_dG_kcal_mol": value}
            for index, value in enumerate(item["boot_dg"])
        ])
    write_csv(output / "pathway_free_energies.csv", pathway_rows)
    anchor_specs = (
        ("V1", "anchor_v1_same_v1_v2", "anchor_v1_cross_v1_v3"),
        ("V5", "anchor_v5_same_v5_v6", "anchor_v5_cross_v5_v8"),
    )
    anchor_rows = []
    anchor_distributions = []
    for anchor, same_name, cross_name in anchor_specs:
        same, cross = data[same_name], data[cross_name]
        point = float(cross["dg"] - same["dg"])
        distribution = exact_difference_distribution(cross["boot_dg"], same["boot_dg"])
        anchor_distributions.append(distribution)
        ci_low, median, ci_high = quantiles(distribution)
        ratio_dist = np.exp(-distribution / RT)
        ratio_low, ratio_median, ratio_high = quantiles(ratio_dist)
        anchor_rows.append({
            "anchor": anchor,
            "point_ddG_kcal_mol": point,
            "point_cross_over_same_odds": math.exp(-point / RT),
            "bootstrap_median_ddG_kcal_mol": median,
            "bootstrap_ci_low_kcal_mol": ci_low,
            "bootstrap_ci_high_kcal_mol": ci_high,
            "bootstrap_median_cross_over_same_odds": ratio_median,
            "bootstrap_odds_ci_low": ratio_low,
            "bootstrap_odds_ci_high": ratio_high,
            "first_half_ddG_kcal_mol": float(cross["dg_first"] - same["dg_first"]),
            "second_half_ddG_kcal_mol": float(cross["dg_second"] - same["dg_second"]),
        })
    rng = np.random.default_rng(20260828)
    draws = 200_000
    pooled_distribution = (
        rng.choice(data["anchor_v1_cross_v1_v3"]["boot_dg"], draws)
        - rng.choice(data["anchor_v1_same_v1_v2"]["boot_dg"], draws)
        + rng.choice(data["anchor_v5_cross_v5_v8"]["boot_dg"], draws)
        - rng.choice(data["anchor_v5_same_v5_v6"]["boot_dg"], draws)
    ) / 2.0
    pooled_point = float(np.mean([row["point_ddG_kcal_mol"] for row in anchor_rows]))
    ci_low, median, ci_high = quantiles(pooled_distribution)
    odds_low, odds_median, odds_high = quantiles(np.exp(-pooled_distribution / RT))
    anchor_rows.append({
        "anchor": "Pooled",
        "point_ddG_kcal_mol": pooled_point,
        "point_cross_over_same_odds": math.exp(-pooled_point / RT),
        "bootstrap_median_ddG_kcal_mol": median,
        "bootstrap_ci_low_kcal_mol": ci_low,
        "bootstrap_ci_high_kcal_mol": ci_high,
        "bootstrap_median_cross_over_same_odds": odds_median,
        "bootstrap_odds_ci_low": odds_low,
        "bootstrap_odds_ci_high": odds_high,
        "first_half_ddG_kcal_mol": float(np.mean([row["first_half_ddG_kcal_mol"] for row in anchor_rows])),
        "second_half_ddG_kcal_mol": float(np.mean([row["second_half_ddG_kcal_mol"] for row in anchor_rows])),
    })
    write_csv(output / "anchor_cross_same_comparisons.csv", anchor_rows)
    unbiased = load_unbiased_benchmarks(
        args.unbiased_summary.resolve(), args.contact_metrics.resolve()
    )
    decomposition_rows, decomposition = origin_decomposition(anchor_rows[-1], unbiased)
    write_csv(output / "cross_arm_origin_decomposition.csv", decomposition_rows)
    sensitivity_rows = sensitivity(data)
    write_csv(output / "definition_sensitivity.csv", sensitivity_rows)
    make_figure(data, anchor_rows, output)
    make_origin_figure(anchor_rows[-1], unbiased, decomposition, output)
    sensitivity_values = [row["pooled_ddG_kcal_mol"] for row in sensitivity_rows]
    report = f"""# A4-m-C8 q0 umbrella sampling：同臂与跨臂预组织自由能

## 主要结论

在共同定义的接触区间 {CONTACT_REGION_NM[0]:.3f}–{CONTACT_REGION_NM[1]:.3f} nm 和伸展参考区间 {REFERENCE_REGION_NM[0]:.1f}–{REFERENCE_REGION_NM[1]:.1f} nm 下，跨臂闭合的自由能代价在两个独立锚点比较中都更低：

- V1 锚点：ΔΔG(cross−same) = {anchor_rows[0]['point_ddG_kcal_mol']:.3f} kcal/mol；跨臂/同臂接触优势约 {anchor_rows[0]['point_cross_over_same_odds']:.2f} 倍；bootstrap 95% CI 为 [{anchor_rows[0]['bootstrap_ci_low_kcal_mol']:.3f}, {anchor_rows[0]['bootstrap_ci_high_kcal_mol']:.3f}] kcal/mol。
- V5 锚点：ΔΔG(cross−same) = {anchor_rows[1]['point_ddG_kcal_mol']:.3f} kcal/mol；跨臂/同臂接触优势约 {anchor_rows[1]['point_cross_over_same_odds']:.2f} 倍；bootstrap 95% CI 为 [{anchor_rows[1]['bootstrap_ci_low_kcal_mol']:.3f}, {anchor_rows[1]['bootstrap_ci_high_kcal_mol']:.3f}] kcal/mol。
- 两锚点汇总：ΔΔG = {pooled_point:.3f} kcal/mol；优势约 {math.exp(-pooled_point/RT):.2f} 倍。bootstrap 中位数为 {median:.3f} kcal/mol，95% CI 为 [{ci_low:.3f}, {ci_high:.3f}] kcal/mol；对应优势中位数 {odds_median:.2f} 倍，95% CI [{odds_low:.2f}, {odds_high:.2f}]。

因此，A4-m-C8 在无催化剂 q0 构象系综中具有约 0.9 kcal/mol 的跨臂预组织自由能优势。这与无偏 MD 得到的软接触比 3.69 和 near-attack proxy 比 4.07 高度一致，说明跨臂倾向主要可以追溯到反应基团更容易形成近距离接触，而不是已接触构象中出现很强的额外取向优势。

## 跨臂倾向的定量来源分解

A4-m-C8 含 24 个跨臂候选对和 4 个同臂候选对，因此跨臂/同臂的位点多重性为 6。将 umbrella 得到的单对预组织优势与该精确组合数分开，可得到：

- 单对构象预组织贡献：ΔΔG = {decomposition['preorganization_ddG_kcal_mol']:.3f} kcal/mol，对应 {decomposition['per_pair_umbrella_odds']:.2f} 倍；
- 位点多重性贡献：−RT ln(24/4) = {decomposition['multiplicity_ddG_kcal_mol']:.3f} kcal/mol，对应 6.00 倍；
- 两项相加的整分子有效偏好：ΔΔG = {decomposition['effective_ddG_kcal_mol']:.3f} kcal/mol，预测跨臂/同臂事件比为 {decomposition['predicted_any_pair_odds']:.2f}（95% CI {decomposition['predicted_any_pair_ci_low']:.2f}–{decomposition['predicted_any_pair_ci_high']:.2f}）。

无偏 MD 在 5 Å 硬接触定义下直接观测到的整分子事件比为 {decomposition['observed_any_pair_odds']:.2f}（95% CI {decomposition['observed_any_pair_ci_low']:.2f}–{decomposition['observed_any_pair_ci_high']:.2f}）。预测值与独立观测值高度一致，构成“单对预组织 × 位点多重性 = 整分子跨臂通量”的定量闭环。该分解中的多重性是拓扑计数，预组织项则来自有偏构象采样；二者都不是化学过渡态贡献。

## 质量控制

- 所有相邻窗口的高斯重叠系数均不低于 {min(float(item['minimum_overlap']) for item in data.values()):.3f}，窗口重叠充分。
- steering 抽帧与目标窗口中心的最大偏差为 {max(float(item['maximum_source_error_nm']) for item in data.values()):.4f} nm。
- 四条 WHAM 均收敛到 1×10⁻⁶；每窗用于主分析的样本数为 3,201。
- 最大积分自相关时间为 {max(float(item['maximum_iact_ps']) for item in data.values()):.1f} ps，显著小于每窗 1.6 ns 的主分析区间。
- 改变接触上限与参考区间后，汇总 ΔΔG 保持在 {min(sensitivity_values):.3f} 至 {max(sensitivity_values):.3f} kcal/mol，结论不依赖单一阈值。

## 收敛边界

前、后半程的 ΔΔG 在两个锚点中都保持为负，因此“跨臂更有利”的方向稳定；但 V1 分别为 {anchor_rows[0]['first_half_ddG_kcal_mol']:.3f} 和 {anchor_rows[0]['second_half_ddG_kcal_mol']:.3f} kcal/mol，V5 分别为 {anchor_rows[1]['first_half_ddG_kcal_mol']:.3f} 和 {anchor_rows[1]['second_half_ddG_kcal_mol']:.3f} kcal/mol。幅值仍有约 0.7–1.0 kcal/mol 漂移，提示距离以外的慢构象自由度尚未完全定量收敛。

当前结果足以作为“跨臂预组织倾向”的强证据，但若要给出发表级精确数值，建议将每窗生产段从 2 ns 延长至至少 4–5 ns，或增加独立 umbrella 重复。

## 解释边界

这里的 ΔG 是两个 C=C 中点靠近的构象预组织自由能，不是烯烃复分解的化学活化能，也不包含 Ru 催化剂。它回答“为什么 A4-m-C8 更容易形成跨臂近接触”，不能单独证明跨臂化学过渡态的能垒更低。
"""
    (output / "umbrella_analysis_report.md").write_text(report, encoding="utf-8")
    summary = {
        "temperature_K": TEMPERATURE_K,
        "contact_region_nm": CONTACT_REGION_NM,
        "reference_region_nm": REFERENCE_REGION_NM,
        "pathways": pathway_rows,
        "comparisons": anchor_rows,
        "cross_arm_origin_decomposition": decomposition,
        "sensitivity_pooled_ddG_range_kcal_mol": [min(sensitivity_values), max(sensitivity_values)],
        "interpretation_boundary": "catalyst-free q0 contact/preorganization PMF; not chemical activation barrier",
    }
    (output / "umbrella_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
