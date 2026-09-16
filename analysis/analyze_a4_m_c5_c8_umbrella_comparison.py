#!/usr/bin/env python3
"""Compare A4-m-C5 and A4-m-C8 q0 umbrella PMFs on one thermodynamic scale.

The primary observable is the contact odds relative to every histogram-supported
non-contact distance, not to a fixed long-distance interval.  This definition is
needed because the C5 same-arm coordinate cannot physically reach the full C8
distance range.
"""

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
CONTACT_LOW_NM = 0.325
CONTACT_SPLIT_NM = 0.500
MIN_HISTOGRAM_COUNT = 5.0
BOOTSTRAP_DRAWS = 300_000
SEED = 20260828

SYSTEMS = {
    "C5": {
        "root": "results/a4_m_c5_umbrella_bohrium/manual_download",
        "anchors": {
            "V1": ("anchor_v1_same_v1_v8", "anchor_v1_cross_v1_v5"),
            "V4": ("anchor_v4_same_v4_v5", "anchor_v4_cross_v4_v8"),
        },
    },
    "C8": {
        "root": "results/a4_m_c8_umbrella_bohrium/download",
        "anchors": {
            "V1": ("anchor_v1_same_v1_v2", "anchor_v1_cross_v1_v3"),
            "V5": ("anchor_v5_same_v5_v6", "anchor_v5_cross_v5_v8"),
        },
    },
}


def read_xvg(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#@&":
            continue
        rows.append([float(value) for value in stripped.split()])
    result = np.asarray(rows, dtype=float)
    if result.ndim != 2 or result.shape[0] < 2:
        raise ValueError(f"insufficient XVG data: {path}")
    return result


def read_bootstrap_profiles(path: Path) -> list[np.ndarray]:
    profiles: list[np.ndarray] = []
    current: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#@":
            continue
        if stripped[0] == "&":
            if current:
                profiles.append(np.asarray(current, dtype=float))
                current = []
            continue
        current.append([float(value) for value in stripped.split()[:2]])
    if current:
        profiles.append(np.asarray(current, dtype=float))
    if len(profiles) != 200:
        raise ValueError(f"expected 200 WHAM bootstrap profiles in {path}; got {len(profiles)}")
    return profiles


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def q025_50_975(values: np.ndarray) -> tuple[float, float, float]:
    result = np.quantile(values, (0.025, 0.5, 0.975))
    return tuple(float(value) for value in result)


def region_log_integral(
    profile: np.ndarray,
    supported: np.ndarray,
    low: float,
    high: float,
) -> float:
    """Numerically integrate exp(-G/RT) without crossing unsupported PMF bins."""
    x = profile[:, 0]
    g = profile[:, 1]
    valid = supported & np.isfinite(x) & np.isfinite(g) & (x >= low) & (x <= high)
    grid = x[valid]
    values = g[valid]
    if grid.size < 2:
        raise ValueError(f"fewer than two supported PMF bins in {low:.3f}..{high:.3f} nm")

    # Add an exact boundary only when supported points bracket it; never extrapolate
    # into a histogram-empty tail.
    for boundary in (low, high):
        left = np.flatnonzero(supported & np.isfinite(g) & (x < boundary))
        right = np.flatnonzero(supported & np.isfinite(g) & (x > boundary))
        if left.size and right.size:
            i, j = left[-1], right[0]
            spacing = float(np.median(np.diff(x)))
            if x[j] - x[i] <= 2.1 * spacing:
                boundary_g = float(np.interp(boundary, x[[i, j]], g[[i, j]]))
                grid = np.r_[grid, boundary]
                values = np.r_[values, boundary_g]

    order = np.argsort(grid)
    grid, values = grid[order], values[order]
    grid, unique = np.unique(grid, return_index=True)
    values = values[unique]
    shift = float(np.min(values))
    integral = float(np.trapezoid(np.exp(-(values - shift) / RT), grid))
    if not np.isfinite(integral) or integral <= 0:
        raise ValueError("non-positive Boltzmann integral")
    return math.log(integral) - shift / RT


def contact_free_energy(
    profile: np.ndarray,
    supported: np.ndarray,
    split_nm: float = CONTACT_SPLIT_NM,
) -> float:
    support_x = profile[supported, 0]
    if support_x.size < 4:
        raise ValueError("insufficient histogram-supported PMF bins")
    log_contact = region_log_integral(profile, supported, CONTACT_LOW_NM, split_nm)
    log_noncontact = region_log_integral(profile, supported, split_nm, float(support_x.max()))
    return -RT * (log_contact - log_noncontact)


def normalized_profile(profile: np.ndarray, supported: np.ndarray, split_nm: float) -> np.ndarray:
    """Shift G so the supported non-contact Boltzmann integral equals one."""
    support_x = profile[supported, 0]
    log_noncontact = region_log_integral(profile, supported, split_nm, float(support_x.max()))
    aligned = profile.copy()
    aligned[:, 1] = aligned[:, 1] + RT * log_noncontact
    aligned[~supported, 1] = np.nan
    return aligned


def load_pathway(pathway: Path, min_count: float = MIN_HISTOGRAM_COUNT) -> dict[str, object]:
    wham = pathway / "wham"
    pmf = read_xvg(wham / "pmf.xvg")[:, :2]
    first = read_xvg(wham / "pmf_first_half.xvg")[:, :2]
    second = read_xvg(wham / "pmf_second_half.xvg")[:, :2]
    histogram = read_xvg(wham / "histograms.xvg")
    if not np.allclose(pmf[:, 0], histogram[:, 0]):
        raise ValueError(f"PMF/histogram grids differ: {pathway}")
    total_count = np.sum(histogram[:, 1:], axis=1)
    supported = total_count >= min_count
    boots = read_bootstrap_profiles(wham / "bootstrap_profiles.xvg")
    for profile in boots:
        if not np.allclose(profile[:, 0], pmf[:, 0]):
            raise ValueError(f"bootstrap/PMF grids differ: {pathway}")

    with (pathway / "window_diagnostics.csv").open(encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    with (pathway / "window_sources.csv").open(encoding="utf-8", newline="") as handle:
        sources = list(csv.DictReader(handle))
    overlaps = [
        float(row["left_neighbor_gaussian_overlap"])
        for row in diagnostics
        if row.get("left_neighbor_gaussian_overlap", "").strip()
    ]
    source_errors = [float(row["absolute_error_nm"]) for row in sources]
    boot_dg = np.asarray([contact_free_energy(profile, supported) for profile in boots])
    return {
        "path": pathway,
        "pmf": pmf,
        "first": first,
        "second": second,
        "histogram_count": total_count,
        "supported": supported,
        "support_min_nm": float(pmf[supported, 0].min()),
        "support_max_nm": float(pmf[supported, 0].max()),
        "dg": contact_free_energy(pmf, supported),
        "dg_first": contact_free_energy(first, supported),
        "dg_second": contact_free_energy(second, supported),
        "boot_dg": boot_dg,
        "boot_ci": q025_50_975(boot_dg),
        "minimum_overlap": min(overlaps),
        "maximum_source_error_nm": max(source_errors),
        "window_count": len(diagnostics),
    }


def exact_difference(cross: np.ndarray, same: np.ndarray) -> np.ndarray:
    return (cross[:, None] - same[None, :]).ravel()


def build_analysis(project_root: Path) -> tuple[dict[str, object], list[dict[str, object]],
                                                 list[dict[str, object]], list[dict[str, object]]]:
    loaded: dict[str, object] = {}
    pathway_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []

    for system, spec in SYSTEMS.items():
        system_root = project_root / str(spec["root"])
        system_data: dict[str, object] = {}
        for anchor, names in spec["anchors"].items():
            same_name, cross_name = names
            pair: dict[str, object] = {}
            for topology, name in (("same", same_name), ("cross", cross_name)):
                item = load_pathway(system_root / name / "pathway")
                pair[topology] = item
                low, median, high = item["boot_ci"]
                pathway_rows.append({
                    "system": system,
                    "anchor": anchor,
                    "topology": topology,
                    "pathway": name,
                    "contact_dG_kcal_mol": item["dg"],
                    "bootstrap_median_kcal_mol": median,
                    "bootstrap_ci_low_kcal_mol": low,
                    "bootstrap_ci_high_kcal_mol": high,
                    "first_half_dG_kcal_mol": item["dg_first"],
                    "second_half_dG_kcal_mol": item["dg_second"],
                    "support_min_nm": item["support_min_nm"],
                    "support_max_nm": item["support_max_nm"],
                    "minimum_neighbor_overlap": item["minimum_overlap"],
                    "maximum_source_error_nm": item["maximum_source_error_nm"],
                    "window_count": item["window_count"],
                })
            same, cross = pair["same"], pair["cross"]
            distribution = exact_difference(cross["boot_dg"], same["boot_dg"])
            low, median, high = q025_50_975(distribution)
            ddg = float(cross["dg"] - same["dg"])
            first_ddg = float(cross["dg_first"] - same["dg_first"])
            second_ddg = float(cross["dg_second"] - same["dg_second"])
            pair["ddg"] = ddg
            pair["ddg_boot"] = distribution
            pair["ddg_ci"] = (low, median, high)
            pair["first_ddg"] = first_ddg
            pair["second_ddg"] = second_ddg
            system_data[anchor] = pair
            anchor_rows.append({
                "system": system,
                "anchor": anchor,
                "same_pathway": same_name,
                "cross_pathway": cross_name,
                "point_ddG_cross_minus_same_kcal_mol": ddg,
                "bootstrap_median_kcal_mol": median,
                "bootstrap_ci_low_kcal_mol": low,
                "bootstrap_ci_high_kcal_mol": high,
                "point_cross_over_same_odds": math.exp(-ddg / RT),
                "bootstrap_odds_ci_low": math.exp(-high / RT),
                "bootstrap_odds_ci_high": math.exp(-low / RT),
                "first_half_ddG_kcal_mol": first_ddg,
                "second_half_ddG_kcal_mol": second_ddg,
            })
        loaded[system] = system_data

    rng = np.random.default_rng(SEED)
    system_rows: list[dict[str, object]] = []
    pooled_distributions: dict[str, np.ndarray] = {}
    for system, system_data in loaded.items():
        anchors = list(system_data)
        draws = []
        for anchor in anchors:
            values = system_data[anchor]["ddg_boot"]
            draws.append(values[rng.integers(0, values.size, BOOTSTRAP_DRAWS)])
        pooled = np.mean(np.vstack(draws), axis=0)
        pooled_distributions[system] = pooled
        point = float(np.mean([system_data[anchor]["ddg"] for anchor in anchors]))
        low, median, high = q025_50_975(pooled)
        system_rows.append({
            "quantity": system,
            "point_ddG_kcal_mol": point,
            "bootstrap_median_kcal_mol": median,
            "bootstrap_ci_low_kcal_mol": low,
            "bootstrap_ci_high_kcal_mol": high,
            "point_cross_over_same_odds": math.exp(-point / RT),
            "bootstrap_odds_ci_low": math.exp(-high / RT),
            "bootstrap_odds_ci_high": math.exp(-low / RT),
            "interpretation": "negative_ddG_means_cross_contact_favored",
        })

    contrast = pooled_distributions["C8"] - pooled_distributions["C5"]
    c5_point = next(float(row["point_ddG_kcal_mol"]) for row in system_rows if row["quantity"] == "C5")
    c8_point = next(float(row["point_ddG_kcal_mol"]) for row in system_rows if row["quantity"] == "C8")
    low, median, high = q025_50_975(contrast)
    system_rows.append({
        "quantity": "C8_minus_C5",
        "point_ddG_kcal_mol": c8_point - c5_point,
        "bootstrap_median_kcal_mol": median,
        "bootstrap_ci_low_kcal_mol": low,
        "bootstrap_ci_high_kcal_mol": high,
        "point_cross_over_same_odds": math.exp(-(c8_point - c5_point) / RT),
        "bootstrap_odds_ci_low": math.exp(-high / RT),
        "bootstrap_odds_ci_high": math.exp(-low / RT),
        "interpretation": "negative_contrast_means_C8_more_cross_favored_than_C5",
    })

    loaded["pooled_distributions"] = pooled_distributions
    loaded["contrast_distribution"] = contrast
    return loaded, pathway_rows, anchor_rows, system_rows


def build_sensitivity(project_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for min_count in (1.0, 5.0, 20.0, 50.0):
        for split in (0.45, 0.50, 0.55):
            points: dict[str, float] = {}
            for system, spec in SYSTEMS.items():
                root = project_root / str(spec["root"])
                anchor_values = []
                for _anchor, (same_name, cross_name) in spec["anchors"].items():
                    same = load_pathway(root / same_name / "pathway", min_count)
                    cross = load_pathway(root / cross_name / "pathway", min_count)
                    anchor_values.append(
                        contact_free_energy(cross["pmf"], cross["supported"], split)
                        - contact_free_energy(same["pmf"], same["supported"], split)
                    )
                points[system] = float(np.mean(anchor_values))
            rows.append({
                "minimum_histogram_count": int(min_count),
                "contact_split_nm": split,
                "C5_pooled_ddG_kcal_mol": points["C5"],
                "C8_pooled_ddG_kcal_mol": points["C8"],
                "C8_minus_C5_kcal_mol": points["C8"] - points["C5"],
            })
    return rows


def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    for suffix, kwargs in (
        (".png", {"dpi": 300}),
        (".pdf", {}),
        (".svg", {}),
        (".tiff", {"dpi": 600, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", facecolor="white", **kwargs)


def make_comparison_figure(
    loaded: dict[str, object], anchor_rows: list[dict[str, object]],
    system_rows: list[dict[str, object]], output: Path,
) -> None:
    configure_plotting()
    colors = {"C5": "#D55E00", "C8": "#0072B2"}
    fig, axes = plt.subplots(1, 3, figsize=(7.20, 2.75), gridspec_kw={"width_ratios": [1.25, 0.9, 1.25]})

    # a, anchor-specific estimates
    ax = axes[0]
    ordered = [
        next(row for row in anchor_rows if row["system"] == "C5" and row["anchor"] == "V1"),
        next(row for row in anchor_rows if row["system"] == "C5" and row["anchor"] == "V4"),
        next(row for row in anchor_rows if row["system"] == "C8" and row["anchor"] == "V1"),
        next(row for row in anchor_rows if row["system"] == "C8" and row["anchor"] == "V5"),
    ]
    y = np.arange(len(ordered))[::-1]
    for ypos, row in zip(y, ordered):
        x = float(row["point_ddG_cross_minus_same_kcal_mol"])
        lo = float(row["bootstrap_ci_low_kcal_mol"])
        hi = float(row["bootstrap_ci_high_kcal_mol"])
        ax.errorbar(x, ypos, xerr=[[x - lo], [hi - x]], fmt="o", color=colors[str(row["system"])],
                    capsize=2.2, lw=1.1, ms=4.2, zorder=3)
    ax.axvline(0, color="#666666", lw=0.8, ls="--")
    ax.set_yticks(y, [f"{row['system']} {row['anchor']}" for row in ordered])
    ax.set_xlabel(r"$\Delta\Delta G_{cross-same}$ (kcal mol$^{-1}$)")
    ax.set_title("Anchor-specific contact preference", loc="left")
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)

    # b, fixed-anchor system means and direct contrast
    ax = axes[1]
    ordered_system = [next(row for row in system_rows if row["quantity"] == key)
                      for key in ("C5", "C8", "C8_minus_C5")]
    labels = ["C5", "C8", "C8 − C5"]
    y = np.arange(3)[::-1]
    for ypos, row, label in zip(y, ordered_system, labels):
        x = float(row["point_ddG_kcal_mol"])
        lo = float(row["bootstrap_ci_low_kcal_mol"])
        hi = float(row["bootstrap_ci_high_kcal_mol"])
        color = colors.get(label, "#333333")
        ax.errorbar(x, ypos, xerr=[[x - lo], [hi - x]], fmt="o", color=color,
                    capsize=2.2, lw=1.1, ms=4.2, zorder=3)
    ax.axvline(0, color="#666666", lw=0.8, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel(r"$\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax.set_title("Fixed-anchor means", loc="left")
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)

    # c, half-trajectory stability; all signs remain negative but magnitudes drift.
    ax = axes[2]
    x = np.arange(3)
    for row in ordered:
        system = str(row["system"])
        anchor = str(row["anchor"])
        values = [float(row["first_half_ddG_kcal_mol"]),
                  float(row["point_ddG_cross_minus_same_kcal_mol"]),
                  float(row["second_half_ddG_kcal_mol"])]
        marker = "o" if anchor == "V1" else "s"
        ax.plot(x, values, marker=marker, ms=3.7, lw=1.0, color=colors[system],
                alpha=0.90, label=f"{system} {anchor}")
    ax.axhline(0, color="#666666", lw=0.8, ls="--")
    ax.set_xticks(x, ["First half", "Full", "Second half"])
    ax.set_ylabel(r"$\Delta\Delta G_{cross-same}$ (kcal mol$^{-1}$)")
    ax.set_title("Time-split stability", loc="left")
    ax.grid(axis="y", color="#DDDDDD", lw=0.5)
    ax.legend(ncol=2, loc="lower left", handlelength=1.5, columnspacing=0.8)

    for label, ax in zip("abc", axes):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    fig.suptitle("Distance-only preorganization does not distinguish C5 from C8", y=1.02,
                 fontsize=9.2, fontweight="bold")
    fig.tight_layout(w_pad=1.35)
    save_figure(fig, output / "a4_m_c5_c8_umbrella_comparison")
    plt.close(fig)


def make_c5_pmf_figure(loaded: dict[str, object], output: Path) -> None:
    configure_plotting()
    colors = {"same": "#555555", "cross": "#009E73"}
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 2.65), sharey=True)
    for panel, (ax, anchor) in enumerate(zip(axes, ("V1", "V4"))):
        for topology in ("same", "cross"):
            item = loaded["C5"][anchor][topology]
            aligned = normalized_profile(item["pmf"], item["supported"], CONTACT_SPLIT_NM)
            ax.plot(aligned[:, 0], aligned[:, 1], lw=1.3, color=colors[topology], label=topology.capitalize())
        ax.axvspan(CONTACT_LOW_NM, CONTACT_SPLIT_NM, color="#F0E442", alpha=0.20, lw=0)
        ax.axvline(CONTACT_SPLIT_NM, color="#777777", lw=0.7, ls="--")
        ax.set_xlabel("C=C pair distance (nm)")
        ax.set_title(f"C5 anchor {anchor}", loc="left")
        ax.grid(color="#E2E2E2", lw=0.45)
        ax.text(-0.12, 1.06, chr(ord("a") + panel), transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
    axes[0].set_ylabel("Free energy aligned to non-contact partition\n(kcal mol$^{-1}$)")
    axes[1].legend(loc="upper right")
    fig.suptitle("A4-m-C5 supported PMFs", y=1.02, fontsize=9.2, fontweight="bold")
    fig.tight_layout(w_pad=1.1)
    save_figure(fig, output / "a4_m_c5_supported_pmf_profiles")
    plt.close(fig)


def make_summary(loaded: dict[str, object], system_rows: list[dict[str, object]],
                 sensitivity_rows: list[dict[str, object]]) -> dict[str, object]:
    by_quantity = {str(row["quantity"]): row for row in system_rows}
    contrast = loaded["contrast_distribution"]
    primary_sensitivity = [row for row in sensitivity_rows if row["contact_split_nm"] == 0.5]
    return {
        "temperature_K": TEMPERATURE_K,
        "RT_kcal_mol": RT,
        "contact_region_nm": [CONTACT_LOW_NM, CONTACT_SPLIT_NM],
        "noncontact_definition": "all histogram-supported bins above 0.500 nm",
        "minimum_histogram_count_primary": MIN_HISTOGRAM_COUNT,
        "bootstrap_profiles_per_pathway": 200,
        "system_estimates": by_quantity,
        "probability_C8_more_cross_favored_than_C5": float(np.mean(contrast < 0)),
        "support_threshold_sensitivity_at_0.5_nm": primary_sensitivity,
        "main_conclusion": (
            "Both C5 and C8 favor cross-arm contact for the selected anchors, but C8 is not more "
            "cross-favored than C5. The opposite experimental product topology therefore cannot be "
            "assigned to this one-dimensional distance preorganization coordinate alone."
        ),
        "scope_limit": (
            "Intervals propagate WHAM within-path bootstrap uncertainty for these fixed selected "
            "anchors; they do not quantify between-anchor/pair generalization uncertainty."
        ),
    }


def write_report(output: Path, summary: dict[str, object], anchor_rows: list[dict[str, object]],
                 pathway_rows: list[dict[str, object]], sensitivity_rows: list[dict[str, object]]) -> None:
    systems = summary["system_estimates"]
    c5, c8, contrast = systems["C5"], systems["C8"], systems["C8_minus_C5"]
    anchor_lines = []
    for row in anchor_rows:
        anchor_lines.append(
            f"| {row['system']} | {row['anchor']} | {float(row['point_ddG_cross_minus_same_kcal_mol']):+.3f} | "
            f"[{float(row['bootstrap_ci_low_kcal_mol']):+.3f}, {float(row['bootstrap_ci_high_kcal_mol']):+.3f}] | "
            f"{float(row['point_cross_over_same_odds']):.2f} |"
        )
    min_overlap = min(float(row["minimum_neighbor_overlap"]) for row in pathway_rows)
    sensitivity_05 = [row for row in sensitivity_rows if row["contact_split_nm"] == 0.5]
    c5_range = (min(float(row["C5_pooled_ddG_kcal_mol"]) for row in sensitivity_05),
                max(float(row["C5_pooled_ddG_kcal_mol"]) for row in sensitivity_05))
    c8_range = (min(float(row["C8_pooled_ddG_kcal_mol"]) for row in sensitivity_05),
                max(float(row["C8_pooled_ddG_kcal_mol"]) for row in sensitivity_05))
    text = f"""# A4-m-C5 与 A4-m-C8 umbrella sampling 对比

## 核心结论

在统一定义下（0.325–0.500 nm 为接触态；每条 PMF 中所有有直方图支持且大于 0.500 nm 的距离为非接触态），**C5 和 C8 的所选锚点都表现为跨臂接触更有利**。固定两个锚点取平均后：

- C5：ΔΔG(cross−same) = {float(c5['point_ddG_kcal_mol']):+.3f} kcal/mol，bootstrap 95% 区间 [{float(c5['bootstrap_ci_low_kcal_mol']):+.3f}, {float(c5['bootstrap_ci_high_kcal_mol']):+.3f}]，对应跨臂/同臂接触 odds = {float(c5['point_cross_over_same_odds']):.2f}。
- C8：ΔΔG(cross−same) = {float(c8['point_ddG_kcal_mol']):+.3f} kcal/mol，bootstrap 95% 区间 [{float(c8['bootstrap_ci_low_kcal_mol']):+.3f}, {float(c8['bootstrap_ci_high_kcal_mol']):+.3f}]，对应 odds = {float(c8['point_cross_over_same_odds']):.2f}。
- 系统差异 C8−C5 = {float(contrast['point_ddG_kcal_mol']):+.3f} kcal/mol，bootstrap 95% 区间 [{float(contrast['bootstrap_ci_low_kcal_mol']):+.3f}, {float(contrast['bootstrap_ci_high_kcal_mol']):+.3f}]；P(C8 比 C5 更偏跨臂) = {float(summary['probability_C8_more_cross_favored_than_C5']):.3f}。

因此，目前数据**不能证明 C8 在简单距离接触预组织上比 C5 更偏跨臂**；点估计甚至略朝相反方向，但系统差异区间跨过 0。更准确的结论是：实验中 C5 偏同臂、C8 偏跨臂的来源不在这一维 C=C 距离接触自由能本身，至少还需要考察反应取向、near-attack geometry、催化剂可接近性以及成键/金属杂环过渡态中的应变。

## 锚点分解

负值表示跨臂接触更有利。

| 系统 | 锚点 | ΔΔG (kcal/mol) | bootstrap 95% 区间 | cross/same odds |
|---|---:|---:|---:|---:|
{chr(10).join(anchor_lines)}

C5 的 V1 与 V4 差异明显：V1 的区间跨 0，而 V4 强烈偏跨臂。这是当前最重要的采样警告——两个手工选定锚点不能代表完整构象系综。

## 质量控制

- 四条 C5 和四条 C8 PMF 都使用 200 条 WHAM bootstrap profile。
- 主分析只保留总直方图计数 ≥ 5 的 PMF bin，避免把 C5 同臂 PMF 高距离端未采样的零值误当成真实自由能。
- 八条路径的最小相邻窗 Gaussian overlap 为 {min_overlap:.3f}，窗口连接性良好。
- 将支持阈值从 1 改到 50、接触边界从 0.45 改到 0.55 nm 的结果见 `support_sensitivity.csv`。在 0.500 nm 边界下，C5 固定锚点平均仅在 [{c5_range[0]:+.3f}, {c5_range[1]:+.3f}] kcal/mol 之间变化，C8 在 [{c8_range[0]:+.3f}, {c8_range[1]:+.3f}] kcal/mol 之间变化。
- 前后半程分析中四个锚点的符号均为负，但幅度有漂移，说明“跨臂方向”较稳健，而精确数值尚未完全收敛。

## 统计边界

这里的 pooled bootstrap 区间只传播了**固定路径内部**的 WHAM 重采样不确定性；它没有覆盖不同锚点、不同烯烃配对选择带来的系综不确定性。尤其 C5 两个锚点相差约 1 kcal/mol，不能把 pooled 区间解释成整个分子的严格置信区间。

## 下一步建议

1. 不必继续仅沿 C=C 距离增加同类 umbrella 窗口；它更可能把当前否定性结论做得更精确，而不会解释选择性反转。
2. 从接触区帧中定义生产性 near-attack geometry，例如距离 + 两个双键相对取向/二面角，比较 P(NAC|contact) 的同臂与跨臂差异。
3. 对 C5 与 C8 各扩展多个锚点和多个同臂/跨臂配对，采用分层 bootstrap，显式纳入锚点间方差。
4. 最后从生产性 NAC 中选代表帧加入催化剂，做同一理论水平下的反应路径/过渡态自由能；这一步才可直接接近“反应能垒来源”。

## 图件

- `a4_m_c5_c8_umbrella_comparison.*`：锚点结果、固定锚点均值、前后半程稳定性。
- `a4_m_c5_supported_pmf_profiles.*`：去除无支持 bin 后的 C5 PMF；曲线按非接触态配分函数对齐。
"""
    (output / "report.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output = (args.output or project_root / "results/a4_m_c5_c8_umbrella_comparison").resolve()
    output.mkdir(parents=True, exist_ok=True)
    loaded, pathway_rows, anchor_rows, system_rows = build_analysis(project_root)
    sensitivity_rows = build_sensitivity(project_root)
    write_csv(output / "pathway_contact_free_energies.csv", pathway_rows)
    write_csv(output / "anchor_comparisons.csv", anchor_rows)
    write_csv(output / "system_comparisons.csv", system_rows)
    write_csv(output / "support_sensitivity.csv", sensitivity_rows)
    summary = make_summary(loaded, system_rows, sensitivity_rows)
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(output, summary, anchor_rows, pathway_rows, sensitivity_rows)
    make_comparison_figure(loaded, anchor_rows, system_rows, output)
    make_c5_pmf_figure(loaded, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
