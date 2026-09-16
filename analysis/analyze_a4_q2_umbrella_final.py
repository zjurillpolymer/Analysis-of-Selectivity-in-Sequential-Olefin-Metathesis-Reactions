#!/usr/bin/env python3
"""Final q2 reuse/switch umbrella comparison for A4-m-C5 and A4-m-C8."""

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

from analysis.analyze_a4_m_c5_c8_umbrella_comparison import (
    CONTACT_LOW_NM,
    RT,
    contact_free_energy,
    load_pathway,
    read_bootstrap_profiles,
    read_xvg,
)


PATHWAYS = {
    "c5_reuse_v1_v6": {
        "system": "C5", "route": "reuse", "pair": "V1–V6",
        "path": "results/a4_q2_umbrella_bohrium/download/c5_reuse_v1_v6/pathway",
        "color": "#3775BA",
    },
    "c5_reuse_v2_v4": {
        "system": "C5", "route": "reuse", "pair": "V2–V4",
        "path": "results/a4_q2_umbrella_bohrium/download/c5_reuse_v2_v4/pathway",
        "color": "#0F4D92",
    },
    "c5_switch_v4_v6": {
        "system": "C5", "route": "switch", "pair": "V4–V6",
        "path": "results/a4_q2_umbrella_bohrium/download/c5_switch_v4_v6/pathway",
        "color": "#D2693C",
    },
    "c8_reuse_v5_v7": {
        "system": "C8", "route": "reuse", "pair": "V5–V7",
        "path": "results/a4_q2_umbrella_bohrium/resumed_download/c8_reuse_v5_v7/pathway",
        "color": "#0F4D92",
    },
    "c8_switch_v4_v5": {
        "system": "C8", "route": "switch", "pair": "V4–V5",
        "path": "results/a4_q2_umbrella_bohrium/partial/c8_switch_v4_v5/pathway",
        "color": "#D2693C",
    },
}

BOOTSTRAP_DRAWS = 300_000
SEED = 20260829


def quantiles(values: np.ndarray) -> tuple[float, float, float]:
    low, median, high = np.quantile(values, (0.025, 0.5, 0.975))
    return float(low), float(median), float(high)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def supported_bootstrap_profiles(pathway: Path, min_count: float = 5.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pmf = read_xvg(pathway / "wham/pmf.xvg")[:, :2]
    histogram = read_xvg(pathway / "wham/histograms.xvg")
    supported = np.sum(histogram[:, 1:], axis=1) >= min_count
    boots = np.stack([profile[:, 1] for profile in read_bootstrap_profiles(pathway / "wham/bootstrap_profiles.xvg")])
    return pmf, supported, boots


def analyze(project_root: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, object]] = {}
    pathway_rows: list[dict[str, object]] = []
    pmf_rows: list[dict[str, object]] = []
    for name, spec in PATHWAYS.items():
        pathway = project_root / str(spec["path"])
        item = load_pathway(pathway)
        pmf, supported, profile_matrix = supported_bootstrap_profiles(pathway)
        low_profile, median_profile, high_profile = np.quantile(profile_matrix, (0.025, 0.5, 0.975), axis=0)
        loaded[name] = {
            **spec,
            **item,
            "pmf": pmf,
            "supported": supported,
            "profile_low": low_profile,
            "profile_median": median_profile,
            "profile_high": high_profile,
        }
        low, median, high = item["boot_ci"]
        pathway_rows.append({
            "pathway": name,
            "system": spec["system"],
            "route": spec["route"],
            "pair": spec["pair"],
            "contact_dG_kcal_mol": item["dg"],
            "bootstrap_median_kcal_mol": median,
            "bootstrap_ci_low_kcal_mol": low,
            "bootstrap_ci_high_kcal_mol": high,
            "first_half_contact_dG_kcal_mol": item["dg_first"],
            "second_half_contact_dG_kcal_mol": item["dg_second"],
            "minimum_neighbor_overlap": item["minimum_overlap"],
            "maximum_source_error_nm": item["maximum_source_error_nm"],
            "support_min_nm": item["support_min_nm"],
            "support_max_nm": item["support_max_nm"],
            "window_count": item["window_count"],
        })
        for index, (distance, free_energy) in enumerate(pmf):
            pmf_rows.append({
                "pathway": name,
                "system": spec["system"],
                "route": spec["route"],
                "pair": spec["pair"],
                "distance_nm": distance,
                "pmf_kcal_mol": free_energy,
                "bootstrap_median_kcal_mol": median_profile[index],
                "bootstrap_ci_low_kcal_mol": low_profile[index],
                "bootstrap_ci_high_kcal_mol": high_profile[index],
                "histogram_supported": int(supported[index]),
            })

    rng = np.random.default_rng(SEED)
    c5_r1 = loaded["c5_reuse_v1_v6"]["boot_dg"]
    c5_r2 = loaded["c5_reuse_v2_v4"]["boot_dg"]
    c5_s = loaded["c5_switch_v4_v6"]["boot_dg"]
    c8_r = loaded["c8_reuse_v5_v7"]["boot_dg"]
    c8_s = loaded["c8_switch_v4_v5"]["boot_dg"]
    c5_reuse_draw = 0.5 * (
        c5_r1[rng.integers(0, len(c5_r1), BOOTSTRAP_DRAWS)]
        + c5_r2[rng.integers(0, len(c5_r2), BOOTSTRAP_DRAWS)]
    )
    c5_delta = c5_s[rng.integers(0, len(c5_s), BOOTSTRAP_DRAWS)] - c5_reuse_draw
    c8_delta = (
        c8_s[rng.integers(0, len(c8_s), BOOTSTRAP_DRAWS)]
        - c8_r[rng.integers(0, len(c8_r), BOOTSTRAP_DRAWS)]
    )
    reversal = c8_delta - c5_delta
    c5_point = float(loaded["c5_switch_v4_v6"]["dg"] - 0.5 * (
        loaded["c5_reuse_v1_v6"]["dg"] + loaded["c5_reuse_v2_v4"]["dg"]
    ))
    c8_point = float(loaded["c8_switch_v4_v5"]["dg"] - loaded["c8_reuse_v5_v7"]["dg"])
    contrast_specs = (
        ("C5 switch − mean(reuse)", c5_point, c5_delta, "positive = reuse favored"),
        ("C8 switch − reuse", c8_point, c8_delta, "negative = switch favored"),
        ("C8 − C5 route contrast", c8_point - c5_point, reversal, "negative = preference reversal"),
    )
    contrast_rows: list[dict[str, object]] = []
    for label, point, distribution, interpretation in contrast_specs:
        low, median, high = quantiles(distribution)
        contrast_rows.append({
            "quantity": label,
            "point_kcal_mol": point,
            "bootstrap_median_kcal_mol": median,
            "bootstrap_ci_low_kcal_mol": low,
            "bootstrap_ci_high_kcal_mol": high,
            "switch_over_reuse_contact_odds": math.exp(-point / RT),
            "interpretation": interpretation,
        })

    sensitivity_rows: list[dict[str, object]] = []
    for min_count in (1.0, 5.0, 20.0):
        point_values: dict[str, float] = {}
        for name, spec in PATHWAYS.items():
            pathway = project_root / str(spec["path"])
            pmf = read_xvg(pathway / "wham/pmf.xvg")[:, :2]
            hist = read_xvg(pathway / "wham/histograms.xvg")
            supported = np.sum(hist[:, 1:], axis=1) >= min_count
            for split in (0.45, 0.50, 0.55):
                point_values[f"{name}|{split}"] = contact_free_energy(pmf, supported, split)
        for split in (0.45, 0.50, 0.55):
            c5 = point_values[f"c5_switch_v4_v6|{split}"] - 0.5 * (
                point_values[f"c5_reuse_v1_v6|{split}"] + point_values[f"c5_reuse_v2_v4|{split}"]
            )
            c8 = point_values[f"c8_switch_v4_v5|{split}"] - point_values[f"c8_reuse_v5_v7|{split}"]
            sensitivity_rows.append({
                "minimum_histogram_count": min_count,
                "contact_split_nm": split,
                "c5_switch_minus_reuse_kcal_mol": c5,
                "c8_switch_minus_reuse_kcal_mol": c8,
                "c8_minus_c5_contrast_kcal_mol": c8 - c5,
                "expected_signs_preserved": int(c5 > 0 and c8 < 0 and c8 - c5 < 0),
            })

    write_csv(output / "q2_pathway_contact_free_energies.csv", pathway_rows)
    write_csv(output / "q2_route_contrasts.csv", contrast_rows)
    write_csv(output / "q2_contact_definition_sensitivity.csv", sensitivity_rows)
    write_csv(output / "q2_pmf_source_data.csv", pmf_rows)
    make_figure(loaded, pathway_rows, contrast_rows, output)
    write_report(pathway_rows, contrast_rows, sensitivity_rows, output)
    return {
        "pathways": pathway_rows,
        "contrasts": contrast_rows,
        "sensitivity": sensitivity_rows,
    }


def make_figure(
    loaded: dict[str, dict[str, object]],
    pathway_rows: list[dict[str, object]],
    contrast_rows: list[dict[str, object]],
    output: Path,
) -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams.update({
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })
    fig = plt.figure(figsize=(7.2, 5.6))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.05), hspace=0.42, wspace=0.38)
    ax_c5 = fig.add_subplot(grid[0, 0])
    ax_c8 = fig.add_subplot(grid[0, 1], sharey=ax_c5)
    ax_contact = fig.add_subplot(grid[1, 0])
    ax_contrast = fig.add_subplot(grid[1, 1])

    for ax, system, names in (
        (ax_c5, "C5", ("c5_reuse_v1_v6", "c5_reuse_v2_v4", "c5_switch_v4_v6")),
        (ax_c8, "C8", ("c8_reuse_v5_v7", "c8_switch_v4_v5")),
    ):
        ax.axvspan(CONTACT_LOW_NM, 0.50, color="#D9EAF5", alpha=0.65, lw=0, label="contact region")
        for name in names:
            item = loaded[name]
            mask = item["supported"]
            x = item["pmf"][:, 0]
            label = f"{item['route']} {item['pair']}"
            ax.fill_between(
                x, item["profile_low"], item["profile_high"], where=mask,
                color=item["color"], alpha=0.14, linewidth=0,
            )
            ax.plot(x[mask], item["pmf"][mask, 1], color=item["color"], lw=1.45, label=label)
        ax.set_xlim(0.32, 2.70)
        ax.set_ylim(-1.8, 10.5)
        ax.set_xlabel("vinyl midpoint distance (nm)")
        ax.set_title(f"A4-m-{system} q2", fontsize=8, pad=4)
        ax.legend(fontsize=6.1, loc="upper right", handlelength=1.6)
    ax_c5.set_ylabel("PMF (kcal mol$^{-1}$)")

    display_order = [
        "c5_reuse_v1_v6", "c5_reuse_v2_v4", "c5_switch_v4_v6",
        "c8_reuse_v5_v7", "c8_switch_v4_v5",
    ]
    row_by_name = {row["pathway"]: row for row in pathway_rows}
    y = np.arange(len(display_order))[::-1]
    labels = []
    for yi, name in zip(y, display_order):
        row, item = row_by_name[name], loaded[name]
        point = float(row["contact_dG_kcal_mol"])
        lo, hi = float(row["bootstrap_ci_low_kcal_mol"]), float(row["bootstrap_ci_high_kcal_mol"])
        ax_contact.plot([lo, hi], [yi, yi], color=item["color"], lw=1.6)
        ax_contact.plot(point, yi, "o", color=item["color"], ms=4.2, mec="white", mew=0.45)
        labels.append(f"{item['system']} {item['route']} {item['pair']}")
    ax_contact.set_yticks(y, labels)
    ax_contact.set_xlabel(r"contact free energy, $\Delta G_{contact}$ (kcal mol$^{-1}$)")
    ax_contact.set_xlim(0.3, 7.5)
    ax_contact.axhline(1.5, color="#BDBDBD", lw=0.7)
    ax_contact.text(0.98, 0.98, "lower = easier contact", transform=ax_contact.transAxes,
                    ha="right", va="top", fontsize=6.2, color="#5E5E5E")

    colors = ("#3775BA", "#D2693C", "#4D4D4D")
    y2 = np.arange(len(contrast_rows))[::-1]
    contrast_labels = ("C5", "C8", "C8 − C5")
    for yi, row, color in zip(y2, contrast_rows, colors):
        point = float(row["point_kcal_mol"])
        lo, hi = float(row["bootstrap_ci_low_kcal_mol"]), float(row["bootstrap_ci_high_kcal_mol"])
        ax_contrast.plot([lo, hi], [yi, yi], color=color, lw=2.0)
        ax_contrast.plot(point, yi, "o", color=color, ms=5.0, mec="white", mew=0.5)
        ax_contrast.text(point, yi + 0.18, f"{point:+.2f}", color=color, ha="center", va="bottom", fontsize=6.2)
    ax_contrast.axvline(0, color="#767676", ls="--", lw=0.9)
    ax_contrast.set_yticks(y2, contrast_labels)
    ax_contrast.set_xlim(-6.2, 2.3)
    ax_contrast.set_xlabel(r"route contrast, $\Delta\Delta G$ = switch − reuse (kcal mol$^{-1}$)")
    ax_contrast.text(0.02, 0.98, "switch favored", transform=ax_contrast.transAxes,
                     ha="left", va="top", fontsize=6.2, color="#D2693C")
    ax_contrast.text(0.98, 0.98, "reuse favored", transform=ax_contrast.transAxes,
                     ha="right", va="top", fontsize=6.2, color="#3775BA")

    for label, ax in zip("abcd", (ax_c5, ax_c8, ax_contact, ax_contrast)):
        ax.text(-0.15, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold",
                ha="left", va="bottom")
    fig.suptitle("Opposite q2 path-memory preference in A4-m-C5 and A4-m-C8", fontsize=9, y=0.995)
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.105, top=0.94)
    base = output / "q2_reuse_switch_umbrella_summary"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_report(
    pathway_rows: list[dict[str, object]],
    contrast_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    output: Path,
) -> None:
    path_by_name = {row["pathway"]: row for row in pathway_rows}
    c5, c8, reversal = contrast_rows
    signs_preserved = sum(int(row["expected_signs_preserved"]) for row in sensitivity_rows)
    lines = [
        "# A4 q2 reuse/switch umbrella final analysis",
        "",
        "## Main result",
        "",
        (
            "The catalyst-free q2 conformational free-energy landscape shows a route-preference reversal: "
            f"C5 favors reuse (switch − mean(reuse) = {float(c5['point_kcal_mol']):+.3f} kcal mol⁻¹; "
            f"95% bootstrap CI {float(c5['bootstrap_ci_low_kcal_mol']):+.3f} to "
            f"{float(c5['bootstrap_ci_high_kcal_mol']):+.3f}), whereas C8 favors switch "
            f"(switch − reuse = {float(c8['point_kcal_mol']):+.3f} kcal mol⁻¹; 95% CI "
            f"{float(c8['bootstrap_ci_low_kcal_mol']):+.3f} to "
            f"{float(c8['bootstrap_ci_high_kcal_mol']):+.3f})."
        ),
        "",
        (
            f"The C8-minus-C5 reversal contrast is {float(reversal['point_kcal_mol']):+.3f} kcal mol⁻¹ "
            f"(95% CI {float(reversal['bootstrap_ci_low_kcal_mol']):+.3f} to "
            f"{float(reversal['bootstrap_ci_high_kcal_mol']):+.3f})."
        ),
        "",
        (
            f"On the contact-odds scale, C5 reuse is favored over switch by "
            f"{1.0 / float(c5['switch_over_reuse_contact_odds']):.1f}-fold, whereas C8 switch is "
            f"favored over reuse by {float(c8['switch_over_reuse_contact_odds']):.1f}-fold."
        ),
        "",
        "## Pathway-level contact free energies",
        "",
        "| System | Route | Pair | ΔG_contact (kcal mol⁻¹) | 95% bootstrap CI | Min. overlap | Half-trajectory ΔG |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in PATHWAYS:
        row = path_by_name[name]
        lines.append(
            f"| {row['system']} | {row['route']} | {row['pair']} | "
            f"{float(row['contact_dG_kcal_mol']):.3f} | "
            f"[{float(row['bootstrap_ci_low_kcal_mol']):.3f}, {float(row['bootstrap_ci_high_kcal_mol']):.3f}] | "
            f"{float(row['minimum_neighbor_overlap']):.3f} | "
            f"{float(row['first_half_contact_dG_kcal_mol']):.3f} / "
            f"{float(row['second_half_contact_dG_kcal_mol']):.3f} |"
        )
    lines.extend([
        "",
        "## Definition and robustness",
        "",
        (
            f"ΔG_contact = −RT ln[P({CONTACT_LOW_NM:.3f}–0.500 nm) / "
            "P(0.500 nm–histogram-supported maximum)]. All PMFs use 100–500 ps of each "
            "500 ps production window at 313.15 K. C5 uses 25 windows (0.35–2.75 nm); "
            "C8 uses 24 windows (0.35–2.65 nm)."
        ),
        "",
        f"The expected C5-positive/C8-negative/reversal-negative signs were preserved in {signs_preserved}/{len(sensitivity_rows)} sensitivity settings (contact split 0.45, 0.50, or 0.55 nm; histogram threshold 1, 5, or 20 counts).",
        "",
        "## Interpretation limits",
        "",
        "- This is a one-dimensional, catalyst-free conformational preorganization PMF. It is not the Ru-catalyzed chemical activation barrier.",
        "- C5 reuse is represented by two pair-specific PMFs and pooled by their arithmetic mean; C5 switch and both C8 routes each use one selected pair.",
        "- Each window contributes 400 ps after discarding the first 100 ps. Half-trajectory consistency is supportive but not a substitute for longer independent replicas.",
        "- The result establishes the origin of route availability at the conformational-contact level; a subsequent QM or QM/MM calculation is required to test whether chemical barriers preserve or modify this preference.",
        "",
    ])
    (output / "q2_umbrella_final_report.md").write_text("\n".join(lines), encoding="utf-8")
    zh_lines = [
        "# A4 q2 reuse/switch umbrella 最终分析",
        "",
        "## 核心结论",
        "",
        (
            f"无催化剂的 q2 构象自由能显示出明确的路径偏好反转：C5 中 switch − mean(reuse) = "
            f"{float(c5['point_kcal_mol']):+.3f} kcal mol⁻¹（95% bootstrap CI "
            f"{float(c5['bootstrap_ci_low_kcal_mol']):+.3f} 至 {float(c5['bootstrap_ci_high_kcal_mol']):+.3f}），"
            "因此 reuse 更有利；C8 中 switch − reuse = "
            f"{float(c8['point_kcal_mol']):+.3f} kcal mol⁻¹（95% CI "
            f"{float(c8['bootstrap_ci_low_kcal_mol']):+.3f} 至 {float(c8['bootstrap_ci_high_kcal_mol']):+.3f}），"
            "因此 switch 更有利。"
        ),
        "",
        (
            f"C8−C5 的偏好反转量为 {float(reversal['point_kcal_mol']):+.3f} kcal mol⁻¹（95% CI "
            f"{float(reversal['bootstrap_ci_low_kcal_mol']):+.3f} 至 "
            f"{float(reversal['bootstrap_ci_high_kcal_mol']):+.3f}）。换算为接触优势比，C5 的 reuse "
            f"相对 switch 约有 {1.0 / float(c5['switch_over_reuse_contact_odds']):.1f} 倍优势；C8 的 switch "
            f"相对 reuse 约有 {float(c8['switch_over_reuse_contact_odds']):.1f} 倍优势。"
        ),
        "",
        "## 单路径结果",
        "",
        "| 系统 | 路径 | Pair | ΔG_contact (kcal mol⁻¹) | 95% bootstrap CI | 最小相邻窗口 overlap |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name in PATHWAYS:
        row = path_by_name[name]
        zh_lines.append(
            f"| {row['system']} | {row['route']} | {row['pair']} | "
            f"{float(row['contact_dG_kcal_mol']):.3f} | "
            f"[{float(row['bootstrap_ci_low_kcal_mol']):.3f}, {float(row['bootstrap_ci_high_kcal_mol']):.3f}] | "
            f"{float(row['minimum_neighbor_overlap']):.3f} |"
        )
    zh_lines.extend([
        "",
        "## 稳健性和边界",
        "",
        f"接触定义为 0.325–0.500 nm。改变接触分界（0.45、0.50、0.55 nm）和直方图支持阈值（1、5、20 counts）后，{signs_preserved}/{len(sensitivity_rows)} 组设置均保持 C5 reuse 有利、C8 switch 有利以及 C8−C5 为负的符号。",
        "",
        "- 这里的 reuse/switch 描述 q2 后反应臂对的构象记忆，不应直接改写为 same-arm/cross-arm。",
        "- 这是一维、无催化剂的接触预组织自由能，不是 Ru 催化的化学反应能垒。",
        "- 每窗使用 100–500 ps，共400 ps；当前结果适合作为构象来源证据，但正式发表前最好增加独立重复或延长窗口采样。",
        "- 下一步可从接触窗口提取代表构象，进行含 Ru 催化剂的 QM 或 QM/MM 反应路径和能垒计算。",
        "",
    ])
    (output / "q2_umbrella_final_report_zh.md").write_text("\n".join(zh_lines), encoding="utf-8")
    legend = (
        "**Figure | Opposite q2 conformational path-memory preference in A4-m-C5 and A4-m-C8.** "
        "**a,b,** Pair-specific midpoint-distance PMFs for C5 and C8; shading denotes the 95% WHAM "
        "bootstrap envelope and the blue band marks the contact interval (0.325–0.500 nm). "
        "**c,** Contact free energies obtained by Boltzmann integration of the contact and "
        "histogram-supported non-contact regions; points are WHAM estimates and bars are 95% "
        "bootstrap intervals. **d,** Route contrasts defined as ΔΔG = ΔG_contact(switch) − "
        "ΔG_contact(reuse); positive values favor reuse and negative values favor switch. C5 reuse "
        "is the mean of two pair-specific pathways. Simulations used 24–25 umbrella windows, "
        "100–500 ps production data per window, 313.15 K, and 200 trajectory-bootstrap WHAM profiles."
    )
    (output / "q2_umbrella_figure_legend.md").write_text(legend + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "results/a4_q2_umbrella_analysis/final")
    args = parser.parse_args()
    result = analyze(root, args.output.resolve())
    print(json.dumps(result["contrasts"], ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
