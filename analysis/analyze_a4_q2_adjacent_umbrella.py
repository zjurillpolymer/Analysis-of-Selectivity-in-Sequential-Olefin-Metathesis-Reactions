#!/usr/bin/env python3
"""Analyze anchor-matched same/cross PMFs on an adjacent-q2 background."""

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
    CONTACT_SPLIT_NM,
    RT,
    contact_free_energy,
    normalized_profile,
    q025_50_975,
    read_bootstrap_profiles,
    read_xvg,
)


PATHWAYS = {
    "c5_adj_same_v2_v3": {"system": "C5", "topology": "same-arm", "pair": "V2–V3"},
    "c5_adj_cross_v2_v1": {"system": "C5", "topology": "cross-arm", "pair": "V2–V1"},
    "c8_adj_same_v1_v2": {"system": "C8", "topology": "same-arm", "pair": "V1–V2"},
    "c8_adj_cross_v1_v4": {"system": "C8", "topology": "cross-arm", "pair": "V1–V4"},
}
PAIRS = {
    "C5": ("c5_adj_same_v2_v3", "c5_adj_cross_v2_v1"),
    "C8": ("c8_adj_same_v1_v2", "c8_adj_cross_v1_v4"),
}
BOOTSTRAP_DRAWS = 300_000
SEED = 20260829


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pathway(
    raw_pathway: Path,
    wham: Path,
    minimum_count: float = 5.0,
    split_nm: float = CONTACT_SPLIT_NM,
) -> dict[str, object]:
    pmf = read_xvg(wham / "pmf.xvg")[:, :2]
    first = read_xvg(wham / "pmf_first_half.xvg")[:, :2]
    second = read_xvg(wham / "pmf_second_half.xvg")[:, :2]
    histogram = read_xvg(wham / "histograms.xvg")
    if not np.allclose(pmf[:, 0], histogram[:, 0]):
        raise ValueError(f"PMF/histogram grids differ: {wham}")
    supported = np.sum(histogram[:, 1:], axis=1) >= minimum_count
    profiles = read_bootstrap_profiles(wham / "bootstrap_profiles.xvg")
    for profile in profiles:
        if not np.allclose(profile[:, 0], pmf[:, 0]):
            raise ValueError(f"bootstrap/PMF grids differ: {wham}")
    bootstrap_dg = np.asarray([
        contact_free_energy(profile, supported, split_nm) for profile in profiles
    ])
    aligned = normalized_profile(pmf, supported, split_nm)
    aligned_bootstrap = np.stack([
        normalized_profile(profile, supported, split_nm)[:, 1] for profile in profiles
    ])
    profile_low = np.full(pmf.shape[0], np.nan)
    profile_median = np.full(pmf.shape[0], np.nan)
    profile_high = np.full(pmf.shape[0], np.nan)
    profile_low[supported], profile_median[supported], profile_high[supported] = np.quantile(
        aligned_bootstrap[:, supported], (0.025, 0.5, 0.975), axis=0,
    )
    with (raw_pathway / "window_diagnostics.csv").open(encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    with (raw_pathway / "window_sources.csv").open(encoding="utf-8", newline="") as handle:
        sources = list(csv.DictReader(handle))
    overlaps = [
        float(row["left_neighbor_gaussian_overlap"])
        for row in diagnostics if row.get("left_neighbor_gaussian_overlap", "").strip()
    ]
    recompute = json.loads((wham / "recompute.json").read_text(encoding="utf-8"))
    support_x = pmf[supported, 0]
    return {
        "pmf": pmf,
        "aligned_pmf": aligned,
        "supported": supported,
        "profile_low": profile_low,
        "profile_median": profile_median,
        "profile_high": profile_high,
        "dg": contact_free_energy(pmf, supported, split_nm),
        "dg_first": contact_free_energy(first, supported, split_nm),
        "dg_second": contact_free_energy(second, supported, split_nm),
        "bootstrap_dg": bootstrap_dg,
        "bootstrap_ci": q025_50_975(bootstrap_dg),
        "support_min_nm": float(support_x.min()),
        "support_max_nm": float(support_x.max()),
        "minimum_overlap": min(overlaps),
        "maximum_source_error_nm": max(float(row["absolute_error_nm"]) for row in sources),
        "maximum_mean_center_error_nm": max(abs(float(row["mean_minus_center_nm"])) for row in diagnostics),
        "window_count": len(diagnostics),
        "recompute": recompute,
    }


def exact_difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left[:, None] - right[None, :]).ravel()


def analyze(project_root: Path, output: Path) -> dict[str, object]:
    raw_root = project_root / "results/a4_q2_adjacent_umbrella_bohrium/download"
    wham_root = project_root / "results/a4_q2_adjacent_umbrella_analysis/wham_recomputed"
    output.mkdir(parents=True, exist_ok=True)
    loaded: dict[str, dict[str, object]] = {}
    pathway_rows: list[dict[str, object]] = []
    source_rows: list[dict[str, object]] = []
    for name, spec in PATHWAYS.items():
        raw_pathway, wham = raw_root / name / "pathway", wham_root / name
        item = load_pathway(raw_pathway, wham)
        loaded[name] = {**spec, **item}
        low, median, high = item["bootstrap_ci"]
        pathway_rows.append({
            "pathway": name,
            "system": spec["system"],
            "topology": spec["topology"],
            "pair": spec["pair"],
            "contact_dG_kcal_mol": item["dg"],
            "bootstrap_median_kcal_mol": median,
            "bootstrap_ci_low_kcal_mol": low,
            "bootstrap_ci_high_kcal_mol": high,
            "first_half_contact_dG_kcal_mol": item["dg_first"],
            "second_half_contact_dG_kcal_mol": item["dg_second"],
            "support_min_nm": item["support_min_nm"],
            "support_max_nm": item["support_max_nm"],
            "minimum_neighbor_overlap": item["minimum_overlap"],
            "maximum_source_error_nm": item["maximum_source_error_nm"],
            "maximum_mean_center_error_nm": item["maximum_mean_center_error_nm"],
            "window_count": item["window_count"],
        })
        for index, (distance, free_energy) in enumerate(item["aligned_pmf"]):
            source_rows.append({
                "pathway": name,
                "system": spec["system"],
                "topology": spec["topology"],
                "pair": spec["pair"],
                "distance_nm": distance,
                "aligned_pmf_kcal_mol": free_energy,
                "bootstrap_median_kcal_mol": item["profile_median"][index],
                "bootstrap_ci_low_kcal_mol": item["profile_low"][index],
                "bootstrap_ci_high_kcal_mol": item["profile_high"][index],
                "histogram_supported": int(item["supported"][index]),
            })

    contrasts: dict[str, np.ndarray] = {}
    contrast_rows: list[dict[str, object]] = []
    point_values: dict[str, float] = {}
    for system, (same_name, cross_name) in PAIRS.items():
        same, cross = loaded[same_name], loaded[cross_name]
        distribution = exact_difference(cross["bootstrap_dg"], same["bootstrap_dg"])
        point = float(cross["dg"] - same["dg"])
        contrasts[system], point_values[system] = distribution, point
        low, median, high = q025_50_975(distribution)
        contrast_rows.append({
            "quantity": system,
            "definition": "cross-arm minus same-arm",
            "point_ddG_kcal_mol": point,
            "bootstrap_median_kcal_mol": median,
            "bootstrap_ci_low_kcal_mol": low,
            "bootstrap_ci_high_kcal_mol": high,
            "cross_over_same_contact_odds": math.exp(-point / RT),
            "probability_ddG_below_zero": float(np.mean(distribution < 0)),
            "first_half_ddG_kcal_mol": float(cross["dg_first"] - same["dg_first"]),
            "second_half_ddG_kcal_mol": float(cross["dg_second"] - same["dg_second"]),
        })
    rng = np.random.default_rng(SEED)
    c8_minus_c5 = (
        contrasts["C8"][rng.integers(0, contrasts["C8"].size, BOOTSTRAP_DRAWS)]
        - contrasts["C5"][rng.integers(0, contrasts["C5"].size, BOOTSTRAP_DRAWS)]
    )
    point = point_values["C8"] - point_values["C5"]
    low, median, high = q025_50_975(c8_minus_c5)
    contrast_rows.append({
        "quantity": "C8_minus_C5",
        "definition": "C8(cross−same) minus C5(cross−same)",
        "point_ddG_kcal_mol": point,
        "bootstrap_median_kcal_mol": median,
        "bootstrap_ci_low_kcal_mol": low,
        "bootstrap_ci_high_kcal_mol": high,
        "cross_over_same_contact_odds": math.exp(-point / RT),
        "probability_ddG_below_zero": float(np.mean(c8_minus_c5 < 0)),
        "first_half_ddG_kcal_mol": "",
        "second_half_ddG_kcal_mol": "",
    })

    sensitivity_rows: list[dict[str, object]] = []
    for minimum_count in (1.0, 5.0, 10.0, 20.0, 50.0):
        for split_nm in (0.45, 0.50, 0.55):
            try:
                values: dict[str, float] = {}
                for system, (same_name, cross_name) in PAIRS.items():
                    same = load_pathway(raw_root / same_name / "pathway", wham_root / same_name,
                                        minimum_count, split_nm)
                    cross = load_pathway(raw_root / cross_name / "pathway", wham_root / cross_name,
                                         minimum_count, split_nm)
                    values[system] = float(cross["dg"] - same["dg"])
                sensitivity_rows.append({
                    "minimum_histogram_count": minimum_count,
                    "contact_split_nm": split_nm,
                    "valid": 1,
                    "c5_cross_minus_same_kcal_mol": values["C5"],
                    "c8_cross_minus_same_kcal_mol": values["C8"],
                    "c8_minus_c5_kcal_mol": values["C8"] - values["C5"],
                    "expected_signs_preserved": int(
                        values["C5"] < 0 and values["C8"] < 0
                        and values["C8"] - values["C5"] < 0
                    ),
                    "failure": "",
                })
            except ValueError as error:
                sensitivity_rows.append({
                    "minimum_histogram_count": minimum_count,
                    "contact_split_nm": split_nm,
                    "valid": 0,
                    "c5_cross_minus_same_kcal_mol": "",
                    "c8_cross_minus_same_kcal_mol": "",
                    "c8_minus_c5_kcal_mol": "",
                    "expected_signs_preserved": "",
                    "failure": str(error),
                })

    write_csv(output / "adjacent_q2_pathway_contact_free_energies.csv", pathway_rows)
    write_csv(output / "adjacent_q2_same_cross_contrasts.csv", contrast_rows)
    write_csv(output / "adjacent_q2_contact_sensitivity.csv", sensitivity_rows)
    write_csv(output / "adjacent_q2_pmf_source_data.csv", source_rows)
    make_figure(loaded, pathway_rows, contrast_rows, output)
    write_reports(pathway_rows, contrast_rows, sensitivity_rows, output)
    summary = {
        "contact_region_nm": [CONTACT_LOW_NM, CONTACT_SPLIT_NM],
        "temperature_K": 313.15,
        "pathways": pathway_rows,
        "contrasts": contrast_rows,
        "valid_sensitivity_settings": sum(int(row["valid"]) for row in sensitivity_rows),
        "main_conclusion": (
            "On an adjacent-q2 background, cross-arm contact is favored in both C5 and C8, "
            "but the preference is substantially stronger in C8."
        ),
    }
    (output / "adjacent_q2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return summary


def make_figure(
    loaded: dict[str, dict[str, object]],
    pathway_rows: list[dict[str, object]],
    contrast_rows: list[dict[str, object]],
    output: Path,
) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    topology_colors = {"same-arm": "#4C566A", "cross-arm": "#D55E00"}
    system_colors = {"C5": "#4C78A8", "C8": "#D55E00", "C8_minus_C5": "#333333"}
    fig = plt.figure(figsize=(7.2, 5.35))
    grid = fig.add_gridspec(2, 2, height_ratios=(1.04, 1.0), hspace=0.43, wspace=0.42)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])]

    for ax, system in zip(axes[:2], ("C5", "C8")):
        ax.axvspan(CONTACT_LOW_NM, CONTACT_SPLIT_NM, color="#F0E6B2", alpha=0.65, lw=0)
        for name in PAIRS[system]:
            item = loaded[name]
            mask, x = item["supported"], item["pmf"][:, 0]
            color = topology_colors[str(item["topology"])]
            ax.fill_between(x, item["profile_low"], item["profile_high"], where=mask,
                            color=color, alpha=0.14, linewidth=0)
            ax.plot(x, item["aligned_pmf"][:, 1], color=color, lw=1.45,
                    label=f"{item['topology']} {item['pair']}")
        ax.axvline(CONTACT_SPLIT_NM, color="#8A8A8A", lw=0.7, ls="--")
        ax.set_xlim(0.30, 2.80)
        ax.set_ylim(-1.5, 12.5)
        ax.set_xlabel("vinyl midpoint distance (nm)")
        ax.set_title(f"A4-m-{system}, adjacent q2", fontsize=8, pad=4)
        ax.legend(loc="upper right", fontsize=6.1, handlelength=1.7)
    axes[0].set_ylabel("PMF aligned to non-contact partition\n(kcal mol$^{-1}$)")

    row_by_name = {str(row["pathway"]): row for row in pathway_rows}
    order = list(PATHWAYS)
    y = np.arange(len(order))[::-1]
    for ypos, name in zip(y, order):
        row, item = row_by_name[name], loaded[name]
        point = float(row["contact_dG_kcal_mol"])
        low, high = float(row["bootstrap_ci_low_kcal_mol"]), float(row["bootstrap_ci_high_kcal_mol"])
        color = topology_colors[str(item["topology"])]
        axes[2].plot([low, high], [ypos, ypos], color=color, lw=1.8)
        axes[2].plot(point, ypos, "o", color=color, ms=4.5, mec="white", mew=0.5)
    axes[2].set_yticks(y, [
        f"{loaded[name]['system']} {loaded[name]['topology']} {loaded[name]['pair']}" for name in order
    ])
    axes[2].set_xlabel(r"contact free energy, $\Delta G_{contact}$ (kcal mol$^{-1}$)")
    axes[2].set_xlim(1.2, 11.0)
    axes[2].text(0.98, 0.97, "lower = easier contact", transform=axes[2].transAxes,
                 ha="right", va="top", fontsize=6.2, color="#555555")

    y2 = np.arange(3)[::-1]
    labels = ("C5", "C8", "C8 − C5")
    for ypos, row, label in zip(y2, contrast_rows, labels):
        point = float(row["point_ddG_kcal_mol"])
        low, high = float(row["bootstrap_ci_low_kcal_mol"]), float(row["bootstrap_ci_high_kcal_mol"])
        color = system_colors[str(row["quantity"])]
        axes[3].plot([low, high], [ypos, ypos], color=color, lw=2.0)
        axes[3].plot(point, ypos, "o", color=color, ms=5.0, mec="white", mew=0.5)
        axes[3].text(point, ypos + 0.18, f"{point:+.2f}", color=color,
                     ha="center", va="bottom", fontsize=6.2)
    axes[3].axvline(0, color="#777777", lw=0.9, ls="--")
    axes[3].set_yticks(y2, labels)
    axes[3].set_xlim(-9.2, 0.8)
    axes[3].set_xlabel(r"$\Delta\Delta G$ = cross-arm − same-arm (kcal mol$^{-1}$)")
    axes[3].text(0.02, 0.97, "cross-arm favored", transform=axes[3].transAxes,
                 ha="left", va="top", fontsize=6.2, color="#D55E00")

    for label, ax in zip("abcd", axes):
        ax.text(-0.15, 1.05, label, transform=ax.transAxes, fontsize=9,
                fontweight="bold", ha="left", va="bottom")
    fig.suptitle("Adjacent-q2 preorganization favors cross-arm contact more strongly in A4-m-C8",
                 fontsize=9, y=0.995)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.105, top=0.94)
    base = output / "adjacent_q2_same_cross_umbrella_summary"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def write_reports(
    pathway_rows: list[dict[str, object]],
    contrast_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
    output: Path,
) -> None:
    by_name = {str(row["pathway"]): row for row in pathway_rows}
    c5, c8, difference = contrast_rows
    valid = [row for row in sensitivity_rows if int(row["valid"])]
    preserved = sum(int(row["expected_signs_preserved"]) for row in valid)
    c5_odds, c8_odds = float(c5["cross_over_same_contact_odds"]), float(c8["cross_over_same_contact_odds"])
    lines = [
        "# 相邻 q2 背景下 A4-m-C5/A4-m-C8 同臂与跨臂接触自由能",
        "",
        "## 核心结论",
        "",
        (
            f"在两个已反应位置为相邻关系的 q2 背景中，C5 的 ΔΔG(cross−same) = "
            f"{float(c5['point_ddG_kcal_mol']):+.3f} kcal/mol（95% bootstrap CI "
            f"{float(c5['bootstrap_ci_low_kcal_mol']):+.3f} 至 "
            f"{float(c5['bootstrap_ci_high_kcal_mol']):+.3f}），对应跨臂接触相对同臂约 "
            f"{c5_odds:.1f} 倍；C8 的 ΔΔG(cross−same) = "
            f"{float(c8['point_ddG_kcal_mol']):+.3f} kcal/mol（95% CI "
            f"{float(c8['bootstrap_ci_low_kcal_mol']):+.3f} 至 "
            f"{float(c8['bootstrap_ci_high_kcal_mol']):+.3f}），对应约 {c8_odds:.2g} 倍。"
        ),
        "",
        (
            f"C8−C5 的差异为 {float(difference['point_ddG_kcal_mol']):+.3f} kcal/mol（95% CI "
            f"{float(difference['bootstrap_ci_low_kcal_mol']):+.3f} 至 "
            f"{float(difference['bootstrap_ci_high_kcal_mol']):+.3f}；"
            f"P(C8 比 C5 更偏跨臂) = {float(difference['probability_ddG_below_zero']):.3f}）。"
            "因此，相邻 q2 构象背景下，两者都允许跨臂接触，但 C8 的跨臂预组织优势显著更强。"
        ),
        "",
        "## 单路径结果",
        "",
        "| 系统 | 拓扑 | Pair | ΔG_contact (kcal/mol) | 95% bootstrap CI | 支持距离 (nm) | 前/后半程 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name in PATHWAYS:
        row = by_name[name]
        lines.append(
            f"| {row['system']} | {row['topology']} | {row['pair']} | "
            f"{float(row['contact_dG_kcal_mol']):.3f} | "
            f"[{float(row['bootstrap_ci_low_kcal_mol']):.3f}, "
            f"{float(row['bootstrap_ci_high_kcal_mol']):.3f}] | "
            f"{float(row['support_min_nm']):.3f}–{float(row['support_max_nm']):.3f} | "
            f"{float(row['first_half_contact_dG_kcal_mol']):.3f}/"
            f"{float(row['second_half_contact_dG_kcal_mol']):.3f} |"
        )
    lines.extend([
        "",
        "## WHAM 修正与质量控制",
        "",
        "原始远程 WHAM 对所有路径统一使用 2.65 nm 作为 PMF 零点。C5 同臂路径实际仅采样到约 1.67 nm，因此该零点落在完全未采样区，原始 `pmf.xvg` 被写成全零。这里没有重跑或修改 MD 轨迹，而是使用同一批 TPR/pullx 数据，将每条路径的 WHAM 上限和零点放回各自实际采样区间后重新计算。修正后四条 PMF 均非零，且每条均得到 200 条不同的 bootstrap profile。",
        "",
        f"在可计算的 {len(valid)} 组敏感性设置中，{preserved}/{len(valid)} 均保持 C5 跨臂有利、C8 跨臂有利以及 C8 比 C5 更偏跨臂的符号。接触边界测试为 0.45/0.50/0.55 nm，直方图阈值测试为 1/5/10/20/50 counts；其余设置因接触区支持点不足而被明确标记为不可计算，而非填入零值。",
        "",
        "C5 的前/后半程 cross−same 分别为 "
        f"{float(c5['first_half_ddG_kcal_mol']):+.3f}/{float(c5['second_half_ddG_kcal_mol']):+.3f} kcal/mol；"
        "C8 分别为 "
        f"{float(c8['first_half_ddG_kcal_mol']):+.3f}/{float(c8['second_half_ddG_kcal_mol']):+.3f} kcal/mol，方向一致。",
        "",
        "## 对反应来源的含义",
        "",
        "这组结果支持“C8 的跨臂来源包含更强的构象接触预组织”这一判断。但它同时显示 C5 在相邻 q2 背景下也偏跨臂，因此不能把 C5 的最终同臂产物简单归因于跨臂接触完全不可达。C5 的实验选择性仍可能主要在更早的 q0/q1 分支、双键相对取向、催化剂进入方式或金属杂环过渡态应变中产生。",
        "",
        "## 解释边界",
        "",
        "- 每个系统目前只有一个锚点匹配的 same/cross pair，bootstrap 只反映固定路径内的不确定性，不包含配对和初始构象之间的方差。",
        "- 反应坐标只有乙烯基中点距离，没有限制双键夹角、二面角和 Ru 配位几何。",
        "- 这是无催化剂经典力场下的接触自由能，不是烯烃复分解的化学活化能垒。",
        "- C8 同臂在最短距离端仍表现出较强几何阻碍；正式发表前建议增加独立初始构象重复，并用二维 distance+orientation 或 NAC 指标验证。",
        "",
    ])
    (output / "adjacent_q2_final_report_zh.md").write_text("\n".join(lines), encoding="utf-8")
    legend = (
        "**Figure | Same-arm and cross-arm contact preorganization on an adjacent-q2 background.** "
        "**a,b,** Histogram-supported midpoint-distance PMFs for A4-m-C5 and A4-m-C8, aligned to "
        "their non-contact partition functions. Shading shows 95% trajectory-bootstrap envelopes "
        "from 200 WHAM profiles; the pale band marks the 0.325–0.500 nm contact interval. "
        "**c,** Contact free energies obtained by Boltzmann integration of contact and supported "
        "non-contact regions. **d,** Contrasts defined as ΔΔG = ΔGcontact(cross-arm) − "
        "ΔGcontact(same-arm); negative values favor cross-arm contact. Points are WHAM estimates "
        "and bars are 95% bootstrap intervals. Each pathway contains 24 windows and uses "
        "100–500 ps from each 500 ps production trajectory at 313.15 K. Source data are provided "
        "in adjacent_q2_pmf_source_data.csv."
    )
    (output / "adjacent_q2_figure_legend.md").write_text(legend + "\n", encoding="utf-8")
    qa = """# Adjacent-q2 figure and analysis QA

- Core conclusion: adjacent-q2 cross-arm contact is favored in both systems and substantially more strongly in C8.
- Figure archetype: quantitative grid with the contrast forest plot as the hero evidence.
- Backend: Python/matplotlib only.
- Output contract: 7.2-inch double-column figure; editable SVG/PDF plus 300-dpi PNG and 600-dpi LZW TIFF.
- Statistics: 200 trajectory-bootstrap WHAM profiles per pathway; intervals are 2.5–97.5 percentiles. No between-pair or between-replica variance is represented.
- Source data: `adjacent_q2_pmf_source_data.csv`, `adjacent_q2_pathway_contact_free_energies.csv`, and `adjacent_q2_same_cross_contrasts.csv`.
- Integrity: no microscopy or image manipulation; all panels are generated directly from numerical source data.
- Reviewer risk: one pair per topology, one-dimensional coordinate, unequal physically supported distance ranges, and no Ru catalyst or chemical reaction coordinate.
- WHAM audit: the original C5 same-arm PMF was all zero because `zprof0=2.65 nm` lay outside its sampled support. All four pathways were recomputed on pathway-specific sampled ranges before comparison.
"""
    (output / "adjacent_q2_qa_notes.md").write_text(qa, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "results/a4_q2_adjacent_umbrella_analysis/final",
    )
    args = parser.parse_args()
    result = analyze(project_root, args.output.resolve())
    print(json.dumps(result["contrasts"], ensure_ascii=False, indent=2))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
