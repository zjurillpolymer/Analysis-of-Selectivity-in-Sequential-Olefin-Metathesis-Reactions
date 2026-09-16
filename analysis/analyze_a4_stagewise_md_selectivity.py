#!/usr/bin/env python3
"""Build a stagewise A4-m-C5/C8 selectivity map from existing MD analyses.

The analysis deliberately stops short of a product-yield prediction.  It
combines comparable, per-pair catalyst-free NAC free-energy proxies at q0, q1,
and q2, then separates intrinsic per-pair preference from candidate-count
multiplicity through

    P_alt(lambda) = m_alt r**lambda / (m_alt r**lambda + m_ref),
    r = exp(-DeltaDeltaG_alt-ref / RT).

lambda=1 assumes closure propensity is proportional to NAC occupancy; lambda=0
retains candidate multiplicity only.  Missing q2 histories prevent multiplying
the local decisions into a complete product distribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


R_KCAL = 0.00198720425864083
TEMPERATURE_K = 313.15
RT = R_KCAL * TEMPERATURE_K
PRIMARY_SETTING = "d5_axis45_side45"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ddg_ci_from_ratio_ci(low_ratio: float, high_ratio: float) -> tuple[float, float]:
    return -RT * math.log(high_ratio), -RT * math.log(low_ratio)


def q0_rows(root: Path) -> list[dict[str, object]]:
    specs = (
        ("C5", root / "results/a4_m_c5_contact_origin/full_geometry_summary.json"),
        ("C8", root / "results/a4_m_c8_contact_origin/full_geometry_summary.json"),
    )
    rows: list[dict[str, object]] = []
    for system, path in specs:
        payload = json_load(path)
        overall = payload["overall"]
        intervals = payload["ratio_95pct_CI"]
        free = payload["free_energy_proxies_kcal_mol"]
        total_low, total_high = ddg_ci_from_ratio_ci(*intervals["nac_cross_over_same"])
        contact_low, contact_high = ddg_ci_from_ratio_ci(*intervals["contact_cross_over_same"])
        orientation_low, orientation_high = ddg_ci_from_ratio_ci(
            *intervals["orientation_cross_over_same"]
        )
        rows.append({
            "context_id": "q0_topology",
            "system": system,
            "state": f"A4_m_{system}_q0",
            "stage": "q0_to_q1",
            "decision": "cross_minus_same",
            "alternative": "cross",
            "reference": "same",
            "alternative_pair_count": 24,
            "reference_pair_count": 4,
            "replicas": int(payload["replicas"]),
            "frames": int(payload["frames"]),
            "ddG_contact_kcal_mol": free["contact_ddG_cross_minus_same"],
            "ddG_contact_ci_low_kcal_mol": contact_low,
            "ddG_contact_ci_high_kcal_mol": contact_high,
            "ddG_orientation_kcal_mol": free["orientation_ddG_cross_minus_same"],
            "ddG_orientation_ci_low_kcal_mol": orientation_low,
            "ddG_orientation_ci_high_kcal_mol": orientation_high,
            "ddG_total_kcal_mol": free["nac_ddG_cross_minus_same"],
            "ddG_total_ci_low_kcal_mol": total_low,
            "ddG_total_ci_high_kcal_mol": total_high,
            "source_method": "unbiased_MD_full_trajectory_NAC",
        })
    return rows


def invert_contrast(row: Mapping[str, str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for component in ("contact", "orientation", "total_nac"):
        point = float(row[f"ddG_{component}_point_kcal_mol"])
        low = float(row[f"ddG_{component}_ci_low_kcal_mol"])
        high = float(row[f"ddG_{component}_ci_high_kcal_mol"])
        output[f"ddG_{component}"] = -point
        output[f"ddG_{component}_low"] = -high
        output[f"ddG_{component}_high"] = -low
    return output


def keep_contrast(row: Mapping[str, str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for component in ("contact", "orientation", "total_nac"):
        output[f"ddG_{component}"] = float(row[f"ddG_{component}_point_kcal_mol"])
        output[f"ddG_{component}_low"] = float(row[f"ddG_{component}_ci_low_kcal_mol"])
        output[f"ddG_{component}_high"] = float(row[f"ddG_{component}_ci_high_kcal_mol"])
    return output


def q1_rows(root: Path) -> list[dict[str, object]]:
    source = root / (
        "results/a4_postreaction_nac_analysis/b6-gromacs-parallel-qpd79/"
        "free_energy_contrasts.csv"
    )
    selected = [row for row in read_csv(source) if row["setting_id"] == PRIMARY_SETTING]
    rows: list[dict[str, object]] = []
    for row in selected:
        system = "C5" if row["system"].endswith("C5") else "C8"
        if row["contrast"] == "cross_minus_same":
            values = keep_contrast(row)
            context_id = "q1_topology"
            alternative, reference = "cross", "same"
            alt_count, ref_count = 13, 2
        elif row["contrast"] == "reuse_minus_switch":
            values = invert_contrast(row)
            context_id = "q1_memory"
            alternative, reference = "switch", "reuse"
            alt_count, ref_count = 12, 1
        else:
            continue
        rows.append({
            "context_id": context_id,
            "system": system,
            "state": row["state"],
            "stage": "q1_to_q2",
            "decision": f"{alternative}_minus_{reference}",
            "alternative": alternative,
            "reference": reference,
            "alternative_pair_count": alt_count,
            "reference_pair_count": ref_count,
            "replicas": int(row["replicas"]),
            "frames": "",
            "ddG_contact_kcal_mol": values["ddG_contact"],
            "ddG_contact_ci_low_kcal_mol": values["ddG_contact_low"],
            "ddG_contact_ci_high_kcal_mol": values["ddG_contact_high"],
            "ddG_orientation_kcal_mol": values["ddG_orientation"],
            "ddG_orientation_ci_low_kcal_mol": values["ddG_orientation_low"],
            "ddG_orientation_ci_high_kcal_mol": values["ddG_orientation_high"],
            "ddG_total_kcal_mol": values["ddG_total_nac"],
            "ddG_total_ci_low_kcal_mol": values["ddG_total_nac_low"],
            "ddG_total_ci_high_kcal_mol": values["ddG_total_nac_high"],
            "source_method": "unbiased_MD_NAC",
        })
    return rows


def q2_rows(root: Path) -> list[dict[str, object]]:
    source = root / "results/a4_q2_unified_nac_analysis/final/q2_unified_decomposition.csv"
    rows: list[dict[str, object]] = []
    for row in read_csv(source):
        system = "C5" if row["system"].endswith("C5") else "C8"
        if row["branch"] == "adjacent":
            context_id = "q2_adjacent_topology"
            alt_count, ref_count = 5, 1
        else:
            context_id = "q2_opposite_memory"
            alt_count, ref_count = 4, 2
        rows.append({
            "context_id": context_id,
            "system": system,
            "state": row["state"],
            "stage": "q2_to_q3",
            "decision": row["contrast"],
            "alternative": row["alternative_group"].replace("_arm", ""),
            "reference": row["reference_group"].replace("_arm", ""),
            "alternative_pair_count": alt_count,
            "reference_pair_count": ref_count,
            "replicas": int(row["replicas"]),
            "frames": "",
            "ddG_contact_kcal_mol": float(row["ddG_contact_point_kcal_mol"]),
            "ddG_contact_ci_low_kcal_mol": float(row["ddG_contact_ci_low_kcal_mol"]),
            "ddG_contact_ci_high_kcal_mol": float(row["ddG_contact_ci_high_kcal_mol"]),
            "ddG_orientation_kcal_mol": float(row["ddG_orientation_point_kcal_mol"]),
            "ddG_orientation_ci_low_kcal_mol": float(row["ddG_orientation_ci_low_kcal_mol"]),
            "ddG_orientation_ci_high_kcal_mol": float(row["ddG_orientation_ci_high_kcal_mol"]),
            "ddG_total_kcal_mol": float(row["ddG_total_nac_point_kcal_mol"]),
            "ddG_total_ci_low_kcal_mol": float(row["ddG_total_nac_ci_low_kcal_mol"]),
            "ddG_total_ci_high_kcal_mol": float(row["ddG_total_nac_ci_high_kcal_mol"]),
            "source_method": "unbiased_MD_NAC",
        })
    return rows


def enrich_rows(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        ddg = float(row["ddG_total_kcal_mol"])
        low = float(row["ddG_total_ci_low_kcal_mol"])
        high = float(row["ddG_total_ci_high_kcal_mol"])
        odds = math.exp(-ddg / RT)
        alt_count = int(row["alternative_pair_count"])
        ref_count = int(row["reference_pair_count"])
        any_odds = alt_count * odds / ref_count
        row.update({
            "direction_resolved_95pct": not (low <= 0.0 <= high),
            "per_pair_alternative_over_reference_odds": odds,
            "per_pair_alternative_probability": odds / (1.0 + odds),
            "candidate_count_ratio_alternative_over_reference": alt_count / ref_count,
            "any_candidate_alternative_over_reference_odds_lambda1": any_odds,
            "any_candidate_alternative_probability_lambda1": any_odds / (1.0 + any_odds),
        })
        output.append(row)
    return output


def sensitivity_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        odds = float(row["per_pair_alternative_over_reference_odds"])
        alt_count = int(row["alternative_pair_count"])
        ref_count = int(row["reference_pair_count"])
        for response_exponent in np.linspace(0.0, 1.5, 151):
            any_odds = alt_count * odds ** float(response_exponent) / ref_count
            output.append({
                "context_id": row["context_id"],
                "system": row["system"],
                "decision": row["decision"],
                "response_exponent_lambda": float(response_exponent),
                "any_candidate_alternative_probability": any_odds / (1.0 + any_odds),
                "any_candidate_alternative_over_reference_odds": any_odds,
            })
    return output


def probability_interval(row: Mapping[str, object], response_exponent: float = 1.0) -> tuple[float, float]:
    alt_count = int(row["alternative_pair_count"])
    ref_count = int(row["reference_pair_count"])
    low_ddg = float(row["ddG_total_ci_low_kcal_mol"])
    high_ddg = float(row["ddG_total_ci_high_kcal_mol"])

    def probability(ddg: float) -> float:
        odds = alt_count * math.exp(-ddg / RT) ** response_exponent / ref_count
        return odds / (1.0 + odds)

    return probability(high_ddg), probability(low_ddg)


def make_figure(output: Path, rows: Sequence[Mapping[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".mplconfig"))
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
    })
    c5, c8, contact, orientation = "#2878B5", "#D95F02", "#4C78A8", "#E7BA52"
    contexts = [
        ("q0_topology", "q0 topology\ncross - same"),
        ("q1_topology", "q1 topology\ncross - same"),
        ("q1_memory", "q1 memory\nswitch - reuse"),
        ("q2_adjacent_topology", "q2 adjacent\ncross - same"),
        ("q2_opposite_memory", "q2 opposite\nswitch - reuse"),
    ]
    lookup = {(str(row["context_id"]), str(row["system"])): row for row in rows}
    fig = plt.figure(figsize=(7.2, 6.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.05, 0.95), height_ratios=(1.0, 1.0))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    matrix = np.asarray([
        [float(lookup[(context, system)]["ddG_total_kcal_mol"]) for system in ("C5", "C8")]
        for context, _ in contexts
    ])
    image = ax_a.imshow(
        matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-5.0, vcenter=0.0, vmax=5.0), aspect="auto"
    )
    for i, (context, _) in enumerate(contexts):
        for j, system in enumerate(("C5", "C8")):
            row = lookup[(context, system)]
            value = float(row["ddG_total_kcal_mol"])
            color = "white" if abs(value) > 2.7 else "black"
            ax_a.text(j, i, f"{value:+.2f}", ha="center", va="center", color=color, fontsize=7.2)
            if not bool(row["direction_resolved_95pct"]):
                ax_a.add_patch(plt.Rectangle((j - 0.47, i - 0.47), 0.94, 0.94,
                                             fill=False, ec="#6E6E6E", lw=1.5, ls="--"))
    ax_a.set_xticks((0, 1), ("A4-m-C5", "A4-m-C8"))
    ax_a.set_yticks(range(len(contexts)), [label for _, label in contexts])
    ax_a.set_title(r"$\bf{a}$  Selectivity diverges in route memory", loc="left", pad=7)
    ax_a.tick_params(length=0)
    cbar = fig.colorbar(image, ax=ax_a, fraction=0.047, pad=0.035)
    cbar.set_label(r"$\Delta\Delta G_{\rm alt-ref}^{\rm NAC}$ (kcal mol$^{-1}$)")
    cbar.set_ticks((-5, 0, 5), labels=(r"$\leq-5$", "0", r"$\geq+5$"))
    ax_a.text(0.0, -0.16, "Negative favours cross/switch; dashed border: 95% CI crosses zero",
              transform=ax_a.transAxes, fontsize=6.3, va="top")

    memory_contexts = [
        ("q1_memory", "C5", "C5 q1"),
        ("q1_memory", "C8", "C8 q1"),
        ("q2_opposite_memory", "C5", "C5 q2 opposite"),
        ("q2_opposite_memory", "C8", "C8 q2 opposite"),
    ]
    y = np.arange(len(memory_contexts))
    for yi, (context, system, _) in enumerate(memory_contexts):
        row = lookup[(context, system)]
        contact_value = float(row["ddG_contact_kcal_mol"])
        total_value = float(row["ddG_total_kcal_mol"])
        ax_b.plot((0, contact_value), (yi, yi), color=contact, lw=5.0, solid_capstyle="butt")
        ax_b.plot((contact_value, total_value), (yi, yi), color=orientation, lw=5.0,
                  solid_capstyle="butt")
        ax_b.scatter((total_value,), (yi,), s=18, color=c5 if system == "C5" else c8,
                     edgecolor="white", linewidth=0.5, zorder=3)
    ax_b.axvline(0, color="#777777", lw=0.8, ls="--")
    ax_b.set_yticks(y, [label for _, _, label in memory_contexts])
    ax_b.invert_yaxis()
    ax_b.set_xlabel(r"switch - reuse $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax_b.set_title(r"$\bf{b}$  Contact accessibility carries the memory signal", loc="left", pad=7)
    ax_b.plot([], [], color=contact, lw=5, label="contact")
    ax_b.plot([], [], color=orientation, lw=5, label="orientation | contact")
    ax_b.legend(loc="upper right", fontsize=6.4)

    lam = np.linspace(0.0, 1.5, 151)
    curve_specs = [
        ("q1_memory", "C5", c5, "-", "C5 q1"),
        ("q2_opposite_memory", "C5", c5, "--", "C5 q2 opposite"),
        ("q1_memory", "C8", c8, "-", "C8 q1"),
        ("q2_opposite_memory", "C8", c8, "--", "C8 q2 opposite"),
    ]
    endpoint_offsets = {"C5 q1": -0.025, "C5 q2 opposite": -0.015,
                        "C8 q1": -0.035, "C8 q2 opposite": 0.012}
    for context, system, color, style, label in curve_specs:
        row = lookup[(context, system)]
        odds = float(row["per_pair_alternative_over_reference_odds"])
        ratio = int(row["alternative_pair_count"]) / int(row["reference_pair_count"])
        values = ratio * odds ** lam
        probabilities = values / (1.0 + values)
        ax_c.plot(lam, probabilities, color=color, ls=style, lw=1.7)
        ax_c.text(1.515, probabilities[-1] + endpoint_offsets[label], label,
                  color=color, va="center", fontsize=6.5, clip_on=False)
    ax_c.axhline(0.5, color="#888888", lw=0.7, ls=":")
    ax_c.axvline(1.0, color="#888888", lw=0.7, ls=":")
    ax_c.text(1.0, 0.025, r"$\lambda=1$", ha="center", va="bottom", color="#666666")
    ax_c.set_xlim(0.0, 1.62)
    ax_c.set_ylim(0.0, 1.04)
    ax_c.set_xlabel(r"NAC-to-closure response exponent, $\lambda$")
    ax_c.set_ylabel("Probability of any switch candidate")
    ax_c.set_title(r"$\bf{c}$  Multiplicity can mask C5 per-pair reuse preference", loc="left", pad=7)
    ax_c.text(0.0, -0.22,
              r"$P_{\rm switch}=m_{\rm switch}r^\lambda/(m_{\rm switch}r^\lambda+m_{\rm reuse})$; "
              r"$\lambda=0$ is multiplicity-only and $\lambda=1$ is proportional to NAC occupancy.",
              transform=ax_c.transAxes, fontsize=6.3, va="top")

    base = output / "a4_stagewise_md_selectivity"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def build_report(rows: Sequence[Mapping[str, object]]) -> str:
    lookup = {(str(row["context_id"]), str(row["system"])): row for row in rows}
    lines = [
        "# A4-m-C5/C8 的阶段性 MD 选择性图谱",
        "",
        "## 核心结论",
        "",
        "现有无偏 MD 并不支持“C5 从一开始就偏同臂、C8 从一开始就偏跨臂”的简单图景。"
        "q0 与 q1 的单对 cross−same NAC 对两种分子都为负；真正稳定的 C5/C8 分叉出现在"
        "反应历史记忆：C5 在特定历史下偏向 reuse，而 C8 强烈偏向 switch。",
        "",
        "## 统一的局部决策",
        "",
        "所有数值统一为 alternative−reference；负值有利于 cross/switch，正值有利于 same/reuse。",
        "",
        "| 决策环境 | C5 ΔΔG | C8 ΔΔG | 解释 |",
        "|---|---:|---:|---|",
    ]
    labels = (
        ("q0_topology", "q0：cross−same"),
        ("q1_topology", "q1：cross−same"),
        ("q1_memory", "q1：switch−reuse"),
        ("q2_adjacent_topology", "q2 adjacent：cross−same"),
        ("q2_opposite_memory", "q2 opposite：switch−reuse"),
    )
    for context, label in labels:
        c5 = lookup[(context, "C5")]
        c8 = lookup[(context, "C8")]
        c5_text = f"{float(c5['ddG_total_kcal_mol']):+.3f} [{float(c5['ddG_total_ci_low_kcal_mol']):+.3f}, {float(c5['ddG_total_ci_high_kcal_mol']):+.3f}]"
        c8_text = f"{float(c8['ddG_total_kcal_mol']):+.3f} [{float(c8['ddG_total_ci_low_kcal_mol']):+.3f}, {float(c8['ddG_total_ci_high_kcal_mol']):+.3f}]"
        if context in {"q0_topology", "q1_topology", "q2_adjacent_topology"}:
            interpretation = "两者均偏 cross；不是选择性反转来源"
        elif context == "q1_memory":
            interpretation = "C5 点估计偏 reuse 但区间未定；C8 稳健偏 switch"
        else:
            interpretation = "C5 稳健偏 reuse；C8 稳健偏 switch"
        lines.append(f"| {label} | {c5_text} | {c8_text} | {interpretation} |")

    c5_q1 = lookup[("q1_memory", "C5")]
    c8_q1 = lookup[("q1_memory", "C8")]
    c5_q2 = lookup[("q2_opposite_memory", "C5")]
    c8_q2 = lookup[("q2_opposite_memory", "C8")]
    lines.extend([
        "",
        "## 反应历史记忆的来源",
        "",
        f"- q1 C5：switch−reuse = {float(c5_q1['ddG_total_kcal_mol']):+.3f} kcal mol⁻¹，其中接触项 "
        f"{float(c5_q1['ddG_contact_kcal_mol']):+.3f}，取向项 {float(c5_q1['ddG_orientation_kcal_mol']):+.3f}。",
        f"- q1 C8：switch−reuse = {float(c8_q1['ddG_total_kcal_mol']):+.3f}，其中接触项 "
        f"{float(c8_q1['ddG_contact_kcal_mol']):+.3f}，取向项 {float(c8_q1['ddG_orientation_kcal_mol']):+.3f}。",
        f"- q2 opposite C5：switch−reuse = {float(c5_q2['ddG_total_kcal_mol']):+.3f}，其中接触项 "
        f"{float(c5_q2['ddG_contact_kcal_mol']):+.3f}。",
        f"- q2 opposite C8：switch−reuse = {float(c8_q2['ddG_total_kcal_mol']):+.3f}，其中接触项 "
        f"{float(c8_q2['ddG_contact_kcal_mol']):+.3f}。",
        "",
        "因此，记忆效应主要来自两个候选乙烯基能否进入接触区；在已经接触的条件下，相对取向只提供次要修正。",
        "",
        "## 单对偏好与候选数多重性",
        "",
        "按 k ∝ P(NAC)（λ=1）仅作局部概率映射：",
        "",
    ])
    for row in (c5_q1, c8_q1, c5_q2, c8_q2):
        lines.append(
            f"- {row['system']} {row['context_id']}：单对 switch 概率 "
            f"{100*float(row['per_pair_alternative_probability']):.1f}%；计入 "
            f"{row['alternative_pair_count']}:{row['reference_pair_count']} 个 switch:reuse 候选后，"
            f"任一 switch 概率 {100*float(row['any_candidate_alternative_probability_lambda1']):.1f}%。"
        )
    lines.extend([
        "",
        "C5 q1 的单个 reuse pair 虽优于单个 switch pair，但 12 个 switch 候选会在 λ=1 下把总 switch 概率推到约 77%。"
        "C5 q2 opposite 只有 4:2 的候选数比，此时 reuse 的单对优势足以保留下来。"
        "因此讨论实验产物时必须同时报告单对构象偏好和候选数，不能只引用一个 ΔΔG。",
        "按点估计计算，C5 q1 需要 λ > 1.93 才能让任一 reuse 的总权重超过全部 switch；但该 q1 点估计的区间仍跨零。"
        "C5 q2 opposite 的翻转阈值仅为 λ = 0.447，因而在 λ=1 映射下稳定表现为 reuse 占优。",
        "",
        "## 为什么目前不能直接给出 R2/R4 产率",
        "",
        "q2 adjacent 与 q2 opposite 都是 q1 发生 switch 后的两个代表性历史。当前没有 q1 选择 reuse 后形成的 q2-reuse 状态，"
        "也没有 q1 选择 same-arm 后的 q2 状态。因此这些局部概率不能沿 q0→q1→q2 简单相乘成完整产物分布。",
        "",
        "最小补算应优先构建 C5、C8 各一个 q2-reuse 状态，并各做 3 条 2 ns 无偏显式 DCM 轨迹；"
        "这只增加 12 ns 总生产采样，却能补上当前反应网络最关键的缺边。C5 q1 的 reuse−switch 区间仍跨零，"
        "若希望做严格统计，再单独增加 C5 q1 的独立重复。",
        "",
        "## 解释边界",
        "",
        "- 所有 ΔΔG 都是无催化剂 NAC 构象自由能代理，不是 Ru 催化活化能垒。",
        "- λ 是 NAC 占据率到闭合倾向的响应指数，不是由当前数据拟合出的速率常数。",
        "- q0 使用 4 条 4 ns 轨迹；q1/q2 当前统计单位为 3 条独立 2 ns 轨迹。",
        "- 本图是阶段性局部决策图谱，不是完整动力学网络。",
    ])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output", type=Path,
        default=root / "results/a4_stagewise_md_selectivity/final",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = enrich_rows([*q0_rows(root), *q1_rows(root), *q2_rows(root)])
    expected = {(context, system) for context in (
        "q0_topology", "q1_topology", "q1_memory",
        "q2_adjacent_topology", "q2_opposite_memory",
    ) for system in ("C5", "C8")}
    observed = {(str(row["context_id"]), str(row["system"])) for row in rows}
    if observed != expected:
        raise ValueError(f"decision-table mismatch: missing={expected-observed}, extra={observed-expected}")

    write_csv(output / "stagewise_md_decisions.csv", rows)
    sensitivity = sensitivity_rows(rows)
    write_csv(output / "nac_response_sensitivity.csv", sensitivity)
    make_figure(output, rows)
    (output / "stagewise_md_selectivity_report_zh.md").write_text(
        build_report(rows), encoding="utf-8"
    )
    (output / "figure_contract.md").write_text(
        "# Figure contract\n\n"
        "- Core conclusion: topology does not invert between C5 and C8; route-memory contact accessibility creates the divergence.\n"
        "- Archetype: quantitative grid with a dominant stage map and two mechanism/robustness panels.\n"
        "- Backend: Python only.\n"
        "- Final size: 7.2 × 6.0 inch, double-column.\n"
        "- Panel a: unified stagewise NAC contrasts.\n"
        "- Panel b: contact/orientation decomposition of route memory.\n"
        "- Panel c: candidate-multiplicity sensitivity to the NAC response exponent.\n"
        "- Reviewer risk: local decisions are not a complete product-yield network; q2 reuse/same histories are missing.\n",
        encoding="utf-8",
    )
    (output / "figure_legend_en.md").write_text(
        "**Fig. 1 | Stagewise MD map of A4-m-C5 and A4-m-C8 configurational selectivity.** "
        "a, Alternative-minus-reference near-attack-conformation (NAC) free-energy proxies from unbiased explicit-dichloromethane MD; negative values favour cross-arm or switch alternatives. Dashed borders mark comparisons whose replica-bootstrap 95% interval includes zero. "
        "b, Contact and conditional-orientation contributions to switch-minus-reuse selectivity after one reaction and in the opposite-q2 history. "
        "c, Probability of any switch candidate after accounting for candidate multiplicity as a function of the NAC-to-closure response exponent. lambda=1 assumes closure propensity proportional to NAC occupancy; lambda=0 retains multiplicity only. q0 uses four independent 4-ns trajectories and q1/q2 use three independent 2-ns trajectories per state. These catalyst-free configurational proxies do not represent Ru-catalysed activation barriers, and the sampled states do not form a complete product-yield network. Source data are provided as CSV files.\n",
        encoding="utf-8",
    )
    (output / "figure_qa_notes.md").write_text(
        "# Figure QA notes\n\n"
        "- Core claim and five decision contexts are declared in `figure_contract.md`.\n"
        "- Quantitative source data are in `stagewise_md_decisions.csv`; lambda sensitivity is in `nac_response_sensitivity.csv`.\n"
        "- q0: four independent 4-ns trajectories per system; q1/q2: three independent 2-ns trajectories per state.\n"
        "- Intervals are hierarchical block/replica bootstrap for q0 and replica bootstrap for q1/q2. No p values are claimed.\n"
        "- The dashed heatmap border denotes an interval crossing zero; it is not a significance star.\n"
        "- SVG and PDF retain editable text; TIFF is exported at 600 dpi and PNG at 300 dpi.\n"
        "- Color is redundant with signed numeric labels and line style, so interpretation does not depend on red/blue perception alone.\n"
        "- Image panels are absent; no crop, contrast, pseudo-colour microscopy, or other image-integrity operation applies.\n"
        "- Principal reviewer risk: q2-reuse and q2-same histories are unsampled, so the figure is explicitly a local-decision map rather than a yield model.\n",
        encoding="utf-8",
    )
    qa = {
        "decision_rows": len(rows),
        "sensitivity_rows": len(sensitivity),
        "all_expected_contexts_present": observed == expected,
        "all_probabilities_valid": all(
            0.0 < float(row["any_candidate_alternative_probability_lambda1"]) < 1.0
            for row in rows
        ),
        "additivity_max_abs_error_kcal_mol": max(
            abs(float(row["ddG_contact_kcal_mol"]) + float(row["ddG_orientation_kcal_mol"])
                - float(row["ddG_total_kcal_mol"])) for row in rows
        ),
        "route_memory_direction_pattern": {
            "C5_q2_opposite_reuse_favored": float(
                next(row for row in rows if row["context_id"] == "q2_opposite_memory" and row["system"] == "C5")["ddG_total_kcal_mol"]
            ) > 0,
            "C8_q1_switch_favored": float(
                next(row for row in rows if row["context_id"] == "q1_memory" and row["system"] == "C8")["ddG_total_kcal_mol"]
            ) < 0,
            "C8_q2_opposite_switch_favored": float(
                next(row for row in rows if row["context_id"] == "q2_opposite_memory" and row["system"] == "C8")["ddG_total_kcal_mol"]
            ) < 0,
        },
        "interpretation_boundary": "partial local-decision map; not a complete yield model",
    }
    (output / "qa_summary.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()
