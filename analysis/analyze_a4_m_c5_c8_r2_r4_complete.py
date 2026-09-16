#!/usr/bin/env python3
"""Unified A4-m-C5--C8 q1/q2 propagation into target-only R2/R4 yields.

This analysis implements the reaction graph agreed for the A4 series:

* q0 is fixed to cross;
* q1 full reuse -> two R2 molecules;
* q1 partial reuse -> one R4 in the target-only model;
* q1 full switch -> q2 opposite;
* q2 opposite reuse -> two R2, while q2 opposite switch -> one R4.

Unbiased probabilities are obtained by summing the soft-NAC weights of every
candidate pair in each class before averaging independent replicas.  Umbrella
probabilities multiply each representative-path soft-NAC probability by the
corresponding candidate multiplicity.  The latter are reported as sensitivity
and mechanistic evidence, not silently substituted for all-pair populations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PRIMARY = "d5_axis45_side45"
SYSTEMS = ("C5", "C6", "C7", "C8")
Q1_KEYS = ("full_reuse", "partial_reuse", "full_switch")
Q1_HISTORY = {
    "reuse_arm_pair": "full_reuse",
    "one_reacted_arm": "partial_reuse",
    "untouched_arm_pair": "full_switch",
}
Q1_MULTIPLICITY = {"full_reuse": 1, "partial_reuse": 8, "full_switch": 4}
Q2_MULTIPLICITY = {"R2x2": 2, "R4": 4}
RT_KCAL = 0.00198720425864083 * 313.15


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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


def normalize(values: dict[str, float], keys: Iterable[str]) -> dict[str, float]:
    keys = tuple(keys)
    total = sum(values.get(key, 0.0) for key in keys)
    if total <= 0:
        raise ValueError(f"non-positive normalization total: {values}")
    return {key: values.get(key, 0.0) / total for key in keys}


def mean_replica_weights(
    by_replica: dict[str, dict[str, float]], keys: Iterable[str]
) -> dict[str, float]:
    keys = tuple(keys)
    return {
        key: float(np.mean([values.get(key, 0.0) for values in by_replica.values()]))
        for key in keys
    }


def q1_unbiased_by_replica(root: Path, system: str) -> dict[str, dict[str, float]]:
    if system in ("C5", "C8"):
        extended = root / "results/a4_m_c5_c8_q1_10ns_analysis_all_pairs/pair_nac_statistics.csv"
        if extended.is_file():
            path = extended
            rows = [
                row for row in read_csv(path)
                if row["state"] == f"A4_m_{system}_1_cis"
                and row["setting_id"] == PRIMARY
            ]
        else:
            path = root / "results/a4_postreaction_nac_analysis/reuse_switch_dynamics/reuse_switch_pair_summary.csv"
            rows = [row for row in read_csv(path) if row["state"] == f"A4_m_{system}_1_cis"]
    else:
        path = root / "results/a4_m_c6_c7_q1_10ns_analysis_all_pairs/pair_nac_statistics.csv"
        rows = [
            row for row in read_csv(path)
            if row["state"] == f"A4_m_{system}_1_cis"
            and row["setting_id"] == PRIMARY
        ]
    by_replica: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        category = Q1_HISTORY.get(row["history_class"])
        if category is not None:
            by_replica[row["replica"]][category] += float(row["mean_soft_nac"])
    if len(by_replica) != 3:
        raise ValueError(f"{system}: expected 3 q1 replicas, found {len(by_replica)}")
    return {replica: dict(values) for replica, values in by_replica.items()}


def q2_unbiased_by_replica(root: Path, system: str) -> dict[str, dict[str, float]]:
    if system in ("C5", "C8"):
        path = root / "results/a4_postreaction_nac_analysis/pair_nac_statistics.csv"
    else:
        path = root / "results/a4_m_c6_c7_pilot_analysis_all_pairs/pair_nac_statistics.csv"
    rows = [
        row for row in read_csv(path)
        if row["state"] == f"A4_m_{system}_2_opposite_cis"
        and row["setting_id"] == PRIMARY
    ]
    by_replica: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        outcome = "R2x2" if row["history_class"] == "reuse_arm_pair" else "R4"
        by_replica[row["replica"]][outcome] += float(row["mean_soft_nac"])
    if len(by_replica) != 3:
        raise ValueError(f"{system}: expected 3 q2 replicas, found {len(by_replica)}")
    return {replica: dict(values) for replica, values in by_replica.items()}


def q1_umbrella(root: Path, system: str) -> dict[str, float] | None:
    if system in ("C5", "C8"):
        # C5/C8 q1 was analyzed with the same one-dimensional contact PMF
        # protocol as the downloaded umbrella pathways.  Convert each class
        # contact ΔG into a contact odds probability, then apply the agreed
        # q1 candidate multiplicities before normalization.  This is a PMF
        # sensitivity estimate, not a replacement for all-pair unbiased NAC.
        path = root / "results/a4_m_c5_c8_q1_three_class_umbrella_analysis/category_contact_free_energies.csv"
        if not path.is_file():
            return None
        rows = read_csv(path)
        raw = {}
        for category in Q1_KEYS:
            row = next(row for row in rows
                       if row["system"] == system and row["category"] == category)
            dg = float(row["contact_dG_kcal_mol"])
            scaled = max(-700.0, min(700.0, dg / RT_KCAL))
            contact_probability = 1.0 / (1.0 + math.exp(scaled))
            raw[category] = Q1_MULTIPLICITY[category] * contact_probability
        return normalize(raw, Q1_KEYS)
    if system not in ("C6", "C7"):
        return None
    rows = read_csv(
        root / "results/a4_m_c6_c7_q1_three_class_umbrella_2d_analysis/category_reweight_summary.csv"
    )
    raw = {
        category: Q1_MULTIPLICITY[category]
        * float(next(
            row["soft_nac_probability"] for row in rows
            if row["system"] == system and row["category"] == category
        ))
        for category in Q1_KEYS
    }
    return normalize(raw, Q1_KEYS)


def q2_umbrella(root: Path, system: str) -> dict[str, float]:
    if system in ("C5", "C8"):
        path = root / "results/a4_q2_umbrella_2d_analysis/final/group_reweight_summary.csv"
        prefix = system
    else:
        path = root / "results/a4_m_c6_c7_q2_umbrella_2d_analysis/group_reweight_summary.csv"
        prefix = f"{system}_q2"
    rows = read_csv(path)
    values = {row["group"]: float(row["soft_nac_probability"]) for row in rows}
    raw = {
        "R2x2": Q2_MULTIPLICITY["R2x2"] * values[f"{prefix}_opposite_reuse"],
        "R4": Q2_MULTIPLICITY["R4"] * values[f"{prefix}_opposite_switch"],
    }
    return normalize(raw, ("R2x2", "R4"))


def propagate(q1: dict[str, float], q2: dict[str, float]) -> dict[str, float]:
    n_r2 = 2.0 * q1["full_reuse"] + 2.0 * q1["full_switch"] * q2["R2x2"]
    n_r4 = q1["partial_reuse"] + q1["full_switch"] * q2["R4"]
    molecular_total = n_r2 + n_r4
    mass_total = n_r2 + 2.0 * n_r4
    return {
        "R2_count_proxy": n_r2,
        "R4_count_proxy": n_r4,
        "R2_molecule_fraction": n_r2 / molecular_total,
        "R4_molecule_fraction": n_r4 / molecular_total,
        "R2_mass_fraction": n_r2 / mass_total,
        "R4_mass_fraction": 2.0 * n_r4 / mass_total,
    }


def experimental_r2(root: Path, system: str) -> float:
    rows = [
        row for row in read_csv(root / "analysis/experimental_targets.csv")
        if row["system"] == f"A4_m_{system}" and int(row["ring_size"]) in (2, 4)
    ]
    values = {int(row["ring_size"]): float(row["ring_wt_percent"]) for row in rows}
    return values[2] / (values[2] + values[4])


def bootstrap_unbiased(
    q1_reps: dict[str, dict[str, float]],
    q2_reps: dict[str, dict[str, float]],
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, float, float]:
    q1_names = tuple(sorted(q1_reps))
    q2_names = tuple(sorted(q2_reps))
    estimates: list[float] = []
    for _ in range(samples):
        q1_draw = [q1_names[index] for index in rng.integers(0, len(q1_names), len(q1_names))]
        q2_draw = [q2_names[index] for index in rng.integers(0, len(q2_names), len(q2_names))]
        q1_raw = {
            key: float(np.mean([q1_reps[rep].get(key, 0.0) for rep in q1_draw]))
            for key in Q1_KEYS
        }
        q2_raw = {
            key: float(np.mean([q2_reps[rep].get(key, 0.0) for rep in q2_draw]))
            for key in ("R2x2", "R4")
        }
        result = propagate(normalize(q1_raw, Q1_KEYS), normalize(q2_raw, ("R2x2", "R4")))
        estimates.append(result["R2_mass_fraction"])
    low, median, high = np.quantile(estimates, (0.025, 0.5, 0.975))
    return float(low), float(median), float(high)


def analyze(root: Path, output: Path, bootstrap: int, seed: int) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    q1_sources: dict[tuple[str, str], dict[str, float]] = {}
    q2_sources: dict[tuple[str, str], dict[str, float]] = {}
    branch_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []

    for system in SYSTEMS:
        q1_reps = q1_unbiased_by_replica(root, system)
        q2_reps = q2_unbiased_by_replica(root, system)
        q1_unbiased = normalize(mean_replica_weights(q1_reps, Q1_KEYS), Q1_KEYS)
        q2_unbiased = normalize(mean_replica_weights(q2_reps, ("R2x2", "R4")), ("R2x2", "R4"))
        q1_sources[(system, "unbiased_all_pairs")] = q1_unbiased
        q2_sources[(system, "unbiased_all_pairs")] = q2_unbiased
        q2_sources[(system, "umbrella_representative")] = q2_umbrella(root, system)
        q1_us = q1_umbrella(root, system)
        if q1_us is not None:
            q1_sources[(system, "umbrella_representative")] = q1_us

        for (source_system, source), probabilities in list(q1_sources.items()):
            if source_system != system:
                continue
            for category, probability in probabilities.items():
                branch_rows.append({
                    "system": system,
                    "stage": "q1",
                    "evidence": source,
                    "category": category,
                    "candidate_multiplicity": Q1_MULTIPLICITY[category],
                    "probability": probability,
                })
        for (source_system, source), probabilities in list(q2_sources.items()):
            if source_system != system:
                continue
            for category, probability in probabilities.items():
                branch_rows.append({
                    "system": system,
                    "stage": "q2_opposite",
                    "evidence": source,
                    "category": category,
                    "candidate_multiplicity": Q2_MULTIPLICITY[category],
                    "probability": probability,
                })

        low, median, high = bootstrap_unbiased(q1_reps, q2_reps, rng, bootstrap)
        bootstrap_rows.append({
            "system": system,
            "bootstrap_samples": bootstrap,
            "R2_mass_fraction_ci95_low": low,
            "R2_mass_fraction_median": median,
            "R2_mass_fraction_ci95_high": high,
            "experimental_R2_mass_fraction_target_normalized": experimental_r2(root, system),
        })

    prediction_rows: list[dict[str, object]] = []
    decomposition_rows: list[dict[str, object]] = []
    for system in SYSTEMS:
        q1_names = [source for (s, source) in q1_sources if s == system]
        q2_names = [source for (s, source) in q2_sources if s == system]
        exp_r2 = experimental_r2(root, system)
        for q1_source in q1_names:
            for q2_source in q2_names:
                result = propagate(q1_sources[(system, q1_source)], q2_sources[(system, q2_source)])
                prediction_rows.append({
                    "system": system,
                    "q1_evidence": q1_source,
                    "q2_opposite_evidence": q2_source,
                    **result,
                    "experimental_R2_mass_fraction_target_normalized": exp_r2,
                    "R2_mass_fraction_error": result["R2_mass_fraction"] - exp_r2,
                })
                decomposition_rows.append({
                    "system": system,
                    "q1_evidence": q1_source,
                    "q2_opposite_evidence": q2_source,
                    "R2_direct_q1_full_reuse": q1_sources[(system, q1_source)]["full_reuse"],
                    "R2_via_q1_full_switch_then_q2_R2x2": (
                        q1_sources[(system, q1_source)]["full_switch"]
                        * q2_sources[(system, q2_source)]["R2x2"]
                    ),
                    "R4_direct_q1_partial_reuse": q1_sources[(system, q1_source)]["partial_reuse"],
                    "R4_via_q1_full_switch_then_q2_R4": (
                        q1_sources[(system, q1_source)]["full_switch"]
                        * q2_sources[(system, q2_source)]["R4"]
                    ),
                    "predicted_R2_mass_fraction": result["R2_mass_fraction"],
                    "predicted_R4_mass_fraction": result["R4_mass_fraction"],
                })

    write_csv(output / "branch_probabilities.csv", branch_rows)
    write_csv(output / "source_combination_predictions.csv", prediction_rows)
    write_csv(output / "yield_pathway_decomposition.csv", decomposition_rows)
    write_csv(output / "unbiased_bootstrap.csv", bootstrap_rows)

    def branch(system: str, stage: str, evidence: str, category: str) -> float:
        return next(
            float(row["probability"]) for row in branch_rows
            if row["system"] == system and row["stage"] == stage
            and row["evidence"] == evidence and row["category"] == category
        )

    def prediction(system: str, q1: str, q2: str) -> dict[str, object]:
        return next(
            row for row in prediction_rows
            if row["system"] == system and row["q1_evidence"] == q1
            and row["q2_opposite_evidence"] == q2
        )

    q1_umbrella_dg_path = root / "results/a4_m_c5_c8_q1_three_class_umbrella_analysis/category_contact_free_energies.csv"
    q1_umbrella_dg_rows = read_csv(q1_umbrella_dg_path) if q1_umbrella_dg_path.is_file() else []
    q1_umbrella_dg = {
        (row["system"], row["category"]): float(row["contact_dG_kcal_mol"])
        for row in q1_umbrella_dg_rows
    }

    report = [
        "# A4-m-C5–C8：R2/R4 产率的统一反应图分析",
        "",
        "> q0 固定为 cross；只在 R2/R4 内比较。R1、R3 不参与目标产物归一化。",
        "> 本报告取代旧的 q1 二分类拼接方式；q1 统一拆成 full reuse、partial reuse、full switch。",
        "",
        "## 1. 唯一采用的产物映射",
        "",
        "- q1 full reuse：生成 2 个 R2。",
        "- q1 partial reuse：进入 adjacent；在 R2/R4 目标模型中生成 1 个 R4。",
        "- q1 full switch：进入 opposite；q2 reuse 生成 2 个 R2，q2 switch 生成 1 个 R4。",
        "",
        "```text",
        "N(R2) = 2 p(full reuse) + 2 p(full switch) p(opposite -> R2x2)",
        "N(R4) = p(partial reuse) + p(full switch) p(opposite -> R4)",
        "w(R2) = N(R2) / [N(R2) + 2 N(R4)]",
        "```",
        "",
        "其中 q1 三类候选数为 1:8:4；q2 opposite 的 R2×2:R4 matching 数为 2:4。",
        "由于 2 个 R2 与 1 个 R4 含有相同的单体总质量，质量分数还可严格化简为：",
        "",
        "```text",
        "w(R2) = p(q1 full reuse) + p(q1 full switch) p(q2 opposite -> R2x2)",
        "w(R4) = p(q1 partial reuse) + p(q1 full switch) p(q2 opposite -> R4)",
        "```",
        "",
        "## 2. 全候选无偏 MD 的分支概率",
        "",
        "| 体系 | q1 full reuse | q1 partial reuse | q1 full switch | q2 opposite→R2×2 | q2 opposite→R4 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for system in SYSTEMS:
        report.append(
            f"| {system} | {branch(system, 'q1', 'unbiased_all_pairs', 'full_reuse'):.6f} | "
            f"{branch(system, 'q1', 'unbiased_all_pairs', 'partial_reuse'):.6f} | "
            f"{branch(system, 'q1', 'unbiased_all_pairs', 'full_switch'):.6f} | "
            f"{branch(system, 'q2_opposite', 'unbiased_all_pairs', 'R2x2'):.6f} | "
            f"{branch(system, 'q2_opposite', 'unbiased_all_pairs', 'R4'):.6f} |"
        )
    report.extend([
        "",
        "C5–C8 的 q1 现在均来自 3×10 ns。所有概率都先在每条重复内对该类全部候选对求和，",
        "再跨重复平均并归一化，因此已经包含候选多重性，不再额外乘 1:8:4。",
        "",
        "## 3. 四个体系按同一公式得到的产率",
        "",
        "这里先以全候选无偏 MD 作为分支人口证据；同时列出 q1/q2 umbrella 组合，作为受限自由能敏感性检查。实验值均在 R2+R4 内重新归一化。",
        "",
        "| 体系 | q2 证据 | 预测 R2 wt | 预测 R4 wt | 实验 R2 wt | 实验 R4 wt |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for system in SYSTEMS:
        for q2_source in ("unbiased_all_pairs", "umbrella_representative"):
            row = prediction(system, "unbiased_all_pairs", q2_source)
            exp_r2 = float(row["experimental_R2_mass_fraction_target_normalized"])
            report.append(
                f"| {system} | {q2_source} | {float(row['R2_mass_fraction']):.3f} | "
                f"{float(row['R4_mass_fraction']):.3f} | {exp_r2:.3f} | {1-exp_r2:.3f} |"
            )
    report.extend([
        "",
        "### q1 umbrella 分支作为输入",
        "",
        "C5/C8 的 q1 umbrella 结果来自每类 2 条独立路径；先把接触 ΔG 转成接触概率，再乘 q1 候选多重性 1:8:4 并归一化。它与无偏 MD 的 all-pair soft-NAC 分支不是同一个统计量，因此只作为独立的自由能敏感性结果。",
        "",
        "| 体系 | q1 full reuse | q1 partial reuse | q1 full switch | q2 证据 | 预测 R2 wt | 预测 R4 wt |",
        "|---|---:|---:|---:|---|---:|---:|",
    ])
    for system in SYSTEMS:
        for q2_source in ("unbiased_all_pairs", "umbrella_representative"):
            row = prediction(system, "umbrella_representative", q2_source)
            report.append(
                f"| {system} | {branch(system, 'q1', 'umbrella_representative', 'full_reuse'):.6f} | "
                f"{branch(system, 'q1', 'umbrella_representative', 'partial_reuse'):.6f} | "
                f"{branch(system, 'q1', 'umbrella_representative', 'full_switch'):.6f} | "
                f"{q2_source} | {float(row['R2_mass_fraction']):.3f} | {float(row['R4_mass_fraction']):.3f} |"
            )
    report.extend([
        "",
        "全候选无偏→无偏的更新系列为 "
        + "、".join(
            f"{system} {float(prediction(system, 'unbiased_all_pairs', 'unbiased_all_pairs')['R2_mass_fraction']):.3f}"
            for system in SYSTEMS
        )
        + "（R2 wt 分数）；实验为 "
        + "、".join(f"{system} {experimental_r2(root, system):.3f}" for system in SYSTEMS)
        + "。",
        "四个体系均已按同一反应图与同一 3×10 ns q1 采样长度处理；剩余偏差必须由分支采样、反应坐标或模型边界解释，不能再归因于 C5/C8 只有 2 ns。",
        "",
        "无偏→无偏结果可进一步拆成以下四条互斥路径：",
        "",
        "| 体系 | R2: q1 full reuse | R2: full switch→q2 R2×2 | R4: q1 partial reuse | R4: full switch→q2 R4 |",
        "|---|---:|---:|---:|---:|",
    ])
    for system in SYSTEMS:
        row = next(
            item for item in decomposition_rows
            if item["system"] == system
            and item["q1_evidence"] == "unbiased_all_pairs"
            and item["q2_opposite_evidence"] == "unbiased_all_pairs"
        )
        report.append(
            f"| {system} | {float(row['R2_direct_q1_full_reuse']):.3f} | "
            f"{float(row['R2_via_q1_full_switch_then_q2_R2x2']):.3f} | "
            f"{float(row['R4_direct_q1_partial_reuse']):.3f} | "
            f"{float(row['R4_via_q1_full_switch_then_q2_R4']):.3f} |"
        )
    report.extend([
        "",
        "C5 的 R2 同时来自 direct full reuse 和 opposite 状态的 reuse matching；C6 也有两条来源。",
        "C7/C8 的 opposite 状态几乎只去 R4，所以它们的 R2 产率近似等于 q1 full-reuse 分支人口。",
        "",
        "按独立重复重采样得到的无偏→无偏 R2 wt 95% 区间为："
        + "、".join(
            f"{row['system']} {float(row['R2_mass_fraction_ci95_low']):.3f}–{float(row['R2_mass_fraction_ci95_high']):.3f}"
            for row in bootstrap_rows
        )
        + "。区间宽度直接反映三条独立重复之间的盆地差异。",
        "",
        "## 4. C5/C8 10 ns 补充实验的含义",
        "",
        f"C5 的 q1 full-reuse 更新人口为 {branch('C5', 'q1', 'unbiased_all_pairs', 'full_reuse'):.3f}；"
        f"C8 为 {branch('C8', 'q1', 'unbiased_all_pairs', 'full_reuse'):.3f}。C8 因而不再是短轨迹给出的近零值，"
        "说明延长采样确实找到了此前缺失的 reuse 构象，但该人口仍不足以单独解释实验 R2。",
        "",
        "C5 的三个重复仍给出显著不同的 full-reuse 权重，因此其点估计不能视为严格收敛；C8 三个重复均采到非负但较小的 full-reuse 权重，方向比旧结果稳定。"
        "由于 R3 已按模型定义排除，这里的差异不能用改变目标产物归一化来修补。",
        "",
        "## 5. C5/C8 q1 umbrella 的结果",
        "",
        "C5/C8 的 12 条 q1 umbrella 路径已经全部完成并下载：每个体系的 full reuse、partial reuse、full switch 各 2 条独立路径；每条路径 25 个距离窗口（0.35–2.75 nm，0.10 nm 间隔），每窗平衡 100 ps、生产 500 ps。WHAM 在生产段跳过前 100 ps，并用 200 条 trajectory-bootstrap 曲线给出不确定度。",
        "",
        "接触 ΔG（接触区 <0.50 nm 相对于支持的非接触区）如下；数值越低表示该类别进入近接触构象越有利：",
        "",
        "| 体系 | full reuse ΔG | partial reuse ΔG | full switch ΔG |",
        "|---|---:|---:|---:|",
    ])
    for system in ("C5", "C8"):
        report.append(
            f"| {system} | {q1_umbrella_dg.get((system, 'full_reuse'), float('nan')):.3f} | "
            f"{q1_umbrella_dg.get((system, 'partial_reuse'), float('nan')):.3f} | "
            f"{q1_umbrella_dg.get((system, 'full_switch'), float('nan')):.3f} |"
        )
    report.extend([
        "",
        "C5 的 PMF 顺序是 full reuse < full switch < partial reuse，因而自由能模型给出 q1 分支约 0.468/0.247/0.285；C8 的顺序是 full switch < partial reuse < full reuse，给出约 0.026/0.506/0.468。这里的分支已经乘入 1:8:4 候选多重性。",
        "",
        "按同一 R2/R4 反应图传播，C5 的 q1-umbrella→q2-umbrella 预测为 R2/R4 = 0.687/0.313，C8 为 0.027/0.973；q2 换用无偏 all-pair 时分别为 C5 0.668/0.332、C8 0.026/0.974。",
        "这些结果说明：C5 的 q1 自由能证据比 3×10 ns 无偏人口更支持 R2，但 C8 仍强烈支持 R4；umbrella 能解释路径接近接触的热力学倾向，却不能替代无偏轨迹对真实动力学分支人口的估计。",
        "",
        "因此，C5/C8 的 q1 umbrella 补充实验已经闭环；后续比较应同时保留 `unbiased_all_pairs` 与 `umbrella_representative` 两条证据链，不把二者静默合并。",
        "",
        "## 6. 文件",
        "",
        "- `branch_probabilities.csv`：四体系、两层反应图的统一分支概率。",
        "- `source_combination_predictions.csv`：所有可用无偏/umbrella 组合的产率传播。",
        "- `yield_pathway_decomposition.csv`：R2、R4 的直接与 q2 间接路径贡献。",
        "- `unbiased_bootstrap.csv`：按独立重复重采样的无偏→无偏 R2 wt 区间。",
        "- `../a4_m_c5_c8_q1_three_class_umbrella_analysis/`：12 条 q1 umbrella 的 WHAM、路径/类别接触自由能和三类对比。",
    ])
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "systems": list(SYSTEMS),
        "primary_nac_setting": PRIMARY,
        "q0": "cross fixed",
        "target_products": ["R2", "R4"],
        "q1_candidate_multiplicity": Q1_MULTIPLICITY,
        "q2_opposite_candidate_multiplicity": Q2_MULTIPLICITY,
        "bootstrap_samples": bootstrap,
        "files": [
            "branch_probabilities.csv",
            "source_combination_predictions.csv",
            "yield_pathway_decomposition.csv",
            "unbiased_bootstrap.csv",
            "REPORT.md",
        ],
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "predictions": prediction_rows}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/a4_m_c5_c8_r2_r4_complete",
    )
    parser.add_argument("--bootstrap", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    analyze(PROJECT, args.output.resolve(), args.bootstrap, args.seed)


if __name__ == "__main__":
    main()
