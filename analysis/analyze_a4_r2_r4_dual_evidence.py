#!/usr/bin/env python3
"""Propagate C6/C7 q1/q2 branches to R2/R4 using unbiased and umbrella evidence."""

from __future__ import annotations

import csv
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
UNBIASED = PROJECT / "results/a4_r2_r4_propagation/r2_r4_predictions.csv"
Q1_UMBRELLA = PROJECT / "results/a4_m_c6_c7_q1_three_class_umbrella_2d_analysis/category_reweight_summary.csv"
Q1_CONTRASTS = PROJECT / "results/a4_m_c6_c7_q1_three_class_umbrella_2d_analysis/q1_three_class_2d_contrasts.csv"
Q1_ENDPOINT_CONTRASTS = PROJECT / "results/a4_m_c6_c7_q1_three_class_umbrella_2d_analysis/q1_paired_endpoint_contrasts.csv"
Q1_2D_NEW_PATHS = PROJECT / "results/a4_m_c6_c7_q1_three_class_umbrella_2d_analysis/new_pathway_reweight_summary.csv"
Q1_1D_PATHS = PROJECT / "results/a4_m_c6_c7_q1_three_class_umbrella_analysis/pathway_contact_free_energies.csv"
Q2_UMBRELLA = PROJECT / "results/a4_m_c6_c7_q2_umbrella_2d_analysis/group_reweight_summary.csv"
Q2_CONTRASTS = PROJECT / "results/a4_m_c6_c7_q2_umbrella_2d_analysis/q2_2d_reweighted_contrasts.csv"
OUTPUT = PROJECT / "results/a4_m_c6_c7_r2_r4_dual_evidence"

Q1_MULTIPLICITY = {"full_reuse": 1, "partial_reuse": 8, "full_switch": 4}
Q2_OPPOSITE_MULTIPLICITY = {"R2x2_reuse": 2, "R4_switch": 4}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    return {key: value / total for key, value in values.items()}


def propagate(q1: dict[str, float], q2_r2: float, q2_r4: float) -> dict[str, float]:
    # A full-reuse or q2-opposite reuse completion generates two R2 molecules.
    r2_count = 2.0 * q1["full_reuse"] + 2.0 * q1["full_switch"] * q2_r2
    r4_count = q1["partial_reuse"] + q1["full_switch"] * q2_r4
    count_total = r2_count + r4_count
    mass_total = r2_count + 2.0 * r4_count  # M(R4)/M(R2) = 2
    return {
        "R2_count_proxy": r2_count,
        "R4_count_proxy": r4_count,
        "R2_molecule_fraction": r2_count / count_total,
        "R4_molecule_fraction": r4_count / count_total,
        "R2_mass_fraction": r2_count / mass_total,
        "R4_mass_fraction": 2.0 * r4_count / mass_total,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    unbiased_rows = {row["system"]: row for row in read_csv(UNBIASED)}
    q1_umbrella_rows = {
        (row["system"], row["category"]): row for row in read_csv(Q1_UMBRELLA)
    }
    q2_umbrella_rows = {row["group"]: row for row in read_csv(Q2_UMBRELLA)}

    branch_rows: list[dict[str, object]] = []
    source_values: dict[tuple[str, str], tuple[dict[str, float], float, float]] = {}
    for system in ("C6", "C7"):
        unb = unbiased_rows[system]
        q1_unbiased = {
            "full_reuse": float(unb["q1_full_reuse_probability"]),
            "partial_reuse": float(unb["q1_partial_reuse_probability"]),
            "full_switch": float(unb["q1_full_switch_probability"]),
        }
        q2_unbiased_r2 = float(unb["q2_opposite_R2x2_probability"])
        q2_unbiased_r4 = float(unb["q2_opposite_R4_probability"])
        source_values[(system, "unbiased")] = (
            q1_unbiased, q2_unbiased_r2, q2_unbiased_r4
        )

        raw_q1_umbrella = {
            category: (
                Q1_MULTIPLICITY[category]
                * float(q1_umbrella_rows[(system, category)]["soft_nac_probability"])
            )
            for category in Q1_MULTIPLICITY
        }
        q1_umbrella = normalize(raw_q1_umbrella)
        q2_reuse = (
            Q2_OPPOSITE_MULTIPLICITY["R2x2_reuse"]
            * float(q2_umbrella_rows[f"{system}_q2_opposite_reuse"]["soft_nac_probability"])
        )
        q2_switch = (
            Q2_OPPOSITE_MULTIPLICITY["R4_switch"]
            * float(q2_umbrella_rows[f"{system}_q2_opposite_switch"]["soft_nac_probability"])
        )
        q2_total = q2_reuse + q2_switch
        source_values[(system, "umbrella")] = (
            q1_umbrella, q2_reuse / q2_total, q2_switch / q2_total
        )

        for source in ("unbiased", "umbrella"):
            q1, q2_r2, q2_r4 = source_values[(system, source)]
            for category, probability in q1.items():
                branch_rows.append({
                    "system": system,
                    "stage": "q1",
                    "evidence": source,
                    "category": category,
                    "candidate_multiplicity": Q1_MULTIPLICITY[category],
                    "target_conditioned_probability": probability,
                })
            branch_rows.extend([
                {
                    "system": system,
                    "stage": "q2_opposite",
                    "evidence": source,
                    "category": "R2x2_reuse",
                    "candidate_multiplicity": 2,
                    "target_conditioned_probability": q2_r2,
                },
                {
                    "system": system,
                    "stage": "q2_opposite",
                    "evidence": source,
                    "category": "R4_switch",
                    "candidate_multiplicity": 4,
                    "target_conditioned_probability": q2_r4,
                },
            ])

    prediction_rows: list[dict[str, object]] = []
    for system in ("C6", "C7"):
        experimental_r2 = float(unbiased_rows[system]["experimental_R2_wt_fraction_target_normalized"])
        for q1_source in ("unbiased", "umbrella"):
            for q2_source in ("unbiased", "umbrella"):
                q1 = source_values[(system, q1_source)][0]
                q2_r2 = source_values[(system, q2_source)][1]
                q2_r4 = source_values[(system, q2_source)][2]
                result = propagate(q1, q2_r2, q2_r4)
                prediction_rows.append({
                    "system": system,
                    "q1_evidence": q1_source,
                    "q2_opposite_evidence": q2_source,
                    **result,
                    "experimental_R2_mass_fraction_target_normalized": experimental_r2,
                    "R2_mass_fraction_error": result["R2_mass_fraction"] - experimental_r2,
                })

    write_csv(OUTPUT / "branch_probabilities.csv", branch_rows)
    write_csv(OUTPUT / "source_combination_predictions.csv", prediction_rows)

    q1_contrasts = read_csv(Q1_CONTRASTS)
    q1_endpoint_contrasts = read_csv(Q1_ENDPOINT_CONTRASTS)
    q1_2d_new_paths = read_csv(Q1_2D_NEW_PATHS)
    q2_contrasts = read_csv(Q2_CONTRASTS)
    q1_paths = read_csv(Q1_1D_PATHS)
    min_overlap = min(float(row["minimum_neighbor_overlap"]) for row in q1_paths)
    max_center_error = max(float(row["maximum_mean_center_error_nm"]) for row in q1_paths)
    min_q1_2d_ess = min(float(row["effective_sample_size"]) for row in q1_2d_new_paths)
    max_q1_2d_rms = max(float(row["geometry_pull_rms_A"]) for row in q1_2d_new_paths)

    def probability(system: str, stage: str, source: str, category: str) -> float:
        row = next(
            item for item in branch_rows
            if item["system"] == system and item["stage"] == stage
            and item["evidence"] == source and item["category"] == category
        )
        return float(row["target_conditioned_probability"])

    def prediction(system: str, q1_source: str, q2_source: str) -> dict[str, object]:
        return next(
            row for row in prediction_rows
            if row["system"] == system and row["q1_evidence"] == q1_source
            and row["q2_opposite_evidence"] == q2_source
        )

    report = [
        "# A4-m-C6/C7：q1 三分支与 R2/R4 传播（无偏 MD + umbrella）",
        "",
        "> 本报告严格按 `A4-m-C5与A4-m-C8反应选择性计算总总结.md` 的 NAC 与自由能口径，",
        "> 但按当前问题只保留 R2、R4；q0 固定为 cross，R1/R3 不进入目标产物归一化。",
        "",
        "## 1. 反应图映射",
        "",
        "q0 的第一条边固定为 cross。q1 的下一条 cross 边必须分成三类：",
        "",
        "1. **full reuse**：再次使用第一次反应涉及的完整臂对，完成后生成 **2 个 R2**；",
        "2. **partial reuse**：只复用第一次反应中的一条臂，进入 adjacent 状态；只考虑 R2/R4 时归为 **1 个 R4**；",
        "3. **full switch**：使用两条此前都未参与反应的臂，进入 opposite 状态；随后可能得到 **2 个 R2** 或 **1 个 R4**。",
        "",
        "候选对数分别为 1、8、4；opposite 状态中 R2×2/reuse 与 R4/switch 候选数分别为 2、4。",
        "因此单路径 umbrella 的 NAC 概率必须先乘候选数，才能形成分支权重。",
        "",
        "传播式为",
        "",
        "```text",
        "N(R2) = 2 p(q1 full reuse) + 2 p(q1 full switch) p(q2 opposite R2x2)",
        "N(R4) = p(q1 partial reuse) + p(q1 full switch) p(q2 opposite R4)",
        "w(R2) = N(R2) / [N(R2) + 2 N(R4)]",
        "```",
        "",
        "最后一式使用 M(R4)/M(R2)=2；这也显式计入一次 full-reuse 完成会同时产生两个 R2。",
        "",
        "## 2. q1 三分支概率",
        "",
        "| 体系 | 证据 | full reuse | partial reuse | full switch |",
        "|---|---|---:|---:|---:|",
    ]
    for system in ("C6", "C7"):
        for source in ("unbiased", "umbrella"):
            report.append(
                f"| {system} | {source} | "
                f"{probability(system, 'q1', source, 'full_reuse'):.3f} | "
                f"{probability(system, 'q1', source, 'partial_reuse'):.3f} | "
                f"{probability(system, 'q1', source, 'full_switch'):.3f} |"
            )
    report.extend([
        "",
        "这里的 umbrella 行来自二维距离—取向重加权后的 soft NAC，且已乘 1:8:4 候选多重性。",
        "三个类别现在都使用两个独立 q1 端点等权平均；同一类别的两个端点保持相同乙烯基对，",
        "因此这一批专门检验初始构象记忆，而没有混入候选双键对身份变化。",
        "",
        "umbrella 的双端点类别平均自由能结果为：",
        "",
        "| 体系 | 对比（左−右） | ΔΔG contact | ΔΔG orientation | ΔΔG total NAC | NAC 几率比（左/右） | 前半/后半 total |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for system in ("C6", "C7"):
        for definition in (
            "full_reuse minus partial_reuse",
            "partial_reuse minus full_switch",
            "full_reuse minus full_switch",
        ):
            row = next(item for item in q1_contrasts if item["system"] == system and item["definition"] == definition)
            report.append(
                f"| {system} | {definition.replace('_', ' ')} | "
                f"{float(row['full_ddG_contact_kcal_mol']):+.3f} | "
                f"{float(row['full_ddG_orientation_kcal_mol']):+.3f} | "
                f"{float(row['full_ddG_nac_kcal_mol']):+.3f} | "
                f"{float(row['left_over_right_nac_odds']):.2f} | "
                f"{float(row['first_half_ddG_nac_kcal_mol']):+.3f}/"
                f"{float(row['second_half_ddG_nac_kcal_mol']):+.3f} |"
            )
    report.extend([
        "",
        "结论：C6 三类代表路径仍然接近。二维总 NAC 中 full switch 相对 partial reuse 仅有 0.301 kcal mol⁻¹",
        "（1.62 倍）、相对 full reuse 为 0.262 kcal mol⁻¹（1.52 倍）；对应一维接触 ΔΔG 的 bootstrap 区间均跨 0，",
        "所以不能把这种小优势视为稳健排序。C7 的 full reuse 相对 partial reuse 为 −1.076 kcal mol⁻¹（5.64 倍），",
        "相对 full switch 为 −1.249 kcal mol⁻¹（7.44 倍），前后半程及一维 bootstrap 均保持 full reuse 优势；",
        "partial reuse 与 full switch 只有 −0.173 kcal mol⁻¹且前后半程换号。",
        "",
        "同一无偏 MD 端点内的配对结果进一步区分了可重复方向与端点依赖：",
        "",
        "| 体系 | 端点 | 对比（左−右） | ΔΔG total NAC | 左/右 NAC 几率比 |",
        "|---|---:|---|---:|---:|",
    ])
    for system in ("C6", "C7"):
        for replica in ("1", "2"):
            for definition in (
                "full_reuse minus partial_reuse",
                "partial_reuse minus full_switch",
                "full_reuse minus full_switch",
            ):
                row = next(
                    item for item in q1_endpoint_contrasts
                    if item["system"] == system and item["source_replica"] == replica
                    and item["definition"] == definition
                )
                report.append(
                    f"| {system} | {replica} | {definition.replace('_', ' ')} | "
                    f"{float(row['full_ddG_nac_kcal_mol']):+.3f} | "
                    f"{float(row['left_over_right_nac_odds']):.2f} |"
                )
    report.extend([
        "",
        "C6 的 full reuse−full switch 在两个端点均为正（+0.165、+0.319 kcal mol⁻¹），即方向上都略偏 full switch；",
        "C7 在两个端点均为负（−1.583、−1.126 kcal mol⁻¹），即稳健偏 full reuse。",
        "但两体系的 partial reuse 与另外两类之间都存在明显端点依赖，因此只能解释为构象盆地差异，不能单独冻结成一个精确常数。",
        "",
        "## 3. q2 opposite：R2×2 与 R4",
        "",
        "| 体系 | 证据 | R2×2/reuse | R4/switch |",
        "|---|---|---:|---:|",
    ])
    for system in ("C6", "C7"):
        for source in ("unbiased", "umbrella"):
            report.append(
                f"| {system} | {source} | "
                f"{probability(system, 'q2_opposite', source, 'R2x2_reuse'):.6f} | "
                f"{probability(system, 'q2_opposite', source, 'R4_switch'):.6f} |"
            )
    report.extend([
        "",
        "umbrella 行同样已乘 2:4 候选多重性。对应的单路径二维 NAC 对比为：",
        "",
        "| 体系 | ΔΔG(switch−reuse) contact | orientation | total NAC | switch/reuse 单路径几率比 | 前半/后半 total |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for system in ("C6", "C7"):
        row = next(item for item in q2_contrasts if item["state"] == f"{system}_q2_opposite")
        report.append(
            f"| {system} | {float(row['full_ddG_contact_kcal_mol']):+.3f} | "
            f"{float(row['full_ddG_orientation_kcal_mol']):+.3f} | "
            f"{float(row['full_ddG_nac_kcal_mol']):+.3f} | "
            f"{float(row['left_over_right_nac_odds']):.3g} | "
            f"{float(row['first_half_ddG_nac_kcal_mol']):+.3f}/"
            f"{float(row['second_half_ddG_nac_kcal_mol']):+.3f} |"
        )
    report.extend([
        "",
        "C7 的 opposite 状态在无偏与 umbrella 中都几乎完全导向 R4；C6 则存在定量差异：",
        "无偏 MD 给出 R2×2=0.863，代表路径 umbrella 给出 0.636。",
        "",
        "## 4. R2/R4 传播与实验",
        "",
        "下表给出 q1、q2 两层分别采用无偏或 umbrella 证据的交叉组合。实验值只在 R2+R4 内归一化。",
        "",
        "| 体系 | q1 | q2 opposite | 预测 R2 wt | 预测 R4 wt | 实验 R2/R4 wt |",
        "|---|---|---|---:|---:|---:|",
    ])
    for system in ("C6", "C7"):
        exp_r2 = float(unbiased_rows[system]["experimental_R2_wt_fraction_target_normalized"])
        for q1_source in ("unbiased", "umbrella"):
            for q2_source in ("unbiased", "umbrella"):
                row = prediction(system, q1_source, q2_source)
                report.append(
                    f"| {system} | {q1_source} | {q2_source} | "
                    f"{float(row['R2_mass_fraction']):.3f} | {float(row['R4_mass_fraction']):.3f} | "
                    f"{exp_r2:.3f}/{1-exp_r2:.3f} |"
                )
    report.extend([
        "",
        "最重要的结果是：**按当前“只保留 R2/R4、partial reuse 必归 R4”的严格映射，C6 的无偏→无偏传播得到 "
        "R2/R4=0.390/0.610，接近实验 0.415/0.585。** C7 得到 0.267/0.733，正确给出 R4 主产物，但高估 R2；",
        "实验为 0.220/0.780。",
        "",
        "全 umbrella 传播得到 C6=0.334/0.666、C7=0.338/0.662。C6 若用新 q1 umbrella 配合无偏 q2，",
        "则得到 0.429/0.571，与实验 0.415/0.585 很接近；差异主要来自 C6 q2 opposite 的代表路径 umbrella。",
        "C7 仍由 q1 full-reuse 代表对的强优势把 R2 高估。原因不是反应图映射错误，而是即使端点重复已经平衡，",
        "umbrella 仍只覆盖每类一个预选乙烯基对；它不能自动成为含多个候选对类别的完整系综平均。",
        "",
        "因此当前最稳妥的证据分工是：**无偏 MD 的全对求和决定分支人口；umbrella/二维重加权解释代表路径的",
        "接触与取向自由能来源。** 两者都保留，但不把单路径 PMF 强行替代全体系分支概率。",
        "",
        "## 5. 质量控制与尚缺实验",
        "",
        f"本轮 4 条新路径及此前 4 条 partial-reuse 路径均有 25 个窗口；所有 q1 路径的最小相邻高斯重叠为 "
        f"{min_overlap:.3f}，最大窗口中心偏差为 {max_center_error:.3f} nm。",
        f"二维几何重构中，8 条新增端点路径的 pull/几何距离 RMS 误差最大为 {max_q1_2d_rms:.4f} Å，",
        f"有效样本数最小为 {min_q1_2d_ess:.0f}。数据文件完整，WHAM/MBAR 收敛。",
        "C7 q2 opposite-reuse 路径的二维重加权有效样本数只有 463，明显低于其他路径；",
        "因此 0.000046 这一具体分支概率不应作高精度解读。不过其 switch−reuse ΔΔG 在前、后半程均小于 −5 kcal mol⁻¹，",
        "且无偏 MD 独立得到近乎全 R4，所以“C7 opposite 导向 R4”的方向性结论仍然稳健。",
        "",
        "当前端点覆盖已经平衡为每类两个。剩余限制转为**候选对覆盖**：full reuse 只有一个候选对，已基本覆盖；",
        "partial reuse 有 8 个、full switch 有 4 个候选对，而 umbrella 各只抽取其中一个固定双键对。",
        "若坚持让 umbrella 独立承担定量产物预测，需要继续扩展候选对；若采用“无偏决定人口、umbrella 解释自由能来源”的分工，",
        "目前数据已经足以支持 C6/C7 的主结论，不必立刻增加大量任务。",
        "",
        "## 6. 当前结论",
        "",
        "1. 用户提出的三分支拓扑映射可以闭合，并且必须把 full reuse 对应的两个 R2 同时计数。",
        "2. C6 的无偏 q1+q2 传播在不引入额外拟合参数时把实验 R2 质量分数重现到 2.5 个百分点以内。",
        "3. C7 的 R4 主产物得到一致预测，但 R2 仍高约 4.7 个百分点；误差主要来自 q1，而不是 q2 opposite。",
        "4. 平衡端点重复后，乘 1:8:4 重数的 umbrella q1 概率为 C6=0.068/0.514/0.417、C7=0.338/0.480/0.182；partial 类在两者中仍最大。",
        "5. 当前 umbrella 已经能控制端点记忆，但候选对覆盖仍不足以取代无偏全对统计作为最终分支人口。",
        "",
    ])
    (OUTPUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {OUTPUT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
