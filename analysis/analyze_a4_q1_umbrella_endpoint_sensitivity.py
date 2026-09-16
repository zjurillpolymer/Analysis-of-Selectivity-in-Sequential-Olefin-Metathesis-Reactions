#!/usr/bin/env python3
"""Audit three-endpoint q1 umbrellas and propagate paired endpoint uncertainty.

Uses the original A4 summary's soft-NAC definition, per-pair averaging and
10,000 paired-replica bootstrap.  q2 is held fixed in the bootstrap: the
reported interval is an endpoint-sensitivity interval, not total yield error.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from analysis.analyze_a4_m_c5_c8_q1_umbrella_2d import JOBS, MULTIPLICITY
from analysis.analyze_a4_q2_umbrella_2d_reweight import (
    RT_KCAL, mbar_weights, orientation_geometry_matching_pull,
    read_csv, read_numeric_xvg, read_pullx, soft_lower, soft_upper, write_csv,
)

CLASSES = tuple(MULTIPLICITY)
COMPARISONS = ((0, 1), (1, 2), (0, 2))
EXPERIMENT = {"C5": (75, 24), "C6": (39, 55), "C7": (20, 71), "C8": (13, 63)}


def branch(nac: np.ndarray) -> np.ndarray:
    raw = nac * np.array([MULTIPLICITY[c] for c in CLASSES])
    return raw / raw.sum(axis=-1, keepdims=True)


def yield_r2(q1: np.ndarray, q2: float) -> np.ndarray:
    return q1[..., 0] + q1[..., 2] * q2


def q2_probabilities(system: str, half: str = "") -> float:
    if system in ("C5", "C8"):
        path = PROJECT / "results/a4_q2_umbrella_2d_analysis/final/group_reweight_summary.csv"
        prefix = f"{system}_opposite_"
    else:
        path = PROJECT / "results/a4_m_c6_c7_q2_umbrella_2d_analysis/group_reweight_summary.csv"
        prefix = f"{system}_q2_opposite_"
    rows = {r["group"]: r for r in read_csv(path)}
    field = f"{half}soft_nac_probability"
    reuse = float(rows[prefix + "reuse"][field])
    switch = float(rows[prefix + "switch"][field])
    return 2 * reuse / (2 * reuse + 4 * switch)


def quantiles(values: np.ndarray) -> tuple[float, float]:
    return tuple(float(x) for x in np.quantile(values, (0.025, 0.975)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path,
                        default=PROJECT / "results/a4_m_c5_c8_q1_umbrella_2d_3rep_analysis")
    parser.add_argument("--supplement-root", type=Path,
                        default=PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_rep3_download")
    args = parser.parse_args()
    output = args.analysis_root.resolve()
    rows = read_csv(output / "pathway_reweight_summary.csv")
    by_key = {(r["system"].replace("A4_m_", ""), int(r["replica"]), r["route_family"]): r for r in rows}
    if len(by_key) != 18:
        raise ValueError(f"expected 18 unique endpoint/class paths, got {len(by_key)}")

    pathways = {alias: PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_download" / job / "extracted/pathway"
                for alias, (job, _) in JOBS.items()}
    for meta in args.supplement_root.glob("*/extracted/pathway/pathway.json"):
        pathways[json.loads(meta.read_text())["pathway"]] = meta.parent

    qa_rows, threshold_by_key, domain_nac = [], {}, {}
    for key, row in by_key.items():
        name = row["pathway"]
        pathway = pathways[name]
        meta = json.loads((pathway / "pathway.json").read_text())
        distances, axes, sides, window_ids, centers, overlaps, center_errors = [], [], [], [], [], [], []
        histograms = []
        for idx, window in enumerate(sorted((pathway / "windows").glob("window_*"))):
            for filename in ("production.gro", "production.cpt", "production.tpr", "production.xtc", "production_pullx.xvg"):
                if not (window / filename).is_file() or (window / filename).stat().st_size == 0:
                    raise FileNotFoundError(window / filename)
            full = read_numeric_xvg(window / "production_pullx.xvg")
            if not math.isclose(float(full[-1, 0]), 500.0, abs_tol=1e-5):
                raise ValueError(f"{window}: incomplete production time")
            pull = read_pullx(window / "production_pullx.xvg", 100.0, 500.0)
            cache = output / "cache" / name / window.name
            coords = read_numeric_xvg(cache / "reactive_pair_coordinates_nm.xvg")
            boxes = read_numeric_xvg(cache / "box_vectors_nm.xvg")
            if len(coords) != len(pull) or not np.allclose(coords[:, 0], pull[:, 0]):
                raise ValueError(f"{window}: cached time mismatch")
            _, axis, side, physical = orientation_geometry_matching_pull(coords, boxes, pull[:, 1])
            if not np.all(physical):
                raise ValueError(f"{window}: invalid vinyl bond length")
            center = float(window.name.rsplit("_", 1)[-1].removesuffix("nm"))
            centers.append(center)
            distances.append(pull[:, 1]); axes.append(axis); sides.append(side)
            window_ids.append(np.full(len(pull), idx, dtype=int))
            hist = np.histogram(pull[:, 1], np.arange(0.0, 4.02, 0.02))[0].astype(float)
            histograms.append(hist / hist.sum())
            center_errors.append(abs(float(pull[:, 1].mean()) - center))
        if len(centers) != 25:
            raise ValueError(f"{name}: expected 25 windows")
        overlaps = [float(np.minimum(a, b).sum()) for a, b in zip(histograms[:-1], histograms[1:])]
        distance = np.concatenate(distances); axis = np.concatenate(axes); side = np.concatenate(sides)
        weights, _, _, _ = mbar_weights(distance, np.asarray(centers), np.concatenate(window_ids))
        primary_nac = soft_upper(distance, 0.50, 0.05) * soft_upper(axis, 45, 7.5) * soft_lower(side, 45, 7.5)
        nac_weights = weights * primary_nac
        nac_weights /= nac_weights.sum()
        in_domain = (distance >= 0.32) & (distance <= 2.78)
        domain_nac[key] = float(np.sum(weights[in_domain] * primary_nac[in_domain]) / weights[in_domain].sum())
        def mass90_frames(values: np.ndarray) -> int:
            return int(np.searchsorted(np.cumsum(np.sort(values)[::-1]), 0.90) + 1)
        qa_rows.append({
            "pathway": name, "system": key[0], "replica": key[1], "category": key[2],
            "windows": len(centers), "frames": len(distance), "production_end_ps": 500.0,
            "minimum_neighbor_histogram_overlap_0_02nm_bins": min(overlaps),
            "maximum_mean_center_error_nm": max(center_errors),
            "geometry_pull_rms_A": float(row["geometry_pull_rms_A"]),
            "geometry_pull_max_A": float(row["geometry_pull_max_A"]),
            "reweight_ESS_not_time_decorrelated": float(row["effective_sample_size"]),
            "NAC_contribution_ESS_not_time_decorrelated": float(1 / np.sum(nac_weights ** 2)),
            "frames_containing_90pct_unbiased_weight": mass90_frames(weights),
            "frames_containing_90pct_NAC_weight": mass90_frames(nac_weights),
            "maximum_observed_pull_distance_nm": float(distance.max()),
            "weight_outside_common_WHAM_domain_0_32_2_78nm": float(weights[~in_domain].sum()),
            "NAC_probability_conditioned_on_common_WHAM_domain": domain_nac[key],
            "source_topology_sha256": hashlib.sha256((pathway / "system.top").read_bytes()).hexdigest(),
            "bias_dimensions": 1, "analysis_dimensions": 2,
        })
        thresholds = {}
        for d, a, s in itertools.product((0.45, 0.50, 0.55), (30, 45, 60), (30, 45, 60)):
            nac = soft_upper(distance, d, 0.05) * soft_upper(axis, a, 7.5) * soft_lower(side, s, 7.5)
            thresholds[(d, a, s)] = float(np.sum(weights * nac))
        threshold_by_key[key] = thresholds
        print(f"audited={name}", flush=True)

    replica_rows, contrast_rows, uncertainty_rows, sensitivity_rows, chain_rows = [], [], [], [], []
    rng = np.random.default_rng(20260913)
    for system in ("C5", "C8"):
        q2 = q2_probabilities(system)
        matrix = np.array([[float(by_key[(system, r, c)]["soft_nac_probability"]) for c in CLASSES] for r in (1, 2, 3)])
        contact = np.array([[float(by_key[(system, r, c)]["soft_contact_probability"]) for c in CLASSES] for r in (1, 2, 3)])
        indices = rng.integers(0, 3, (10000, 3))
        nac_draws = matrix[indices].mean(axis=1); contact_draws = contact[indices].mean(axis=1)
        q1 = branch(matrix.mean(axis=0)); predictions = yield_r2(branch(nac_draws), q2)
        low, high = quantiles(predictions)
        uncertainty_rows.append({"system": system, "R2_mass_fraction": float(yield_r2(q1, q2)),
                                 "q1_endpoint_bootstrap_ci_low": low, "q1_endpoint_bootstrap_ci_high": high,
                                 "replicas": 3, "bootstrap_draws": 10000, "q2_held_fixed": q2})
        for r in (1, 2, 3):
            p = branch(matrix[r - 1])
            replica_rows.append({"system": system, "replica": r,
                                 **dict(zip(CLASSES, p)), "q2_R2x2_fixed": q2,
                                 "R2_mass_fraction": float(yield_r2(p, q2))})
        for a, b in COMPARISONS:
            dc = -RT_KCAL * math.log(contact[:, a].mean() / contact[:, b].mean())
            dn = -RT_KCAL * math.log(matrix[:, a].mean() / matrix[:, b].mean())
            dd = -RT_KCAL * np.log(nac_draws[:, a] / nac_draws[:, b])
            lo, hi = quantiles(dd)
            contrast_rows.append({"system": system, "definition": f"{CLASSES[a]} minus {CLASSES[b]}",
                                  "ddG_contact_kcal_mol": dc, "ddG_orientation_kcal_mol": dn - dc,
                                  "ddG_NAC_kcal_mol": dn, "alpha_NAC": dn / RT_KCAL,
                                  "q1_endpoint_bootstrap_ci_low": lo, "q1_endpoint_bootstrap_ci_high": hi,
                                  "left_over_right_per_pair_NAC_ratio": math.exp(-dn / RT_KCAL)})
        for omitted in (1, 2, 3):
            p = branch(np.delete(matrix, omitted - 1, axis=0).mean(axis=0))
            sensitivity_rows.append({"system": system, "setting": f"leave_out_endpoint_{omitted}",
                                     **dict(zip(CLASSES, p)), "q2_R2x2": q2,
                                     "R2_mass_fraction": float(yield_r2(p, q2))})
        for half in ("first_half_", "second_half_"):
            half_nac = np.array([[float(by_key[(system, r, c)][half + "soft_nac_probability"])
                                  for c in CLASSES] for r in (1, 2, 3)]).mean(axis=0)
            p = branch(half_nac)
            for fixed in (True, False):
                q2_value = q2 if fixed else q2_probabilities(system, half)
                sensitivity_rows.append({"system": system, "setting": half + ("q2_fixed" if fixed else "q2_same_half"),
                                         **dict(zip(CLASSES, p)), "q2_R2x2": q2_value,
                                         "R2_mass_fraction": float(yield_r2(p, q2_value))})
        chain_rows.extend([{"system": system, "outcome": outcome, "mass_fraction_contribution": float(v)}
                           for outcome, v in zip(("R2_direct_full_reuse", "R2_full_switch_q2_reuse",
                                                  "R4_partial_reuse", "R4_full_switch_q2_switch"),
                                                 (q1[0], q1[2] * q2, q1[1], q1[2] * (1 - q2)))])

    threshold_rows = []
    for system in ("C5", "C8"):
        for d, a, s in itertools.product((0.45, 0.50, 0.55), (30, 45, 60), (30, 45, 60)):
            means = np.array([np.mean([threshold_by_key[(system, r, c)][(d, a, s)] for r in (1, 2, 3)]) for c in CLASSES])
            p = branch(means)
            threshold_rows.append({"system": system, "distance_nm": d, "axis_max_deg": a, "side_min_deg": s,
                                   "full_reuse_minus_full_switch_ddG_kcal_mol": -RT_KCAL * math.log(means[0] / means[2]),
                                   **dict(zip(CLASSES, p)), "q2_primary_held_fixed": True,
                                   "R2_mass_fraction_q2_primary_fixed": float(yield_r2(p, q2_probabilities(system)))})

    domain_rows = []
    for system in ("C5", "C8"):
        p = branch(np.array([np.mean([domain_nac[(system, r, c)] for r in (1, 2, 3)]) for c in CLASSES]))
        domain_rows.append({"system": system, "definition": "NAC conditional on each path's distance in 0.32-2.78 nm; q2 primary fixed",
                            **dict(zip(CLASSES, p)), "range_conditioned_R2_mass_fraction": float(yield_r2(p, q2_probabilities(system)))})
    series = []
    baseline = {r["system"]: r for r in read_csv(PROJECT / "results/a4_m_c5_c8_q1_umbrella_2d_analysis/q1_2d_yield_prediction.csv")}
    for system in ("C5", "C6", "C7", "C8"):
        if system in ("C5", "C8"):
            info = next(r for r in uncertainty_rows if r["system"] == system)
            prediction = info["R2_mass_fraction"]
            endpoints = 3
        else:
            groups = read_csv(PROJECT / "results/a4_m_c6_c7_q1_three_class_umbrella_2d_analysis/category_reweight_summary.csv")
            means = np.array([float(next(r for r in groups if r["system"] == system and r["category"] == c)["soft_nac_probability"]) for c in CLASSES])
            prediction = float(yield_r2(branch(means), q2_probabilities(system)))
            endpoints = 2
        experiment = EXPERIMENT[system][0] / sum(EXPERIMENT[system])
        series.append({"system": system, "q1_endpoints_per_class": endpoints,
                       "predicted_R2_mass_fraction": prediction, "predicted_R4_mass_fraction": 1 - prediction,
                       "experimental_R2_target_normalized": experiment,
                       "experimental_R4_target_normalized": 1 - experiment,
                       "R2_error_percentage_points": 100 * (prediction - experiment),
                       "previous_q1_2endpoint_2d_R2": float(baseline[system]["R2_mass_fraction"]) if system in baseline else ""})

    for filename, values in (("quality_audit.csv", qa_rows), ("replica_branch_predictions.csv", replica_rows),
                             ("endpoint_bootstrap_yield.csv", uncertainty_rows), ("endpoint_bootstrap_contrasts.csv", contrast_rows),
                             ("endpoint_half_sensitivity.csv", sensitivity_rows), ("yield_pathway_decomposition.csv", chain_rows),
                             ("q1_27_gate_sensitivity.csv", threshold_rows), ("four_system_2d_yield_comparison.csv", series),
                             ("common_distance_domain_conditioning.csv", domain_rows)):
        write_csv(output / filename, values)
    report = ["# C5/C8 q1 umbrella：第三独立端点补充与二维产率分析", "",
              "沿用原总结的逐帧 soft-NAC、per-pair 平均、自由能分解及 paired replica bootstrap；不拟合实验产率。",
              "本批增加 6 条路径、150 个窗口；合并后每体系每类 3 个独立端点，共 18 条路径、450 个窗口、360450 帧。",
              "距离是唯一施加偏置的坐标；二维 F(d,Ω) 来自无偏帧权重和逐帧取向统计，不是双坐标施加 restraint 的二维 umbrella。", "",
              "## 1. 方法与产物映射", "",
              "每窗平衡 100 ps、生产 500 ps，分析生产段 100–500 ps，温度 313.15 K，力常数 500 kJ mol⁻¹ nm⁻²。",
              "NAC = S≤(d;0.50 nm,0.05 nm) × S≤(axis;45°,7.5°) × S≥(side;45°,7.5°)。",
              "ΔΔG(A−B)=−RT ln(P_A/P_B)，负值偏 A。概率先在每条路径上重加权，再对独立端点等权平均；最后乘候选数 1:8:4。",
              "q0 固定 cross。q1 full reuse→2 R2；partial reuse→1 R4；full switch→opposite。opposite reuse→2 R2、switch→1 R4，候选数 2:4。",
              "由于 2 R2 与 1 R4 含相同单体总质量，w(R2)=p(full reuse)+p(full switch)·p(q2 reuse)。未改变 R1/R3 排除假设。", "",
              "## 2. 四体系模拟与实验比较（仅在 R2+R4 内归一化）", "",
              "实验原始R2/R4收率分别为C5：75/24、C6：39/55、C7：20/71、C8：13/63；下表比较的是各自R2+R4内的质量占比，而非绝对收率。", "",
              "| 体系 | q1 每类端点数 | 模拟 R2 | 模拟 R4 | 实验 R2 | 实验 R4 | R2 偏差/百分点 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in series:
        report.append(f"| {r['system']} | {r['q1_endpoints_per_class']} | {r['predicted_R2_mass_fraction']:.1%} | {r['predicted_R4_mass_fraction']:.1%} | {r['experimental_R2_target_normalized']:.1%} | {r['experimental_R4_target_normalized']:.1%} | {r['R2_error_percentage_points']:+.1f} |")
    report += ["", "C6/C7 使用已完成的二维结果，本轮未新增采样；不能把两端点与三端点结果描述成完全相同的收敛程度。", "",
               "## 3. 第三端点的影响与不确定性", "",
               "| 体系 | 原2端点二维 R2 | 新3端点二维 R2 | q1端点bootstrap 95%区间（q2固定） |",
               "|---|---:|---:|---:|"]
    for r in uncertainty_rows:
        old = float(baseline[r['system']]['R2_mass_fraction'])
        report.append(f"| {r['system']} | {old:.1%} | {r['R2_mass_fraction']:.1%} | [{r['q1_endpoint_bootstrap_ci_low']:.1%}, {r['q1_endpoint_bootstrap_ci_high']:.1%}] |")
    report += ["", "采用 10000 次 paired endpoint bootstrap（seed=20260913），三类共享端点编号。仅有3个端点，此为小样本诊断区间；不包含窗内时间相关性、q2不确定性、力场或产物映射误差。", "",
               "| 体系 | 独立端点 | full reuse | partial reuse | full switch | R2（q2固定） |",
               "|---|---:|---:|---:|---:|---:|"]
    for r in replica_rows:
        report.append(f"| {r['system']} | {r['replica']} | {r['full_reuse']:.1%} | {r['partial_reuse']:.1%} | {r['full_switch']:.1%} | {r['R2_mass_fraction']:.1%} |")
    report += ["", "## 4. 自由能分解", "",
               "| 体系 | A−B | ΔΔG contact | ΔΔG orientation | ΔΔG NAC / kcal mol⁻¹ | 端点95%区间 | α=NAC/RT |",
               "|---|---|---:|---:|---:|---:|---:|"]
    for r in contrast_rows:
        report.append(f"| {r['system']} | {r['definition']} | {r['ddG_contact_kcal_mol']:+.3f} | {r['ddG_orientation_kcal_mol']:+.3f} | {r['ddG_NAC_kcal_mol']:+.3f} | [{r['q1_endpoint_bootstrap_ci_low']:+.3f}, {r['q1_endpoint_bootstrap_ci_high']:+.3f}] | {r['alpha_NAC']:+.3f} |")
    report += ["", "这里α是相应per-pair类别对比的无量纲自由能差，不把三类q1压成一个reuse/switch参数。", "",
               "## 5. 半段、删端点与阈值诊断", "",
               "| 体系 | 诊断 | R2 |", "|---|---|---:|"]
    for r in sensitivity_rows:
        report.append(f"| {r['system']} | {r['setting']} | {r['R2_mass_fraction']:.1%} |")
    for system in ("C5", "C8"):
        selected = [r for r in threshold_rows if r['system'] == system]
        ddg = [r['full_reuse_minus_full_switch_ddG_kcal_mol'] for r in selected]
        predictions = [r['R2_mass_fraction_q2_primary_fixed'] for r in selected]
        report += ["", f"{system} 的27套q1门控扫描：full reuse−full switch 在 {sum(x < 0 for x in ddg)}/27 套为负、{sum(x > 0 for x in ddg)}/27 套为正；传播R2范围为 {min(predictions):.1%}–{max(predictions):.1%}。q2保持主门控固定，故不是整条反应网络的阈值扫描。"]
    report += ["", "## 6. 数据完整性与解释边界", "",
               f"18/18条路径包含25个完整500 ps生产窗口；最小邻窗直方图重叠系数（0.02 nm bins）={min(r['minimum_neighbor_histogram_overlap_0_02nm_bins'] for r in qa_rows):.3f}。",
               f"重构几何与GROMACS pull距离最大RMS误差={max(r['geometry_pull_rms_A'] for r in qa_rows):.4f} Å，最大逐帧误差={max(r['geometry_pull_max_A'] for r in qa_rows):.4f} Å；全部乙烯基键长通过1.15–1.55 Å检查。",
               "重加权ESS只度量权重集中程度，不能当作时间去相关后的独立帧数；WHAM方程收敛和窗口重叠也不等同于慢构象盆地已收敛。",
               "三个端点复核同一组选定乙烯基对，并没有覆盖partial reuse的8个或full switch的4个全部候选对。multiply-by-multiplicity仍是类别代表性假设。",
               "", "### 6.1 边缘/范围外权重诊断", "",
               "原WHAM范围为0.32–2.78 nm，而当前逐帧重加权概率对全部观测距离归一化。以下检查范围外帧是否主导归一化。", "",
               "| 路径 | 权重ESS（非时间独立数） | 范围外权重 | 最大观测距离/nm |",
               "|---|---:|---:|---:|"]
    for r in qa_rows:
        if r['reweight_ESS_not_time_decorrelated'] < 2000 or r['weight_outside_common_WHAM_domain_0_32_2_78nm'] > 0.10:
            report.append(f"| {r['pathway']} | {r['reweight_ESS_not_time_decorrelated']:.1f} | {r['weight_outside_common_WHAM_domain_0_32_2_78nm']:.1%} | {r['maximum_observed_pull_distance_nm']:.3f} |")
    report += ["", "若仅条件于各路径距离位于共同WHAM范围内，再按同一候选数传播（q2主结果固定）：", ""]
    for r in domain_rows:
        report.append(f"- {r['system']}：range-conditioned R2={r['range_conditioned_R2_mass_fraction']:.1%}。")
    report += ["", "这是距离范围条件化的敏感性检查，不是全构象系综的无条件产率，不能直接替换主结果或当作拟合改善。范围外权重集中与半段漂移说明需要更充分的范围/慢盆地采样。",
               "",
               "umbrella是稀有近接触构象的主要自由能证据；无偏MD用于交叉检查。两者均为无催化剂构象预组织代理，不是Ru催化化学活化能垒或绝对收率。",
               "原始结果和原2端点报告均保留。本轮未提交新计算。", "",
               "## 7. 本轮结论", "",
               "1. 六个新任务均成功下载，150个新窗口完整。新增第三端点后，C5的R2点估计由70.5%变为70.2%，C8由3.0%变为3.3%；C8与实验的差距并未解决。",
               "2. C5的总R2占比变化小不代表q1分支已收敛：第三端点full reuse的per-pair NAC概率比前两个低超过2000倍。full reuse与full switch→q2 reuse两条R2来源互相补偿，掩盖了分支差异。",
               "3. C8在分析段前后半段的R2估计为1.0%和8.4%，部分路径权重集中程度ESS低至9.8。共同距离范围条件化仅将R2改为3.2%，表明范围外归一化本身不足以解释实验17.1%的R2。",
               "4. 当前模拟为C5>C7≈C6>C8，而实验为C5>C6>C7>C8。因此模型尚未完整再现四体系产物比例；不能以这组点估计宣布定量预测成功，也不能用窄的三端点bootstrap区间替代时间收敛检验。",
               "5. 距离umbrella已经完成，距离–取向二维重加权也已完成；双坐标同时加偏置的二维umbrella尚未完成。下一轮收敛补充应优先针对C5 full reuse和C8低ESS/半段漂移路径，并复核q2与类别代表性；本轮没有擅自改变R1/R3排除或产物映射。", "",
               "新任务ID：20730987、20730985、20730988、20730989、20730990、20730986。",
               "原始下载目录：results/a4_m_c5_c8_q1_three_class_umbrella_rep3_download。", ""]
    (output / "SUPPLEMENT_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(output), "series": series, "uncertainty": uncertainty_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
