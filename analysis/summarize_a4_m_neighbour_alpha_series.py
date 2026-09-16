#!/usr/bin/env python3
"""Summarize the A4-m-C5--C8 mixed-q2 neighbour-arm alpha series."""

from __future__ import annotations

import csv
import math
from pathlib import Path


R_KCAL = 0.00198720425864083
TEMPERATURE_K = 313.15
RT = R_KCAL * TEMPERATURE_K
PRIMARY = "d5_axis45_side45"

SOURCES = (
    Path("results/a4_q2_missing_md_mixed_10ns_analysis_neighbour_only"),
    Path("results/a4_m_c6_c7_mixed_10ns_v2_analysis"),
)

EXPERIMENT = {
    "A4_m_C5": {"spacer": 5, "R2_wt_pct": 75, "R4_wt_pct": 24},
    "A4_m_C6": {"spacer": 6, "R2_wt_pct": 39, "R4_wt_pct": 55},
    "A4_m_C7": {"spacer": 7, "R2_wt_pct": 20, "R4_wt_pct": 71},
    "A4_m_C8": {"spacer": 8, "R2_wt_pct": 13, "R4_wt_pct": 63},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_all(project: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    contrasts: list[dict[str, str]] = []
    replicas: list[dict[str, str]] = []
    for source in SOURCES:
        root = project / source
        contrasts.extend(read_csv(root / "free_energy_contrasts.csv"))
        replicas.extend(read_csv(root / "replica_group_statistics.csv"))
    return contrasts, replicas


def replica_ddg(
    replicas: list[dict[str, str]], system: str, contrast: str
) -> list[float]:
    if contrast == "reuse_minus_switch":
        grouping, numerator, denominator = "route_family", "reuse", "switch"
    elif contrast == "cross_minus_same":
        grouping, numerator, denominator = "topology", "cross_arm", "same_arm"
    else:
        raise ValueError(contrast)
    selected = [
        row
        for row in replicas
        if row["system"] == system
        and row["setting_id"] == PRIMARY
        and row["grouping"] == grouping
        and row["group"] in (numerator, denominator)
    ]
    by_replica: dict[int, dict[str, float]] = {}
    for row in selected:
        by_replica.setdefault(int(row["replica"]), {})[row["group"]] = float(
            row["per_pair_nac"]
        )
    values = []
    for replica in sorted(by_replica):
        groups = by_replica[replica]
        values.append(-RT * math.log(groups[numerator] / groups[denominator]))
    if len(values) != 3:
        raise ValueError(f"expected 3 replicas for {system}/{contrast}, found {len(values)}")
    return values


def summarize(
    contrasts: list[dict[str, str]], replicas: list[dict[str, str]], contrast: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for system, experimental in EXPERIMENT.items():
        rows = [
            row
            for row in contrasts
            if row["system"] == system and row["contrast"] == contrast
        ]
        primary = next(row for row in rows if row["setting_id"] == PRIMARY)
        dg = float(primary["ddG_total_nac_point_kcal_mol"])
        low = float(primary["ddG_total_nac_ci_low_kcal_mol"])
        high = float(primary["ddG_total_nac_ci_high_kcal_mol"])
        setting_values = [float(row["ddG_total_nac_point_kcal_mol"]) for row in rows]
        replicate_values = replica_ddg(replicas, system, contrast)
        output.append(
            {
                "system": system.replace("_", "-"),
                **experimental,
                "contrast": contrast,
                "ddG_contact_kcal_mol": float(primary["ddG_contact_point_kcal_mol"]),
                "ddG_orientation_kcal_mol": float(primary["ddG_orientation_point_kcal_mol"]),
                "ddG_total_NAC_kcal_mol": dg,
                "ddG_total_ci_low_kcal_mol": low,
                "ddG_total_ci_high_kcal_mol": high,
                "alpha_NAC": dg / RT,
                "alpha_ci_low": low / RT,
                "alpha_ci_high": high / RT,
                "rho_reuse_over_switch": math.exp(-dg / RT)
                if contrast == "reuse_minus_switch"
                else "",
                "favored": (
                    "reuse" if dg < 0 else "switch"
                )
                if contrast == "reuse_minus_switch"
                else ("cross" if dg < 0 else "same"),
                "robust_replica_CI": high < 0 or low > 0,
                "threshold_direction_consistent": all(value < 0 for value in setting_values)
                or all(value > 0 for value in setting_values),
                "threshold_ddG_min_kcal_mol": min(setting_values),
                "threshold_ddG_max_kcal_mol": max(setting_values),
                "replica_1_ddG_kcal_mol": replicate_values[0],
                "replica_2_ddG_kcal_mol": replicate_values[1],
                "replica_3_ddG_kcal_mol": replicate_values[2],
            }
        )
    return output


def f(value: object, digits: int = 3) -> str:
    return f"{float(value):+.{digits}f}"


def write_report(output: Path, alpha_rows: list[dict[str, object]], topology_rows: list[dict[str, object]]) -> None:
    lines = [
        "# A4-m-C5 至 C8 相邻臂反应选择性计算总总结",
        "",
        "> 版本：2026-09-10  ",
        "> 模型范围：只考虑同臂及 A4 环上相邻两臂的乙烯基候选；对位/非相邻臂候选不进入概率计算；不讨论 R3。",
        "",
        "## 摘要",
        "",
        "C5、C6、C7、C8 的 mixed-q2 状态现已全部完成相同协议的 3 条独立、每条 10 ns 显式二氯甲烷无偏 MD。每个体系的主要参数定义为 `alpha_NAC = ΔΔG(reuse-switch)/RT`；alpha<0 表示 reuse，alpha>0 表示 switch。",
        "",
        "统一相邻臂口径下，C5 明确偏 reuse；C6 明确偏 switch；C8 明确偏 switch。C7 的总体点估计偏 switch，但三个重复的方向不一致，95% replica-bootstrap CI 跨过 0，因此目前只能记为未定。27 套 NAC 距离/角度阈值对四个体系都给出相同的点估计方向，说明 C7 的不确定性来自独立轨迹之间的构象盆地差异，而不是某一个阈值的选择。",
        "",
        "## alpha_NAC 主结果",
        "",
        f"313.15 K 下 RT={RT:.4f} kcal mol^-1。rho=exp(-alpha) 是单对 reuse/switch NAC 几率比。",
        "",
        "| 体系 | ΔΔG reuse-switch | 95% CI | alpha_NAC | alpha 95% CI | rho reuse/switch | 结论 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in alpha_rows:
        conclusion = str(row["favored"])
        if not bool(row["robust_replica_CI"]):
            conclusion += "（重复间未定）"
        lines.append(
            f"| {row['system']} | {f(row['ddG_total_NAC_kcal_mol'])} | "
            f"[{f(row['ddG_total_ci_low_kcal_mol'])}, {f(row['ddG_total_ci_high_kcal_mol'])}] | "
            f"{f(row['alpha_NAC'], 2)} | [{f(row['alpha_ci_low'], 2)}, {f(row['alpha_ci_high'], 2)}] | "
            f"{float(row['rho_reuse_over_switch']):.4g} | {conclusion} |"
        )
    lines += [
        "",
        "## C6 与 C7 的新增 10 ns 结果",
        "",
        "| 体系 | ΔΔG contact | ΔΔG orientation | ΔΔG total NAC | reuse/switch |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in alpha_rows[1:3]:
        lines.append(
            f"| {row['system']} | {f(row['ddG_contact_kcal_mol'])} | "
            f"{f(row['ddG_orientation_kcal_mol'])} | {f(row['ddG_total_NAC_kcal_mol'])} | "
            f"{float(row['rho_reuse_over_switch']):.4g} |"
        )
    lines += [
        "",
        "C6 的总差值为正且 CI 全部位于 0 以上，单对 switch/reuse 优势约为 7.53 倍；其中主要贡献来自接触可及性。C7 的单对 switch/reuse 点估计约为 1.77 倍，但 CI 跨 0，不能据此断言真实总体偏好。",
        "",
        "## 重复间稳定性",
        "",
        "| 体系 | replica 1 ΔΔG | replica 2 ΔΔG | replica 3 ΔΔG | 27 阈值方向 |",
        "|---|---:|---:|---:|---|",
    ]
    for row in alpha_rows:
        lines.append(
            f"| {row['system']} | {f(row['replica_1_ddG_kcal_mol'])} | "
            f"{f(row['replica_2_ddG_kcal_mol'])} | {f(row['replica_3_ddG_kcal_mol'])} | "
            f"{'27/27 一致' if row['threshold_direction_consistent'] else '不一致'} |"
        )
    lines += [
        "",
        "C6 的三个重复均偏 switch，但两个重复中的 reuse 对几乎从未进入接触区，因此 CI 上界很大；这说明方向稳定，但自由能差的绝对大小仍受稀有事件检测限影响。C7 的 replica 1 和 3 偏 reuse，replica 2 强烈偏 switch，说明 10 ns 内仍保留明显构象盆地记忆。",
        "",
        "## same/cross 辅助结果",
        "",
        "| 体系 | ΔΔG cross-same | 95% CI | cross/same NAC odds | 结论 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in topology_rows:
        odds = math.exp(-float(row["ddG_total_NAC_kcal_mol"]) / RT)
        conclusion = "cross" if float(row["ddG_total_NAC_kcal_mol"]) < 0 else "same"
        if not bool(row["robust_replica_CI"]):
            conclusion += "（重复间未定）"
        lines.append(
            f"| {row['system']} | {f(row['ddG_total_NAC_kcal_mol'])} | "
            f"[{f(row['ddG_total_ci_low_kcal_mol'])}, {f(row['ddG_total_ci_high_kcal_mol'])}] | "
            f"{odds:.3g} | {conclusion} |"
        )
    lines += [
        "",
        "四个体系的 cross-same 点估计都偏 cross，但真正区分 R2/R4 路径的仍是 cross 内部的 reuse/switch，而不是把 same 与 cross 直接对应到产物。",
        "",
        "## 与实验和单参数模型的关系",
        "",
        "实验上 C5 以 R2 为主，而 C6、C7、C8 以 R4 为主。相邻臂 mixed-q2 结果在定性分类上得到 C5=reuse、C6=switch、C8=switch；C7 的点估计为 switch，但统计上未定。因而当前数据支持从 C5 到较长臂体系发生 reuse-to-switch 转换。",
        "",
        "但是，四个 alpha_NAC 点估计并不随 C 单调增加：C6 的 alpha 高于 C7，且 C7 的重复间差异很大。因此目前可以为每个 spacer 报告一个有效 alpha_NAC，却不能把四点当作已经验证的单一平滑 alpha(C) 规律。若最终目标是只保留 alpha 一个参数，最稳妥的做法是把 C7 标记为区间约束，而不是强行给出精确点值。",
        "",
        "早期 2 ns q1/opposite-q2 pilot 中，C6 往往偏 reuse、C7 偏 switch；与本次 mixed-q2 的 C6 switch 结果不同。这说明 alpha 的表观值还可能依赖已经形成的两条反应边属于何种历史。mixed-q2 与 C5/C8 的关键验证状态完全同构，因此本报告将其作为主比较，旧 pilot 只作为状态依赖证据。",
        "",
        "## 计算边界",
        "",
        "- 数值是无催化剂底物的构象预组织/NAC 自由能代理，不是 Ru 催化反应的活化自由能垒。",
        "- 所有对比都是 per-pair；候选数多重性没有混入 alpha。",
        "- 非相邻臂候选已排除，R3 及终止、副反应等不在本模型范围内。",
        "- 6 个新增任务全部使用逐帧盒矢量处理 PBC，保留帧比例均为 100%。",
        "",
        "## 数据文件",
        "",
        "- `alpha_neighbour_series.csv`：C5-C8 alpha 主表及重复级数值。",
        "- `cross_same_neighbour_series.csv`：same/cross 辅助表。",
        "- `../a4_m_c6_c7_mixed_10ns_v2_analysis/free_energy_contrasts.csv`：C6/C7 27 阈值完整结果。",
        "- `../a4_m_c6_c7_mixed_10ns_v2_analysis/block_stability/`：500 ps 分块稳定性结果。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    contrasts, replicas = load_all(project)
    alpha_rows = summarize(contrasts, replicas, "reuse_minus_switch")
    topology_rows = summarize(contrasts, replicas, "cross_minus_same")
    output = project / "results/a4_m_c5_c8_neighbour_alpha_series"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "alpha_neighbour_series.csv", alpha_rows)
    write_csv(output / "cross_same_neighbour_series.csv", topology_rows)
    write_report(output, alpha_rows, topology_rows)
    print(output / "REPORT.md")


if __name__ == "__main__":
    main()
