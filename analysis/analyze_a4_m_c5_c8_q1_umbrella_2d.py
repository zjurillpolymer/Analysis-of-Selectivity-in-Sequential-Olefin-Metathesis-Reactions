#!/usr/bin/env python3
"""Distance/orientation reweighting for the completed C5/C8 q1 umbrellas.

The simulations bias midpoint distance, while this analysis reconstructs the
unbiased frame weights and applies the same vinyl-axis/side-angle NAC gate used
for the validated C6/C7 and q2 two-dimensional analyses.  Thus the q1 branch
weights are based on soft-NAC probability rather than distance-only contact.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.analyze_a4_m_c6_c7_q1_umbrella_2d import prepare_topology  # noqa: E402
from analysis.analyze_a4_q2_umbrella_2d_reweight import (  # noqa: E402
    RT_KCAL,
    analyze_pathway,
    write_csv,
)


JOBS = {
    "c5_q1_full_reuse_v1_v6_rep1": ("20730620", "full_reuse"),
    "c5_q1_full_reuse_v1_v6_rep2": ("20730628", "full_reuse"),
    "c5_q1_partial_reuse_v1_v5_rep1": ("20730619", "partial_reuse"),
    "c5_q1_partial_reuse_v1_v5_rep2": ("20730627", "partial_reuse"),
    "c5_q1_full_switch_v2_v4_rep1": ("20730629", "full_switch"),
    "c5_q1_full_switch_v2_v4_rep2": ("20730622", "full_switch"),
    "c8_q1_full_reuse_v5_v7_rep1": ("20730624", "full_reuse"),
    "c8_q1_full_reuse_v5_v7_rep2": ("20730625", "full_reuse"),
    "c8_q1_partial_reuse_v1_v5_rep1": ("20730626", "partial_reuse"),
    "c8_q1_partial_reuse_v1_v5_rep2": ("20730618", "partial_reuse"),
    "c8_q1_full_switch_v1_v3_rep1": ("20730623", "full_switch"),
    "c8_q1_full_switch_v1_v3_rep2": ("20730621", "full_switch"),
}

CATEGORIES = {
    "C5": {
        "full_reuse": ("c5_q1_full_reuse_v1_v6_rep1", "c5_q1_full_reuse_v1_v6_rep2"),
        "partial_reuse": ("c5_q1_partial_reuse_v1_v5_rep1", "c5_q1_partial_reuse_v1_v5_rep2"),
        "full_switch": ("c5_q1_full_switch_v2_v4_rep1", "c5_q1_full_switch_v2_v4_rep2"),
    },
    "C8": {
        "full_reuse": ("c8_q1_full_reuse_v5_v7_rep1", "c8_q1_full_reuse_v5_v7_rep2"),
        "partial_reuse": ("c8_q1_partial_reuse_v1_v5_rep1", "c8_q1_partial_reuse_v1_v5_rep2"),
        "full_switch": ("c8_q1_full_switch_v1_v3_rep1", "c8_q1_full_switch_v1_v3_rep2"),
    },
}
MULTIPLICITY = {"full_reuse": 1, "partial_reuse": 8, "full_switch": 4}


def aggregate(results: list[object]) -> dict[str, object]:
    metrics = [result.metrics for result in results]
    half = {
        name: [result.half_metrics[name] for result in results]
        for name in ("first_half", "second_half")
    }

    def one(values: list[dict[str, float]]) -> dict[str, float]:
        contact = float(np.mean([value["soft_contact_probability"] for value in values]))
        nac = float(np.mean([value["soft_nac_probability"] for value in values]))
        orientation = nac / contact
        return {
            "soft_contact_probability": contact,
            "conditional_orientation": orientation,
            "soft_nac_probability": nac,
            "contact_free_energy_kcal_mol": -RT_KCAL * math.log(contact),
            "orientation_free_energy_kcal_mol": -RT_KCAL * math.log(orientation),
            "nac_free_energy_kcal_mol": -RT_KCAL * math.log(nac),
        }

    return {"metrics": one(metrics), "half_metrics": {name: one(values) for name, values in half.items()}}


def contrast(system: str, left: str, right: str, groups: dict[str, dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {"system": system, "definition": f"{left} minus {right}"}
    for prefix, half in (("full", None), ("first_half", "first_half"), ("second_half", "second_half")):
        lm = groups[left]["metrics"] if half is None else groups[left]["half_metrics"][half]
        rm = groups[right]["metrics"] if half is None else groups[right]["half_metrics"][half]
        for key in ("soft_contact_probability", "conditional_orientation", "soft_nac_probability"):
            result[f"{prefix}_ddG_{key}_kcal_mol"] = -RT_KCAL * math.log(float(lm[key]) / float(rm[key]))
    result["reuse_or_switch_nac_odds"] = math.exp(-float(result["full_ddG_soft_nac_probability_kcal_mol"]) / RT_KCAL)
    return result


def q2_umbrella(root: Path, system: str) -> dict[str, float]:
    path = root / "results/a4_q2_umbrella_2d_analysis/final/group_reweight_summary.csv"
    values = {row["group"]: float(row["soft_nac_probability"]) for row in csv.DictReader(path.open())}
    reuse = values[f"{system}_opposite_reuse"]
    switch = values[f"{system}_opposite_switch"]
    raw = {"R2x2": 2.0 * reuse, "R4": 4.0 * switch}
    total = sum(raw.values())
    return {key: value / total for key, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_download")
    parser.add_argument(
        "--supplement-root",
        type=Path,
        default=None,
        help="optional Bohrium download root containing additional */extracted/pathway results",
    )
    parser.add_argument("--output", type=Path, default=PROJECT / "results/a4_m_c5_c8_q1_umbrella_2d_analysis")
    parser.add_argument("--gmx", default="gmx")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology_root = output / "reactive_pair_topologies"
    path_specs = {
        alias: (root / job_id / "extracted/pathway", category)
        for alias, (job_id, category) in JOBS.items()
    }
    categories = {
        system: {category: list(aliases) for category, aliases in definitions.items()}
        for system, definitions in CATEGORIES.items()
    }
    if args.supplement_root is not None:
        supplement_root = args.supplement_root.resolve()
        for metadata_path in sorted(supplement_root.glob("*/extracted/pathway/pathway.json")):
            pathway = metadata_path.parent
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            alias = str(metadata["pathway"])
            if alias in path_specs:
                raise ValueError(f"duplicate pathway alias in supplement: {alias}")
            system = str(metadata["system"]).replace("A4_m_", "")
            category = str(metadata["route_family"])
            if system not in categories or category not in categories[system]:
                raise ValueError(f"unsupported supplement category: {system}/{category}")
            path_specs[alias] = (pathway, category)
            categories[system][category].append(alias)

    results = {}
    pathway_rows = []
    for alias, (pathway, category) in path_specs.items():
        prepare_topology(pathway, topology_root)
        result = analyze_pathway(pathway, PROJECT, output, args.gmx, topology_root)
        results[alias] = result
        pathway_rows.append({
            "pathway": alias,
            "system": result.metadata["system"],
            "route_family": category,
            "replica": result.metadata.get("replica", ""),
            "windows": len(result.centers_nm),
            "frames": len(result.distance_nm),
            "mbar_iterations": result.mbar_iterations,
            "mbar_residual": result.mbar_residual,
            "effective_sample_size": result.effective_sample_size,
            "geometry_pull_rms_A": result.geometry_pull_rms_A,
            "geometry_pull_max_A": result.geometry_pull_max_A,
            **result.metrics,
            **{f"{half}_{key}": value for half, values in result.half_metrics.items() for key, value in values.items()},
        })
        print(f"analyzed={alias} windows={len(result.centers_nm)} frames={len(result.distance_nm)}", flush=True)

    groups, category_rows = {}, []
    for system, definitions in categories.items():
        groups[system] = {}
        for category, aliases in definitions.items():
            pooled = aggregate([results[alias] for alias in aliases])
            groups[system][category] = pooled
            category_rows.append({
                "system": system,
                "category": category,
                "pathways": ";".join(aliases),
                **pooled["metrics"],
                **{f"{half}_{key}": value for half, values in pooled["half_metrics"].items() for key, value in values.items()},
            })

    contrast_rows = []
    for system in ("C5", "C8"):
        contrast_rows.extend([
            contrast(system, "full_reuse", "partial_reuse", groups[system]),
            contrast(system, "partial_reuse", "full_switch", groups[system]),
            contrast(system, "full_reuse", "full_switch", groups[system]),
        ])

    branch_rows, yield_rows = [], []
    for system in ("C5", "C8"):
        raw = {category: MULTIPLICITY[category] * groups[system][category]["metrics"]["soft_nac_probability"]
               for category in MULTIPLICITY}
        total = sum(raw.values())
        q1 = {category: value / total for category, value in raw.items()}
        for category, probability in q1.items():
            branch_rows.append({"system": system, "category": category, "probability": probability,
                                "soft_nac_probability": groups[system][category]["metrics"]["soft_nac_probability"],
                                "candidate_multiplicity": MULTIPLICITY[category]})
        q2 = q2_umbrella(PROJECT, system)
        r2 = q1["full_reuse"] + q1["full_switch"] * q2["R2x2"]
        yield_rows.append({"system": system, "q1_source": "q1_2d_reweight", "q2_source": "q2_2d_reweight",
                           "q1_full_reuse": q1["full_reuse"], "q1_partial_reuse": q1["partial_reuse"],
                           "q1_full_switch": q1["full_switch"], "q2_R2x2": q2["R2x2"], "q2_R4": q2["R4"],
                           "R2_mass_fraction": r2, "R4_mass_fraction": 1.0 - r2})

    write_csv(output / "pathway_reweight_summary.csv", pathway_rows)
    write_csv(output / "category_reweight_summary.csv", category_rows)
    write_csv(output / "q1_2d_contrasts.csv", contrast_rows)
    write_csv(output / "q1_2d_branch_probabilities.csv", branch_rows)
    write_csv(output / "q1_2d_yield_prediction.csv", yield_rows)
    summary = {"method": "distance-bias WHAM/MBAR-equivalent frame reweighting plus frame-resolved orientation gate",
               "temperature_K": 313.15, "pathway_count": len(pathway_rows), "categories": category_rows,
               "contrasts": contrast_rows, "branch_probabilities": branch_rows, "yield_predictions": yield_rows}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# A4-m-C5/C8 q1 二维距离–取向重加权",
        "",
        f"本分析使用已完成的 {len(pathway_rows)} 条距离 umbrella 轨迹；通过 MBAR 等价重加权恢复无偏帧权重，并显式计算乙烯基轴向/侧向取向门控。因此这里的 q1 分支采用 soft-NAC，而不是此前仅按距离定义的 contact 概率。",
        "",
        "## q1 三类自由能",
        "",
        "| 体系 | 类别 | soft contact | conditional orientation | soft NAC | NAC ΔG (kcal/mol) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in category_rows:
        report.append(f"| {row['system']} | {row['category']} | {float(row['soft_contact_probability']):.6g} | {float(row['conditional_orientation']):.6g} | {float(row['soft_nac_probability']):.6g} | {float(row['nac_free_energy_kcal_mol']):.3f} |")
    report += [
        "",
        "## 候选多重性归一化后的 q1 分支",
        "",
        "| 体系 | full reuse | partial reuse | full switch |",
        "|---|---:|---:|---:|",
    ]
    for system in ("C5", "C8"):
        row = {r["category"]: r for r in branch_rows if r["system"] == system}
        report.append(f"| {system} | {float(row['full_reuse']['probability']):.6f} | {float(row['partial_reuse']['probability']):.6f} | {float(row['full_switch']['probability']):.6f} |")
    report += [
        "",
        "## 产率传播",
        "",
        "| 体系 | q1/q2 证据 | R2 | R4 |",
        "|---|---|---:|---:|",
    ]
    for row in yield_rows:
        report.append(f"| {row['system']} | q1 2D + q2 2D | {float(row['R2_mass_fraction']):.3f} | {float(row['R4_mass_fraction']):.3f} |")
    report += [
        "",
        "二维重加权把取向门控纳入 q1 NAC；它是 C5/C8 后续应优先采用的 umbrella 口径。该结果仍是代表路径的 equal-path 类别平均，不等于完整候选集合的无偏动力学人口。",
        "以上为点估计，不是采样已收敛的声明。须结合独立端点、前后半段和权重集中程度诊断，不能仅凭WHAM收敛或窗口完整判断产物比例已可靠。",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"pathways": len(pathway_rows), "output": str(output), "yield_predictions": yield_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
