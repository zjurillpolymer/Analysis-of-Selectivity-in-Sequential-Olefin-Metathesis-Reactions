#!/usr/bin/env python3
"""Distance-orientation reweighting of the completed A4-m-C6/C7 q2 umbrellas."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.analyze_a4_m_c6_c7_q1_umbrella_2d import prepare_topology  # noqa: E402
from analysis.analyze_a4_q2_umbrella_2d_reweight import (  # noqa: E402
    RT_KCAL,
    PathwayResult,
    analyze_pathway,
    write_csv,
)


GROUPS = {
    "C6_q2_adjacent_cross": ("c6_q2_adjacent_cross_v4_v7",),
    "C6_q2_adjacent_same": ("c6_q2_adjacent_same_v4_v6",),
    "C6_q2_opposite_switch": ("c6_q2_opposite_switch_v3_v7",),
    "C6_q2_opposite_reuse": ("c6_q2_opposite_reuse_v3_v4",),
    "C7_q2_adjacent_cross": ("c7_q2_adjacent_cross_v2_v8",),
    "C7_q2_adjacent_same": ("c7_q2_adjacent_same_v7_v8",),
    "C7_q2_opposite_switch": ("c7_q2_opposite_switch_v2_v6",),
    "C7_q2_opposite_reuse": ("c7_q2_opposite_reuse_v2_v5",),
}

CONTRASTS = (
    ("C6_q2_adjacent", "cross", "same", "C6_q2_adjacent_cross", "C6_q2_adjacent_same"),
    ("C7_q2_adjacent", "cross", "same", "C7_q2_adjacent_cross", "C7_q2_adjacent_same"),
    ("C6_q2_opposite", "switch", "reuse", "C6_q2_opposite_switch", "C6_q2_opposite_reuse"),
    ("C7_q2_opposite", "switch", "reuse", "C7_q2_opposite_switch", "C7_q2_opposite_reuse"),
)


def aggregate(selected: list[PathwayResult]) -> dict[str, object]:
    def metrics(values: list[dict[str, float]]) -> dict[str, float]:
        contact = sum(value["soft_contact_probability"] for value in values) / len(values)
        nac = sum(value["soft_nac_probability"] for value in values) / len(values)
        orientation = nac / contact
        return {
            "soft_contact_probability": contact,
            "conditional_orientation": orientation,
            "soft_nac_probability": nac,
            "contact_free_energy_kcal_mol": -RT_KCAL * math.log(contact),
            "orientation_free_energy_kcal_mol": -RT_KCAL * math.log(orientation),
            "nac_free_energy_kcal_mol": -RT_KCAL * math.log(nac),
        }

    return {
        "metrics": metrics([item.metrics for item in selected]),
        "half_metrics": {
            half: metrics([item.half_metrics[half] for item in selected])
            for half in ("first_half", "second_half")
        },
    }


def contrast(state: str, left_label: str, right_label: str,
             left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    record: dict[str, object] = {
        "state": state,
        "contrast": f"{left_label}_minus_{right_label}",
    }
    for prefix, half in (("full", None), ("first_half", "first_half"),
                         ("second_half", "second_half")):
        left_values = left["metrics"] if half is None else left["half_metrics"][half]
        right_values = right["metrics"] if half is None else right["half_metrics"][half]
        contact = -RT_KCAL * math.log(
            float(left_values["soft_contact_probability"])
            / float(right_values["soft_contact_probability"])
        )
        orientation = -RT_KCAL * math.log(
            float(left_values["conditional_orientation"])
            / float(right_values["conditional_orientation"])
        )
        nac = -RT_KCAL * math.log(
            float(left_values["soft_nac_probability"])
            / float(right_values["soft_nac_probability"])
        )
        record[f"{prefix}_ddG_contact_kcal_mol"] = contact
        record[f"{prefix}_ddG_orientation_kcal_mol"] = orientation
        record[f"{prefix}_ddG_nac_kcal_mol"] = nac
        record[f"{prefix}_additivity_residual_kcal_mol"] = nac - contact - orientation
    record["left_over_right_nac_odds"] = math.exp(
        -float(record["full_ddG_nac_kcal_mol"]) / RT_KCAL
    )
    record["half_difference_ddG_nac_kcal_mol"] = (
        float(record["second_half_ddG_nac_kcal_mol"])
        - float(record["first_half_ddG_nac_kcal_mol"])
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=PROJECT)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument(
        "--raw-root", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q2_extracted",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q2_umbrella_2d_analysis",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology_root = output / "reactive_pair_topologies"

    paths: dict[str, Path] = {}
    for pathway in args.raw_root.resolve().glob("*/pathway"):
        metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
        paths[str(metadata["pathway"])] = pathway
    expected = {name for members in GROUPS.values() for name in members}
    if set(paths) != expected:
        raise ValueError(f"q2 pathway mismatch: missing={expected-set(paths)}, extra={set(paths)-expected}")

    results: dict[str, PathwayResult] = {}
    pathway_rows: list[dict[str, object]] = []
    for name, pathway in sorted(paths.items()):
        prepare_topology(pathway, topology_root)
        result = analyze_pathway(pathway, workspace, output, args.gmx, topology_root)
        results[name] = result
        pathway_rows.append({
            "pathway": name,
            "state": result.metadata["state"],
            "system": result.metadata["system"],
            "route_family": result.metadata["route_family"],
            "windows": len(result.centers_nm),
            "frames": len(result.distance_nm),
            "mbar_iterations": result.mbar_iterations,
            "mbar_residual": result.mbar_residual,
            "effective_sample_size": result.effective_sample_size,
            "geometry_pull_rms_A": result.geometry_pull_rms_A,
            "geometry_pull_max_A": result.geometry_pull_max_A,
            **result.metrics,
            **{
                f"{half}_{key}": value
                for half, values in result.half_metrics.items() for key, value in values.items()
            },
        })
        print(f"analyzed={name} windows={len(result.centers_nm)} frames={len(result.distance_nm)}",
              flush=True)

    groups = {
        group: aggregate([results[name] for name in members])
        for group, members in GROUPS.items()
    }
    group_rows: list[dict[str, object]] = []
    for group, members in GROUPS.items():
        item = groups[group]
        group_rows.append({
            "group": group,
            "pathways": ";".join(members),
            "pathway_count": len(members),
            **item["metrics"],
            **{
                f"{half}_{key}": value
                for half, values in item["half_metrics"].items() for key, value in values.items()
            },
        })
    contrasts = [
        contrast(state, left_label, right_label, groups[left], groups[right])
        for state, left_label, right_label, left, right in CONTRASTS
    ]
    write_csv(output / "pathway_reweight_summary.csv", pathway_rows)
    write_csv(output / "group_reweight_summary.csv", group_rows)
    write_csv(output / "q2_2d_reweighted_contrasts.csv", contrasts)
    summary = {
        "temperature_K": 313.15,
        "method": "distance-bias WHAM/MBAR-equivalent frame reweighting plus the original soft orientation gate",
        "scope": "eight representative C6/C7 q2 pathways",
        "pathways": pathway_rows,
        "groups": group_rows,
        "contrasts": contrasts,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(contrasts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
