#!/usr/bin/env python3
"""Analyze balanced two-endpoint q1 route classes for C6/C7."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


PARTIAL_PATHWAYS = {
    "c6_q1_partial_reuse_v1_v5_rep1": "20729814",
    "c6_q1_partial_reuse_v1_v5_rep2": "20729811",
    "c7_q1_partial_reuse_v5_v6_rep1": "20729812",
    "c7_q1_partial_reuse_v5_v6_rep2": "20729813",
}

FULL_REPLICATE_PATHWAYS = {
    "c6_q1_full_reuse_v1_v7_rep2": "20729862",
    "c6_q1_full_switch_v3_v4_rep2": "20729860",
    "c7_q1_full_reuse_v2_v5_rep1": "20729863",
    "c7_q1_full_switch_v1_v8_rep1": "20729861",
}

ORIGINAL_NAMES = {
    "C6": {
        "full_reuse": "c6_q1_reuse_v1_v7",
        "full_switch": "c6_q1_switch_v3_v4",
    },
    "C7": {
        "full_reuse": "c7_q1_reuse_v2_v5",
        "full_switch": "c7_q1_switch_v1_v8",
    },
}

PARTIAL_NAMES = {
    "C6": (
        "c6_q1_partial_reuse_v1_v5_rep1",
        "c6_q1_partial_reuse_v1_v5_rep2",
    ),
    "C7": (
        "c7_q1_partial_reuse_v5_v6_rep1",
        "c7_q1_partial_reuse_v5_v6_rep2",
    ),
}

FULL_REPLICATE_NAMES = {
    "C6": {
        "full_reuse": "c6_q1_full_reuse_v1_v7_rep2",
        "full_switch": "c6_q1_full_switch_v3_v4_rep2",
    },
    "C7": {
        "full_reuse": "c7_q1_full_reuse_v2_v5_rep1",
        "full_switch": "c7_q1_full_switch_v1_v8_rep1",
    },
}

ENDPOINT_CLASS_PATHS = {
    ("C6", 1): {
        "full_reuse": "c6_q1_reuse_v1_v7",
        "partial_reuse": "c6_q1_partial_reuse_v1_v5_rep1",
        "full_switch": "c6_q1_switch_v3_v4",
    },
    ("C6", 2): {
        "full_reuse": "c6_q1_full_reuse_v1_v7_rep2",
        "partial_reuse": "c6_q1_partial_reuse_v1_v5_rep2",
        "full_switch": "c6_q1_full_switch_v3_v4_rep2",
    },
    ("C7", 1): {
        "full_reuse": "c7_q1_full_reuse_v2_v5_rep1",
        "partial_reuse": "c7_q1_partial_reuse_v5_v6_rep1",
        "full_switch": "c7_q1_full_switch_v1_v8_rep1",
    },
    ("C7", 2): {
        "full_reuse": "c7_q1_reuse_v2_v5",
        "partial_reuse": "c7_q1_partial_reuse_v5_v6_rep2",
        "full_switch": "c7_q1_switch_v1_v8",
    },
}

PROBABILITY_KEYS = (
    "soft_contact_probability",
    "conditional_orientation",
    "soft_nac_probability",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def metric_set_from_row(row: Mapping[str, str], half: str | None = None) -> dict[str, float]:
    prefix = "" if half is None else f"{half}_"
    return {key: float(row[f"{prefix}{key}"]) for key in PROBABILITY_KEYS}


def metric_set_from_result(result: PathwayResult, half: str | None = None) -> dict[str, float]:
    values = result.metrics if half is None else result.half_metrics[half]
    return {key: float(values[key]) for key in PROBABILITY_KEYS}


def aggregate(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    contact = sum(item["soft_contact_probability"] for item in values) / len(values)
    nac = sum(item["soft_nac_probability"] for item in values) / len(values)
    orientation = nac / contact
    return {
        "soft_contact_probability": contact,
        "conditional_orientation": orientation,
        "soft_nac_probability": nac,
        "contact_free_energy_kcal_mol": -RT_KCAL * math.log(contact),
        "orientation_free_energy_kcal_mol": -RT_KCAL * math.log(orientation),
        "nac_free_energy_kcal_mol": -RT_KCAL * math.log(nac),
    }


def compare(system: str, left_name: str, right_name: str,
            left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object]:
    record: dict[str, object] = {
        "system": system,
        "definition": f"{left_name} minus {right_name}",
        "left_category": left_name,
        "right_category": right_name,
    }
    for prefix, half in (("full", None), ("first_half", "first_half"),
                         ("second_half", "second_half")):
        left_metrics = left["metrics"] if half is None else left["half_metrics"][half]
        right_metrics = right["metrics"] if half is None else right["half_metrics"][half]
        contact = -RT_KCAL * math.log(
            float(left_metrics["soft_contact_probability"])
            / float(right_metrics["soft_contact_probability"])
        )
        orientation = -RT_KCAL * math.log(
            float(left_metrics["conditional_orientation"])
            / float(right_metrics["conditional_orientation"])
        )
        nac = -RT_KCAL * math.log(
            float(left_metrics["soft_nac_probability"])
            / float(right_metrics["soft_nac_probability"])
        )
        record[f"{prefix}_ddG_contact_kcal_mol"] = contact
        record[f"{prefix}_ddG_orientation_kcal_mol"] = orientation
        record[f"{prefix}_ddG_nac_kcal_mol"] = nac
        record[f"{prefix}_additivity_residual_kcal_mol"] = nac - contact - orientation
    record["left_over_right_nac_odds"] = math.exp(
        -float(record["full_ddG_nac_kcal_mol"]) / RT_KCAL
    )
    record["alpha_nac"] = float(record["full_ddG_nac_kcal_mol"]) / RT_KCAL
    record["half_difference_ddG_nac_kcal_mol"] = (
        float(record["second_half_ddG_nac_kcal_mol"])
        - float(record["first_half_ddG_nac_kcal_mol"])
    )
    return record


def plot_contrasts(output: Path, rows: list[dict[str, object]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharex=True, constrained_layout=True)
    definitions = (
        "full_reuse minus partial_reuse",
        "partial_reuse minus full_switch",
        "full_reuse minus full_switch",
    )
    labels = ("full reuse − partial", "partial − full switch", "full reuse − full switch")
    colors = ("#4C78A8", "#E69F00", "#009E73")
    for axis, system in zip(axes, ("C6", "C7")):
        selected = {str(row["definition"]): row for row in rows if row["system"] == system}
        y = range(len(definitions))
        for index, definition in enumerate(definitions):
            row = selected[definition]
            contact = float(row["full_ddG_contact_kcal_mol"])
            orientation = float(row["full_ddG_orientation_kcal_mol"])
            total = float(row["full_ddG_nac_kcal_mol"])
            axis.barh(index, contact, color=colors[index], alpha=0.78, height=0.50)
            axis.barh(index, orientation, left=contact, color="#BBBBBB", height=0.50)
            axis.scatter(total, index, color="black", marker="D", s=22, zorder=4)
            first = float(row["first_half_ddG_nac_kcal_mol"])
            second = float(row["second_half_ddG_nac_kcal_mol"])
            axis.plot([first, second], [index + 0.21] * 2, color="0.35", lw=0.8)
            axis.scatter([first, second], [index + 0.21] * 2,
                         color=["#56B4E9", "#D55E00"], s=12, zorder=4)
        axis.axvline(0.0, color="0.25", lw=0.8)
        axis.set_yticks(list(y), labels)
        axis.invert_yaxis()
        axis.set_title(f"A4-m-{system} q1")
        axis.set_xlabel("left − right ΔΔG (kcal mol$^{-1}$)")
        axis.spines[["top", "right"]].set_visible(False)
    base = output / "q1_three_class_2d_contrasts"
    fig.savefig(base.with_suffix(".png"), dpi=350)
    fig.savefig(base.with_suffix(".svg"))
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=PROJECT)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument(
        "--partial-root", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_partial_reuse_umbrella_download",
    )
    parser.add_argument(
        "--full-replicate-root", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_full_route_replicates_download",
    )
    parser.add_argument(
        "--original-summary", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_umbrella_2d_analysis/pathway_reweight_summary.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_three_class_umbrella_2d_analysis",
    )
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology_root = output / "reactive_pair_topologies"

    new_results: dict[str, PathwayResult] = {}
    pathway_rows: list[dict[str, object]] = []
    locations = {
        **{
            name: args.partial_root.resolve() / job_id / "extracted/pathway"
            for name, job_id in PARTIAL_PATHWAYS.items()
        },
        **{
            name: args.full_replicate_root.resolve() / job_id / "extracted/pathway"
            for name, job_id in FULL_REPLICATE_PATHWAYS.items()
        },
    }
    for name, pathway in locations.items():
        prepare_topology(pathway, topology_root)
        result = analyze_pathway(pathway, workspace, output, args.gmx, topology_root)
        new_results[name] = result
        pathway_rows.append({
            "pathway": name,
            "state": result.metadata["state"],
            "system": result.metadata["system"],
            "route_family": result.metadata["route_family"],
            "source_replica": result.metadata["replica"],
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

    original_rows = {row["pathway"]: row for row in read_csv(args.original_summary.resolve())}

    def item_from_row(row: Mapping[str, str]) -> dict[str, object]:
        return {
            "metrics": aggregate([metric_set_from_row(row)]),
            "half_metrics": {
                half: aggregate([metric_set_from_row(row, half)])
                for half in ("first_half", "second_half")
            },
        }

    def item_from_result(result: PathwayResult) -> dict[str, object]:
        return {
            "metrics": aggregate([metric_set_from_result(result)]),
            "half_metrics": {
                half: aggregate([metric_set_from_result(result, half)])
                for half in ("first_half", "second_half")
            },
        }

    path_items = {
        **{name: item_from_row(row) for name, row in original_rows.items()},
        **{name: item_from_result(result) for name, result in new_results.items()},
    }
    categories: dict[str, dict[str, dict[str, object]]] = {}
    category_rows: list[dict[str, object]] = []
    for system in ("C6", "C7"):
        categories[system] = {}
        for category in ("full_reuse", "full_switch"):
            name = ORIGINAL_NAMES[system][category]
            row = original_rows[name]
            replicate = new_results[FULL_REPLICATE_NAMES[system][category]]
            category_result = {
                "paths": (name, FULL_REPLICATE_NAMES[system][category]),
                "metrics": aggregate([
                    metric_set_from_row(row), metric_set_from_result(replicate)
                ]),
                "half_metrics": {
                    half: aggregate([
                        metric_set_from_row(row, half),
                        metric_set_from_result(replicate, half),
                    ])
                    for half in ("first_half", "second_half")
                },
            }
            categories[system][category] = category_result
        names = PARTIAL_NAMES[system]
        category_result = {
            "paths": names,
            "metrics": aggregate([metric_set_from_result(new_results[name]) for name in names]),
            "half_metrics": {
                half: aggregate([
                    metric_set_from_result(new_results[name], half) for name in names
                ])
                for half in ("first_half", "second_half")
            },
        }
        categories[system]["partial_reuse"] = category_result

        for category in ("full_reuse", "partial_reuse", "full_switch"):
            item = categories[system][category]
            category_rows.append({
                "system": system,
                "category": category,
                "pathways": ";".join(item["paths"]),
                "pathway_count": len(item["paths"]),
                **item["metrics"],
                **{
                    f"{half}_{key}": value
                    for half, values in item["half_metrics"].items() for key, value in values.items()
                },
            })

    contrasts: list[dict[str, object]] = []
    for system, groups in categories.items():
        contrasts.extend([
            compare(system, "full_reuse", "partial_reuse",
                    groups["full_reuse"], groups["partial_reuse"]),
            compare(system, "partial_reuse", "full_switch",
                    groups["partial_reuse"], groups["full_switch"]),
            compare(system, "full_reuse", "full_switch",
                    groups["full_reuse"], groups["full_switch"]),
        ])

    endpoint_contrasts: list[dict[str, object]] = []
    for (system, replica), members in ENDPOINT_CLASS_PATHS.items():
        for left_name, right_name in (
            ("full_reuse", "partial_reuse"),
            ("partial_reuse", "full_switch"),
            ("full_reuse", "full_switch"),
        ):
            row = compare(
                system, left_name, right_name,
                path_items[members[left_name]], path_items[members[right_name]],
            )
            row["source_replica"] = replica
            row["left_pathway"] = members[left_name]
            row["right_pathway"] = members[right_name]
            endpoint_contrasts.append(row)

    write_csv(output / "new_pathway_reweight_summary.csv", pathway_rows)
    write_csv(
        output / "partial_pathway_reweight_summary.csv",
        [row for row in pathway_rows if row["route_family"] == "partial_reuse"],
    )
    write_csv(output / "category_reweight_summary.csv", category_rows)
    write_csv(output / "q1_three_class_2d_contrasts.csv", contrasts)
    write_csv(output / "q1_paired_endpoint_contrasts.csv", endpoint_contrasts)
    plot_contrasts(output, contrasts)
    summary = {
        "temperature_K": 313.15,
        "method": "distance-bias WHAM/MBAR-equivalent frame reweighting plus the original soft orientation gate",
        "scope": "each q1 category is an equal-path average of two independent unbiased-MD endpoints",
        "new_pathways": pathway_rows,
        "categories": category_rows,
        "contrasts": contrasts,
        "paired_endpoint_contrasts": endpoint_contrasts,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(contrasts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
