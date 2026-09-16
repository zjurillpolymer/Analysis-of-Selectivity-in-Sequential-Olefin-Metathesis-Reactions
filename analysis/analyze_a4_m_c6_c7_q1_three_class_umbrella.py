#!/usr/bin/env python3
"""Analyze A4-m-C6/C7 q1 full-reuse, partial-reuse, and full-switch PMFs.

The full-reuse/full-switch paths are the previously completed representative
umbrella calculations.  Partial reuse is represented by two independently
selected q1 endpoints per spacer.  All paths are reprocessed with the same
one-dimensional WHAM/contact protocol used in the C5/C8 summary.

This script deliberately keeps two concepts separate:

1. per-path contact thermodynamics from umbrella sampling; and
2. class multiplicity / whole-ensemble population from unbiased MD.

Accordingly, the equal-path partial-reuse pool below is a reproducibility
summary of the two umbrella paths, not a replacement for q1 branch weights.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.analyze_a4_m_c6_c7_umbrella_final import (  # noqa: E402
    RT,
    load_pathway,
    recompute_wham,
    write_csv,
)
from analysis.analyze_a4_m_c5_c8_umbrella_comparison import q025_50_975  # noqa: E402


ORIGINAL = {
    "c6_q1_full_reuse_v1_v7": ("20726912", "c6_q1_reuse_v1_v7"),
    "c6_q1_full_switch_v3_v4": ("20726914", "c6_q1_switch_v3_v4"),
    "c7_q1_full_reuse_v2_v5": ("20726913", "c7_q1_reuse_v2_v5"),
    "c7_q1_full_switch_v1_v8": ("20726915", "c7_q1_switch_v1_v8"),
}

PARTIAL = {
    "c6_q1_partial_reuse_v1_v5_rep1": ("20729814", "c6_q1_partial_reuse_v1_v5_rep1"),
    "c6_q1_partial_reuse_v1_v5_rep2": ("20729811", "c6_q1_partial_reuse_v1_v5_rep2"),
    "c7_q1_partial_reuse_v5_v6_rep1": ("20729812", "c7_q1_partial_reuse_v5_v6_rep1"),
    "c7_q1_partial_reuse_v5_v6_rep2": ("20729813", "c7_q1_partial_reuse_v5_v6_rep2"),
}

FULL_REPLICATES = {
    "c6_q1_full_reuse_v1_v7_rep2": ("20729862", "c6_q1_full_reuse_v1_v7_rep2"),
    "c6_q1_full_switch_v3_v4_rep2": ("20729860", "c6_q1_full_switch_v3_v4_rep2"),
    "c7_q1_full_reuse_v2_v5_rep1": ("20729863", "c7_q1_full_reuse_v2_v5_rep1"),
    "c7_q1_full_switch_v1_v8_rep1": ("20729861", "c7_q1_full_switch_v1_v8_rep1"),
}

CATEGORIES = {
    "C6": {
        "full_reuse": (
            "c6_q1_full_reuse_v1_v7",
            "c6_q1_full_reuse_v1_v7_rep2",
        ),
        "partial_reuse": (
            "c6_q1_partial_reuse_v1_v5_rep1",
            "c6_q1_partial_reuse_v1_v5_rep2",
        ),
        "full_switch": (
            "c6_q1_full_switch_v3_v4",
            "c6_q1_full_switch_v3_v4_rep2",
        ),
    },
    "C7": {
        "full_reuse": (
            "c7_q1_full_reuse_v2_v5",
            "c7_q1_full_reuse_v2_v5_rep1",
        ),
        "partial_reuse": (
            "c7_q1_partial_reuse_v5_v6_rep1",
            "c7_q1_partial_reuse_v5_v6_rep2",
        ),
        "full_switch": (
            "c7_q1_full_switch_v1_v8",
            "c7_q1_full_switch_v1_v8_rep1",
        ),
    },
}


def probability_from_dg(dg: np.ndarray | float) -> np.ndarray:
    """Convert contact-vs-noncontact free energy to contact probability."""
    values = np.asarray(dg, dtype=float)
    scaled = np.clip(values / RT, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(scaled))


def dg_from_probability(probability: np.ndarray | float) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-15, 1.0 - 1e-15)
    return -RT * np.log(values / (1.0 - values))


def pool_point(items: list[dict[str, object]], key: str) -> float:
    probabilities = [float(probability_from_dg(float(item[key]))) for item in items]
    return float(dg_from_probability(np.mean(probabilities)))


def pool_bootstrap(items: list[dict[str, object]]) -> np.ndarray:
    arrays = [np.asarray(item["bootstrap_dg"], dtype=float) for item in items]
    if len(arrays) == 1:
        return arrays[0]
    if len(arrays) != 2:
        raise ValueError("current exact pooling implementation expects one or two paths")
    first = probability_from_dg(arrays[0])[:, None]
    second = probability_from_dg(arrays[1])[None, :]
    return dg_from_probability(0.5 * (first + second)).ravel()


def build_category(items: list[dict[str, object]]) -> dict[str, object]:
    distribution = pool_bootstrap(items)
    low, median, high = q025_50_975(distribution)
    return {
        "dg": pool_point(items, "dg"),
        "dg_first": pool_point(items, "dg_first"),
        "dg_second": pool_point(items, "dg_second"),
        "bootstrap_dg": distribution,
        "bootstrap_ci": (low, median, high),
        "path_count": len(items),
    }


def bootstrap_difference(left: np.ndarray, right: np.ndarray,
                         maximum_pairs: int = 2_000_000) -> np.ndarray:
    """Return an exact or reproducibly sampled independent-bootstrap contrast."""
    pair_count = len(left) * len(right)
    if pair_count <= maximum_pairs:
        return (left[:, None] - right[None, :]).ravel()
    rng = np.random.default_rng(20260912 + len(left) + 3 * len(right))
    left_index = rng.integers(0, len(left), size=maximum_pairs)
    right_index = rng.integers(0, len(right), size=maximum_pairs)
    return left[left_index] - right[right_index]


def contrast(label: str, left_name: str, right_name: str,
             left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    point = float(left["dg"] - right["dg"])
    distribution = bootstrap_difference(
        np.asarray(left["bootstrap_dg"], dtype=float),
        np.asarray(right["bootstrap_dg"], dtype=float),
    )
    low, median, high = q025_50_975(distribution)
    return {
        "system": label,
        "definition": f"{left_name} minus {right_name}",
        "left_category": left_name,
        "right_category": right_name,
        "point_ddG_kcal_mol": point,
        "bootstrap_median_kcal_mol": median,
        "bootstrap_ci_low_kcal_mol": low,
        "bootstrap_ci_high_kcal_mol": high,
        "left_over_right_contact_odds": math.exp(-point / RT),
        "probability_ddG_below_zero": float(np.mean(distribution < 0.0)),
        "first_half_ddG_kcal_mol": float(left["dg_first"] - right["dg_first"]),
        "second_half_ddG_kcal_mol": float(left["dg_second"] - right["dg_second"]),
        "alpha_contact": point / RT,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument(
        "--original-root", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_umbrella_v2_download",
    )
    parser.add_argument(
        "--partial-root", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_partial_reuse_umbrella_download",
    )
    parser.add_argument(
        "--full-replicate-root", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_full_route_replicates_download",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_three_class_umbrella_analysis",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    wham_root = output / "wham_recomputed"
    output.mkdir(parents=True, exist_ok=True)

    pathways: dict[str, Path] = {}
    expected: dict[str, str] = {}
    for alias, (job_id, pathway_name) in ORIGINAL.items():
        pathways[alias] = args.original_root.resolve() / job_id / "extracted/pathway"
        expected[alias] = pathway_name
    for alias, (job_id, pathway_name) in PARTIAL.items():
        pathways[alias] = args.partial_root.resolve() / job_id / "extracted/pathway"
        expected[alias] = pathway_name
    for alias, (job_id, pathway_name) in FULL_REPLICATES.items():
        pathways[alias] = args.full_replicate_root.resolve() / job_id / "extracted/pathway"
        expected[alias] = pathway_name

    for alias, pathway in pathways.items():
        metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
        if metadata["pathway"] != expected[alias]:
            raise ValueError(f"{alias}: expected {expected[alias]}, found {metadata['pathway']}")

    recompute_records = []
    for alias, pathway in sorted(pathways.items()):
        recompute_records.append(recompute_wham(pathway, wham_root / alias, args.gmx))
        print(f"recomputed {alias}", flush=True)
    (output / "recompute_manifest.json").write_text(
        json.dumps(recompute_records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    loaded = {
        alias: load_pathway(pathway, wham_root / alias)
        for alias, pathway in pathways.items()
    }
    pathway_rows: list[dict[str, object]] = []
    for alias, item in sorted(loaded.items()):
        metadata = item["metadata"]
        low, median, high = item["bootstrap_ci"]
        pathway_rows.append({
            "analysis_alias": alias,
            "pathway": metadata["pathway"],
            "system": metadata["system"],
            "route_family": (
                metadata["route_family"] if metadata["route_family"] != "reuse" else "full_reuse"
            ),
            "pair": f"V{metadata['left']}-V{metadata['right']}",
            "source_replica": metadata.get("replica", ""),
            "contact_dG_kcal_mol": item["dg"],
            "bootstrap_median_kcal_mol": median,
            "bootstrap_ci_low_kcal_mol": low,
            "bootstrap_ci_high_kcal_mol": high,
            "first_half_contact_dG_kcal_mol": item["dg_first"],
            "second_half_contact_dG_kcal_mol": item["dg_second"],
            "support_min_nm": item["support_min_nm"],
            "support_max_nm": item["support_max_nm"],
            "minimum_neighbor_overlap": item["minimum_neighbor_overlap"],
            "maximum_mean_center_error_nm": item["maximum_mean_center_error_nm"],
            "window_count": item["window_count"],
        })

    categories: dict[str, dict[str, dict[str, object]]] = {}
    category_rows: list[dict[str, object]] = []
    for system, groups in CATEGORIES.items():
        categories[system] = {}
        for category, aliases in groups.items():
            pooled = build_category([loaded[alias] for alias in aliases])
            categories[system][category] = pooled
            low, median, high = pooled["bootstrap_ci"]
            category_rows.append({
                "system": system,
                "category": category,
                "representative_paths": ";".join(aliases),
                "path_count": pooled["path_count"],
                "contact_dG_kcal_mol": pooled["dg"],
                "bootstrap_median_kcal_mol": median,
                "bootstrap_ci_low_kcal_mol": low,
                "bootstrap_ci_high_kcal_mol": high,
                "first_half_contact_dG_kcal_mol": pooled["dg_first"],
                "second_half_contact_dG_kcal_mol": pooled["dg_second"],
            })

    comparisons: list[dict[str, object]] = []
    for system, groups in categories.items():
        comparisons.extend([
            contrast(system, "full_reuse", "partial_reuse",
                     groups["full_reuse"], groups["partial_reuse"]),
            contrast(system, "partial_reuse", "full_switch",
                     groups["partial_reuse"], groups["full_switch"]),
            contrast(system, "full_reuse", "full_switch",
                     groups["full_reuse"], groups["full_switch"]),
        ])

    sensitivity: list[dict[str, object]] = []
    for minimum_count in (1.0, 5.0, 20.0):
        for split_nm in (0.45, 0.50, 0.55):
            values = {
                alias: load_pathway(pathways[alias], wham_root / alias, minimum_count, split_nm)
                for alias in pathways
            }
            for system, groups in CATEGORIES.items():
                pooled = {
                    name: build_category([values[alias] for alias in aliases])
                    for name, aliases in groups.items()
                }
                for left_name, right_name in (
                    ("full_reuse", "partial_reuse"),
                    ("partial_reuse", "full_switch"),
                    ("full_reuse", "full_switch"),
                ):
                    ddg = float(pooled[left_name]["dg"] - pooled[right_name]["dg"])
                    sensitivity.append({
                        "system": system,
                        "definition": f"{left_name} minus {right_name}",
                        "minimum_histogram_count": minimum_count,
                        "contact_split_nm": split_nm,
                        "ddG_kcal_mol": ddg,
                        "left_over_right_contact_odds": math.exp(-ddg / RT),
                    })

    write_csv(output / "pathway_contact_free_energies.csv", pathway_rows)
    write_csv(output / "category_contact_free_energies.csv", category_rows)
    write_csv(output / "three_class_contrasts.csv", comparisons)
    write_csv(output / "contact_definition_sensitivity.csv", sensitivity)
    summary = {
        "temperature_K": 313.15,
        "contact_split_nm": 0.50,
        "minimum_histogram_count": 5.0,
        "method_note": (
            "partial_reuse pools two independently selected representative paths by equal-weight "
            "contact probability; class multiplicity and whole-ensemble branch populations are not included"
        ),
        "pathways": pathway_rows,
        "categories": category_rows,
        "contrasts": comparisons,
        "sensitivity": sensitivity,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pathways": len(pathway_rows), "contrasts": comparisons,
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
