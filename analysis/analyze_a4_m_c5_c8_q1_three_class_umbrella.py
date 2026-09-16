#!/usr/bin/env python3
"""Analyze the completed A4-m-C5/C8 q1 three-class umbrella calculations.

The twelve pathways are two independent replicas for each of the three q1
classes (full reuse, partial reuse and full switch) for C5 and C8.  Each path
is reprocessed with the same WHAM/contact definition used for the earlier
C6/C7 analysis, then the two replicas of a class are pooled in contact
probability space.  The pooled values are thermodynamic diagnostics; unbiased
MD branch populations remain a separate quantity.
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

from analysis.analyze_a4_m_c6_c7_q1_three_class_umbrella import (  # noqa: E402
    build_category,
    contrast,
)
from analysis.analyze_a4_m_c6_c7_umbrella_final import (  # noqa: E402
    RT,
    load_pathway,
    recompute_wham,
    write_csv,
)


JOBS = {
    "c5_q1_full_reuse_v1_v6_rep1": ("20730620", "c5_q1_full_reuse_v1_v6"),
    "c5_q1_full_reuse_v1_v6_rep2": ("20730628", "c5_q1_full_reuse_v1_v6"),
    "c5_q1_partial_reuse_v1_v5_rep1": ("20730619", "c5_q1_partial_reuse_v1_v5"),
    "c5_q1_partial_reuse_v1_v5_rep2": ("20730627", "c5_q1_partial_reuse_v1_v5"),
    "c5_q1_full_switch_v2_v4_rep1": ("20730629", "c5_q1_full_switch_v2_v4"),
    "c5_q1_full_switch_v2_v4_rep2": ("20730622", "c5_q1_full_switch_v2_v4"),
    "c8_q1_full_reuse_v5_v7_rep1": ("20730624", "c8_q1_full_reuse_v5_v7"),
    "c8_q1_full_reuse_v5_v7_rep2": ("20730625", "c8_q1_full_reuse_v5_v7"),
    "c8_q1_partial_reuse_v1_v5_rep1": ("20730626", "c8_q1_partial_reuse_v1_v5"),
    "c8_q1_partial_reuse_v1_v5_rep2": ("20730618", "c8_q1_partial_reuse_v1_v5"),
    "c8_q1_full_switch_v1_v3_rep1": ("20730623", "c8_q1_full_switch_v1_v3"),
    "c8_q1_full_switch_v1_v3_rep2": ("20730621", "c8_q1_full_switch_v1_v3"),
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


def bootstrap_difference(left: np.ndarray, right: np.ndarray,
                         maximum_pairs: int = 2_000_000) -> np.ndarray:
    pair_count = len(left) * len(right)
    if pair_count <= maximum_pairs:
        return (left[:, None] - right[None, :]).ravel()
    rng = np.random.default_rng(20260912 + len(left) + 3 * len(right))
    return left[rng.integers(0, len(left), maximum_pairs)] - right[
        rng.integers(0, len(right), maximum_pairs)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_download")
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_analysis")
    parser.add_argument("--gmx", default="gmx")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    wham_root = output / "wham_recomputed"
    output.mkdir(parents=True, exist_ok=True)

    pathways = {}
    for alias, (job_id, _expected_base_name) in JOBS.items():
        pathway = root / job_id / "extracted/pathway"
        metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
        if metadata.get("pathway") != alias:
            raise ValueError(f"{alias}: expected {alias}, found {metadata.get('pathway')}")
        pathways[alias] = pathway

    records = []
    for alias in sorted(pathways):
        records.append(recompute_wham(pathways[alias], wham_root / alias, args.gmx))
        print(f"recomputed {alias}", flush=True)
    (output / "recompute_manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    loaded = {alias: load_pathway(pathway, wham_root / alias)
              for alias, pathway in pathways.items()}
    pathway_rows = []
    for alias in sorted(loaded):
        item = loaded[alias]
        metadata = item["metadata"]
        low, median, high = item["bootstrap_ci"]
        pathway_rows.append({
            "analysis_alias": alias,
            "pathway": metadata["pathway"],
            "system": metadata["system"],
            "route_family": metadata["route_family"],
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

    categories = {}
    category_rows = []
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

    comparisons = []
    for system, groups in categories.items():
        for left, right in (("full_reuse", "partial_reuse"),
                            ("partial_reuse", "full_switch"),
                            ("full_reuse", "full_switch")):
            comparisons.append(contrast(system, left, right, groups[left], groups[right]))

    sensitivity = []
    for minimum_count in (1.0, 5.0, 20.0):
        for split_nm in (0.45, 0.50, 0.55):
            values = {alias: load_pathway(pathways[alias], wham_root / alias,
                                          minimum_count, split_nm)
                      for alias in pathways}
            for system, groups in CATEGORIES.items():
                pooled = {name: build_category([values[alias] for alias in aliases])
                          for name, aliases in groups.items()}
                for left, right in (("full_reuse", "partial_reuse"),
                                    ("partial_reuse", "full_switch"),
                                    ("full_reuse", "full_switch")):
                    ddg = float(pooled[left]["dg"] - pooled[right]["dg"])
                    sensitivity.append({
                        "system": system,
                        "definition": f"{left} minus {right}",
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
        "method_note": "two independent replicas pooled by equal-weight contact probability; this is a PMF/path thermodynamic comparison and does not replace unbiased q1 branch populations",
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
