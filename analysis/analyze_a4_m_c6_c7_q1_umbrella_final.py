#!/usr/bin/env python3
"""Analyze the completed A4-m-C6/C7 q1 reuse/switch umbrella paths.

This follows the validated C5/C8 one-dimensional umbrella protocol: WHAM is
recomputed on each observed coordinate range, contact odds use only
histogram-supported bins, and uncertainty comes from trajectory-bootstrap
WHAM profiles.  Results describe the four preselected representative paths,
not an exhaustive average over every A4 cross-arm pair.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.analyze_a4_m_c6_c7_umbrella_final import (  # noqa: E402
    RT,
    contrast_row,
    load_pathway,
    recompute_wham,
    write_csv,
)


JOB_TO_PATHWAY = {
    "20726912": "c6_q1_reuse_v1_v7",
    "20726913": "c7_q1_reuse_v2_v5",
    "20726914": "c6_q1_switch_v3_v4",
    "20726915": "c7_q1_switch_v1_v8",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument(
        "--download-root", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_umbrella_v2_download",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_umbrella_analysis",
    )
    args = parser.parse_args()
    download_root = args.download_root.resolve()
    output = args.output.resolve()
    wham_root = output / "wham_recomputed"
    output.mkdir(parents=True, exist_ok=True)

    pathways: dict[str, Path] = {}
    for job_id, expected_name in JOB_TO_PATHWAY.items():
        pathway = download_root / job_id / "extracted/pathway"
        metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
        name = str(metadata["pathway"])
        if name != expected_name:
            raise ValueError(f"{job_id}: expected {expected_name}, found {name}")
        pathways[name] = pathway

    recompute_records = []
    for name, pathway in sorted(pathways.items()):
        recompute_records.append(recompute_wham(pathway, wham_root / name, args.gmx))
        print(f"recomputed {name}", flush=True)
    (output / "recompute_manifest.json").write_text(
        json.dumps(recompute_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded = {
        name: load_pathway(pathway, wham_root / name)
        for name, pathway in pathways.items()
    }
    pathway_rows: list[dict[str, object]] = []
    for name, item in sorted(loaded.items()):
        metadata = item["metadata"]
        low, median, high = item["bootstrap_ci"]
        pathway_rows.append({
            "pathway": name,
            "system": metadata["system"],
            "stage": metadata["stage"],
            "route_family": metadata["route_family"],
            "topology": metadata["topology"],
            "pair": f"V{metadata['left']}-V{metadata['right']}",
            "source_replica": metadata["replica"],
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

    specs = (
        ("C6_q1", "c6_q1_reuse_v1_v7", "c6_q1_switch_v3_v4"),
        ("C7_q1", "c7_q1_reuse_v2_v5", "c7_q1_switch_v1_v8"),
    )
    contrasts = [
        contrast_row(label, "reuse minus switch", loaded[reuse], loaded[switch])
        for label, reuse, switch in specs
    ]
    for row in contrasts:
        row["alpha_umbrella"] = float(row["point_ddG_kcal_mol"]) / RT

    sensitivity: list[dict[str, object]] = []
    for minimum_count in (1.0, 5.0, 20.0):
        for split_nm in (0.45, 0.50, 0.55):
            values = {
                name: load_pathway(pathways[name], wham_root / name, minimum_count, split_nm)["dg"]
                for name in pathways
            }
            for label, reuse, switch in specs:
                ddg = float(values[reuse] - values[switch])
                sensitivity.append({
                    "quantity": label,
                    "definition": "reuse minus switch",
                    "minimum_histogram_count": minimum_count,
                    "contact_split_nm": split_nm,
                    "ddG_kcal_mol": ddg,
                    "reuse_over_switch_contact_odds": math.exp(-ddg / RT),
                    "alpha_umbrella": ddg / RT,
                })

    write_csv(output / "pathway_contact_free_energies.csv", pathway_rows)
    write_csv(output / "selectivity_contrasts.csv", contrasts)
    write_csv(output / "contact_definition_sensitivity.csv", sensitivity)
    summary = {
        "temperature_K": 313.15,
        "contact_split_nm": 0.50,
        "minimum_histogram_count": 5.0,
        "scope": "four prescreened representative q1 paths; A4 has no opposite-arm exclusion",
        "pathways": pathway_rows,
        "contrasts": contrasts,
        "sensitivity": sensitivity,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pathways": len(pathway_rows), "contrasts": contrasts,
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
