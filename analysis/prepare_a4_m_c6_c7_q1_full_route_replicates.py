#!/usr/bin/env python3
"""Prepare independent-endpoint q1 full-reuse/full-switch umbrella replicates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.prepare_a4_m_c6_c7_q1_umbrella import prepare_pathway
from analysis.prepare_a4_m_c6_c7_missing_umbrella import package_job


PATHWAYS = (
    {
        "pathway": "c6_q1_full_reuse_v1_v7_rep2",
        "system": "A4_m_C6",
        "state": "A4_m_C6_1_cis",
        "comparison": "q1_three_class_independent_endpoint",
        "route_family": "full_reuse",
        "history_class": "reuse_arm_pair",
        "topology": "cross_arm",
        "left": 1,
        "right": 7,
        "replica": 2,
        "selection_reason": "same full-reuse pair as the existing path, independent unbiased q1 endpoint",
    },
    {
        "pathway": "c6_q1_full_switch_v3_v4_rep2",
        "system": "A4_m_C6",
        "state": "A4_m_C6_1_cis",
        "comparison": "q1_three_class_independent_endpoint",
        "route_family": "full_switch",
        "history_class": "untouched_arm_pair",
        "topology": "cross_arm",
        "left": 3,
        "right": 4,
        "replica": 2,
        "selection_reason": "same full-switch pair as the existing path, independent unbiased q1 endpoint",
    },
    {
        "pathway": "c7_q1_full_reuse_v2_v5_rep1",
        "system": "A4_m_C7",
        "state": "A4_m_C7_1_cis",
        "comparison": "q1_three_class_independent_endpoint",
        "route_family": "full_reuse",
        "history_class": "reuse_arm_pair",
        "topology": "cross_arm",
        "left": 2,
        "right": 5,
        "replica": 1,
        "selection_reason": "same full-reuse pair as the existing path, independent unbiased q1 endpoint",
    },
    {
        "pathway": "c7_q1_full_switch_v1_v8_rep1",
        "system": "A4_m_C7",
        "state": "A4_m_C7_1_cis",
        "comparison": "q1_three_class_independent_endpoint",
        "route_family": "full_switch",
        "history_class": "untouched_arm_pair",
        "topology": "cross_arm",
        "left": 1,
        "right": 8,
        "replica": 1,
        "selection_reason": "same full-switch pair as the existing path, independent unbiased q1 endpoint",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=PROJECT / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs",
    )
    parser.add_argument(
        "--download-root", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_q1_10ns_download",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "umbrella/a4_m_c6_c7_q1_full_route_replicates",
    )
    parser.add_argument(
        "--job-root", type=Path,
        default=PROJECT / "bohrium_submit/a4_m_c6_c7_q1_full_route_replicates",
    )
    args = parser.parse_args()
    output, job_root = args.output.resolve(), args.job_root.resolve()
    if output.exists() or job_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output} or {job_root}")
    output.mkdir(parents=True)
    job_root.mkdir(parents=True)

    records = []
    for config in PATHWAYS:
        record = prepare_pathway(
            output,
            args.run_root.resolve(),
            args.download_root.resolve(),
            dict(config),
        )
        package_job(job_root, output, record)
        records.append(record)

    manifest = {
        "format": "a4-m-c6-c7-q1-full-route-independent-endpoints-v1",
        "project_id": 4377225,
        "dispatch": "one NVIDIA T4 job per independent q1 endpoint",
        "purpose": "balance endpoint coverage across full reuse, partial reuse, and full switch",
        "pathway_count": len(records),
        "total_window_sampling_ns": sum(
            int(row["window_count"])
            * (float(row["equilibration_ps_per_window"]) + float(row["production_ps_per_window"]))
            / 1000.0
            for row in records
        ),
        "pathways": records,
    }
    (output / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (job_root / "jobs.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
