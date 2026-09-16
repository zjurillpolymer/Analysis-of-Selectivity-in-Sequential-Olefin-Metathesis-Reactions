#!/usr/bin/env python3
"""Prepare third independent C5/C8 q1 umbrella endpoints.

The first two endpoint replicas are already represented in the completed q1
umbrella set.  This supplement adds replica 3 for each of the three q1
history classes, using the same distance-umbrella protocol and the same
validated topology construction.  The resulting trajectories are analysed
with the post-hoc distance/orientation two-dimensional reweighting script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.prepare_a4_m_c5_c8_q1_three_class_umbrella import (
    find_replica,
    package_job,
    prepare_pathway,
)


PATHWAYS = (
    {
        "pathway": "c5_q1_full_reuse_v1_v6_rep3",
        "system": "A4_m_C5",
        "state": "A4_m_C5_1_cis",
        "route_family": "full_reuse",
        "history_class": "reuse_arm_pair",
        "left": 1,
        "right": 6,
        "replica": 3,
        "selection_reason": "third independent endpoint for the unique q0-reacted arm pair",
    },
    {
        "pathway": "c5_q1_partial_reuse_v1_v5_rep3",
        "system": "A4_m_C5",
        "state": "A4_m_C5_1_cis",
        "route_family": "partial_reuse",
        "history_class": "one_reacted_arm",
        "left": 1,
        "right": 5,
        "replica": 3,
        "selection_reason": "third independent endpoint for the validated partial-reuse pair",
    },
    {
        "pathway": "c5_q1_full_switch_v2_v4_rep3",
        "system": "A4_m_C5",
        "state": "A4_m_C5_1_cis",
        "route_family": "full_switch",
        "history_class": "untouched_arm_pair",
        "left": 2,
        "right": 4,
        "replica": 3,
        "selection_reason": "third independent endpoint for the validated untouched-arm pair",
    },
    {
        "pathway": "c8_q1_full_reuse_v5_v7_rep3",
        "system": "A4_m_C8",
        "state": "A4_m_C8_1_cis",
        "route_family": "full_reuse",
        "history_class": "reuse_arm_pair",
        "left": 5,
        "right": 7,
        "replica": 3,
        "selection_reason": "third independent endpoint for the unique q0-reacted arm pair",
    },
    {
        "pathway": "c8_q1_partial_reuse_v1_v5_rep3",
        "system": "A4_m_C8",
        "state": "A4_m_C8_1_cis",
        "route_family": "partial_reuse",
        "history_class": "one_reacted_arm",
        "left": 1,
        "right": 5,
        "replica": 3,
        "selection_reason": "third independent endpoint for the validated partial-reuse pair",
    },
    {
        "pathway": "c8_q1_full_switch_v1_v3_rep3",
        "system": "A4_m_C8",
        "state": "A4_m_C8_1_cis",
        "route_family": "full_switch",
        "history_class": "untouched_arm_pair",
        "left": 1,
        "right": 3,
        "replica": 3,
        "selection_reason": "third independent endpoint for the validated untouched-arm pair",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-root",
        type=Path,
        default=PROJECT / "results/a4_m_c5_c8_q1_10ns_download",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=PROJECT / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "umbrella/a4_m_c5_c8_q1_three_class_umbrella_rep3",
    )
    parser.add_argument(
        "--job-root",
        type=Path,
        default=PROJECT / "bohrium_submit/a4_m_c5_c8_q1_three_class_umbrella_rep3",
    )
    args = parser.parse_args()
    download_root, run_root, output, job_root = (
        path.resolve() for path in (args.download_root, args.run_root, args.output, args.job_root)
    )
    if output.exists() or job_root.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {job_root}")
    output.mkdir(parents=True)
    job_root.mkdir(parents=True)

    records = []
    for config in PATHWAYS:
        record = prepare_pathway(output, download_root, run_root, dict(config))
        package_job(job_root, output, record)
        records.append(record)

    manifest = {
        "format": "a4-m-c5-c8-q1-three-class-umbrella-rep3-v1",
        "project_id": 4377225,
        "dispatch": "one NVIDIA T4 job per independent q1 endpoint",
        "pathway_count": len(records),
        "paths_per_system": {"full_reuse": 1, "partial_reuse": 1, "full_switch": 1},
        "protocol": "same distance umbrella as the completed q1 set; analyse with 2D distance/orientation reweighting",
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
