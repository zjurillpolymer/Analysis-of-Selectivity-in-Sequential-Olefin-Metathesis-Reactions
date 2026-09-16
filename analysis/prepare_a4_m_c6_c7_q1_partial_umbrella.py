#!/usr/bin/env python3
"""Prepare q1 partial-reuse (adjacent) umbrella paths for A4-m-C6/C7.

The q1 all-pair analysis already separates the three relevant cross-arm
classes.  The existing q1 umbrella set covered full reuse (same reacted arm
pair) and full switch (the untouched arm pair), but not the intermediate
partial-reuse class.  These four jobs add two independently seeded endpoints
for one representative, replica-stable partial-reuse pair in each spacer.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.prepare_a4_m_c6_c7_q1_umbrella import (
    find_downloaded_replica,
    package_job,
    prepare_pathway,
)


PATHWAYS = (
    {
        "pathway": "c6_q1_partial_reuse_v1_v5_rep1",
        "system": "A4_m_C6",
        "state": "A4_m_C6_1_cis",
        "comparison": "partial_reuse_representative",
        "route_family": "partial_reuse",
        "history_class": "one_reacted_arm",
        "topology": "cross_arm",
        "left": 1,
        "right": 5,
        "replica": 1,
        "selection_reason": "high and replica-stable q1 one-reacted-arm NAC pair",
    },
    {
        "pathway": "c6_q1_partial_reuse_v1_v5_rep2",
        "system": "A4_m_C6",
        "state": "A4_m_C6_1_cis",
        "comparison": "partial_reuse_representative",
        "route_family": "partial_reuse",
        "history_class": "one_reacted_arm",
        "topology": "cross_arm",
        "left": 1,
        "right": 5,
        "replica": 2,
        "selection_reason": "high and replica-stable q1 one-reacted-arm NAC pair",
    },
    {
        "pathway": "c7_q1_partial_reuse_v5_v6_rep1",
        "system": "A4_m_C7",
        "state": "A4_m_C7_1_cis",
        "comparison": "partial_reuse_representative",
        "route_family": "partial_reuse",
        "history_class": "one_reacted_arm",
        "topology": "cross_arm",
        "left": 5,
        "right": 6,
        "replica": 1,
        "selection_reason": "highest and replica-stable q1 one-reacted-arm NAC pair",
    },
    {
        "pathway": "c7_q1_partial_reuse_v5_v6_rep2",
        "system": "A4_m_C7",
        "state": "A4_m_C7_1_cis",
        "comparison": "partial_reuse_representative",
        "route_family": "partial_reuse",
        "history_class": "one_reacted_arm",
        "topology": "cross_arm",
        "left": 5,
        "right": 6,
        "replica": 2,
        "selection_reason": "highest and replica-stable q1 one-reacted-arm NAC pair",
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
        default=PROJECT / "umbrella/a4_m_c6_c7_q1_partial_reuse_umbrella",
    )
    parser.add_argument(
        "--job-root", type=Path,
        default=PROJECT / "bohrium_submit/a4_m_c6_c7_q1_partial_reuse_umbrella",
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
        "format": "a4-m-c6-c7-q1-partial-reuse-umbrella-v1",
        "project_id": 4377225,
        "dispatch": "one NVIDIA T4 job per independent partial-reuse endpoint",
        "model_scope": "A4 all active cross-arm pairs; partial reuse means exactly one q0-reacted arm is reused",
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
