#!/usr/bin/env python3
"""Submit the four prepared A4-m-C6/C7 q1 umbrella pathways."""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def submit(task: Path) -> dict[str, object]:
    record_path = task / "submission.json"
    if record_path.is_file():
        return json.loads(record_path.read_text(encoding="utf-8"))
    result = subprocess.run(
        ["bohr", "job", "submit", "-i", str(task / "job.json"), "--yes", "--output", "json"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"{task.name}: {result.stderr or result.stdout}")
    response = json.loads(result.stdout)
    if not response.get("ok"):
        raise RuntimeError(f"{task.name}: {response}")
    record = {
        "task": task.name,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitted",
        "bohr_job_id": int(response["data"]["bohrJobId"]),
        "internal_job_id": int(response["data"]["jobId"]),
        "bohr_job_group_id": int(response["data"]["bohrJobGroupId"]),
        "internal_job_group_id": int(response["data"]["jobGroupId"]),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project / "bohrium_submit/a4_m_c6_c7_q1_umbrella")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    tasks = sorted(path.parent for path in root.glob("*/job.json"))
    if len(tasks) != 4:
        raise RuntimeError(f"expected 4 prepared jobs, found {len(tasks)}")
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(submit, task): task for task in tasks}
        for future in as_completed(futures):
            try:
                record = future.result()
                records.append(record)
                print(json.dumps(record), flush=True)
            except Exception as exc:
                failures.append(str(exc))
                print(f"ERROR {exc}", flush=True)
    summary = {
        "jobs_expected": len(tasks),
        "jobs_submitted_or_existing": len(records),
        "failures": failures,
        "records": sorted(records, key=lambda row: str(row["task"])),
    }
    (root / "submission_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
