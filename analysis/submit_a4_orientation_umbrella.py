#!/usr/bin/env python3
"""Submit the targeted A4 orientation-umbrella pilot jobs through Bohrium."""

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
        check=False, capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError(f"{task.name}: {result.stderr or result.stdout}")
    response = json.loads(result.stdout)
    if not response.get("ok"):
        raise RuntimeError(f"{task.name}: {response}")
    data = response["data"]
    record = {
        "task": task.name,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "status": "submitted",
        "bohr_job_id": int(data["bohrJobId"]),
        "internal_job_id": int(data["jobId"]),
        "bohr_job_group_id": int(data["bohrJobGroupId"]),
        "internal_job_group_id": int(data["jobGroupId"]),
    }
    record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project / "bohrium_submit/a4_orientation_umbrella_pilot")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--paths", nargs="+", help="Select task directory names")
    args = parser.parse_args()
    root = args.root.resolve()
    tasks = sorted(path.parent for path in root.glob("*/job.json"))
    if not tasks:
        raise RuntimeError(f"no prepared jobs under {root}")
    if args.paths:
        selected = set(args.paths)
        tasks = [task for task in tasks if task.name in selected]
        if len(tasks) != len(selected):
            raise ValueError("unknown selected task")
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
    existing = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("*/submission.json")]
    summary = {
        "jobs_expected": len(list(root.glob("*/job.json"))),
        "jobs_selected_this_call": len(tasks),
        "jobs_submitted_or_existing": len(existing),
        "failures": failures,
        "records": sorted(existing, key=lambda row: str(row["task"])),
    }
    (root / "submission_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
