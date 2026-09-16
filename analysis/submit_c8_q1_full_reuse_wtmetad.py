#!/usr/bin/env python3
"""Submit the three targeted C8 q1 full-reuse WT-MetaD replicas."""

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
    root_default = Path(__file__).resolve().parents[1] / "bohrium_submit/c8_q1_full_reuse_wtmetad"
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--replicas", nargs="+", choices=("rep1", "rep2", "rep3"))
    args = parser.parse_args()
    root = args.root.resolve()
    tasks = sorted(path.parent for path in root.glob("*/job.json"))
    if args.replicas:
        selected = set(args.replicas)
        tasks = [task for task in tasks if task.name in selected]
    if not tasks:
        raise SystemExit(f"no jobs found under {root}")
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(submit, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as exc:
                failures.append(f"{task.name}: {exc}")
                print(f"ERROR {task.name}: {exc}", flush=True)
    records = [json.loads(path.read_text(encoding="utf-8")) for path in root.glob("*/submission.json")]
    summary = {
        "jobs_expected": 3,
        "jobs_selected_this_call": len(tasks),
        "jobs_submitted_or_existing": len(records),
        "failures": failures,
        "records": sorted(records, key=lambda row: str(row["task"])),
    }
    (root / "submission_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
