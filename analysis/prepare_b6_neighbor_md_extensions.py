#!/usr/bin/env python3
"""Prepare portable 2 -> 10 ns B6 neighbour-only checkpoint extensions."""
from __future__ import annotations
import argparse, csv, json, shutil
from pathlib import Path

STATES = {
    "B6_p_C6_q1_V1_V2_cis": 20748984,
    "B6_p_C6_q2_neighbor_full_reuse_cis": 20749009,
    "B6_p_C6_q2_neighbor_one_arm_reuse_cis": 20749047,
    "B6_p_C6_q2_neighbor_full_switch_d1_cis": 20749048,
    "B6_p_C6_q2_neighbor_full_switch_d2_cis": 20749051,
}

RUN_SCRIPT = r'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
exec > >(tee -a md_extension_job.log) 2>&1
status=0
archive_results() {{
  status=$?
  tar -czf {archive} workspace/gromacs_dcm/runs/{state} md_extension_job.log
  exit "$status"
}}
trap archive_results EXIT
export OMP_NUM_THREADS=4
nvidia-smi -L
gmx --version
python3 workspace/analysis/extend_explicit_dcm_q0.py \
  --root workspace --states {state} --replicas 3 --target-total-ns 10 --ntomp 4
for replica in 01 02 03; do
  test -s workspace/gromacs_dcm/runs/{state}/replica_${{replica}}/production_2ns.xtc
  test -s workspace/gromacs_dcm/runs/{state}/replica_${{replica}}/extension_metadata.json
  python3 -c 'import json, pathlib; p=pathlib.Path("workspace/gromacs_dcm/runs/{state}/replica_${{replica}}/extension_metadata.json"); assert json.loads(p.read_text())["target_total_ns"] == 10.0'
done
'''

def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", type=Path, default=project.parent / "bohrium_download/b6_neighbor_md_pilot_20260917")
    parser.add_argument("--output", type=Path, default=project.parent / "bohrium_submit/b6_neighbor_md_extensions_2to10ns_20260918")
    args = parser.parse_args()
    download, output = args.download.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    records = []
    for state, job_id in STATES.items():
        source_workspace = download / str(job_id) / "extracted/workspace"
        source_manifest = source_workspace / "gromacs_dcm/state_manifest.csv"
        with source_manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        row = next(item for item in rows if item["state"] == state)
        source_run = source_workspace / row["output_dir"]
        task, workspace = output / state, output / state / "workspace"
        copy_required(source_manifest, workspace / "gromacs_dcm/state_manifest.csv")
        copy_required(project / "analysis/extend_explicit_dcm_q0.py", workspace / "analysis/extend_explicit_dcm_q0.py")
        copy_required(project / "analysis/run_explicit_dcm_gromacs.py", workspace / "analysis/run_explicit_dcm_gromacs.py")
        target_run = workspace / row["output_dir"]
        for replica in range(1, 4):
            source_replica = source_run / f"replica_{replica:02d}"
            target_replica = target_run / f"replica_{replica:02d}"
            for suffix in (".tpr", ".cpt", ".xtc", ".edr", ".log", ".gro"):
                copy_required(source_replica / f"production_2ns{suffix}", target_replica / f"production_2ns{suffix}")
        archive = f"{state}_md_extension_results.tar.gz"
        run_script = task / "run_extension.sh"
        run_script.write_text(RUN_SCRIPT.format(state=state, archive=archive), encoding="utf-8")
        run_script.chmod(0o755)
        job = {
            "job_name": f"B6 neighbor {state} checkpoint extension 2to10ns",
            "command": "bash run_extension.sh", "log_file": "md_extension_job.log",
            "backward_files": [archive, "md_extension_job.log"], "project_id": 4377225,
            "machine_type": "c4_m15_1 * NVIDIA T4", "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
            "input_directory": str(task), "max_reschedule_times": 2, "max_run_time": 720, "nnode": 1,
        }
        (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        records.append({"state": state, "source_job": job_id, "task": str(task), "archive": archive})
    summary = {"status": "prepared_not_submitted", "project_id": 4377225, "original_total_ns": 2.0, "target_total_ns": 10.0, "continuation": "checkpoint_noappend_no_velocity_reinitialization", "states": records}
    (output / "extension_jobs.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
