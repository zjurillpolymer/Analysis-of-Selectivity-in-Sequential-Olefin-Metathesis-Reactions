#!/usr/bin/env python3
"""Prepare six independent C5/C8 q1 10-ns Bohrium GPU jobs."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


STATES = ("A4_m_C5_1_cis", "A4_m_C8_1_cis")

RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
exec > >(tee -a q1_10ns_job.log) 2>&1

status=0
archive_results() {{
  status=$?
  tar -czf {archive} nac workspace/gromacs_dcm/runs/{state}/replica_{replica:02d}
  exit "$status"
}}
trap archive_results EXIT

export OMP_NUM_THREADS=4
nvidia-smi -L
gmx --version

python3 workspace/analysis/run_explicit_dcm_gromacs.py \
  --root workspace \
  --states {state} \
  --replicas 3 \
  --replica-indices {replica} \
  --ntomp 4 \
  --dcm-model ch2cl2_dc

python3 workspace/analysis/extend_explicit_dcm_q0.py \
  --root workspace \
  --states {state} \
  --replicas 3 \
  --replica-indices {replica} \
  --target-total-ns 10 \
  --ntomp 4

mkdir -p nac
python3 workspace/analysis/export_explicit_dcm_postreaction_nac.py \
  --root workspace \
  --states {state} \
  --replicas 3 \
  --replica-indices {replica} \
  --output nac
"""


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=project / "results/a4_postreaction_nac_prepared_q1")
    parser.add_argument("--archive-root", type=Path, default=project / "autodl-tmp/Molecular_dynamics_project")
    parser.add_argument("--output", type=Path, default=project / "bohrium_submit/a4_m_c5_c8_q1_10ns")
    args = parser.parse_args()
    bundle, archive_root, output = (path.resolve() for path in (args.bundle, args.archive_root, args.output))
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    manifest = list(csv.DictReader((bundle / "gromacs_dcm/state_manifest.csv").open(encoding="utf-8")))
    by_state = {row["state"]: row for row in manifest}
    missing = set(STATES) - set(by_state)
    if missing:
        raise ValueError(f"q1 states missing from manifest: {sorted(missing)}")

    records = []
    for state in STATES:
        row = by_state[state]
        archived_state = archive_root / row["output_dir"]
        for replica in range(1, 4):
            task_name = f"{state}_replica_{replica:02d}"
            task = output / task_name
            workspace = task / "workspace"
            shutil.copytree(bundle, workspace)
            for script in (
                "run_explicit_dcm_gromacs.py",
                "extend_explicit_dcm_q0.py",
                "export_explicit_dcm_postreaction_nac.py",
            ):
                copy_file(project / "analysis" / script, workspace / "analysis" / script)
            target_state = workspace / row["output_dir"]
            target_state.mkdir(parents=True, exist_ok=True)
            existing_setup = target_state / "setup"
            if existing_setup.exists():
                shutil.rmtree(existing_setup)
            shutil.copytree(archived_state / "setup", target_state / "setup")
            shutil.copytree(archived_state / "base_equilibration", target_state / "base_equilibration")
            target_replica = target_state / f"replica_{replica:02d}"
            target_replica.mkdir()
            source_replica = archived_state / f"replica_{replica:02d}"
            for path in source_replica.glob("reseed_nvt*"):
                copy_file(path, target_replica / path.name)
            for suffix in (".tpr", ".gro", ".cpt"):
                if not (target_replica / f"reseed_nvt{suffix}").is_file():
                    raise FileNotFoundError(f"incomplete q1 reseed checkpoint: {source_replica}")

            archive = f"{task_name}_results.tar.gz"
            run_script = task / "run_q1_10ns.sh"
            run_script.write_text(
                RUN_SCRIPT.format(state=state, replica=replica, archive=archive),
                encoding="utf-8",
            )
            run_script.chmod(0o755)
            job = {
                "job_name": f"A4-m {state} replica {replica:02d} q1 10ns",
                "command": "bash run_q1_10ns.sh",
                "log_file": "q1_10ns_job.log",
                "backward_files": [archive, "q1_10ns_job.log"],
                "project_id": 4377225,
                "machine_type": "c4_m15_1 * NVIDIA T4",
                "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
                "input_directory": str(task),
                "max_reschedule_times": 2,
                "max_run_time": 720,
                "nnode": 1,
            }
            (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
            records.append({"task": task_name, "state": state, "replica": replica, "archive": archive})

    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "READY",
        "jobs": len(records),
        "states": list(STATES),
        "replicas_per_state": 3,
        "target_total_ns_per_replica": 10.0,
        "purpose": "C5/C8 q1 full-reuse rare-event and branch convergence",
        "gpu_policy": "one NVIDIA T4 per state/replica job",
        "tasks": records,
    }
    (output / "jobs.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
