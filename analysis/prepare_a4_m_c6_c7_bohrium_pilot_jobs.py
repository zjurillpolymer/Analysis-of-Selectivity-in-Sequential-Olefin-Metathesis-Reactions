#!/usr/bin/env python3
"""Build 18 direct Bohrium jobs for the C6/C7 q1/q2 3x2 ns pilot.

Each job owns one state/replica pair and one T4 GPU.  The inputs reuse the
audited GAFF2/AM1-BCC topology and completed NPT/reseed checkpoints from the
local server archive; only the missing production trajectory is generated.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


STATES = (
    "A4_m_C6_1_cis",
    "A4_m_C6_2_adjacent_cis",
    "A4_m_C6_2_opposite_cis",
    "A4_m_C7_1_cis",
    "A4_m_C7_2_adjacent_cis",
    "A4_m_C7_2_opposite_cis",
)


RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
exec > >(tee -a pilot_job.log) 2>&1

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

mkdir -p nac
python3 workspace/analysis/export_explicit_dcm_postreaction_nac.py \
  --root workspace \
  --states {state} \
  --replicas 3 \
  --replica-indices {replica} \
  --output nac
"""


def copy_static_bundle(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)
    analysis = destination / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=project / "results/a4_m_c6_c7_postreaction_nac_bundle",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=project / "autodl-tmp/Molecular_dynamics_project",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "bohrium_submit/a4_m_c6_c7_pilot",
    )
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    archive_root = args.archive_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    rows = list(csv.DictReader((bundle / "gromacs_dcm/state_manifest.csv").open()))
    by_state = {row["state"]: row for row in rows}
    if set(STATES) - set(by_state):
        raise ValueError("selected states are missing from the manifest")

    records = []
    for state in STATES:
        row = by_state[state]
        archived_state = archive_root / row["output_dir"]
        for replica in range(1, 4):
            task_name = f"{state}_replica_{replica:02d}"
            task = output / task_name
            workspace = task / "workspace"
            copy_static_bundle(bundle, workspace)
            for script_name in (
                "run_explicit_dcm_gromacs.py",
                "export_explicit_dcm_postreaction_nac.py",
            ):
                shutil.copy2(project / "analysis" / script_name, workspace / "analysis" / script_name)

            target_state = workspace / row["output_dir"]
            target_state.mkdir(parents=True, exist_ok=True)
            shutil.copytree(archived_state / "setup", target_state / "setup")
            shutil.copytree(archived_state / "base_equilibration", target_state / "base_equilibration")
            source_replica = archived_state / f"replica_{replica:02d}"
            target_replica = target_state / f"replica_{replica:02d}"
            target_replica.mkdir()
            for path in source_replica.glob("reseed_nvt*"):
                shutil.copy2(path, target_replica / path.name)
            if not all((target_replica / f"reseed_nvt{suffix}").is_file() for suffix in (".tpr", ".gro", ".cpt")):
                raise FileNotFoundError(f"incomplete reseed checkpoint: {source_replica}")

            archive = f"{task_name}_results.tar.gz"
            run_script = task / "run_pilot.sh"
            run_script.write_text(
                RUN_SCRIPT.format(state=state, replica=replica, archive=archive),
                encoding="utf-8",
            )
            run_script.chmod(0o755)
            job = {
                "job_name": f"A4-m {state} replica {replica:02d} pilot",
                "command": "bash run_pilot.sh",
                "log_file": "pilot_job.log",
                "backward_files": [archive, "pilot_job.log"],
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

    audit = {
        "status": "READY",
        "jobs": len(records),
        "replicas_per_state": 3,
        "production_ns_per_replica": 2.0,
        "gpu_policy": "one NVIDIA T4 per state/replica job",
        "starting_point": "completed archived NPT plus replica reseed checkpoint",
        "tasks": records,
    }
    (output / "pilot_jobs.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
