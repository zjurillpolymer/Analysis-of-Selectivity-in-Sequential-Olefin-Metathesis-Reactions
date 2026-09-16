#!/usr/bin/env python3
"""Build six direct Bohrium jobs for C6/C7 mixed-q2 3 x 10 ns MD."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


STATES = (
    "A4_m_C6_2_mixed_same_cross_cis",
    "A4_m_C7_2_mixed_same_cross_cis",
)


RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
exec > >(tee -a mixed_10ns_job.log) 2>&1

archive_results() {{
  status=$?
  items=(workspace/gromacs_dcm/runs/{state}/replica_{replica:02d})
  if [[ -d nac ]]; then
    items+=(nac)
  fi
  tar -czf {archive} "${{items[@]}}"
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


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=project / "results/a4_m_c6_c7_q2_missing_local_prepared/workspace",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "bohrium_submit/a4_m_c6_c7_mixed_10ns_compact",
    )
    args = parser.parse_args()
    source = args.workspace.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    manifest_path = source / "gromacs_dcm/state_manifest.csv"
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    by_state = {row["state"]: row for row in rows}
    missing = set(STATES) - set(by_state)
    if missing:
        raise ValueError(f"states missing from manifest: {sorted(missing)}")

    density_gate = (
        source
        / "gromacs_dcm/solvent_validation/ch2cl2_dc/run/density_gate.json"
    )
    if not density_gate.is_file() or not json.loads(density_gate.read_text()).get("passed"):
        raise RuntimeError("validated ch2cl2_dc density gate is missing")

    records: list[dict[str, object]] = []
    for state in STATES:
        row = by_state[state]
        state_source = source / row["output_dir"]
        for replica in range(1, 4):
            task_name = f"{state}_replica_{replica:02d}"
            task = output / task_name
            workspace = task / "workspace"
            (workspace / "gromacs_dcm").mkdir(parents=True)

            # Copy the small shared configuration and just the selected state setup.
            shutil.copy2(source / "gromacs_dcm/state_manifest.csv", workspace / "gromacs_dcm/state_manifest.csv")
            shutil.copytree(source / "gromacs_dcm/mdp", workspace / "gromacs_dcm/mdp")
            # The post-reaction NAC exporter needs the q2 history/mapping
            # tables at runtime; keep these small metadata files in every
            # standalone Bohrium task.
            results_target = workspace / "results" / "react2"
            results_target.mkdir(parents=True)
            for name in ("vinyl_mapping.csv", "structure_qc.csv"):
                shutil.copy2(source / "results" / "react2" / name, results_target / name)
            gate_target = workspace / "gromacs_dcm/solvent_validation/ch2cl2_dc/run"
            gate_target.mkdir(parents=True)
            shutil.copy2(density_gate, gate_target / "density_gate.json")
            state_target = workspace / row["output_dir"]
            state_target.mkdir(parents=True)
            setup_target = state_target / "setup"
            setup_target.mkdir()
            for name in ("system.gro", "system.top", "metadata.json", "index.ndx"):
                shutil.copy2(state_source / "setup" / name, setup_target / name)
            conformers_target = setup_target / "initial_conformers"
            conformers_target.mkdir()
            for source_replica in sorted(
                (state_source / "setup/initial_conformers").glob("replica_*")
            ):
                target_replica = conformers_target / source_replica.name
                target_replica.mkdir()
                for name in ("system.gro", "system.top", "metadata.json", "index.ndx"):
                    shutil.copy2(source_replica / name, target_replica / name)

            analysis = workspace / "analysis"
            analysis.mkdir()
            for script in (
                "run_explicit_dcm_gromacs.py",
                "extend_explicit_dcm_q0.py",
                "export_explicit_dcm_postreaction_nac.py",
            ):
                shutil.copy2(project / "analysis" / script, analysis / script)

            archive = f"{task_name}_results.tar.gz"
            run_script = task / "run_mixed_10ns.sh"
            run_script.write_text(
                RUN_SCRIPT.format(state=state, replica=replica, archive=archive),
                encoding="utf-8",
            )
            run_script.chmod(0o755)
            job = {
                "job_name": f"A4-m {state} replica {replica:02d} 10ns",
                "command": "bash run_mixed_10ns.sh",
                "log_file": "mixed_10ns_job.log",
                "backward_files": [archive, "mixed_10ns_job.log"],
                "project_id": 4377225,
                "machine_type": "c4_m15_1 * NVIDIA T4",
                "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
                "input_directory": str(task),
                "max_reschedule_times": 2,
                "max_run_time": 720,
                "nnode": 1,
            }
            (task / "job.json").write_text(
                json.dumps(job, indent=2) + "\n", encoding="utf-8"
            )
            records.append(
                {"task": task_name, "state": state, "replica": replica, "archive": archive}
            )

    audit = {
        "status": "READY",
        "jobs": len(records),
        "states": list(STATES),
        "replicas_per_state": 3,
        "target_total_ns_per_replica": 10.0,
        "gpu_policy": "one NVIDIA T4 per state/replica job",
        "tasks": records,
    }
    (output / "jobs.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
