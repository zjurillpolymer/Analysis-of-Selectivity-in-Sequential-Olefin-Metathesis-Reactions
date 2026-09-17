#!/usr/bin/env python3
"""Prepare one Bohrium T4 pilot job per validated B6 q1/q2 state."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from verify_b6_md_setup import check_setup


RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec > >(tee -a md_pilot_job.log) 2>&1

archive_results() {
  status=$?
  tar -czf {archive} workspace
  exit "$status"
}
trap archive_results EXIT

nvidia-smi -L
gmx --version
python3 workspace/analysis/run_explicit_dcm_gromacs.py \\
  --root workspace \\
  --states {state} \\
  --replicas 3 \\
  --ntomp 4 \\
  --dcm-model ch2cl2_dc

for replica in 01 02 03; do
  test -s workspace/gromacs_dcm/runs/{state}/replica_${{replica}}/production_2ns.xtc
  test -s workspace/gromacs_dcm/runs/{state}/replica_${{replica}}/production_2ns.gro
done
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup-root", type=Path, required=True,
                        help="downloaded and extracted CPU setup workspace")
    parser.add_argument("--density-gate", type=Path,
                        default=project.parent / "gromacs_dcm_server_bundle/gromacs_dcm/solvent_validation/ch2cl2_dc/run/density_gate.json")
    parser.add_argument("--output", type=Path,
                        default=project.parent / "bohrium_submit/b6_neighbor_md_pilot_20260917")
    args = parser.parse_args()
    setup_root = args.setup_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source = setup_root / "gromacs_dcm"
    with (source / "state_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5 or not all(row["system"] == "B6_p_C6" for row in rows):
        raise ValueError("setup manifest must contain exactly five B6-p-C6 q1/q2 states")
    for row in rows:
        check_setup(setup_root / row["output_dir"] / "setup", row["state"])
    check_setup(source / "solvent_validation/ch2cl2_dc/setup", None)

    gate_path = args.density_gate.resolve()
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("dcm_model") != "ch2cl2_dc" or not gate.get("passed"):
        raise ValueError("the cited pure-DCM density gate did not pass")
    reference_model = project.parent / "gromacs_dcm_server_bundle/gromacs_dcm/models/ch2cl2_dc"
    for filename in ("dcm.mol2", "dcm.frcmod", "dcm.pdb"):
        if digest(source / "models/ch2cl2_dc" / filename) != digest(reference_model / filename):
            raise ValueError(f"density-gate DCM model differs from setup: {filename}")

    records = []
    for row in rows:
        state = row["state"]
        task = output / state
        workspace = task / "workspace"
        (workspace / "analysis").mkdir(parents=True)
        (workspace / "gromacs_dcm").mkdir()
        shutil.copy2(project / "analysis/run_explicit_dcm_gromacs.py", workspace / "analysis")
        shutil.copytree(source / "mdp", workspace / "gromacs_dcm/mdp")
        with (workspace / "gromacs_dcm/state_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerow(row)
        target_setup = workspace / row["output_dir"] / "setup"
        shutil.copytree(setup_root / row["output_dir"] / "setup", target_setup)
        gate_target = workspace / "gromacs_dcm/solvent_validation/ch2cl2_dc/run/density_gate.json"
        gate_target.parent.mkdir(parents=True)
        shutil.copy2(gate_path, gate_target)
        provenance = {
            "density_gate_source": str(gate_path),
            "density_gate_sha256": digest(gate_path),
            "dcm_model_sha256": {
                filename: digest(reference_model / filename)
                for filename in ("dcm.mol2", "dcm.frcmod", "dcm.pdb")
            },
            "reason_for_reuse": "identical CH2Cl2-DC model and temperature/pressure protocol",
        }
        (workspace / "density_gate_provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        archive = f"{state}_md_pilot_results.tar.gz"
        run_script = task / "run_md.sh"
        run_script.write_text(RUN_SCRIPT.format(state=state, archive=archive), encoding="utf-8")
        run_script.chmod(0o755)
        job = {
            "job_name": f"B6 neighbor {state} 3x2ns pilot",
            "command": "bash run_md.sh",
            "log_file": "md_pilot_job.log",
            "backward_files": [archive, "md_pilot_job.log"],
            "project_id": 4377225,
            "machine_type": "c4_m15_1 * NVIDIA T4",
            "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
            "input_directory": str(task),
            "max_reschedule_times": 2,
            "max_run_time": 720,
            "nnode": 1,
        }
        (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        records.append({"state": state, "task": str(task), "archive": archive})
    summary = {
        "status": "prepared_not_submitted",
        "setup_source": str(setup_root),
        "one_gpu_per_state": True,
        "replicas_per_state": 3,
        "production_ns_per_replica": 2.0,
        "tasks": records,
    }
    (output / "pilot_jobs.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
