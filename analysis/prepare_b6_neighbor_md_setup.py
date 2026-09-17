#!/usr/bin/env python3
"""Stage audited B6 q1/q2 Amber parameters for explicit-DCM system setup.

The output is a Bohrium CPU input directory. The four invalid q2 parameter
sets from job 20735563 are rejected by the SHA256 provenance manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


STATES = (
    "B6_p_C6_q1_V1_V2_cis",
    "B6_p_C6_q2_neighbor_full_reuse_cis",
    "B6_p_C6_q2_neighbor_one_arm_reuse_cis",
    "B6_p_C6_q2_neighbor_full_switch_d1_cis",
    "B6_p_C6_q2_neighbor_full_switch_d2_cis",
)

REQUIRED_PARAMETER_FILES = (
    "parameterization_metadata.json",
    "slu.mol2",
    "slu.frcmod",
    "slu.pdb",
)

RUN_SCRIPT = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec > >(tee -a md_setup_job.log) 2>&1

archive_results() {
  status=$?
  tar -czf B6_neighbor_md_setup_results.tar.gz workspace
  exit "$status"
}
trap archive_results EXIT

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

command -v antechamber
command -v parmchk2
command -v tleap
python3 -c 'import parmed; print("ParmEd", parmed.__version__)'
test -f workspace/prepacked_boxes.json

python3 workspace/analysis/verify_b6_parameter_sources.py \\
  --root workspace \\
  --manifest workspace/validated_parameter_sources.json

python3 workspace/analysis/setup_explicit_dcm_gromacs.py \\
  --root workspace \\
  --states \\
    B6_p_C6_q1_V1_V2_cis \\
    B6_p_C6_q2_neighbor_full_reuse_cis \\
    B6_p_C6_q2_neighbor_one_arm_reuse_cis \\
    B6_p_C6_q2_neighbor_full_switch_d1_cis \\
    B6_p_C6_q2_neighbor_full_switch_d2_cis \\
  --jobs 2 \\
  --dcm-model ch2cl2_dc \\
  --sqm-grms-tol 0.05 \\
  --defer-grompp \\
  --use-prepacked-boxes

python3 workspace/analysis/verify_b6_md_setup.py --root workspace
"""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=project.parent / "bohrium_submit/b6_neighbor_md_setup_20260917",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    source_root = project / "gromacs_dcm_server_bundle"
    provenance_path = project / "b6_workflow/bohrium/validated_parameter_sources.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if set(provenance["states"]) != set(STATES):
        raise ValueError("unexpected audited B6 state set")

    source_manifest = source_root / "gromacs_dcm/state_manifest.csv"
    with source_manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = [row for row in rows if row["state"] in STATES]
    if len(selected) != len(STATES):
        raise ValueError("B6 q1/q2 state missing or duplicated in manifest")

    for state in STATES:
        folder = project / provenance["root"] / state / "setup/solute_parameters"
        record = provenance["states"][state]
        for filename in REQUIRED_PARAMETER_FILES:
            path = folder / filename
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        for filename, key in (
            ("slu.mol2", "slu_mol2_sha256"),
            ("slu.frcmod", "slu_frcmod_sha256"),
        ):
            if digest(folder / filename) != record[key]:
                raise ValueError(f"uncertified parameter: {state}/{filename}")

    workspace = output / "workspace"
    (workspace / "analysis").mkdir(parents=True)
    for filename in (
        "setup_explicit_dcm_gromacs.py",
        "verify_b6_parameter_sources.py",
        "verify_b6_md_setup.py",
    ):
        shutil.copy2(project / "analysis" / filename, workspace / "analysis" / filename)
    staged_provenance = {**provenance, "root": "gromacs_dcm/runs"}
    (workspace / "validated_parameter_sources.json").write_text(
        json.dumps(staged_provenance, indent=2) + "\n", encoding="utf-8"
    )
    (workspace / "structures").mkdir()
    for row in selected:
        source = source_root / row["source_pdb"]
        for path in (source, source.with_name(source.stem + "_antechamber.mol2")):
            if not path.is_file():
                raise FileNotFoundError(path)
            shutil.copy2(path, workspace / "structures" / path.name)

    gromacs_source = source_root / "gromacs_dcm"
    gromacs_target = workspace / "gromacs_dcm"
    gromacs_target.mkdir()
    shutil.copytree(gromacs_source / "mdp", gromacs_target / "mdp")
    shutil.copytree(gromacs_source / "models", gromacs_target / "models")
    shutil.copy2(gromacs_source / "dcm.pdb", gromacs_target / "dcm.pdb")
    with (gromacs_target / "state_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(selected)
    for state in STATES:
        source = gromacs_source / "runs" / state / "setup/solute_parameters"
        target = gromacs_target / "runs" / state / "setup/solute_parameters"
        shutil.copytree(source, target)

    run_script = output / "run_setup.sh"
    run_script.write_text(RUN_SCRIPT, encoding="utf-8")
    run_script.chmod(0o755)
    job = {
        "job_name": "B6 neighbor q1-q2 prepacked DCM setup",
        "command": "bash run_setup.sh",
        "log_file": "md_setup_job.log",
        "backward_files": ["B6_neighbor_md_setup_results.tar.gz", "md_setup_job.log"],
        "project_id": 4377225,
        "machine_type": "c8_m16_cpu",
        "image_address": "registry.dp.tech/dptech/amber:24_rc2",
        "input_directory": str(output),
        "max_reschedule_times": 2,
        "max_run_time": 720,
        "nnode": 1,
    }
    (output / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "states": list(STATES), "job": job}, indent=2))


if __name__ == "__main__":
    main()
