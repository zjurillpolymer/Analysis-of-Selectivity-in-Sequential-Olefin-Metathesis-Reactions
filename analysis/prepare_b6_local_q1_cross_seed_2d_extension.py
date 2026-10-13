#!/usr/bin/env python3
"""Prepare 5 ns follow-up trajectories from the two completed 2D test endpoints."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DOWNLOAD = PROJECT.parent / "bohrium_download/b6_local_q1_cross_seed_2d_mixing_20261001"
OUT = PROJECT.parent / "bohrium_submit/b6_local_q1_cross_seed_2d_extension_20261001"
PREVIOUS = PROJECT.parent / "bohrium_submit/b6_local_q1_cross_seed_2d_mixing_20261001"
JOBS = {"A6_origin": 20807020, "A2_origin": 20807021}
PROJECT_ID = 4377225


def replace_setting(text: str, key: str, value: str) -> str:
    output = []
    for line in text.splitlines():
        if "=" in line and line.split("=", 1)[0].strip() == key:
            output.append(f"{key:24s} = {value}")
        else:
            output.append(line)
    if not any(line.strip().startswith(key + " ") for line in output):
        raise ValueError(f"mdp setting not found: {key}")
    return "\n".join(output) + "\n"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    manifest = []
    for origin, job_id in JOBS.items():
        old_work = DOWNLOAD / str(job_id) / "extracted/work"
        task = OUT / origin
        task.mkdir()
        for source, target in ((old_work / "production.gro", "start.gro"),
                               (PREVIOUS / origin / "system.top", "system.top"),
                               (PREVIOUS / origin / "index.ndx", "index.ndx")):
            if not source.is_file():
                raise FileNotFoundError(source)
            shutil.copy2(source, task / target)
        equil_base = (PREVIOUS / origin / "equil.mdp").read_text(encoding="utf-8")
        production_base = (PREVIOUS / origin / "production.mdp").read_text(encoding="utf-8")
        (task / "equil.mdp").write_text(
            replace_setting(replace_setting(equil_base, "nsteps", "250000"), "gen-seed",
                            "20261004" if origin == "A6_origin" else "20261005"), encoding="utf-8")
        (task / "production.mdp").write_text(
            replace_setting(production_base, "nsteps", "2500000"), encoding="utf-8")
        run = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p results
exec > >(tee results/run.log) 2>&1
export OMP_NUM_THREADS=4
export GMX_MAXBACKUP=-1
archive() {
  status=$?
  tar -czf results/extension_5ns_results.tar.gz equil.mdp production.mdp start.gro index.ndx \\
    equil.gro equil.cpt extension.gro extension.cpt extension.xtc extension_pullx.xvg \\
    equil_pullx.xvg equil.log extension.log results/run.log 2>/dev/null || true
  exit "$status"
}
trap archive EXIT
gmx grompp -f equil.mdp -c start.gro -p system.top -n index.ndx -o equil.tpr -maxwarn 0
gmx mdrun -deffnm equil -ntmpi 1 -ntomp 4 -pin off -nb gpu -pme gpu -bonded cpu -update cpu -px equil_pullx.xvg
gmx grompp -f production.mdp -c equil.gro -t equil.cpt -p system.top -n index.ndx -o extension.tpr -maxwarn 0
gmx mdrun -s extension.tpr -deffnm extension -ntmpi 1 -ntomp 4 -pin off -nb gpu -pme gpu -bonded cpu -update cpu -px extension_pullx.xvg
"""
        (task / "run.sh").write_text(run, encoding="utf-8")
        (task / "run.sh").chmod(0o755)
        details = {
            "origin": origin, "previous_job": job_id,
            "start": "final production.gro from completed 2D test",
            "new_velocity_seed": 20261004 if origin == "A6_origin" else 20261005,
            "equilibration_ps": 500, "production_ps": 5000,
            "restraints_nm": {"V3-V12": 1.75, "V3-V10": 1.75},
        }
        (task / "extension.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
        job = {
            "job_name": f"B6 q1 pair2 2D mixing 5ns follow-up {origin}",
            "command": "bash run.sh", "log_file": "results/run.log",
            "backward_files": ["results/extension_5ns_results.tar.gz", "results/run.log"],
            "project_id": PROJECT_ID, "machine_type": "c4_m15_1 * NVIDIA T4",
            "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
            "input_directory": str(task), "max_reschedule_times": 1,
            "max_run_time": 240, "nnode": 1,
        }
        (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        manifest.append({**details, "task_directory": str(task)})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
