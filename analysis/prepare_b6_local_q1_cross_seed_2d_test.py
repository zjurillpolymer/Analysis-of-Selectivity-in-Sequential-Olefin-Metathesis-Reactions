#!/usr/bin/env python3
"""Prepare two reciprocal-start 2D umbrella *mixing tests* for B6 q1 pair 2.

This does not replace the 25-window PMF. Both starts run under the same two
distance restraints; their hidden contact fingerprints should converge if the
orthogonal conformational degrees of freedom mix on this time scale.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from analyze_b6_local_q1_joint_mbar import DOWNLOAD, JOBS, SOURCE_METADATA


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT.parent / "bohrium_submit/b6_local_q1_cross_seed_2d_mixing_20261001"
PAIR = 2
CENTER_NM = 1.75
EQUIL_PS = 1000
PRODUCTION_PS = 2000
DT_PS = 0.002
PROJECT_ID = 4377225
SOURCES = {
    # A6 terminal geometry has strong positive fingerprint PC1; A2 terminal
    # geometry has negative PC1. Both complete GROs contain all 9219 atoms.
    "A6_origin": (JOBS[PAIR]["A6"], "window_013_1.650nm"),
    "A2_origin": (JOBS[PAIR]["A2"], "window_012_1.550nm"),
}


def mdp(base: Path, production: bool, seed: int) -> str:
    original = base.read_text(encoding="utf-8")
    lines = []
    for line in original.splitlines():
        if not line.strip().startswith("pull"):
            lines.append(line)
    edits = {
        "nsteps": str(round((PRODUCTION_PS if production else EQUIL_PS) / DT_PS)),
        "nstxout-compressed": "250" if production else "0",
        "gen-seed": str(seed) if not production else "-1",
    }
    for i, line in enumerate(lines):
        if "=" in line:
            key = line.split("=", 1)[0].strip()
            if key in edits:
                lines[i] = f"{key:24s} = {edits[key]}"
    lines.extend([
        "pull                     = yes",
        "pull-ncoords             = 2",
        "pull-ngroups             = 3",
        "pull-group1-name         = V3_midpoint",
        "pull-group2-name         = V12_midpoint",
        "pull-group3-name         = V10_midpoint",
        "pull-group1-pbcatom      = 189",
        "pull-group2-pbcatom      = 207",
        "pull-group3-pbcatom      = 203",
        "pull-pbc-ref-prev-step-com = yes",
    ])
    for coord, groups in ((1, "1 2"), (2, "1 3")):
        lines.extend([
            f"pull-coord{coord}-type        = umbrella",
            f"pull-coord{coord}-geometry    = distance",
            f"pull-coord{coord}-groups      = {groups}",
            f"pull-coord{coord}-dim         = Y Y Y",
            f"pull-coord{coord}-start       = no",
            f"pull-coord{coord}-init        = {CENTER_NM:.6f}",
            f"pull-coord{coord}-rate        = 0.0",
            f"pull-coord{coord}-k           = 500.0",
        ])
    lines.extend(["pull-nstxout             = 250", "pull-nstfout             = 250"])
    return "\n".join(lines) + "\n"


def make_task(origin: str, job_id: int, window: str, mapping: dict[str, list[int]]) -> dict[str, object]:
    source = DOWNLOAD / str(job_id) / "extracted/pathway"
    start = source / "windows" / window / "production.gro"
    if int(start.read_text(encoding="utf-8").splitlines()[1]) != 9219:
        raise ValueError(f"incomplete full-system source GRO: {start}")
    base = source / "window_mdps"
    task = OUT / origin
    task.mkdir(parents=True)
    shutil.copy2(start, task / "start.gro")
    shutil.copy2(source / "system.top", task / "system.top")
    index = (source / "index.ndx").read_text(encoding="utf-8")
    for label in ("V3", "V12", "V10"):
        if f"[ {label}_midpoint ]" not in index:
            index += f"\n[ {label}_midpoint ]\n" + " ".join(map(str, mapping[label])) + "\n"
    (task / "index.ndx").write_text(index, encoding="utf-8")
    for file_name, prod in (("equil.mdp", False), ("production.mdp", True)):
        template = base / f"window_013_1.650nm_{'production' if prod else 'equil'}.mdp"
        (task / file_name).write_text(mdp(template, prod, 20261002 if origin == "A6_origin" else 20261003), encoding="utf-8")
    run = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p results
exec > >(tee results/run.log) 2>&1
export OMP_NUM_THREADS=4
export GMX_MAXBACKUP=-1
archive() {
  status=$?
  tar -czf results/cross_seed_2d_results.tar.gz equil.mdp production.mdp start.gro index.ndx \\
    equil.gro equil.cpt production.gro production.cpt production.xtc \\
    equil_pullx.xvg production_pullx.xvg equil.log production.log results/run.log 2>/dev/null || true
  exit "$status"
}
trap archive EXIT
gmx grompp -f equil.mdp -c start.gro -p system.top -n index.ndx -o equil.tpr -maxwarn 0
gmx mdrun -deffnm equil -ntmpi 1 -ntomp 4 -pin off -nb gpu -pme gpu -bonded cpu -update cpu -px equil_pullx.xvg
gmx grompp -f production.mdp -c equil.gro -t equil.cpt -p system.top -n index.ndx -o production.tpr -maxwarn 0
gmx mdrun -deffnm production -ntmpi 1 -ntomp 4 -pin off -nb gpu -pme gpu -bonded cpu -update cpu -px production_pullx.xvg
"""
    (task / "run.sh").write_text(run, encoding="utf-8")
    (task / "run.sh").chmod(0o755)
    metadata = {
        "purpose": "reciprocal-start 2D restraint mixing test, not a PMF window set",
        "origin": origin, "source_job": job_id, "source_window": window,
        "target_distances_nm": {"V3-V12": CENTER_NM, "V3-V10": CENTER_NM},
        "equilibration_ps": EQUIL_PS, "production_ps": PRODUCTION_PS,
        "source_full_system_atoms": 9219,
    }
    (task / "test.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    job = {
        "job_name": f"B6 q1 pair2 cross-seed 2D mixing {origin}",
        "command": "bash run.sh",
        "log_file": "results/run.log",
        "backward_files": ["results/cross_seed_2d_results.tar.gz", "results/run.log"],
        "project_id": PROJECT_ID, "machine_type": "c4_m15_1 * NVIDIA T4",
        "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
        "input_directory": str(task), "max_reschedule_times": 1,
        "max_run_time": 180, "nnode": 1,
    }
    (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    return {**metadata, "task_directory": str(task)}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite prepared or submitted task directory: {OUT}")
    OUT.mkdir(parents=True)
    mapping = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))["vinyl_mapping_gromacs_1based"]
    if mapping["V10"][0] != 203:
        raise ValueError(f"unexpected V10 pbcatom: {mapping['V10']}")
    records = [make_task(origin, *source, mapping) for origin, source in SOURCES.items()]
    (OUT / "manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
