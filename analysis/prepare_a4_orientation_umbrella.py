#!/usr/bin/env python3
"""Prepare targeted orientation-umbrella validation jobs for A4 C5/C8.

This pilot is not a full distance-by-orientation grid. The distance is
restrained at the contact value (0.50 nm), while the angle between the two
vinyl-bond vectors is sampled with seven angular windows.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
ANGLE_CENTERS_DEG = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0)
DISTANCE_CENTER_NM = 0.50
DISTANCE_FORCE_KJ_MOL_NM2 = 500.0
ANGLE_FORCE_KJ_MOL_RAD2 = 50.0
DT_PS = 0.002
EQUILIBRATION_PS = 100.0
PRODUCTION_PS = 1000.0

PATHS = (
    {
        "name": "c5_q2_reuse_v1_v6",
        "system": "C5",
        "route": "reuse",
        "source": "results/a4_q2_umbrella_bohrium/download/c5_reuse_v1_v6/pathway",
        "left": 1,
        "right": 6,
    },
    {
        "name": "c5_q2_switch_v4_v6",
        "system": "C5",
        "route": "switch",
        "source": "results/a4_q2_umbrella_bohrium/download/c5_switch_v4_v6/pathway",
        "left": 4,
        "right": 6,
    },
    {
        "name": "c8_q2_reuse_v5_v7",
        "system": "C8",
        "route": "reuse",
        "source": "results/a4_q2_umbrella_bohrium/resumed_download/c8_reuse_v5_v7/pathway",
        "left": 5,
        "right": 7,
    },
    {
        "name": "c8_q2_switch_v4_v5",
        "system": "C8",
        "route": "switch",
        "source": "results/a4_q2_umbrella_bohrium/partial/c8_switch_v4_v5/pathway",
        "left": 4,
        "right": 5,
    },
)


def parse_index_text(text: str) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip()
            groups[current] = []
        elif current and stripped and not stripped.startswith((";", "#")):
            groups[current].extend(int(value) for value in stripped.split())
    return groups


def append_axis_groups(index_text: str, left: int, right: int) -> tuple[str, dict[str, list[int]]]:
    groups = parse_index_text(index_text)
    left_atoms = groups[f"V{left}_midpoint"]
    right_atoms = groups[f"V{right}_midpoint"]
    if len(left_atoms) != 2 or len(right_atoms) != 2:
        raise ValueError(f"expected two vinyl atoms per group, got {left_atoms}/{right_atoms}")
    additions = {
        f"V{left}_axis_a": [left_atoms[0]],
        f"V{left}_axis_b": [left_atoms[1]],
        f"V{right}_axis_a": [right_atoms[0]],
        f"V{right}_axis_b": [right_atoms[1]],
    }
    chunks = [index_text.rstrip()]
    for name, atoms in additions.items():
        chunks.extend((f"\n[ {name} ]", " ".join(map(str, atoms))))
    return "\n".join(chunks) + "\n", additions


def common_mdp(nsteps: int, continuation: bool, gen_vel: bool) -> str:
    return f"""integrator = md
dt = {DT_PS:g}
nsteps = {nsteps}
continuation = {'yes' if continuation else 'no'}
constraint-algorithm = lincs
constraints = h-bonds
lincs-iter = 1
lincs-order = 4
cutoff-scheme = Verlet
nstlist = 20
rlist = 1.2
coulombtype = PME
rcoulomb = 1.2
pme-order = 4
fourierspacing = 0.12
vdwtype = Cut-off
rvdw = 1.2
DispCorr = EnerPres
tcoupl = V-rescale
tc-grps = System
tau-t = 1.0
ref-t = 313.15
pcoupl = C-rescale
pcoupltype = isotropic
tau-p = 5.0
ref-p = 1.0
compressibility = 1.0e-4
gen-vel = {'yes' if gen_vel else 'no'}
gen-temp = 313.15
gen-seed = -1
nstxout = 0
nstvout = 0
nstfout = 0
nstxout-compressed = 250
compressed-x-grps = System
compressed-x-precision = 1000
nstenergy = 500
nstlog = 5000
nstcalcenergy = 100
pbc = xyz
comm-mode = Linear
nstcomm = 100
pull = yes
pull-ncoords = 2
pull-ngroups = 6
pull-print-components = no
pull-group1-name = __DIST_LEFT_NAME__
pull-group2-name = __DIST_RIGHT_NAME__
pull-group3-name = __ANGLE_LEFT_A_NAME__
pull-group4-name = __ANGLE_LEFT_B_NAME__
pull-group5-name = __ANGLE_RIGHT_A_NAME__
pull-group6-name = __ANGLE_RIGHT_B_NAME__
pull-group1-pbcatom = __LEFT_PBC__
pull-group2-pbcatom = __RIGHT_PBC__
pull-pbc-ref-prev-step-com = yes
pull-coord1-type = umbrella
pull-coord1-geometry = distance
pull-coord1-groups = 1 2
pull-coord1-dim = Y Y Y
pull-coord1-start = no
pull-coord1-init = 0.500000
pull-coord1-rate = 0
pull-coord1-k = 500.0
pull-coord2-type = umbrella
pull-coord2-geometry = angle
pull-coord2-groups = 3 4 5 6
pull-coord2-dim = Y Y Y
pull-coord2-start = no
pull-coord2-init = __ANGLE_CENTER__
pull-coord2-rate = 0
pull-coord2-k = 50.0
pull-nstxout = 250
pull-nstfout = 250
"""


def render_mdp(template: str, left: int, right: int, left_atoms: list[int], right_atoms: list[int], angle: float) -> str:
    return (
        template
        .replace("__DIST_LEFT_NAME__", f"V{left}_midpoint")
        .replace("__DIST_RIGHT_NAME__", f"V{right}_midpoint")
        .replace("__ANGLE_LEFT_A_NAME__", f"V{left}_axis_a")
        .replace("__ANGLE_LEFT_B_NAME__", f"V{left}_axis_b")
        .replace("__ANGLE_RIGHT_A_NAME__", f"V{right}_axis_a")
        .replace("__ANGLE_RIGHT_B_NAME__", f"V{right}_axis_b")
        .replace("__LEFT_PBC__", str(left_atoms[0]))
        .replace("__RIGHT_PBC__", str(right_atoms[0]))
        .replace("__ANGLE_CENTER__", f"{angle:.6f}")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "bohrium_submit/a4_orientation_umbrella_pilot")
    parser.add_argument("--project-id", type=int, default=4377225)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for path_spec in PATHS:
        source = PROJECT / str(path_spec["source"])
        source_index = source / "index.ndx"
        source_top = source / "system.top"
        start = source / "windows/window_000_0.350nm/production.gro"
        if not source_index.is_file() or not source_top.is_file() or not start.is_file():
            raise FileNotFoundError(f"missing source files for {path_spec['name']}: {source}")
        source_groups = parse_index_text(source_index.read_text(encoding="utf-8"))
        left = int(path_spec["left"])
        right = int(path_spec["right"])
        left_atoms = source_groups[f"V{left}_midpoint"]
        right_atoms = source_groups[f"V{right}_midpoint"]
        index_text, additions = append_axis_groups(source_index.read_text(encoding="utf-8"), left, right)
        equil_template = common_mdp(int(round(EQUILIBRATION_PS / DT_PS)), False, True)
        production_template = common_mdp(int(round(PRODUCTION_PS / DT_PS)), True, False)
        for window_index, angle in enumerate(ANGLE_CENTERS_DEG):
            task_name = f"{path_spec['name']}_axis_{window_index:03d}_{int(angle):03d}deg"
            task = output / task_name
            task.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_top, task / "system.top")
            shutil.copy2(start, task / "start.gro")
            (task / "index.ndx").write_text(index_text, encoding="utf-8")
            (task / "equilibration.mdp").write_text(
                render_mdp(equil_template, left, right, left_atoms, right_atoms, angle), encoding="utf-8"
            )
            (task / "production.mdp").write_text(
                render_mdp(production_template, left, right, left_atoms, right_atoms, angle), encoding="utf-8"
            )
            metadata = {
                **path_spec,
                "task": task_name,
                "angle_center_deg": angle,
                "distance_center_nm": DISTANCE_CENTER_NM,
                "distance_force_kJ_mol_nm2": DISTANCE_FORCE_KJ_MOL_NM2,
                "angle_force_kJ_mol_rad2": ANGLE_FORCE_KJ_MOL_RAD2,
                "equilibration_ps": EQUILIBRATION_PS,
                "production_ps": PRODUCTION_PS,
                "left_vinyl_atoms": left_atoms,
                "right_vinyl_atoms": right_atoms,
                "axis_groups": additions,
            }
            (task / "task.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            run_script = f"""#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=4
export GMX_MAXBACKUP=-1
mkdir -p results
gmx grompp -f equilibration.mdp -c start.gro -p system.top -n index.ndx -o equilibration.tpr > equilibration_grompp.log 2>&1
gmx mdrun -deffnm equilibration -ntmpi 1 -ntomp 4 -pin on -nb gpu -pme gpu -bonded gpu -update gpu > equilibration_mdrun.log 2>&1
gmx grompp -f production.mdp -c equilibration.gro -t equilibration.cpt -p system.top -n index.ndx -o production.tpr > production_grompp.log 2>&1
gmx mdrun -deffnm production -ntmpi 1 -ntomp 4 -pin on -nb gpu -pme gpu -bonded gpu -update gpu > production_mdrun.log 2>&1
tar -czf results/{task_name}_orientation_results.tar.gz task.json index.ndx production.mdp production.tpr production.cpt production.gro production.xtc production_pullx.xvg production_pullf.xvg equilibration_grompp.log equilibration_mdrun.log production_grompp.log production_mdrun.log
"""
            (task / "run_orientation.sh").write_text(run_script, encoding="utf-8")
            (task / "run_orientation.sh").chmod(0o755)
            job = {
                "job_name": f"A4 orientation pilot {task_name}",
                "command": "bash run_orientation.sh",
                "log_file": "results/orientation_job.log",
                "backward_files": [f"results/{task_name}_orientation_results.tar.gz", "results/orientation_job.log"],
                "project_id": args.project_id,
                "machine_type": "c4_m15_1 * NVIDIA T4",
                "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
                "input_directory": str(task),
                "max_reschedule_times": 2,
                "max_run_time": 1800,
                "nnode": 1,
            }
            (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
            manifest.append(metadata)
    (output / "orientation_manifest.json").write_text(
        json.dumps({
            "purpose": "orientation sampling sufficiency pilot",
            "angle_centers_deg": ANGLE_CENTERS_DEG,
            "path_count": len(PATHS),
            "task_count": len(manifest),
            "paths": PATHS,
            "tasks": manifest,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "tasks": len(manifest), "paths": len(PATHS)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
