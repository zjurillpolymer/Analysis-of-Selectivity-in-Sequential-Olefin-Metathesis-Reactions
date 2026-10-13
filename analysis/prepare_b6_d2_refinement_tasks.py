#!/usr/bin/env python3
"""Prepare intermediate-window and continuation jobs for B6 d2 2D sampling."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "results/b6_d2_targeted_2d_umbrella_results_20260920/raw"
OUTPUT_DEFAULT = PROJECT.parent / "bohrium_submit/b6_d2_refinement_20260920"
PROJECT_ID = 4377225
JOBS = {20760833: "compact", 20760834: "extended"}
D1 = (0.45, 0.55, 0.65)
OLD_D2 = (0.50, 0.80, 1.10, 1.40, 1.70, 2.00)
NEW_D2 = (0.65, 0.95, 1.25, 1.55, 1.85)
K = 500.0
DT = 0.002


def mdp(d1: float, d2: float, ps: int, continuation: bool, gen_vel: bool) -> str:
    return f"""integrator = md
dt = {DT:g}
nsteps = {int(ps / DT)}
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
pull-ngroups = 3
pull-print-components = no
pull-group1-name = V6_midpoint
pull-group2-name = V8_midpoint
pull-group3-name = V10_midpoint
pull-group1-pbcatom = 195
pull-group2-pbcatom = 199
pull-group3-pbcatom = 203
pull-pbc-ref-prev-step-com = yes
pull-coord1-type = umbrella
pull-coord1-geometry = distance
pull-coord1-groups = 1 2
pull-coord1-dim = Y Y Y
pull-coord1-start = no
pull-coord1-init = {d1:.6f}
pull-coord1-rate = 0
pull-coord1-k = {K:.1f}
pull-coord2-type = umbrella
pull-coord2-geometry = distance
pull-coord2-groups = 2 3
pull-coord2-dim = Y Y Y
pull-coord2-start = no
pull-coord2-init = {d2:.6f}
pull-coord2-rate = 0
pull-coord2-k = {K:.1f}
pull-nstxout = 250
pull-nstfout = 250
"""


def run_script(archive: str, rows: list[dict[str, object]], kind: str) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", 'cd "$(dirname "${BASH_SOURCE[0]}")"',
             'GMX=${GMX:-gmx}', 'export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}', 'mkdir -p results']
    for row in rows:
        name = row["window_name"]
        if kind == "refine":
            lines += [
                f"mkdir -p pathway/windows/{name}",
                f"$GMX grompp -f pathway/mdps/{name}_equil.mdp -c pathway/starts/{name}.gro -p pathway/system.top -n pathway/index.ndx -o pathway/windows/{name}/equil.tpr -maxwarn 0",
                f"$GMX mdrun -s pathway/windows/{name}/equil.tpr -deffnm pathway/windows/{name}/equil -ntmpi 1 -ntomp $OMP_NUM_THREADS -nb gpu -pme gpu -bonded gpu -update cpu -pin off",
                f"$GMX grompp -f pathway/mdps/{name}_production.mdp -c pathway/windows/{name}/equil.gro -t pathway/windows/{name}/equil.cpt -p pathway/system.top -n pathway/index.ndx -o pathway/windows/{name}/production.tpr -maxwarn 0",
                f"$GMX mdrun -s pathway/windows/{name}/production.tpr -deffnm pathway/windows/{name}/production -ntmpi 1 -ntomp $OMP_NUM_THREADS -nb gpu -pme gpu -bonded gpu -update cpu -pin off",
            ]
        else:
            lines += [
                f"mkdir -p pathway/windows/{name}",
                f"$GMX grompp -f pathway/mdps/{name}_extension.mdp -c pathway/starts/{name}.gro -t pathway/starts/{name}.cpt -p pathway/system.top -n pathway/index.ndx -o pathway/windows/{name}/extension.tpr -maxwarn 0",
                f"$GMX mdrun -s pathway/windows/{name}/extension.tpr -deffnm pathway/windows/{name}/extension -ntmpi 1 -ntomp $OMP_NUM_THREADS -nb gpu -pme gpu -bonded gpu -update cpu -pin off",
            ]
    lines += [f"tar -czf results/{archive} pathway", ""]
    return "\n".join(lines)


def make_task(output: Path, job: int, seed: str, kind: str) -> dict[str, object]:
    source = RAW / str(job) / "pathway"
    name = f"b6_d2_{'refine' if kind == 'refine' else 'extend'}_{seed}"
    task = output / name; pathway = task / "pathway"; (pathway / "mdps").mkdir(parents=True); (pathway / "starts").mkdir()
    shutil.copy2(source / "system.top", pathway / "system.top"); shutil.copy2(source / "index.ndx", pathway / "index.ndx")
    source_rows = list(csv.DictReader((source / "windows.csv").open()))
    rows = []
    if kind == "refine":
        for d1 in D1:
            candidates = [r for r in source_rows if abs(float(r["d1_center_nm"]) - d1) < 1e-6]
            for d2 in NEW_D2:
                nearest = min(candidates, key=lambda r: abs(float(r["d2_center_nm"]) - d2))
                name_w = f"window_refine_d1_{d1:.2f}_d2_{d2:.2f}"
                shutil.copy2(source / "windows" / nearest["window_name"] / "production.gro", pathway / "starts" / f"{name_w}.gro")
                (pathway / "mdps" / f"{name_w}_equil.mdp").write_text(mdp(d1, d2, 100, False, True))
                (pathway / "mdps" / f"{name_w}_production.mdp").write_text(mdp(d1, d2, 500, True, False))
                rows.append({"window_name": name_w, "d1_center_nm": d1, "d2_center_nm": d2, "source_window": nearest["window_name"]})
    else:
        for old in source_rows:
            name_w = old["window_name"]
            shutil.copy2(source / "windows" / name_w / "production.gro", pathway / "starts" / f"{name_w}.gro")
            shutil.copy2(source / "windows" / name_w / "production.cpt", pathway / "starts" / f"{name_w}.cpt")
            (pathway / "mdps" / f"{name_w}_extension.mdp").write_text(mdp(float(old["d1_center_nm"]), float(old["d2_center_nm"]), 1500, True, False))
            rows.append({"window_name": name_w, "d1_center_nm": float(old["d1_center_nm"]), "d2_center_nm": float(old["d2_center_nm"]), "source_window": name_w})
    with (pathway / "windows.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    metadata = {"pathway": name, "system": "B6_p_C6", "stage": "q1", "route_family": "full_switch_d2",
                "seed": seed, "source_job": job, "kind": kind, "d1": "V6-V8", "d2": "V8-V10",
                "new_d2_centers_nm": list(NEW_D2) if kind == "refine" else [],
                "production_ps": 500 if kind == "refine" else 1500, "grid_windows": len(rows),
                "purpose": "improve d2 overlap and test contact-map convergence from two basin seeds"}
    (pathway / "pathway.json").write_text(json.dumps(metadata, indent=2) + "\n")
    archive = f"{name}_results.tar.gz"; (task / "run.sh").write_text(run_script(archive, rows, kind)); (task / "run.sh").chmod(0o755)
    spec = {"job_name": f"B6 d2 {kind} {seed}", "command": "bash run.sh", "backward_files": [f"results/{archive}"],
            "project_id": PROJECT_ID, "machine_type": "c4_m15_1 * NVIDIA T4", "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
            "input_directory": str(task), "max_reschedule_times": 2, "max_run_time": 4320 if kind == "refine" else 7200, "nnode": 1}
    (task / "job.json").write_text(json.dumps(spec, indent=2) + "\n")
    return {**metadata, "task": str(task), "archive": archive}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT); args = parser.parse_args()
    output = args.output.resolve()
    if output.exists(): raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    records = [make_task(output, job, seed, kind) for job, seed in JOBS.items() for kind in ("refine", "extend")]
    (output / "jobs.json").write_text(json.dumps({"status": "prepared_not_submitted", "project_id": PROJECT_ID, "records": records}, indent=2) + "\n")
    print(json.dumps({"status": "prepared_not_submitted", "tasks": len(records), "output": str(output),
                      "refine_windows_per_task": 15, "extend_windows_per_task": 18}, indent=2))


if __name__ == "__main__": main()
