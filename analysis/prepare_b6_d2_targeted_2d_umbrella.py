#!/usr/bin/env python3
"""Prepare a focused 2D distance umbrella for the B6 full-switch-d2 hidden basin.

The existing one-dimensional V6--V8 umbrella found two structural basins.  At
the same V6--V8 distance, the decisive hidden coordinate is the competing
cross-arm distance V8--V10.  This bundle samples the local q1 contact region
with a small rectangular grid, for diagnosis rather than replacement of the
main q1 umbrella.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PROJECT_ID = 4377225
SOURCE = PROJECT / "results/b6_q1_umbrella_extensions_2ns_analysis_20260920/prepared"
JOBS = {20759498: 1, 20759494: 3}
LEFT, RIGHT, SECOND = 6, 8, 10
ATOM = {6: 195, 8: 199, 10: 203}
D1_CENTERS = (0.45, 0.55, 0.65)
D2_CENTERS = (0.50, 0.80, 1.10, 1.40, 1.70, 2.00)
K = 500.0
EQUIL_PS = 100
PROD_PS = 500
DT_PS = 0.002


def mdp(d1: float, d2: float, production: bool) -> str:
    nsteps = int((PROD_PS if production else EQUIL_PS) / DT_PS)
    return f"""integrator = md
dt = {DT_PS:g}
nsteps = {nsteps}
continuation = {'yes' if production else 'no'}
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
gen-vel = {'yes' if not production else 'no'}
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
pull-group1-pbcatom = {ATOM[6]}
pull-group2-pbcatom = {ATOM[8]}
pull-group3-pbcatom = {ATOM[10]}
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


def append_v10(index_text: str) -> str:
    return index_text.rstrip() + f"\n\n[ V10_midpoint ]\n{ATOM[10]} {ATOM[10] + 1}\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=PROJECT.parent / "bohrium_submit/b6_d2_targeted_2d_umbrella_20260920")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    records = []
    for job, replica in JOBS.items():
        source = SOURCE / str(job) / "pathway"
        start = source / "windows/window_000_0.350nm/production.gro"
        task_name = f"b6_d2_targeted_2d_rep{replica}"
        task = output / task_name
        pathway = task / "pathway"
        mdps = pathway / "window_mdps"
        mdps.mkdir(parents=True)
        shutil.copy2(source / "system.top", pathway / "system.top")
        shutil.copy2(start, pathway / "start.gro")
        (pathway / "index.ndx").write_text(append_v10((source / "index.ndx").read_text()), encoding="utf-8")
        rows = []
        for wi, d1 in enumerate(D1_CENTERS):
            for wj, d2 in enumerate(D2_CENTERS):
                index = wi * len(D2_CENTERS) + wj
                name = f"window_{index:03d}_d1_{d1:.2f}_d2_{d2:.2f}"
                (mdps / f"{name}_equil.mdp").write_text(mdp(d1, d2, False), encoding="utf-8")
                (mdps / f"{name}_production.mdp").write_text(mdp(d1, d2, True), encoding="utf-8")
                rows.append({"window_index": index, "window_name": name,
                             "d1_center_nm": f"{d1:.2f}", "d2_center_nm": f"{d2:.2f}"})
        with (pathway / "windows.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        metadata = {
            "pathway": task_name, "system": "B6_p_C6", "state": "B6_p_C6_q1_V1_V2_cis",
            "stage": "q1", "route_family": "full_switch_d2", "replica": replica,
            "source_job": job, "d1": "V6-V8", "d2": "V8-V10",
            "d1_centers_nm": list(D1_CENTERS), "d2_centers_nm": list(D2_CENTERS),
            "grid_windows": len(rows), "force_constant_kJ_mol_nm2": K,
            "equilibration_ps_per_window": EQUIL_PS, "production_ps_per_window": PROD_PS,
            "purpose": "diagnose and equilibrate the hidden full-switch-d2 structural basin; not a replacement for the main 1D q1 umbrella",
        }
        (pathway / "pathway.json").write_text(json.dumps(metadata, indent=2) + "\n")
        archive = f"{task_name}_results.tar.gz"
        shell = f'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
GMX=${{GMX:-gmx}}
export OMP_NUM_THREADS=${{OMP_NUM_THREADS:-8}}
mkdir -p results
status=0
for row in $(tail -n +2 pathway/windows.csv | cut -d, -f2); do
  equil=pathway/windows/$row
  mkdir -p "$equil"
  $GMX grompp -f pathway/window_mdps/${{row}}_equil.mdp -c pathway/start.gro -p pathway/system.top -n pathway/index.ndx -o "$equil/equil.tpr" -maxwarn 0
  $GMX mdrun -s "$equil/equil.tpr" -deffnm "$equil/equil" -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pme gpu -bonded gpu -update cpu -pin off
  $GMX grompp -f pathway/window_mdps/${{row}}_production.mdp -c "$equil/equil.gro" -t "$equil/equil.cpt" -p pathway/system.top -n pathway/index.ndx -o "$equil/production.tpr" -maxwarn 0
  $GMX mdrun -s "$equil/production.tpr" -deffnm "$equil/production" -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -nb gpu -pme gpu -bonded gpu -update cpu -pin off
done
tar -czf results/{archive} pathway
'''
        (task / "run_umbrella.sh").write_text(shell, encoding="utf-8"); (task / "run_umbrella.sh").chmod(0o755)
        job_spec = {"job_name": f"B6 d2 targeted 2D umbrella rep{replica}", "command": "bash run_umbrella.sh",
                    "backward_files": [f"results/{archive}"], "project_id": PROJECT_ID,
                    "machine_type": "c4_m15_1 * NVIDIA T4", "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
                    "input_directory": str(task), "max_reschedule_times": 2, "max_run_time": 4320, "nnode": 1}
        (task / "job.json").write_text(json.dumps(job_spec, indent=2) + "\n")
        records.append(metadata)
    (output / "jobs.json").write_text(json.dumps({"status": "prepared_not_submitted", "project_id": PROJECT_ID,
        "paths": len(records), "grid": [len(D1_CENTERS), len(D2_CENTERS)], "records": records}, indent=2) + "\n")
    print(json.dumps({"status": "prepared_not_submitted", "paths": len(records), "windows_per_path": len(D1_CENTERS)*len(D2_CENTERS), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
