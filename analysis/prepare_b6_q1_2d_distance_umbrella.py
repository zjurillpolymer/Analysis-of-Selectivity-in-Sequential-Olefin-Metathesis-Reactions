#!/usr/bin/env python3
"""Prepare a small native-GROMACS 2D umbrella pilot for the B6 q1 channels."""
from __future__ import annotations

import json
import math
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.prepare_a4_q2_umbrella_pilot import pair_distance_nm  # noqa: E402
SOURCE = ROOT / "b6_q1_two_cv_metad_pilot_20261006"
OUT = ROOT / "b6_q1_2d_distance_umbrella_bohrium_20261006_v6"
SEEDS = ("seed_05000", "seed_12500")
CENTERS = [round(0.35 + 0.30 * i, 2) for i in range(9)]

def write_runner(path: Path) -> None:
    path.write_text(r'''#!/usr/bin/env bash
set -euo pipefail
export GMX_MAXBACKUP=-1
GMX="${GMX:-gmx}"
NTOMP="${NTOMP:-4}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p results/windows
archive_partial() { tar -czf b6_q1_2d_distance_surface_results.tar.gz results || true; }
trap archive_partial EXIT
python3 - <<'PY'
import csv, json, math, os, pathlib, subprocess
root=pathlib.Path.cwd(); gmx=os.environ.get("GMX","gmx"); ntomp=os.environ.get("NTOMP","4")
centers=[round(0.35+0.30*i,2) for i in range(9)]
grid=[(i,j) for i in range(9) for j in range(9)]
start=json.loads((root/'grid_start.json').read_text())
si,sj=start['i'],start['j']
order=sorted(grid,key=lambda p:(abs(p[0]-si)+abs(p[1]-sj),p[0],p[1]))
done={}; rows=[]
def call(args,log):
    with open(log,"w") as f: subprocess.run(args,cwd=log.parent,stdout=f,stderr=subprocess.STDOUT,check=True)
def mdp(ci,cj,prod):
    nsteps=250000 if prod else 50000
    return f"""integrator = md
dt = 0.002
nsteps = {nsteps}
continuation = {'yes' if prod else 'no'}
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
compressibility = 1e-4
gen-vel = {'no' if prod else 'yes'}
gen-temp = 313.15
gen-seed = -1
nstxout = 0
nstvout = 0
nstfout = 0
nstxout-compressed = {2500 if prod else 0}
compressed-x-grps = Solute
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
pull-group1-name = V2_midpoint
pull-group2-name = V11_midpoint
pull-group3-name = V3_midpoint
pull-group1-pbcatom = 207
pull-group2-pbcatom = 205
pull-group3-pbcatom = 189
pull-pbc-ref-prev-step-com = yes
pull-coord1-type = umbrella
pull-coord1-geometry = distance
pull-coord1-groups = 1 2
pull-coord1-dim = Y Y Y
pull-coord1-start = no
pull-coord1-init = {centers[ci]:.3f}
pull-coord1-rate = 0
pull-coord1-k = 100
pull-coord2-type = umbrella
pull-coord2-geometry = distance
pull-coord2-groups = 1 3
pull-coord2-dim = Y Y Y
pull-coord2-start = no
pull-coord2-init = {centers[cj]:.3f}
pull-coord2-rate = 0
pull-coord2-k = 100
pull-nstxout = 250
pull-nstfout = 250
"""
for ix,(i,j) in enumerate(order):
    name=f'w_d6_{centers[i]:.2f}_d2_{centers[j]:.2f}'
    w=root/'results/windows'/name; w.mkdir(parents=True,exist_ok=True)
    if (w/'production.gro').is_file() and (w/'pullx.xvg').is_file():
        done[(i,j)]=w/'production.gro'; continue
    if (i,j)==(si,sj): src=root/'start.gro'
    else:
        prior=[p for p in done if abs(p[0]-i)+abs(p[1]-j)==1]
        if not prior: prior=[p for p in done]
        if not prior: raise RuntimeError('no adjacent seed window available')
        p=min(prior,key=lambda p:abs(p[0]-i)+abs(p[1]-j)); src=done[p]
    (w/'equil.mdp').write_text(mdp(i,j,False)); (w/'production.mdp').write_text(mdp(i,j,True))
    call([gmx,'grompp','-f','equil.mdp','-c',str(src),'-p',str(root/'system.top'),'-n',str(root/'index.ndx'),'-o','equil.tpr','-maxwarn','0'],w/'equil_grompp.log')
    call([gmx,'mdrun','-deffnm','equil','-ntmpi','1','-ntomp',ntomp,'-pin','off','-nb','gpu','-pme','gpu','-bonded','cpu','-update','cpu','-v'],w/'equil_mdrun.log')
    call([gmx,'grompp','-f','production.mdp','-c','equil.gro','-t','equil.cpt','-p',str(root/'system.top'),'-n',str(root/'index.ndx'),'-o','production.tpr','-maxwarn','0'],w/'production_grompp.log')
    call([gmx,'mdrun','-deffnm','production','-ntmpi','1','-ntomp',ntomp,'-pin','off','-nb','gpu','-pme','gpu','-bonded','cpu','-update','cpu','-px','pullx.xvg','-pf','pullf.xvg','-v'],w/'production_mdrun.log')
    done[(i,j)]=w/'production.gro'
    rows.append([name,centers[i],centers[j],str(src.relative_to(root))])
with open(root/'results/window_manifest.csv','w') as f:
    wr=csv.writer(f); wr.writerow(['window','d6_center_nm','d2_center_nm'])
    for (i,j) in sorted(done): wr.writerow([f'w_d6_{centers[i]:.2f}_d2_{centers[j]:.2f}',centers[i],centers[j]])
with open(root/'results/COMPLETE','w') as f: f.write(f'completed={len(done)} of 81 windows\n')
PY
''', encoding="utf-8")
    path.chmod(0o755)

def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir()
    manifests=[]
    for seed in SEEDS:
        dest=OUT/seed; dest.mkdir()
        shutil.copy2(SOURCE/"system.top",dest/"system.top")
        shutil.copy2(SOURCE/"index.ndx",dest/"index.ndx")
        shutil.copy2(SOURCE/"seeds"/seed/"npt_free.gro",dest/"start.gro")
        gro=dest/"start.gro"
        mapping={"V2":[207,208],"V11":[205,206],"V3":[189,190]}
        d6=pair_distance_nm(gro,mapping,2,11)[0]
        d2=pair_distance_nm(gro,mapping,2,3)[0]
        si=min(range(9),key=lambda i:abs(CENTERS[i]-d6))
        sj=min(range(9),key=lambda j:abs(CENTERS[j]-d2))
        (dest/"grid_start.json").write_text(json.dumps({"d6_initial_nm":d6,"d2_initial_nm":d2,"i":si,"j":sj},indent=2)+"\n")
        idx=(dest/"index.ndx").read_text()
        for name,atoms in (("V2_midpoint","207 208"),("V11_midpoint","205 206"),("V3_midpoint","189 190")):
            if f"[ {name} ]" not in idx:
                idx += f"\n[ {name} ]\n{atoms}\n"
        (dest/"index.ndx").write_text(idx)
        write_runner(dest/"run_surface.sh")
        job={"job_name":f"B6 q1 2D distance umbrella {seed}","command":"bash run_surface.sh",
             "log_file":"job.log","backward_files":["b6_q1_2d_distance_surface_results.tar.gz"],
             "project_id":4377225,"machine_type":"c4_m15_1 * NVIDIA T4",
             "image_address":"registry.dp.tech/dptech/gromacs:2024.2","input_directory":str(dest),
             "max_reschedule_times":1,"max_run_time":4320,"nnode":1}
        (dest/"job.json").write_text(json.dumps(job,indent=2)+"\n")
        manifests.append({"seed":seed,"windows":81,"centers_nm":CENTERS,"spacing_nm":0.30,
                          "k_kj_mol_nm2":100,"equil_ps":100,"production_ps":500,
                          "sampling_ns_per_job":81*0.6,"routes":"d6=V2-V11; d2=V2-V3",
                          "initial_d6_nm":d6,"initial_d2_nm":d2,"first_grid_center_nm":[CENTERS[si],CENTERS[sj]]})
    (OUT/"manifest.json").write_text(json.dumps({"method":"native GROMACS 2D umbrella",
        "plumed_used":False,"seed_jobs":manifests,"total_sampling_ns":sum(x["sampling_ns_per_job"] for x in manifests),
        "warning":"coarse 0.30-nm pilot; assess 2D overlap/convergence before interpreting absolute free energies",
        "arm2_site_multiplicity_applied":False},indent=2)+"\n")
    (OUT/"README.md").write_text("""# B6 q1 2-distance umbrella pilot\n\nTwo T4 jobs; one independent 9x9 joint-distance grid per full-solvent seed (81 windows/job). CVs: d6=V2–V11 and d2=V2–V3. Each cell has 100 ps equilibration + 500 ps production, k=100 kJ mol-1 nm-2, 0.30 nm coarse spacing. Total sampled time is 97.2 ns. GROMACS native pull restraints; no PLUMED/MetaD. Arm-2 multiplicity is not applied to sampling, only later pathway weighting.\n\nEach seed runs windows outward from the central cell using the nearest completed neighbor as the next starting structure. Check every `equil_grompp.log` and production log; successful completion writes results archive and COMPLETE. A genuine 2D PMF requires joint-coordinate WHAM/MBAR, not separate 1D gmx wham calls.\n""",encoding="utf-8")
    archive=OUT.with_suffix(".tar.gz")
    with tarfile.open(archive,"w:gz") as tar: tar.add(OUT,arcname=OUT.name)
    print(OUT); print(archive)

if __name__=="__main__": main()
