#!/usr/bin/env python3
"""Package PLUMED harmonic-umbrella jobs for B6 q1 distance + global Rg."""
from __future__ import annotations

import json
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.prepare_a4_q2_umbrella_pilot import pair_distance_nm  # noqa: E402

SOURCE = ROOT / "b6_q1_two_cv_metad_pilot_20261006"
OUT = ROOT / "b6_q1_distance_rg_umbrella_server_inputs_20261006_v2"
SEEDS = ("seed_05000", "seed_12500")
ROUTES = (("arm6_d6_V2_V11", "d6", 2, 11), ("arm2_d2_V2_V3", "d2", 2, 3))
DISTANCES = [round(0.35 + 0.20 * i, 2) for i in range(13)]
RGS = [0.98, 1.03, 1.08]

RUNNER = r'''#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math, os, pathlib, subprocess, tarfile

ROOT=pathlib.Path(__file__).resolve().parent
GMX=os.environ.get("GMX","gmx")
NTOMP=os.environ.get("NTOMP","8")
os.environ["OMP_NUM_THREADS"]=NTOMP
DIST=[round(0.35+0.20*i,2) for i in range(13)]
RGS=[0.98,1.03,1.08]
SEEDS=["seed_05000","seed_12500"]
ROUTES=["arm6_d6_V2_V11","arm2_d2_V2_V3"]

def call(args,cwd,log):
    with open(log,"w") as f: subprocess.run(args,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)

def mdp(prod):
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
"""

def plumed(path, cv, distance, rg, output):
    d6="v2,v11"; d2="v2,v3"
    text=f"""UNITS LENGTH=nm ENERGY=kj/mol TIME=ps
WHOLEMOLECULES ENTITY0=1-438
v2: COM ATOMS=207,208
v11: COM ATOMS=205,206
v3: COM ATOMS=189,190
d6: DISTANCE ATOMS={d6} NOPBC
d2: DISTANCE ATOMS={d2} NOPBC
rg: GYRATION ATOMS=1-438 WEIGHTS=@Masses NOPBC
rest_d: RESTRAINT ARG={cv} AT={distance:.3f} KAPPA=100
rest_rg: RESTRAINT ARG=rg AT={rg:.3f} KAPPA=500
PRINT ARG=d6,d2,rg,rest_d.bias,rest_rg.bias STRIDE=250 FILE={output}
FLUSH STRIDE=500
"""
    path.write_text(text)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--only-path", choices=ROUTES)
    ap.add_argument("--max-windows", type=int, default=0, help="pilot limit; 0 runs all windows")
    args=ap.parse_args()
    (ROOT/"results").mkdir(exist_ok=True)
    archive=ROOT/"b6_q1_distance_rg_umbrella_results.tar.gz"
    def archive_partial():
        with tarfile.open(archive,"w:gz") as tar: tar.add(ROOT/"results",arcname="results")
    done_total=0
    try:
      for seed in SEEDS:
       for route in ROUTES:
        if args.only_path and route != args.only_path: continue
        folder=ROOT/"paths"/f"{seed}__{route}"
        cfg=json.loads((folder/"pathway.json").read_text())
        dist_start=int(cfg["distance_start_index"]); rg_start=int(cfg["rg_start_index"])
        grid=[(i,j) for i in range(len(DIST)) for j in range(len(RGS))]
        order=sorted(grid,key=lambda p:(abs(p[0]-dist_start)+abs(p[1]-rg_start),p[0],p[1]))
        done={}; count=0
        for i,j in order:
         d,rg=DIST[i],RGS[j]; name=f"d_{d:.2f}_rg_{rg:.2f}"; w=ROOT/"results"/f"{seed}__{route}"/name
         w.mkdir(parents=True,exist_ok=True)
         if (w/"production.gro").is_file() and (w/"COLVAR_prod").is_file():
          done[(i,j)]=w/"production.gro"; continue
         if (i,j)==(dist_start,rg_start): src=folder/"start.gro"
         else:
          prior=[p for p in done if abs(p[0]-i)+abs(p[1]-j)==1]
          if not prior: prior=[p for p in done]
          if not prior: raise RuntimeError("no previously equilibrated neighboring window")
          p=min(prior,key=lambda p:abs(p[0]-i)+abs(p[1]-j)); src=done[p]
         (w/"equil.mdp").write_text(mdp(False)); (w/"production.mdp").write_text(mdp(True))
         plumed(w/"plumed_equil.dat",cfg["cv"],d,rg,"COLVAR_equil")
         plumed(w/"plumed_production.dat",cfg["cv"],d,rg,"COLVAR_prod")
         call([GMX,"grompp","-f","equil.mdp","-c",str(src),"-p",str(folder/"system.top"),"-n",str(folder/"index.ndx"),"-o","equil.tpr","-maxwarn","0"],w,w/"equil_grompp.log")
         call([GMX,"mdrun","-deffnm","equil","-ntmpi","1","-ntomp",NTOMP,"-pin","off","-nb","gpu","-pme","gpu","-bonded","cpu","-update","cpu","-plumed","plumed_equil.dat","-v"],w,w/"equil_mdrun.log")
         call([GMX,"grompp","-f","production.mdp","-c","equil.gro","-t","equil.cpt","-p",str(folder/"system.top"),"-n",str(folder/"index.ndx"),"-o","production.tpr","-maxwarn","0"],w,w/"production_grompp.log")
         call([GMX,"mdrun","-deffnm","production","-ntmpi","1","-ntomp",NTOMP,"-pin","off","-nb","gpu","-pme","gpu","-bonded","cpu","-update","cpu","-plumed","plumed_production.dat","-v"],w,w/"production_mdrun.log")
         done[(i,j)]=w/"production.gro"; count+=1; done_total+=1
         with open(ROOT/"results"/"progress.csv","a") as f: f.write(f"{seed},{route},{d:.2f},{rg:.2f},completed\n")
         if args.max_windows and count>=args.max_windows: break
        print(f"{seed} {route}: processed {count} new windows",flush=True)
    finally:
      archive_partial()
    if not args.max_windows:
      (ROOT/"results"/"COMPLETE").write_text(f"new_windows_completed={done_total}\n")
      archive_partial()

if __name__=="__main__": main()
'''

def main() -> None:
    if OUT.exists(): raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir()
    (OUT/"paths").mkdir()
    for seed in SEEDS:
        source=SOURCE/"seeds"/seed
        for route,cv,left,right in ROUTES:
            dest=OUT/"paths"/f"{seed}__{route}"
            dest.mkdir()
            shutil.copy2(SOURCE/"system.top",dest/"system.top")
            shutil.copy2(SOURCE/"index.ndx",dest/"index.ndx")
            gro=source/"npt_free.gro"
            shutil.copy2(gro,dest/"start.gro")
            mapping={"V2":[207,208],"V11":[205,206],"V3":[189,190]}
            initial=pair_distance_nm(gro,mapping,left,right)[0]
            di=min(range(len(DISTANCES)),key=lambda i:abs(DISTANCES[i]-initial))
            rg_guess=1.00 if seed=="seed_05000" else 1.04
            ri=min(range(len(RGS)),key=lambda i:abs(RGS[i]-rg_guess))
            cfg={"seed":seed,"route":route,"cv":cv,"atom_groups_1based":{"V2":[207,208],"V11":[205,206],"V3":[189,190]},
                 "initial_distance_nm":initial,"initial_rg_estimate_nm":rg_guess,"distance_start_index":di,"rg_start_index":ri,
                 "distance_centers_nm":DISTANCES,"rg_centers_nm":RGS,"distance_k_kj_mol_nm2":100,"rg_k_kj_mol_nm2":500,
                 "equilibration_ps":100,"production_ps":500}
            (dest/"pathway.json").write_text(json.dumps(cfg,indent=2)+"\n")
    (OUT/"run_rg_umbrella.py").write_text(RUNNER,encoding="utf-8")
    (OUT/"run_rg_umbrella.py").chmod(0o755)
    (OUT/"run_all_rg_umbrella.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\nexport GMX=\"${GMX:-/root/gromacs-2026.3/bin/gmx}\"\nexport NTOMP=\"${NTOMP:-8}\"\npython3 -u run_rg_umbrella.py\n",encoding="utf-8")
    (OUT/"run_all_rg_umbrella.sh").chmod(0o755)
    (OUT/"README_SERVER.md").write_text("""# B6 q1 distance + Rg umbrella inputs (PLUMED restraint, not MetaD)\n\nFour paired conditional surfaces: two independent REST2-derived q0-product seeds (5 ns and 12.5 ns), each for d6=V2–V11 and d2=V2–V3. The arm-2 site degeneracy is not applied here; use ×2 only in subsequent pathway weighting. Each surface has 13 distance centers (0.35–2.75 nm, 0.20 nm spacing) × 3 Rg centers (0.98, 1.03, 1.08 nm), 39 windows; 100 ps restrained equilibration + 500 ps production per window. Total nominal sampling: 93.6 ns. Distance K=100 and Rg K=500 kJ mol-1 nm-2. This is an exploratory 2D umbrella pilot; examine overlap and convergence before treating surfaces as quantitative.\n\n## Upload/run on the supplied server\n\nFrom the Mac project directory:\n\n```bash\nscp -P 28158 b6_q1_distance_rg_umbrella_server_inputs_20261006.tar.gz root@connect.westb.seetacloud.com:/root/\n```\n\nOn the server:\n\n```bash\ncd /root\ntar -xzf b6_q1_distance_rg_umbrella_server_inputs_20261006.tar.gz\ncd b6_q1_distance_rg_umbrella_server_inputs_20261006\nexport GMX=/root/gromacs-2026.3/bin/gmx\nexport NTOMP=8\n\"$GMX\" mdrun -h 2>&1 | grep -A1 -- -plumed\n```\n\nRun one 600-ps representative window first to validate PLUMED and T4 offload:\n\n```bash\nnohup env GMX=\"$GMX\" NTOMP=8 python3 -u run_rg_umbrella.py --only-path arm6_d6_V2_V11 --max-windows 1 > rg_preflight.log 2>&1 &\ntail -f rg_preflight.log\n```\n\nIf that completes cleanly, continue all remaining windows; completed windows are skipped:\n\n```bash\nnohup env GMX=\"$GMX\" NTOMP=8 ./run_all_rg_umbrella.sh > rg_umbrella.log 2>&1 &\ntail -f rg_umbrella.log\n```\n\nThe script writes a partial results tarball even if interrupted. Keep the resulting `b6_q1_distance_rg_umbrella_results.tar.gz` for local analysis. PLUMED is only applying harmonic `RESTRAINT` biases; there is no METAD/HILLS.\n""",encoding="utf-8")
    (OUT/"manifest.json").write_text(json.dumps({"method":"GROMACS MD + PLUMED harmonic restraints (umbrella), no metadynamics",
        "seed_count":2,"route_surfaces":4,"windows_per_surface":39,"total_windows":156,
        "total_sampling_ns":156*0.6,"distances_nm":DISTANCES,"rg_nm":RGS,"plumed_used_for":"Rg CV and harmonic restraint"},indent=2)+"\n")
    archive=OUT.with_suffix(".tar.gz")
    with tarfile.open(archive,"w:gz") as tar: tar.add(OUT,arcname=OUT.name)
    print(OUT); print(archive)

if __name__=="__main__": main()
