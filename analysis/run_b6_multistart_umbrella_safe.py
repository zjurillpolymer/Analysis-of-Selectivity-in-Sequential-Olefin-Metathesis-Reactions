#!/usr/bin/env python3
"""Auditable native-GROMACS umbrella runner; no Amber/Packmol needed remotely."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parent
STATUS = {"status": "running", "stage": "initialization"}
BASE = dict(
    integrator="md", dt="0.002", nsteps="500000", continuation="yes",
    constraints="h-bonds", **{"constraint-algorithm": "lincs", "lincs-order": "4", "lincs-iter": "2"},
    **{"cutoff-scheme": "Verlet"}, nstlist="20", rlist="1.2", coulombtype="PME",
    rcoulomb="1.2", **{"pme-order": "4"}, fourierspacing="0.12", vdwtype="Cut-off",
    rvdw="1.2", DispCorr="EnerPres", tcoupl="V-rescale", **{"tc-grps": "System"},
    **{"tau-t": "1.0", "ref-t": "313.15"}, pcoupl="C-rescale", pcoupltype="isotropic",
    **{"tau-p": "5.0", "ref-p": "1.0"}, compressibility="1e-4", pbc="xyz",
    **{"gen-vel": "no"}, nstxout="0", nstvout="0", nstfout="0",
    **{"nstxout-compressed": "500", "compressed-x-grps": "System"},
    nstenergy="500", nstlog="5000", nstcalcenergy="100",
    **{"comm-mode": "Linear"}, nstcomm="100",
)

def save_status(**values):
    STATUS.update(values)
    (ROOT / "run_status.json").write_text(json.dumps(STATUS, indent=2) + "\n")

def command(args, log, stdin=None):
    with log.open("w") as f:
        r = subprocess.run(args, cwd=ROOT, input=stdin, text=True, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"Command exit {r.returncode}: {' '.join(map(str,args))}; log={log.relative_to(ROOT)}")

def params(path, values):
    path.write_text("".join(f"{k} = {v}\n" for k,v in values.items()))

def coords(path):
    lines=path.read_text().splitlines()
    xyz=[tuple(float(l[k:k+8]) for k in (20,28,36)) for l in lines[2:440]]
    box=list(map(float,lines[2+int(lines[1])].split()))
    if len(box)!=3: raise ValueError("Expected orthorhombic box")
    return xyz,box

def distance(path, partner):
    xyz,box=coords(path)
    a=[sum(xyz[i][k] for i in (206,207))/2 for k in range(3)]
    b=[sum(xyz[i][k] for i in (partner-1,partner))/2 for k in range(3)]
    delta=[a[k]-b[k] for k in range(3)]
    delta=[v-L*round(v/L) for v,L in zip(delta,box)]
    return sum(v*v for v in delta)**.5

def pull(config, center, rate=0):
    return {
        "pull":"yes","pull-ncoords":"1","pull-ngroups":"2",
        "pull-group1-name":"V2_midpoint","pull-group2-name":config["group"],
        "pull-group1-pbcatom":"207","pull-group2-pbcatom":str(config["atom"]),
        "pull-pbc-ref-prev-step-com":"yes","pull-coord1-type":"umbrella",
        "pull-coord1-geometry":"distance","pull-coord1-groups":"1 2",
        "pull-coord1-dim":"Y Y Y","pull-coord1-start":"no",
        "pull-coord1-init":str(center),"pull-coord1-rate":str(rate),
        "pull-coord1-k":"500","pull-nstxout":"250","pull-nstfout":"250",
    }

def simulate(name, values, start, cpt, args, restrain=False):
    prefix=ROOT/name
    prefix.parent.mkdir(parents=True, exist_ok=True)
    save_status(stage=name)
    finished=Path(str(prefix)+".done.json")
    if finished.exists():
        return Path(str(prefix)+".gro"),Path(str(prefix)+".cpt")
    mdp=Path(str(prefix)+".mdp"); tpr=Path(str(prefix)+".tpr")
    params(mdp, values)
    if not tpr.exists():
        cmd=[args.gmx,"grompp","-f",str(mdp),"-c",str(start),"-p",str(ROOT/"system.top"),
             "-n",str(ROOT/"index.ndx"),"-o",str(tpr),"-po",str(prefix)+"_processed.mdp","-maxwarn","0"]
        if restrain: cmd+=["-r",str(ROOT/"start.gro")]
        if cpt is not None: cmd+=["-t",str(cpt)]
        command(cmd,Path(str(prefix)+"_grompp.log"))
    cmd=[args.gmx,"mdrun","-s",str(tpr),"-deffnm",str(prefix),"-ntmpi","1",
         "-ntomp",str(args.ntomp),"-pin","off","-nb","cpu" if args.cpu else "gpu",
         "-pme","cpu","-bonded","cpu","-update","cpu","-cpt","5","-v"]
    checkpoint=Path(str(prefix)+".cpt")
    if checkpoint.exists(): cmd+=["-cpi",str(checkpoint),"-append"]
    if values.get("pull")=="yes":
        cmd+=["-px",str(prefix)+"_pullx.xvg","-pf",str(prefix)+"_pullf.xvg"]
    command(cmd,Path(str(prefix)+"_mdrun.log"))
    log=Path(str(prefix)+".log").read_text(errors="replace")
    if "LINCS WARNING" in log or "not finite" in log:
        raise RuntimeError(f"Numerical-instability check failed: {prefix}.log")
    if values["integrator"]=="steep" and "converged to Fmax" not in log:
        raise RuntimeError(f"EM did not reach requested Fmax: {prefix}.log")
    if not Path(str(prefix)+".gro").is_file():
        raise RuntimeError(f"Missing final coordinates: {prefix}")
    finished.write_text(json.dumps({"finished":True,"nsteps":values["nsteps"]})+"\n")
    return Path(str(prefix)+".gro"), checkpoint

def run(args):
    cfg=json.loads((ROOT/"config.json").read_text())
    os.environ["OMP_NUM_THREADS"]=str(args.ntomp)
    # Never carry a stale failure marker into a continuation.
    save_status(status="running", error=None)
    common=BASE.copy()
    em={k:v for k,v in common.items() if k not in (
        "tcoupl","tc-grps","tau-t","ref-t","pcoupl","pcoupltype","tau-p","ref-p",
        "compressibility","gen-vel","continuation","comm-mode","nstcomm")}
    em.update(integrator="steep",nsteps="50000",emtol="100",emstep="0.005",
              define="-DPOSRES",**{"nstxout-compressed":"0"})
    emgro,_=simulate("preparation/em",em,ROOT/"start.gro",None,args,True)
    nvt=common|dict(nsteps=str(1000 if args.smoke else 50000),pcoupl="no",continuation="no",
                   define="-DPOSRES",**{"gen-vel":"yes","gen-temp":"313.15","gen-seed":str(cfg["seed"])})
    nvtgro,nvtcpt=simulate("preparation/nvt",nvt,emgro,None,args,True)
    npt=common|dict(nsteps=str(1000 if args.smoke else 250000),define="-DPOSRES",
                   **{"refcoord-scaling":"com"})
    nptgro,nptcpt=simulate("preparation/npt",npt,nvtgro,nvtcpt,args,True)
    free=common|dict(nsteps=str(1000 if args.smoke else 50000))
    start,cpt=simulate("preparation/release",free,nptgro,nptcpt,args)
    d0=distance(start,cfg["atom"])
    centers=cfg["centers_nm"]; nearest=min(centers,key=lambda x:abs(x-d0))
    # Two outward chains sharing one central equilibrated window, no long jump.
    order=[nearest]+sorted([x for x in centers if x<nearest],reverse=True)+sorted(x for x in centers if x>nearest)
    if args.smoke: order=order[:1]
    (ROOT/"window_order.json").write_text(json.dumps({"start_distance_nm":d0,"centers":order},indent=2)+"\n")
    center_start=None
    (ROOT/"progress.csv").write_text("center_nm,status\n")
    for target in order:
        if target>nearest and center_start is not None:
            start,cpt=center_start
            center_start=None
        dstart=distance(start,cfg["atom"])
        duration=2 if args.smoke else max(100,abs(target-dstart)/.002)
        steps=round(duration/.002)
        values=common|dict(nsteps=str(steps))|pull(cfg,dstart,(target-dstart)/(steps*.002))
        # Each stage starts its time origin at zero; pull center and velocity are explicit.
        prefix=f"windows/window_{target:.2f}nm"
        rampgro,rampcpt=simulate(prefix+"/ramp",values,start,cpt,args)
        eq=common|dict(nsteps=str(1000 if args.smoke else 500000))|pull(cfg,target)
        eqgro,eqcpt=simulate(prefix+"/equil",eq,rampgro,rampcpt,args)
        production=common|dict(nsteps=str(1000 if args.smoke else 1500000))|pull(cfg,target)
        pg,pc=simulate(prefix+"/production",production,eqgro,eqcpt,args)
        with (ROOT/"progress.csv").open("a") as f: f.write(f"{target:.2f},completed\n")
        if target==nearest: center_start=(eqgro,eqcpt)
        start,cpt=eqgro,eqcpt
    save_status(status="smoke_passed" if args.smoke else "completed",stage="finished")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gmx",default=os.environ.get("GMX","gmx"))
    p.add_argument("--ntomp",type=int,default=int(os.environ.get("NTOMP","8")))
    p.add_argument("--cpu",action="store_true")
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    try:
        run(args)
    except Exception as e:
        save_status(status="failed",error=str(e))
        raise
    finally:
        # Archive logs and preparation too, including failures before the first window.
        with tarfile.open(ROOT/"umbrella_pilot_results.tar.gz","w:gz") as t:
            for name in ("start.gro","system.top","index.ndx","config.json","run_umbrella.py",
                         "geometry_audit.json","run_status.json","progress.csv","window_order.json",
                         "preparation","windows"):
                if (ROOT/name).exists(): t.add(ROOT/name,arcname=name)
if __name__=="__main__": main()

