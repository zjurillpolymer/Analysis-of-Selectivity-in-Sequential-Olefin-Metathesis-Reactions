#!/usr/bin/env python3
"""Pack fresh DCM boxes, audit PBC contacts, prepare six T4 jobs, preserve old inputs."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"bohrium_submit/b6_postq0_multistart_umbrella_fixed_20261012"
OLD=ROOT/"bohrium_submit/b6_postq0_multistart_umbrella_20261011"
SEL=ROOT/"results/b6_postq0_unbiased_30ns_20261011/timescale_analysis/umbrella_seed_selection/structures"
TOP=ROOT/"b6_q1_two_cv_metad_pilot_20261006/system.top"
TEMPLATE=ROOT/"b6_q1_two_cv_metad_pilot_20261006/seeds/seed_05000/npt_free.gro"
DCM=ROOT/"gromacs_dcm/models/ch2cl2_dc/dcm.pdb"
GMX="/opt/homebrew/bin/gmx"
PACKMOL="/opt/homebrew/Caskroom/miniconda/base/envs/chem_dl/bin/packmol"
RUNNER=ROOT/"analysis/run_b6_multistart_umbrella_safe.py"

def command(cmd,cwd,log,stdin=None):
    with log.open("w") as f:
        r=subprocess.run(cmd,cwd=cwd,input=stdin,text=True,stdout=f,stderr=subprocess.STDOUT)
    if r.returncode: raise RuntimeError(f"{cmd}: {log}")

def gro(path):
    lines=path.read_text().splitlines(); n=int(lines[1])
    x=np.array([[float(l[k:k+8]) for k in (20,28,36)] for l in lines[2:2+n]])
    b=np.array(list(map(float,lines[n+2].split())))
    return lines,x,b

def topology():
    text=TOP.read_text()
    matches=list(re.finditer(r"(?m)^\s*\[\s*moleculetype\s*\]",text))
    if len(matches)!=2: raise ValueError("expected solute and DCM molecule definitions")
    atoms=text[matches[0].start():matches[1].start()]
    atomsec=re.search(r"\[\s*atoms\s*\](.*?)(?=\[)",atoms,re.S).group(1)
    rows=[l.split(";")[0].split() for l in atomsec.splitlines() if l.split(";")[0].strip()]
    if len(rows)!=438: raise ValueError("wrong solute atom count")
    posres="\n#ifdef POSRES\n[ position_restraints ]\n"
    posres+="".join(f"{r[0]} 1 1000 1000 1000\n" for r in rows if float(r[7])>2)
    posres+="#endif\n\n"
    return text[:matches[1].start()]+posres+text[matches[1].start():]

def build_box(candidate,out):
    out.mkdir()
    source=SEL/f"{candidate}.gro"
    ls,x,b=gro(source)
    lt,xt,bt=gro(TEMPLATE)
    if len(x)!=438 or len(b)!=3: raise ValueError("unexpected source")
    # Use source box, translate only; keep the whole-molecule conformation.
    x=x+(b/2-(x.min(0)+x.max(0))/2)
    if (x.min(0)<.2).any() or (x.max(0)>b-.2).any():
        raise ValueError("solute too close to periodic boundaries; explicit larger-box decision required")
    atoms=[lt[2+i][:20]+"".join(f"{v:8.3f}" for v in pos) for i,pos in enumerate(x)]
    (out/"solute.gro").write_text("whole solute selected frame\n438\n"+"\n".join(atoms)+"\n"+" ".join(map(str,b))+"\n")
    command([GMX,"editconf","-f","solute.gro","-o","solute.pdb"],out,out/"editconf.log")
    shutil.copy2(DCM,out/"dcm.pdb")
    hi=b*10-1 # Packmol coordinates and tolerances are ANGSTROMS.
    inp=("tolerance 2.0\nfiletype pdb\noutput packed.pdb\nseed 20261012\n"
         "structure solute.pdb\n number 1\n fixed 0 0 0 0 0 0\nend structure\n"
         f"structure dcm.pdb\n number 2927\n resnumbers 3\n inside box 1 1 1 {hi[0]} {hi[1]} {hi[2]}\nend structure\n")
    (out/"packmol.inp").write_text(inp)
    with (out/"packmol.inp").open() as pack_input, (out/"packmol.log").open("w") as pack_log:
        result=subprocess.run([PACKMOL],cwd=out,stdin=pack_input,text=True,stdout=pack_log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"Packmol failed: {out/'packmol.log'}")
    if "Success!" not in (out/"packmol.log").read_text():
        raise RuntimeError("Packmol did not report success")
    packed=[l for l in (out/"packed.pdb").read_text().splitlines() if l.startswith(("ATOM  ","HETATM"))]
    if len(packed)!=9219: raise ValueError("wrong packed atom count")
    names=[l[12:16].strip() for l in packed]
    expected=[l[10:15].strip() for l in lt[2:2+9219]]
    if names!=expected: raise ValueError("atom identity/order mismatch")
    xp=np.array([[float(l[k:k+8])/10 for k in (30,38,46)] for l in packed])
    if np.max(np.abs(xp[:438]-x))>.001: raise ValueError("solute moved during packing")
    # Fixed-width original labels preserve topology ordering; drop old velocities.
    lines=[lt[2+i][:20]+"".join(f"{v:8.3f}" for v in pos) for i,pos in enumerate(xp)]
    (out/"start.gro").write_text("Fresh Packmol DCM box\n9219\n"+"\n".join(lines)+"\n"+" ".join(map(str,b))+"\n")
    _,xp,b=gro(out/"start.gro") # audit exactly the submitted rounded coordinates
    tree=cKDTree(xp[438:]%b,boxsize=b)
    ds,_=tree.query(xp[:438]%b)
    # Exclude same-molecule solvent contacts; rigid DCM internal geometry is retained.
    pairs=np.array(list(tree.query_pairs(.19)),dtype=int)
    if pairs.size:
        pairs=pairs[pairs[:,0]//3!=pairs[:,1]//3]
    if ds.min()<.19 or len(pairs): raise ValueError("periodic intermolecular overlap check failed")
    audit={"solute_atoms":438,"solvent_atoms":8781,"dcm_molecules":2927,"box_nm":b.tolist(),
           "min_solute_solvent_nm":float(ds.min()),"solvent_intermolecular_pairs_under_0_19nm":len(pairs),
           "source":str(source),"packmol_tolerance_A":2.0,"atom_order_verified":True}
    (out/"geometry_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    (out/"em.mdp").write_text("""integrator = steep
nsteps = 50000
emtol = 100
emstep = 0.005
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
pbc = xyz
constraints = h-bonds
constraint-algorithm = lincs
lincs-order = 4
""")
    print(out.name,audit,flush=True)

def main():
    if OUT.exists(): raise FileExistsError(OUT)
    OUT.mkdir()
    manifest=json.loads((OLD/"manifest.json").read_text())
    boxes={}
    for j in manifest["jobs"]:
        seed=j["replica"]
        if seed not in boxes:
            d=OUT/("_box_"+seed); build_box(j["candidate_id"],d); boxes[seed]=d
    jobs=[]
    top=topology()
    for j in manifest["jobs"]:
        d=OUT/j["name"]; d.mkdir(); src=OLD/j["name"]
        for name in ("start.gro","geometry_audit.json"): shutil.copy2(boxes[j["replica"]]/name,d/name)
        (d/"system.top").write_text(top); shutil.copy2(src/"index.ndx",d/"index.ndx")
        shutil.copy2(RUNNER,d/"run_umbrella.py")
        arm6="arm6" in j["path"]
        cfg={"seed":700000+int(j["candidate_id"].split("_")[-1].replace("ps","")),
             "group":"V11_midpoint" if arm6 else "V3_midpoint","atom":205 if arm6 else 189,
             "centers_nm":manifest["centers_nm"],"equilibration_ns":1.0,"production_ns":3.0,
             "preparation":"heavy-atom restrained EM, 100 ps NVT, 500 ps NPT, 100 ps release"}
        (d/"config.json").write_text(json.dumps(cfg,indent=2)+"\n")
        job=json.loads((src/"job.json").read_text())
        job.update(job_name="B6 repaired "+j["name"],command="python3 -u run_umbrella.py",
                   input_directory=str(d),max_reschedule_times=0)
        (d/"job.json").write_text(json.dumps(job,indent=2)+"\n")
        jobs.append(j|{"input_directory":str(d)})
    manifest.update(jobs=jobs,method="fresh Packmol DCM, EM/NVT/NPT, gradual window seeding, native 1D umbrella",
                    supersedes=str(OLD),preparation_ns=.7,window_ramp_min_ps=100,
                    window_equil_ns=1.0,window_production_ns=3.0)
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("Prepared",len(jobs),"jobs",flush=True)

if __name__=="__main__": main()

