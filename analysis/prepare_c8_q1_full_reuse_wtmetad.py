#!/usr/bin/env python3
"""Prepare a targeted WT-MetaD supplement for A4-m-C8 q1 full-reuse."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {
    "rep1": ROOT / "bohrium_submit/a4_m_c5_c8_q1_three_class_umbrella/c8_q1_full_reuse_v5_v7_rep1/pathway",
    "rep2": ROOT / "bohrium_submit/a4_m_c5_c8_q1_three_class_umbrella/c8_q1_full_reuse_v5_v7_rep2/pathway",
    "rep3": ROOT / "bohrium_submit/a4_m_c5_c8_q1_three_class_umbrella_rep3/c8_q1_full_reuse_v5_v7_rep3/pathway",
}


def metad_dat() -> str:
    return """# Targeted supplement only: global conformational exploration.
# The V5--V7 reaction distance is deliberately NOT biased here.
UNITS LENGTH=nm TIME=ps ENERGY=kJ/mol
WHOLEMOLECULES ENTITY0=1-247
rg: GYRATION TYPE=RADIUS ATOMS=1-247 MASS
k2: GYRATION TYPE=KAPPA2 ATOMS=1-247 MASS
metad: METAD ARG=rg,k2 PACE=500 HEIGHT=1.0 SIGMA=0.05,0.025 BIASFACTOR=10 TEMP=313.15 FILE=HILLS GRID_MIN=0.30,0.0 GRID_MAX=4.50,1.0 GRID_BIN=168,100 CALC_RCT
PRINT STRIDE=500 ARG=rg,k2,metad.bias FILE=COLVAR FMT=%16.8f
FLUSH STRIDE=5000
"""


def mdp() -> str:
    return """integrator               = md
dt                       = 0.002
nsteps                   = 10000000
continuation             = no
constraint-algorithm     = lincs
constraints              = h-bonds
lincs-iter               = 1
lincs-order              = 4
cutoff-scheme            = Verlet
nstlist                  = 20
rlist                    = 1.2
coulombtype              = PME
rcoulomb                 = 1.2
pme-order                = 4
fourierspacing           = 0.12
vdwtype                  = Cut-off
rvdw                     = 1.2
DispCorr                 = EnerPres
tcoupl                   = V-rescale
tc-grps                  = System
tau-t                    = 1.0
ref-t                    = 313.15
pcoupl                   = C-rescale
pcoupltype               = isotropic
tau-p                    = 5.0
ref-p                    = 1.0
compressibility          = 1.0e-4
gen-vel                  = no
nstxout                  = 0
nstvout                  = 0
nstfout                  = 0
nstxout-compressed       = 2500
compressed-x-grps        = Solute
compressed-x-precision   = 1000
nstenergy                = 5000
nstlog                   = 5000
nstcalcenergy            = 100
pbc                      = xyz
comm-mode                = Linear
nstcomm                  = 100
"""


def run_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
export GMX_MAXBACKUP=-1

gmx_bin="${GMX:-gmx}"
if ! "$gmx_bin" mdrun -h 2>&1 | grep -q -- '-plumed'; then
  echo 'ERROR: this GROMACS binary has no -plumed support' >&2
  exit 42
fi

if [[ -z "${PLUMED_KERNEL:-}" ]]; then
  for candidate in \
      /usr/local/lib/libplumedKernel.so \
      /usr/lib/libplumedKernel.so \
      /opt/conda/lib/libplumedKernel.so \
      /opt/conda/lib/libplumedKernel.dylib; do
    if [[ -f "$candidate" ]]; then
      export PLUMED_KERNEL="$candidate"
      break
    fi
  done
fi
if [[ -z "${PLUMED_KERNEL:-}" ]]; then
  echo 'ERROR: PLUMED kernel was not found; refuse to run an unverified MetaD job' >&2
  exit 43
fi

"$gmx_bin" grompp -f metad.mdp -c start.gro -p system.top -n index.ndx -o metad.tpr -maxwarn 0
"$gmx_bin" mdrun -deffnm metad -plumed plumed.dat \
  -ntmpi 1 -ntomp "${OMP_NUM_THREADS:-8}" -pin on \
  -nb gpu -pme gpu -bonded gpu -update gpu \
  -v
'''


def job_json(name: str, directory: Path) -> dict[str, object]:
    return {
        "job_name": f"A4-m C8 q1 full-reuse targeted WT-MetaD {name}",
        "command": "bash run_wtmetad.sh",
        "log_file": "metad.log",
        "backward_files": [
            "metad.tpr", "metad.xtc", "metad.gro", "metad.edr", "metad.log",
            "COLVAR", "HILLS", "plumed.out", "run_manifest.json",
        ],
        "project_id": 4377225,
        "machine_type": "c4_m15_1 * NVIDIA T4",
        "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
        "input_directory": str(directory),
        "max_reschedule_times": 2,
        "max_run_time": 4320,
        "nnode": 1,
    }


def prepare(output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, object]] = []
    for replica, source in SOURCE_DIRS.items():
        if not source.is_dir():
            raise FileNotFoundError(source)
        target = output / replica
        target.mkdir(parents=True, exist_ok=True)
        for filename in ("start.gro", "system.top", "index.ndx"):
            shutil.copy2(source / filename, target / filename)
        (target / "plumed.dat").write_text(metad_dat(), encoding="utf-8")
        (target / "metad.mdp").write_text(mdp(), encoding="utf-8")
        run_path = target / "run_wtmetad.sh"
        run_path.write_text(run_script(), encoding="utf-8")
        run_path.chmod(0o755)
        payload = job_json(replica, target)
        (target / "job.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        manifest = {
            "system": "A4_m_C8",
            "state": "A4_m_C8_1_cis",
            "stage": "q1",
            "route": "full_reuse",
            "active_pair": ["V5", "V7"],
            "replica": replica,
            "method": "targeted well-tempered metadynamics supplement",
            "primary_method_unchanged": "distance umbrella sampling",
            "biased_cvs": {
                "rg": "solute mass-weighted radius of gyration, atoms 1-247, nm",
                "k2": "solute mass-weighted relative shape anisotropy, atoms 1-247, dimensionless",
            },
            "reaction_coordinate_biased": False,
            "temperature_K": 313.15,
            "duration_ns": 20.0,
            "pace_steps": 500,
            "height_kJ_mol": 1.0,
            "bias_factor": 10.0,
            "sigma": {"rg_nm": 0.05, "k2": 0.025},
            "source_pathway": str(source),
            "purpose": "discover persistent global-shape basins absent from ordinary MD/umbrella trajectories",
            "not_for": "replacing WHAM/MBAR or estimating the primary reaction-path free energy",
        }
        (target / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        jobs.append({"replica": replica, "directory": str(target), "job": payload})
    top_manifest = {
        "name": "c8_q1_full_reuse_targeted_wtmetad",
        "system": "A4-m-C8",
        "stage": "q1",
        "route": "full-reuse",
        "replicas": 3,
        "duration_ns_per_replica": 20.0,
        "primary_method": "existing distance umbrella",
        "supplement": "global-CV WT-MetaD",
        "biased_cvs": ["mass-weighted solute Rg", "mass-weighted solute KAPPA2"],
        "reaction_distance_bias": False,
        "jobs": jobs,
    }
    (output / "submission_manifest.json").write_text(json.dumps(top_manifest, indent=2) + "\n", encoding="utf-8")
    return top_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "bohrium_submit/c8_q1_full_reuse_wtmetad")
    args = parser.parse_args()
    payload = prepare(args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "replicas": payload["replicas"], "duration_ns_per_replica": payload["duration_ns_per_replica"]}, indent=2))


if __name__ == "__main__":
    main()
