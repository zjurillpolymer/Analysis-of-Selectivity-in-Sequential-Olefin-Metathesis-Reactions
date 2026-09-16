#!/usr/bin/env python3
"""Prepare a small, paired C8 q1 full-reuse exchange test.

Each of three replicas gets two otherwise matched starts at the same V5--V7
distance window: an old ordinary-umbrella start and a compact MetaD-derived
start.  This package is intentionally only prepared locally; submission is a
separate, explicitly authorized action because it incurs Bohrium cost.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from scipy.spatial import cKDTree

PROJECT = Path(__file__).resolve().parents[1]
N_SOLUTE = 247
TARGETS = {
    "rep1": {"target_nm": 0.75, "normal_window": "window_004_0.750nm"},
    "rep2": {"target_nm": 0.45, "normal_window": "window_001_0.450nm"},
    "rep3": {"target_nm": 0.75, "normal_window": "window_004_0.750nm"},
}
BASE_PATHS = {
    "rep1": PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_download/20730624/extracted/pathway",
    "rep2": PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_download/20730625/extracted/pathway",
    "rep3": PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_rep3_download/20730989/extracted/pathway",
}
CANDIDATES = {
    rep: PROJECT / f"results/c8_q1_colvars_wtmetad_analysis_20260914/candidate_1_{rep}_best_soft_nac_solute_only.gro"
    for rep in TARGETS
}


def replace_dcm_count(topology: str, count: int) -> str:
    lines = topology.splitlines()
    in_molecules = False
    changed = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            in_molecules = stripped.strip("[] ").strip() == "molecules"
            continue
        if in_molecules and re.match(r"^DCM\s+\d+\s*$", stripped):
            prefix = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{prefix}DCM               {count}"
            changed += 1
    if changed != 1:
        raise ValueError(f"expected one DCM molecule count in [ molecules ], changed {changed}")
    return "\n".join(lines) + ("\n" if topology.endswith("\n") else "")


def index_text(n_atoms: int) -> str:
    def lines(values: list[int]) -> str:
        return "\n".join(" ".join(map(str, values[i:i + 15])) for i in range(0, len(values), 15))
    return (
        "[ System ]\n" + lines(list(range(1, n_atoms + 1))) + "\n\n"
        "[ Solute ]\n" + lines(list(range(1, N_SOLUTE + 1))) + "\n\n"
        "[ V5_midpoint ]\n84 110\n\n"
        "[ V7_midpoint ]\n105 111\n"
    )


def compact_full_system(template: Path, candidate: Path, output: Path, cutoff_a: float = 2.0) -> int:
    """Insert a solute-only candidate into old solvent and remove overlaps."""
    base = mda.Universe(str(template))
    cand = mda.Universe(str(candidate))
    if len(cand.atoms) != N_SOLUTE or len(base.atoms) < N_SOLUTE:
        raise ValueError(f"unexpected atom count for {template} / {candidate}")
    if cand.atoms.names.tolist() != base.atoms[:N_SOLUTE].names.tolist():
        raise ValueError(f"solute atom order mismatch for {candidate}")
    base.dimensions = np.array(base.dimensions, dtype=float)
    base.atoms[:N_SOLUTE].positions = cand.atoms.positions
    box_a = base.dimensions[:3].astype(float)
    solute_xyz = base.atoms[:N_SOLUTE].positions.astype(float)
    solvent = base.atoms[N_SOLUTE:]
    tree = cKDTree(solute_xyz, boxsize=box_a)
    bad_resids: set[int] = set()
    for residue in solvent.residues:
        distance, _ = tree.query(residue.atoms.positions.astype(float), k=1)
        if float(np.min(distance)) < cutoff_a:
            bad_resids.add(int(residue.ix))
    keep_solvent = solvent[np.array([int(atom.residue.ix) not in bad_resids for atom in solvent], dtype=bool)]
    kept = base.atoms[:N_SOLUTE] + keep_solvent
    output.parent.mkdir(parents=True, exist_ok=True)
    with mda.Writer(str(output), n_atoms=len(kept), reindex=True) as writer:
        writer.write(kept)
    return len(keep_solvent) // 3


def mdp(target_nm: float, production: bool) -> str:
    nsteps = 500_000 if production else 100_000
    return f"""integrator               = md
dt                       = 0.002
nsteps                   = {nsteps}
continuation             = {'yes' if production else 'no'}
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
gen-vel                  = {'no' if production else 'yes'}
gen-temp                 = 313.15
gen-seed                 = -1
nstxout                  = 0
nstvout                  = 0
nstfout                  = 0
nstxout-compressed       = {2500 if production else 0}
compressed-x-grps        = Solute
compressed-x-precision   = 1000
nstenergy                = 500
nstlog                   = 5000
nstcalcenergy            = 100
pbc                      = xyz
comm-mode                = Linear
nstcomm                  = 100
pull                     = yes
pull-ncoords             = 1
pull-ngroups             = 2
pull-group1-name         = V5_midpoint
pull-group2-name         = V7_midpoint
pull-group1-pbcatom      = 84
pull-group2-pbcatom      = 105
pull-pbc-ref-prev-step-com = yes
pull-coord1-type         = umbrella
pull-coord1-geometry     = distance
pull-coord1-groups       = 1 2
pull-coord1-dim          = Y Y Y
pull-coord1-start        = no
pull-coord1-init         = {target_nm:.3f}
pull-coord1-rate         = 0
pull-coord1-k            = 500
pull-nstxout             = 250
pull-nstfout             = 250
"""


def run_script() -> str:
    return r'''#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
gmx_bin="${GMX:-gmx}"
"$gmx_bin" grompp -f equil.mdp -c start.gro -p system.top -n index.ndx -o equil.tpr -maxwarn 0
"$gmx_bin" mdrun -deffnm equil -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -pin off -nb gpu -pme gpu -bonded cpu -update cpu -v
"$gmx_bin" grompp -f production.mdp -c equil.gro -t equil.cpt -p system.top -n index.ndx -o production.tpr -maxwarn 0
"$gmx_bin" mdrun -deffnm production -ntmpi 1 -ntomp "$OMP_NUM_THREADS" -pin off -nb gpu -pme gpu -bonded cpu -update cpu -v
'''


def job_json(directory: Path, name: str) -> dict[str, object]:
    return {
        "job_name": f"A4-m-C8 q1 full-reuse exchange test {name}",
        "command": "bash run_exchange.sh",
        "log_file": "production.log",
        "backward_files": ["equil.gro", "equil.cpt", "equil.tpr", "equil.log",
                           "production.gro", "production.cpt", "production.tpr",
                           "production.xtc", "production.edr", "production.log"],
        "project_id": 4377225,
        "machine_type": "c4_m15_1 * NVIDIA T4",
        "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
        "input_directory": str(directory),
        "max_reschedule_times": 2,
        "max_run_time": 720,
        "nnode": 1,
    }


def prepare(output: Path) -> dict[str, object]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty {output}")
    output.mkdir(parents=True, exist_ok=True)
    jobs = []
    for rep, base_path in BASE_PATHS.items():
        target_nm = float(TARGETS[rep]["target_nm"])
        normal_window = str(TARGETS[rep]["normal_window"])
        template = base_path / "windows" / normal_window / "production.gro"
        top_text = (base_path / "system.top").read_text(encoding="utf-8")
        for kind in ("normal", "compact"):
            name = f"{rep}_{kind}"
            target = output / name
            target.mkdir(parents=True, exist_ok=True)
            if kind == "normal":
                shutil.copy2(template, target / "start.gro")
                dcm_count = 2985
            else:
                dcm_count = compact_full_system(
                    base_path / "start.gro", CANDIDATES[rep], target / "start.gro"
                )
            (target / "system.top").write_text(replace_dcm_count(top_text, dcm_count), encoding="utf-8")
            (target / "index.ndx").write_text(index_text(N_SOLUTE + 3 * dcm_count), encoding="utf-8")
            (target / "equil.mdp").write_text(mdp(target_nm, False), encoding="utf-8")
            (target / "production.mdp").write_text(mdp(target_nm, True), encoding="utf-8")
            run = target / "run_exchange.sh"
            run.write_text(run_script(), encoding="utf-8")
            run.chmod(0o755)
            manifest = {
                "system": "A4_m_C8",
                "state": "A4_m_C8_1_cis",
                "stage": "q1",
                "route": "full_reuse",
                "active_pair": ["V5", "V7"],
                "replica": rep,
                "seed_kind": kind,
                "target_distance_nm": target_nm,
                "production_ns": 1.0,
                "equilibration_ps": 200.0,
                "purpose": "test whether old distance umbrella can exchange normal and MetaD-discovered compact global conformations",
                "primary_method_unchanged": "distance umbrella with orientation postprocessing",
                "submit_status": "prepared_only",
                "dcm_molecules": dcm_count,
                "normal_seed_source": str(template),
                "compact_seed_source": str(CANDIDATES[rep]),
            }
            (target / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            payload = job_json(target, name)
            (target / "job.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            jobs.append({"name": name, "directory": str(target), "seed_kind": kind, "replica": rep, "dcm_molecules": dcm_count})
    manifest = {
        "name": "c8_q1_full_reuse_exchange_test",
        "system": "A4-m-C8",
        "jobs": jobs,
        "job_count": len(jobs),
        "duration_ns_per_job": 1.0,
        "submission_status": "prepared_only",
        "decision_rule": "compare Rg trajectories and compact/normal occupancy within each matched distance target; persistent disagreement supports joint distance×Rg umbrella",
    }
    (output / "submission_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# C8 q1 full-reuse exchange test\n\n"
        "Six paired 1-ns GPU jobs are prepared but not submitted. Each replica has an old normal seed and a MetaD-derived compact seed at the same V5–V7 distance target. Compact starts remove overlapping DCM molecules and update the topology count; they must be re-equilibrated before production.\n\n"
        "This is an exchange diagnostic, not a new yield estimate. Submit only after explicit approval.\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "bohrium_submit/c8_q1_full_reuse_exchange_test")
    args = parser.parse_args()
    manifest = prepare(args.output.resolve())
    print(json.dumps({"output": str(args.output.resolve()), "job_count": manifest["job_count"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
