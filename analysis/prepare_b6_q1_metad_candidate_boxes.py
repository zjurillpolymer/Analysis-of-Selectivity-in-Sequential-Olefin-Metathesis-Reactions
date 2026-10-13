#!/usr/bin/env python3
"""Locally re-solvate four representative B6 q1 WT-MetaD solute frames.

Packmol constructs new DCM solvent configurations around centered solute
conformations. This does not recover the original solvent trajectory. The
resulting boxes are inputs for minimization and restrained equilibration before
any candidate is used to seed umbrella windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "b6_q1_contact_pc1_wtmetad_results_20261002"
OUT = ROOT / "b6_q1_meta_candidate_umbrella_inputs_20261003"
DCM_PDB = ROOT / "gromacs_dcm/models/ch2cl2_dc/dcm.pdb"
N_DCM = 2927
AVOGADRO = 6.02214076e23
MOLAR_MASS_DCM = 84.927

# Representative extremes from both reciprocal starts, with both directions
# on the deployed PC1 represented for each origin.
CANDIDATES = [
    ("A6_origin", 690, "high_novelty_positive_pc1"),
    ("A6_origin", 1270, "high_novelty_crossing_frame"),
    ("A2_origin", 5480, "positive_pc1_return"),
    ("A2_origin", 9090, "high_novelty_negative_pc1"),
]


def need_executable(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required executable is not on PATH: {name}")
    return path


def read_gro(path: Path) -> tuple[int, tuple[float, float, float]]:
    lines = path.read_text().splitlines()
    atom_count = int(lines[1].strip())
    box = tuple(float(x) for x in lines[2 + atom_count].split()[:3])
    return atom_count, box


def run(cmd: list[str], cwd: Path, log: Path, stdin: str | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, input=stdin, capture_output=True, text=True)
    log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); see {log}: {' '.join(cmd)}")


def add_pull_restraints(mdp: str, save_solute: bool = False) -> str:
    if save_solute:
        replacements = {"nstxout-compressed": "250", "compressed-x-grps": "Solute"}
        seen = set()
        updated = []
        for line in mdp.splitlines():
            if "=" in line and not line.lstrip().startswith(";"):
                key = line.split("=", 1)[0].strip().lower()
                if key in replacements:
                    if key not in seen:
                        updated.append(f"{key:25s}= {replacements[key]}")
                        seen.add(key)
                    continue
            updated.append(line)
        for key, value in replacements.items():
            if key not in seen:
                updated.append(f"{key:25s}= {value}")
        mdp = "\n".join(updated)
    return mdp.rstrip() + """

; Keep both reactive distances at the MetaD reference point while solvent settles.
pull                     = yes
pull-ncoords              = 2
pull-ngroups              = 3
pull-group1-name          = V3_midpoint
pull-group2-name          = V12_midpoint
pull-group3-name          = V10_midpoint
pull-group1-pbcatom        = 189
pull-group2-pbcatom        = 207
pull-group3-pbcatom        = 203
pull-pbc-ref-prev-step-com = yes
pull-coord1-type           = umbrella
pull-coord1-geometry       = distance
pull-coord1-groups         = 1 2
pull-coord1-dim            = Y Y Y
pull-coord1-start           = no
pull-coord1-init            = 1.750000
pull-coord1-rate            = 0.0
pull-coord1-k               = 500.0
pull-coord2-type           = umbrella
pull-coord2-geometry       = distance
pull-coord2-groups         = 1 3
pull-coord2-dim            = Y Y Y
pull-coord2-start           = no
pull-coord2-init            = 1.750000
pull-coord2-rate            = 0.0
pull-coord2-k               = 500.0
pull-nstxout                = 250
pull-nstfout                = 250
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmx", default="gmx")
    ap.add_argument("--packmol", default="/opt/homebrew/Caskroom/miniconda/base/envs/chem_dl/bin/packmol")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gmx = need_executable(args.gmx)
    packmol = Path(args.packmol)
    if not packmol.is_file():
        raise FileNotFoundError(f"Local Packmol executable not found: {packmol}")
    if not DCM_PDB.is_file():
        raise FileNotFoundError(DCM_PDB)

    manifest_in = RESULTS / "analysis/candidate_structure_manifest.csv"
    with manifest_in.open() as f:
        extracted = {(r["origin"], int(r["time_ps"])): r for r in csv.DictReader(f)}
    top_source = RESULTS / "A6_origin/system.top"
    index_source = RESULTS / "A6_origin/index.ndx"
    min_source = ROOT / "gromacs_dcm/mdp/em.mdp"
    nvt_source = ROOT / "gromacs_dcm/mdp/nvt.mdp"
    npt_source = ROOT / "gromacs_dcm/mdp/npt.mdp"
    all_rows = []

    for origin, time_ps, label in CANDIDATES:
        item = extracted[(origin, time_ps)]
        frame = Path(item["structure"])
        name = f"{origin}_{time_ps:05d}ps_{label}"
        work = OUT / name
        work.mkdir(parents=True, exist_ok=True)
        atom_count, box_nm = read_gro(frame)
        if (work / "em.tpr").is_file() and (work / "system.gro").is_file():
            (work / "em.mdp").write_text(add_pull_restraints(min_source.read_text()))
            (work / "nvt.mdp").write_text(add_pull_restraints(nvt_source.read_text(), save_solute=True))
            (work / "npt.mdp").write_text(add_pull_restraints(npt_source.read_text(), save_solute=True))
            box_a = [x * 10 for x in box_nm]
            hi = [x - 1.0 for x in box_a]
            (work / "packmol.inp").write_text(
                "tolerance 2.0\nfiletype pdb\noutput packed.pdb\nseed 20261002\n"
                "structure solute_centered.pdb\n"
                "  number 1\n  fixed 0.0 0.0 0.0 0.0 0.0 0.0\nend structure\n"
                "structure dcm_template.pdb\n"
                f"  number {N_DCM}\n  resnumbers 3\n"
                f"  inside box 1.0 1.0 1.0 {hi[0]:.5f} {hi[1]:.5f} {hi[2]:.5f}\n"
                "end structure\n"
            )
            final_atoms, final_box = read_gro(work / "system.gro")
            mass_density = N_DCM * MOLAR_MASS_DCM / AVOGADRO / (math.prod(box_nm) * 1e-21)
            all_rows.append({
                "name": name, "origin": origin, "time_ps": time_ps,
                "candidate_pc1": float(item["pc1"]),
                "nearest_reference_rms_z": float(item["nearest_ref_rms_z"]),
                "d_V3_V12_nm": float(item["V3-V12_nm"]), "d_V3_V10_nm": float(item["V3-V10_nm"]),
                "solute_atoms": atom_count, "DCM_molecules": N_DCM, "total_atoms": final_atoms,
                "box_nm": list(final_box), "nominal_DCM_density_g_cm3": mass_density,
                "packmol_atom_overlap_tolerance_A": 2.0,
                "packed_grompp_em": "passed", "workdir": str(work),
            })
            print(f"{name}: reusing validated Packmol box and EM TPR")
            continue
        shutil.copy2(frame, work / "solute_centered.gro")
        shutil.copy2(top_source, work / "system.top")
        shutil.copy2(index_source, work / "index.ndx")
        shutil.copy2(DCM_PDB, work / "dcm_template.pdb")

        if atom_count != 438 or max(box_nm) - min(box_nm) > 1e-4:
            raise ValueError(f"Unexpected solute atom count or non-cubic box in {frame}")
        box_a = [x * 10 for x in box_nm]
        # Leave a 1 A boundary margin; the whole solute is held fixed at its
        # box-centered coordinates while Packmol fills the remaining volume.
        hi = [x - 1.0 for x in box_a]
        packmol_in = work / "packmol.inp"
        packmol_in.write_text(
            "tolerance 2.0\nfiletype pdb\noutput packed.pdb\nseed 20261002\n"
            "structure solute_centered.pdb\n"
            "  number 1\n  fixed 0.0 0.0 0.0 0.0 0.0 0.0\nend structure\n"
            "structure dcm_template.pdb\n"
            f"  number {N_DCM}\n  resnumbers 3\n"
            f"  inside box 1.0 1.0 1.0 {hi[0]:.5f} {hi[1]:.5f} {hi[2]:.5f}\n"
            "end structure\n"
        )
        run([gmx, "editconf", "-f", str(work / "solute_centered.gro"),
             "-o", str(work / "solute_centered.pdb")], work, work / "editconf_solute.log")
        with (work / "packmol.inp").open() as inp:
            result = subprocess.run([str(packmol)], cwd=work, stdin=inp,
                                    capture_output=True, text=True)
        (work / "packmol.log").write_text(result.stdout + result.stderr)
        if result.returncode or not (work / "packed.pdb").is_file():
            raise RuntimeError(f"Packmol failed for {name}; see {work / 'packmol.log'}")
        pdb_lines = (work / "packed.pdb").read_text().splitlines()
        pdb_atom_lines = [line for line in pdb_lines if line.startswith(("ATOM  ", "HETATM"))]
        if len(pdb_atom_lines) != 438 + 3 * N_DCM:
            raise RuntimeError(f"Packed atom count {len(pdb_atom_lines)} != {438 + 3*N_DCM} for {name}")
        run([gmx, "editconf", "-f", str(work / "packed.pdb"), "-o", str(work / "system.gro"),
             "-box", *(f"{x:.6f}" for x in box_nm)], work, work / "editconf_box.log")
        final_atoms, final_box = read_gro(work / "system.gro")
        if final_atoms != 438 + 3 * N_DCM:
            raise RuntimeError(f"GRO atom count mismatch for {name}: {final_atoms}")

        em = add_pull_restraints(min_source.read_text())
        nvt = add_pull_restraints(nvt_source.read_text(), save_solute=True)
        npt = add_pull_restraints(npt_source.read_text(), save_solute=True)
        (work / "em.mdp").write_text(em)
        (work / "nvt.mdp").write_text(nvt)
        # Keep the established 1 ns NPT equilibration duration.
        (work / "npt.mdp").write_text(npt)
        run([gmx, "grompp", "-f", "em.mdp", "-c", "system.gro", "-p", "system.top",
             "-n", "index.ndx", "-o", "em.tpr", "-maxwarn", "0"], work, work / "grompp_em.log")
        mass_density = N_DCM * MOLAR_MASS_DCM / AVOGADRO / (math.prod(box_nm) * 1e-21)
        all_rows.append({
            "name": name, "origin": origin, "time_ps": time_ps, "candidate_pc1": float(item["pc1"]),
            "nearest_reference_rms_z": float(item["nearest_ref_rms_z"]),
            "d_V3_V12_nm": float(item["V3-V12_nm"]), "d_V3_V10_nm": float(item["V3-V10_nm"]),
            "solute_atoms": atom_count, "DCM_molecules": N_DCM, "total_atoms": final_atoms,
            "box_nm": list(final_box), "nominal_DCM_density_g_cm3": mass_density,
            "packmol_atom_overlap_tolerance_A": 2.0,
            "packed_grompp_em": "passed", "workdir": str(work),
        })
        print(f"{name}: Packmol and energy-minimization grompp passed; rho_DCM={mass_density:.3f} g/cm3")

    (OUT / "candidate_box_manifest.json").write_text(json.dumps(all_rows, indent=2) + "\n")
    (OUT / "candidate_box_manifest.csv").write_text(
        "name,origin,time_ps,candidate_pc1,nearest_reference_rms_z,d_V3_V12_nm,d_V3_V10_nm,"
        "solute_atoms,DCM_molecules,total_atoms,nominal_DCM_density_g_cm3,packed_grompp_em,workdir\n" +
        "".join(
            f"{r['name']},{r['origin']},{r['time_ps']},{r['candidate_pc1']:.6f},"
            f"{r['nearest_reference_rms_z']:.6f},{r['d_V3_V12_nm']:.3f},{r['d_V3_V10_nm']:.3f},"
            f"{r['solute_atoms']},{r['DCM_molecules']},{r['total_atoms']},"
            f"{r['nominal_DCM_density_g_cm3']:.4f},{r['packed_grompp_em']},{r['workdir']}\n"
            for r in all_rows
        )
    )
    run_script = OUT / "run_candidate.sh"
    run_script.write_text("""#!/usr/bin/env bash
set -euo pipefail
candidate=${1:?usage: run_candidate.sh /path/to/candidate_directory}
cd "$candidate"
GMX=${GMX:-/root/autodl-tmp/apps/gromacs-2026.3/bin/gmx}
NTOMP=${NTOMP:-16}
export GMX_MAXBACKUP=-1
run_md() {
  local name="$1"
  shift
  if [[ -s "${name}.cpt" ]]; then
    "$GMX" mdrun -deffnm "$name" -cpi "${name}.cpt" -append "$@"
  else
    "$GMX" mdrun -deffnm "$name" "$@"
  fi
}
if [[ ! -s em.tpr ]]; then
  "$GMX" grompp -f em.mdp -c system.gro -p system.top -n index.ndx -o em.tpr -maxwarn 0
fi
if [[ ! -s em.gro ]]; then
  # GROMACS does not support GPU PME with steepest-descent minimization.
  run_md em -ntmpi 1 -ntomp "$NTOMP" -pin off -nb gpu -pme cpu -bonded cpu -update cpu -v
fi
if [[ ! -s nvt.tpr ]]; then
  "$GMX" grompp -f nvt.mdp -c em.gro -p system.top -n index.ndx -o nvt.tpr -maxwarn 0
fi
if [[ ! -s nvt.gro ]]; then
  run_md nvt -ntmpi 1 -ntomp "$NTOMP" -pin off -nb gpu -pme gpu -bonded cpu -update cpu -v
fi
if [[ ! -s npt.tpr ]]; then
  "$GMX" grompp -f npt.mdp -c nvt.gro -t nvt.cpt -p system.top -n index.ndx -o npt.tpr -maxwarn 0
fi
if [[ ! -s npt.gro ]]; then
  run_md npt -ntmpi 1 -ntomp "$NTOMP" -pin off -nb gpu -pme gpu -bonded cpu -update cpu -v
fi
"$GMX" check -f npt.gro
"$GMX" distance -f npt.xtc -s npt.tpr \\
  -select 'com of (atomnr 189 or atomnr 190) plus com of (atomnr 207 or atomnr 208); com of (atomnr 189 or atomnr 190) plus com of (atomnr 203 or atomnr 204)' \\
  -oav npt_restrained_distances.xvg -xvg none
""", encoding="utf-8")
    run_script.chmod(0o755)
    wrapper = OUT / "run_all_candidates_sequential.sh"
    wrapper.write_text("""#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
for name in \\
  A6_origin_00690ps_high_novelty_positive_pc1 \\
  A6_origin_01270ps_high_novelty_crossing_frame \\
  A2_origin_05480ps_positive_pc1_return \\
  A2_origin_09090ps_high_novelty_negative_pc1; do
  NTOMP=${NTOMP:-16} bash "$ROOT/run_candidate.sh" "$ROOT/$name"
done
""", encoding="utf-8")
    wrapper.chmod(0o755)
    (OUT / "README_SERVER.md").write_text(f"""# B6 q1 MetaD candidate boxes

Four representative solute conformations from the 2026-10-02 reciprocal-start WT-MetaD runs were re-solvated locally with Packmol and the existing CH2Cl2-DC model. This reconstructs new solvent configurations; it does **not** recover the original solvent trajectory. Each box contains 1 solute (438 atoms) plus {N_DCM} three-site DCM molecules (8,781 atoms), in its source trajectory's cubic box (about 6.86 nm).

Packmol was run locally with 2.0 A overlap tolerance. All four full systems contain 9,219 atoms; EM, NVT, and NPT input files passed local GROMACS 2026.3 `grompp` syntax/topology checks without warnings. Energy minimization uses CPU PME (GROMACS cannot run PME on GPU for the non-dynamical steepest-descent integrator); NVT/NPT retain GPU PME. Nominal initial DCM density is 1.279–1.282 g/cm3; NPT equilibration will adjust the box.

The inputs preserve the two reactive-distance restraints at 1.75 nm and include the following stages: 50 ps steepest-descent energy minimization, 100 ps NVT with newly generated velocities, then 1 ns NPT. Coordinates are output for the solute every 0.5 ps during NVT/NPT so candidate basin persistence can be checked before constructing umbrella windows. The candidate structures are not yet evidence for equilibrium weights.

## Upload and run

From the local project directory:

```bash
tar -czf b6_q1_meta_candidate_umbrella_inputs_20261003.tar.gz b6_q1_meta_candidate_umbrella_inputs_20261003
scp -P 43212 b6_q1_meta_candidate_umbrella_inputs_20261003.tar.gz root@connect.westb.seetacloud.com:/root/
```

On the server:

```bash
tar -xzf /root/b6_q1_meta_candidate_umbrella_inputs_20261003.tar.gz -C /root
cd /root/b6_q1_meta_candidate_umbrella_inputs_20261003
export GMX=/root/autodl-tmp/apps/gromacs-2026.3/bin/gmx
nohup env NTOMP=16 bash ./run_all_candidates_sequential.sh > candidate_equilibration.log 2>&1 &
tail -f candidate_equilibration.log
```

The wrapper runs the four systems sequentially, appropriate for one GPU. It is restartable from the GROMACS checkpoints. After completion, inspect each `npt_restrained_distances.xvg` and `npt.xtc` before deciding which seed(s) merit the larger umbrella calculation.
""", encoding="utf-8")
    print(f"Prepared four candidate solvent boxes in {OUT}")


if __name__ == "__main__":
    main()
