#!/usr/bin/env python3
"""Package reciprocal-start WT-MetaD tests biasing a distance-only contact PC1.

This is a basin-discovery supplement. Two reactive distances remain harmonically
restrained at 1.75 nm; only the other-vinyl distance fingerprint PC1 is biased.
"""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

from analyze_b6_local_q1_joint_mbar import SOURCE_METADATA
from diagnose_b6_local_q1_hidden_conformations import FEATURE_NAMES


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
MATCHED = PROJECT / (
    "results/b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001/"
    "joint_mbar/hidden_conformation_diagnostic/matched_frames_and_contacts.csv"
)
DOWNLOAD = WORKSPACE / "bohrium_download/b6_local_q1_cross_seed_2d_extension_20261001"
SOURCE_TASKS = WORKSPACE / "bohrium_submit/b6_local_q1_cross_seed_2d_extension_20261001"
OUT = WORKSPACE / "b6_q1_contact_pc1_wtmetad_server_20261002"
SOURCES = {"A6_origin": 20807597, "A2_origin": 20807598}
DT_PS = 0.002
BIASFACTOR = 10
HEIGHT_KJ = 0.5
SIGMA_PC = 0.5
PACE_STEPS = 500
TEMP_K = 313.15


def distance_only_pc1() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    features = [name + "_nm" for name in FEATURE_NAMES if name not in ("V3-V12", "V3-V10")]
    rows = list(csv.DictReader(MATCHED.open(encoding="utf-8")))
    rows = [row for row in rows if int(row["pair"]) == 2]
    values = np.asarray([[float(row[name]) for name in features] for row in rows])
    family = np.asarray([row["family"] for row in rows])
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (values - mean) / scale
    _, singular, axes = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    axis = axes[0]
    score = z @ axis
    if score[family == "A6"].mean() < score[family == "A2"].mean():
        axis = -axis
        score = -score
    centers = {group: float(score[family == group].mean()) for group in ("A6", "A2")}
    explained = float(singular[0] ** 2 / np.sum(singular ** 2))
    return features, mean, scale, axis, {
        "training_data": str(MATCHED), "training_pair": 2,
        "features": features, "excluded_reaction_distances": ["V3-V12", "V3-V10"],
        "feature_count": len(features), "pca1_explained_variance_fraction": explained,
        "training_mean_pc1": centers, "bias_cv": "sum_i (loading_i / training_sd_i) * distance_i",
        "center_offset_omitted": "A constant PC offset does not alter the metadynamics forces.",
        "extension_projection_mean_pc1_first_vs_last_1ns": {
            "A6_origin": [1.82, 2.23], "A2_origin": [1.11, -1.00]
        },
    }


def write_plumed(features: list[str], scale: np.ndarray, axis: np.ndarray,
                 mapping: dict[str, list[int]]) -> str:
    lines = [
        "UNITS LENGTH=nm TIME=ps ENERGY=kj/mol",
        "WHOLEMOLECULES ENTITY0=1-438",
    ]
    for index in range(3, 13):
        atoms = mapping[f"V{index}"]
        lines.append(f"v{index}: COM ATOMS={atoms[0]},{atoms[1]}")
    args, coeffs = [], []
    for i, name in enumerate(features):
        left, right = name.removesuffix("_nm").split("-")
        label = f"d_{left}_{right}"
        lines.append(f"{label}: DISTANCE ATOMS=v{left[1:]},v{right[1:]}")
        args.append(label)
        coeffs.append(float(axis[i] / scale[i]))
    lines.append(f"pc1: COMBINE ARG={','.join(args)} COEFFICIENTS={','.join(f'{x:.12g}' for x in coeffs)} PERIODIC=NO")
    lines.append(
        f"meta: METAD ARG=pc1 SIGMA={SIGMA_PC} HEIGHT={HEIGHT_KJ} PACE={PACE_STEPS} "
        f"BIASFACTOR={BIASFACTOR} TEMP={TEMP_K} FILE=HILLS"
    )
    lines.append("PRINT ARG=pc1 FILE=COLVAR STRIDE=250")
    return "\n".join(lines) + "\n"


def make_run(task: Path) -> None:
    text = """#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
GMX=${GMX:-/root/autodl-tmp/apps/gromacs-2026.3/bin/gmx}
NTOMP=${NTOMP:-16}
export GMX_MAXBACKUP=-1
if [[ -z "${PLUMED_KERNEL:-}" ]]; then
  for candidate in "${CONDA_PREFIX:-}/lib/libplumedKernel.so" /root/miniconda3/envs/plumed_runtime/lib/libplumedKernel.so /root/miniconda3/envs/gmx_gpu/lib/libplumedKernel.so /root/autodl-tmp/apps/gromacs-2026.3/lib/libplumedKernel.so; do
    if [[ -s "$candidate" ]]; then PLUMED_KERNEL="$candidate"; break; fi
  done
fi
if [[ -z "${PLUMED_KERNEL:-}" || ! -s "$PLUMED_KERNEL" ]]; then
  echo "PLUMED_KERNEL missing; set it to a readable libplumedKernel.so" >&2; exit 2
fi
export PLUMED_KERNEL
mkdir -p results
exec > >(tee results/wtmetad.log) 2>&1
archive() {
  status=$?
  tar -czf results/wtmetad_results.tar.gz equil.mdp production.mdp plumed.dat start.gro system.top index.ndx \\
    equil.gro equil.cpt wtmeta.tpr wtmeta.gro wtmeta.cpt wtmeta.xtc wtmeta_pullx.xvg \\
    HILLS COLVAR wtmeta.log results/wtmetad.log 2>/dev/null || true
  exit "$status"
}
trap archive EXIT
if [[ ! -s equil.gro || ! -s equil.cpt ]]; then
  "$GMX" grompp -f equil.mdp -c start.gro -p system.top -n index.ndx -o equil.tpr -maxwarn 0
  "$GMX" mdrun -deffnm equil -ntmpi 1 -ntomp "$NTOMP" -pin off -nb gpu -pme gpu -bonded cpu -update cpu
fi
if [[ ! -s wtmeta.tpr ]]; then
  "$GMX" grompp -f production.mdp -c equil.gro -t equil.cpt -p system.top -n index.ndx -o wtmeta.tpr -maxwarn 0
fi
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then echo "Prepared WT-MetaD TPR and PC1 input; PREPARE_ONLY=1"; exit 0; fi
if [[ -s wtmeta.cpt ]]; then
  "$GMX" mdrun -s wtmeta.tpr -deffnm wtmeta -plumed plumed.dat -ntmpi 1 -ntomp "$NTOMP" -pin off -nb gpu -pme gpu -bonded cpu -update cpu -v -cpi wtmeta.cpt -append
else
  "$GMX" mdrun -s wtmeta.tpr -deffnm wtmeta -plumed plumed.dat -ntmpi 1 -ntomp "$NTOMP" -pin off -nb gpu -pme gpu -bonded cpu -update cpu -v
fi
"""
    (task / "run_wtmetad.sh").write_text(text, encoding="utf-8")
    (task / "run_wtmetad.sh").chmod(0o755)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    features, mean, scale, axis, cv_meta = distance_only_pc1()
    mapping = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))["vinyl_mapping_gromacs_1based"]
    OUT.mkdir(parents=True)
    tasks = []
    for origin, job in SOURCES.items():
        task = OUT / origin
        task.mkdir()
        work = DOWNLOAD / str(job) / "extracted/work"
        shutil.copy2(work / "extension.gro", task / "start.gro")
        shutil.copy2(SOURCE_TASKS / origin / "system.top", task / "system.top")
        shutil.copy2(SOURCE_TASKS / origin / "index.ndx", task / "index.ndx")
        eq = (SOURCE_TASKS / origin / "equil.mdp").read_text(encoding="utf-8")
        prod = (SOURCE_TASKS / origin / "production.mdp").read_text(encoding="utf-8")
        for key, value in (("nsteps", "250000"), ("gen-seed", "20261012" if origin == "A6_origin" else "20261013")):
            eq = "\n".join(f"{key:24s} = {value}" if "=" in row and row.split("=",1)[0].strip()==key else row for row in eq.splitlines()) + "\n"
        prod = "\n".join(f"{key:24s} = {value}" if "=" in row and row.split("=",1)[0].strip()==key else row
                          for row in prod.splitlines()
                          for key, value in [("nsteps", "5000000")]) + "\n"
        (task / "equil.mdp").write_text(eq, encoding="utf-8")
        (task / "production.mdp").write_text(prod, encoding="utf-8")
        (task / "plumed.dat").write_text(write_plumed(features, scale, axis, mapping), encoding="utf-8")
        make_run(task)
        metadata = {
            "origin": origin, "source_bohrium_job": job,
            "method": "WT-MetaD on distance-only PC1 of 43 non-reactive vinyl-center pair distances",
            "restraint_distances": {"V3-V12": 1.75, "V3-V10": 1.75},
            "equilibration_ps": 500, "production_ps": 10000,
            "metad": {"sigma_pc1": SIGMA_PC, "height_kj_mol": HEIGHT_KJ,
                      "pace_steps": PACE_STEPS, "biasfactor": BIASFACTOR, "temperature_K": TEMP_K},
            "interpretation": "basin discovery only; not a PMF or unbiased selectivity estimate",
        }
        (task / "cv_metadata.json").write_text(json.dumps({**metadata, **cv_meta}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tasks.append(metadata)
    readme = f"""# B6 q1 contact-PC1 WT-MetaD server package

Two reciprocal starts probe the slow contact rearrangement seen under identical d(V3,V12)=d(V3,V10)=1.75 nm restraints. The biased CV is the first principal component of the 43 other vinyl-center pair distances. It was fit on the distance-matched pair-2 umbrella frames; the first component explains {cv_meta['pca1_explained_variance_fraction']:.3f} of their standardized distance variance. The PC direction separates the A6 and A2 family contact fingerprints in the training data and follows their distinct 5 ns continuation behavior.

This is basin discovery only. It cannot replace the two-dimensional umbrella PMF or supply a free-energy ratio. The GROMACS executable on the target server must report PLUMED support; the Bohrium 2024.2 binary used for recent runs does not expose a PLUMED interface, so this package is intended for the user's GROMACS 2026.3 PLUMED-enabled server.

## Upload and run

From the project directory on the Mac:

```bash
tar -czf b6_q1_contact_pc1_wtmetad_server_20261002.tar.gz b6_q1_contact_pc1_wtmetad_server_20261002
scp -P 43212 b6_q1_contact_pc1_wtmetad_server_20261002.tar.gz root@connect.westb.seetacloud.com:/root/
```

On the server:

```bash
tar -xzf /root/b6_q1_contact_pc1_wtmetad_server_20261002.tar.gz -C /root
cd /root/b6_q1_contact_pc1_wtmetad_server_20261002
source /root/miniconda3/etc/profile.d/conda.sh
conda activate gmx_gpu
export GMX=/root/autodl-tmp/apps/gromacs-2026.3/bin/gmx
./check_plumed.sh
```

First run a short PLUMED smoke test in only one directory:

```bash
cd A6_origin
PREPARE_ONLY=1 NTOMP=16 bash ./run_wtmetad.sh
```

Then run a 10-step PLUMED smoke check in a separate output folder (it does not alter the full-run HILLS/COLVAR):

```bash
cd A6_origin
mkdir -p smoke
cd smoke
"$GMX" mdrun -s ../wtmeta.tpr -deffnm smoke -plumed ../plumed.dat -nsteps 10 -ntmpi 1 -ntomp 4 -pin off -nb gpu -pme gpu -bonded cpu -update cpu
cd ../..
```

Then run both full trajectories sequentially on the single GPU:

```bash
NTOMP=16 bash ./run_all_sequential.sh
```

Each trajectory has 0.5 ns un-biased (apart from the two distance restraints) equilibration followed by 10 ns WT-MetaD on PC1. Compare `COLVAR`, `HILLS`, and the other-distance fingerprints from the trajectory; any newly visited basin must be checked with an identical umbrella calculation before combining free energies.
"""
    (OUT / "README_SERVER.md").write_text(readme, encoding="utf-8")
    shutil.copy2(WORKSPACE / "results/b6_q1_wtmetad_results_20260919/raw/b6_q1_targeted_wtmetad_server_20260918/check_plumed.sh", OUT / "check_plumed.sh")
    (OUT / "check_plumed.sh").chmod(0o755)
    seq = """#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
for dir in A6_origin A2_origin; do
  (cd "$ROOT/$dir" && NTOMP=${NTOMP:-16} bash ./run_wtmetad.sh)
done
"""
    (OUT / "run_all_sequential.sh").write_text(seq, encoding="utf-8")
    (OUT / "run_all_sequential.sh").chmod(0o755)
    (OUT / "cv_definition.json").write_text(json.dumps(cv_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
