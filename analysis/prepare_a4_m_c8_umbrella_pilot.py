#!/usr/bin/env python3
"""Prepare a portable GROMACS umbrella pilot for A4-m-C8 q0.

Four pair-specific pathways are prepared as two anchor-matched comparisons:
V1-V2 vs V1-V3 and V5-V6 vs V5-V8.  Each pathway starts from a complete
explicit-DCM production endpoint and uses inward/outward steering to seed a
common grid of umbrella windows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


PAIR_CONFIGS = (
    {"pathway": "anchor_v1_same_v1_v2", "topology": "same_arm", "left": 1, "right": 2,
     "replica": 4, "maximum_nm": 2.225},
    {"pathway": "anchor_v1_cross_v1_v3", "topology": "cross_arm", "left": 1, "right": 3,
     "replica": 3, "maximum_nm": 3.050},
    {"pathway": "anchor_v5_same_v5_v6", "topology": "same_arm", "left": 5, "right": 6,
     "replica": 3, "maximum_nm": 2.225},
    {"pathway": "anchor_v5_cross_v5_v8", "topology": "cross_arm", "left": 5, "right": 8,
     "replica": 1, "maximum_nm": 3.050},
)

MINIMUM_NM = 0.350
WINDOW_SPACING_NM = 0.075
FORCE_CONSTANT = 500.0
PULL_RATE_NM_PS = 0.0005
DT_PS = 0.002
STEERING_MARGIN_PS = 300.0


def last_distance_nm(csv_path: Path, column: str) -> float:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty distance CSV: {csv_path}")
    return float(rows[-1][column]) / 10.0


def window_centers(maximum_nm: float) -> list[float]:
    count = int(round((maximum_nm - MINIMUM_NM) / WINDOW_SPACING_NM))
    centers = [MINIMUM_NM + index * WINDOW_SPACING_NM for index in range(count + 1)]
    if not math.isclose(centers[-1], maximum_nm, abs_tol=1e-9):
        raise ValueError(f"maximum {maximum_nm} is not on the window grid")
    return centers


def common_mdp(*, nsteps: int, continuation: bool, gen_vel: bool, xtc_group: str, xtc_stride: int) -> str:
    return f"""integrator               = md
dt                       = {DT_PS:g}
nsteps                   = {nsteps}
continuation             = {'yes' if continuation else 'no'}
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
gen-vel                  = {'yes' if gen_vel else 'no'}
gen-temp                 = 313.15
gen-seed                 = -1
nstxout                  = 0
nstvout                  = 0
nstfout                  = 0
nstxout-compressed       = {xtc_stride}
compressed-x-grps        = {xtc_group}
compressed-x-precision   = 1000
nstenergy                = 500
nstlog                   = 5000
nstcalcenergy            = 100
pbc                      = xyz
comm-mode                = Linear
nstcomm                  = 100
"""


def pull_block(left: int, right: int, mapping: dict[str, list[int]], rate: float, center: float | None) -> str:
    left_atoms = mapping[f"V{left}"]
    right_atoms = mapping[f"V{right}"]
    start = center is None
    initial = 0.0 if center is None else center
    return f"""pull                     = yes
pull-ncoords             = 1
pull-ngroups             = 2
pull-group1-name         = V{left}_midpoint
pull-group2-name         = V{right}_midpoint
pull-group1-pbcatom      = {left_atoms[0]}
pull-group2-pbcatom      = {right_atoms[0]}
pull-pbc-ref-prev-step-com = yes
pull-coord1-type         = umbrella
pull-coord1-geometry     = distance
pull-coord1-groups       = 1 2
pull-coord1-dim          = Y Y Y
pull-coord1-start        = {'yes' if start else 'no'}
pull-coord1-init         = {initial:.6f}
pull-coord1-rate         = {rate:.7f}
pull-coord1-k            = {FORCE_CONSTANT:.1f}
pull-nstxout             = 250
pull-nstfout             = 250
"""


def steering_mdp(
    left: int,
    right: int,
    mapping: dict[str, list[int]],
    initial_nm: float,
    target_nm: float,
) -> str:
    rate = math.copysign(PULL_RATE_NM_PS, target_nm - initial_nm)
    duration_ps = abs(target_nm - initial_nm) / PULL_RATE_NM_PS + STEERING_MARGIN_PS
    nsteps = int(math.ceil(duration_ps / DT_PS))
    return common_mdp(
        nsteps=nsteps, continuation=True, gen_vel=False, xtc_group="System", xtc_stride=2500
    ) + pull_block(left, right, mapping, rate, None)


def umbrella_mdp(
    left: int,
    right: int,
    mapping: dict[str, list[int]],
    center_nm: float,
    *,
    production: bool,
) -> str:
    if production:
        base = common_mdp(
            nsteps=1_000_000, continuation=True, gen_vel=False,
            xtc_group="Solute", xtc_stride=250,
        )
    else:
        base = common_mdp(
            nsteps=100_000, continuation=False, gen_vel=True,
            xtc_group="Solute", xtc_stride=0,
        )
    return base + pull_block(left, right, mapping, 0.0, center_nm)


def append_pair_groups(index_text: str, left: int, right: int, mapping: dict[str, list[int]]) -> str:
    blocks = [index_text.rstrip(), ""]
    for vinyl in (left, right):
        blocks.extend(
            [f"[ V{vinyl}_midpoint ]", " ".join(str(value) for value in mapping[f"V{vinyl}"]), ""]
        )
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--run-root", type=Path,
        default=root / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs/A4_m_C8_q0",
    )
    parser.add_argument("--output", type=Path, default=root / "umbrella/a4_m_c8_q0_pilot")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((run_root / "setup/metadata.json").read_text(encoding="utf-8"))
    mapping = {
        str(name): [int(value) for value in values]
        for name, values in metadata["vinyl_mapping_gromacs_1based"].items()
    }
    original_index = (run_root / "setup/index.ndx").read_text(encoding="utf-8")
    manifest = []
    for config in PAIR_CONFIGS:
        pathway = output / str(config["pathway"])
        pathway.mkdir(parents=True, exist_ok=True)
        replica = int(config["replica"])
        replica_root = run_root / f"replica_{replica:02d}"
        left, right = int(config["left"]), int(config["right"])
        column = f"V{left}_V{right}_{'同臂' if config['topology'] == 'same_arm' else '跨臂'}"
        initial_nm = last_distance_nm(replica_root / "vinyl_distances.csv", column)
        centers = window_centers(float(config["maximum_nm"]))
        shutil.copy2(run_root / "setup/system.top", pathway / "system.top")
        shutil.copy2(replica_root / "production_2ns.gro", pathway / "start.gro")
        shutil.copy2(replica_root / "production_2ns.cpt", pathway / "start.cpt")
        (pathway / "index.ndx").write_text(
            append_pair_groups(original_index, left, right, mapping), encoding="utf-8"
        )
        inward_target = MINIMUM_NM - 0.10
        outward_target = float(config["maximum_nm"]) + 0.10
        (pathway / "steer_in.mdp").write_text(
            steering_mdp(left, right, mapping, initial_nm, inward_target), encoding="utf-8"
        )
        (pathway / "steer_out.mdp").write_text(
            steering_mdp(left, right, mapping, initial_nm, outward_target), encoding="utf-8"
        )
        windows = pathway / "window_mdps"
        windows.mkdir(exist_ok=True)
        window_rows = []
        for index, center in enumerate(centers):
            name = f"window_{index:03d}_{center:.3f}nm"
            (windows / f"{name}_equil.mdp").write_text(
                umbrella_mdp(left, right, mapping, center, production=False), encoding="utf-8"
            )
            (windows / f"{name}_production.mdp").write_text(
                umbrella_mdp(left, right, mapping, center, production=True), encoding="utf-8"
            )
            window_rows.append({"window_index": index, "window_name": name, "center_nm": center})
        with (pathway / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("window_index", "window_name", "center_nm"))
            writer.writeheader()
            writer.writerows(window_rows)
        pathway_metadata = {
            **config,
            "pair_column": column,
            "initial_distance_nm": initial_nm,
            "minimum_nm": MINIMUM_NM,
            "maximum_nm": config["maximum_nm"],
            "window_spacing_nm": WINDOW_SPACING_NM,
            "window_count": len(centers),
            "force_constant_kJ_mol_nm2": FORCE_CONSTANT,
            "steering_rate_nm_ps": PULL_RATE_NM_PS,
            "equilibration_ps_per_window": 200.0,
            "production_ps_per_window": 2000.0,
            "source_replica": f"replica_{replica:02d}",
        }
        (pathway / "pathway.json").write_text(
            json.dumps(pathway_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        manifest.append(pathway_metadata)
    submission = {
        "analysis": "A4-m-C8 q0 pair-specific umbrella pilot",
        "gromacs_image": "registry.dp.tech/dptech/gromacs:2024.2",
        "temperature_K": 313.15,
        "pathways": manifest,
        "recommended_dispatch": "one independent GPU job per pathway",
        "command": (
            "python3 analysis/run_a4_m_c8_umbrella_path.py "
            "--pathway PATHWAY_DIR --ntomp 4 --gpu --stage all"
        ),
        "interpretation": (
            "Pair-specific 1D midpoint-distance PMFs quantify contact preorganization only. "
            "They do not include Ru chemistry or cross-arm pair multiplicity."
        ),
    }
    (output / "submission_manifest.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(submission, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
