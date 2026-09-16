#!/usr/bin/env python3
"""Prepare five q2 path-memory umbrella pilots from downloaded full-system endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path


PATHWAYS = (
    {"pathway": "c5_reuse_v1_v6", "system": "A4_m_C5", "state": "A4_m_C5_2_opposite_cis",
     "state_index": 2, "replica": 2, "route_family": "reuse", "left": 1, "right": 6,
     "maximum_nm": 2.75},
    {"pathway": "c5_reuse_v2_v4", "system": "A4_m_C5", "state": "A4_m_C5_2_opposite_cis",
     "state_index": 2, "replica": 3, "route_family": "reuse", "left": 2, "right": 4,
     "maximum_nm": 2.75},
    {"pathway": "c5_switch_v4_v6", "system": "A4_m_C5", "state": "A4_m_C5_2_opposite_cis",
     "state_index": 2, "replica": 2, "route_family": "switch", "left": 4, "right": 6,
     "maximum_nm": 2.75},
    {"pathway": "c8_reuse_v5_v7", "system": "A4_m_C8", "state": "A4_m_C8_2_opposite_cis",
     "state_index": 4, "replica": 3, "route_family": "reuse", "left": 5, "right": 7,
     "maximum_nm": 2.65},
    {"pathway": "c8_switch_v4_v5", "system": "A4_m_C8", "state": "A4_m_C8_2_opposite_cis",
     "state_index": 4, "replica": 3, "route_family": "switch", "left": 4, "right": 5,
     "maximum_nm": 2.65},
)

MINIMUM_NM = 0.35
WINDOW_SPACING_NM = 0.10
FORCE_CONSTANT = 500.0
STEERING_FORCE_CONSTANT = 1000.0
PULL_RATE_NM_PS = 0.0005
DT_PS = 0.002
EQUILIBRATION_PS = 100.0
PRODUCTION_PS = 500.0
WHAM_BEGIN_PS = 100.0


def centers(maximum_nm: float) -> list[float]:
    count = int(round((maximum_nm - MINIMUM_NM) / WINDOW_SPACING_NM))
    result = [MINIMUM_NM + index * WINDOW_SPACING_NM for index in range(count + 1)]
    if not math.isclose(result[-1], maximum_nm, abs_tol=1e-9):
        raise ValueError("maximum is not on the requested window grid")
    return result


def gro_geometry(path: Path) -> tuple[dict[int, tuple[float, float, float]], tuple[float, float, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[1].strip())
    atoms: dict[int, tuple[float, float, float]] = {}
    for line in lines[2:2 + atom_count]:
        atom = int(line[15:20])
        atoms[atom] = (float(line[20:28]), float(line[28:36]), float(line[36:44]))
    box_fields = [float(value) for value in lines[2 + atom_count].split()]
    if len(box_fields) != 3:
        raise ValueError(f"only orthorhombic GRO boxes are supported: {path}")
    return atoms, (box_fields[0], box_fields[1], box_fields[2])


def minimum_image(delta: float, box: float) -> float:
    return delta - box * round(delta / box)


def midpoint(atoms: dict[int, tuple[float, float, float]], pair: list[int], box: tuple[float, float, float]) -> tuple[float, float, float]:
    first, second = atoms[pair[0]], atoms[pair[1]]
    return tuple(
        (first[axis] + 0.5 * minimum_image(second[axis] - first[axis], box[axis])) % box[axis]
        for axis in range(3)
    )


def pair_distance_nm(
    gro: Path, mapping: dict[str, list[int]], left: int, right: int,
) -> tuple[float, float]:
    atoms, box = gro_geometry(gro)
    left_mid = midpoint(atoms, mapping[f"V{left}"], box)
    right_mid = midpoint(atoms, mapping[f"V{right}"], box)
    distance = math.sqrt(sum(minimum_image(right_mid[i] - left_mid[i], box[i]) ** 2 for i in range(3)))
    return distance, min(box) / 2.0


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


def pull_block(
    left: int,
    right: int,
    mapping: dict[str, list[int]],
    rate: float,
    center: float | None,
    force_constant: float = FORCE_CONSTANT,
) -> str:
    return f"""pull                     = yes
pull-ncoords             = 1
pull-ngroups             = 2
pull-group1-name         = V{left}_midpoint
pull-group2-name         = V{right}_midpoint
pull-group1-pbcatom      = {mapping[f'V{left}'][0]}
pull-group2-pbcatom      = {mapping[f'V{right}'][0]}
pull-pbc-ref-prev-step-com = yes
pull-coord1-type         = umbrella
pull-coord1-geometry     = distance
pull-coord1-groups       = 1 2
pull-coord1-dim          = Y Y Y
pull-coord1-start        = {'yes' if center is None else 'no'}
pull-coord1-init         = {0.0 if center is None else center:.6f}
pull-coord1-rate         = {rate:.7f}
pull-coord1-k            = {force_constant:.1f}
pull-nstxout             = 250
pull-nstfout             = 250
"""


def steering_mdp(left: int, right: int, mapping: dict[str, list[int]], initial: float, target: float) -> str:
    rate = math.copysign(PULL_RATE_NM_PS, target - initial)
    # The moving reference must stop at ``target``.  Extending nsteps while
    # keeping a non-zero rate would overshoot it and can cross the half-box
    # distance for the compact C8 system.
    duration_ps = abs(target - initial) / PULL_RATE_NM_PS
    nsteps = int(math.ceil(duration_ps / DT_PS))
    return common_mdp(
        nsteps=nsteps, continuation=False, gen_vel=True, xtc_group="System", xtc_stride=2500,
    ) + pull_block(left, right, mapping, rate, None, STEERING_FORCE_CONSTANT)


def umbrella_mdp(left: int, right: int, mapping: dict[str, list[int]], center: float, production: bool) -> str:
    duration = PRODUCTION_PS if production else EQUILIBRATION_PS
    base = common_mdp(
        nsteps=int(round(duration / DT_PS)),
        continuation=production,
        gen_vel=not production,
        xtc_group="Solute",
        xtc_stride=250 if production else 0,
    )
    return base + pull_block(left, right, mapping, 0.0, center)


def append_pair_groups(index_text: str, left: int, right: int, mapping: dict[str, list[int]]) -> str:
    blocks = [index_text.rstrip(), ""]
    for vinyl in (left, right):
        blocks.extend([f"[ V{vinyl}_midpoint ]", " ".join(map(str, mapping[f"V{vinyl}"])), ""])
    return "\n".join(blocks)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-root", type=Path,
        default=root / "results/a4_postreaction_nac_exports/b6-gromacs-parallel-kcl7v",
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=root / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs",
    )
    parser.add_argument("--output", type=Path, default=root / "umbrella/a4_q2_path_memory_pilot")
    args = parser.parse_args()
    export_root, run_root, output = args.export_root.resolve(), args.run_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for config in PATHWAYS:
        state = str(config["state"])
        replica = int(config["replica"])
        setup = run_root / state / "setup"
        endpoint = (
            export_root / f"gromacs-run-{int(config['state_index']):02d}-{replica:02d}"
            / state / f"replica_{replica:02d}"
        )
        metadata = json.loads((setup / "metadata.json").read_text(encoding="utf-8"))
        mapping = {str(key): [int(value) for value in values] for key, values in metadata["vinyl_mapping_gromacs_1based"].items()}
        left, right = int(config["left"]), int(config["right"])
        maximum_nm = float(config["maximum_nm"])
        pathway = output / str(config["pathway"])
        pathway.mkdir(parents=True, exist_ok=True)
        start = endpoint / "production_2ns.gro"
        initial_nm, half_box_nm = pair_distance_nm(start, mapping, left, right)
        outward_target_nm = maximum_nm
        if outward_target_nm >= 0.49 * (2.0 * half_box_nm):
            raise ValueError(f"{pathway.name}: outward target violates half-box limit {half_box_nm:.3f} nm")
        shutil.copy2(setup / "system.top", pathway / "system.top")
        shutil.copy2(start, pathway / "start.gro")
        (pathway / "index.ndx").write_text(
            append_pair_groups((setup / "index.ndx").read_text(encoding="utf-8"), left, right, mapping),
            encoding="utf-8",
        )
        (pathway / "steer_in.mdp").write_text(
            steering_mdp(left, right, mapping, initial_nm, MINIMUM_NM - 0.10), encoding="utf-8",
        )
        (pathway / "steer_out.mdp").write_text(
            steering_mdp(left, right, mapping, initial_nm, outward_target_nm), encoding="utf-8",
        )
        mdp_root = pathway / "window_mdps"
        mdp_root.mkdir(exist_ok=True)
        rows = []
        for index, center in enumerate(centers(maximum_nm)):
            name = f"window_{index:03d}_{center:.3f}nm"
            (mdp_root / f"{name}_equil.mdp").write_text(
                umbrella_mdp(left, right, mapping, center, False), encoding="utf-8",
            )
            (mdp_root / f"{name}_production.mdp").write_text(
                umbrella_mdp(left, right, mapping, center, True), encoding="utf-8",
            )
            rows.append({"window_index": index, "window_name": name, "center_nm": f"{center:.3f}"})
        with (pathway / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("window_index", "window_name", "center_nm"))
            writer.writeheader()
            writer.writerows(rows)
        pathway_metadata = {
            **config,
            "initial_distance_nm": initial_nm,
            "half_box_limit_nm": half_box_nm,
            "minimum_nm": MINIMUM_NM,
            "maximum_nm": maximum_nm,
            "window_spacing_nm": WINDOW_SPACING_NM,
            "window_count": len(rows),
            "force_constant_kJ_mol_nm2": FORCE_CONSTANT,
            "steering_rate_nm_ps": PULL_RATE_NM_PS,
            "equilibration_ps_per_window": EQUILIBRATION_PS,
            "production_ps_per_window": PRODUCTION_PS,
            "wham_begin_ps": WHAM_BEGIN_PS,
            "wham_end_ps": PRODUCTION_PS,
            "start_mode": "full-system endpoint GRO with newly generated velocities",
            "source_endpoint": str(start.relative_to(root)),
        }
        (pathway / "pathway.json").write_text(
            json.dumps(pathway_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        manifest.append(pathway_metadata)
    submission = {
        "analysis": "A4 q2 path-memory umbrella pilot",
        "gromacs_image": "registry.dp.tech/dptech/gromacs:2024.2",
        "temperature_K": 313.15,
        "pathways": manifest,
        "total_window_sampling_ns": sum(
            int(row["window_count"]) * (EQUILIBRATION_PS + PRODUCTION_PS) / 1000.0
            for row in manifest
        ),
        "interpretation": (
            "These midpoint-distance PMFs quantify catalyst-free conformational preorganization for "
            "reuse/switch path-memory pairs; they are not Ru-catalyzed chemical activation barriers."
        ),
    }
    (output / "submission_manifest.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(submission, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
