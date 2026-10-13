#!/usr/bin/env python3
"""Build matched B6 q1 distance-umbrella inputs from physical REST2 frames.

The REST2 XTC contains only the 438-atom solute. This script extracts three
physical-state frames, re-solvates each with the same DCM model and molecule
count using local GROMACS insert-molecules, then builds paired-path umbrella
inputs. It never submits or runs MD production.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


TIMES_PS = (5000, 12500, 19500)
LBOX_NM = 6.84432
N_SOLUTE = 438
N_DCM = 2927
WINDOWS = [round(0.35 + i * 0.10, 2) for i in range(25)]
KB_KJ_MOL_K = 0.008314462618
TEMP_K = 313.15


def run(command: list[str], *, cwd: Path, stdin: str | None = None) -> str:
    result = subprocess.run(
        command, cwd=cwd, input=stdin, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout[-5000:]}")
    return result.stdout


def gro_read(path: Path) -> tuple[list[str], int, np.ndarray, np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    count = int(lines[1].strip())
    coords = np.asarray([
        [float(line[20:28]), float(line[28:36]), float(line[36:44])]
        for line in lines[2:2 + count]
    ], dtype=float)
    box = np.asarray([float(x) for x in lines[2 + count].split()[:3]], dtype=float)
    if len(coords) != count:
        raise ValueError(f"bad GRO atom count in {path}")
    return lines, count, coords, box


def write_gro(path: Path, title: str, atom_lines: list[str], box: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{title}\n{len(atom_lines):5d}\n" + "\n".join(atom_lines)
        + "\n" + " ".join(f"{x:10.5f}" for x in box) + "\n",
        encoding="utf-8",
    )


def centered_solute(source: Path, destination: Path) -> None:
    lines, count, coords, _ = gro_read(source)
    if count != N_SOLUTE:
        raise ValueError(f"expected {N_SOLUTE} solute atoms in {source}, found {count}")
    box = np.full(3, LBOX_NM)
    geometric_center = (coords.min(axis=0) + coords.max(axis=0)) / 2
    coords = coords + (box / 2 - geometric_center)
    if np.any(coords.min(axis=0) < 0.05) or np.any(coords.max(axis=0) > box - 0.05):
        raise ValueError(f"solute does not fit in {LBOX_NM} nm box: {source}")
    atom_lines = []
    for line, xyz in zip(lines[2:2 + count], coords):
        atom_lines.append(line[:20] + "".join(f"{value:8.3f}" for value in xyz) + line[44:])
    write_gro(destination, f"B6 q0 REST2 physical frame; {source.name}", atom_lines, box)


def make_ndx(path: Path) -> None:
    groups = {
        "System": list(range(1, 9219 + 1)),
        "Solute": list(range(1, N_SOLUTE + 1)),
        "V2_midpoint": [207, 208],
        "V11_midpoint": [205, 206],
        "V3_midpoint": [189, 190],
    }
    chunks = []
    for name, atoms in groups.items():
        chunks.append(f"[ {name} ]\n")
        chunks.extend(" ".join(str(x) for x in atoms[i:i + 15]) + "\n" for i in range(0, len(atoms), 15))
        chunks.append("\n")
    path.write_text("".join(chunks), encoding="utf-8")


def make_posres_top(source: Path, destination: Path) -> None:
    topology = source.read_text(encoding="utf-8")
    marker = "[ moleculetype ]"
    first = topology.find(marker)
    second = topology.find(marker, first + len(marker))
    if first < 0 or second < 0:
        raise ValueError("could not identify the solute/solvent moleculetype boundary")
    posres = "\n[ position_restraints ]\n; ai funct fcx fcy fcz (kJ mol-1 nm-2)\n"
    posres += "".join(f"{i:5d} 1 1000 1000 1000\n" for i in range(1, N_SOLUTE + 1))
    destination.write_text(topology[:second] + posres + topology[second:], encoding="utf-8")


def mdp_text(*, nsteps: int, pressure: bool, gen_vel: bool,
             out_stride: int, pull_groups: tuple[str, str] | None = None,
             pull_centers: list[float] | None = None, pull_init: bool = False,
             pull_rate: float = 0.0, pull_k: float = 500.0,
             nstenergy: int = 500) -> str:
    lines = [
        "integrator = md", "dt = 0.002", f"nsteps = {nsteps}",
        "continuation = no" if gen_vel else "continuation = yes",
        "constraint-algorithm = lincs", "constraints = h-bonds", "lincs-iter = 1", "lincs-order = 4",
        "cutoff-scheme = Verlet", "nstlist = 20", "rlist = 1.2", "coulombtype = PME",
        "rcoulomb = 1.2", "pme-order = 4", "fourierspacing = 0.12", "vdwtype = Cut-off",
        "rvdw = 1.2", "DispCorr = EnerPres", "tcoupl = V-rescale", "tc-grps = System",
        "tau-t = 1.0", f"ref-t = {TEMP_K}",
        "pcoupl = C-rescale" if pressure else "pcoupl = no",
    ]
    if pressure:
        lines += ["pcoupltype = isotropic", "tau-p = 5.0", "ref-p = 1.0", "compressibility = 1.0e-4"]
    lines += [
        f"gen-vel = {'yes' if gen_vel else 'no'}",
        f"gen-temp = {TEMP_K}", "gen-seed = -1", "nstxout = 0", "nstvout = 0", "nstfout = 0",
        f"nstxout-compressed = {out_stride}", "compressed-x-grps = System" if pull_groups and pull_init else "compressed-x-grps = Solute",
        "compressed-x-precision = 1000", f"nstenergy = {nstenergy}", "nstlog = 5000", "nstcalcenergy = 100",
        "pbc = xyz", "comm-mode = Linear", "nstcomm = 100",
    ]
    if pull_groups:
        g1, g2 = pull_groups
        lines += [
            "pull = yes", "pull-ncoords = 1", "pull-ngroups = 2",
            f"pull-group1-name = {g1}", f"pull-group2-name = {g2}",
            "pull-group1-pbcatom = 207",
            "pull-group2-pbcatom = 205" if g2 == "V11_midpoint" else "pull-group2-pbcatom = 189",
            "pull-pbc-ref-prev-step-com = yes", "pull-coord1-type = umbrella",
            "pull-coord1-geometry = distance", "pull-coord1-groups = 1 2",
            "pull-coord1-dim = Y Y Y", f"pull-coord1-start = {'yes' if pull_init else 'no'}",
        ]
        if pull_init:
            lines += ["pull-coord1-init = 0.0", f"pull-coord1-rate = {pull_rate:.7f}", f"pull-coord1-k = {pull_k:.1f}"]
        else:
            if not pull_centers or len(pull_centers) != 1:
                raise ValueError("window MDP requires one pull center")
            lines += [f"pull-coord1-init = {pull_centers[0]:.3f}", "pull-coord1-rate = 0.0000000", f"pull-coord1-k = {pull_k:.1f}"]
        lines += ["pull-nstxout = 250", "pull-nstfout = 250"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gmx", default=os.environ.get("GMX", "gmx"))
    args = parser.parse_args()
    project = args.project.resolve()
    output = (args.output or project / "b6_q1_rest2_umbrella_inputs_20261006").resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    results = project / "b6_rest2_hrex_20ns_results"
    analysis = project / "results/b6_rest2_hrex_20ns_analysis"
    tpr = results / "replicas/r1000/topol.tpr"
    xtc = analysis / "r1000_whole_5ps.xtc"
    top = results / "replicas/r1000/topol.top"
    setup_gro = project / "b6_v1_v12_rest2_prep_20261006/workspace/gromacs_dcm/runs/B6_p_C6_q1_V1_V12_cis/setup/system.gro"
    for required in (tpr, xtc, top, setup_gro):
        if not required.is_file():
            raise FileNotFoundError(required)

    shutil.copy2(top, output / "system.top")
    make_posres_top(output / "system.top", output / "system_posres.top")
    shutil.copy2(project / "b6_q1_novel_seed_umbrella_inputs_20261003/run_umbrella_path.py", output / "run_umbrella_path.py")
    make_ndx(output / "index.ndx")
    setup_lines = setup_gro.read_text(encoding="utf-8").splitlines()
    # The setup GRO atom order is SLU then DCM; use its first DCM molecule as the rigid insert template.
    dcm_lines = [setup_lines[0], "3", *setup_lines[2 + N_SOLUTE:2 + N_SOLUTE + 3], setup_lines[-1]]
    (output / "dcm_template.gro").write_text("\n".join(dcm_lines) + "\n", encoding="utf-8")

    equil = output / "equilibration"
    equil.mkdir()
    (equil / "em.mdp").write_text(
        "integrator = steep\nnsteps = 50000\nemtol = 100\nemstep = 0.01\n"
        "cutoff-scheme = Verlet\nnstlist = 20\nrlist = 1.2\ncoulombtype = PME\nrcoulomb = 1.2\n"
        "pme-order = 4\nfourierspacing = 0.12\nvdwtype = Cut-off\nrvdw = 1.2\n"
        "DispCorr = EnerPres\npbc = xyz\nconstraints = h-bonds\nconstraint-algorithm = lincs\n",
        encoding="utf-8",
    )
    (equil / "nvt_rest.mdp").write_text(mdp_text(nsteps=50000, pressure=False, gen_vel=True, out_stride=0), encoding="utf-8")
    (equil / "npt_rest.mdp").write_text(mdp_text(nsteps=50000, pressure=True, gen_vel=False, out_stride=0), encoding="utf-8")
    (equil / "npt_free.mdp").write_text(mdp_text(nsteps=500000, pressure=True, gen_vel=False, out_stride=2500), encoding="utf-8")

    seed_names = {5000: "rest2_frame_05000ps", 12500: "rest2_frame_12500ps", 19500: "rest2_frame_19500ps"}
    route_defs = {
        "to_arm6_V11": {"right": "V11_midpoint", "atoms": [205, 206], "multiplicity": 1, "label": "arm1_to_arm6"},
        "to_arm2_V3": {"right": "V3_midpoint", "atoms": [189, 190], "multiplicity": 2, "label": "arm1_to_arm2_two_equivalent_sites"},
    }
    manifest = {
        "system": "B6 q0 product V1-V12, physical REST2 frames",
        "method": "matched 1D distance umbrellas; orientation is unrestrained/marginalized and should be diagnosed per window",
        "seed_times_ps": list(TIMES_PS), "seed_count": 3, "path_count": 6,
        "channels": route_defs, "windows_per_path": len(WINDOWS), "window_centers_nm": WINDOWS,
        "equilibration_ps_per_window": 100, "production_ps_per_window": 500,
        "total_umbrella_sampling_ns": 90.0,
        "multiplicity_note": "arm-2 V3/V4 treated as symmetry-equivalent; one representative V3 path is simulated and its statistical weight is multiplied by 2 in later route weighting.",
        "population_note": "seed-specific conditional PMFs must remain separate; three snapshots are sensitivity starts, not basin populations or independent equilibrium weights.",
    }
    for time_ps in TIMES_PS:
        seed = seed_names[time_ps]
        seed_root = output / "seed_systems" / seed
        seed_root.mkdir(parents=True)
        candidate = seed_root / "solute_frame.gro"
        run([
            args.gmx, "trjconv", "-f", str(xtc), "-s", str(tpr), "-n", str(output / "index.ndx"),
            "-dump", str(time_ps), "-o", str(candidate),
        ], cwd=project, stdin="1\n")
        centered = seed_root / "solute_centered.gro"
        centered_solute(candidate, centered)
        packed = seed_root / "packed.gro"
        log = run([
            args.gmx, "insert-molecules", "-f", str(centered), "-ci", str(output / "dcm_template.gro"),
            "-nmol", str(N_DCM), "-box", *(f"{LBOX_NM:.5f}" for _ in range(3)),
            "-try", "200", "-seed", str(10000 + time_ps), "-o", str(packed),
        ], cwd=project)
        (seed_root / "packing.log").write_text(log, encoding="utf-8")
        _, atom_count, _, _ = gro_read(packed)
        if atom_count != 9219:
            raise RuntimeError(f"{seed}: expected 9219 atoms after packing; got {atom_count}")
        (seed_root / "seed.json").write_text(json.dumps({
            "seed": seed, "source_physical_state": "r1000", "source_time_ps": time_ps,
            "source_trajectory": str(xtc.relative_to(project)), "full_system_atoms": atom_count,
            "DCM_molecules": N_DCM, "box_nm": LBOX_NM,
            "note": "solute conformation from whole-solute REST2 trajectory; solvent newly packed independently and must be equilibrated before umbrellas",
        }, indent=2) + "\n", encoding="utf-8")

        for route, definition in route_defs.items():
            path_name = f"{seed}__{route}"
            path = output / "pathways" / path_name
            (path / "window_mdps").mkdir(parents=True)
            shutil.copy2(packed, path / "packed_un_equilibrated.gro")
            shutil.copy2(output / "system.top", path / "system.top")
            shutil.copy2(output / "index.ndx", path / "index.ndx")
            pair_label = "V2-V11" if route == "to_arm6_V11" else "V2-V3"
            start_distance = float(np.linalg.norm(
                gro_read(centered)[2][206:208].mean(axis=0)
                - gro_read(centered)[2][definition["atoms"][0] - 1:definition["atoms"][1]].mean(axis=0)
            ))
            maximum_leg_ps = max(abs(start_distance - min(WINDOWS)), abs(max(WINDOWS) - start_distance)) / 0.0005
            nsteps_steer = math.ceil(maximum_leg_ps / 0.002)
            common = {
                "pathway": path_name, "stage": "q1 conditional seed-specific paired umbrella",
                "seed": seed, "source_state": "REST2 physical state r1000",
                "source_time_ps": time_ps, "route_family": definition["label"],
                "left": "V2", "right": definition["right"], "atom_numbers_1based": [207, 208, *definition["atoms"]],
                "site_multiplicity_for_later_route_weighting_only": definition["multiplicity"],
                "initial_distance_nm": start_distance, "minimum_nm": min(WINDOWS), "maximum_nm": max(WINDOWS),
                "window_spacing_nm": 0.1, "window_count": len(WINDOWS),
                "force_constant_kJ_mol_nm2": 500.0, "steering_rate_nm_ps": 0.0005,
                "steering_nsteps": nsteps_steer, "equilibration_ps_per_window": 100.0,
                "production_ps_per_window": 500.0, "wham_begin_ps": 100.0,
                "interpretation": "conditional PMF within one REST2 snapshot seed; not a basin population",
            }
            (path / "pathway.json").write_text(json.dumps(common, indent=2) + "\n", encoding="utf-8")
            with (path / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["window_name", "center_nm"])
                writer.writeheader()
                for i, center in enumerate(WINDOWS):
                    writer.writerow({"window_name": f"window_{i:03d}_{center:.3f}nm", "center_nm": f"{center:.3f}"})
            left, right = "V2_midpoint", definition["right"]
            for direction, rate in (("in", -0.0005), ("out", 0.0005)):
                (path / f"steer_{direction}.mdp").write_text(mdp_text(
                    nsteps=nsteps_steer, pressure=True, gen_vel=True, out_stride=2500,
                    pull_groups=(left, right), pull_init=True, pull_rate=rate, pull_k=1000,
                    nstenergy=500,
                ), encoding="utf-8")
            for i, center in enumerate(WINDOWS):
                name = f"window_{i:03d}_{center:.3f}nm"
                for stage, nsteps in (("equil", 50000), ("production", 250000)):
                    (path / "window_mdps" / f"{name}_{stage}.mdp").write_text(mdp_text(
                        nsteps=nsteps, pressure=True, gen_vel=(stage == "equil"), out_stride=250 if stage == "production" else 0,
                        pull_groups=(left, right), pull_centers=[center], pull_init=False, pull_k=500,
                    ), encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {output}: 3 packed q0 seeds, 6 paired q1 pathways, 150 windows.")


if __name__ == "__main__":
    main()
