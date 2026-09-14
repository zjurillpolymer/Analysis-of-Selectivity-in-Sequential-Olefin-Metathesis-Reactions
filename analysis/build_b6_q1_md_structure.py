#!/usr/bin/env python3
"""Build an MD-ready B6 cis-q1 structure from reactive q0 frames.

The chemical edit is exact (loss of C2H4 and a new Z C=C).  Candidate
conformations are taken from well-separated q0 frames with the shortest V3--V6
distance, closed with a restrained MMFF94 minimization, then relaxed without
restraints.  The lowest-energy conformer passing graph and geometry checks is
written as PDB plus an explicit-bond MOL2 seed for AmberTools/GAFF2.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

import analyze_react1_structures as react1
import build_b6_q1_review as review
import build_q2_branch_structures as metathesis


DEFAULT_CONSUMED = (3, 6)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--left", type=int, default=DEFAULT_CONSUMED[0])
    parser.add_argument("--right", type=int, default=DEFAULT_CONSUMED[1])
    parser.add_argument("--candidate-frames", type=int, default=8)
    parser.add_argument("--minimum-separation-ps", type=float, default=25.0)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    return parser


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def select_candidate_frames(
    distances: Path, pair: tuple[int, int], count: int, minimum_separation_ps: float
) -> tuple[str, list[dict[str, float | int]]]:
    rows = read_csv(distances)
    if not rows:
        raise ValueError(f"distance table is empty: {distances}")
    pair_label = f"V{min(pair)}_V{max(pair)}_"
    columns = [name for name in rows[0] if name.startswith(pair_label)]
    if len(columns) != 1:
        raise ValueError(f"expected one {pair_label} distance column, found {columns}")
    column = columns[0]
    ranked = sorted(
        (
            {
                "frame": int(row["frame"]),
                "time_ps": float(row["absolute_time_ps"]),
                "distance_A": float(row[column]),
            }
            for row in rows
        ),
        key=lambda row: (float(row["distance_A"]), int(row["frame"])),
    )
    selected: list[dict[str, float | int]] = []
    for row in ranked:
        if all(
            abs(float(row["time_ps"]) - float(chosen["time_ps"]))
            >= minimum_separation_ps
            for chosen in selected
        ):
            selected.append(row)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} frames satisfy {minimum_separation_ps:g} ps separation"
        )
    return column, selected


def extract_solute_frame(
    gmx: str, tpr: Path, xtc: Path, index: Path, time_ps: float, output: Path
) -> None:
    command = [
        gmx,
        "trjconv",
        "-s",
        str(tpr),
        "-f",
        str(xtc),
        "-n",
        str(index),
        "-o",
        str(output),
        "-dump",
        f"{time_ps:.6f}",
        "-pbc",
        "whole",
    ]
    result = subprocess.run(
        command,
        input="Solute\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0 or not output.is_file():
        raise RuntimeError(
            f"gmx trjconv failed at {time_ps:g} ps (exit {result.returncode}):\n"
            f"{result.stdout[-4000:]}"
        )


def pdb_coordinates(path: Path, expected_atoms: int) -> np.ndarray:
    coordinates: list[tuple[float, float, float]] = []
    serials: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        serials.append(int(line[6:11]))
        coordinates.append(
            (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        )
    if serials != list(range(1, expected_atoms + 1)):
        raise ValueError(f"nonsequential or incomplete solute frame: {path}")
    return np.asarray(coordinates, dtype=float)


def molecule_at_coordinates(template: Chem.Mol, coordinates: np.ndarray) -> Chem.Mol:
    if template.GetNumAtoms() != len(coordinates):
        raise ValueError("template/frame atom-count mismatch")
    molecule = Chem.Mol(template)
    molecule.RemoveAllConformers()
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    for index, xyz in enumerate(coordinates):
        conformer.SetAtomPosition(index, tuple(float(value) for value in xyz))
    molecule.AddConformer(conformer, assignId=True)
    return molecule


def new_bond_indices(
    molecule: Chem.Mol, retained_serials: tuple[int, int]
) -> tuple[int, int, int, int]:
    left, right = (
        metathesis.atom_index_by_source_serial(molecule, serial)
        for serial in retained_serials
    )
    bond = molecule.GetBondBetweenAtoms(left, right)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError("new q1 double bond is absent")
    stereo_atoms = list(bond.GetStereoAtoms())
    if len(stereo_atoms) != 2:
        raise ValueError("new V3--V6 double bond lacks stereo atoms")
    return stereo_atoms[0], left, right, stereo_atoms[1]


def geometry(
    molecule: Chem.Mol, retained_serials: tuple[int, int]
) -> tuple[float, float, str]:
    first, left, right, fourth = new_bond_indices(molecule, retained_serials)
    conformer = molecule.GetConformer()
    points = [
        np.asarray(conformer.GetAtomPosition(index), dtype=float)
        for index in (first, left, right, fourth)
    ]
    length = float(np.linalg.norm(points[1] - points[2]))
    dihedral = react1.dihedral_degrees(*points)
    return length, dihedral, react1.geometry_label(length, dihedral)


def close_and_optimize(
    molecule: Chem.Mol, retained_serials: tuple[int, int], max_iters: int
) -> tuple[Chem.Mol, dict[str, float | int | str]]:
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise ValueError("MMFF94 parameters are incomplete")
    first, left, right, fourth = new_bond_indices(molecule, retained_serials)
    properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94")
    if properties is None:
        raise ValueError("cannot create MMFF94 properties")

    restrained = AllChem.MMFFGetMoleculeForceField(
        molecule, properties, nonBondedThresh=100.0
    )
    restrained.MMFFAddDistanceConstraint(
        left, right, False, 1.32, 1.36, 250.0
    )
    restrained.MMFFAddTorsionConstraint(
        first, left, right, fourth, False, -20.0, 20.0, 100.0
    )
    restrained.Initialize()
    restrained_status = int(restrained.Minimize(maxIts=max_iters))

    relaxed = AllChem.MMFFGetMoleculeForceField(
        molecule, properties, nonBondedThresh=100.0
    )
    relaxed.Initialize()
    relaxed_status = int(relaxed.Minimize(maxIts=max_iters))
    energy = float(relaxed.CalcEnergy())
    length, dihedral, label = geometry(molecule, retained_serials)
    if not math.isfinite(energy) or label != "cis":
        raise ValueError(
            f"invalid relaxed candidate: E={energy}, length={length}, "
            f"dihedral={dihedral}, geometry={label}"
        )
    return molecule, {
        "restrained_status": restrained_status,
        "relaxed_status": relaxed_status,
        "mmff_energy_kcal_mol": energy,
        "new_bond_length_A": length,
        "new_bond_dihedral_deg": dihedral,
        "new_bond_geometry": label,
    }


def active_mapping(
    product: Chem.Mol, original_mapping: dict[int, tuple[int, int]], consumed: tuple[int, int]
) -> dict[int, tuple[int, int]]:
    source_to_output = {
        atom.GetIntProp("source_q1_serial"): atom.GetIdx() + 1
        for atom in product.GetAtoms()
    }
    active = sorted(set(original_mapping) - set(consumed))
    return {
        vinyl: tuple(source_to_output[serial] for serial in original_mapping[vinyl])
        for vinyl in active
    }


def write_manifest_row(
    manifest_path: Path,
    source_pdb: str,
    mapping: dict[int, tuple[int, int]],
    state: str,
) -> None:
    rows = read_csv(manifest_path)
    if not rows:
        raise ValueError(f"manifest is empty: {manifest_path}")
    original = next(row for row in rows if row["state"] == "B6_p_C6_q0")
    original_schema = json.loads(original["pair_schema_json"])
    active = set(mapping)
    schema = [
        pair
        for pair in original_schema
        if int(pair["left"]) in active and int(pair["right"]) in active
    ]
    row = {
        "state": state,
        "stage": "q1",
        "system": "B6_p_C6",
        "branch": "first_reaction_cis",
        "source_pdb": source_pdb,
        "net_charge": "0",
        "active_vinyl_count": str(len(mapping)),
        "active_pair_count": str(len(schema)),
        "vinyl_mapping_json": json.dumps(
            {f"V{key}": list(value) for key, value in mapping.items()},
            separators=(",", ":"),
        ),
        "pair_schema_json": json.dumps(schema, ensure_ascii=False, separators=(",", ":")),
        "output_dir": f"gromacs_dcm/runs/{state}",
    }
    by_state = {existing["state"]: existing for existing in rows}
    by_state[state] = row
    ordered = [
        by_state[existing["state"]]
        for existing in rows
        if existing["state"] != state
    ] + [row]
    fields = list(rows[0])
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)


def main() -> None:
    args = build_parser().parse_args()
    consumed = tuple(sorted((args.left, args.right)))
    if (
        args.candidate_frames < 1
        or args.minimum_separation_ps < 0
        or args.max_iters < 1
        or len(consumed) != 2
        or consumed[0] == consumed[1]
        or not all(1 <= value <= 12 for value in consumed)
    ):
        raise SystemExit("invalid candidate count, frame separation, or max iterations")
    state = f"B6_p_C6_q1_V{consumed[0]}_V{consumed[1]}_cis"
    root = Path(args.root).resolve()
    bundle = root / "gromacs_dcm_server_bundle"
    output_dir = bundle / "structures"
    result_dir = root / f"results/explicit_dcm_b6_q1_v{consumed[0]}_v{consumed[1]}_build"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = output_dir / f"{state}.pdb"
    output_mol2 = output_dir / f"{state}_antechamber.mol2"
    audit_path = result_dir / "structure_qc.json"
    if output_pdb.exists() and output_mol2.exists() and audit_path.exists() and not args.force:
        print(audit_path.read_text(encoding="utf-8"))
        return

    gmx = shutil.which("gmx")
    if gmx is None:
        raise RuntimeError("gmx is required to extract q0 candidate frames")
    q0_dir = bundle / "gromacs_dcm/runs/B6_p_C6_q0"
    replica = q0_dir / "replica_01"
    distance_column, candidates = select_candidate_frames(
        replica / "vinyl_distances.csv",
        consumed,
        args.candidate_frames,
        args.minimum_separation_ps,
    )
    source = bundle / "structures/B6_p_C6.pdb"
    template = metathesis.load_rdkit_pdb(source)
    original_mapping = review.read_mapping(root)
    initial_formula = rdMolDescriptors.CalcMolFormula(template)

    screened: list[dict[str, float | int | str | bool]] = []
    products: dict[int, Chem.Mol] = {}
    with tempfile.TemporaryDirectory(prefix="b6_q1_frames_") as temporary:
        temporary_path = Path(temporary)
        for candidate_index, candidate in enumerate(candidates):
            frame_pdb = temporary_path / f"frame_{int(candidate['frame']):05d}.pdb"
            extract_solute_frame(
                gmx,
                replica / "production_2ns.tpr",
                replica / "production_2ns.xtc",
                q0_dir / "setup/index.ndx",
                float(candidate["time_ps"]),
                frame_pdb,
            )
            try:
                frame_molecule = molecule_at_coordinates(
                    template, pdb_coordinates(frame_pdb, template.GetNumAtoms())
                )
                terminal_retained = [
                    metathesis.terminal_and_retained_indices(
                        frame_molecule, original_mapping[vinyl]
                    )
                    for vinyl in consumed
                ]
                retained_serials = tuple(
                    frame_molecule.GetAtomWithIdx(retained).GetIntProp("source_q1_serial")
                    for _, retained in terminal_retained
                )
                product = metathesis.edit_second_metathesis(
                    frame_molecule, original_mapping, consumed
                )
                product, qc = close_and_optimize(
                    product, retained_serials, args.max_iters
                )
                products[candidate_index] = product
                screened.append({**candidate, **qc, "passed": True, "error": ""})
            except Exception as error:
                screened.append(
                    {**candidate, "passed": False, "error": str(error)}
                )

    passing_indices = [
        index for index, row in enumerate(screened) if bool(row["passed"])
    ]
    if not passing_indices:
        raise RuntimeError(f"no q0-derived q1 conformer passed: {screened}")
    selected_index = min(
        passing_indices,
        key=lambda index: float(screened[index]["mmff_energy_kcal_mol"]),
    )
    product = products[selected_index]
    selected = screened[selected_index]
    product_formula = rdMolDescriptors.CalcMolFormula(product)
    delta = review.formula_delta(template, product)
    if delta != {"C": 2, "H": 4}:
        raise ValueError(f"unexpected formula delta: {delta}")
    if product.GetNumAtoms() != 438:
        raise ValueError(f"unexpected q1 atom count: {product.GetNumAtoms()}")
    cycle_rank_initial = template.GetNumBonds() - template.GetNumAtoms() + 1
    cycle_rank_product = product.GetNumBonds() - product.GetNumAtoms() + 1
    if cycle_rank_product != cycle_rank_initial + 1:
        raise ValueError("metathesis product did not add exactly one graph cycle")

    source_to_output = metathesis.renumber_pdb_metadata(product)
    Chem.MolToPDBFile(product, str(output_pdb), flavor=0)
    metathesis.write_antechamber_mol2(product, output_mol2)
    mapping = active_mapping(product, original_mapping, consumed)
    if len(mapping) != 10:
        raise ValueError(f"expected 10 active vinyl groups, found {mapping}")
    relative_source = output_pdb.relative_to(bundle).as_posix()
    write_manifest_row(
        bundle / "gromacs_dcm/state_manifest.csv", relative_source, mapping, state
    )

    audit = {
        "status": "PASS",
        "state": state,
        "method": "q0 reactive-frame extraction + restrained/unrestrained MMFF94",
        "source_q0_state": "B6_p_C6_q0",
        "source_replica": 1,
        "distance_column": distance_column,
        "screened_candidate_count": len(candidates),
        "passing_candidate_count": len(passing_indices),
        "minimum_candidate_separation_ps": args.minimum_separation_ps,
        "selected_source_frame": selected["frame"],
        "selected_source_time_ps": selected["time_ps"],
        "selected_pre_reaction_distance_A": selected["distance_A"],
        "selected_mmff_energy_kcal_mol": selected["mmff_energy_kcal_mol"],
        "new_bond_length_A": selected["new_bond_length_A"],
        "new_bond_dihedral_deg": selected["new_bond_dihedral_deg"],
        "new_bond_geometry": selected["new_bond_geometry"],
        "consumed_vinyls": [f"V{consumed[0]}", f"V{consumed[1]}"],
        "new_double_bond_source_serials": list(retained_serials),
        "new_double_bond_output_serials": [source_to_output[serial] for serial in retained_serials],
        "initial_formula": initial_formula,
        "product_formula": product_formula,
        "formula_delta": delta,
        "initial_atom_count": template.GetNumAtoms(),
        "product_atom_count": product.GetNumAtoms(),
        "adds_one_graph_cycle": True,
        "active_vinyl_count": len(mapping),
        "active_pair_count": len(mapping) * (len(mapping) - 1) // 2,
        "active_vinyl_mapping_product_pdb_1based": {
            f"V{key}": list(value) for key, value in mapping.items()
        },
        "output_pdb": str(output_pdb.relative_to(root)),
        "output_explicit_bond_mol2": str(output_mol2.relative_to(root)),
        "screened_candidates": screened,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
