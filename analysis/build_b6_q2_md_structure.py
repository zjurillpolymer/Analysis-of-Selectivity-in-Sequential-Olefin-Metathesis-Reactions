#!/usr/bin/env python3
"""Build MD-ready B6 q2 V3-V6 + V11-V12 structures from q1 trajectories.

For each independent q1 replica, well-separated V11-V12 near-contact frames
are screened.  The exact second metathesis edit removes C2H4 and creates a Z
double bond, followed by restrained and unrestrained MMFF94 minimization.  One
passing low-energy q2 conformer is retained per q1 replica; the lowest-energy
of those three is the canonical AmberTools parameterization seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

import analyze_react1_structures as react1
import build_b6_q1_md_structure as q1_builder
import build_q2_branch_structures as metathesis


STATE = "B6_p_C6_q2_V3_V6_V11_V12_cis"
Q1_STATE = "B6_p_C6_q1_V3_V6_cis"
SECOND_PAIR = (11, 12)
EXPECTED_ACTIVE = (1, 2, 4, 5, 7, 8, 9, 10)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--q1-workspace",
        default=(
            "results/b6_q1_v3_v6_50ns_download/tmp/output/var/folders/2r/"
            "923m0n5527d9hvtrnwldp3yc0000gn/T/b6-gromacs-upload-1kxsiogy/workspace"
        ),
    )
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--candidate-frames-per-replica", type=int, default=5)
    parser.add_argument("--minimum-time-ps", type=float, default=5000.0)
    parser.add_argument("--minimum-separation-ps", type=float, default=500.0)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def q1_manifest_row(workspace: Path) -> dict[str, str]:
    rows = read_rows(workspace / "gromacs_dcm/state_manifest.csv")
    return next(row for row in rows if row["state"] == Q1_STATE)


def q1_mapping(row: dict[str, str]) -> dict[int, tuple[int, int]]:
    return {
        int(label.removeprefix("V")): tuple(int(value) for value in serials)
        for label, serials in json.loads(row["vinyl_mapping_json"]).items()
    }


def select_candidate_frames(
    distances: Path,
    count: int,
    minimum_time_ps: float,
    minimum_separation_ps: float,
) -> tuple[str, list[dict[str, float | int]]]:
    rows = read_rows(distances)
    if not rows:
        raise ValueError(f"empty distance table: {distances}")
    columns = [name for name in rows[0] if name.startswith("V11_V12_")]
    if len(columns) != 1:
        raise ValueError(f"expected one V11-V12 column, found {columns}")
    column = columns[0]
    ranked = sorted(
        (
            {
                "frame": int(row["frame"]),
                "time_ps": float(row["absolute_time_ps"]),
                "distance_A": float(row[column]),
            }
            for row in rows
            if float(row["absolute_time_ps"]) >= minimum_time_ps
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
            f"only {len(selected)} q1 frames satisfy the selection constraints"
        )
    return column, selected


def source_index(molecule: Chem.Mol, serial: int) -> int:
    return metathesis.atom_index_by_source_serial(molecule, serial)


def retained_serials(
    molecule: Chem.Mol, mapping: dict[int, tuple[int, int]]
) -> tuple[int, int]:
    result: list[int] = []
    for vinyl in SECOND_PAIR:
        _, retained = metathesis.terminal_and_retained_indices(molecule, mapping[vinyl])
        result.append(molecule.GetAtomWithIdx(retained).GetIntProp("source_q1_serial"))
    return result[0], result[1]


def second_bond_indices(
    molecule: Chem.Mol, retained: tuple[int, int]
) -> tuple[int, int, int, int]:
    left, right = (source_index(molecule, serial) for serial in retained)
    bond = molecule.GetBondBetweenAtoms(left, right)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError("new V11-V12 double bond is absent")
    stereo_atoms = list(bond.GetStereoAtoms())
    if len(stereo_atoms) != 2:
        raise ValueError("new V11-V12 double bond lacks stereo atoms")
    return stereo_atoms[0], left, right, stereo_atoms[1]


def close_and_optimize(
    molecule: Chem.Mol, retained: tuple[int, int], max_iters: int
) -> tuple[Chem.Mol, dict[str, object]]:
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise ValueError("MMFF94 parameters are incomplete")
    first, left, right, fourth = second_bond_indices(molecule, retained)
    properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94")
    if properties is None:
        raise ValueError("cannot create MMFF94 properties")

    restrained = AllChem.MMFFGetMoleculeForceField(
        molecule, properties, nonBondedThresh=100.0
    )
    restrained.MMFFAddDistanceConstraint(left, right, False, 1.32, 1.36, 250.0)
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
    geometry_rows = metathesis.stereo_geometry_rows(molecule)
    if not math.isfinite(energy):
        raise ValueError("MMFF94 returned a non-finite energy")
    if len(geometry_rows) != 2 or any(
        row["geometry_classification"] != "cis" for row in geometry_rows
    ):
        raise ValueError(f"expected two relaxed cis internal alkenes: {geometry_rows}")
    new_bond = next(
        row for row in geometry_rows
        if set(int(value) for value in str(row["bond_atom_indices"]).split(";"))
        == {left + 1, right + 1}
    )
    return molecule, {
        "restrained_status": restrained_status,
        "relaxed_status": relaxed_status,
        "mmff_energy_kcal_mol": energy,
        "second_bond_length_A": new_bond["bond_length_A"],
        "second_bond_dihedral_deg": new_bond["substituent_dihedral_deg"],
        "stereo_geometry": geometry_rows,
    }


def active_mapping(
    product: Chem.Mol, mapping: dict[int, tuple[int, int]]
) -> dict[int, tuple[int, int]]:
    source_to_output = {
        atom.GetIntProp("source_q1_serial"): atom.GetIdx() + 1
        for atom in product.GetAtoms()
    }
    return {
        vinyl: tuple(source_to_output[serial] for serial in mapping[vinyl])
        for vinyl in EXPECTED_ACTIVE
    }


def write_manifest_row(
    manifest_path: Path,
    q1_row: dict[str, str],
    source_pdb: str,
    mapping: dict[int, tuple[int, int]],
) -> None:
    rows = read_rows(manifest_path)
    q1_schema = json.loads(q1_row["pair_schema_json"])
    active = set(mapping)
    schema = [
        pair for pair in q1_schema
        if int(pair["left"]) in active and int(pair["right"]) in active
    ]
    row = {
        "state": STATE,
        "stage": "q2",
        "system": "B6_p_C6",
        "branch": "natural_second_V11_V12_cis",
        "source_pdb": source_pdb,
        "net_charge": "0",
        "active_vinyl_count": str(len(mapping)),
        "active_pair_count": str(len(schema)),
        "vinyl_mapping_json": json.dumps(
            {f"V{key}": list(value) for key, value in mapping.items()},
            separators=(",", ":"),
        ),
        "pair_schema_json": json.dumps(
            schema, ensure_ascii=False, separators=(",", ":")
        ),
        "output_dir": f"gromacs_dcm/runs/{STATE}",
    }
    if len(schema) != 28:
        raise ValueError(f"expected 28 q2 pairs, found {len(schema)}")
    by_state = {existing["state"]: existing for existing in rows}
    by_state[STATE] = row
    ordered = [
        by_state[existing["state"]]
        for existing in rows
        if existing["state"] != STATE
    ] + [row]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ordered)


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.replicas < 1
        or args.candidate_frames_per_replica < 1
        or args.minimum_time_ps < 0
        or args.minimum_separation_ps < 0
        or args.max_iters < 1
    ):
        raise SystemExit("invalid replica, candidate, time, separation, or iteration value")
    root = Path(args.root).resolve()
    q1_workspace = (root / args.q1_workspace).resolve()
    bundle = root / "gromacs_dcm_server_bundle"
    structures = bundle / "structures"
    result_dir = root / "results/explicit_dcm_b6_q2_v3_v6_v11_v12_build"
    conformer_dir = structures / f"{STATE}_initial_conformers"
    result_dir.mkdir(parents=True, exist_ok=True)
    conformer_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = structures / f"{STATE}.pdb"
    output_mol2 = structures / f"{STATE}_antechamber.mol2"
    audit_path = result_dir / "structure_qc.json"
    if output_pdb.exists() and output_mol2.exists() and audit_path.exists() and not args.force:
        print(audit_path.read_text(encoding="utf-8"))
        return

    gmx = shutil.which("gmx")
    if gmx is None:
        raise RuntimeError("gmx is required to extract q1 candidate frames")
    q1_row = q1_manifest_row(q1_workspace)
    mapping = q1_mapping(q1_row)
    q1_source = bundle / q1_row["source_pdb"]
    template = metathesis.load_rdkit_pdb(q1_source)
    retained = retained_serials(template, mapping)
    initial_formula = rdMolDescriptors.CalcMolFormula(template)
    q1_run = q1_workspace / q1_row["output_dir"]
    screened: list[dict[str, object]] = []
    selected_products: dict[int, Chem.Mol] = {}
    selected_rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="b6_q2_frames_") as temporary:
        temporary_path = Path(temporary)
        for replica in range(1, args.replicas + 1):
            replica_dir = q1_run / f"replica_{replica:02d}"
            distance_column, candidates = select_candidate_frames(
                replica_dir / "vinyl_distances.csv",
                args.candidate_frames_per_replica,
                args.minimum_time_ps,
                args.minimum_separation_ps,
            )
            products: dict[int, Chem.Mol] = {}
            replica_rows: list[dict[str, object]] = []
            for candidate_index, candidate in enumerate(candidates):
                frame_pdb = temporary_path / (
                    f"replica_{replica:02d}_frame_{int(candidate['frame']):06d}.pdb"
                )
                q1_builder.extract_solute_frame(
                    gmx,
                    replica_dir / "production_50ns.tpr",
                    replica_dir / "production_2ns.xtc",
                    q1_run / "setup/index.ndx",
                    float(candidate["time_ps"]),
                    frame_pdb,
                )
                base = {
                    "replica": replica,
                    "distance_column": distance_column,
                    **candidate,
                }
                try:
                    frame = q1_builder.molecule_at_coordinates(
                        template,
                        q1_builder.pdb_coordinates(frame_pdb, template.GetNumAtoms()),
                    )
                    product = metathesis.edit_second_metathesis(
                        frame, mapping, SECOND_PAIR
                    )
                    product, qc = close_and_optimize(product, retained, args.max_iters)
                    products[candidate_index] = product
                    row = {**base, **qc, "passed": True, "error": ""}
                except Exception as error:
                    row = {**base, "passed": False, "error": str(error)}
                replica_rows.append(row)
                screened.append(row)
            passing = [
                index for index, row in enumerate(replica_rows) if bool(row["passed"])
            ]
            if not passing:
                raise RuntimeError(
                    f"no q2 conformer passed for replica {replica}: {replica_rows}"
                )
            selected_index = min(
                passing,
                key=lambda index: float(replica_rows[index]["mmff_energy_kcal_mol"]),
            )
            selected_product = products[selected_index]
            metathesis.renumber_pdb_metadata(selected_product)
            replica_pdb = conformer_dir / f"replica_{replica:02d}.pdb"
            Chem.MolToPDBFile(selected_product, str(replica_pdb), flavor=0)
            selected_products[replica] = selected_product
            selected_rows.append(
                {
                    **replica_rows[selected_index],
                    "output_pdb": str(replica_pdb.relative_to(root)),
                }
            )

    canonical_row = min(
        selected_rows, key=lambda row: float(row["mmff_energy_kcal_mol"])
    )
    canonical_replica = int(canonical_row["replica"])
    canonical = selected_products[canonical_replica]
    source_to_output = metathesis.renumber_pdb_metadata(canonical)
    Chem.MolToPDBFile(canonical, str(output_pdb), flavor=0)
    metathesis.write_antechamber_mol2(canonical, output_mol2)

    product_formula = rdMolDescriptors.CalcMolFormula(canonical)
    if canonical.GetNumAtoms() != 432:
        raise ValueError(f"expected 432 q2 atoms, found {canonical.GetNumAtoms()}")
    if initial_formula != "C184H230O24" or product_formula != "C182H226O24":
        raise ValueError(f"unexpected formulas: {initial_formula} -> {product_formula}")
    cycle_q1 = template.GetNumBonds() - template.GetNumAtoms() + 1
    cycle_q2 = canonical.GetNumBonds() - canonical.GetNumAtoms() + 1
    if cycle_q2 != cycle_q1 + 1:
        raise ValueError("q2 product did not add exactly one graph cycle")
    mapping_q2 = active_mapping(canonical, mapping)
    if tuple(mapping_q2) != EXPECTED_ACTIVE:
        raise ValueError(f"unexpected active q2 mapping: {mapping_q2}")
    relative_source = output_pdb.relative_to(bundle).as_posix()
    write_manifest_row(
        bundle / "gromacs_dcm/state_manifest.csv",
        q1_row,
        relative_source,
        mapping_q2,
    )

    audit = {
        "status": "PASS",
        "state": STATE,
        "method": (
            "three q1 replicas; five separated near-contact frames each; exact C2H4 "
            "loss; restrained/unrestrained MMFF94"
        ),
        "source_q1_state": Q1_STATE,
        "q1_workspace": str(q1_workspace.relative_to(root)),
        "source_q1_formula": initial_formula,
        "q2_formula": product_formula,
        "formula_delta": {"C": 2, "H": 4},
        "q1_atom_count": template.GetNumAtoms(),
        "q2_atom_count": canonical.GetNumAtoms(),
        "adds_one_graph_cycle": True,
        "first_reaction": ["V3", "V6"],
        "second_reaction": ["V11", "V12"],
        "second_double_bond_q1_source_serials": list(retained),
        "second_double_bond_q2_output_serials": [
            source_to_output[retained[0]], source_to_output[retained[1]]
        ],
        "selected_conformers": selected_rows,
        "canonical_replica": canonical_replica,
        "canonical_mmff_energy_kcal_mol": canonical_row["mmff_energy_kcal_mol"],
        "screened_candidate_count": len(screened),
        "passing_candidate_count": sum(bool(row["passed"]) for row in screened),
        "minimum_time_ps": args.minimum_time_ps,
        "minimum_candidate_separation_ps": args.minimum_separation_ps,
        "active_vinyls": [f"V{vinyl}" for vinyl in mapping_q2],
        "active_vinyl_count": len(mapping_q2),
        "active_pair_count": len(mapping_q2) * (len(mapping_q2) - 1) // 2,
        "active_vinyl_mapping_product_pdb_1based": {
            f"V{key}": list(value) for key, value in mapping_q2.items()
        },
        "output_pdb": str(output_pdb.relative_to(root)),
        "output_explicit_bond_mol2": str(output_mol2.relative_to(root)),
        "replica_initial_conformer_dir": str(conformer_dir.relative_to(root)),
        "screened_candidates": screened,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
