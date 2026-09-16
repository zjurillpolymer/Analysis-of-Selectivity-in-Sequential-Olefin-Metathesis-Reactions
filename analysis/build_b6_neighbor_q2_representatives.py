#!/usr/bin/env python3
"""Build the small set of B6 neighbour-only q2 representative structures.

The preceding q0 -> q1 structure is fixed once.  This script applies the q1
representative pairs from ``results/b6_neighbor_orbit_table/manifest.json``
and writes one cis q2 structure per physical orbit.  It intentionally does
not expand the 17 labelled q1 candidates into 17 molecular dynamics systems;
their multiplicities remain in the orbit table and are used later in the
conditional probability.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import build_q2_branch_structures as metathesis


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_pair(text: str) -> tuple[int, int]:
    values = tuple(sorted(int(item.removeprefix("V")) for item in text.split("-")))
    if len(values) != 2 or values[0] == values[1]:
        raise ValueError(f"invalid vinyl pair: {text}")
    return values


def source_index(molecule: Chem.Mol, serial: int) -> int:
    matches = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.HasProp("source_q1_serial")
        and atom.GetIntProp("source_q1_serial") == serial
    ]
    if len(matches) != 1:
        raise ValueError(f"source serial {serial} maps to {matches}")
    return matches[0]


def bond_by_serials(molecule: Chem.Mol, serials: tuple[int, int]) -> Chem.Bond:
    indices = tuple(source_index(molecule, serial) for serial in serials)
    bond = molecule.GetBondBetweenAtoms(*indices)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError(f"expected double bond between source serials {serials}")
    return bond


def q1_source(root: Path) -> tuple[Chem.Mol, dict[int, tuple[int, int]], dict[str, str]]:
    bundle = root / "gromacs_dcm_server_bundle"
    rows = read_csv(bundle / "gromacs_dcm/state_manifest.csv")
    row = next(item for item in rows if item["state"] == "B6_p_C6_q1_V1_V2_cis")
    mapping = {
        int(label.removeprefix("V")): tuple(int(value) for value in serials)
        for label, serials in json.loads(row["vinyl_mapping_json"]).items()
    }
    source = bundle / row["source_pdb"]
    return metathesis.load_rdkit_pdb(source), mapping, row


def update_manifest(
    path: Path,
    source_pdb: str,
    state: str,
    branch: str,
    mapping: dict[int, tuple[int, int]],
) -> None:
    rows = read_csv(path)
    original = next(item for item in rows if item["state"] == "B6_p_C6_q1_V1_V2_cis")
    original_schema = json.loads(original["pair_schema_json"])
    active = set(mapping)
    schema = [
        pair
        for pair in original_schema
        if int(pair["left"]) in active and int(pair["right"]) in active
    ]
    row = {
        "state": state,
        "stage": "q2",
        "system": "B6_p_C6",
        "branch": branch,
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
    by_state = {item["state"]: item for item in rows}
    by_state[state] = row
    ordered = [by_state[item["state"]] for item in rows if item["state"] != state]
    ordered.append(row)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ordered)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--conformers", type=int, default=4)
    args = parser.parse_args()
    if args.conformers < 1:
        parser.error("--conformers must be positive")
    root = Path(args.root).resolve()
    bundle = root / "gromacs_dcm_server_bundle"
    q1, mapping, q1_row = q1_source(root)
    q1_audit_path = root / "results/explicit_dcm_b6_q1_v1_v2_build/structure_qc.json"
    q1_audit = json.loads(q1_audit_path.read_text(encoding="utf-8"))
    q1_first_bond = tuple(
        int(value) for value in q1_audit["new_double_bond_output_serials"]
    )
    # The q1 mapping contains only active vinyls; the inherited V1--V2 bond
    # is therefore read from the q1 structure audit instead of the mapping.
    first_bond = bond_by_serials(q1, q1_first_bond)
    if first_bond.GetStereo() != Chem.BondStereo.STEREOZ:
        raise ValueError("inherited q0 -> q1 V1-V2 bond is not Z/cis")

    orbit_manifest = json.loads(
        (root / "results/b6_neighbor_orbit_table/manifest.json").read_text(encoding="utf-8")
    )
    representatives = orbit_manifest["representative_q1_pairs"]
    output_dir = bundle / "structures"
    manifest_path = bundle / "gromacs_dcm/state_manifest.csv"
    audits: list[dict[str, object]] = []
    for orbit_index, (category, pair_text) in enumerate(representatives.items()):
        pair = parse_pair(pair_text)
        if not set(pair).issubset(mapping):
            raise ValueError(f"{category} pair {pair_text} is not active in q1")
        state = f"B6_p_C6_q2_neighbor_{category}_cis"
        output_pdb = output_dir / f"{state}.pdb"
        output_mol2 = output_dir / f"{state}_antechamber.mol2"
        audit_path = root / f"results/{state}/structure_qc.json"
        if output_pdb.exists() and output_mol2.exists() and audit_path.exists() and not args.force:
            audits.append(json.loads(audit_path.read_text(encoding="utf-8")))
            continue

        terminal_retained = [
            metathesis.terminal_and_retained_indices(q1, mapping[vinyl])
            for vinyl in pair
        ]
        second_serials = tuple(
            q1.GetAtomWithIdx(retained).GetIntProp("source_q1_serial")
            for _, retained in terminal_retained
        )
        product = metathesis.edit_second_metathesis(q1, mapping, pair)
        # A graph edit alone leaves the new C=C at the original inter-arm
        # separation (5--27 A in the first Amber batch). Re-embed the whole
        # q2 molecule with both internal alkenes constrained to Z/cis before
        # exporting coordinates for parameterization or MD.
        product, mmff_energy, mmff_status = metathesis.embed_and_optimize(
            product, seed=20260916 + orbit_index, conformers=args.conformers
        )
        if mmff_status != 0:
            raise ValueError(f"{category} MMFF94 optimization did not converge")
        second_bond = bond_by_serials(product, second_serials)
        if second_bond.GetStereo() != Chem.BondStereo.STEREOZ:
            raise ValueError(f"{category} new bond is not Z/cis")
        geometric_stereo = Chem.Mol(product)
        Chem.AssignStereochemistryFrom3D(geometric_stereo)
        for bond_serials in (q1_first_bond, second_serials):
            if bond_by_serials(geometric_stereo, bond_serials).GetStereo() != Chem.BondStereo.STEREOZ:
                raise ValueError(f"{category} optimized coordinates do not preserve both cis alkenes")
        conformer = product.GetConformer()
        bond_lengths = [
            conformer.GetAtomPosition(bond.GetBeginAtomIdx()).Distance(
                conformer.GetAtomPosition(bond.GetEndAtomIdx())
            )
            for bond in product.GetBonds()
        ]
        new_bond_length = bond_lengths[second_bond.GetIdx()]
        if not 1.20 <= new_bond_length <= 1.50 or max(bond_lengths) > 2.20:
            raise ValueError(
                f"{category} optimized bond geometry is invalid: "
                f"new C=C {new_bond_length:.2f} A; max bond {max(bond_lengths):.2f} A"
            )
        initial_atoms = Counter(atom.GetSymbol() for atom in q1.GetAtoms())
        final_atoms = Counter(atom.GetSymbol() for atom in product.GetAtoms())
        if initial_atoms - final_atoms != Counter({"C": 2, "H": 4}):
            raise ValueError(f"{category} unexpected formula delta: {initial_atoms - final_atoms}")
        if product.GetNumAtoms() != q1.GetNumAtoms() - 6:
            raise ValueError(f"{category} unexpected atom count: {product.GetNumAtoms()}")
        cycle_q1 = q1.GetNumBonds() - q1.GetNumAtoms() + 1
        cycle_q2 = product.GetNumBonds() - product.GetNumAtoms() + 1
        if cycle_q2 != cycle_q1 + 1:
            raise ValueError(f"{category} q2 edit did not add one graph cycle")

        Chem.MolToPDBFile(product, str(output_pdb), flavor=0)
        metathesis.write_antechamber_mol2(product, output_mol2)
        source_to_output = {
            atom.GetIntProp("source_q1_serial"): atom.GetIdx() + 1
            for atom in product.GetAtoms()
        }
        active = sorted(set(mapping) - set(pair))
        active_mapping = {
            vinyl: tuple(source_to_output[serial] for serial in mapping[vinyl])
            for vinyl in active
        }
        relative_source = output_pdb.relative_to(bundle).as_posix()
        update_manifest(manifest_path, relative_source, state, category, active_mapping)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit = {
            "status": "PASS",
            "state": state,
            "orbit": category,
            "source_q1": str((bundle / q1_row["source_pdb"]).relative_to(root)),
            "first_reaction": ["V1", "V2"],
            "second_reaction": [f"V{pair[0]}", f"V{pair[1]}"],
            "second_reaction_source_serials": list(second_serials),
            "both_internal_double_bonds": "Z/cis",
            "geometric_stereo_from_3d": "Z/cis for both internal double bonds",
            "new_double_bond_length_A": round(new_bond_length, 4),
            "maximum_bond_length_A": round(max(bond_lengths), 4),
            "mmff_energy_kcal_mol": round(mmff_energy, 4),
            "mmff_optimization_status": mmff_status,
            "etkdg_conformers": args.conformers,
            "etkdg_seed": 20260916 + orbit_index,
            "q1_formula": rdMolDescriptors.CalcMolFormula(q1),
            "q2_formula": rdMolDescriptors.CalcMolFormula(product),
            "q1_atom_count": q1.GetNumAtoms(),
            "q2_atom_count": product.GetNumAtoms(),
            "active_vinyl_count": len(active_mapping),
            "active_pair_count": len(active_mapping) * (len(active_mapping) - 1) // 2,
            "output_pdb": str(output_pdb.relative_to(root)),
            "output_explicit_bond_mol2": str(output_mol2.relative_to(root)),
        }
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audits.append(audit)

    print(json.dumps({"status": "PASS", "built": audits}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
