#!/usr/bin/env python3
"""Build the common B6 q4 topology from the two converged q3 histories.

The q3 analysis found that

    V3--V6 + V11--V12 + V8--V10 + V7--V9

is the same fourth-bond graph whether V8--V10 or V7--V9 is formed third.
This script therefore edits the missing fourth bond in both q3 branches,
using separated near-contact frames from all six q3 trajectories.  The
result is one q4 Amber topology candidate plus six independent starting
conformers; no new reaction-path assumption is hidden in the construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

import build_b6_q1_md_structure as frame_tools
import build_q2_branch_structures as metathesis


ROOT = Path(__file__).resolve().parents[1]
Q3_BRANCHES = (
    {
        "state": "B6_p_C6_q3_V3_V6_V11_V12_V8_V10_cis",
        "pair": (7, 9),
        "source_role": "q3_V8_V10_then_V7_V9",
    },
    {
        "state": "B6_p_C6_q3_V3_V6_V11_V12_V7_V9_cis",
        "pair": (8, 10),
        "source_role": "q3_V7_V9_then_V8_V10",
    },
)
Q4_STATE = "B6_p_C6_q4_V3_V6_V11_V12_V8_V10_V7_V9_cis"
ACTIVE_Q4 = (1, 2, 4, 5)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def mapping(row: dict[str, str]) -> dict[int, tuple[int, int]]:
    return {
        int(label.removeprefix("V")): tuple(int(value) for value in serials)
        for label, serials in json.loads(row["vinyl_mapping_json"]).items()
    }


def select_frames(path: Path, pair: tuple[int, int], count: int, min_ps: float, sep_ps: float):
    data = rows(path)
    prefix = f"V{pair[0]}_V{pair[1]}_"
    columns = [name for name in data[0] if name.startswith(prefix)]
    if len(columns) != 1:
        raise ValueError(f"expected one {prefix} column in {path}, found {columns}")
    column = columns[0]
    ranked = sorted(
        (
            {"frame": int(row["frame"]), "time_ps": float(row["absolute_time_ps"]),
             "distance_A": float(row[column])}
            for row in data if float(row["absolute_time_ps"]) >= min_ps
        ),
        key=lambda item: (item["distance_A"], item["frame"]),
    )
    selected = []
    for item in ranked:
        if all(abs(item["time_ps"] - old["time_ps"]) >= sep_ps for old in selected):
            selected.append(item)
            if len(selected) == count:
                return column, selected
    raise ValueError(f"only {len(selected)} separated frames found in {path}")


def source_index(molecule: Chem.Mol, serial: int) -> int:
    return metathesis.atom_index_by_source_serial(molecule, serial)


def active_mapping(product: Chem.Mol, source_mapping: dict[int, tuple[int, int]]):
    source_to_output = {
        atom.GetIntProp("source_q1_serial"): atom.GetIdx() + 1
        for atom in product.GetAtoms()
    }
    return {
        vinyl: tuple(source_to_output[serial] for serial in source_mapping[vinyl])
        for vinyl in ACTIVE_Q4
    }


def q4_edit(molecule: Chem.Mol, source_mapping: dict[int, tuple[int, int]], pair):
    product = metathesis.edit_second_metathesis(molecule, source_mapping, pair)
    retained = []
    for vinyl in pair:
        _, idx = metathesis.terminal_and_retained_indices(molecule, source_mapping[vinyl])
        retained.append(molecule.GetAtomWithIdx(idx).GetIntProp("source_q1_serial"))
    bond = product.GetBondBetweenAtoms(*(source_index(product, serial) for serial in retained))
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() != Chem.BondStereo.STEREOZ:
        raise ValueError(f"q4 bond V{pair[0]}-V{pair[1]} is not Z/cis")
    return product, tuple(retained)


def close_and_optimize(product: Chem.Mol, retained: tuple[int, int], max_iters: int = 10_000):
    """Regularize the newly formed q4 Z alkene before Amber parameterization."""
    if not AllChem.MMFFHasAllMoleculeParams(product):
        raise ValueError("MMFF94 parameters are incomplete for q4")
    left, right = (source_index(product, serial) for serial in retained)
    bond = product.GetBondBetweenAtoms(left, right)
    stereo_atoms = list(bond.GetStereoAtoms()) if bond is not None else []
    if len(stereo_atoms) != 2:
        raise ValueError("new q4 double bond lacks stereo atoms")
    first, fourth = stereo_atoms
    properties = AllChem.MMFFGetMoleculeProperties(product, mmffVariant="MMFF94")
    if properties is None:
        raise ValueError("cannot create q4 MMFF94 properties")
    restrained = AllChem.MMFFGetMoleculeForceField(product, properties, nonBondedThresh=100.0)
    restrained.MMFFAddDistanceConstraint(left, right, False, 1.32, 1.36, 250.0)
    restrained.MMFFAddTorsionConstraint(first, left, right, fourth, False, -20.0, 20.0, 100.0)
    restrained.Initialize()
    restrained_status = int(restrained.Minimize(maxIts=max_iters))
    relaxed = AllChem.MMFFGetMoleculeForceField(product, properties, nonBondedThresh=100.0)
    relaxed.Initialize()
    relaxed_status = int(relaxed.Minimize(maxIts=max_iters))
    energy = float(relaxed.CalcEnergy())
    geometry_rows = metathesis.stereo_geometry_rows(product)
    if not math.isfinite(energy):
        raise ValueError("q4 MMFF94 returned a non-finite energy")
    if len(geometry_rows) != 4 or any(row["geometry_classification"] != "cis" for row in geometry_rows):
        raise ValueError(f"expected four relaxed cis internal alkenes: {geometry_rows}")
    new_bond = next(
        row for row in geometry_rows
        if set(int(value) for value in str(row["bond_atom_indices"]).split(";")) == {left + 1, right + 1}
    )
    length = float(new_bond["bond_length_A"])
    if not 1.25 <= length <= 1.45:
        raise ValueError(f"new q4 C=C remains unphysical after optimization: {length:.3f} A")
    return product, {
        "restrained_status": restrained_status,
        "relaxed_status": relaxed_status,
        "mmff_energy_kcal_mol": energy,
        "fourth_bond_length_A": length,
        "fourth_bond_dihedral_deg": float(new_bond["substituent_dihedral_deg"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--q3-workspace", default="results/b6_q3_20ns_merged/workspace")
    parser.add_argument("--frames-per-replica", type=int, default=1)
    parser.add_argument("--minimum-time-ps", type=float, default=5000.0)
    parser.add_argument("--minimum-separation-ps", type=float, default=1000.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    workspace = (root / args.q3_workspace).resolve()
    bundle = workspace
    manifest_path = bundle / "gromacs_dcm/state_manifest.csv"
    manifest = rows(manifest_path)
    output_dir = bundle / "structures"
    conformer_dir = output_dir / f"{Q4_STATE}_initial_conformers"
    audit_dir = root / "results/explicit_dcm_b6_q4_build"
    output_pdb = output_dir / f"{Q4_STATE}.pdb"
    output_mol2 = output_dir / f"{Q4_STATE}_antechamber.mol2"
    audit_path = audit_dir / "structure_qc.json"
    if output_pdb.exists() and output_mol2.exists() and audit_path.exists() and not args.force:
        print(audit_path.read_text(encoding="utf-8"))
        return
    gmx = shutil.which("gmx")
    if gmx is None:
        raise RuntimeError("gmx is required to extract q3 near-contact frames")
    conformer_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    products = []
    for branch in Q3_BRANCHES:
        row = next(item for item in manifest if item["state"] == branch["state"])
        source_mapping = mapping(row)
        template = metathesis.load_rdkit_pdb(bundle / row["source_pdb"])
        run_root = bundle / row["output_dir"]
        for replica in range(1, 4):
            replica_dir = run_root / f"replica_{replica:02d}"
            column, candidates = select_frames(
                replica_dir / "vinyl_distances.csv", branch["pair"],
                args.frames_per_replica, args.minimum_time_ps, args.minimum_separation_ps,
            )
            candidate = candidates[0]
            frame_pdb = audit_dir / "frames" / f"{branch['source_role']}_replica_{replica:02d}.pdb"
            frame_pdb.parent.mkdir(parents=True, exist_ok=True)
            frame_tools.extract_solute_frame(
                gmx, replica_dir / "production_20ns.tpr", replica_dir / "production_2ns.xtc",
                run_root / "setup/index.ndx", candidate["time_ps"], frame_pdb,
            )
            frame = frame_tools.molecule_at_coordinates(template, frame_tools.pdb_coordinates(frame_pdb, template.GetNumAtoms()))
            product, retained = q4_edit(frame, source_mapping, branch["pair"])
            product, geometry_qc = close_and_optimize(product, retained)
            if product.GetNumAtoms() != 420 or rdMolDescriptors.CalcMolFormula(product) != "C178H218O24":
                raise ValueError(f"unexpected q4 formula/atom count: {rdMolDescriptors.CalcMolFormula(product)}, {product.GetNumAtoms()}")
            metathesis.renumber_pdb_metadata(product)
            out = conformer_dir / f"replica_{len(products)+1:02d}.pdb"
            Chem.MolToPDBFile(product, str(out), flavor=0)
            products.append(product)
            selected.append({"source_role": branch["source_role"], "replica": replica, "distance_column": column, **candidate, **geometry_qc, "output_pdb": str(out.relative_to(root))})
    canonical = products[0]
    Chem.MolToPDBFile(canonical, str(output_pdb), flavor=0)
    metathesis.write_antechamber_mol2(canonical, output_mol2)
    # q4 rows inherit the q3 schema but retain only the four unreacted vinyls.
    q3_row = next(item for item in manifest if item["state"] == Q3_BRANCHES[0]["state"])
    q4_mapping = active_mapping(canonical, mapping(q3_row))
    q3_schema = json.loads(q3_row["pair_schema_json"])
    schema = [pair for pair in q3_schema if int(pair["left"]) in ACTIVE_Q4 and int(pair["right"]) in ACTIVE_Q4]
    q4_row = dict(q3_row)
    q4_row.update({"state": Q4_STATE, "stage": "q4", "branch": "common_q4_from_both_q3_histories", "source_pdb": str(output_pdb.relative_to(bundle)), "active_vinyl_count": "4", "active_pair_count": str(len(schema)), "vinyl_mapping_json": json.dumps({f"V{k}": list(v) for k,v in q4_mapping.items()}, separators=(",", ":")), "pair_schema_json": json.dumps(schema, separators=(",", ":")), "output_dir": f"gromacs_dcm/runs/{Q4_STATE}"})
    by_state = {item["state"]: item for item in manifest}
    by_state[Q4_STATE] = q4_row
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(by_state.values())
    audit = {"status": "PASS", "state": Q4_STATE, "formula": "C178H218O24", "atom_count": 420, "fourth_bond_graph": [["V3","V6"],["V11","V12"],["V8","V10"],["V7","V9"]], "active_vinyls": [f"V{x}" for x in ACTIVE_Q4], "selected_conformers": selected, "output_pdb": str(output_pdb.relative_to(root)), "output_explicit_bond_mol2": str(output_mol2.relative_to(root)), "initial_conformer_dir": str(conformer_dir.relative_to(root)), "next_step": "parameterize one q4 topology, then run 3 independent q4 pilot replicas before constructing q5"
    }
    audit_dir.mkdir(parents=True, exist_ok=True); audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
