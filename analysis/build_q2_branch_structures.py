#!/usr/bin/env python3
"""Construct and audit the two symmetry-distinct A4 q2 branch structures.

Each structure starts from the validated cis q1 PDB.  A second terminal-alkene
metathesis removes C2H4, connects the two retained alkene carbons with a cis
double bond, regenerates a 3D conformer with ETKDG, and pre-optimizes it with
MMFF94.  The selected adjacent/opposite pairs are representative topology
states, not additional independent chemical classes.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds, rdMolDescriptors

import analyze_react1_structures as react1
import mtmr_baseline as baseline


@dataclass(frozen=True)
class BranchConfig:
    system: str
    branch: str
    second_pair: tuple[int, int]
    selection_basis: str

    @property
    def q1_state(self) -> str:
        return f"{self.system}_1_cis"

    @property
    def q2_state(self) -> str:
        return f"{self.system}_2_{self.branch}_cis"


BRANCH_CONFIGS: tuple[BranchConfig, ...] = (
    BranchConfig("A4_m_C5", "adjacent", (4, 6), "largest mapped q1 soft-contact contribution"),
    BranchConfig("A4_m_C5", "opposite", (3, 5), "largest mapped q1 soft-contact contribution"),
    BranchConfig("A4_m_C6", "adjacent", (1, 5), "closest valid q1-PDB pair; q1 atom labels unresolved"),
    BranchConfig("A4_m_C6", "opposite", (5, 6), "closest valid q1-PDB pair; q1 atom labels unresolved"),
    BranchConfig("A4_m_C7", "adjacent", (1, 5), "largest mapped q1 soft-contact contribution"),
    BranchConfig("A4_m_C7", "opposite", (1, 8), "largest mapped q1 soft-contact contribution"),
    BranchConfig("A4_m_C8", "adjacent", (3, 7), "largest mapped q1 soft-contact contribution"),
    BranchConfig("A4_m_C8", "opposite", (1, 3), "largest mapped q1 soft-contact contribution"),
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def q1_mapping(root: Path, state: str) -> dict[int, tuple[int, int]]:
    rows = read_rows(root / "results" / "react1" / "vinyl_mapping.csv")
    mapping = {
        int(row["vinyl"].removeprefix("V")): tuple(
            int(value) for value in row["product_pdb_atoms"].split(";")
        )
        for row in rows
        if row["state"] == state and row["vinyl_status"] == "active"
    }
    if len(mapping) != 6:
        raise ValueError(f"expected six q1 active vinyls for {state}, found {mapping}")
    return mapping


def load_rdkit_pdb(path: Path) -> Chem.Mol:
    molecule = Chem.MolFromPDBFile(
        str(path), removeHs=False, sanitize=False, proximityBonding=False
    )
    if molecule is None:
        raise ValueError(f"RDKit cannot read {path}")
    rdDetermineBonds.DetermineBondOrders(molecule, charge=0)
    Chem.SanitizeMol(molecule)
    Chem.AssignStereochemistryFrom3D(molecule, replaceExistingTags=True)
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info is None:
            raise ValueError(f"atom {atom.GetIdx()} in {path} lacks PDB serial metadata")
        atom.SetIntProp("source_q1_serial", int(info.GetSerialNumber()))
    return molecule


def atom_index_by_source_serial(molecule: Chem.Mol, serial: int) -> int:
    matches = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.HasProp("source_q1_serial") and atom.GetIntProp("source_q1_serial") == serial
    ]
    if len(matches) != 1:
        raise ValueError(f"source q1 serial {serial} maps to {matches}")
    return matches[0]


def terminal_and_retained_indices(
    molecule: Chem.Mol, atom_serials: tuple[int, int]
) -> tuple[int, int]:
    indices = tuple(atom_index_by_source_serial(molecule, serial) for serial in atom_serials)
    heavy_degrees = tuple(
        sum(neighbor.GetAtomicNum() > 1 for neighbor in molecule.GetAtomWithIdx(index).GetNeighbors())
        for index in indices
    )
    if heavy_degrees[0] == 1 and heavy_degrees[1] > 1:
        return indices[0], indices[1]
    if heavy_degrees[1] == 1 and heavy_degrees[0] > 1:
        return indices[1], indices[0]
    raise ValueError(
        f"active vinyl {atom_serials} lacks one terminal carbon; heavy degrees={heavy_degrees}"
    )


def set_cis_double_bond(molecule: Chem.Mol, left: int, right: int) -> None:
    bond = molecule.GetBondBetweenAtoms(left, right)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError("new retained-carbon connection is not a double bond")
    left_neighbors = [
        atom.GetIdx()
        for atom in molecule.GetAtomWithIdx(left).GetNeighbors()
        if atom.GetIdx() != right and atom.GetAtomicNum() > 1
    ]
    right_neighbors = [
        atom.GetIdx()
        for atom in molecule.GetAtomWithIdx(right).GetNeighbors()
        if atom.GetIdx() != left and atom.GetAtomicNum() > 1
    ]
    if len(left_neighbors) != 1 or len(right_neighbors) != 1:
        raise ValueError("new internal alkene does not have one heavy substituent on each carbon")
    bond.SetStereoAtoms(left_neighbors[0], right_neighbors[0])
    bond.SetStereo(Chem.BondStereo.STEREOZ)


def edit_second_metathesis(
    molecule: Chem.Mol,
    mapping: dict[int, tuple[int, int]],
    consumed_pair: tuple[int, int],
) -> Chem.Mol:
    terminal_retained = [
        terminal_and_retained_indices(molecule, mapping[vinyl]) for vinyl in consumed_pair
    ]
    retained_serials = [
        molecule.GetAtomWithIdx(retained).GetIntProp("source_q1_serial")
        for _, retained in terminal_retained
    ]
    delete_indices: set[int] = set()
    for terminal, _ in terminal_retained:
        terminal_atom = molecule.GetAtomWithIdx(terminal)
        delete_indices.add(terminal)
        delete_indices.update(
            neighbor.GetIdx() for neighbor in terminal_atom.GetNeighbors() if neighbor.GetAtomicNum() == 1
        )
    removed_formula = Counter(
        molecule.GetAtomWithIdx(index).GetSymbol() for index in delete_indices
    )
    if removed_formula != Counter({"C": 2, "H": 4}):
        raise ValueError(f"second metathesis would remove {removed_formula}, expected C2H4")
    editable = Chem.RWMol(molecule)
    for index in sorted(delete_indices, reverse=True):
        editable.RemoveAtom(index)
    product = editable.GetMol()
    Chem.SanitizeMol(product)
    retained = [atom_index_by_source_serial(product, serial) for serial in retained_serials]
    editable = Chem.RWMol(product)
    editable.AddBond(retained[0], retained[1], Chem.BondType.DOUBLE)
    product = editable.GetMol()
    Chem.SanitizeMol(product)
    retained = [atom_index_by_source_serial(product, serial) for serial in retained_serials]
    set_cis_double_bond(product, retained[0], retained[1])
    if not AllChem.MMFFHasAllMoleculeParams(product):
        raise ValueError("MMFF94 parameters are incomplete for edited product")
    return product


def embed_and_optimize(
    molecule: Chem.Mol, seed: int, conformers: int = 8
) -> tuple[Chem.Mol, float, int]:
    molecule.RemoveAllConformers()
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.useRandomCoords = True
    parameters.useMacrocycleTorsions = True
    parameters.useMacrocycle14config = True
    parameters.maxIterations = 3000
    parameters.pruneRmsThresh = -1.0
    conformer_ids = list(AllChem.EmbedMultipleConfs(molecule, numConfs=conformers, params=parameters))
    if not conformer_ids:
        raise RuntimeError("ETKDG failed to generate a q2 macrocycle conformer")
    results = AllChem.MMFFOptimizeMoleculeConfs(
        molecule, numThreads=0, maxIters=5000, nonBondedThresh=100.0
    )
    candidates = [
        (float(energy), int(status), int(conformer_id))
        for conformer_id, (status, energy) in zip(conformer_ids, results)
        if math.isfinite(float(energy))
    ]
    if not candidates:
        raise RuntimeError("MMFF94 returned no finite q2 conformer energies")
    energy, status, selected_id = min(candidates)
    selected_conformer = Chem.Conformer(molecule.GetConformer(selected_id))
    molecule.RemoveAllConformers()
    molecule.AddConformer(selected_conformer, assignId=True)
    return molecule, energy, status


def renumber_pdb_metadata(molecule: Chem.Mol) -> dict[int, int]:
    source_to_output: dict[int, int] = {}
    for atom in molecule.GetAtoms():
        output_serial = atom.GetIdx() + 1
        source_serial = atom.GetIntProp("source_q1_serial")
        source_to_output[source_serial] = output_serial
        symbol = atom.GetSymbol().upper()
        atom_name = f" {symbol:<3}" if len(symbol) == 1 else f"{symbol:<4}"
        info = Chem.AtomPDBResidueInfo(
            atomName=atom_name,
            serialNumber=output_serial,
            residueName="UNK",
            residueNumber=1,
            isHeteroAtom=False,
        )
        atom.SetMonomerInfo(info)
    return source_to_output


def sybyl_atom_type(atom: Chem.Atom) -> str:
    if atom.GetIsAromatic():
        return f"{atom.GetSymbol()}.ar"
    has_multiple_bond = any(
        bond.GetBondTypeAsDouble() > 1.0 for bond in atom.GetBonds()
    )
    if atom.GetSymbol() == "C":
        return "C.2" if has_multiple_bond else "C.3"
    if atom.GetSymbol() == "O":
        return "O.2" if has_multiple_bond else "O.3"
    if atom.GetSymbol() == "N":
        return "N.2" if has_multiple_bond else "N.3"
    return atom.GetSymbol()


def write_antechamber_mol2(
    molecule: Chem.Mol,
    path: Path,
    residue_name: str = "SLU",
) -> None:
    """Export explicit RDKit bond orders for partial Antechamber judgement."""
    conformer = molecule.GetConformer()
    element_counts: Counter[str] = Counter()
    atom_lines: list[str] = []
    for index, atom in enumerate(molecule.GetAtoms(), 1):
        element = atom.GetSymbol()
        element_counts[element] += 1
        name = f"{element}{element_counts[element]}"
        position = conformer.GetAtomPosition(atom.GetIdx())
        atom_lines.append(
            f"{index:7d} {name:<8} {position.x:10.4f} {position.y:10.4f} "
            f"{position.z:10.4f} {sybyl_atom_type(atom):<8} "
            f"1 {residue_name:<8} 0.000000"
        )
    bond_lines: list[str] = []
    for index, bond in enumerate(molecule.GetBonds(), 1):
        if bond.GetIsAromatic():
            bond_type = "ar"
        else:
            order = bond.GetBondTypeAsDouble()
            if order not in {1.0, 2.0, 3.0}:
                raise ValueError(f"unsupported MOL2 bond order: {order}")
            bond_type = str(int(order))
        bond_lines.append(
            f"{index:6d} {bond.GetBeginAtomIdx()+1:6d} "
            f"{bond.GetEndAtomIdx()+1:6d} {bond_type}"
        )
    path.write_text(
        "\n".join(
            [
                "@<TRIPOS>MOLECULE",
                f"{residue_name}_EXPLICIT_BONDS",
                f"{molecule.GetNumAtoms()} {molecule.GetNumBonds()} 1 0 0",
                "SMALL",
                "NO_CHARGES",
                "",
                "@<TRIPOS>ATOM",
                *atom_lines,
                "@<TRIPOS>BOND",
                *bond_lines,
                "@<TRIPOS>SUBSTRUCTURE",
                f"     1 {residue_name:<8} 1 RESIDUE 1 A {residue_name:<8} 1 ROOT",
                "",
            ]
        ),
        encoding="utf-8",
    )


def double_bond_geometry(molecule: Chem.Mol, bond: Chem.Bond) -> tuple[float, float]:
    left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    stereo_atoms = list(bond.GetStereoAtoms())
    if len(stereo_atoms) != 2:
        raise ValueError("stereogenic double bond lacks stereo atoms")
    conformer = molecule.GetConformer()
    points = [
        np.asarray(conformer.GetAtomPosition(index), dtype=float)
        for index in (stereo_atoms[0], left, right, stereo_atoms[1])
    ]
    length = float(np.linalg.norm(points[1] - points[2]))
    dihedral = react1.dihedral_degrees(*points)
    return length, dihedral


def stereo_geometry_rows(molecule: Chem.Mol) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bond in molecule.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        length, dihedral = double_bond_geometry(molecule, bond)
        rows.append(
            {
                "bond_atom_indices": f"{bond.GetBeginAtomIdx()+1};{bond.GetEndAtomIdx()+1}",
                "rdkit_stereo": str(bond.GetStereo()),
                "bond_length_A": length,
                "substituent_dihedral_deg": dihedral,
                "geometry_classification": react1.geometry_label(length, dihedral),
            }
        )
    return rows


def active_pair_header(
    active: Sequence[int], terminal_to_arm: dict[int, int]
) -> list[str]:
    columns = ["step", "time_ps", "energy_eV"]
    for left, right in combinations(active, 2):
        kind = "同臂" if terminal_to_arm[left] == terminal_to_arm[right] else "跨臂"
        columns.append(f"V{left}_V{right}_{kind}")
    return columns


def branch_role(
    first_pair: tuple[int, int],
    second_pair: tuple[int, int],
    terminal_to_arm: dict[int, int],
) -> str:
    first_arms = {terminal_to_arm[vinyl] for vinyl in first_pair}
    second_arms = {terminal_to_arm[vinyl] for vinyl in second_pair}
    shared = len(first_arms & second_arms)
    if shared == 1:
        return "adjacent"
    if shared == 0:
        return "opposite"
    return "same_edge"


def construct_one(
    root: Path,
    output_dir: Path,
    config: BranchConfig,
    conformers: int,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    q1_path = root / "react_1" / f"{config.q1_state}.pdb"
    mapping = q1_mapping(root, config.q1_state)
    if not set(config.second_pair) <= set(mapping):
        raise ValueError(f"second pair {config.second_pair} is not active in {config.q1_state}")
    system_data, _ = baseline.analyse_system(
        root, root / "trace" / f"{config.system}_md", baseline.DEFAULT_THRESHOLDS
    )
    if system_data is None:
        raise RuntimeError(f"cannot load {config.system} topology")
    terminal_to_arm = {
        vinyl: int(system_data.terminal_to_monomer[vinyl - 1]) for vinyl in range(1, 9)
    }
    first_row = next(
        row
        for row in read_rows(root / "results" / "react1" / "structure_qc.csv")
        if row["state"] == config.q1_state
    )
    first_pair = tuple(int(value.removeprefix("V")) for value in first_row["reacted_pair"].split("-"))
    observed_branch = branch_role(first_pair, config.second_pair, terminal_to_arm)
    if observed_branch != config.branch:
        raise ValueError(
            f"configured {config.branch} pair {config.second_pair} is actually {observed_branch}"
        )

    q1_molecule = load_rdkit_pdb(q1_path)
    q2_molecule = edit_second_metathesis(q1_molecule, mapping, config.second_pair)
    seed = baseline.stable_seed(20260722, config.q2_state) % (2**31 - 1)
    q2_molecule, mmff_energy, optimization_status = embed_and_optimize(
        q2_molecule, seed, conformers
    )
    geometry_rows = stereo_geometry_rows(q2_molecule)
    if len(geometry_rows) != 2 or any(
        row["geometry_classification"] != "cis" for row in geometry_rows
    ):
        raise ValueError(f"expected two cis internal alkenes in {config.q2_state}: {geometry_rows}")
    source_to_output = renumber_pdb_metadata(q2_molecule)
    q2_path = output_dir / f"{config.q2_state}.pdb"
    Chem.MolToPDBFile(q2_molecule, str(q2_path), flavor=0)
    write_antechamber_mol2(
        q2_molecule,
        output_dir / f"{config.q2_state}_antechamber.mol2",
    )

    q1_graph = react1.read_pdb(q1_path)
    q2_graph = react1.read_pdb(q2_path)
    edit = react1.expected_edit(q1_graph, q2_graph, mapping, config.second_pair)
    if edit is None:
        raise ValueError(f"saved PDB does not reconcile to the intended edit: {q2_path}")
    length, dihedral = react1.new_bond_geometry(q2_graph, edit)
    formula_delta = q1_graph.formula_counts - q2_graph.formula_counts
    active = sorted(set(mapping) - set(config.second_pair))
    output_mapping = {
        vinyl: tuple(source_to_output[serial] for serial in mapping[vinyl]) for vinyl in active
    }
    qc = {
        "state": config.q2_state,
        "system": config.system,
        "branch": config.branch,
        "status": "PASS",
        "source_q1_pdb": str(q1_path.relative_to(root)),
        "output_q2_pdb": str(q2_path.relative_to(root)),
        "first_reacted_pair": f"V{first_pair[0]}-V{first_pair[1]}",
        "second_reacted_pair": f"V{config.second_pair[0]}-V{config.second_pair[1]}",
        "selection_basis": config.selection_basis,
        "q1_formula": react1.formula_text(q1_graph.formula_counts),
        "q2_formula": react1.formula_text(q2_graph.formula_counts),
        "q1_to_q2_formula_delta": react1.formula_text(formula_delta),
        "formula_delta_is_C2H4": formula_delta == Counter({"C": 2, "H": 4}),
        "q1_cycle_rank": q1_graph.cycle_rank,
        "q2_cycle_rank": q2_graph.cycle_rank,
        "adds_one_cycle": q2_graph.cycle_rank == q1_graph.cycle_rank + 1,
        "active_vinyl_count": len(active),
        "active_pair_count": len(active) * (len(active) - 1) // 2,
        "new_bond_length_A": length,
        "new_bond_dihedral_deg": dihedral,
        "new_bond_geometry": react1.geometry_label(length, dihedral),
        "stereogenic_internal_double_bonds": len(geometry_rows),
        "all_internal_double_bonds_cis": all(
            row["geometry_classification"] == "cis" for row in geometry_rows
        ),
        "mmff_energy_kcal_mol": mmff_energy,
        "mmff_optimization_status": optimization_status,
        "rdkit_formula": rdMolDescriptors.CalcMolFormula(q2_molecule),
    }
    if not all(
        (
            qc["formula_delta_is_C2H4"],
            qc["adds_one_cycle"],
            qc["active_vinyl_count"] == 4,
            qc["active_pair_count"] == 6,
            qc["new_bond_geometry"] == "cis",
            qc["all_internal_double_bonds_cis"],
        )
    ):
        qc["status"] = "ERROR"
        raise ValueError(f"q2 QC failed: {qc}")
    mapping_rows = [
        {
            "state": config.q2_state,
            "vinyl": f"V{vinyl}",
            "arm_zero_based": terminal_to_arm[vinyl],
            "vinyl_status": "active" if vinyl in output_mapping else "consumed",
            "q1_pdb_atoms": (
                f"{mapping[vinyl][0]};{mapping[vinyl][1]}" if vinyl in mapping else ""
            ),
            "q2_pdb_atoms": (
                f"{output_mapping[vinyl][0]};{output_mapping[vinyl][1]}"
                if vinyl in output_mapping
                else ""
            ),
        }
        for vinyl in range(1, 9)
    ]
    expected_dir = output_dir.parent / "results" / "react2" / "expected_schemas"
    expected_dir.mkdir(parents=True, exist_ok=True)
    header = active_pair_header(active, terminal_to_arm)
    (expected_dir / f"{config.q2_state}_trajectory_header.txt").write_text(
        ",".join(header) + "\n", encoding="utf-8"
    )
    (expected_dir / f"{config.q2_state}_active_vinyl_mapping.txt").write_text(
        "\n".join(
            f"V{vinyl}: ({output_mapping[vinyl][0]}, {output_mapping[vinyl][1]})"
            for vinyl in active
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "state": config.q2_state,
        "status": "READY_PILOT",
        "branch": config.branch,
        "source_pdb": str(q2_path.relative_to(root)),
        "first_reacted_pair": qc["first_reacted_pair"],
        "second_reacted_pair": qc["second_reacted_pair"],
        "active_vinyls": ";".join(f"V{vinyl}" for vinyl in active),
        "active_pair_count": 6,
        "suggested_output_dir": f"two_reaction/{config.q2_state}_md",
        "pilot_cycles": 20,
        "target_cycles_if_needed": 100,
        "sampling_ps_per_cycle": 25,
        "mapping_file": f"results/react2/expected_schemas/{config.q2_state}_active_vinyl_mapping.txt",
        "header_file": f"results/react2/expected_schemas/{config.q2_state}_trajectory_header.txt",
    }
    return qc, mapping_rows, manifest


def write_report(output: Path, qc_rows: Sequence[dict[str, object]]) -> None:
    lines = [
        "# A4 q2 adjacent/opposite branch structures",
        "",
        "Eight cis q2 PDBs were constructed from the validated cis q1 structures: one adjacent and one opposite second-reaction state for each A4_m C5--C8 system.",
        "",
        "Each edit removes C2H4, adds one graph cycle, leaves four active vinyl groups (six monitored pairs), and contains two cis internal metathesis alkenes after ETKDG/MMFF94 pre-optimization.",
        "",
        "| state | branch | second pair | formula | new C=C (A) | dihedral | MMFF energy |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in qc_rows:
        lines.append(
            f"| {row['state']} | {row['branch']} | {row['second_reacted_pair']} | "
            f"{row['q2_formula']} | {float(row['new_bond_length_A']):.3f} | "
            f"{float(row['new_bond_dihedral_deg']):.1f} | {float(row['mmff_energy_kcal_mol']):.1f} |"
        )
    lines.extend(
        [
            "",
            "C6 pair identity is selected from the q1 PDB topology/static geometry because the saved C6 q1 trajectory atom serials cannot be reconciled to that PDB. Its adjacent/opposite topology labels remain valid.",
            "",
            "These MMFF structures are starting conformers only. Production trajectories must use the same force field, perturb-minimize-equilibrate protocol, temperature, constraints, and sampling schedule as q0/q1; mixing an unrelated MD engine would invalidate q2/q1 comparisons.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(root: Path, conformers: int = 8) -> dict[str, object]:
    output_dir = root / "react_2"
    result_dir = root / "results" / "react2"
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    qc_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for config in BRANCH_CONFIGS:
        qc, mapping, manifest = construct_one(root, output_dir, config, conformers)
        qc_rows.append(qc)
        mapping_rows.extend(mapping)
        manifest_rows.append(manifest)
    baseline.write_csv(result_dir / "structure_qc.csv", qc_rows)
    baseline.write_csv(result_dir / "vinyl_mapping.csv", mapping_rows)
    baseline.write_csv(result_dir / "md_run_manifest.csv", manifest_rows)
    write_report(result_dir, qc_rows)
    return {
        "states": len(qc_rows),
        "passed": sum(row["status"] == "PASS" for row in qc_rows),
        "output": str(output_dir),
        "results": str(result_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--conformers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(Path(args.root).resolve(), conformers=args.conformers)
    print(
        f"Constructed {summary['passed']}/{summary['states']} validated q2 branch PDBs. "
        f"Structures: {summary['output']}; audit: {summary['results']}"
    )


if __name__ == "__main__":
    main()
