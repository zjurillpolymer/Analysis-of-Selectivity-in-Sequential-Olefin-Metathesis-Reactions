#!/usr/bin/env python3
"""Build and draw the two B6 q3 chemical graphs selected from q2 MD.

This is a review-only step.  It applies an exact third metathesis edit to the
validated q2 graph, checks C2H4 loss, Z/cis stereochemistry, atom mapping and
cycle count, then writes large annotated PNG/SVG drawings and SDF files.  It
does not prepare AmberTools inputs or submit remote calculations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen, rdDepictor, rdMolDescriptors

import build_q2_branch_structures as metathesis


Q2_STATE = "B6_p_C6_q2_V3_V6_V11_V12_cis"
INHERITED_BONDS = {
    "V3*/V6*": (191, 196),
    "V11*/V12*": (205, 206),
}


@dataclass(frozen=True)
class Candidate:
    label: str
    pair: tuple[int, int]
    role: str
    selection_basis: str

    @property
    def state(self) -> str:
        left, right = self.pair
        return f"B6_p_C6_q3_V3_V6_V11_V12_V{left}_V{right}_cis"


CANDIDATES = (
    Candidate(
        "V8_V10",
        (8, 10),
        "primary",
        "highest q2 consensus score; top-5 in all replicas and 14/15 blocks",
    ),
    Candidate(
        "V7_V9",
        (7, 9),
        "sensitivity",
        "top-5 in all replicas and 11/15 blocks; longest observed contact event",
    ),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    result.add_argument("--output", default="b6_q3_review")
    return result


def read_q2(root: Path) -> tuple[Chem.Mol, dict[int, tuple[int, int]], Path]:
    bundle = root / "gromacs_dcm_server_bundle"
    manifest = bundle / "gromacs_dcm/state_manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        row = next(item for item in csv.DictReader(handle) if item["state"] == Q2_STATE)
    mapping = {
        int(label.removeprefix("V")): tuple(int(value) for value in serials)
        for label, serials in json.loads(row["vinyl_mapping_json"]).items()
    }
    source = bundle / row["source_pdb"]
    return metathesis.load_rdkit_pdb(source), mapping, source


def source_index(molecule: Chem.Mol, serial: int) -> int:
    return metathesis.atom_index_by_source_serial(molecule, serial)


def serial_for_index(molecule: Chem.Mol, index: int) -> int:
    return molecule.GetAtomWithIdx(index).GetIntProp("source_q1_serial")


def terminal_retained_serials(
    molecule: Chem.Mol, mapping: dict[int, tuple[int, int]]
) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for vinyl, serials in mapping.items():
        terminal, retained = metathesis.terminal_and_retained_indices(molecule, serials)
        result[vinyl] = (
            serial_for_index(molecule, terminal),
            serial_for_index(molecule, retained),
        )
    return result


def element_counts(molecule: Chem.Mol) -> Counter[str]:
    return Counter(atom.GetSymbol() for atom in molecule.GetAtoms())


def checked_double_bond(
    molecule: Chem.Mol, serials: tuple[int, int], label: str
) -> Chem.Bond:
    indices = tuple(source_index(molecule, serial) for serial in serials)
    bond = molecule.GetBondBetweenAtoms(*indices)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError(f"{label} double bond is absent")
    if bond.GetStereo() != Chem.BondStereo.STEREOZ:
        raise ValueError(f"{label} double bond is not Z/cis: {bond.GetStereo()}")
    return bond


def output_active_mapping(
    product: Chem.Mol,
    mapping: dict[int, tuple[int, int]],
    consumed: tuple[int, int],
) -> dict[int, tuple[int, int]]:
    source_to_output = {
        atom.GetIntProp("source_q1_serial"): atom.GetIdx() + 1
        for atom in product.GetAtoms()
    }
    return {
        vinyl: tuple(source_to_output[serial] for serial in mapping[vinyl])
        for vinyl in sorted(set(mapping) - set(consumed))
    }


def depict(
    product: Chem.Mol,
    candidate: Candidate,
    mapping: dict[int, tuple[int, int]],
    terminal_retained: dict[int, tuple[int, int]],
    new_bond_serials: tuple[int, int],
    png: Path,
    svg: Path,
) -> None:
    molecule = Chem.RemoveHs(Chem.Mol(product))
    rdCoordGen.AddCoords(molecule)

    active = sorted(set(mapping) - set(candidate.pair))
    active_atoms: set[int] = set()
    active_bonds: set[int] = set()
    for vinyl in active:
        indices = [source_index(molecule, serial) for serial in mapping[vinyl]]
        active_atoms.update(indices)
        bond = molecule.GetBondBetweenAtoms(*indices)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError(f"active V{vinyl} is not a double bond")
        active_bonds.add(bond.GetIdx())
        terminal = source_index(molecule, terminal_retained[vinyl][0])
        molecule.GetAtomWithIdx(terminal).SetProp("atomNote", f"V{vinyl}")

    inherited_atoms: set[int] = set()
    inherited_bond_indices: set[int] = set()
    for label, serials in INHERITED_BONDS.items():
        indices = [source_index(molecule, serial) for serial in serials]
        bond = molecule.GetBondBetweenAtoms(*indices)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError(f"inherited {label} double bond is absent")
        inherited_atoms.update(indices)
        inherited_bond_indices.add(bond.GetIdx())
        for atom_label, index in zip(label.split("/"), indices):
            molecule.GetAtomWithIdx(index).SetProp("atomNote", atom_label)

    new_indices = [source_index(molecule, serial) for serial in new_bond_serials]
    new_bond = molecule.GetBondBetweenAtoms(*new_indices)
    if new_bond is None or new_bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError("new q3 double bond is absent")
    for atom_label, index in zip(
        (f"V{candidate.pair[0]}*", f"V{candidate.pair[1]}*"), new_indices
    ):
        molecule.GetAtomWithIdx(index).SetProp("atomNote", atom_label)

    active_color = (0.18, 0.43, 0.88)
    inherited_color = (0.88, 0.18, 0.14)
    new_color = (0.58, 0.12, 0.78)
    atom_colors = {index: active_color for index in active_atoms}
    atom_colors.update({index: inherited_color for index in inherited_atoms})
    atom_colors.update({index: new_color for index in new_indices})
    bond_colors = {index: active_color for index in active_bonds}
    bond_colors.update({index: inherited_color for index in inherited_bond_indices})
    bond_colors[new_bond.GetIdx()] = new_color
    highlighted_atoms = active_atoms | inherited_atoms | set(new_indices)
    highlighted_bonds = active_bonds | inherited_bond_indices | {new_bond.GetIdx()}
    legend = (
        f"B6_p_C6 q3 {candidate.role}: third metathesis V{candidate.pair[0]}-V{candidate.pair[1]} (Z/cis)\n"
        "purple = new q3 C=C; red = inherited q1/q2 C=C; blue = six unreacted terminal vinyls"
    )
    for target, width, height in ((png, 3200, 2850), (svg, 3200, 2850)):
        drawer = (
            Draw.MolDraw2DCairo(width, height)
            if target.suffix == ".png"
            else Draw.MolDraw2DSVG(width, height)
        )
        options = drawer.drawOptions()
        options.addStereoAnnotation = True
        options.bondLineWidth = 2.0
        options.legendFontSize = 31
        options.annotationFontScale = 0.76
        # Atom notes on this large branched macrocycle extend beyond the graph
        # bounding box, so use generous padding to keep V-labels visible.
        options.padding = 0.19
        drawer.DrawMolecule(
            molecule,
            legend=legend,
            highlightAtoms=sorted(highlighted_atoms),
            highlightBonds=sorted(highlighted_bonds),
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        drawing = drawer.GetDrawingText()
        if target.suffix == ".png":
            target.write_bytes(drawing)
        else:
            target.write_text(drawing, encoding="utf-8")


def build_candidate(
    root: Path,
    output: Path,
    q2: Chem.Mol,
    mapping: dict[int, tuple[int, int]],
    source: Path,
    candidate: Candidate,
) -> dict[str, object]:
    branch_dir = output / candidate.label
    branch_dir.mkdir(parents=True, exist_ok=True)
    terminal_retained = terminal_retained_serials(q2, mapping)
    product = metathesis.edit_second_metathesis(q2, mapping, candidate.pair)
    new_bond_serials = tuple(terminal_retained[vinyl][1] for vinyl in candidate.pair)

    for label, serials in INHERITED_BONDS.items():
        checked_double_bond(product, serials, label)
    checked_double_bond(product, new_bond_serials, "new q3")
    delta = element_counts(q2) - element_counts(product)
    if delta != Counter({"C": 2, "H": 4}):
        raise ValueError(f"unexpected q2-to-q3 formula delta: {delta}")
    if product.GetNumAtoms() != 426:
        raise ValueError(f"expected 426 q3 atoms, found {product.GetNumAtoms()}")
    cycle_q2 = q2.GetNumBonds() - q2.GetNumAtoms() + 1
    cycle_q3 = product.GetNumBonds() - product.GetNumAtoms() + 1
    if cycle_q3 != cycle_q2 + 1:
        raise ValueError("q3 edit did not add exactly one graph cycle")

    png = branch_dir / f"{candidate.state}_review.png"
    svg = branch_dir / f"{candidate.state}_review.svg"
    sdf = branch_dir / f"{candidate.state}_review_only.sdf"
    depict(
        product,
        candidate,
        mapping,
        terminal_retained,
        new_bond_serials,
        png,
        svg,
    )
    depicted = Chem.Mol(product)
    rdDepictor.Compute2DCoords(depicted)
    writer = Chem.SDWriter(str(sdf))
    writer.write(depicted)
    writer.close()

    active_mapping = output_active_mapping(product, mapping, candidate.pair)
    audit = {
        "status": "PASS",
        "purpose": "2D chemical-graph review only; not an MD-ready 3D conformer",
        "state": candidate.state,
        "role": candidate.role,
        "selection_basis": candidate.selection_basis,
        "source_q2": str(source.relative_to(root)),
        "reaction_history": [["V3", "V6"], ["V11", "V12"], [f"V{candidate.pair[0]}", f"V{candidate.pair[1]}"]],
        "new_internal_double_bond_q2_serials": list(new_bond_serials),
        "all_three_internal_double_bonds": "Z/cis",
        "q2_formula": rdMolDescriptors.CalcMolFormula(q2),
        "q3_formula": rdMolDescriptors.CalcMolFormula(product),
        "q2_to_q3_formula_delta": {"C": 2, "H": 4},
        "q2_atom_count": q2.GetNumAtoms(),
        "q3_atom_count": product.GetNumAtoms(),
        "adds_one_graph_cycle": True,
        "active_vinyls": [f"V{vinyl}" for vinyl in active_mapping],
        "active_vinyl_count": len(active_mapping),
        "active_pair_count": len(active_mapping) * (len(active_mapping) - 1) // 2,
        "active_vinyl_mapping_q3_sdf_1based": {
            f"V{vinyl}": list(serials) for vinyl, serials in active_mapping.items()
        },
        "png": png.name,
        "svg": svg.name,
        "sdf": sdf.name,
    }
    (branch_dir / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    args = parser().parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    q2, mapping, source = read_q2(root)
    if set(mapping) != {1, 2, 4, 5, 7, 8, 9, 10}:
        raise ValueError(f"unexpected q2 active-vinyl mapping: {mapping}")
    if rdMolDescriptors.CalcMolFormula(q2) != "C182H226O24":
        raise ValueError(f"unexpected q2 formula: {rdMolDescriptors.CalcMolFormula(q2)}")

    audits = [
        build_candidate(root, output, q2, mapping, source, candidate)
        for candidate in CANDIDATES
    ]
    summary = {
        "status": "PASS" if all(audit["status"] == "PASS" for audit in audits) else "FAIL",
        "source_state": Q2_STATE,
        "candidate_count": len(audits),
        "candidates": audits,
        "next_step": "visual user review before any AmberTools parameterization",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
