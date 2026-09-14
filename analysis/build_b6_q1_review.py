#!/usr/bin/env python3
"""Build an exact 2D review drawing for a candidate B6 q1 metathesis product."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen, rdDepictor, rdMolDescriptors

import build_q2_branch_structures as metathesis


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    result.add_argument("--left", type=int, default=3)
    result.add_argument("--right", type=int, default=6)
    result.add_argument("--output", default="b6_q1_review/V3_V6")
    return result


def read_mapping(root: Path) -> dict[int, tuple[int, int]]:
    manifest = root / "gromacs_dcm_server_bundle/gromacs_dcm/state_manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        row = next(item for item in csv.DictReader(handle) if item["state"] == "B6_p_C6_q0")
    return {
        int(label.removeprefix("V")): tuple(int(value) for value in serials)
        for label, serials in json.loads(row["vinyl_mapping_json"]).items()
    }


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


def formula_delta(initial: Chem.Mol, product: Chem.Mol) -> dict[str, int]:
    delta = element_counts(initial) - element_counts(product)
    return {element: int(count) for element, count in sorted(delta.items())}


def depict(
    product: Chem.Mol,
    mapping: dict[int, tuple[int, int]],
    terminal_retained: dict[int, tuple[int, int]],
    consumed: tuple[int, int],
    png: Path,
    svg: Path,
) -> None:
    molecule = Chem.RemoveHs(Chem.Mol(product))
    rdCoordGen.AddCoords(molecule)

    active = sorted(set(mapping) - set(consumed))
    active_atoms: set[int] = set()
    active_bonds: set[int] = set()
    for vinyl in active:
        indices = [source_index(molecule, serial) for serial in mapping[vinyl]]
        active_atoms.update(indices)
        bond = molecule.GetBondBetweenAtoms(*indices)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError(f"active V{vinyl} is not a double bond in the product")
        active_bonds.add(bond.GetIdx())
        terminal_index = source_index(molecule, terminal_retained[vinyl][0])
        molecule.GetAtomWithIdx(terminal_index).SetProp("atomNote", f"V{vinyl}")

    new_indices = [
        source_index(molecule, terminal_retained[vinyl][1]) for vinyl in consumed
    ]
    new_bond = molecule.GetBondBetweenAtoms(*new_indices)
    if new_bond is None or new_bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError("candidate product does not contain the expected new C=C")
    for vinyl, index in zip(consumed, new_indices):
        molecule.GetAtomWithIdx(index).SetProp("atomNote", f"V{vinyl}*")

    active_color = (0.20, 0.45, 0.90)
    new_color = (0.90, 0.15, 0.12)
    highlighted_atoms = sorted(active_atoms | set(new_indices))
    highlighted_bonds = sorted(active_bonds | {new_bond.GetIdx()})
    atom_colors = {index: active_color for index in active_atoms}
    atom_colors.update({index: new_color for index in new_indices})
    bond_colors = {index: active_color for index in active_bonds}
    bond_colors[new_bond.GetIdx()] = new_color

    legend = (
        f"B6_p_C6 q1 candidate: V{consumed[0]}-V{consumed[1]} cis metathesis\n"
        "red = new internal C=C; blue = ten unreacted terminal vinyl groups"
    )
    for target, width, height in ((png, 2600, 2200), (svg, 2600, 2200)):
        if target.suffix == ".png":
            drawer = Draw.MolDraw2DCairo(width, height)
        else:
            drawer = Draw.MolDraw2DSVG(width, height)
        options = drawer.drawOptions()
        options.addStereoAnnotation = True
        options.bondLineWidth = 2.0
        options.legendFontSize = 32
        options.annotationFontScale = 0.8
        options.padding = 0.08
        drawer.DrawMolecule(
            molecule,
            legend=legend,
            highlightAtoms=highlighted_atoms,
            highlightBonds=highlighted_bonds,
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        drawing = drawer.GetDrawingText()
        if target.suffix == ".png":
            target.write_bytes(drawing)
        else:
            target.write_text(drawing, encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    consumed = tuple(sorted((args.left, args.right)))
    if consumed[0] == consumed[1] or not all(1 <= value <= 12 for value in consumed):
        raise ValueError("consumed vinyls must be two distinct values from 1 through 12")

    source = root / "gromacs_dcm_server_bundle/structures/B6_p_C6.pdb"
    initial = metathesis.load_rdkit_pdb(source)
    mapping = read_mapping(root)
    terminal_retained = terminal_retained_serials(initial, mapping)
    product = metathesis.edit_second_metathesis(initial, mapping, consumed)

    delta = formula_delta(initial, product)
    if delta != {"C": 2, "H": 4}:
        raise ValueError(f"unexpected reaction formula delta: {delta}")
    new_retained_serials = [terminal_retained[vinyl][1] for vinyl in consumed]
    new_indices = [source_index(product, serial) for serial in new_retained_serials]
    new_bond = product.GetBondBetweenAtoms(*new_indices)
    if new_bond is None or new_bond.GetStereo() != Chem.BondStereo.STEREOZ:
        raise ValueError("new metathesis double bond is not tagged cis/Z")

    png = output / f"B6_p_C6_q1_V{consumed[0]}_V{consumed[1]}_review.png"
    svg = output / f"B6_p_C6_q1_V{consumed[0]}_V{consumed[1]}_review.svg"
    depict(product, mapping, terminal_retained, consumed, png, svg)

    sdf_product = Chem.Mol(product)
    rdDepictor.Compute2DCoords(sdf_product)
    sdf = output / f"B6_p_C6_q1_V{consumed[0]}_V{consumed[1]}_review_only.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(sdf_product)
    writer.close()

    audit = {
        "status": "PASS",
        "purpose": "2D chemical-graph review only; not an MD-ready 3D conformer",
        "source": str(source.relative_to(root)),
        "consumed_vinyls": [f"V{value}" for value in consumed],
        "deleted_terminal_carbons_source_serial": [
            terminal_retained[vinyl][0] for vinyl in consumed
        ],
        "retained_carbons_new_double_bond_source_serial": new_retained_serials,
        "new_double_bond_stereo": "Z/cis",
        "initial_formula": rdMolDescriptors.CalcMolFormula(initial),
        "product_formula": rdMolDescriptors.CalcMolFormula(product),
        "formula_delta": delta,
        "initial_atom_count": initial.GetNumAtoms(),
        "product_atom_count": product.GetNumAtoms(),
        "initial_bond_count": initial.GetNumBonds(),
        "product_bond_count": product.GetNumBonds(),
        "active_vinyls": [f"V{value}" for value in sorted(set(mapping) - set(consumed))],
        "png": png.name,
        "svg": svg.name,
        "sdf": sdf.name,
    }
    audit_path = output / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
