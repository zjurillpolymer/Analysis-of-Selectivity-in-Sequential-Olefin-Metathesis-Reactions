#!/usr/bin/env python3
"""Build and draw the exact B6 q2 V3-V6 + V11-V12 chemical graph for review."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen, rdDepictor, rdMolDescriptors

import build_q2_branch_structures as metathesis


STATE = "B6_p_C6_q2_V3_V6_V11_V12_cis"
FIRST_PAIR = (3, 6)
SECOND_PAIR = (11, 12)
FIRST_BOND_Q1_SERIALS = (191, 196)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    result.add_argument("--output", default="b6_q2_review/V3_V6__V11_V12")
    return result


def read_q1(root: Path) -> tuple[Chem.Mol, dict[int, tuple[int, int]], Path]:
    bundle = root / "gromacs_dcm_server_bundle"
    manifest = bundle / "gromacs_dcm/state_manifest.csv"
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        row = next(
            item for item in csv.DictReader(handle)
            if item["state"] == "B6_p_C6_q1_V3_V6_cis"
        )
    mapping = {
        int(label.removeprefix("V")): tuple(int(value) for value in serials)
        for label, serials in json.loads(row["vinyl_mapping_json"]).items()
    }
    source = bundle / row["source_pdb"]
    return metathesis.load_rdkit_pdb(source), mapping, source


def source_index(molecule: Chem.Mol, serial: int) -> int:
    matches = [
        atom.GetIdx() for atom in molecule.GetAtoms()
        if atom.HasProp("source_q1_serial")
        and atom.GetIntProp("source_q1_serial") == serial
    ]
    if len(matches) != 1:
        raise ValueError(f"q1 source serial {serial} maps to {matches}")
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


def bond_by_serials(molecule: Chem.Mol, serials: tuple[int, int]) -> Chem.Bond:
    indices = tuple(source_index(molecule, serial) for serial in serials)
    bond = molecule.GetBondBetweenAtoms(*indices)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError(f"expected double bond between q1 serials {serials}")
    return bond


def depict(
    product: Chem.Mol,
    active_mapping: dict[int, tuple[int, int]],
    terminal_retained: dict[int, tuple[int, int]],
    second_bond_serials: tuple[int, int],
    png: Path,
    svg: Path,
) -> None:
    molecule = Chem.RemoveHs(Chem.Mol(product))
    rdCoordGen.AddCoords(molecule)
    active = sorted(set(active_mapping) - set(SECOND_PAIR))
    active_atoms: set[int] = set()
    active_bonds: set[int] = set()
    for vinyl in active:
        indices = [source_index(molecule, serial) for serial in active_mapping[vinyl]]
        active_atoms.update(indices)
        bond = molecule.GetBondBetweenAtoms(*indices)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError(f"active V{vinyl} is not a double bond")
        active_bonds.add(bond.GetIdx())
        terminal = source_index(molecule, terminal_retained[vinyl][0])
        molecule.GetAtomWithIdx(terminal).SetProp("atomNote", f"V{vinyl}")

    formed = {
        "V3*/V6*": FIRST_BOND_Q1_SERIALS,
        "V11*/V12*": second_bond_serials,
    }
    formed_atoms: set[int] = set()
    formed_bonds: set[int] = set()
    for label, serials in formed.items():
        indices = [source_index(molecule, serial) for serial in serials]
        bond = molecule.GetBondBetweenAtoms(*indices)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError(f"formed {label} double bond is absent")
        formed_atoms.update(indices)
        formed_bonds.add(bond.GetIdx())
        for atom_label, index in zip(label.split("/"), indices):
            molecule.GetAtomWithIdx(index).SetProp("atomNote", atom_label)

    active_color = (0.20, 0.45, 0.90)
    formed_color = (0.90, 0.15, 0.12)
    atom_colors = {index: active_color for index in active_atoms}
    atom_colors.update({index: formed_color for index in formed_atoms})
    bond_colors = {index: active_color for index in active_bonds}
    bond_colors.update({index: formed_color for index in formed_bonds})
    legend = (
        "B6_p_C6 q2 candidate: V3-V6 + V11-V12 cis metathesis\n"
        "red = two formed internal C=C; blue = eight unreacted terminal vinyl groups"
    )
    for target, width, height in ((png, 2800, 2300), (svg, 2800, 2300)):
        drawer = (
            Draw.MolDraw2DCairo(width, height)
            if target.suffix == ".png"
            else Draw.MolDraw2DSVG(width, height)
        )
        options = drawer.drawOptions()
        options.addStereoAnnotation = True
        options.bondLineWidth = 2.0
        options.legendFontSize = 32
        options.annotationFontScale = 0.78
        options.padding = 0.07
        drawer.DrawMolecule(
            molecule,
            legend=legend,
            highlightAtoms=sorted(active_atoms | formed_atoms),
            highlightBonds=sorted(active_bonds | formed_bonds),
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
    q1, mapping, source = read_q1(root)
    if set(SECOND_PAIR) - set(mapping):
        raise ValueError("V11/V12 are not both active in the q1 manifest")
    terminal_retained = terminal_retained_serials(q1, mapping)
    product = metathesis.edit_second_metathesis(q1, mapping, SECOND_PAIR)
    second_bond_serials = tuple(terminal_retained[vinyl][1] for vinyl in SECOND_PAIR)

    first_bond = bond_by_serials(product, FIRST_BOND_Q1_SERIALS)
    second_bond = bond_by_serials(product, second_bond_serials)
    if first_bond.GetStereo() != Chem.BondStereo.STEREOZ:
        raise ValueError("the inherited V3-V6 double bond is not Z/cis")
    if second_bond.GetStereo() != Chem.BondStereo.STEREOZ:
        raise ValueError("the new V11-V12 double bond is not Z/cis")
    delta = element_counts(q1) - element_counts(product)
    if delta != Counter({"C": 2, "H": 4}):
        raise ValueError(f"unexpected q1-to-q2 formula delta: {delta}")
    if product.GetNumAtoms() != 432:
        raise ValueError(f"expected 432 q2 atoms, found {product.GetNumAtoms()}")
    cycle_q1 = q1.GetNumBonds() - q1.GetNumAtoms() + 1
    cycle_q2 = product.GetNumBonds() - product.GetNumAtoms() + 1
    if cycle_q2 != cycle_q1 + 1:
        raise ValueError("q2 edit did not add exactly one graph cycle")

    png = output / f"{STATE}_review.png"
    svg = output / f"{STATE}_review.svg"
    depict(product, mapping, terminal_retained, second_bond_serials, png, svg)
    depicted = Chem.Mol(product)
    rdDepictor.Compute2DCoords(depicted)
    sdf = output / f"{STATE}_review_only.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(depicted)
    writer.close()
    audit = {
        "status": "PASS",
        "purpose": "2D chemical-graph review only; not an MD-ready 3D conformer",
        "state": STATE,
        "source_q1": str(source.relative_to(root)),
        "first_reaction": ["V3", "V6"],
        "second_reaction": ["V11", "V12"],
        "first_internal_double_bond_q1_serials": list(FIRST_BOND_Q1_SERIALS),
        "second_internal_double_bond_q1_serials": list(second_bond_serials),
        "both_internal_double_bonds": "Z/cis",
        "q1_formula": rdMolDescriptors.CalcMolFormula(q1),
        "q2_formula": rdMolDescriptors.CalcMolFormula(product),
        "q1_to_q2_formula_delta": {"C": 2, "H": 4},
        "q1_atom_count": q1.GetNumAtoms(),
        "q2_atom_count": product.GetNumAtoms(),
        "adds_one_graph_cycle": True,
        "active_vinyls": [
            f"V{vinyl}" for vinyl in sorted(set(mapping) - set(SECOND_PAIR))
        ],
        "active_vinyl_count": 8,
        "active_pair_count": 28,
        "png": png.name,
        "svg": svg.name,
        "sdf": sdf.name,
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
