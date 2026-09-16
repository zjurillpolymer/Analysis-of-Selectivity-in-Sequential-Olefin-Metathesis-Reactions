#!/usr/bin/env python3
"""Build exact 2D RDKit review structures for missing A4 q2 histories.

For A4-m-C5 and A4-m-C8, construct one q2-reuse product and one representative
q2-same-arm product from the validated q1 cis structure.  The same-arm pair is
selected by the largest mean q1 soft-NAC value among the two available
same-arm pairs.  Outputs are chemical-graph review drawings and 2D SDF files;
they are not MD-ready three-dimensional conformers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen, rdDepictor, rdMolDescriptors

import analyze_react1_structures as react1
import build_q2_branch_structures as metathesis


PAIR_RE = re.compile(r"^V(\d+)_V(\d+)_")
PRIMARY_SETTING = "d5_axis45_side45"


@dataclass(frozen=True)
class Candidate:
    system: str
    history: str
    first_pair: tuple[int, int]
    second_pair: tuple[int, int]
    arm_by_vinyl: Mapping[int, int]
    selection_basis: str
    same_arm_alternatives: tuple[tuple[int, int, float], ...] = ()

    @property
    def q1_state(self) -> str:
        return f"{self.system}_1_cis"

    @property
    def q2_state(self) -> str:
        return f"{self.system}_2_{self.history}_cis_review"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_index(molecule: Chem.Mol, serial: int) -> int:
    matches = [
        atom.GetIdx()
        for atom in molecule.GetAtoms()
        if atom.HasProp("source_q1_serial")
        and atom.GetIntProp("source_q1_serial") == serial
    ]
    if len(matches) != 1:
        raise ValueError(f"q1 serial {serial} maps to {matches}")
    return matches[0]


def serial_for_index(molecule: Chem.Mol, index: int) -> int:
    return molecule.GetAtomWithIdx(index).GetIntProp("source_q1_serial")


def terminal_retained_serials(
    molecule: Chem.Mol, mapping: Mapping[int, tuple[int, int]]
) -> dict[int, tuple[int, int]]:
    output: dict[int, tuple[int, int]] = {}
    for vinyl, serials in mapping.items():
        terminal, retained = metathesis.terminal_and_retained_indices(molecule, serials)
        output[vinyl] = (
            serial_for_index(molecule, terminal),
            serial_for_index(molecule, retained),
        )
    return output


def first_pair(root: Path, state: str) -> tuple[int, int]:
    row = next(
        item
        for item in read_csv(root / "results/react1/structure_qc.csv")
        if item["state"] == state
    )
    return tuple(sorted(int(value.removeprefix("V")) for value in row["reacted_pair"].split("-")))


def arm_mapping(root: Path, state: str) -> dict[int, int]:
    rows = [
        row
        for row in read_csv(root / "results/react1/vinyl_mapping.csv")
        if row["state"] == state
    ]
    mapping = {int(row["vinyl"].removeprefix("V")): int(row["arm_zero_based"]) for row in rows}
    if set(mapping) != set(range(1, 9)):
        raise ValueError(f"incomplete arm mapping for {state}: {mapping}")
    return mapping


def same_pair_nac_means(
    root: Path,
    system: str,
    statistics_path: Path | None = None,
) -> dict[tuple[int, int], float]:
    path = statistics_path or root / (
        "results/a4_postreaction_nac_analysis/b6-gromacs-parallel-qpd79/"
        "pair_nac_statistics.csv"
    )
    buckets: dict[tuple[int, int], list[float]] = {}
    for row in read_csv(path):
        if (
            row["system"] != system
            or row["stage"] != "q1"
            or row["setting_id"] != PRIMARY_SETTING
            or row["route_family"] != "same_arm"
        ):
            continue
        match = PAIR_RE.match(row["pair"])
        if match is None:
            raise ValueError(f"cannot parse pair label: {row['pair']}")
        pair = tuple(sorted((int(match.group(1)), int(match.group(2)))))
        buckets.setdefault(pair, []).append(float(row["mean_soft_nac"]))
    return {pair: sum(values) / len(values) for pair, values in buckets.items()}


def select_candidates(
    root: Path,
    system: str,
    statistics_path: Path | None = None,
) -> tuple[Candidate, Candidate]:
    state = f"{system}_1_cis"
    active_mapping = metathesis.q1_mapping(root, state)
    active = sorted(active_mapping)
    arms = arm_mapping(root, state)
    first = first_pair(root, state)
    first_arm_pair = tuple(sorted((arms[first[0]], arms[first[1]])))
    reuse = [
        pair for pair in combinations(active, 2)
        if arms[pair[0]] != arms[pair[1]]
        and tuple(sorted((arms[pair[0]], arms[pair[1]]))) == first_arm_pair
    ]
    if len(reuse) != 1:
        raise ValueError(f"expected one reuse candidate in {state}, found {reuse}")

    same = [pair for pair in combinations(active, 2) if arms[pair[0]] == arms[pair[1]]]
    means = same_pair_nac_means(root, system, statistics_path)
    if set(same) != set(means):
        raise ValueError(f"same-arm NAC inventory mismatch for {state}: pairs={same}, means={means}")
    ranked = tuple(sorted(
        ((left, right, means[(left, right)]) for left, right in same),
        key=lambda item: (-item[2], item[0], item[1]),
    ))
    selected_same = ranked[0][:2]
    return (
        Candidate(
            system=system,
            history="reuse",
            first_pair=first,
            second_pair=reuse[0],
            arm_by_vinyl=arms,
            selection_basis="unique active pair that repeats the first reacted arm pair",
        ),
        Candidate(
            system=system,
            history="same_arm",
            first_pair=first,
            second_pair=selected_same,
            arm_by_vinyl=arms,
            selection_basis=f"largest mean q1 soft-NAC among same-arm pairs ({PRIMARY_SETTING})",
            same_arm_alternatives=ranked,
        ),
    )


def formula_delta(initial: Chem.Mol, product: Chem.Mol) -> dict[str, int]:
    initial_counts = Counter(atom.GetSymbol() for atom in initial.GetAtoms())
    product_counts = Counter(atom.GetSymbol() for atom in product.GetAtoms())
    return {element: int(count) for element, count in sorted((initial_counts - product_counts).items())}


def active_vinyl_bonds(
    molecule: Chem.Mol,
    mapping: Mapping[int, tuple[int, int]],
    consumed: Sequence[int],
) -> dict[int, int]:
    output: dict[int, int] = {}
    for vinyl in sorted(set(mapping) - set(consumed)):
        indices = tuple(source_index(molecule, serial) for serial in mapping[vinyl])
        bond = molecule.GetBondBetweenAtoms(*indices)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError(f"V{vinyl} is not an active C=C after the edit")
        output[vinyl] = bond.GetIdx()
    if len(output) != 4:
        raise ValueError(f"expected four active vinyls, found {output}")
    return output


def internal_double_bonds(molecule: Chem.Mol) -> list[int]:
    output = [
        bond.GetIdx()
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}
    ]
    if len(output) != 2:
        raise ValueError(f"expected two cis/Z internal C=C bonds, found {output}")
    return output


def draw_candidate(
    product: Chem.Mol,
    mapping: Mapping[int, tuple[int, int]],
    terminal_retained: Mapping[int, tuple[int, int]],
    candidate: Candidate,
    png: Path,
    svg: Path,
) -> dict[str, object]:
    molecule = Chem.RemoveHs(Chem.Mol(product))
    rdCoordGen.AddCoords(molecule)
    active = active_vinyl_bonds(molecule, mapping, candidate.second_pair)
    second_indices = [
        source_index(molecule, terminal_retained[vinyl][1])
        for vinyl in candidate.second_pair
    ]
    second_bond = molecule.GetBondBetweenAtoms(*second_indices)
    if second_bond is None or second_bond.GetStereo() not in {
        Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS
    }:
        raise ValueError("second metathesis bond is not cis/Z")
    internal = internal_double_bonds(molecule)
    first_bonds = [index for index in internal if index != second_bond.GetIdx()]
    if len(first_bonds) != 1:
        raise ValueError(f"cannot identify first metathesis C=C: {internal}")
    first_bond = molecule.GetBondWithIdx(first_bonds[0])

    active_atoms: set[int] = set()
    for vinyl, bond_index in active.items():
        bond = molecule.GetBondWithIdx(bond_index)
        active_atoms.update((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        terminal_serial = terminal_retained[vinyl][0]
        terminal_index = source_index(molecule, terminal_serial)
        arm = candidate.arm_by_vinyl[vinyl]
        molecule.GetAtomWithIdx(terminal_index).SetProp("atomNote", f"V{vinyl}/A{arm}")
    for vinyl, index in zip(candidate.second_pair, second_indices):
        molecule.GetAtomWithIdx(index).SetProp("atomNote", f"V{vinyl}*/A{candidate.arm_by_vinyl[vinyl]}")
    molecule.GetAtomWithIdx(first_bond.GetBeginAtomIdx()).SetProp("atomNote", "site A")
    molecule.GetAtomWithIdx(second_indices[0]).SetProp(
        "atomNote", molecule.GetAtomWithIdx(second_indices[0]).GetProp("atomNote") + "/site B"
    )

    active_color = (0.18, 0.45, 0.85)
    first_color = (0.45, 0.30, 0.68)
    second_color = (0.88, 0.18, 0.12)
    highlighted_atoms = sorted(
        active_atoms
        | set(second_indices)
        | {first_bond.GetBeginAtomIdx(), first_bond.GetEndAtomIdx()}
    )
    highlighted_bonds = sorted(set(active.values()) | {first_bond.GetIdx(), second_bond.GetIdx()})
    atom_colors = {index: active_color for index in active_atoms}
    atom_colors.update({first_bond.GetBeginAtomIdx(): first_color, first_bond.GetEndAtomIdx(): first_color})
    atom_colors.update({index: second_color for index in second_indices})
    bond_colors = {index: active_color for index in active.values()}
    bond_colors[first_bond.GetIdx()] = first_color
    bond_colors[second_bond.GetIdx()] = second_color

    first_arms = tuple(candidate.arm_by_vinyl[value] for value in candidate.first_pair)
    second_arms = tuple(candidate.arm_by_vinyl[value] for value in candidate.second_pair)
    legend = (
        f"{candidate.system.replace('_', '-')} q2-{candidate.history}\n"
        f"V{candidate.first_pair[0]}-V{candidate.first_pair[1]} (A{first_arms[0]}-A{first_arms[1]})"
        f"  +  V{candidate.second_pair[0]}-V{candidate.second_pair[1]} "
        f"(A{second_arms[0]}-A{second_arms[1]})\n"
        "purple/red = the two reacted sites (order not assigned); blue = four active vinyls"
    )
    for target in (png, svg):
        drawer = Draw.MolDraw2DCairo(1500, 1120) if target.suffix == ".png" else Draw.MolDraw2DSVG(1500, 1120)
        options = drawer.drawOptions()
        options.addStereoAnnotation = True
        options.bondLineWidth = 2.1
        options.legendFontSize = 29
        options.annotationFontScale = 0.75
        options.padding = 0.06
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
    return {
        "active_vinyls": sorted(active),
        "first_internal_bond_index_1based": first_bond.GetIdx() + 1,
        "second_internal_bond_index_1based": second_bond.GetIdx() + 1,
    }


def build_one(root: Path, output: Path, candidate: Candidate) -> dict[str, object]:
    q1_path = root / "react_1" / f"{candidate.q1_state}.pdb"
    initial = metathesis.load_rdkit_pdb(q1_path)
    mapping = metathesis.q1_mapping(root, candidate.q1_state)
    terminal_retained = terminal_retained_serials(initial, mapping)
    product = metathesis.edit_second_metathesis(initial, mapping, candidate.second_pair)
    delta = formula_delta(initial, product)
    if delta != {"C": 2, "H": 4}:
        raise ValueError(f"unexpected formula delta for {candidate.q2_state}: {delta}")
    if rdMolDescriptors.CalcNumRings(product) != rdMolDescriptors.CalcNumRings(initial) + 1:
        raise ValueError(f"second edit does not add one ring for {candidate.q2_state}")

    png = output / f"{candidate.q2_state}.png"
    svg = output / f"{candidate.q2_state}.svg"
    drawing = draw_candidate(product, mapping, terminal_retained, candidate, png, svg)
    sdf_molecule = Chem.Mol(product)
    rdDepictor.Compute2DCoords(sdf_molecule)
    sdf = output / f"{candidate.q2_state}.sdf"
    writer = Chem.SDWriter(str(sdf))
    writer.write(sdf_molecule)
    writer.close()

    first_arms = tuple(candidate.arm_by_vinyl[value] for value in candidate.first_pair)
    second_arms = tuple(candidate.arm_by_vinyl[value] for value in candidate.second_pair)
    expected_history = (
        tuple(sorted(first_arms)) == tuple(sorted(second_arms))
        if candidate.history == "reuse"
        else second_arms[0] == second_arms[1]
    )
    if not expected_history:
        raise ValueError(f"history classification failed for {candidate.q2_state}")
    return {
        "state": candidate.q2_state,
        "system": candidate.system,
        "history": candidate.history,
        "status": "PASS",
        "purpose": "2D chemical-graph review only; not an MD-ready 3D conformer",
        "source_q1": str(q1_path.relative_to(root)),
        "first_pair": [f"V{value}" for value in candidate.first_pair],
        "first_arms": list(first_arms),
        "second_pair": [f"V{value}" for value in candidate.second_pair],
        "second_arms": list(second_arms),
        "selection_basis": candidate.selection_basis,
        "pair_order_has_physical_meaning": False,
        "same_arm_alternatives": [
            {"pair": [f"V{left}", f"V{right}"], "mean_soft_nac": mean}
            for left, right, mean in candidate.same_arm_alternatives
        ],
        "q1_formula": rdMolDescriptors.CalcMolFormula(initial),
        "q2_formula": rdMolDescriptors.CalcMolFormula(product),
        "formula_delta": delta,
        "q1_ring_count": rdMolDescriptors.CalcNumRings(initial),
        "q2_ring_count": rdMolDescriptors.CalcNumRings(product),
        "new_double_bond_stereo": "Z/cis",
        **drawing,
        "png": png.name,
        "svg": svg.name,
        "sdf": sdf.name,
    }


def composite(paths: Sequence[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if len(images) != 4 or len({image.size for image in images}) != 1:
        raise ValueError("expected four equal-sized review panels")
    width, height = images[0].size
    canvas = Image.new("RGB", (2 * width, 2 * height), "white")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 2) * width, (index // 2) * height))
    canvas.save(output, dpi=(300, 300))


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output", type=Path,
        default=root / "results/a4_q2_missing_rdkit_review",
    )
    parser.add_argument("--systems", nargs="+", default=["A4_m_C5", "A4_m_C8"])
    parser.add_argument(
        "--q1-statistics",
        type=Path,
        help="pair_nac_statistics.csv used to choose the same-arm q1 candidate",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = [
        candidate
        for system in args.systems
        for candidate in select_candidates(root, system, args.q1_statistics)
    ]
    audits = [build_one(root, output, candidate) for candidate in candidates]
    composite(
        [output / str(audit["png"]) for audit in audits],
        output / "a4_q2_missing_histories_review_grid.png",
    )
    (output / "audit.json").write_text(
        json.dumps({"status": "PASS", "candidates": audits}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "PASS",
        "states": [audit["state"] for audit in audits],
        "grid": str(output / "a4_q2_missing_histories_review_grid.png"),
    }, indent=2))


if __name__ == "__main__":
    main()
