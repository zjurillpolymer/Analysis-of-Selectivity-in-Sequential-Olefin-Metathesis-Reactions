#!/usr/bin/env python3
"""Build RDKit q1/q2 review structures for A4_p_C6 and A4_p_C8.

The construction follows the already-used A4 workflow:

* q1: consume one representative cross-arm vinyl pair;
* q2: from q1, consume one same-arm pair on a still-complete arm;
* every newly formed internal alkene is explicitly Z (cis).

The output is intended for chemical-graph review before Amber preparation or MD.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdCoordGen, rdMolDescriptors

import analyze_react1_structures as react1
import build_q2_branch_structures as metathesis
import mtmr_baseline as baseline


@dataclass(frozen=True)
class SystemPlan:
    system: str
    first_pair: tuple[int, int]
    second_pair: tuple[int, int]

    @property
    def q1_state(self) -> str:
        return f"{self.system}_1_cis"

    @property
    def q2_state(self) -> str:
        return f"{self.system}_2_mixed_same_cross_cis"


# Representative labels are selected from the existing q0 trajectories.  The
# first pair is the largest q0 mean soft-contact cross-arm pair; among the
# complete arms untouched by q1, the second pair has the largest q0 mean
# same-arm soft contact.  The labels remain subject to the user's graph review.
PLANS: tuple[SystemPlan, ...] = (
    SystemPlan("A4_p_C6", (6, 7), (1, 2)),
    SystemPlan("A4_p_C8", (2, 4), (3, 8)),
)


def formula_delta(initial: Chem.Mol, product: Chem.Mol) -> dict[str, int]:
    before = Counter(atom.GetSymbol() for atom in initial.GetAtoms())
    after = Counter(atom.GetSymbol() for atom in product.GetAtoms())
    return dict(sorted((before - after).items()))


def graph_cycle_rank(molecule: Chem.Mol) -> int:
    """Return the heavy-atom cyclomatic number E - V + components."""
    heavy = {atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1}
    edges = sum(
        bond.GetBeginAtomIdx() in heavy and bond.GetEndAtomIdx() in heavy
        for bond in molecule.GetBonds()
    )
    seen: set[int] = set()
    components = 0
    for start in heavy:
        if start in seen:
            continue
        components += 1
        seen.add(start)
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in molecule.GetAtomWithIdx(current).GetNeighbors():
                index = neighbor.GetIdx()
                if index in heavy and index not in seen:
                    seen.add(index)
                    stack.append(index)
    return int(edges) - len(heavy) + components


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


def terminal_retained_serials(
    molecule: Chem.Mol, mapping: Mapping[int, tuple[int, int]]
) -> dict[int, tuple[int, int]]:
    result: dict[int, tuple[int, int]] = {}
    for vinyl, serials in mapping.items():
        terminal, retained = metathesis.terminal_and_retained_indices(molecule, serials)
        result[vinyl] = (
            molecule.GetAtomWithIdx(terminal).GetIntProp("source_q1_serial"),
            molecule.GetAtomWithIdx(retained).GetIntProp("source_q1_serial"),
        )
    return result


def arm_map(root: Path, system: str) -> tuple[dict[int, int], baseline.SystemData]:
    data, _ = baseline.analyse_system(
        root, root / "trace" / f"{system}_md", baseline.DEFAULT_THRESHOLDS
    )
    if data is None:
        raise RuntimeError(f"cannot read q0 trajectory metadata for {system}")
    return (
        {vinyl: int(data.terminal_to_monomer[vinyl - 1]) + 1 for vinyl in range(1, 9)},
        data,
    )


def pair_soft_score(data: baseline.SystemData, pair: tuple[int, int]) -> float:
    edge = tuple(sorted((pair[0] - 1, pair[1] - 1)))
    index = data.pair_vertices.index(edge)
    return float(data.cycle_soft_q[:, index].mean())


def save_3d_product(molecule: Chem.Mol, path: Path) -> dict[int, int]:
    source_to_output = metathesis.renumber_pdb_metadata(molecule)
    Chem.MolToPDBFile(molecule, str(path), flavor=0)
    return source_to_output


def save_sdf(molecule: Chem.Mol, path: Path) -> None:
    writer = Chem.SDWriter(str(path))
    writer.write(molecule)
    writer.close()


def active_bonds(
    molecule: Chem.Mol,
    mapping: Mapping[int, tuple[int, int]],
    active: Sequence[int],
) -> dict[int, int]:
    result: dict[int, int] = {}
    for vinyl in active:
        indices = tuple(source_index(molecule, serial) for serial in mapping[vinyl])
        bond = molecule.GetBondBetweenAtoms(*indices)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise ValueError(f"V{vinyl} is not an active C=C")
        result[vinyl] = bond.GetIdx()
    return result


def cis_internal_bonds(molecule: Chem.Mol) -> list[int]:
    return [
        bond.GetIdx()
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetStereo() in {Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS}
    ]


def draw_state(
    molecule: Chem.Mol,
    mapping: Mapping[int, tuple[int, int]],
    active: Sequence[int],
    arms: Mapping[int, int],
    state: str,
    reacted_pairs: Sequence[tuple[int, int]],
    latest_retained_serials: tuple[int, int],
    output_png: Path,
    output_svg: Path,
) -> None:
    drawing_mol = Chem.RemoveHs(Chem.Mol(molecule))
    rdCoordGen.AddCoords(drawing_mol)
    active_by_vinyl = active_bonds(drawing_mol, mapping, active)
    internals = cis_internal_bonds(drawing_mol)
    if len(internals) != len(reacted_pairs):
        raise ValueError(
            f"{state}: expected {len(reacted_pairs)} cis internal bonds, found {internals}"
        )

    blue = (0.16, 0.43, 0.82)
    purple = (0.52, 0.27, 0.67)
    red = (0.88, 0.16, 0.10)
    reaction_colors = (purple, red)
    atom_colors: dict[int, tuple[float, float, float]] = {}
    bond_colors: dict[int, tuple[float, float, float]] = {}
    highlighted_atoms: set[int] = set()
    highlighted_bonds: set[int] = set()

    for vinyl, bond_index in active_by_vinyl.items():
        bond = drawing_mol.GetBondWithIdx(bond_index)
        indices = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        terminal = min(
            indices,
            key=lambda index: sum(
                neighbor.GetAtomicNum() > 1
                for neighbor in drawing_mol.GetAtomWithIdx(index).GetNeighbors()
            ),
        )
        drawing_mol.GetAtomWithIdx(terminal).SetProp(
            "atomNote", f"V{vinyl}/A{arms[vinyl]}"
        )
        highlighted_atoms.update(indices)
        highlighted_bonds.add(bond_index)
        atom_colors.update({index: blue for index in indices})
        bond_colors[bond_index] = blue

    # The most recently formed bond is located from the retained-carbon source
    # serials.  At q2 the other Z bond is therefore the q1 bond.
    latest_indices = tuple(
        source_index(drawing_mol, serial) for serial in latest_retained_serials
    )
    latest_bond = drawing_mol.GetBondBetweenAtoms(*latest_indices)
    if latest_bond is None or latest_bond.GetIdx() not in internals:
        raise ValueError(f"{state}: cannot locate the latest formed Z bond")
    if len(reacted_pairs) == 1:
        ordered_internal = [latest_bond.GetIdx()]
    else:
        ordered_internal = [
            next(index for index in internals if index != latest_bond.GetIdx()),
            latest_bond.GetIdx(),
        ]

    for order, bond_index in enumerate(ordered_internal):
        color = reaction_colors[order]
        bond = drawing_mol.GetBondWithIdx(bond_index)
        indices = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        drawing_mol.GetAtomWithIdx(indices[0]).SetProp("atomNote", f"q{order + 1}: Z")
        highlighted_atoms.update(indices)
        highlighted_bonds.add(bond_index)
        atom_colors.update({index: color for index in indices})
        bond_colors[bond_index] = color

    pair_text = "  +  ".join(
        f"q{number}: V{pair[0]}-V{pair[1]} (A{arms[pair[0]]}-A{arms[pair[1]]}, Z)"
        for number, pair in enumerate(reacted_pairs, 1)
    )
    legend = (
        f"{state.replace('_', '-')}\n{pair_text}\n"
        "purple/red = formed internal C=C; blue = remaining vinyl groups"
    )
    for target in (output_png, output_svg):
        drawer = (
            Draw.MolDraw2DCairo(1550, 1120)
            if target.suffix == ".png"
            else Draw.MolDraw2DSVG(1550, 1120)
        )
        options = drawer.drawOptions()
        options.addStereoAnnotation = True
        options.bondLineWidth = 2.2
        options.legendFontSize = 27
        options.annotationFontScale = 0.78
        options.padding = 0.055
        drawer.DrawMolecule(
            drawing_mol,
            legend=legend,
            highlightAtoms=sorted(highlighted_atoms),
            highlightBonds=sorted(highlighted_bonds),
            highlightAtomColors=atom_colors,
            highlightBondColors=bond_colors,
        )
        drawer.FinishDrawing()
        payload = drawer.GetDrawingText()
        if target.suffix == ".png":
            target.write_bytes(payload)
        else:
            target.write_text(payload, encoding="utf-8")


def validate_plan(arms: Mapping[int, int], plan: SystemPlan) -> None:
    if arms[plan.first_pair[0]] == arms[plan.first_pair[1]]:
        raise ValueError(f"{plan.system}: q1 pair must be cross-arm")
    if arms[plan.second_pair[0]] != arms[plan.second_pair[1]]:
        raise ValueError(f"{plan.system}: q2 pair must be same-arm")
    if set(plan.first_pair) & set(plan.second_pair):
        raise ValueError(f"{plan.system}: q2 pair must be on an untouched arm")


def build_system(root: Path, output: Path, plan: SystemPlan, conformers: int) -> dict[str, object]:
    arms, data = arm_map(root, plan.system)
    validate_plan(arms, plan)
    q0_path = root / "structures" / f"{plan.system}.pdb"
    q0 = metathesis.load_rdkit_pdb(q0_path)
    q0_mapping = react1.vinyl_mapping(
        root / "trace" / f"{plan.system}_md" / "md_simulation.log"
    )
    q0_terminal_retained = terminal_retained_serials(q0, q0_mapping)

    q1 = metathesis.edit_second_metathesis(q0, q0_mapping, plan.first_pair)
    q1, q1_energy, q1_status = metathesis.embed_and_optimize(
        q1, baseline.stable_seed(20260905, plan.q1_state) % (2**31 - 1), conformers
    )
    q1_geometry = metathesis.stereo_geometry_rows(q1)
    if formula_delta(q0, q1) != {"C": 2, "H": 4}:
        raise ValueError(f"{plan.q1_state}: formula delta is not C2H4")
    if graph_cycle_rank(q1) != graph_cycle_rank(q0) + 1:
        raise ValueError(f"{plan.q1_state}: graph cycle count did not increase by one")
    if len(q1_geometry) != 1 or q1_geometry[0]["geometry_classification"] != "cis":
        raise ValueError(f"{plan.q1_state}: cis geometry QC failed: {q1_geometry}")

    q1_path = output / f"{plan.q1_state}.pdb"
    q1_source_to_output = save_3d_product(q1, q1_path)
    save_sdf(q1, output / f"{plan.q1_state}.sdf")
    q1_active = sorted(set(q0_mapping) - set(plan.first_pair))
    q1_mapping = {
        vinyl: tuple(q1_source_to_output[serial] for serial in q0_mapping[vinyl])
        for vinyl in q1_active
    }
    draw_state(
        q1,
        q0_mapping,
        q1_active,
        arms,
        plan.q1_state,
        (plan.first_pair,),
        tuple(q0_terminal_retained[value][1] for value in plan.first_pair),
        output / f"{plan.q1_state}.png",
        output / f"{plan.q1_state}.svg",
    )

    q1_reloaded = metathesis.load_rdkit_pdb(q1_path)
    q1_terminal_retained = terminal_retained_serials(q1_reloaded, q1_mapping)
    q2 = metathesis.edit_second_metathesis(q1_reloaded, q1_mapping, plan.second_pair)
    q2, q2_energy, q2_status = metathesis.embed_and_optimize(
        q2, baseline.stable_seed(20260905, plan.q2_state) % (2**31 - 1), conformers
    )
    q2_geometry = metathesis.stereo_geometry_rows(q2)
    if formula_delta(q1_reloaded, q2) != {"C": 2, "H": 4}:
        raise ValueError(f"{plan.q2_state}: formula delta is not C2H4")
    if graph_cycle_rank(q2) != graph_cycle_rank(q1_reloaded) + 1:
        raise ValueError(f"{plan.q2_state}: graph cycle count did not increase by one")
    if len(q2_geometry) != 2 or any(
        row["geometry_classification"] != "cis" for row in q2_geometry
    ):
        raise ValueError(f"{plan.q2_state}: cis geometry QC failed: {q2_geometry}")

    q2_path = output / f"{plan.q2_state}.pdb"
    q2_source_to_output = save_3d_product(q2, q2_path)
    save_sdf(q2, output / f"{plan.q2_state}.sdf")
    q2_active = sorted(set(q1_mapping) - set(plan.second_pair))
    q2_mapping = {
        vinyl: tuple(q2_source_to_output[serial] for serial in q1_mapping[vinyl])
        for vinyl in q2_active
    }
    draw_state(
        q2,
        q1_mapping,
        q2_active,
        arms,
        plan.q2_state,
        (plan.first_pair, plan.second_pair),
        tuple(q1_terminal_retained[value][1] for value in plan.second_pair),
        output / f"{plan.q2_state}.png",
        output / f"{plan.q2_state}.svg",
    )

    untouched_same_pairs = [
        tuple(vinyl for vinyl in range(1, 9) if arms[vinyl] == arm)
        for arm in sorted(set(arms.values()))
        if not set(vinyl for vinyl in range(1, 9) if arms[vinyl] == arm) & set(plan.first_pair)
    ]
    return {
        "system": plan.system,
        "status": "PASS",
        "purpose": "RDKit chemical-graph review before Amber/MD",
        "arm_mapping_1based": {f"V{vinyl}": f"A{arm}" for vinyl, arm in arms.items()},
        "q1": {
            "state": plan.q1_state,
            "reacted_pair": [f"V{value}" for value in plan.first_pair],
            "arms": [f"A{arms[value]}" for value in plan.first_pair],
            "selection": "largest q0 mean soft-contact cross-arm pair",
            "q0_mean_soft_contact": pair_soft_score(data, plan.first_pair),
            "formula": rdMolDescriptors.CalcMolFormula(q1),
            "active_vinyls": [f"V{value}" for value in q1_active],
            "new_internal_alkenes": 1,
            "all_new_internal_alkenes": "Z/cis",
            "mmff_energy_kcal_mol": q1_energy,
            "mmff_optimization_status": q1_status,
            "mapping": {f"V{k}": list(v) for k, v in q1_mapping.items()},
        },
        "q2": {
            "state": plan.q2_state,
            "reacted_pair": [f"V{value}" for value in plan.second_pair],
            "arms": [f"A{arms[value]}" for value in plan.second_pair],
            "selection": "largest q0 mean soft-contact among complete untouched same-arm pairs",
            "q0_mean_soft_contact": pair_soft_score(data, plan.second_pair),
            "eligible_untouched_same_arm_pairs": [
                {
                    "pair": [f"V{value}" for value in pair],
                    "q0_mean_soft_contact": pair_soft_score(data, pair),
                }
                for pair in untouched_same_pairs
            ],
            "formula": rdMolDescriptors.CalcMolFormula(q2),
            "active_vinyls": [f"V{value}" for value in q2_active],
            "new_internal_alkenes": 2,
            "all_new_internal_alkenes": "Z/cis",
            "mmff_energy_kcal_mol": q2_energy,
            "mmff_optimization_status": q2_status,
            "mapping": {f"V{k}": list(v) for k, v in q2_mapping.items()},
        },
        "q0_formula": rdMolDescriptors.CalcMolFormula(q0),
        "q0_to_q1_delta": formula_delta(q0, q1),
        "q1_to_q2_delta": formula_delta(q1_reloaded, q2),
    }


def make_grid(paths: Sequence[Path], output: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if len(images) != 4 or len({image.size for image in images}) != 1:
        raise ValueError("expected four equal-sized panels")
    width, height = images[0].size
    canvas = Image.new("RGB", (2 * width, 2 * height), "white")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 2) * width, (index // 2) * height))
    canvas.save(output, dpi=(300, 300))


def write_review(output: Path, audits: Sequence[dict[str, object]]) -> None:
    lines = [
        "# A4-p-C6 / A4-p-C8 q1-q2 RDKit review",
        "",
        "Only the established A4 same/cross/reuse workflow is represented. No Amber parameters or MD inputs have been generated.",
        "",
        "| system | q1 cross-arm pair | q2 same-arm pair | q1 active | q2 active | stereo |",
        "|---|---|---|---|---|---|",
    ]
    for audit in audits:
        q1 = audit["q1"]
        q2 = audit["q2"]
        lines.append(
            f"| {audit['system']} | {'-'.join(q1['reacted_pair'])} | "
            f"{'-'.join(q2['reacted_pair'])} | {', '.join(q1['active_vinyls'])} | "
            f"{', '.join(q2['active_vinyls'])} | all formed C=C are Z/cis |"
        )
    lines.extend([
        "",
        "Color key: purple is the first formed internal alkene, red is the second, and blue marks the remaining terminal vinyl groups.",
        "",
        "The q2 label is the complete untouched same-arm pair with the larger q0 mean soft-contact value. Final atom-label choice should be confirmed from the drawings before Amber preparation.",
    ])
    (output / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument(
        "--output",
        type=Path,
        default=root_default / "results" / "a4_p_c6_c8_q1_q2_rdkit_review",
    )
    parser.add_argument("--conformers", type=int, default=8)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audits = [build_system(root, output, plan, args.conformers) for plan in PLANS]
    grid_paths = [
        output / f"{plan.q1_state}.png" for plan in PLANS
    ] + [
        output / f"{plan.q2_state}.png" for plan in PLANS
    ]
    # Arrange each system by row: C6 q1/q2, then C8 q1/q2.
    grid_paths = [
        output / f"{PLANS[0].q1_state}.png",
        output / f"{PLANS[0].q2_state}.png",
        output / f"{PLANS[1].q1_state}.png",
        output / f"{PLANS[1].q2_state}.png",
    ]
    grid = output / "A4_p_C6_C8_q1_q2_review_grid.png"
    make_grid(grid_paths, grid)
    (output / "audit.json").write_text(
        json.dumps({"status": "PASS", "systems": audits}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_review(output, audits)
    print(json.dumps({
        "status": "PASS",
        "states": [
            state for plan in PLANS for state in (plan.q1_state, plan.q2_state)
        ],
        "grid": str(grid),
    }, indent=2))


if __name__ == "__main__":
    main()
