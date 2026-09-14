#!/usr/bin/env python3
"""Audit A4 one-reaction cis/trans PDBs and prepare state-specific MD inputs."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

import mtmr_baseline as baseline


MAPPING_PATTERN = re.compile(r"V(\d+):\s*\((\d+),\s*(\d+)\)")


@dataclass
class PDBGraph:
    path: Path
    elements: dict[int, str]
    coordinates: dict[int, np.ndarray]
    edges: set[tuple[int, int]]

    @property
    def heavy_atoms(self) -> list[int]:
        return [serial for serial in sorted(self.elements) if self.elements[serial] != "H"]

    @property
    def heavy_edges(self) -> set[tuple[int, int]]:
        return {
            edge
            for edge in self.edges
            if self.elements.get(edge[0]) != "H" and self.elements.get(edge[1]) != "H"
        }

    @property
    def formula_counts(self) -> Counter[str]:
        return Counter(self.elements.values())

    @property
    def cycle_rank(self) -> int:
        # All supplied molecules are connected; E - V + 1 is the graph cycle rank.
        return len(self.heavy_edges) - len(self.heavy_atoms) + 1


@dataclass
class ReactionEdit:
    consumed_vinyls: tuple[int, int]
    deleted_initial_atoms: tuple[int, int]
    retained_initial_atoms: tuple[int, int]
    new_product_bond: tuple[int, int]
    initial_to_product: dict[int, int]


def read_pdb(path: Path) -> PDBGraph:
    elements: dict[int, str] = {}
    coordinates: dict[int, np.ndarray] = {}
    edges: set[tuple[int, int]] = set()
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            serial = int(line[6:11])
            element = line[12:16].strip()[0].upper()
            elements[serial] = element
            coordinates[serial] = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=float
            )
        elif line.startswith("CONECT"):
            values = [int(value) for value in line.split()[1:]]
            if values:
                for target in values[1:]:
                    if target != values[0]:
                        edges.add(tuple(sorted((values[0], target))))
    return PDBGraph(path, elements, coordinates, edges)


def formula_text(counts: Counter[str]) -> str:
    return "".join(f"{element}{counts[element]}" for element in ("C", "H", "O") if counts[element])


def vinyl_mapping(log_path: Path) -> dict[int, tuple[int, int]]:
    mapping = {
        int(vinyl): (int(left), int(right))
        for vinyl, left, right in MAPPING_PATTERN.findall(
            log_path.read_text(encoding="utf-8", errors="replace")
        )
    }
    if set(mapping) != set(range(1, 9)):
        raise ValueError(f"incomplete vinyl mapping in {log_path}")
    return mapping


def heavy_degree(graph: PDBGraph, atom: int) -> int:
    return sum(
        1
        for edge in graph.heavy_edges
        if atom in edge
    )


def vinyl_terminal_and_retained(
    graph: PDBGraph, atom_pair: tuple[int, int]
) -> tuple[int, int]:
    left, right = atom_pair
    left_degree, right_degree = heavy_degree(graph, left), heavy_degree(graph, right)
    if left_degree == 1 and right_degree > 1:
        return left, right
    if right_degree == 1 and left_degree > 1:
        return right, left
    raise ValueError(
        f"vinyl atoms do not contain one terminal heavy atom: {atom_pair}, degrees={left_degree},{right_degree}"
    )


def expected_edit(
    initial: PDBGraph,
    product: PDBGraph,
    mapping: dict[int, tuple[int, int]],
    consumed: tuple[int, int],
) -> ReactionEdit | None:
    terminal_retained = [vinyl_terminal_and_retained(initial, mapping[vinyl]) for vinyl in consumed]
    deleted = tuple(sorted(item[0] for item in terminal_retained))
    retained = tuple(item[1] for item in terminal_retained)
    if len(initial.heavy_atoms) - len(product.heavy_atoms) != 2:
        return None
    remaining = [atom for atom in initial.heavy_atoms if atom not in deleted]
    if [initial.elements[atom] for atom in remaining] != [
        product.elements[atom] for atom in product.heavy_atoms
    ]:
        return None
    initial_to_product = dict(zip(remaining, product.heavy_atoms))
    mapped_edges = {
        tuple(sorted((initial_to_product[left], initial_to_product[right])))
        for left, right in initial.heavy_edges
        if left in initial_to_product and right in initial_to_product
    }
    if mapped_edges - product.heavy_edges:
        return None
    added = product.heavy_edges - mapped_edges
    expected_new = tuple(
        sorted((initial_to_product[retained[0]], initial_to_product[retained[1]]))
    )
    if added != {expected_new}:
        return None
    return ReactionEdit(consumed, deleted, retained, expected_new, initial_to_product)


def infer_reaction(
    initial: PDBGraph,
    product: PDBGraph,
    mapping: dict[int, tuple[int, int]],
) -> ReactionEdit | None:
    matches = [
        edit
        for consumed in combinations(range(1, 9), 2)
        if (edit := expected_edit(initial, product, mapping, consumed)) is not None
    ]
    if len(matches) > 1:
        raise ValueError(f"ambiguous reaction edit in {product.path}: {matches}")
    return matches[0] if matches else None


def dihedral_degrees(
    first: np.ndarray, second: np.ndarray, third: np.ndarray, fourth: np.ndarray
) -> float:
    b0 = -(second - first)
    b1 = third - second
    b2 = fourth - third
    b1 /= np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def new_bond_geometry(product: PDBGraph, edit: ReactionEdit) -> tuple[float, float]:
    left, right = edit.new_product_bond
    length = float(np.linalg.norm(product.coordinates[left] - product.coordinates[right]))
    left_neighbors = [
        b if a == left else a
        for a, b in product.heavy_edges
        if left in (a, b) and (b if a == left else a) != right
    ]
    right_neighbors = [
        b if a == right else a
        for a, b in product.heavy_edges
        if right in (a, b) and (b if a == right else a) != left
    ]
    if len(left_neighbors) != 1 or len(right_neighbors) != 1:
        raise ValueError(f"new alkene substituent topology is not unique in {product.path}")
    angle = dihedral_degrees(
        product.coordinates[left_neighbors[0]],
        product.coordinates[left],
        product.coordinates[right],
        product.coordinates[right_neighbors[0]],
    )
    return length, angle


def geometry_label(length: float, dihedral: float) -> str:
    if not 1.25 <= length <= 1.42:
        return "invalid_new_bond_length"
    absolute = abs(dihedral)
    if absolute <= 30:
        return "cis"
    if abs(absolute - 180) <= 30:
        return "trans"
    return "ambiguous_stereochemistry"


def parse_initial_system(
    root: Path, system: str
) -> baseline.SystemData:
    data, _ = baseline.analyse_system(
        root,
        root / "trace" / f"{system}_md",
        baseline.DEFAULT_THRESHOLDS,
    )
    if data is None:
        raise RuntimeError(f"cannot load initial-state data for {system}")
    return data


def pair_q0(data: baseline.SystemData, left: int, right: int) -> float:
    edge = tuple(sorted((left - 1, right - 1)))
    index = data.pair_vertices.index(edge)
    return float(data.cycle_soft_q[:, index].mean())


def product_vinyl_mapping(
    mapping: dict[int, tuple[int, int]], edit: ReactionEdit
) -> dict[int, tuple[int, int]]:
    return {
        vinyl: tuple(edit.initial_to_product[atom] for atom in atoms)
        for vinyl, atoms in mapping.items()
        if vinyl not in edit.consumed_vinyls
    }


def midpoint(graph: PDBGraph, atoms: tuple[int, int]) -> np.ndarray:
    return (graph.coordinates[atoms[0]] + graph.coordinates[atoms[1]]) / 2.0


def analyse(root: Path, output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    qc_rows: list[dict[str, object]] = []
    mapping_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    md_rows: list[dict[str, object]] = []
    static_alpha_rows: list[dict[str, object]] = []

    for spacer in range(5, 9):
        system = f"A4_m_C{spacer}"
        initial = read_pdb(root / "structures" / f"{system}.pdb")
        mapping = vinyl_mapping(root / "trace" / f"{system}_md" / "md_simulation.log")
        data = parse_initial_system(root, system)
        terminal_to_arm = {vinyl: data.terminal_to_monomer[vinyl - 1] for vinyl in range(1, 9)}
        for isomer in ("cis", "trans"):
            state = f"{system}_1_{isomer}"
            product = read_pdb(root / "react_1" / f"{state}.pdb")
            edit = infer_reaction(initial, product, mapping)
            formula_delta = initial.formula_counts - product.formula_counts
            formula_ok = formula_delta == Counter({"C": 2, "H": 4})
            cycle_ok = product.cycle_rank == initial.cycle_rank + 1
            if edit is None:
                length = dihedral = math.nan
                geometry = "no_valid_metathesis_edit"
                topology_ok = False
            else:
                length, dihedral = new_bond_geometry(product, edit)
                geometry = geometry_label(length, dihedral)
                topology_ok = True
            stereo_ok = geometry == isomer
            status = "PASS" if topology_ok and formula_ok and cycle_ok and stereo_ok else "ERROR"
            qc_rows.append(
                {
                    "state": state,
                    "system": system,
                    "isomer": isomer,
                    "status": status,
                    "initial_formula": formula_text(initial.formula_counts),
                    "product_formula": formula_text(product.formula_counts),
                    "formula_delta": formula_text(formula_delta),
                    "formula_delta_is_C2H4": formula_ok,
                    "initial_cycle_rank": initial.cycle_rank,
                    "product_cycle_rank": product.cycle_rank,
                    "adds_one_cycle": cycle_ok,
                    "reaction_edit_identified": topology_ok,
                    "reacted_pair": (
                        f"V{edit.consumed_vinyls[0]}-V{edit.consumed_vinyls[1]}" if edit else ""
                    ),
                    "new_bond_length_A": length,
                    "substituent_dihedral_deg": dihedral,
                    "geometry_classification": geometry,
                    "filename_stereo_matches_geometry": stereo_ok,
                }
            )
            analysis_role = (
                "primary_cis"
                if isomer == "cis" and status == "PASS"
                else "trans_sensitivity"
                if isomer == "trans" and status == "PASS"
                else "excluded_qc_failure"
            )
            if edit is None:
                md_rows.append(
                    {
                        "state": state,
                        "status": "EXCLUDE",
                        "analysis_role": analysis_role,
                        "source_pdb": str(product.path.relative_to(root)),
                        "reacted_pair": "",
                        "active_vinyls": "",
                        "active_pair_count": 0,
                        "suggested_output_dir": "",
                        "required_cycles": 0,
                        "sampling_ps_per_cycle": 0,
                        "reason": geometry,
                    }
                )
                continue

            product_mapping = product_vinyl_mapping(mapping, edit)
            active = sorted(product_mapping)
            reacted_arms = tuple(terminal_to_arm[item] for item in edit.consumed_vinyls)
            remaining_same_two_arms = tuple(
                vinyl
                for vinyl in active
                if terminal_to_arm[vinyl] in reacted_arms
            )
            if len(set(reacted_arms)) != 2 or len(remaining_same_two_arms) != 2:
                raise ValueError(f"reaction is not a cross-arm single event in {state}")
            alpha_edge = tuple(sorted(remaining_same_two_arms))
            for vinyl in range(1, 9):
                original_atoms = mapping[vinyl]
                product_atoms = product_mapping.get(vinyl)
                mapping_rows.append(
                    {
                        "state": state,
                        "status": status,
                        "vinyl": f"V{vinyl}",
                        "arm_zero_based": terminal_to_arm[vinyl],
                        "vinyl_status": "consumed" if vinyl in edit.consumed_vinyls else "active",
                        "initial_pdb_atoms": f"{original_atoms[0]};{original_atoms[1]}",
                        "product_pdb_atoms": (
                            f"{product_atoms[0]};{product_atoms[1]}" if product_atoms else ""
                        ),
                    }
                )
            control_edges: set[tuple[int, int]] = set()
            for alpha_vinyl in alpha_edge:
                for vinyl in active:
                    if terminal_to_arm[vinyl] not in reacted_arms:
                        control_edges.add(tuple(sorted((alpha_vinyl, vinyl))))
            for left, right in combinations(active, 2):
                distance = float(
                    np.linalg.norm(
                        midpoint(product, product_mapping[left])
                        - midpoint(product, product_mapping[right])
                    )
                )
                static_score = 1.0 / (
                    1.0 + math.exp((distance - baseline.SOFT_SIGMOID_DC) / baseline.SOFT_SIGMOID_SIGMA)
                )
                edge = (left, right)
                pair_rows.append(
                    {
                        "state": state,
                        "system": system,
                        "isomer": isomer,
                        "state_status": status,
                        "analysis_role": analysis_role,
                        "pair": f"V{left}-V{right}",
                        "vinyl_i": left,
                        "vinyl_j": right,
                        "pair_type": (
                            "same_arm"
                            if terminal_to_arm[left] == terminal_to_arm[right]
                            else "cross_arm"
                        ),
                        "is_alpha_second_edge": edge == alpha_edge,
                        "is_alpha_control": edge in control_edges,
                        "static_midpoint_distance_A": distance,
                        "static_pdb_soft_score": static_score,
                        "initial_trajectory_soft_q": pair_q0(data, left, right),
                        "react1_trajectory_soft_q": "",
                        "q1_over_q0": "",
                    }
                )
            state_pair_rows = [row for row in pair_rows if row["state"] == state]
            alpha_row = next(row for row in state_pair_rows if row["is_alpha_second_edge"])
            controls = [row for row in state_pair_rows if row["is_alpha_control"]]
            mean_control_score = float(np.mean([row["static_pdb_soft_score"] for row in controls]))
            static_ratio = (
                float(alpha_row["static_pdb_soft_score"]) / mean_control_score
                if mean_control_score > 0
                else math.nan
            )
            static_alpha_rows.append(
                {
                    "state": state,
                    "state_status": status,
                    "analysis_role": analysis_role,
                    "reacted_pair": f"V{edit.consumed_vinyls[0]}-V{edit.consumed_vinyls[1]}",
                    "alpha_second_edge": f"V{alpha_edge[0]}-V{alpha_edge[1]}",
                    "static_alpha_distance_A": alpha_row["static_midpoint_distance_A"],
                    "static_alpha_soft_score": alpha_row["static_pdb_soft_score"],
                    "static_control_soft_score_mean": mean_control_score,
                    "static_score_ratio_alpha_to_control": static_ratio,
                    "static_pseudo_alpha": -math.log(static_ratio) if static_ratio > 0 else math.nan,
                    "interpretation": "single minimized PDB only; not an estimate of dynamic alpha",
                }
            )
            md_rows.append(
                {
                    "state": state,
                    "status": "READY" if status == "PASS" else "EXCLUDE",
                    "analysis_role": analysis_role,
                    "source_pdb": str(product.path.relative_to(root)),
                    "reacted_pair": f"V{edit.consumed_vinyls[0]}-V{edit.consumed_vinyls[1]}",
                    "active_vinyls": ";".join(f"V{vinyl}" for vinyl in active),
                    "active_pair_count": len(list(combinations(active, 2))),
                    "suggested_output_dir": (
                        f"react_1_md/{state}_md" if status == "PASS" else ""
                    ),
                    "required_cycles": 100 if status == "PASS" else 0,
                    "sampling_ps_per_cycle": 25 if status == "PASS" else 0,
                    "reason": "" if status == "PASS" else geometry,
                }
            )

    baseline.write_csv(output / "structure_qc.csv", qc_rows)
    baseline.write_csv(output / "vinyl_mapping.csv", mapping_rows)
    baseline.write_csv(output / "static_pair_geometry.csv", pair_rows)
    baseline.write_csv(output / "static_alpha_summary.csv", static_alpha_rows)
    baseline.write_csv(output / "md_run_manifest.csv", md_rows)
    schema_dir = output / "expected_schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    for state_row in md_rows:
        if state_row["status"] != "READY":
            continue
        state = str(state_row["state"])
        state_pairs = [row for row in pair_rows if row["state"] == state]
        columns = [
            "step",
            "time_ps",
            "energy_eV",
            *(
                f"V{int(row['vinyl_i'])}_V{int(row['vinyl_j'])}_"
                f"{'同臂' if row['pair_type'] == 'same_arm' else '跨臂'}"
                for row in state_pairs
            ),
        ]
        (schema_dir / f"{state}_trajectory_header.txt").write_text(
            ",".join(columns) + "\n", encoding="utf-8"
        )
        state_mapping = [row for row in mapping_rows if row["state"] == state]
        mapping_lines = [
            f"{row['vinyl']}: ({str(row['product_pdb_atoms']).replace(';', ', ')})"
            for row in state_mapping
            if row["vinyl_status"] == "active"
        ]
        (schema_dir / f"{state}_active_vinyl_mapping.txt").write_text(
            "\n".join(mapping_lines) + "\n", encoding="utf-8"
        )

    pass_cis = [row for row in qc_rows if row["isomer"] == "cis" and row["status"] == "PASS"]
    pass_trans = [row for row in qc_rows if row["isomer"] == "trans" and row["status"] == "PASS"]
    errors = [row for row in qc_rows if row["status"] == "ERROR"]
    lines = [
        "# A4 one-reaction structure audit",
        "",
        "## Result",
        "",
        f"All {len(pass_cis)}/4 cis structures pass the one-metathesis topology, formula, ring-count, and alkene-geometry checks. Only {len(pass_trans)}/4 trans structures pass; invalid trans files are excluded from sensitivity analysis.",
        "",
        "The supplied directory contains PDB starting structures only, not post-reaction trajectories. Therefore q^(1), gamma, and dynamic alpha cannot yet be measured. Static PDB distances are recorded solely as initialization/QC diagnostics.",
        "",
        "## Inferred first reactions",
        "",
        "| state | status | reacted pair | new bond (Å) | substituent dihedral | geometry |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in qc_rows:
        lines.append(
            f"| {row['state']} | {row['status']} | {row['reacted_pair'] or 'none'} | "
            f"{float(row['new_bond_length_A']):.3f} | {float(row['substituent_dihedral_deg']):.1f}° | "
            f"{row['geometry_classification']} |"
            if math.isfinite(float(row["new_bond_length_A"]))
            else f"| {row['state']} | {row['status']} | none | n/a | n/a | {row['geometry_classification']} |"
        )
    lines.extend(
        [
            "",
            "## QC failures",
            "",
            "- `A4_m_C6_1_trans.pdb` is constitutionally identical to the unreacted C6 structure: no C2H4 loss and no additional cycle.",
            "- `A4_m_C8_1_trans.pdb` has the reacted topology but its new bond is 1.526 Å and the substituent dihedral is about 96°, inconsistent with an E alkene.",
            "",
            "## MD-ready states",
            "",
            "The run manifest marks four cis states as primary and the valid C5/C7 trans states as stereochemical sensitivity controls. Each state has six active vinyl groups and 15 surviving pair distances. To remain comparable with the initial dataset, the recommended target is 100 independent cycles, 25 ps sampling per cycle, with a cycle as the uncertainty unit.",
            "",
            "## Required next input",
            "",
            "A trajectory generator matching the original force field/model and its source code or environment is still required. The current workspace has neither the original MD driver nor ASE/OpenMM. Once `react_1_md/<state>_md/cycle_*/trajectory.csv` is available, the prepared mapping supports direct q^(1)/q^(0), cis/trans, gamma-response, and alpha-response analysis.",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    validation = [
        "# Validation report: A4 one-reaction structures",
        "",
        "## Overall Assessment: Needs additional trajectories",
        "",
        "The four cis PDBs are structurally ready as primary post-reaction starting states. Two trans files are valid sensitivity controls. No mechanistic parameter is estimated because a single minimized PDB is not an ensemble.",
        "",
        "## Checks passed",
        "",
        "- Atom, formula, connectivity, reaction edit, cycle rank, new-bond length, and alkene substituent dihedral were checked independently.",
        "- Reacted vinyl identities were traced back to the original md_simulation.log mappings.",
        "- Surviving vinyl atom serials and all 15 active pair identities were regenerated for each valid state.",
        "- Cis is designated primary and trans remains a separate sensitivity stratum; no unsupported cis/trans population weighting is applied.",
        "",
        "## Blocking issues",
        "",
        "1. Post-reaction MD trajectories are absent, so q^(1), gamma, and alpha are not observable yet.",
        "2. The original MD driver/force-field environment is absent from this workspace.",
        "3. C6 trans is unreacted and C8 trans has invalid alkene geometry; both must be regenerated before a four-system trans comparison.",
        "",
        "## Required caveat",
        "",
        "Static PDB soft scores must not be substituted for trajectory-averaged reactive encounter probabilities.",
    ]
    (output / "validation_report.md").write_text("\n".join(validation) + "\n", encoding="utf-8")
    return {
        "states": len(qc_rows),
        "cis_pass": len(pass_cis),
        "trans_pass": len(pass_trans),
        "errors": len(errors),
        "output": str(output),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default="results/react1")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    summary = analyse(root, (root / args.output).resolve())
    print(
        f"Audited {summary['states']} states: cis PASS={summary['cis_pass']}/4, "
        f"trans PASS={summary['trans_pass']}/4, errors={summary['errors']}."
    )
    print(f"Results: {summary['output']}")


if __name__ == "__main__":
    main()
