#!/usr/bin/env python3
"""Build one q1 and the three q2 branch structures for A4_p-C6/C8.

q1 is a single cross-arm metathesis.  From that q1, q2 is split into the
mixed (same-arm), adjacent (cross-arm sharing one q1 arm), and opposite
(cross-arm on the two other arms) topologies.  All new internal alkenes are
written as Z/cis.  This script stops at RDKit review structures; it does not
prepare Amber or launch MD.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

import analyze_react1_structures as react1
import build_a4_p_q1_q2_review as base
import build_q2_branch_structures as metathesis
import mtmr_baseline as baseline


SYSTEMS = ("A4_p_C6", "A4_p_C8")
BRANCHES = ("mixed", "adjacent", "opposite")


def classify(first: tuple[int, int], second: tuple[int, int], arms: dict[int, int]) -> str | None:
    first_arms = {arms[value] for value in first}
    second_arms = {arms[value] for value in second}
    if len(second_arms) == 1:
        return "mixed"
    shared = len(first_arms & second_arms)
    return "adjacent" if shared == 1 else "opposite" if shared == 0 else None


def select_first_and_q2(
    root: Path, system: str
) -> tuple[tuple[int, int], dict[str, tuple[int, int]], dict[int, int], baseline.SystemData, dict[str, list[dict[str, object]]]]:
    data, _ = baseline.analyse_system(
        root, root / "trace" / f"{system}_md", baseline.DEFAULT_THRESHOLDS
    )
    if data is None:
        raise RuntimeError(f"cannot load q0 data for {system}")
    arms = {vinyl: int(data.terminal_to_monomer[vinyl - 1]) for vinyl in range(1, 9)}
    scores = {
        tuple(sorted((left + 1, right + 1))): float(data.cycle_soft_q[:, index].mean())
        for index, (left, right) in enumerate(data.pair_vertices)
    }
    cross = [
        pair for pair in combinations(range(1, 9), 2)
        if arms[pair[0]] != arms[pair[1]]
    ]
    first = min(cross, key=lambda pair: (-scores[pair], pair))
    active = set(range(1, 9)) - set(first)
    candidates: dict[str, list[dict[str, object]]] = {branch: [] for branch in BRANCHES}
    for pair in combinations(sorted(active), 2):
        branch = classify(first, pair, arms)
        if branch is None:
            continue
        candidates[branch].append({
            "pair": pair,
            "score": scores[pair],
            "arms": (arms[pair[0]], arms[pair[1]]),
        })
    selected = {
        branch: min(rows, key=lambda row: (-float(row["score"]), row["pair"]))["pair"]
        for branch, rows in candidates.items()
    }
    return first, selected, {k: v + 1 for k, v in arms.items()}, data, candidates


def build_system(root: Path, output: Path, system: str, conformers: int) -> dict[str, object]:
    first, second_by_branch, arms, data, candidates = select_first_and_q2(root, system)
    q0 = metathesis.load_rdkit_pdb(root / "structures" / f"{system}.pdb")
    q0_mapping = react1.vinyl_mapping(root / "trace" / f"{system}_md" / "md_simulation.log")
    q0_retained = base.terminal_retained_serials(q0, q0_mapping)

    q1_state = f"{system}_1_cis"
    q1 = metathesis.edit_second_metathesis(q0, q0_mapping, first)
    q1, q1_energy, q1_status = metathesis.embed_and_optimize(
        q1, baseline.stable_seed(20260905, q1_state) % (2**31 - 1), conformers
    )
    q1_geometry = metathesis.stereo_geometry_rows(q1)
    if base.formula_delta(q0, q1) != {"C": 2, "H": 4} or len(q1_geometry) != 1:
        raise ValueError(f"{q1_state}: q1 QC failed")
    if q1_geometry[0]["geometry_classification"] != "cis":
        raise ValueError(f"{q1_state}: new double bond is not cis")
    q1_path = output / f"{q1_state}.pdb"
    q1_source_to_output = base.save_3d_product(q1, q1_path)
    base.save_sdf(q1, output / f"{q1_state}.sdf")
    q1_mapping = {
        vinyl: tuple(q1_source_to_output[serial] for serial in q0_mapping[vinyl])
        for vinyl in sorted(set(q0_mapping) - set(first))
    }
    base.draw_state(
        q1, q0_mapping, sorted(q1_mapping), arms, q1_state, (first,),
        tuple(q0_retained[value][1] for value in first),
        output / f"{q1_state}.png", output / f"{q1_state}.svg",
    )

    branch_audits: dict[str, dict[str, object]] = {}
    for branch in BRANCHES:
        second = second_by_branch[branch]
        q2_state = f"{system}_2_{branch}_cis"
        q1_reloaded = metathesis.load_rdkit_pdb(q1_path)
        q1_retained = base.terminal_retained_serials(q1_reloaded, q1_mapping)
        q2 = metathesis.edit_second_metathesis(q1_reloaded, q1_mapping, second)
        q2, q2_energy, q2_status = metathesis.embed_and_optimize(
            q2, baseline.stable_seed(20260905, q2_state) % (2**31 - 1), conformers
        )
        q2_geometry = metathesis.stereo_geometry_rows(q2)
        if base.formula_delta(q1_reloaded, q2) != {"C": 2, "H": 4}:
            raise ValueError(f"{q2_state}: formula delta is not C2H4")
        if len(q2_geometry) != 2 or any(
            row["geometry_classification"] != "cis" for row in q2_geometry
        ):
            raise ValueError(f"{q2_state}: cis geometry QC failed")
        q2_path = output / f"{q2_state}.pdb"
        q2_source_to_output = base.save_3d_product(q2, q2_path)
        base.save_sdf(q2, output / f"{q2_state}.sdf")
        q2_active = sorted(set(q1_mapping) - set(second))
        q2_mapping = {
            vinyl: tuple(q2_source_to_output[serial] for serial in q1_mapping[vinyl])
            for vinyl in q2_active
        }
        base.draw_state(
            q2, q1_mapping, q2_active, arms, q2_state, (first, second),
            tuple(q1_retained[value][1] for value in second),
            output / f"{q2_state}.png", output / f"{q2_state}.svg",
        )
        branch_audits[branch] = {
            "state": q2_state,
            "reacted_pair": [f"V{value}" for value in second],
            "arms": [f"A{arms[value]}" for value in second],
            "q0_mean_soft_contact": next(
                float(row["score"]) for row in candidates[branch] if row["pair"] == second
            ),
            "active_vinyls": [f"V{value}" for value in q2_active],
            "formula": rdMolDescriptors.CalcMolFormula(q2),
            "new_internal_alkenes": 2,
            "all_new_internal_alkenes": "Z/cis",
            "mmff_energy_kcal_mol": q2_energy,
            "mmff_optimization_status": q2_status,
            "mapping": {f"V{k}": list(v) for k, v in q2_mapping.items()},
            "candidate_pairs_ranked": [
                {
                    "pair": [f"V{x}" for x in row["pair"]],
                    "q0_mean_soft_contact": row["score"],
                    "arms": [f"A{x}" for x in row["arms"]],
                }
                for row in sorted(candidates[branch], key=lambda row: (-float(row["score"]), row["pair"]))
            ],
        }

    return {
        "system": system,
        "status": "PASS",
        "q0_formula": rdMolDescriptors.CalcMolFormula(q0),
        "arm_mapping_1based": {f"V{k}": f"A{v}" for k, v in arms.items()},
        "q1": {
            "state": q1_state,
            "reacted_pair": [f"V{value}" for value in first],
            "arms": [f"A{arms[value]}" for value in first],
            "selection": "largest q0 mean soft-contact cross-arm pair",
            "q0_mean_soft_contact": float(
                data.cycle_soft_q[:, data.pair_vertices.index((first[0] - 1, first[1] - 1))].mean()
            ),
            "active_vinyls": [f"V{value}" for value in sorted(q1_mapping)],
            "formula": rdMolDescriptors.CalcMolFormula(q1),
            "new_internal_alkenes": 1,
            "all_new_internal_alkenes": "Z/cis",
            "mmff_energy_kcal_mol": q1_energy,
            "mmff_optimization_status": q1_status,
            "mapping": {f"V{k}": list(v) for k, v in q1_mapping.items()},
        },
        "q2": branch_audits,
        "q0_to_q1_delta": base.formula_delta(q0, q1),
    }


def grid(output_path: Path, paths: list[Path]) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    if len(images) != 8 or len({image.size for image in images}) != 1:
        raise ValueError("expected eight equal-sized panels")
    width, height = images[0].size
    canvas = Image.new("RGB", (4 * width, 2 * height), "white")
    for index, image in enumerate(images):
        canvas.paste(image, ((index % 4) * width, (index // 4) * height))
    canvas.save(output_path, dpi=(300, 300))


def write_review(output: Path, audits: list[dict[str, object]]) -> None:
    lines = [
        "# A4-p-C6 / A4-p-C8 q1 + three q2 branches",
        "",
        "q1 is one cross-arm reaction. q2 is represented by mixed (same-arm), adjacent (cross-arm sharing one q1 arm), and opposite (cross-arm on the two other arms). All formed internal alkenes are Z/cis. Amber and MD have not been run.",
        "",
        "| system | q1 | q2-mixed | q2-adjacent | q2-opposite |",
        "|---|---|---|---|---|",
    ]
    for audit in audits:
        q1 = audit["q1"]
        q2 = audit["q2"]
        values = ["-".join(q1["reacted_pair"])] + ["-".join(q2[b]["reacted_pair"]) for b in BRANCHES]
        lines.append(f"| {audit['system']} | " + " | ".join(values) + " |")
    lines.extend([
        "",
        "图中紫色为 q1 新生双键，红色为 q2 新生双键，蓝色为 q2 后仍保留的乙烯基。位点候选按 q0 平均 soft-contact 在各拓扑类别内排序，当前取最高者供审核。",
    ])
    (output / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument(
        "--output", type=Path,
        default=root_default / "results" / "a4_p_c6_c8_q1_q2_three_branches_rdkit_review",
    )
    parser.add_argument("--conformers", type=int, default=8)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    audits = [build_system(root, output, system, args.conformers) for system in SYSTEMS]
    paths: list[Path] = []
    for system in SYSTEMS:
        paths.append(output / f"{system}_1_cis.png")
        paths.extend(output / f"{system}_2_{branch}_cis.png" for branch in BRANCHES)
    grid_path = output / "A4_p_C6_C8_q1_q2_three_branches_review_grid.png"
    grid(grid_path, paths)
    (output / "audit.json").write_text(json.dumps({"status": "PASS", "systems": audits}, indent=2) + "\n", encoding="utf-8")
    write_review(output, audits)
    print(json.dumps({"status": "PASS", "grid": str(grid_path)}, indent=2))


if __name__ == "__main__":
    main()
