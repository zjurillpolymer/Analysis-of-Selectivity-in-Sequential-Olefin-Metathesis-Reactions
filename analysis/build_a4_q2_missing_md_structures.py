#!/usr/bin/env python3
"""Build MD-ready 3D structures for the four reviewed A4 q2 histories.

The two reacted alkene pairs define a final q2 graph; their order is deliberately
not assigned.  For each C5/C8 reuse or mixed same+cross graph, this script:

* performs the exact second metathesis edit on the validated q1 cis graph;
* samples ETKDGv3 macrocycle conformers and pre-optimizes them with MMFF94;
* retains a low-energy, geometrically diverse conformer set for MD replicas;
* writes a canonical PDB, an explicit-bond MOL2, a GROMACS state manifest,
  and a machine-readable chemistry/geometry audit.

MMFF energies are used only to reject poor starting geometries and rank the
initial conformers.  They are not interpreted as reaction free energies.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

import analyze_react1_structures as react1
import build_a4_q2_missing_review as review
import build_q2_branch_structures as metathesis
import mtmr_baseline as baseline


def physical_state(candidate: review.Candidate) -> str:
    role = "reuse" if candidate.history == "reuse" else "mixed_same_cross"
    return f"{candidate.system}_2_{role}_cis"


def one_conformer(molecule: Chem.Mol, conformer_id: int) -> Chem.Mol:
    result = Chem.Mol(molecule)
    conformer = Chem.Conformer(result.GetConformer(conformer_id))
    result.RemoveAllConformers()
    result.AddConformer(conformer, assignId=True)
    return result


def embed_optimize_all(
    molecule: Chem.Mol,
    *,
    seed: int,
    conformer_count: int,
    max_iters: int,
) -> tuple[Chem.Mol, list[dict[str, object]]]:
    molecule = Chem.Mol(molecule)
    molecule.RemoveAllConformers()
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = int(seed)
    parameters.useRandomCoords = True
    parameters.useMacrocycleTorsions = True
    parameters.useMacrocycle14config = True
    parameters.maxIterations = 3000
    parameters.pruneRmsThresh = -1.0
    conformer_ids = list(
        AllChem.EmbedMultipleConfs(
            molecule, numConfs=conformer_count, params=parameters
        )
    )
    if not conformer_ids:
        raise RuntimeError("ETKDGv3 generated no conformers")
    optimization = AllChem.MMFFOptimizeMoleculeConfs(
        molecule,
        numThreads=0,
        maxIters=max_iters,
        nonBondedThresh=100.0,
    )
    rows = []
    for conformer_id, (status, energy) in zip(conformer_ids, optimization):
        if not math.isfinite(float(energy)):
            continue
        rows.append(
            {
                "conformer_id": int(conformer_id),
                "mmff_status": int(status),
                "mmff_energy_kcal_mol": float(energy),
            }
        )
    if not rows:
        raise RuntimeError("MMFF94 returned no finite conformer energies")
    rows.sort(key=lambda row: float(row["mmff_energy_kcal_mol"]))
    minimum = float(rows[0]["mmff_energy_kcal_mol"])
    for row in rows:
        row["relative_mmff_energy_kcal_mol"] = (
            float(row["mmff_energy_kcal_mol"]) - minimum
        )
    return molecule, rows


def conformer_rms(molecule: Chem.Mol, left: int, right: int) -> float:
    # Hydrogen coordinates matter for packing, but heavy-atom RMSD is a more
    # stable diversity criterion for these flexible macrocycles.
    heavy = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() > 1]
    left_xyz = np.asarray(
        [molecule.GetConformer(left).GetAtomPosition(index) for index in heavy],
        dtype=float,
    )
    right_xyz = np.asarray(
        [molecule.GetConformer(right).GetAtomPosition(index) for index in heavy],
        dtype=float,
    )
    left_xyz -= left_xyz.mean(axis=0)
    right_xyz -= right_xyz.mean(axis=0)
    covariance = left_xyz.T @ right_xyz
    u, _, vt = np.linalg.svd(covariance)
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    rotation = u @ vt
    delta = left_xyz @ rotation - right_xyz
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def choose_diverse(
    molecule: Chem.Mol,
    rows: Sequence[dict[str, object]],
    *,
    replicas: int,
    minimum_rms_A: float,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for row in rows:
        conformer_id = int(row["conformer_id"])
        rms_values = [
            conformer_rms(molecule, conformer_id, int(other["conformer_id"]))
            for other in selected
        ]
        enriched = {
            **row,
            "minimum_heavy_atom_rmsd_to_earlier_selected_A": (
                min(rms_values) if rms_values else None
            ),
        }
        if not rms_values or min(rms_values) >= minimum_rms_A:
            selected.append(enriched)
        else:
            rejected.append(enriched)
        if len(selected) == replicas:
            break
    # Very similar low-energy basins are still valid independent MD starts.
    # Fill any shortfall by energy rank while recording the smaller RMSD.
    if len(selected) < replicas:
        selected_ids = {int(row["conformer_id"]) for row in selected}
        for row in [*rejected, *rows]:
            conformer_id = int(row["conformer_id"])
            if conformer_id in selected_ids:
                continue
            rms_values = [
                conformer_rms(molecule, conformer_id, int(other["conformer_id"]))
                for other in selected
            ]
            selected.append(
                {
                    **row,
                    "minimum_heavy_atom_rmsd_to_earlier_selected_A": min(rms_values),
                }
            )
            selected_ids.add(conformer_id)
            if len(selected) == replicas:
                break
    if len(selected) != replicas:
        raise RuntimeError(
            f"only {len(selected)} conformers available for {replicas} replicas"
        )
    return selected


def nonbonded_minimum_distance(molecule: Chem.Mol) -> float:
    conformer = molecule.GetConformer()
    excluded: set[tuple[int, int]] = set()
    for atom in molecule.GetAtoms():
        left = atom.GetIdx()
        for right in Chem.FindAtomEnvironmentOfRadiusN(molecule, 2, left):
            bond = molecule.GetBondWithIdx(right)
            # The helper returns bond indices, not atom indices.
            excluded.add(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))
    for bond in molecule.GetBonds():
        excluded.add(tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))))
    # Explicitly exclude all 1-3 atom pairs.
    for center in molecule.GetAtoms():
        neighbors = [atom.GetIdx() for atom in center.GetNeighbors()]
        excluded.update(tuple(sorted(pair)) for pair in combinations(neighbors, 2))
    minimum = math.inf
    for left in range(molecule.GetNumAtoms()):
        left_xyz = np.asarray(conformer.GetAtomPosition(left), dtype=float)
        for right in range(left + 1, molecule.GetNumAtoms()):
            if (left, right) in excluded:
                continue
            right_xyz = np.asarray(conformer.GetAtomPosition(right), dtype=float)
            minimum = min(minimum, float(np.linalg.norm(left_xyz - right_xyz)))
    return minimum


def active_mapping(
    molecule: Chem.Mol,
    source_mapping: Mapping[int, tuple[int, int]],
    consumed: Sequence[int],
    source_to_output: Mapping[int, int],
) -> dict[int, tuple[int, int]]:
    result = {
        vinyl: tuple(source_to_output[serial] for serial in source_mapping[vinyl])
        for vinyl in sorted(set(source_mapping) - set(consumed))
    }
    if len(result) != 4:
        raise ValueError(f"expected four active vinyls, found {result}")
    graph_edges = {
        tuple(sorted((bond.GetBeginAtomIdx() + 1, bond.GetEndAtomIdx() + 1)))
        for bond in molecule.GetBonds()
        if bond.GetBondType() == Chem.BondType.DOUBLE
    }
    if any(tuple(sorted(atoms)) not in graph_edges for atoms in result.values()):
        raise ValueError(f"active-vinyl mapping does not point to C=C bonds: {result}")
    return result


def pair_schema(
    mapping: Mapping[int, tuple[int, int]],
    arm_by_vinyl: Mapping[int, int],
) -> list[dict[str, object]]:
    rows = []
    for left, right in combinations(sorted(mapping), 2):
        same = arm_by_vinyl[left] == arm_by_vinyl[right]
        rows.append(
            {
                "left": left,
                "right": right,
                "topology": "same_arm" if same else "cross_arm",
                "column": f"V{left}_V{right}_{'同臂' if same else '跨臂'}",
            }
        )
    return rows


def write_manifest(path: Path, rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_export_compatibility_metadata(
    root: Path,
    output: Path,
    candidates: Sequence[review.Candidate],
    manifests: Sequence[Mapping[str, object]],
) -> None:
    """Write the q2 history tables consumed by the compact NAC exporter.

    The generic exporter predates these reuse/mixed states and reads the
    project-style ``results/react2`` tables to recover arm identities and the
    two reacted sites.  These tables are metadata only: pair order is retained
    for atom bookkeeping and explicitly marked as physically unordered.
    """
    manifest_by_state = {str(row["state"]): row for row in manifests}
    mapping_rows: list[dict[str, object]] = []
    structure_rows: list[dict[str, object]] = []
    for candidate in candidates:
        state = physical_state(candidate)
        manifest = manifest_by_state[state]
        active_mapping = {
            int(label.removeprefix("V")): tuple(int(value) for value in atoms)
            for label, atoms in json.loads(str(manifest["vinyl_mapping_json"])).items()
        }
        q1_mapping = metathesis.q1_mapping(root, candidate.q1_state)
        for vinyl in range(1, 9):
            mapping_rows.append(
                {
                    "state": state,
                    "vinyl": f"V{vinyl}",
                    "arm_zero_based": candidate.arm_by_vinyl[vinyl],
                    "vinyl_status": "active" if vinyl in active_mapping else "consumed",
                    "q1_pdb_atoms": (
                        ";".join(str(value) for value in q1_mapping[vinyl])
                        if vinyl in q1_mapping
                        else ""
                    ),
                    "q2_pdb_atoms": (
                        ";".join(str(value) for value in active_mapping[vinyl])
                        if vinyl in active_mapping
                        else ""
                    ),
                }
            )
        structure_rows.append(
            {
                "state": state,
                "system": candidate.system,
                "branch": manifest["branch"],
                "status": "PASS",
                "first_reacted_pair": (
                    f"V{candidate.first_pair[0]}-V{candidate.first_pair[1]}"
                ),
                "second_reacted_pair": (
                    f"V{candidate.second_pair[0]}-V{candidate.second_pair[1]}"
                ),
                "pair_order_has_physical_meaning": "false",
            }
        )
    metadata_dir = output / "results/react2"
    write_manifest(metadata_dir / "vinyl_mapping.csv", mapping_rows)
    write_manifest(metadata_dir / "structure_qc.csv", structure_rows)


def copy_protocol_inputs(root: Path, output: Path) -> None:
    shutil.copytree(
        root / "gromacs_dcm/mdp", output / "gromacs_dcm/mdp", dirs_exist_ok=True
    )
    shutil.copytree(
        root / "gromacs_dcm_server_bundle/gromacs_dcm/models/ch2cl2_dc",
        output / "gromacs_dcm/models/ch2cl2_dc",
        dirs_exist_ok=True,
    )
    density = Path("gromacs_dcm/solvent_validation/ch2cl2_dc/run/density_gate.json")
    source = root / "gromacs_dcm_server_bundle" / density
    if not source.is_file():
        raise FileNotFoundError(f"validated DCM density gate is missing: {source}")
    destination = output / density
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_candidate(
    root: Path,
    output: Path,
    candidate: review.Candidate,
    *,
    conformer_count: int,
    replicas: int,
    minimum_rms_A: float,
    max_iters: int,
) -> tuple[dict[str, object], dict[str, object]]:
    state = physical_state(candidate)
    q1_path = root / "react_1" / f"{candidate.q1_state}.pdb"
    q1 = metathesis.load_rdkit_pdb(q1_path)
    q1_mapping = metathesis.q1_mapping(root, candidate.q1_state)
    product = metathesis.edit_second_metathesis(
        q1, q1_mapping, candidate.second_pair
    )
    formula_delta = review.formula_delta(q1, product)
    if formula_delta != {"C": 2, "H": 4}:
        raise ValueError(f"{state} has wrong q1->q2 formula delta: {formula_delta}")
    q1_cycle_rank = q1.GetNumBonds() - q1.GetNumAtoms() + 1
    q2_cycle_rank = product.GetNumBonds() - product.GetNumAtoms() + 1
    if q2_cycle_rank != q1_cycle_rank + 1:
        raise ValueError(f"{state} does not add exactly one graph cycle")

    seed = baseline.stable_seed(20260829, state) % (2**31 - 1)
    ensemble, all_rows = embed_optimize_all(
        product,
        seed=seed,
        conformer_count=conformer_count,
        max_iters=max_iters,
    )
    selected = choose_diverse(
        ensemble,
        all_rows,
        replicas=replicas,
        minimum_rms_A=minimum_rms_A,
    )

    structures = output / "structures"
    conformer_dir = structures / f"{state}_initial_conformers"
    conformer_dir.mkdir(parents=True, exist_ok=True)
    selected_molecules: list[Chem.Mol] = []
    for replica, row in enumerate(selected, 1):
        molecule = one_conformer(ensemble, int(row["conformer_id"]))
        metathesis.renumber_pdb_metadata(molecule)
        geometry = metathesis.stereo_geometry_rows(molecule)
        if len(geometry) != 2 or any(
            item["geometry_classification"] != "cis" for item in geometry
        ):
            raise ValueError(f"{state} conformer {replica} failed cis QC: {geometry}")
        minimum_distance = nonbonded_minimum_distance(molecule)
        if minimum_distance < 0.70:
            raise ValueError(
                f"{state} conformer {replica} has an atomic clash: "
                f"{minimum_distance:.3f} A"
            )
        replica_pdb = conformer_dir / f"replica_{replica:02d}.pdb"
        Chem.MolToPDBFile(molecule, str(replica_pdb), flavor=0)
        row["replica"] = replica
        row["minimum_nonbonded_distance_A"] = minimum_distance
        row["internal_double_bonds"] = geometry
        row["output_pdb"] = replica_pdb.relative_to(output).as_posix()
        selected_molecules.append(molecule)

    canonical = selected_molecules[0]
    source_to_output = metathesis.renumber_pdb_metadata(canonical)
    canonical_pdb = structures / f"{state}.pdb"
    canonical_mol2 = structures / f"{state}_antechamber.mol2"
    Chem.MolToPDBFile(canonical, str(canonical_pdb), flavor=0)
    metathesis.write_antechamber_mol2(canonical, canonical_mol2)
    active = active_mapping(
        canonical, q1_mapping, candidate.second_pair, source_to_output
    )
    schema = pair_schema(active, candidate.arm_by_vinyl)

    first_arms = [candidate.arm_by_vinyl[value] for value in candidate.first_pair]
    second_arms = [candidate.arm_by_vinyl[value] for value in candidate.second_pair]
    site_pairs = [
        [f"V{value}" for value in candidate.first_pair],
        [f"V{value}" for value in candidate.second_pair],
    ]
    audit = {
        "status": "PASS",
        "state": state,
        "system": candidate.system,
        "history_class": (
            "two_cross_sites_reusing_same_arm_pair"
            if candidate.history == "reuse"
            else "one_cross_site_plus_one_same_arm_site"
        ),
        "reacted_site_pairs": site_pairs,
        "reacted_site_arm_pairs": [first_arms, second_arms],
        "reacted_site_order_assigned": False,
        "construction_note": (
            "The validated q1 graph is used as an atom-mapping scaffold only; "
            "site A/site B do not imply reaction order."
        ),
        "selection_basis": candidate.selection_basis,
        "source_q1_pdb": q1_path.relative_to(root).as_posix(),
        "formula_q1": rdMolDescriptors.CalcMolFormula(q1),
        "formula_q2": rdMolDescriptors.CalcMolFormula(canonical),
        "formula_delta_q1_to_q2": formula_delta,
        "atom_count_q2": canonical.GetNumAtoms(),
        "q1_cycle_rank": q1_cycle_rank,
        "q2_cycle_rank": q2_cycle_rank,
        "adds_one_graph_cycle": True,
        "internal_cis_double_bond_count": 2,
        "active_vinyls": [f"V{vinyl}" for vinyl in active],
        "active_vinyl_count": len(active),
        "active_pair_count": len(schema),
        "active_vinyl_mapping_product_pdb_1based": {
            f"V{vinyl}": list(atoms) for vinyl, atoms in active.items()
        },
        "etkdg_seed": seed,
        "generated_conformer_count": len(all_rows),
        "requested_conformer_count": conformer_count,
        "selected_replica_conformer_count": len(selected),
        "minimum_selection_rms_A": minimum_rms_A,
        "mmff_role": "starting-geometry screening only; not a reaction barrier",
        "selected_conformers": selected,
        "canonical_pdb": canonical_pdb.relative_to(output).as_posix(),
        "explicit_bond_mol2": canonical_mol2.relative_to(output).as_posix(),
        "initial_conformer_dir": conformer_dir.relative_to(output).as_posix(),
    }
    manifest = {
        "state": state,
        "stage": "q2",
        "system": candidate.system,
        "branch": "reuse" if candidate.history == "reuse" else "mixed_same_cross",
        "source_pdb": canonical_pdb.relative_to(output).as_posix(),
        "net_charge": 0,
        "active_vinyl_count": len(active),
        "active_pair_count": len(schema),
        "vinyl_mapping_json": json.dumps(
            {f"V{key}": list(value) for key, value in active.items()},
            separators=(",", ":"),
        ),
        "pair_schema_json": json.dumps(
            schema, ensure_ascii=False, separators=(",", ":")
        ),
        "output_dir": f"gromacs_dcm/runs/{state}",
    }
    return audit, manifest


def write_readme(output: Path, audits: Sequence[dict[str, object]]) -> None:
    states = [str(audit["state"]) for audit in audits]
    state_args = " ".join(states)
    lines = [
        "# A4 q2 missing-history MD-ready bundle",
        "",
        "This bundle contains four chemically audited q2 graphs: C5/C8 reuse and C5/C8 mixed same+cross. The two reacted sites are unordered; labels only preserve atom mapping.",
        "",
        "Each state has one canonical PDB/MOL2 pair and multiple ETKDGv3/MMFF94 starting conformers. MMFF is used only for initial-geometry screening. The intended production protocol remains explicit CH2Cl2, GAFF2/AM1-BCC, 313.15 K, and independent unbiased MD replicas.",
        "",
        "## Remote parameterization",
        "",
        "```bash",
        f"conda run --no-capture-output -n chem_dl python -m dflow_explicit_dcm.cli submit-amber {state_args} \\",
        f"  --bundle {output} \\",
        "  --record results/a4_q2_missing_md_ready/amber_submission.json",
        "```",
        "",
        "After the Amber job succeeds, use `prepare-local`, then `submit-gromacs`. Start with 4 replicas/state and 2 ns/replica; use the resulting NAC diagnostics to decide whether extension is needed.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output", type=Path, default=root / "results/a4_q2_missing_md_ready"
    )
    parser.add_argument("--conformers", type=int, default=32)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--minimum-rms-A", type=float, default=1.0)
    parser.add_argument("--max-iters", type=int, default=5000)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="refresh NAC-export compatibility tables without rebuilding conformers",
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
    if (
        args.conformers < args.replicas
        or args.replicas < 1
        or args.minimum_rms_A < 0
        or args.max_iters < 1
    ):
        raise SystemExit("invalid conformer/replica/RMS/iteration settings")
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copy_protocol_inputs(root, output)
    candidates = [
        candidate
        for system in args.systems
        for candidate in review.select_candidates(root, system, args.q1_statistics)
    ]
    if args.metadata_only:
        manifest_path = output / "gromacs_dcm/state_manifest.csv"
        with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
            manifests = list(csv.DictReader(handle))
        write_export_compatibility_metadata(root, output, candidates, manifests)
        print(f"metadata_refreshed={output / 'results/react2'}")
        return
    audits: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for candidate in candidates:
        print(f"Building {physical_state(candidate)} ...", flush=True)
        audit, manifest = build_candidate(
            root,
            output,
            candidate,
            conformer_count=args.conformers,
            replicas=args.replicas,
            minimum_rms_A=args.minimum_rms_A,
            max_iters=args.max_iters,
        )
        audits.append(audit)
        manifests.append(manifest)
        audit_dir = output / "audits"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / f"{audit['state']}.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    write_manifest(output / "gromacs_dcm/state_manifest.csv", manifests)
    write_export_compatibility_metadata(root, output, candidates, manifests)
    summary = {
        "status": "PASS",
        "reacted_site_order_assigned": False,
        "states": [audit["state"] for audit in audits],
        "replica_conformers_per_state": args.replicas,
        "audits": [f"audits/{audit['state']}.json" for audit in audits],
    }
    (output / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_readme(output, audits)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
