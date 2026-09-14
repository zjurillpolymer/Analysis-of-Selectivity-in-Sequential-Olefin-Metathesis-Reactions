#!/usr/bin/env python3
"""Build MD-ready structures for both B6 q3 branches from q2 trajectories.

For each q3 branch and each independent q2 replica, five well-separated
near-contact frames are screened.  The exact third metathesis edit removes
C2H4 and creates a Z double bond.  Restrained then unrestrained MMFF94
minimization is used only to regularize the new bond geometry.  One passing
conformer is retained from every source replica, preserving three independent
starting basins for subsequent GROMACS runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

import build_b6_q1_md_structure as frame_tools
import build_q2_branch_structures as metathesis


Q2_STATE = "B6_p_C6_q2_V3_V6_V11_V12_cis"
EXPECTED_Q2_ACTIVE = (1, 2, 4, 5, 7, 8, 9, 10)


@dataclass(frozen=True)
class Branch:
    label: str
    consumed: tuple[int, int]
    role: str

    @property
    def state(self) -> str:
        left, right = self.consumed
        return f"B6_p_C6_q3_V3_V6_V11_V12_V{left}_V{right}_cis"

    @property
    def active(self) -> tuple[int, ...]:
        return tuple(vinyl for vinyl in EXPECTED_Q2_ACTIVE if vinyl not in self.consumed)


BRANCHES = (
    Branch("V8_V10", (8, 10), "primary_third_V8_V10_cis"),
    Branch("V7_V9", (7, 9), "sensitivity_third_V7_V9_cis"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--q2-workspace",
        default=(
            "results/b6_q2_v3_v6_v11_v12_50ns_download/tmp/output/var/folders/2r/"
            "923m0n5527d9hvtrnwldp3yc0000gn/T/b6-gromacs-upload-0l4cbbg1/workspace"
        ),
    )
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--candidate-frames-per-replica", type=int, default=5)
    parser.add_argument("--minimum-time-ps", type=float, default=5000.0)
    parser.add_argument("--minimum-separation-ps", type=float, default=500.0)
    parser.add_argument("--max-iters", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    return parser


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def q2_manifest_row(workspace: Path) -> dict[str, str]:
    rows = read_rows(workspace / "gromacs_dcm/state_manifest.csv")
    return next(row for row in rows if row["state"] == Q2_STATE)


def vinyl_mapping(row: dict[str, str]) -> dict[int, tuple[int, int]]:
    return {
        int(label.removeprefix("V")): tuple(int(value) for value in serials)
        for label, serials in json.loads(row["vinyl_mapping_json"]).items()
    }


def select_candidate_frames(
    distances: Path,
    consumed: tuple[int, int],
    count: int,
    minimum_time_ps: float,
    minimum_separation_ps: float,
) -> tuple[str, list[dict[str, float | int]]]:
    rows = read_rows(distances)
    if not rows:
        raise ValueError(f"empty distance table: {distances}")
    prefix = f"V{consumed[0]}_V{consumed[1]}_"
    columns = [name for name in rows[0] if name.startswith(prefix)]
    if len(columns) != 1:
        raise ValueError(f"expected one {prefix} distance column, found {columns}")
    column = columns[0]
    ranked = sorted(
        (
            {
                "frame": int(row["frame"]),
                "time_ps": float(row["absolute_time_ps"]),
                "distance_A": float(row[column]),
            }
            for row in rows
            if float(row["absolute_time_ps"]) >= minimum_time_ps
        ),
        key=lambda row: (float(row["distance_A"]), int(row["frame"])),
    )
    selected: list[dict[str, float | int]] = []
    for row in ranked:
        if all(
            abs(float(row["time_ps"]) - float(chosen["time_ps"]))
            >= minimum_separation_ps
            for chosen in selected
        ):
            selected.append(row)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(
            f"only {len(selected)} {prefix} frames satisfy the selection constraints"
        )
    return column, selected


def source_index(molecule: Chem.Mol, serial: int) -> int:
    return metathesis.atom_index_by_source_serial(molecule, serial)


def retained_serials(
    molecule: Chem.Mol,
    mapping: dict[int, tuple[int, int]],
    consumed: tuple[int, int],
) -> tuple[int, int]:
    result: list[int] = []
    for vinyl in consumed:
        _, retained = metathesis.terminal_and_retained_indices(molecule, mapping[vinyl])
        result.append(molecule.GetAtomWithIdx(retained).GetIntProp("source_q1_serial"))
    return result[0], result[1]


def new_bond_indices(
    molecule: Chem.Mol, retained: tuple[int, int]
) -> tuple[int, int, int, int]:
    left, right = (source_index(molecule, serial) for serial in retained)
    bond = molecule.GetBondBetweenAtoms(left, right)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        raise ValueError("new q3 double bond is absent")
    stereo_atoms = list(bond.GetStereoAtoms())
    if len(stereo_atoms) != 2:
        raise ValueError("new q3 double bond lacks stereo atoms")
    return stereo_atoms[0], left, right, stereo_atoms[1]


def close_and_optimize(
    molecule: Chem.Mol, retained: tuple[int, int], max_iters: int
) -> tuple[Chem.Mol, dict[str, object]]:
    if not AllChem.MMFFHasAllMoleculeParams(molecule):
        raise ValueError("MMFF94 parameters are incomplete")
    first, left, right, fourth = new_bond_indices(molecule, retained)
    properties = AllChem.MMFFGetMoleculeProperties(molecule, mmffVariant="MMFF94")
    if properties is None:
        raise ValueError("cannot create MMFF94 properties")

    restrained = AllChem.MMFFGetMoleculeForceField(
        molecule, properties, nonBondedThresh=100.0
    )
    restrained.MMFFAddDistanceConstraint(left, right, False, 1.32, 1.36, 250.0)
    restrained.MMFFAddTorsionConstraint(
        first, left, right, fourth, False, -20.0, 20.0, 100.0
    )
    restrained.Initialize()
    restrained_status = int(restrained.Minimize(maxIts=max_iters))

    relaxed = AllChem.MMFFGetMoleculeForceField(
        molecule, properties, nonBondedThresh=100.0
    )
    relaxed.Initialize()
    relaxed_status = int(relaxed.Minimize(maxIts=max_iters))
    energy = float(relaxed.CalcEnergy())
    geometry_rows = metathesis.stereo_geometry_rows(molecule)
    if not math.isfinite(energy):
        raise ValueError("MMFF94 returned a non-finite energy")
    if len(geometry_rows) != 3 or any(
        row["geometry_classification"] != "cis" for row in geometry_rows
    ):
        raise ValueError(f"expected three relaxed cis internal alkenes: {geometry_rows}")
    new_bond = next(
        row
        for row in geometry_rows
        if set(int(value) for value in str(row["bond_atom_indices"]).split(";"))
        == {left + 1, right + 1}
    )
    return molecule, {
        "restrained_status": restrained_status,
        "relaxed_status": relaxed_status,
        "mmff_energy_kcal_mol": energy,
        "third_bond_length_A": new_bond["bond_length_A"],
        "third_bond_dihedral_deg": new_bond["substituent_dihedral_deg"],
        "stereo_geometry": geometry_rows,
    }


def active_mapping(
    product: Chem.Mol,
    mapping: dict[int, tuple[int, int]],
    branch: Branch,
) -> dict[int, tuple[int, int]]:
    source_to_output = {
        atom.GetIntProp("source_q1_serial"): atom.GetIdx() + 1
        for atom in product.GetAtoms()
    }
    return {
        vinyl: tuple(source_to_output[serial] for serial in mapping[vinyl])
        for vinyl in branch.active
    }


def write_manifest_row(
    manifest_path: Path,
    q2_row: dict[str, str],
    branch: Branch,
    source_pdb: str,
    mapping: dict[int, tuple[int, int]],
) -> None:
    rows = read_rows(manifest_path)
    q2_schema = json.loads(q2_row["pair_schema_json"])
    active = set(mapping)
    schema = [
        pair
        for pair in q2_schema
        if int(pair["left"]) in active and int(pair["right"]) in active
    ]
    if len(mapping) != 6 or len(schema) != 15:
        raise ValueError(f"expected six q3 vinyls and 15 pairs: {mapping}, {schema}")
    row = {
        "state": branch.state,
        "stage": "q3",
        "system": "B6_p_C6",
        "branch": branch.role,
        "source_pdb": source_pdb,
        "net_charge": "0",
        "active_vinyl_count": str(len(mapping)),
        "active_pair_count": str(len(schema)),
        "vinyl_mapping_json": json.dumps(
            {f"V{key}": list(value) for key, value in mapping.items()},
            separators=(",", ":"),
        ),
        "pair_schema_json": json.dumps(
            schema, ensure_ascii=False, separators=(",", ":")
        ),
        "output_dir": f"gromacs_dcm/runs/{branch.state}",
    }
    by_state = {existing["state"]: existing for existing in rows}
    by_state[branch.state] = row
    ordered = [
        by_state[existing["state"]]
        for existing in rows
        if existing["state"] != branch.state
    ] + [row]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(ordered)


def build_branch(
    root: Path,
    q2_workspace: Path,
    bundle: Path,
    q2_row: dict[str, str],
    mapping: dict[int, tuple[int, int]],
    template: Chem.Mol,
    branch: Branch,
    replicas: int,
    candidate_count: int,
    minimum_time_ps: float,
    minimum_separation_ps: float,
    max_iters: int,
    force: bool,
    gmx: str,
) -> dict[str, object]:
    structures = bundle / "structures"
    result_dir = root / "results/explicit_dcm_b6_q3_build" / branch.label
    conformer_dir = structures / f"{branch.state}_initial_conformers"
    result_dir.mkdir(parents=True, exist_ok=True)
    conformer_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = structures / f"{branch.state}.pdb"
    output_mol2 = structures / f"{branch.state}_antechamber.mol2"
    audit_path = result_dir / "structure_qc.json"
    if output_pdb.exists() and output_mol2.exists() and audit_path.exists() and not force:
        return json.loads(audit_path.read_text(encoding="utf-8"))

    retained = retained_serials(template, mapping, branch.consumed)
    q2_run = q2_workspace / q2_row["output_dir"]
    screened: list[dict[str, object]] = []
    selected_products: dict[int, Chem.Mol] = {}
    selected_rows: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix=f"b6_q3_{branch.label}_frames_") as temporary:
        temporary_path = Path(temporary)
        for replica in range(1, replicas + 1):
            replica_dir = q2_run / f"replica_{replica:02d}"
            distance_column, candidates = select_candidate_frames(
                replica_dir / "vinyl_distances.csv",
                branch.consumed,
                candidate_count,
                minimum_time_ps,
                minimum_separation_ps,
            )
            index = (
                q2_run
                / "setup/initial_conformers"
                / f"replica_{replica:02d}"
                / "index.ndx"
            )
            products: dict[int, Chem.Mol] = {}
            replica_rows: list[dict[str, object]] = []
            for candidate_index, candidate in enumerate(candidates):
                frame_pdb = temporary_path / (
                    f"{branch.label}_replica_{replica:02d}_"
                    f"frame_{int(candidate['frame']):06d}.pdb"
                )
                frame_tools.extract_solute_frame(
                    gmx,
                    replica_dir / "production_50ns.tpr",
                    replica_dir / "production_2ns.xtc",
                    index,
                    float(candidate["time_ps"]),
                    frame_pdb,
                )
                base: dict[str, object] = {
                    "replica": replica,
                    "distance_column": distance_column,
                    **candidate,
                }
                try:
                    frame = frame_tools.molecule_at_coordinates(
                        template,
                        frame_tools.pdb_coordinates(frame_pdb, template.GetNumAtoms()),
                    )
                    product = metathesis.edit_second_metathesis(
                        frame, mapping, branch.consumed
                    )
                    product, qc = close_and_optimize(product, retained, max_iters)
                    products[candidate_index] = product
                    row = {**base, **qc, "passed": True, "error": ""}
                except Exception as error:
                    row = {**base, "passed": False, "error": str(error)}
                replica_rows.append(row)
                screened.append(row)
            passing = [
                index for index, row in enumerate(replica_rows) if bool(row["passed"])
            ]
            if not passing:
                raise RuntimeError(
                    f"no {branch.label} q3 conformer passed for replica {replica}: "
                    f"{replica_rows}"
                )
            selected_index = min(
                passing,
                key=lambda index: float(replica_rows[index]["mmff_energy_kcal_mol"]),
            )
            selected_product = products[selected_index]
            metathesis.renumber_pdb_metadata(selected_product)
            replica_pdb = conformer_dir / f"replica_{replica:02d}.pdb"
            Chem.MolToPDBFile(selected_product, str(replica_pdb), flavor=0)
            selected_products[replica] = selected_product
            selected_rows.append(
                {
                    **replica_rows[selected_index],
                    "output_pdb": str(replica_pdb.relative_to(root)),
                }
            )

    canonical_row = min(
        selected_rows, key=lambda row: float(row["mmff_energy_kcal_mol"])
    )
    canonical_replica = int(canonical_row["replica"])
    canonical = selected_products[canonical_replica]
    source_to_output = metathesis.renumber_pdb_metadata(canonical)
    Chem.MolToPDBFile(canonical, str(output_pdb), flavor=0)
    metathesis.write_antechamber_mol2(canonical, output_mol2)

    formula = rdMolDescriptors.CalcMolFormula(canonical)
    if canonical.GetNumAtoms() != 426 or formula != "C180H222O24":
        raise ValueError(
            f"unexpected {branch.label} q3 atoms/formula: "
            f"{canonical.GetNumAtoms()}, {formula}"
        )
    cycle_q2 = template.GetNumBonds() - template.GetNumAtoms() + 1
    cycle_q3 = canonical.GetNumBonds() - canonical.GetNumAtoms() + 1
    if cycle_q3 != cycle_q2 + 1:
        raise ValueError(f"{branch.label} q3 did not add exactly one graph cycle")
    mapping_q3 = active_mapping(canonical, mapping, branch)
    relative_source = output_pdb.relative_to(bundle).as_posix()
    write_manifest_row(
        bundle / "gromacs_dcm/state_manifest.csv",
        q2_row,
        branch,
        relative_source,
        mapping_q3,
    )

    audit = {
        "status": "PASS",
        "state": branch.state,
        "branch_role": branch.role,
        "method": (
            "three independent q2 replicas; five separated near-contact frames each; "
            "exact C2H4 loss; restrained/unrestrained MMFF94"
        ),
        "source_q2_state": Q2_STATE,
        "q2_workspace": str(q2_workspace.relative_to(root)),
        "q2_formula": "C182H226O24",
        "q3_formula": formula,
        "formula_delta": {"C": 2, "H": 4},
        "q2_atom_count": template.GetNumAtoms(),
        "q3_atom_count": canonical.GetNumAtoms(),
        "adds_one_graph_cycle": True,
        "reaction_history": [["V3", "V6"], ["V11", "V12"], [f"V{branch.consumed[0]}", f"V{branch.consumed[1]}"]],
        "third_double_bond_q2_source_serials": list(retained),
        "third_double_bond_q3_output_serials": [
            source_to_output[retained[0]],
            source_to_output[retained[1]],
        ],
        "selected_conformers": selected_rows,
        "canonical_replica": canonical_replica,
        "canonical_mmff_energy_kcal_mol": canonical_row["mmff_energy_kcal_mol"],
        "screened_candidate_count": len(screened),
        "passing_candidate_count": sum(bool(row["passed"]) for row in screened),
        "minimum_time_ps": minimum_time_ps,
        "minimum_candidate_separation_ps": minimum_separation_ps,
        "active_vinyls": [f"V{vinyl}" for vinyl in mapping_q3],
        "active_vinyl_count": len(mapping_q3),
        "active_pair_count": len(mapping_q3) * (len(mapping_q3) - 1) // 2,
        "active_vinyl_mapping_product_pdb_1based": {
            f"V{key}": list(value) for key, value in mapping_q3.items()
        },
        "output_pdb": str(output_pdb.relative_to(root)),
        "output_explicit_bond_mol2": str(output_mol2.relative_to(root)),
        "replica_initial_conformer_dir": str(conformer_dir.relative_to(root)),
        "screened_candidates": screened,
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.replicas < 1
        or args.candidate_frames_per_replica < 1
        or args.minimum_time_ps < 0
        or args.minimum_separation_ps < 0
        or args.max_iters < 1
    ):
        raise SystemExit("invalid replica, candidate, time, separation, or iteration value")
    root = Path(args.root).resolve()
    q2_workspace = (root / args.q2_workspace).resolve()
    bundle = root / "gromacs_dcm_server_bundle"
    gmx = shutil.which("gmx")
    if gmx is None:
        raise RuntimeError("gmx is required to extract q2 candidate frames")
    q2_row = q2_manifest_row(q2_workspace)
    mapping = vinyl_mapping(q2_row)
    if tuple(mapping) != EXPECTED_Q2_ACTIVE:
        raise ValueError(f"unexpected q2 mapping: {mapping}")
    q2_source = bundle / q2_row["source_pdb"]
    template = metathesis.load_rdkit_pdb(q2_source)
    if template.GetNumAtoms() != 432 or rdMolDescriptors.CalcMolFormula(template) != "C182H226O24":
        raise ValueError("unexpected q2 source structure")

    audits = [
        build_branch(
            root,
            q2_workspace,
            bundle,
            q2_row,
            mapping,
            template,
            branch,
            args.replicas,
            args.candidate_frames_per_replica,
            args.minimum_time_ps,
            args.minimum_separation_ps,
            args.max_iters,
            args.force,
            gmx,
        )
        for branch in BRANCHES
    ]
    summary = {
        "status": "PASS" if all(audit["status"] == "PASS" for audit in audits) else "FAIL",
        "source_state": Q2_STATE,
        "branches": audits,
        "next_step": "prepare two sequential AmberTools parameterization jobs",
    }
    summary_path = root / "results/explicit_dcm_b6_q3_build/summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
