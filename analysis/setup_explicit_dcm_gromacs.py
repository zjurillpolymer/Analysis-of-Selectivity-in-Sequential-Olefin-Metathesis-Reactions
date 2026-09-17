#!/usr/bin/env python3
"""Parameterize, pack, and validate explicit-DCM GROMACS systems.

This script is intended for the server ``md_tools`` environment.  Solutes use
AmberTools/GAFF2 with AM1-BCC charges.  Explicit DCM defaults to the published
CH2Cl2-DC liquid model, while the original GAFF2/AM1-BCC DCM remains available
as a negative control.  Packmol constructs the boxes, ParmEd converts Amber to
GROMACS, and ``gmx grompp`` performs the final topology check.  The pure-
solvent box must pass the density gate before any solute trajectory is run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DCM_MOLAR_MASS_G_MOL = 84.927
AVOGADRO_NM3_FACTOR = 602.214076  # N_A * 1e-21 cm3/nm3
DEFAULT_INITIAL_DENSITY_G_CM3 = 1.18
DEFAULT_PADDING_NM = 2.0
PURE_DCM_BOX_NM = 5.0
DEFAULT_DCM_MODEL = "ch2cl2_dc"
DCM_MODELS = ("ch2cl2_dc", "gaff2_am1bcc")
SUPPORTED_REACTION_STAGES = tuple(f"q{index}" for index in range(7))
DEFAULT_SQM_GRMS_TOL_KCAL_MOL_A = 0.05
LEGACY_ANTECHAMBER_GRMS_TOL_KCAL_MOL_A = 0.0002
EXPLICIT_BOND_ANTECHAMBER_JUDGE_MODE = 5


@dataclass(frozen=True)
class PDBAtom:
    serial: int
    name: str
    element: str
    position_A: tuple[float, float, float]


@dataclass(frozen=True)
class SetupTask:
    root: str
    row: dict[str, str]
    force: bool
    padding_nm: float
    density_g_cm3: float
    dcm_model: str
    sqm_grms_tol: float
    defer_grompp: bool


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_pdb(path: Path) -> list[PDBAtom]:
    atoms: list[PDBAtom] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        element = line[76:78].strip().upper()
        if not element:
            element = "".join(character for character in line[12:16] if character.isalpha()).upper()
            if element.startswith("CL"):
                element = "CL"
            else:
                element = element[:1]
        atoms.append(
            PDBAtom(
                serial=int(line[6:11]),
                name=line[12:16].strip(),
                element=element,
                position_A=(float(line[30:38]), float(line[38:46]), float(line[46:54])),
            )
        )
    if not atoms:
        raise ValueError(f"no atoms in {path}")
    return atoms


def normalize_residue_name(source: Path, destination: Path, residue_name: str) -> None:
    lines: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            line = f"{line[:17]}{residue_name:>3}{line[20:]}"
        lines.append(line)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def initial_conformer_paths(source_pdb: Path) -> list[Path]:
    """Return a consecutive replica_XX conformer series next to a source PDB."""
    directory = source_pdb.with_name(f"{source_pdb.stem}_initial_conformers")
    if not directory.is_dir():
        return []
    paths = sorted(directory.glob("replica_*.pdb"))
    if not paths:
        raise ValueError(f"initial conformer directory is empty: {directory}")
    expected = [f"replica_{index:02d}.pdb" for index in range(1, len(paths) + 1)]
    observed = [path.name for path in paths]
    if observed != expected:
        raise ValueError(
            f"initial conformers must be consecutive replica_01..replica_N: {observed}"
        )
    return paths


def transplant_pdb_coordinates(
    coordinate_source: Path,
    parameterized_template: Path,
    destination: Path,
) -> None:
    """Put conformer coordinates into a parameterized PDB identity template.

    The q2 conformers share the canonical graph and atom order, while the
    Antechamber PDB supplies the unique GAFF/LEaP atom names.  Keeping those
    identities and changing coordinates only allows one AM1-BCC topology to be
    reused safely across independently packed replica starting structures.
    """
    source_atoms = read_pdb(coordinate_source)
    template_atoms = read_pdb(parameterized_template)
    if len(source_atoms) != len(template_atoms):
        raise ValueError(
            f"conformer/template atom-count mismatch: "
            f"{len(source_atoms)} != {len(template_atoms)}"
        )
    for index, (source_atom, template_atom) in enumerate(
        zip(source_atoms, template_atoms), start=1
    ):
        if source_atom.element != template_atom.element:
            raise ValueError(
                f"conformer/template element mismatch at atom {index}: "
                f"{source_atom.element} != {template_atom.element}"
            )

    template_lines = parameterized_template.read_text(encoding="utf-8").splitlines()
    atom_line_indices = [
        index
        for index, line in enumerate(template_lines)
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    if len(atom_line_indices) != len(source_atoms):
        raise ValueError(f"invalid parameterized PDB template: {parameterized_template}")
    for line_index, source_atom in zip(atom_line_indices, source_atoms):
        line = template_lines[line_index]
        x, y, z = source_atom.position_A
        template_lines[line_index] = (
            f"{line[:30]}{x:8.3f}{y:8.3f}{z:8.3f}{line[54:]}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(template_lines) + "\n", encoding="utf-8")


def mol2_atom_names(path: Path) -> list[str]:
    names: list[str] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
            continue
        if in_atoms and line.strip():
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"invalid MOL2 atom line in {path}: {line}")
            names.append(fields[1])
    if not names:
        raise ValueError(f"no MOL2 atom names in {path}")
    return names


def mol2_atom_identity_records(
    path: Path,
) -> list[tuple[int, str, str, tuple[float, float, float]]]:
    """Return serial, name, element, and coordinates from a MOL2 atom table."""
    records: list[tuple[int, str, str, tuple[float, float, float]]] = []
    in_atoms = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
            continue
        if not in_atoms or not line.strip():
            continue
        fields = line.split()
        if len(fields) < 6:
            raise ValueError(f"invalid MOL2 atom line in {path}: {line}")
        name = fields[1]
        letters = "".join(character for character in name if character.isalpha()).upper()
        element = (
            letters[:2]
            if letters.startswith(("CL", "BR"))
            else letters[:1]
        )
        if not element:
            raise ValueError(f"cannot infer MOL2 element from atom name: {name!r}")
        records.append(
            (
                int(fields[0]),
                name,
                element,
                (float(fields[2]), float(fields[3]), float(fields[4])),
            )
        )
    if not records:
        raise ValueError(f"no MOL2 atom records in {path}")
    return records


def synchronize_pdb_atom_names_from_mol2(mol2: Path, pdb: Path) -> None:
    """Make case-sensitive LEaP PDB names identical to its MOL2 template.

    Antechamber may convert a MOL2 chlorine name such as ``CL1`` to ``Cl1``
    when writing PDB.  LEaP matches residue-template atom names case
    sensitively, leaving every such atom untyped.  The MOL2 atom order is
    preserved by the MOL2-to-PDB conversion, so restore those canonical names
    in the PDB before Packmol replicates the molecule.
    """
    names = mol2_atom_names(mol2)
    lines = pdb.read_text(encoding="utf-8").splitlines()
    atom_lines = [
        index
        for index, line in enumerate(lines)
        if line.startswith(("ATOM  ", "HETATM"))
    ]
    if len(atom_lines) != len(names):
        raise ValueError(
            f"MOL2/PDB atom-count mismatch while synchronizing names: "
            f"{len(names)} != {len(atom_lines)}; {mol2}; {pdb}"
        )
    for line_index, name in zip(atom_lines, names):
        if not 1 <= len(name) <= 4:
            raise ValueError(f"PDB atom name is not 1-4 characters: {name!r}")
        line = lines[line_index]
        current_field = line[12:16]
        leading_spaces = min(
            len(current_field) - len(current_field.lstrip()),
            4 - len(name),
        )
        atom_field = (" " * leading_spaces + name).ljust(4)
        lines[line_index] = f"{line[:12]}{atom_field}{line[16:]}"
    pdb.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_command(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    stdin: str | None = None,
    stdin_path: Path | None = None,
) -> None:
    if stdin is not None and stdin_path is not None:
        raise ValueError("stdin text and stdin_path are mutually exclusive")
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        common = {
            "cwd": cwd,
            "text": True,
            "stdout": log,
            "stderr": subprocess.STDOUT,
            "check": False,
        }
        if stdin_path is not None:
            # Packmol's Fortran input parser rewinds unit 5.  A real seekable
            # file must therefore be attached to stdin; a PIPE can fail with
            # "Fortran runtime error: Illegal seek".
            with stdin_path.open("r", encoding="utf-8") as input_handle:
                result = subprocess.run(list(command), stdin=input_handle, **common)
        else:
            result = subprocess.run(list(command), input=stdin, **common)
        log.write(f"exit={result.returncode}; elapsed_s={time.perf_counter() - started:.1f}\n")
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}; see {log_path}")


def require_environment(
    *,
    require_gromacs: bool = True,
    require_packing: bool = True,
    require_packmol: bool = True,
) -> dict[str, str]:
    tools = {}
    executables = ["antechamber", "parmchk2"]
    if require_packing:
        executables.append("tleap")
        if require_packmol:
            executables.append("packmol")
        if require_gromacs:
            executables.append("gmx")
    for executable in executables:
        resolved = shutil.which(executable)
        if resolved is None:
            raise RuntimeError(f"required executable is not on PATH: {executable}")
        tools[executable] = resolved
    if require_packing:
        try:
            import parmed  # noqa: F401
        except ModuleNotFoundError as error:
            raise RuntimeError("ParmEd is missing from the active environment") from error
    return tools


def mol2_charge_and_count(path: Path) -> tuple[float, int]:
    charge = 0.0
    count = 0
    in_atoms = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
            continue
        if in_atoms and line.strip():
            fields = line.split()
            if len(fields) < 9:
                raise ValueError(f"invalid MOL2 atom line in {path}: {line}")
            charge += float(fields[-1])
            count += 1
    if count == 0:
        raise ValueError(f"no MOL2 atoms in {path}")
    return charge, count


def mol2_charge_residual_limit(atom_count: int) -> float:
    """Return a conservative, size-aware AM1-BCC residual limit.

    Large-molecule AmberTools charge tables accumulate upstream decimal
    rounding before the final six-decimal MOL2 is written.  A fixed 0.010 e
    cutoff therefore rejects some neutral, several-hundred-atom molecules.
    The size-aware term permits at most 5e-5 e uniform correction per atom,
    while the 0.050 e absolute cap still rejects chemically material charge
    errors.  Small molecules retain the historical 0.010 e limit.
    """
    if atom_count < 1:
        raise ValueError(f"invalid MOL2 atom count: {atom_count}")
    return min(0.05, max(0.01, 5.0e-5 * atom_count))


def normalize_mol2_total_charge(
    path: Path,
    target_charge: int,
    max_rounding_residual: float | None = None,
) -> float:
    """Remove a small aggregate AM1-BCC residual while preserving symmetry.

    Antechamber's upstream charge tables and final MOL2 use finite decimal
    precision.  Across hundreds of atoms the aggregate can exceed 0.010 e even
    when ``-nc`` requested an integral charge.  We permit only a very small
    size-scaled correction, spread it uniformly over all atoms (preserving
    charge differences and symmetry), round to six decimals, and put only the
    sub-microelectron remainder on the last atom.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    atom_lines: list[int] = []
    charges: list[float] = []
    in_atoms = False
    for index, line in enumerate(lines):
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
            continue
        if in_atoms and line.strip():
            fields = line.split()
            if len(fields) < 9:
                raise ValueError(f"invalid MOL2 atom line in {path}: {line}")
            atom_lines.append(index)
            charges.append(float(fields[-1]))
    if not charges:
        raise ValueError(f"no MOL2 charges in {path}")
    residual = sum(charges) - float(target_charge)
    if abs(residual) <= 5.0e-7:
        return residual
    residual_limit = (
        mol2_charge_residual_limit(len(charges))
        if max_rounding_residual is None
        else max_rounding_residual
    )
    if residual_limit <= 0.0:
        raise ValueError(f"invalid MOL2 charge residual limit: {residual_limit}")
    if abs(residual) > residual_limit + 1.0e-12:
        raise ValueError(
            f"MOL2 charge residual {residual:+.8f} e is too large to treat as "
            f"aggregate AM1-BCC rounding for {len(charges)} atoms "
            f"(limit {residual_limit:.5f} e; per-atom correction "
            f"{abs(residual) / len(charges):.8f} e): {path}"
        )
    corrected = [round(charge - residual / len(charges), 6) for charge in charges]
    corrected[-1] = round(
        corrected[-1] + float(target_charge) - sum(corrected), 6
    )
    for line_index, charge in zip(atom_lines, corrected):
        prefix = lines[line_index].rsplit(None, 1)[0]
        lines[line_index] = f"{prefix} {charge:.6f}"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    final_charge, _ = mol2_charge_and_count(path)
    if abs(final_charge - target_charge) > 5.0e-7:
        raise ValueError(
            f"failed to normalize MOL2 total charge: {final_charge:+.8f} e in {path}"
        )
    return residual


def sqm_extra_keywords(grms_tol: float) -> str:
    if not 0.0 < grms_tol <= 0.1:
        raise ValueError("SQM grms_tol must be in (0, 0.1] kcal/(mol*A)")
    return f"grms_tol={grms_tol:.8g}"


def cached_sqm_tolerance_satisfies(
    metadata: dict[str, object], requested_tolerance: float
) -> bool:
    """Return whether a cached SQM result is at least as strict as requested."""
    try:
        cached_tolerance = float(
            metadata.get(
                "sqm_grms_tol_kcal_mol_A",
                LEGACY_ANTECHAMBER_GRMS_TOL_KCAL_MOL_A,
            )
        )
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(cached_tolerance)
        and 0.0 < cached_tolerance <= requested_tolerance
    )


def completed_sqm_output(path: Path) -> bool:
    return path.is_file() and "Calculation Completed" in path.read_text(
        encoding="utf-8", errors="replace"
    )


def parse_completed_sqm(
    path: Path,
) -> tuple[list[float], list[tuple[str, float, float, float]]]:
    """Read final AM1 Mulliken charges and optimized coordinates from SQM."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    charge_headers = [
        index
        for index, line in enumerate(lines)
        if "Atom" in line and "Element" in line and "Mulliken Charge" in line
    ]
    if not charge_headers:
        raise ValueError(f"no Mulliken charge table in completed SQM output: {path}")
    charges: list[float] = []
    elements: list[str] = []
    for line in lines[charge_headers[-1] + 1 :]:
        if "Total Mulliken Charge" in line:
            break
        fields = line.split()
        if len(fields) >= 3 and fields[0].isdigit():
            elements.append(fields[-2].upper())
            charges.append(float(fields[-1].replace("D", "E")))
    if not charges:
        raise ValueError(f"empty Mulliken charge table in {path}")

    final_markers = [
        index for index, line in enumerate(lines) if line.strip() == "Final Structure"
    ]
    if not final_markers:
        raise ValueError(f"no Final Structure section in completed SQM output: {path}")
    coordinate_pattern = re.compile(
        r"^\s*QMMM:\s+(\d+)\s+\d+\s+([A-Za-z]+)\s+"
        r"([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s*$"
    )
    coordinates: list[tuple[str, float, float, float]] = []
    for line in lines[final_markers[-1] + 1 :]:
        match = coordinate_pattern.match(line)
        if match is None:
            continue
        serial = int(match.group(1))
        if serial != len(coordinates) + 1:
            raise ValueError(f"nonsequential final SQM coordinates in {path}: {serial}")
        coordinates.append(
            (
                match.group(2).upper(),
                *(float(match.group(index).replace("D", "E")) for index in (3, 4, 5)),
            )
        )
    if len(coordinates) != len(charges):
        raise ValueError(
            f"SQM charge/coordinate count mismatch: {len(charges)} != "
            f"{len(coordinates)}; {path}"
        )
    if any(left != right[0] for left, right in zip(elements, coordinates)):
        raise ValueError(f"SQM charge/coordinate element mismatch in {path}")
    # SQM prints individual Mulliken charges to only three decimal places.
    # Across >500 atoms their text-rounding residual can reach a few
    # millielectrons even though SQM reports an integral total charge.
    if abs(sum(charges) - round(sum(charges))) > 1.0e-2 + 1.0e-12:
        raise ValueError(f"SQM Mulliken charges do not sum near an integer: {sum(charges)}")
    return charges, coordinates


def ac_atom_records(path: Path) -> list[tuple[int, str, float, str]]:
    """Return serial, name, charge, and GAFF2 type from fixed-width AC atoms."""
    records: list[tuple[int, str, float, str]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM"):
            continue
        if len(line) < 68:
            raise ValueError(f"short AC atom line in {path}: {line}")
        records.append(
            (
                int(line[4:11]),
                line[11:17].strip(),
                float(line[55:67]),
                line[67:].strip(),
            )
        )
    if not records or [record[0] for record in records] != list(
        range(1, len(records) + 1)
    ):
        raise ValueError(f"AC atoms are absent or nonsequential: {path}")
    return records


def write_ac_charges(source: Path, destination: Path, charges: Sequence[float]) -> None:
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    atom_indices = [index for index, line in enumerate(lines) if line.startswith("ATOM")]
    if len(atom_indices) != len(charges):
        raise ValueError(
            f"AC/SQM atom-count mismatch: {len(atom_indices)} != {len(charges)}"
        )
    for line_index, charge in zip(atom_indices, charges):
        line = lines[line_index]
        if len(line) < 68:
            raise ValueError(f"short AC atom line in {source}: {line}")
        lines[line_index] = f"{line[:55]}{charge:12.6f}{line[67:]}"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recovered_mol2(
    seed_mol2: Path,
    bcc_ac: Path,
    sqm_coordinates: Sequence[tuple[str, float, float, float]],
    destination: Path,
) -> None:
    """Combine seed bond orders, GAFF2 types, BCC charges, and SQM geometry."""
    records = ac_atom_records(bcc_ac)
    if len(records) != len(sqm_coordinates):
        raise ValueError("BCC AC/SQM coordinate atom-count mismatch")
    lines = seed_mol2.read_text(encoding="utf-8").splitlines()
    atom_indices: list[int] = []
    in_atoms = False
    for index, line in enumerate(lines):
        if line.startswith("@<TRIPOS>"):
            in_atoms = line.strip() == "@<TRIPOS>ATOM"
            continue
        if in_atoms and line.strip():
            atom_indices.append(index)
    if len(atom_indices) != len(records):
        raise ValueError("seed MOL2/BCC AC atom-count mismatch")

    # The explicit-bond seed is only a connectivity template and commonly
    # declares NO_CHARGES.  Recovery writes AM1-BCC values into the atom table,
    # so advertise them consistently; MDAnalysis rejects nonzero charges under
    # a NO_CHARGES header.
    try:
        molecule_start = lines.index("@<TRIPOS>MOLECULE")
    except ValueError as error:
        raise ValueError(f"seed MOL2 has no MOLECULE section: {seed_mol2}") from error
    molecule_fields: list[int] = []
    for index in range(molecule_start + 1, len(lines)):
        if lines[index].startswith("@<TRIPOS>"):
            break
        if lines[index].strip():
            molecule_fields.append(index)
    if len(molecule_fields) < 4:
        raise ValueError(f"incomplete MOL2 MOLECULE header: {seed_mol2}")
    lines[molecule_fields[3]] = "USER_CHARGES"

    for line_index, record, coordinate in zip(atom_indices, records, sqm_coordinates):
        serial, ac_name, charge, atom_type = record
        fields = lines[line_index].split()
        if int(fields[0]) != serial or fields[1] != ac_name:
            raise ValueError(
                f"seed/AC atom identity mismatch at {serial}: {fields[1]} != {ac_name}"
            )
        element, x, y, z = coordinate
        if not ac_name.upper().startswith(element):
            raise ValueError(f"seed/SQM element mismatch at {serial}: {ac_name} != {element}")
        lines[line_index] = (
            f"{serial:7d} {ac_name:<8} {x:10.4f} {y:10.4f} {z:10.4f} "
            f"{atom_type:<8} {int(fields[6]):d} {fields[7]:<8} {charge:.6f}"
        )
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recover_completed_sqm_am1bcc(
    seed_mol2: Path,
    output_dir: Path,
    output_mol2: Path,
    log: Path,
) -> str | None:
    """Recover a completed >500-atom SQM job after Antechamber exits 139."""
    sqm_candidates = (
        output_dir / "sqm.completed.out",
        output_dir / "sqm.out",
    )
    ac_candidates = (
        output_dir / "ANTECHAMBER_AC.completed.AC",
        output_dir / "ANTECHAMBER_AC.AC",
    )
    sqm_path = next((path for path in sqm_candidates if completed_sqm_output(path)), None)
    ac_path = next((path for path in ac_candidates if path.is_file()), None)
    if sqm_path is None or ac_path is None:
        return None
    charges, coordinates = parse_completed_sqm(sqm_path)
    if len(ac_atom_records(ac_path)) != len(charges):
        raise ValueError("completed SQM/AC atom-count mismatch")
    mulliken_ac = output_dir / "sqm_mulliken.ac"
    bcc_ac = output_dir / "sqm_am1bcc.ac"
    write_ac_charges(ac_path, mulliken_ac, charges)
    run_command(
        [
            "am1bcc",
            "-i", str(mulliken_ac),
            "-o", str(bcc_ac),
            "-f", "ac",
            "-j", "0",
            "-s", "2",
        ],
        output_dir,
        log,
    )
    write_recovered_mol2(seed_mol2, bcc_ac, coordinates, output_mol2)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(
            f"Recovered completed SQM output without recomputation: {sqm_path}; "
            "standalone am1bcc -j 0.\n"
        )
    return "completed_sqm_plus_standalone_am1bcc"


def parameterize_molecule(
    source_pdb: Path,
    output_dir: Path,
    residue_name: str,
    net_charge: int,
    force: bool,
    sqm_grms_tol: float = DEFAULT_SQM_GRMS_TOL_KCAL_MOL_A,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mol2 = output_dir / f"{residue_name.lower()}.mol2"
    frcmod = output_dir / f"{residue_name.lower()}.frcmod"
    parameterized_pdb_raw = output_dir / f"{residue_name.lower()}_raw.pdb"
    parameterized_pdb = output_dir / f"{residue_name.lower()}.pdb"
    bonded_input_mol2 = source_pdb.with_name(
        f"{source_pdb.stem}_antechamber.mol2"
    )
    if bonded_input_mol2.is_file():
        antechamber_input = bonded_input_mol2
        antechamber_input_format = "mol2"
        metadata_input_format = "mol2_explicit_pdb_bonds"
    else:
        antechamber_input = source_pdb
        antechamber_input_format = "pdb"
        metadata_input_format = "pdb_legacy_partial_bond_judgement"
    parameterization_metadata = output_dir / "parameterization_metadata.json"
    log = output_dir / "parameterization.log"
    cached_metadata: dict[str, object] = {}
    if parameterization_metadata.is_file():
        cached_metadata = json.loads(parameterization_metadata.read_text(encoding="utf-8"))
    cache_is_current = (
        mol2.is_file()
        and frcmod.is_file()
        and parameterized_pdb.is_file()
        and cached_metadata.get("input_format") == metadata_input_format
        and cached_metadata.get("antechamber_judge_mode")
        == EXPLICIT_BOND_ANTECHAMBER_JUDGE_MODE
        and cached_sqm_tolerance_satisfies(cached_metadata, sqm_grms_tol)
    )
    if force or not cache_is_current:
        charge_recovery = None
        if bonded_input_mol2.is_file() and not force:
            charge_recovery = recover_completed_sqm_am1bcc(
                bonded_input_mol2,
                output_dir,
                mol2,
                log,
            )
        if charge_recovery is None:
            try:
                run_command(
                    [
                        "antechamber",
                        "-i", str(antechamber_input),
                        "-fi", antechamber_input_format,
                        "-o", str(mol2),
                        "-fo", "mol2",
                        "-at", "gaff2",
                        "-c", "bcc",
                        "-nc", str(net_charge),
                        "-rn", residue_name,
                        # The MOL2 seed carries explicit single/double/triple bond
                        # orders, so partial judgement is both correct and tractable.
                        "-j", str(EXPLICIT_BOND_ANTECHAMBER_JUDGE_MODE),
                        "-ek", sqm_extra_keywords(sqm_grms_tol),
                        "-pf", "y",
                        "-s", "2",
                    ],
                    output_dir,
                    log,
                )
            except RuntimeError:
                if not bonded_input_mol2.is_file():
                    raise
                charge_recovery = recover_completed_sqm_am1bcc(
                    bonded_input_mol2,
                    output_dir,
                    mol2,
                    log,
                )
                if charge_recovery is None:
                    raise
        unnormalized_charge, unnormalized_atom_count = mol2_charge_and_count(mol2)
        charge_residual = normalize_mol2_total_charge(mol2, net_charge)
        charge_normalization = {
            "method": "uniform_integral_charge_normalization",
            "original_total_charge_e": unnormalized_charge,
            "target_total_charge_e": net_charge,
            "residual_e": charge_residual,
            "atom_count": unnormalized_atom_count,
            "per_atom_correction_e": -charge_residual / unnormalized_atom_count,
            "accepted_residual_limit_e": mol2_charge_residual_limit(
                unnormalized_atom_count
            ),
        }
        if abs(charge_residual) > 5.0e-7:
            with log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"Normalized aggregate AM1-BCC charge residual "
                    f"{charge_residual:+.8f} e to {net_charge:+d} e "
                    f"({-charge_residual / unnormalized_atom_count:+.8f} e/atom).\n"
                )
        run_command(
            [
                "parmchk2",
                "-i", str(mol2),
                "-f", "mol2",
                "-o", str(frcmod),
                "-s", "gaff2",
            ],
            output_dir,
            log,
        )
        run_command(
            [
                "antechamber",
                "-i", str(mol2),
                "-fi", "mol2",
                "-o", str(parameterized_pdb_raw),
                "-fo", "pdb",
                "-rn", residue_name,
                "-pf", "y",
            ],
            output_dir,
            log,
        )
        normalize_residue_name(parameterized_pdb_raw, parameterized_pdb, residue_name)
        parameterization_metadata.write_text(
            json.dumps(
                {
                    "input_format": metadata_input_format,
                    "antechamber_judge_mode": EXPLICIT_BOND_ANTECHAMBER_JUDGE_MODE,
                    "sqm_grms_tol_kcal_mol_A": sqm_grms_tol,
                    "net_charge": net_charge,
                    "charge_recovery": charge_recovery,
                    "charge_normalization": charge_normalization,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    normalize_mol2_total_charge(mol2, net_charge)
    # Apply this even for cached parameters so older failed setup attempts are
    # repaired without repeating the expensive AM1-BCC calculation.
    synchronize_pdb_atom_names_from_mol2(mol2, parameterized_pdb)
    charge, count = mol2_charge_and_count(mol2)
    if abs(charge - net_charge) > 5.0e-7:
        raise ValueError(
            f"{residue_name} MOL2 charge is {charge:.8f}, expected {net_charge}; {mol2}"
        )
    if count != len(read_pdb(source_pdb)):
        raise ValueError(f"atom count changed during parameterization of {source_pdb}")
    current_metadata = json.loads(parameterization_metadata.read_text(encoding="utf-8"))
    return {
        "mol2": mol2,
        "frcmod": frcmod,
        "pdb": parameterized_pdb,
        "charge": charge,
        "atom_count": count,
        "sqm_grms_tol_kcal_mol_A": sqm_grms_tol,
        "charge_recovery": current_metadata.get("charge_recovery"),
        "charge_normalization": current_metadata.get("charge_normalization"),
        "explicit_input_mol2": bonded_input_mol2 if bonded_input_mol2.is_file() else None,
    }


def coordinate_atom_map(source: Path, parameterized: Path, tolerance_A: float = 0.20) -> dict[int, int]:
    original = read_pdb(source)
    output = read_pdb(parameterized)
    if len(original) != len(output):
        raise ValueError("cannot atom-map PDBs with different atom counts")

    # Antechamber normally preserves the explicit-MOL2 atom order and names.
    # In that case atom identity is stronger evidence than proximity: SQM may
    # legitimately optimize a large molecule by more than ``tolerance_A``.
    # Require unique names and an exact ordered name/element match before using
    # this path; otherwise retain the conservative coordinate-based fallback.
    original_identities = [(atom.name.upper(), atom.element) for atom in original]
    output_identities = [(atom.name.upper(), atom.element) for atom in output]
    names_are_unique = (
        len({identity[0] for identity in original_identities}) == len(original)
        and len({identity[0] for identity in output_identities}) == len(output)
    )
    if names_are_unique and original_identities == output_identities:
        return {atom.serial: index + 1 for index, atom in enumerate(original)}

    unused = set(range(len(output)))
    mapping: dict[int, int] = {}
    for atom in original:
        candidates = []
        for index in unused:
            candidate = output[index]
            if candidate.element != atom.element:
                continue
            distance = math.dist(atom.position_A, candidate.position_A)
            candidates.append((distance, index))
        if not candidates:
            raise ValueError(f"no element-compatible parameterized atom for source atom {atom.serial}")
        distance, index = min(candidates)
        if distance > tolerance_A:
            raise ValueError(
                f"source atom {atom.serial} moved {distance:.3f} A during parameterization; "
                "serial mapping is unsafe"
            )
        unused.remove(index)
        mapping[atom.serial] = index + 1
    return mapping


def explicit_mol2_serial_atom_map(
    source: Path,
    explicit_mol2: Path,
    parameterized: Path,
    source_coordinate_tolerance_A: float = 0.02,
) -> dict[int, int]:
    """Prove a serial map through the explicit MOL2 used by Antechamber.

    Large SQM optimizations invalidate proximity mapping.  The explicit MOL2,
    however, was generated directly from the source PDB and carries stable,
    numbered atom identities that are preserved by the recovery path.
    """
    original = read_pdb(source)
    seed = mol2_atom_identity_records(explicit_mol2)
    output = read_pdb(parameterized)
    if not (len(original) == len(seed) == len(output)):
        raise ValueError("source/explicit-MOL2/parameterized atom-count mismatch")
    seed_names = [record[1].upper() for record in seed]
    if len(set(seed_names)) != len(seed_names):
        raise ValueError("explicit MOL2 atom names are not unique")
    for index, (source_atom, seed_atom, output_atom) in enumerate(
        zip(original, seed, output),
        start=1,
    ):
        seed_serial, seed_name, seed_element, seed_position = seed_atom
        if seed_serial != index:
            raise ValueError(f"explicit MOL2 atom serial is nonsequential: {seed_serial}")
        if source_atom.element != seed_element:
            raise ValueError(
                f"source/explicit-MOL2 element mismatch at {index}: "
                f"{source_atom.element} != {seed_element}"
            )
        source_seed_distance = math.dist(source_atom.position_A, seed_position)
        if source_seed_distance > source_coordinate_tolerance_A:
            raise ValueError(
                f"source/explicit-MOL2 coordinate mismatch at {index}: "
                f"{source_seed_distance:.3f} A"
            )
        if output_atom.name.upper() != seed_name.upper() or output_atom.element != seed_element:
            raise ValueError(
                f"explicit-MOL2/parameterized identity mismatch at {index}: "
                f"{seed_name}/{seed_element} != {output_atom.name}/{output_atom.element}"
            )
    return {atom.serial: index + 1 for index, atom in enumerate(original)}


def box_length_nm(pdb: Path, padding_nm: float) -> float:
    atoms = read_pdb(pdb)
    extents_A = [
        max(atom.position_A[axis] for atom in atoms)
        - min(atom.position_A[axis] for atom in atoms)
        for axis in range(3)
    ]
    return max(extents_A) / 10.0 + 2.0 * padding_nm


def dcm_count(box_nm: float, density_g_cm3: float) -> int:
    return max(
        1,
        round(
            density_g_cm3
            * box_nm**3
            * AVOGADRO_NM3_FACTOR
            / DCM_MOLAR_MASS_G_MOL
        ),
    )


def gro_atom_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"invalid GRO file: {path}")
    count = int(lines[1].strip())
    if count < 1 or len(lines) < count + 3:
        raise ValueError(f"invalid GRO atom count in {path}: {count}")
    return count


def write_solute_index(
    path: Path,
    atom_count: int,
    system_atom_count: int | None = None,
) -> None:
    """Write Solute and, when known, complete System index groups.

    ``system_atom_count`` is optional only for compatibility with setup
    bundles produced before the complete-System group was added.  New setup
    calls always pass it explicitly.
    """
    if system_atom_count is not None and not 0 < atom_count <= system_atom_count:
        raise ValueError(
            f"invalid solute/system atom counts: {atom_count}/{system_atom_count}"
        )
    if system_atom_count is None and atom_count < 1:
        raise ValueError(f"invalid solute atom count: {atom_count}")
    groups = (
        (("System", system_atom_count), ("Solute", atom_count))
        if system_atom_count is not None
        else (("Solute", atom_count),)
    )
    lines: list[str] = []
    for name, count in groups:
        assert count is not None
        values = [str(index) for index in range(1, count + 1)]
        lines.append(f"[ {name} ]")
        for start in range(0, len(values), 15):
            lines.append(" ".join(values[start : start + 15]))
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pack_box(
    destination: Path,
    dcm_pdb: Path,
    dcm_molecules: int,
    box_nm: float,
    log: Path,
    solute_pdb: Path | None = None,
) -> None:
    if os.environ.get("USE_PREPACKED_DCM_BOXES") == "1":
        validate_prepacked_box(destination, dcm_pdb, dcm_molecules, box_nm, solute_pdb)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(f"Reused validated local Packmol box: {destination}\n")
        return
    box_A = box_nm * 10.0
    sections = [
        "tolerance 2.0",
        "filetype pdb",
        f"output {destination}",
        "add_box_sides 1.0",
        "",
    ]
    if solute_pdb is not None:
        center = box_A / 2.0
        sections.extend(
            [
                f"structure {solute_pdb}",
                "  number 1",
                "  resnumbers 3",
                "  center",
                f"  fixed {center:.4f} {center:.4f} {center:.4f} 0. 0. 0.",
                "end structure",
                "",
            ]
        )
    sections.extend(
        [
            f"structure {dcm_pdb}",
            f"  number {dcm_molecules}",
            "  resnumbers 3",
            f"  inside box 1.0 1.0 1.0 {box_A - 1.0:.4f} {box_A - 1.0:.4f} {box_A - 1.0:.4f}",
            "end structure",
            "",
        ]
    )
    packmol_input = destination.with_suffix(".packmol.inp")
    packmol_input.write_text("\n".join(sections), encoding="utf-8")
    run_command([os.environ.get("PACKMOL_BIN", "packmol")], destination.parent, log, stdin_path=packmol_input)
    if not destination.is_file():
        raise RuntimeError(f"Packmol did not create {destination}")
    validate_prepacked_box(destination, dcm_pdb, dcm_molecules, box_nm, solute_pdb)


def validate_prepacked_box(
    destination: Path,
    dcm_pdb: Path,
    dcm_molecules: int,
    box_nm: float,
    solute_pdb: Path | None,
) -> None:
    """Check cached Packmol output before LEaP consumes it."""
    if not destination.is_file():
        raise FileNotFoundError(f"prepacked solvent box is missing: {destination}")
    atoms = read_pdb(destination)
    solvent_template = read_pdb(dcm_pdb)
    solute_template = read_pdb(solute_pdb) if solute_pdb is not None else []
    expected = len(solute_template) + dcm_molecules * len(solvent_template)
    if len(atoms) != expected:
        raise ValueError(f"prepacked box atom count mismatch: {destination}: {len(atoms)} != {expected}")
    for index, template in enumerate(solute_template):
        atom = atoms[index]
        if (atom.name, atom.element) != (template.name, template.element):
            raise ValueError(f"prepacked solute identity mismatch at atom {index + 1}: {destination}")
    offset = len(solute_template)
    for index, atom in enumerate(atoms[offset:]):
        template = solvent_template[index % len(solvent_template)]
        if (atom.name, atom.element) != (template.name, template.element):
            raise ValueError(f"prepacked solvent identity mismatch at atom {offset + index + 1}: {destination}")
    box_A = box_nm * 10.0
    if any(not -0.1 <= coordinate <= box_A + 0.1 for atom in atoms for coordinate in atom.position_A):
        raise ValueError(f"prepacked coordinates outside box: {destination}")


def amber_to_gromacs(prmtop: Path, rst7: Path, top: Path, gro: Path) -> None:
    try:
        import parmed as pmd
    except ModuleNotFoundError as error:
        raise RuntimeError("ParmEd is missing") from error
    structure = pmd.load_file(str(prmtop), xyz=str(rst7))
    structure.save(str(top), format="gromacs", overwrite=True)
    structure.save(str(gro), format="gro", overwrite=True)


TOPOLOGY_SECTION = re.compile(r"^\s*\[\s*([^]]+)\s*]\s*$")


def _topology_data_lines(lines: Sequence[str], start: int, end: int) -> list[str]:
    return [
        line
        for line in lines[start + 1 : end]
        if line.strip() and not line.lstrip().startswith((";", "#"))
    ]


def _topology_sections(lines: Sequence[str]) -> list[tuple[str, int, int]]:
    starts: list[tuple[str, int]] = []
    for index, line in enumerate(lines):
        match = TOPOLOGY_SECTION.match(line)
        if match:
            starts.append((match.group(1).strip().lower(), index))
    return [
        (name, start, starts[position + 1][1] if position + 1 < len(starts) else len(lines))
        for position, (name, start) in enumerate(starts)
    ]


def verify_lorentz_berthelot_topology(top: Path) -> None:
    lines = top.read_text(encoding="utf-8").splitlines()
    defaults = [section for section in _topology_sections(lines) if section[0] == "defaults"]
    if len(defaults) != 1:
        raise ValueError(f"expected exactly one [ defaults ] section in {top}")
    fields = _topology_data_lines(lines, defaults[0][1], defaults[0][2])[0].split()
    if len(fields) < 2 or fields[1] != "2":
        raise ValueError(
            f"DCM/GAFF2 mixing requires GROMACS combination rule 2 "
            f"(Lorentz-Berthelot), found: {' '.join(fields)}"
        )


def rigidify_ch2cl2_dc_topology(top: Path) -> None:
    """Replace only the DCM valence terms with the paper's three constraints.

    LEaP needs bonded parameters to build the intermediate Amber topology, but
    CH2Cl2-DC is a rigid three-site model.  ParmEd writes each residue-sized DCM
    as its own ``moleculetype``.  We therefore replace the three DCM bonds by
    explicit GROMACS constraints, remove its redundant angles, and explicitly
    exclude all three intramolecular nonbonded pairs.  Solute bonds are left
    untouched, so the MDP can continue to constrain only X-H bonds in GAFF2.
    """
    lines = top.read_text(encoding="utf-8").splitlines()
    sections = _topology_sections(lines)
    molecule_sections = [section for section in sections if section[0] == "moleculetype"]
    candidates: list[tuple[int, int]] = []
    for _, start, _ in molecule_sections:
        following = next(
            (
                section_start
                for name, section_start, _ in sections
                if section_start > start and name in {"moleculetype", "system"}
            ),
            len(lines),
        )
        molecule_name = _topology_data_lines(lines, start, following)[0].split()[0]
        if molecule_name == "DCM":
            candidates.append((start, following))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one DCM moleculetype in {top}, found {len(candidates)}")
    block_start, block_end = candidates[0]
    local_sections = [
        section for section in sections if block_start < section[1] < block_end
    ]
    by_name = {name: (start, end) for name, start, end in local_sections}
    if "constraints" in by_name:
        verify_lorentz_berthelot_topology(top)
        return
    if "bonds" not in by_name or "atoms" not in by_name:
        raise ValueError(f"DCM topology lacks [ atoms ] or [ bonds ] in {top}")
    atom_rows = _topology_data_lines(lines, *by_name["atoms"])
    atom_names = [row.split()[4] for row in atom_rows]
    if atom_names != ["C1", "CL1", "CL2"]:
        raise ValueError(f"unexpected CH2Cl2-DC atom order in {top}: {atom_names}")
    bond_rows = _topology_data_lines(lines, *by_name["bonds"])
    pairs = {tuple(sorted(map(int, row.split()[:2]))) for row in bond_rows}
    if pairs != {(1, 2), (1, 3), (2, 3)}:
        raise ValueError(f"unexpected CH2Cl2-DC bond graph in {top}: {sorted(pairs)}")
    if "exclusions" in by_name:
        raise ValueError(f"unexpected pre-existing DCM [ exclusions ] in {top}")

    replacement = [
        "[ constraints ]",
        "; ai  aj  funct  distance_nm; rigid CH2Cl2-DC, DOI 10.1021/jp3002383",
        "    1    2      1      0.182150",
        "    1    3      1      0.182150",
        "    2    3      1      0.293440",
        "",
        "[ exclusions ]",
        "; all intramolecular nonbonded pairs are excluded",
        "    1    2    3",
        "    2    1    3",
        "    3    1    2",
        "",
    ]
    bonds_start, bonds_end = by_name["bonds"]
    remove_ranges = [(bonds_start, bonds_end)]
    if "angles" in by_name:
        remove_ranges.append(by_name["angles"])
    output: list[str] = []
    index = 0
    while index < len(lines):
        matching = next((bounds for bounds in remove_ranges if bounds[0] == index), None)
        if matching is None:
            output.append(lines[index])
            index += 1
            continue
        if index == bonds_start:
            output.extend(replacement)
        index = matching[1]
    top.write_text("\n".join(output) + "\n", encoding="utf-8")
    verify_lorentz_berthelot_topology(top)


def build_system_topology(
    setup_dir: Path,
    packed_pdb: Path,
    box_nm: float,
    dcm_parameters: dict[str, object],
    solute_parameters: dict[str, object] | None,
    root: Path,
    defer_grompp: bool = False,
) -> None:
    tleap_lines = ["source leaprc.gaff2"]
    leap_atom_types = dcm_parameters.get("leap_atom_types", [])
    if leap_atom_types:
        entries = " ".join(
            f'{{ "{name}" "{element}" "{hybridization}" }}'
            for name, element, hybridization in leap_atom_types
        )
        tleap_lines.append(f"addAtomTypes {{ {entries} }}")
    tleap_lines.extend(
        [
            f"loadamberparams {dcm_parameters['frcmod']}",
            f"DCM = loadmol2 {dcm_parameters['mol2']}",
        ]
    )
    if solute_parameters is not None:
        tleap_lines.extend(
            [
                f"loadamberparams {solute_parameters['frcmod']}",
                f"SLU = loadmol2 {solute_parameters['mol2']}",
            ]
        )
    box_A = box_nm * 10.0
    tleap_lines.extend(
        [
            f"SYS = loadpdb {packed_pdb}",
            f"set SYS box {{ {box_A:.6f} {box_A:.6f} {box_A:.6f} }}",
            "check SYS",
            "saveamberparm SYS system.prmtop system.rst7",
            "savepdb SYS system_amber.pdb",
            "quit",
            "",
        ]
    )
    tleap_input = setup_dir / "tleap.in"
    tleap_input.write_text("\n".join(tleap_lines), encoding="utf-8")
    log = setup_dir / "setup.log"
    run_command(["tleap", "-f", str(tleap_input)], setup_dir, log)
    amber_to_gromacs(
        setup_dir / "system.prmtop",
        setup_dir / "system.rst7",
        setup_dir / "system.top",
        setup_dir / "system.gro",
    )
    if dcm_parameters["model"] == "ch2cl2_dc":
        rigidify_ch2cl2_dc_topology(setup_dir / "system.top")
    else:
        verify_lorentz_berthelot_topology(setup_dir / "system.top")
    if defer_grompp:
        return
    run_command(
        [
            "gmx", "grompp",
            "-f", str(root / "gromacs_dcm" / "mdp" / "em.mdp"),
            "-c", "system.gro",
            "-p", "system.top",
            "-o", "em.tpr",
            "-po", "em_expanded.mdp",
            "-maxwarn", "0",
        ],
        setup_dir,
        log,
    )


def prepare_dcm(root: Path, force: bool, model: str = DEFAULT_DCM_MODEL) -> dict[str, object]:
    if model not in DCM_MODELS:
        raise ValueError(f"unknown DCM model: {model}")
    if model == "gaff2_am1bcc":
        result = parameterize_molecule(
            root / "gromacs_dcm" / "dcm.pdb",
            root / "gromacs_dcm" / "parameters" / "dcm_gaff2_am1bcc",
            "DCM",
            0,
            force,
        )
        result.update(
            {
                "model": model,
                "name": "GAFF2/AM1-BCC DCM",
                "doi": None,
                "mixing_rule": "Lorentz-Berthelot",
            }
        )
        return result

    model_dir = root / "gromacs_dcm" / "models" / "ch2cl2_dc"
    metadata = json.loads((model_dir / "model.json").read_text(encoding="utf-8"))
    result: dict[str, object] = {
        "model": model,
        "name": metadata["name"],
        "doi": metadata["doi"],
        "mixing_rule": metadata["mixing_rule"],
        "pdb": model_dir / "dcm.pdb",
        "mol2": model_dir / "dcm.mol2",
        "frcmod": model_dir / "dcm.frcmod",
        "leap_atom_types": [("DU", "C", "sp3"), ("DJ", "Cl", "sp3")],
        "charge": 0.0,
        "atom_count": 3,
    }
    missing = [str(path) for key in ("pdb", "mol2", "frcmod") if not (path := result[key]).is_file()]
    if missing:
        raise FileNotFoundError(f"CH2Cl2-DC model files are missing: {missing}")
    charge, count = mol2_charge_and_count(Path(str(result["mol2"])))
    if abs(charge) > 5.0e-7 or count != 3:
        raise ValueError(f"invalid CH2Cl2-DC MOL2: charge={charge}, atoms={count}")
    if len(read_pdb(Path(str(result["pdb"])))) != 3:
        raise ValueError("invalid CH2Cl2-DC PDB atom count")
    return result


def prepare_pure_dcm(
    root: Path,
    parameters: dict[str, object],
    force: bool,
    density_g_cm3: float,
    defer_grompp: bool = False,
) -> Path:
    setup = root / "gromacs_dcm" / "solvent_validation" / str(parameters["model"]) / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    complete = setup / ("system.gro" if defer_grompp else "em.tpr")
    if complete.is_file() and not force:
        return setup
    count = dcm_count(PURE_DCM_BOX_NM, density_g_cm3)
    packed = setup / "packed.pdb"
    pack_box(
        packed,
        Path(str(parameters["pdb"])),
        count,
        PURE_DCM_BOX_NM,
        setup / "setup.log",
    )
    build_system_topology(
        setup,
        packed,
        PURE_DCM_BOX_NM,
        parameters,
        None,
        root,
        defer_grompp=defer_grompp,
    )
    metadata = {
        "kind": "pure_dcm_density_validation",
        "dcm_model": parameters["model"],
        "dcm_model_name": parameters["name"],
        "dcm_model_doi": parameters["doi"],
        "mixing_rule": parameters["mixing_rule"],
        "box_nm": PURE_DCM_BOX_NM,
        "dcm_molecules": count,
        "initial_pack_density_g_cm3": density_g_cm3,
        "temperature_K": 313.15,
        "pressure_bar": 1.0,
        "required_density_gate": {"target_g_cm3": 1.29, "relative_tolerance": 0.03},
    }
    (setup / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return setup


def prepare_initial_conformer_setups(
    *,
    root: Path,
    state: str,
    source_pdb: Path,
    setup: Path,
    dcm_parameters: dict[str, object],
    solute_parameters: dict[str, object],
    padding_nm: float,
    density_g_cm3: float,
    defer_grompp: bool,
    force: bool,
) -> list[dict[str, object]]:
    """Pack and topology-check one independent solvent box per conformer."""
    conformers = initial_conformer_paths(source_pdb)
    results: list[dict[str, object]] = []
    parameterized_template = Path(str(solute_parameters["pdb"]))
    atom_count = int(solute_parameters["atom_count"])
    for replica, conformer in enumerate(conformers, start=1):
        replica_setup = setup / "initial_conformers" / f"replica_{replica:02d}"
        replica_setup.mkdir(parents=True, exist_ok=True)
        metadata_path = replica_setup / "metadata.json"
        complete = replica_setup / ("system.gro" if defer_grompp else "em.tpr")
        conformer_digest = hashlib.sha256(conformer.read_bytes()).hexdigest()
        if complete.is_file() and metadata_path.is_file() and not force:
            cached = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                cached.get("source_conformer_sha256") == conformer_digest
                and cached.get("dcm_model") == dcm_parameters["model"]
                and cached.get("source_atom_count") == atom_count
            ):
                results.append(cached)
                continue

        solute_pdb = replica_setup / "solute.pdb"
        transplant_pdb_coordinates(conformer, parameterized_template, solute_pdb)
        box_nm = box_length_nm(solute_pdb, padding_nm)
        count = dcm_count(box_nm, density_g_cm3)
        packed = replica_setup / "packed.pdb"
        pack_box(
            packed,
            Path(str(dcm_parameters["pdb"])),
            count,
            box_nm,
            replica_setup / "setup.log",
            solute_pdb=solute_pdb,
        )
        build_system_topology(
            replica_setup,
            packed,
            box_nm,
            dcm_parameters,
            solute_parameters,
            root,
            defer_grompp=defer_grompp,
        )
        write_solute_index(
            replica_setup / "index.ndx",
            atom_count,
            gro_atom_count(replica_setup / "system.gro"),
        )
        metadata: dict[str, object] = {
            "kind": "replica_initial_conformer_setup",
            "state": state,
            "replica": replica,
            "source_conformer": str(conformer.relative_to(root)),
            "source_conformer_sha256": conformer_digest,
            "source_atom_count": atom_count,
            "box_nm": box_nm,
            "padding_nm": padding_nm,
            "dcm_molecules": count,
            "initial_pack_density_g_cm3": density_g_cm3,
            "dcm_model": dcm_parameters["model"],
            "dcm_model_name": dcm_parameters["name"],
            "dcm_model_doi": dcm_parameters["doi"],
            "mixing_rule": dcm_parameters["mixing_rule"],
            "solute_force_field": "GAFF2",
            "solute_charge_model": "AM1-BCC",
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        results.append(metadata)
    return results


def prepare_state(task: SetupTask) -> dict[str, object]:
    root = Path(task.root)
    row = task.row
    state = row["state"]
    run_dir = root / row["output_dir"]
    setup = run_dir / "setup"
    setup.mkdir(parents=True, exist_ok=True)
    source_pdb = root / row["source_pdb"]
    conformers = initial_conformer_paths(source_pdb)
    metadata_path = setup / "metadata.json"
    complete = setup / ("system.gro" if task.defer_grompp else "em.tpr")
    if complete.is_file() and metadata_path.is_file() and not task.force:
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        conformer_setups_current = True
        for replica, conformer in enumerate(conformers, start=1):
            replica_setup = setup / "initial_conformers" / f"replica_{replica:02d}"
            replica_metadata = replica_setup / "metadata.json"
            replica_complete = replica_setup / (
                "system.gro" if task.defer_grompp else "em.tpr"
            )
            if not replica_complete.is_file() or not replica_metadata.is_file():
                conformer_setups_current = False
                break
            replica_cached = json.loads(replica_metadata.read_text(encoding="utf-8"))
            if replica_cached.get("source_conformer_sha256") != hashlib.sha256(
                conformer.read_bytes()
            ).hexdigest():
                conformer_setups_current = False
                break
        if (
            cached.get("dcm_model") == task.dcm_model
            and cached_sqm_tolerance_satisfies(cached, task.sqm_grms_tol)
            and conformer_setups_current
        ):
            return {"state": state, "status": "SKIPPED"}
    try:
        dcm_parameters = prepare_dcm(root, False, task.dcm_model)
        solute_parameters = parameterize_molecule(
            source_pdb,
            setup / "solute_parameters",
            "SLU",
            int(row["net_charge"]),
            task.force,
            task.sqm_grms_tol,
        )
        parameterized_pdb = Path(str(solute_parameters["pdb"]))
        explicit_input = solute_parameters.get("explicit_input_mol2")
        if explicit_input is not None:
            # The explicit-bond MOL2 is generated directly from the source PDB
            # and preserves atom order.  Use that identity proof for every
            # Antechamber path, not only the completed-SQM recovery path:
            # ordinary SQM optimization can also move a large macrocycle far
            # enough that coordinate-nearest mapping becomes unsafe.
            serial_map = explicit_mol2_serial_atom_map(
                source_pdb,
                Path(str(explicit_input)),
                parameterized_pdb,
            )
        else:
            serial_map = coordinate_atom_map(source_pdb, parameterized_pdb)
        box_nm = box_length_nm(Path(str(solute_parameters["pdb"])), task.padding_nm)
        count = dcm_count(box_nm, task.density_g_cm3)
        packed = setup / "packed.pdb"
        pack_box(
            packed,
            Path(str(dcm_parameters["pdb"])),
            count,
            box_nm,
            setup / "setup.log",
            solute_pdb=Path(str(solute_parameters["pdb"])),
        )
        build_system_topology(
            setup,
            packed,
            box_nm,
            dcm_parameters,
            solute_parameters,
            root,
            defer_grompp=task.defer_grompp,
        )
        write_solute_index(
            setup / "index.ndx",
            int(solute_parameters["atom_count"]),
            gro_atom_count(setup / "system.gro"),
        )

        source_mapping = json.loads(row["vinyl_mapping_json"])
        gromacs_mapping = {
            vinyl: [serial_map[int(serial)] for serial in serials]
            for vinyl, serials in source_mapping.items()
        }
        metadata = {
            "state": state,
            "stage": row["stage"],
            "system": row["system"],
            "branch": row["branch"],
            "source_pdb": row["source_pdb"],
            "source_atom_count": solute_parameters["atom_count"],
            "box_nm": box_nm,
            "padding_nm": task.padding_nm,
            "dcm_molecules": count,
            "initial_pack_density_g_cm3": task.density_g_cm3,
            "source_to_gromacs_atom": serial_map,
            "vinyl_mapping_gromacs_1based": gromacs_mapping,
            "pair_schema": json.loads(row["pair_schema_json"]),
            "solute_force_field": "GAFF2",
            "solute_charge_model": "AM1-BCC",
            "sqm_grms_tol_kcal_mol_A": task.sqm_grms_tol,
            "dcm_model": dcm_parameters["model"],
            "dcm_model_name": dcm_parameters["name"],
            "dcm_model_doi": dcm_parameters["doi"],
            "mixing_rule": dcm_parameters["mixing_rule"],
            "solute_net_charge": int(row["net_charge"]),
        }
        (setup / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        conformer_setups = prepare_initial_conformer_setups(
            root=root,
            state=state,
            source_pdb=source_pdb,
            setup=setup,
            dcm_parameters=dcm_parameters,
            solute_parameters=solute_parameters,
            padding_nm=task.padding_nm,
            density_g_cm3=task.density_g_cm3,
            defer_grompp=task.defer_grompp,
            force=task.force,
        )
        if conformer_setups:
            metadata["replica_initial_conformer_setups"] = [
                str(
                    Path("initial_conformers")
                    / f"replica_{int(item['replica']):02d}"
                )
                for item in conformer_setups
            ]
            (setup / "metadata.json").write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        return {
            "state": state,
            "status": "OK",
            "box_nm": box_nm,
            "dcm": count,
            "initial_conformers": len(conformer_setups),
        }
    except Exception as error:
        (setup / "setup_error.log").write_text(
            f"{error}\n\n{traceback.format_exc()}", encoding="utf-8"
        )
        return {"state": state, "status": "ERROR", "error": str(error)}


def parameterize_state_only(task: SetupTask) -> dict[str, object]:
    """Run only the expensive GAFF2/AM1-BCC solute stage.

    This is the remote Amber-image half of the split workflow. Packing,
    ``tleap`` assembly, and ParmEd conversion are deliberately deferred to the
    local workstation, where those lightweight tools are already installed.
    """
    root = Path(task.root)
    row = task.row
    state = row["state"]
    output = root / row["output_dir"] / "setup" / "solute_parameters"
    error_log = output / "parameterization_error.log"
    try:
        parameters = parameterize_molecule(
            root / row["source_pdb"],
            output,
            "SLU",
            int(row["net_charge"]),
            task.force,
            task.sqm_grms_tol,
        )
        if error_log.is_file():
            error_log.unlink()
        return {
            "state": state,
            "status": "OK",
            "atoms": parameters["atom_count"],
            "charge": parameters["charge"],
        }
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        error_log.write_text(
            f"{error}\n\n{traceback.format_exc()}", encoding="utf-8"
        )
        return {"state": state, "status": "ERROR", "error": str(error)}


def select_rows(
    rows: Sequence[dict[str, str]], states: Sequence[str], stages: Sequence[str]
) -> list[dict[str, str]]:
    stage_set = set(stages)
    selected = [row for row in rows if row["stage"] in stage_set]
    if states != ["all"]:
        state_set = set(states)
        known = {row["state"] for row in rows}
        unknown = state_set - known
        if unknown:
            raise ValueError(f"unknown states: {sorted(unknown)}")
        selected = [row for row in selected if row["state"] in state_set]
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--states", nargs="+", default=["all"])
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=SUPPORTED_REACTION_STAGES,
        default=list(SUPPORTED_REACTION_STAGES),
        help="reaction stages to include; B6 supports q0 through q6",
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--padding-nm", type=float, default=DEFAULT_PADDING_NM)
    parser.add_argument("--initial-density", type=float, default=DEFAULT_INITIAL_DENSITY_G_CM3)
    parser.add_argument("--dcm-model", choices=DCM_MODELS, default=DEFAULT_DCM_MODEL)
    parser.add_argument(
        "--sqm-grms-tol",
        type=float,
        default=DEFAULT_SQM_GRMS_TOL_KCAL_MOL_A,
        help="Antechamber/SQM RMS-gradient tolerance in kcal/(mol*A)",
    )
    parser.add_argument("--dcm-only", action="store_true")
    parser.add_argument(
        "--parameterize-only",
        action="store_true",
        help="run only solute GAFF2/AM1-BCC; defer Packmol/tleap/ParmEd locally",
    )
    parser.add_argument(
        "--defer-grompp",
        action="store_true",
        help=(
            "finish after AmberTools/ParmEd writes system.top and system.gro; "
            "a downstream GROMACS image must create em.tpr"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--use-prepacked-boxes", action="store_true",
        help="reuse locally generated Packmol boxes after atom and box checks",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.jobs < 1
        or args.padding_nm < 1.2
        or not 0.8 <= args.initial_density <= 1.4
        or not 0.0 < args.sqm_grms_tol <= 0.1
    ):
        raise SystemExit("invalid jobs, padding, initial density, or SQM tolerance")
    root = Path(args.root).resolve()
    if args.use_prepacked_boxes:
        os.environ["USE_PREPACKED_DCM_BOXES"] = "1"
    rows = read_manifest(root / "gromacs_dcm" / "state_manifest.csv")
    selected = select_rows(rows, args.states, args.stages)
    print(
        f"Selected {len(selected)}/{len(rows)} states; jobs={args.jobs}; "
        f"dcm_model={args.dcm_model}; sqm_grms_tol={args.sqm_grms_tol:g}"
    )
    if not selected and not args.dcm_only:
        raise SystemExit(
            "No states selected. Check --states/--stages and the manifest; "
            "refusing to report success for an empty calculation."
        )
    if args.dry_run:
        required = ["antechamber", "parmchk2"]
        if not args.parameterize_only:
            required.append("tleap")
            if not args.use_prepacked_boxes:
                required.append("packmol")
        if not args.parameterize_only and not args.defer_grompp:
            required.append("gmx")
        missing = [name for name in required if shutil.which(name) is None]
        print(f"Executable preflight: missing={missing or 'none'}")
        for row in selected:
            print(f"  {row['state']}: {row['source_pdb']} -> {row['output_dir']}")
        return
    tools = require_environment(
        require_gromacs=not args.defer_grompp,
        require_packing=not args.parameterize_only,
        require_packmol=not args.use_prepacked_boxes,
    )
    print(f"Environment passed: {tools}")
    if not args.parameterize_only:
        dcm_parameters = prepare_dcm(root, args.force, args.dcm_model)
        pure_setup = prepare_pure_dcm(
            root,
            dcm_parameters,
            args.force,
            args.initial_density,
            defer_grompp=args.defer_grompp,
        )
        print(f"Pure-DCM validation system ready: {pure_setup}")
        if args.dcm_only:
            return
    tasks = [
        SetupTask(
            root=str(root),
            row=row,
            force=args.force,
            padding_nm=args.padding_nm,
            density_g_cm3=args.initial_density,
            dcm_model=args.dcm_model,
            sqm_grms_tol=args.sqm_grms_tol,
            defer_grompp=args.defer_grompp,
        )
        for row in selected
    ]
    counts = {"OK": 0, "SKIPPED": 0, "ERROR": 0}
    worker = parameterize_state_only if args.parameterize_only else prepare_state
    # A one-job request should be genuinely serial.  Besides avoiding needless
    # process startup, this works in restricted macOS environments where
    # ProcessPoolExecutor's semaphore-limit probe is not permitted.
    if args.jobs == 1:
        results = (worker(task) for task in tasks)
        for result in results:
            status = str(result["status"])
            counts[status] += 1
            suffix = f"; {result.get('error')}" if status == "ERROR" else ""
            print(f"[{status}] {result['state']}{suffix}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(worker, task): task for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                status = str(result["status"])
                counts[status] += 1
                suffix = f"; {result.get('error')}" if status == "ERROR" else ""
                print(f"[{status}] {result['state']}{suffix}", flush=True)
    print(f"Setup summary: {counts}")
    if counts["ERROR"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
