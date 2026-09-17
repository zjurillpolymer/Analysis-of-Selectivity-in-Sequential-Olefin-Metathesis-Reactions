#!/usr/bin/env python3
"""Audit completed B6 explicit-DCM boxes before GPU MD submission."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from setup_explicit_dcm_gromacs import (
    gro_atom_count,
    verify_lorentz_berthelot_topology,
)


def check_setup(folder: Path, expected_state: str | None) -> dict[str, object]:
    for filename in ("system.gro", "system.top", "system.prmtop", "system.rst7", "metadata.json", "setup.log"):
        path = folder / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    top = folder / "system.top"
    verify_lorentz_berthelot_topology(top)
    if "rigid CH2Cl2-DC" not in top.read_text(encoding="utf-8"):
        raise ValueError(f"rigid DCM constraints missing: {top}")
    log = (folder / "setup.log").read_text(encoding="utf-8", errors="replace")
    if "Exiting LEaP: Errors = 0" not in log:
        raise ValueError(f"LEaP did not report zero errors: {folder}")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("dcm_model") != "ch2cl2_dc":
        raise ValueError(f"wrong DCM model: {folder}")
    if expected_state is not None and metadata.get("state") != expected_state:
        raise ValueError(f"wrong state metadata: {folder}")
    solvent_count = int(metadata["dcm_molecules"])
    solute_count = 0 if expected_state is None else int(metadata["source_atom_count"])
    expected_atoms = solute_count + 3 * solvent_count
    actual_atoms = gro_atom_count(folder / "system.gro")
    if actual_atoms != expected_atoms:
        raise ValueError(f"atom-count mismatch: {folder}: {actual_atoms} != {expected_atoms}")
    if expected_state is not None:
        mapping = metadata["vinyl_mapping_gromacs_1based"]
        expected_vinyls = 10 if "_q1_" in expected_state else 8
        if len(mapping) != expected_vinyls:
            raise ValueError(f"wrong vinyl mapping size: {expected_state}: {len(mapping)}")
        for pair in mapping.values():
            if len(pair) != 2 or any(not 1 <= int(atom) <= solute_count for atom in pair):
                raise ValueError(f"vinyl mapping outside solute: {expected_state}")
    return {"state": expected_state or "pure_dcm", "atoms": actual_atoms, "dcm": solvent_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / "gromacs_dcm/state_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise ValueError(f"expected five B6 rows, got {len(rows)}")
    reports = [
        check_setup(root / row["output_dir"] / "setup", row["state"])
        for row in rows
    ]
    reports.append(
        check_setup(
            root / "gromacs_dcm/solvent_validation/ch2cl2_dc/setup",
            None,
        )
    )
    for report in reports:
        print(f"[SETUP OK] {report['state']}: {report['atoms']} atoms, {report['dcm']} DCM")


if __name__ == "__main__":
    main()
