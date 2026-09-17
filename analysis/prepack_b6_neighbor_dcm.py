#!/usr/bin/env python3
"""Generate B6 q1/q2 and pure-DCM Packmol boxes on the local workstation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from setup_explicit_dcm_gromacs import (
    DEFAULT_INITIAL_DENSITY_G_CM3,
    DEFAULT_PADDING_NM,
    PURE_DCM_BOX_NM,
    box_length_nm,
    dcm_count,
    pack_box,
    prepare_dcm,
    validate_prepacked_box,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--packmol-bin", type=Path, required=True)
    parser.add_argument("--library-dir", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    root = args.root.resolve()
    packmol_bin = args.packmol_bin.resolve()
    if not packmol_bin.is_file():
        raise FileNotFoundError(packmol_bin)
    os.environ["PACKMOL_BIN"] = str(packmol_bin)
    if args.library_dir:
        libraries = ":".join(str(path.resolve()) for path in args.library_dir)
        os.environ["DYLD_LIBRARY_PATH"] = libraries

    model = prepare_dcm(root, False, "ch2cl2_dc")
    dcm_pdb = Path(str(model["pdb"]))
    with (root / "gromacs_dcm/state_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise ValueError("expected five staged B6 states")
    tasks: list[tuple[str, Path, int, float, Path | None]] = []
    pure_box = root / "gromacs_dcm/solvent_validation/ch2cl2_dc/setup/packed.pdb"
    tasks.append(("pure_dcm", pure_box, dcm_count(PURE_DCM_BOX_NM, DEFAULT_INITIAL_DENSITY_G_CM3), PURE_DCM_BOX_NM, None))
    for row in rows:
        state = row["state"]
        solute = root / row["output_dir"] / "setup/solute_parameters/slu.pdb"
        box_nm = box_length_nm(solute, DEFAULT_PADDING_NM)
        count = dcm_count(box_nm, DEFAULT_INITIAL_DENSITY_G_CM3)
        destination = root / row["output_dir"] / "setup/packed.pdb"
        tasks.append((state, destination, count, box_nm, solute))

    report = []
    for state, destination, count, box_nm, solute in tasks:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            validate_prepacked_box(destination, dcm_pdb, count, box_nm, solute)
            status = "validated_existing"
        else:
            pack_box(destination, dcm_pdb, count, box_nm, destination.parent / "local_packmol.log", solute)
            status = "packed"
        record = {
            "state": state,
            "status": status,
            "dcm_molecules": count,
            "box_nm": box_nm,
            "packed_pdb": str(destination.relative_to(root)),
            "sha256": digest(destination),
        }
        report.append(record)
        print(json.dumps(record), flush=True)
    (root / "prepacked_boxes.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
