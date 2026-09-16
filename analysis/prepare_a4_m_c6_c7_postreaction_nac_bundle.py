#!/usr/bin/env python3
"""Prepare the six C6/C7 q1 and q2 states for the C5/C8 NAC workflow.

This is a filtered bundle builder.  It keeps the validated structures,
mapping/QC tables, explicit DCM model and GROMACS MDP files, while removing
old runtime products so AmberTools and GROMACS rebuild the state cleanly.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


STATES = (
    "A4_m_C6_1_cis",
    "A4_m_C6_2_adjacent_cis",
    "A4_m_C6_2_opposite_cis",
    "A4_m_C7_1_cis",
    "A4_m_C7_2_adjacent_cis",
    "A4_m_C7_2_opposite_cis",
)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=project / "results/b6_q2_v3_v6_v11_v12_prepared/workspace",
        help="validated all-state workspace containing C6/C7 structures and metadata",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "results/a4_m_c6_c7_postreaction_nac_bundle",
    )
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing bundle: {output}")

    manifest_path = source / "gromacs_dcm/state_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or ())
        rows = list(reader)
    by_state = {row["state"]: row for row in rows}
    missing = sorted(set(STATES) - set(by_state))
    if missing:
        raise ValueError(f"states missing from source manifest: {missing}")

    for relative in (
        "gromacs_dcm/mdp",
        "gromacs_dcm/models/ch2cl2_dc",
        "gromacs_dcm/dcm.pdb",
        "gromacs_dcm/protocol.json",
        "gromacs_dcm/environment.yml",
        "gromacs_dcm/solvent_validation/ch2cl2_dc/run/density_gate.json",
        "results/react1/vinyl_mapping.csv",
        "results/react1/structure_qc.csv",
        "results/react2/vinyl_mapping.csv",
        "results/react2/structure_qc.csv",
    ):
        src = source / relative
        dst = output / relative
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            copy_file(src, dst)

    selected_rows = []
    for state in STATES:
        row = by_state[state]
        copy_file(source / row["source_pdb"], output / row["source_pdb"])
        selected_rows.append(row)
    output_manifest = output / "gromacs_dcm/state_manifest.csv"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected_rows)

    audit = {
        "status": "READY",
        "scope": "A4-m-C6/C7 q1 and q2 adjacent/opposite",
        "states": list(STATES),
        "production_coordinates_required": True,
        "old_runtime_products_excluded": True,
        "source": str(source),
        "output": str(output),
    }
    (output / "bundle_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
