#!/usr/bin/env python3
"""Create a compact prepared GROMACS bundle for A4 C5/C8 q1/q2 NAC runs."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


Q1_STATES = ("A4_m_C5_1_cis", "A4_m_C8_1_cis")
Q2_STATES = (
    "A4_m_C5_2_adjacent_cis", "A4_m_C5_2_opposite_cis",
    "A4_m_C8_2_adjacent_cis", "A4_m_C8_2_opposite_cis",
)


def copy_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--prepared-source", type=Path, default=Path("autodl-tmp/Molecular_dynamics_project"))
    parser.add_argument("--output", type=Path, default=Path("results/a4_postreaction_nac_prepared_bundle"))
    parser.add_argument("--stage", choices=("q1", "q2", "all"), default="all")
    args = parser.parse_args()
    project, prepared, output = args.project.resolve(), args.prepared_source.resolve(), args.output.resolve()
    states = list(Q1_STATES if args.stage == "q1" else Q2_STATES if args.stage == "q2" else Q1_STATES + Q2_STATES)
    manifest_path = project / "gromacs_dcm/state_manifest.csv"
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0])
    by_state = {row["state"]: row for row in rows}
    unknown = set(states) - set(by_state)
    if unknown:
        raise ValueError(f"states absent from source manifest: {sorted(unknown)}")
    output.mkdir(parents=True, exist_ok=True)

    for relative in (
        "gromacs_dcm/mdp",
        "results/react1/structure_qc.csv",
        "results/react1/vinyl_mapping.csv",
        "results/react2/structure_qc.csv",
        "results/react2/vinyl_mapping.csv",
    ):
        copy_path(project / relative, output / relative)
    density_gate = Path("gromacs_dcm/solvent_validation/ch2cl2_dc/run/density_gate.json")
    copy_path(
        project / "gromacs_dcm_server_bundle" / density_gate,
        output / density_gate,
    )
    for state in states:
        row = by_state[state]
        copy_path(project / row["source_pdb"], output / row["source_pdb"])
        setup = Path(row["output_dir"]) / "setup"
        copy_path(prepared / setup, output / setup)
    # TPR/checkpoint/trajectory files are GROMACS-version-specific runtime
    # products.  The prepared setups were created locally with GROMACS 2026.3,
    # whereas the Bohrium image currently runs 2024.2.  Never upload those
    # binaries: the remote runtime must rebuild every TPR from MDP/GRO/TOP.
    removed = []
    for pattern in ("*.tpr", "*.cpt", "*.edr", "*.trr", "*.xtc"):
        for path in output.rglob(pattern):
            path.unlink()
            removed.append(path)
    (output / "analysis").mkdir(parents=True, exist_ok=True)
    manifest_out = output / "gromacs_dcm/state_manifest.csv"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    with manifest_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([by_state[state] for state in states])
    print(f"prepared_bundle={output}")
    print(f"states={' '.join(states)}")
    print(f"removed_version_specific_runtime_files={len(removed)}")


if __name__ == "__main__":
    main()
