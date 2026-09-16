#!/usr/bin/env python3
"""Recover compact NAC data from a failed A4 q2 GROMACS workspace.

The affected workflow completed production MD but failed while loading legacy
q2 history tables.  This utility locates the downloaded workspace, injects the
correct compatibility metadata, and reruns only ``gmx traj`` locally.  It does
not rerun dynamics or modify the production trajectory.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import export_explicit_dcm_postreaction_nac as exporter


STATES = (
    "A4_m_C5_2_reuse_cis",
    "A4_m_C5_2_mixed_same_cross_cis",
    "A4_m_C8_2_reuse_cis",
    "A4_m_C8_2_mixed_same_cross_cis",
)


def find_bundle(download: Path) -> Path:
    manifests = list(download.rglob("gromacs_dcm/state_manifest.csv"))
    if len(manifests) != 1:
        raise RuntimeError(
            "expected exactly one recovered state manifest; "
            f"found={manifests}"
        )
    return manifests[0].parents[1]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("download", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument(
        "--metadata-source",
        type=Path,
        default=project / "results/a4_q2_missing_md_ready/results/react2",
    )
    args = parser.parse_args()
    if args.replicas < 1:
        raise SystemExit("replicas must be positive")
    download = args.download.resolve()
    output = args.output.resolve()
    bundle = find_bundle(download)

    metadata_source = args.metadata_source.resolve()
    for name in ("vinyl_mapping.csv", "structure_qc.csv"):
        source = metadata_source / name
        if not source.is_file():
            raise FileNotFoundError(f"recovery metadata is missing: {source}")
        destination = bundle / "results/react2" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    rows = read_manifest(bundle / "gromacs_dcm/state_manifest.csv")
    by_state = {row["state"]: row for row in rows}
    missing_states = set(STATES) - set(by_state)
    if missing_states:
        raise ValueError(f"recovered manifest lacks states: {sorted(missing_states)}")
    missing_production = []
    for state in STATES:
        state_dir = bundle / by_state[state]["output_dir"]
        for replica in range(1, args.replicas + 1):
            prefix = state_dir / f"replica_{replica:02d}/production_2ns"
            for suffix in (".xtc", ".tpr"):
                path = prefix.with_suffix(suffix)
                if not path.is_file():
                    missing_production.append(str(path))
    if missing_production:
        raise FileNotFoundError(
            "production MD is incomplete in the recovered artifact: "
            + "; ".join(missing_production)
        )

    output.mkdir(parents=True, exist_ok=True)
    completed = []
    for state in STATES:
        for replica in range(1, args.replicas + 1):
            print(f"Exporting {state} replica {replica:02d} ...", flush=True)
            completed.append(
                exporter.export_replica(
                    bundle, by_state[state], replica, output, args.gmx
                )
            )
    summary = {
        "status": "PASS",
        "source_workflow": "b6-gromacs-t2s5l",
        "dynamics_rerun": False,
        "recovered_bundle": str(bundle),
        "exported_replicas": len(completed),
        "states": list(STATES),
        "output": str(output),
    }
    (output / "recovery_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
