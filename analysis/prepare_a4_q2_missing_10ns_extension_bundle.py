#!/usr/bin/env python3
"""Merge completed 2 ns mixed-state replicas into a portable 10 ns seed bundle."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


STATES = (
    "A4_m_C5_2_mixed_same_cross_cis",
    "A4_m_C8_2_mixed_same_cross_cis",
)
REQUIRED_PRODUCTION_SUFFIXES = (".tpr", ".cpt", ".xtc", ".gro")


def find_bundle(root: Path) -> Path:
    direct = root / "gromacs_dcm/state_manifest.csv"
    if direct.is_file():
        return root
    manifests = list(root.rglob("gromacs_dcm/state_manifest.csv"))
    if len(manifests) != 1:
        raise RuntimeError(f"expected one bundle below {root}; found={manifests}")
    return manifests[0].parents[1]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_replica(bundle: Path, row: dict[str, str], replica: int) -> Path:
    replica_dir = bundle / row["output_dir"] / f"replica_{replica:02d}"
    missing = [
        str(replica_dir / f"production_2ns{suffix}")
        for suffix in REQUIRED_PRODUCTION_SUFFIXES
        if not (replica_dir / f"production_2ns{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"completed 2 ns replica is incomplete for {row['state']} replica {replica}: {missing}"
        )
    return replica_dir


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepared-record",
        type=Path,
        default=project / "results/a4_q2_missing_md_ready/gromacs_replicas_2_3_submission.json",
        help="record whose bundle field points to the parameterized template",
    )
    parser.add_argument("--replica1-download", type=Path, required=True)
    parser.add_argument("--parallel-download", type=Path, required=True)
    parser.add_argument("--parallel-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prepared_record = json.loads(args.prepared_record.resolve().read_text(encoding="utf-8"))
    template = find_bundle(Path(str(prepared_record["bundle"])).resolve())
    replica1_bundle = find_bundle(args.replica1_download.resolve())
    parallel_root = args.parallel_download.resolve()
    parallel_record = json.loads(args.parallel_record.resolve().read_text(encoding="utf-8"))
    record_states = [str(state) for state in parallel_record["states"]]
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite extension bundle: {output}")
    shutil.copytree(template, output)

    fields, template_rows = read_csv(output / "gromacs_dcm/state_manifest.csv")
    by_state = {row["state"]: row for row in template_rows}
    missing_states = set(STATES) - set(by_state)
    if missing_states:
        raise ValueError(f"template manifest lacks mixed states: {sorted(missing_states)}")
    _, replica1_rows = read_csv(replica1_bundle / "gromacs_dcm/state_manifest.csv")
    replica1_by_state = {row["state"]: row for row in replica1_rows}

    sources: list[dict[str, object]] = []
    for state in STATES:
        destination_row = by_state[state]
        source_replica1 = validate_replica(replica1_bundle, replica1_by_state[state], 1)
        destination_replica1 = output / destination_row["output_dir"] / "replica_01"
        shutil.copytree(source_replica1, destination_replica1, dirs_exist_ok=True)
        sources.append(
            {
                "state": state,
                "replica": 1,
                "source": str(source_replica1),
                "destination": str(destination_replica1.relative_to(output)),
            }
        )
        state_index = record_states.index(state) + 1
        for replica in (2, 3):
            step = f"gromacs-run-{state_index:02d}-{replica:02d}"
            task_bundle = find_bundle(parallel_root / step)
            _, task_rows = read_csv(task_bundle / "gromacs_dcm/state_manifest.csv")
            task_by_state = {row["state"]: row for row in task_rows}
            source_replica = validate_replica(task_bundle, task_by_state[state], replica)
            destination_replica = (
                output / destination_row["output_dir"] / f"replica_{replica:02d}"
            )
            shutil.copytree(source_replica, destination_replica, dirs_exist_ok=True)
            sources.append(
                {
                    "state": state,
                    "replica": replica,
                    "source_step": step,
                    "source": str(source_replica),
                    "destination": str(destination_replica.relative_to(output)),
                }
            )

    write_csv(
        output / "gromacs_dcm/state_manifest.csv",
        fields,
        [by_state[state] for state in STATES],
    )
    compatibility = project / "results/a4_q2_missing_md_ready/results/react2"
    destination_metadata = output / "results/react2"
    destination_metadata.mkdir(parents=True, exist_ok=True)
    for name in ("vinyl_mapping.csv", "structure_qc.csv"):
        shutil.copy2(compatibility / name, destination_metadata / name)

    summary = {
        "status": "READY_10NS_EXTENSION",
        "states": list(STATES),
        "replicas_per_state": 3,
        "completed_total_ns": 2.0,
        "target_total_ns": 10.0,
        "continuation_mode": "checkpoint_noappend_portable",
        "template": str(template),
        "replica_sources": sources,
        "output": str(output),
    }
    (output / "extension_seed_audit.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
