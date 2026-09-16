#!/usr/bin/env python3
"""Export compact vinyl-carbon coordinates for q1/q2 NAC analysis.

This script is designed to run inside the GROMACS container immediately after
production.  It uses ``gmx traj`` rather than MDAnalysis, so the remote image
needs no extra Python packages.  The output contains only selected coordinates
and a self-describing JSON manifest; solvent trajectories remain remote.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Mapping, Sequence


PAIR_RE = re.compile(r"^V(\d+)-V(\d+)$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_reacted_pair(value: str) -> tuple[int, int]:
    match = PAIR_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid reacted-pair label: {value!r}")
    return tuple(sorted((int(match.group(1)), int(match.group(2)))))


def state_history(root: Path, state: str, stage: str) -> tuple[dict[int, int], list[tuple[int, int]]]:
    if stage == "q1":
        mapping_path = root / "results/react1/vinyl_mapping.csv"
        structure_path = root / "results/react1/structure_qc.csv"
        mapping_rows = [row for row in read_csv(mapping_path) if row["state"] == state]
        structure = next(row for row in read_csv(structure_path) if row["state"] == state)
        history = [parse_reacted_pair(structure["reacted_pair"])]
    elif stage == "q2":
        mapping_path = root / "results/react2/vinyl_mapping.csv"
        structure_path = root / "results/react2/structure_qc.csv"
        mapping_rows = [row for row in read_csv(mapping_path) if row["state"] == state]
        structure = next(row for row in read_csv(structure_path) if row["state"] == state)
        history = [
            parse_reacted_pair(structure["first_reacted_pair"]),
            parse_reacted_pair(structure["second_reacted_pair"]),
        ]
    else:
        raise ValueError(f"post-reaction NAC export accepts q1/q2, not {stage!r}")
    if not mapping_rows:
        raise ValueError(f"no vinyl-to-arm mapping for {state}")
    arm_by_vinyl = {
        int(row["vinyl"].removeprefix("V")): int(row["arm_zero_based"])
        for row in mapping_rows
    }
    return arm_by_vinyl, history


def annotate_pairs(
    pair_schema: Sequence[Mapping[str, object]],
    arm_by_vinyl: Mapping[int, int],
    reacted_pairs: Sequence[tuple[int, int]],
) -> list[dict[str, object]]:
    reacted_arm_pairs = {
        tuple(sorted((arm_by_vinyl[left], arm_by_vinyl[right])))
        for left, right in reacted_pairs
    }
    reacted_arms = {arm for pair in reacted_arm_pairs for arm in pair}
    annotations: list[dict[str, object]] = []
    for pair in pair_schema:
        left, right = int(pair["left"]), int(pair["right"])
        left_arm, right_arm = arm_by_vinyl[left], arm_by_vinyl[right]
        arm_pair = tuple(sorted((left_arm, right_arm)))
        topology = str(pair["topology"])
        if topology == "same_arm":
            history_class = "same_arm"
            route_family = "same_arm"
        elif arm_pair in reacted_arm_pairs:
            history_class = "reuse_arm_pair"
            route_family = "reuse"
        elif left_arm in reacted_arms and right_arm in reacted_arms:
            history_class = "switch_between_reacted_arms"
            route_family = "switch"
        elif (left_arm in reacted_arms) ^ (right_arm in reacted_arms):
            history_class = "one_reacted_arm"
            route_family = "switch"
        else:
            history_class = "untouched_arm_pair"
            route_family = "switch"
        annotations.append({
            "pair": str(pair["column"]),
            "left_vinyl": left,
            "right_vinyl": right,
            "left_arm": left_arm,
            "right_arm": right_arm,
            "topology": topology,
            "history_class": history_class,
            "route_family": route_family,
        })
    return annotations


def numeric_xvg_rows(path: Path) -> int:
    rows = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and stripped[0] not in "#@&":
            rows += 1
    return rows


def export_replica(
    root: Path,
    state_row: Mapping[str, str],
    replica: int,
    output: Path,
    gmx: str,
) -> Path:
    state = state_row["state"]
    stage = state_row["stage"]
    state_dir = root / state_row["output_dir"]
    setup = state_dir / "setup"
    metadata_path = setup / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    replica_dir = state_dir / f"replica_{replica:02d}"
    prefix = replica_dir / "production_2ns"
    xtc, tpr = prefix.with_suffix(".xtc"), prefix.with_suffix(".tpr")
    if not xtc.is_file() or not tpr.is_file():
        raise FileNotFoundError(f"production trajectory is incomplete: {prefix}")

    mapping = {
        str(name): [int(value) for value in values]
        for name, values in metadata["vinyl_mapping_gromacs_1based"].items()
    }
    labels = sorted(mapping, key=lambda name: int(name.removeprefix("V")))
    selected_atoms = [atom for label in labels for atom in mapping[label]]
    if len(selected_atoms) != 2 * len(labels) or len(set(selected_atoms)) != len(selected_atoms):
        raise ValueError(f"invalid vinyl atom mapping for {state}")
    arm_by_vinyl, reacted_pairs = state_history(root, state, stage)
    annotations = annotate_pairs(metadata["pair_schema"], arm_by_vinyl, reacted_pairs)

    target = output / state / f"replica_{replica:02d}"
    target.mkdir(parents=True, exist_ok=True)
    index = target / "vinyl_carbons.ndx"
    index.write_text(
        "[ VinylCarbons ]\n" + " ".join(str(atom) for atom in selected_atoms) + "\n",
        encoding="utf-8",
    )
    coordinates = target / "vinyl_coordinates_nm.xvg"
    boxes = target / "box_vectors_nm.xvg"
    command = [
        gmx, "traj", "-f", str(xtc), "-s", str(tpr), "-n", str(index),
        "-ox", str(coordinates), "-ob", str(boxes), "-xvg", "none", "-fp",
    ]
    completed = subprocess.run(
        command,
        input="VinylCarbons\n",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    (target / "gmx_traj.log").write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0 or not coordinates.is_file() or not boxes.is_file():
        raise RuntimeError(f"gmx traj failed for {state} replica {replica:02d}")
    frames = numeric_xvg_rows(coordinates)
    if frames < 2:
        raise ValueError(f"too few exported frames for {state} replica {replica:02d}: {frames}")

    manifest = {
        "format": "explicit-dcm-postreaction-nac-coordinates-v1",
        "state": state,
        "system": state_row["system"],
        "stage": stage,
        "branch": state_row["branch"],
        "replica": replica,
        "temperature_K": 313.15,
        "coordinate_unit": "nm",
        "coordinate_file": coordinates.name,
        "box_file": boxes.name,
        "frames": frames,
        "ordered_vinyl_labels": labels,
        "selected_gromacs_atoms_1based": selected_atoms,
        "vinyl_atom_order": [
            {"vinyl": label, "atoms_1based": mapping[label]} for label in labels
        ],
        "reacted_vinyl_pairs": [f"V{left}-V{right}" for left, right in reacted_pairs],
        "reacted_arm_pairs": [
            list(sorted((arm_by_vinyl[left], arm_by_vinyl[right])))
            for left, right in reacted_pairs
        ],
        "pair_annotations": annotations,
        "source": {
            "xtc": str(xtc.relative_to(root)),
            "tpr": str(tpr.relative_to(root)),
            "metadata": str(metadata_path.relative_to(root)),
        },
    }
    (target / "nac_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for suffix in (".log", ".gro"):
        source = prefix.with_suffix(suffix)
        if source.is_file():
            shutil.copy2(source, target / source.name)
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--states", nargs="+", required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--replica-indices", nargs="+", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gmx", default=os.environ.get("GMX", "gmx"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    rows = read_csv(root / "gromacs_dcm/state_manifest.csv")
    by_state = {row["state"]: row for row in rows}
    unknown = set(args.states) - set(by_state)
    if unknown:
        raise ValueError(f"states absent from manifest: {sorted(unknown)}")
    selected_states = [state for state in args.states if by_state[state]["stage"] in {"q1", "q2"}]
    skipped_states = [state for state in args.states if state not in selected_states]
    replicas = args.replica_indices or list(range(1, args.replicas + 1))
    if any(replica < 1 or replica > args.replicas for replica in replicas):
        raise ValueError("replica indices must lie within 1..replicas")
    completed = []
    for state in selected_states:
        for replica in replicas:
            completed.append(export_replica(root, by_state[state], replica, output, args.gmx))
    print(f"exported_replicas={len(completed)}")
    if skipped_states:
        print(f"skipped_non_q1_q2_states={' '.join(skipped_states)}")
    for path in completed:
        print(path)


if __name__ == "__main__":
    main()
