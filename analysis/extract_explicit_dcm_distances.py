#!/usr/bin/env python3
"""Extract all active vinyl-midpoint distances from explicit-DCM XTC files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Sequence


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: Sequence[dict[str, str]], states: Sequence[str], stages: Sequence[str]
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["stage"] in set(stages)]
    if states != ["all"]:
        requested = set(states)
        unknown = requested - {row["state"] for row in rows}
        if unknown:
            raise ValueError(f"unknown states: {sorted(unknown)}")
        selected = [row for row in selected if row["state"] in requested]
    return selected


def midpoint_distances(
    positions_A,
    mapping: dict[str, list[int]],
    pair_schema: Sequence[dict[str, object]],
) -> list[float]:
    centers = {
        int(vinyl.removeprefix("V")): (
            positions_A[serials[0] - 1] + positions_A[serials[1] - 1]
        )
        / 2.0
        for vinyl, serials in mapping.items()
    }
    return [
        float(math.dist(centers[int(pair["left"])], centers[int(pair["right"])]))
        for pair in pair_schema
    ]


def solute_topology_path(setup: Path) -> Path:
    """Return the solute-only bonded topology, independent of GROMACS TPX."""
    path = setup / "solute_parameters" / "slu.mol2"
    if not path.is_file():
        raise FileNotFoundError(f"solute MOL2 topology is missing: {path}")
    return path


def extract_replica(setup: Path, replica: Path) -> dict[str, object]:
    try:
        import MDAnalysis as mda
        from MDAnalysis.transformations import unwrap
    except ModuleNotFoundError as error:
        raise RuntimeError("MDAnalysis is required for trajectory extraction") from error

    metadata = json.loads((setup / "metadata.json").read_text(encoding="utf-8"))
    xtc = replica / "production_2ns.xtc"
    if not xtc.is_file():
        raise FileNotFoundError(f"production XTC missing in {replica}")
    solute_count = int(metadata["source_atom_count"])
    topology = solute_topology_path(setup)
    # The production XTC contains only the first (solute) atom group.  Reading
    # its AmberTools MOL2 supplies the same atom order plus bonds for PBC
    # unwrapping, and avoids MDAnalysis' version-sensitive GROMACS TPR parser.
    universe = mda.Universe(str(topology), str(xtc))
    if len(universe.atoms) != solute_count:
        raise ValueError(
            f"solute MOL2/XTC atom count mismatch: {len(universe.atoms)} != {solute_count}"
        )
    solute = universe.atoms
    universe.trajectory.add_transformations(unwrap(solute))
    mapping = {
        key: [int(value) for value in values]
        for key, values in metadata["vinyl_mapping_gromacs_1based"].items()
    }
    pairs = list(metadata["pair_schema"])
    header = [
        "frame",
        "time_ps",
        "absolute_time_ps",
        *(str(pair["column"]) for pair in pairs),
    ]
    output = replica / "vinyl_distances.csv"
    first_time: float | None = None
    frame_count = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for frame, timestep in enumerate(universe.trajectory):
            absolute_time = float(timestep.time)
            if first_time is None:
                first_time = absolute_time
            distances = midpoint_distances(solute.positions, mapping, pairs)
            if not all(math.isfinite(value) and value > 0.0 for value in distances):
                raise ValueError(f"nonfinite/nonpositive vinyl distance at frame {frame}")
            writer.writerow(
                [
                    frame,
                    f"{absolute_time - first_time:.4f}",
                    f"{absolute_time:.4f}",
                    *(f"{value:.6f}" for value in distances),
                ]
            )
            frame_count += 1
    return {
        "replica": str(replica),
        "frames": frame_count,
        "output": str(output),
        "topology": str(topology),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--states", nargs="+", default=["all"])
    supported_stages = tuple(f"q{index}" for index in range(7))
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=supported_stages,
        default=list(supported_stages),
    )
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.replicas < 1:
        raise SystemExit("replicas must be positive")
    root = Path(args.root).resolve()
    rows = select_rows(
        read_manifest(root / "gromacs_dcm" / "state_manifest.csv"),
        args.states,
        args.stages,
    )
    tasks = [
        (
            root / row["output_dir"] / "setup",
            root / row["output_dir"] / f"replica_{replica:02d}",
        )
        for row in rows
        for replica in range(1, args.replicas + 1)
    ]
    print(f"Selected {len(rows)} states and {len(tasks)} replica trajectories.")
    if args.dry_run:
        for _, replica in tasks:
            print(replica)
        return
    total_frames = 0
    for setup, replica in tasks:
        result = extract_replica(setup, replica)
        total_frames += int(result["frames"])
        print(f"[OK] {result['replica']}: {result['frames']} frames", flush=True)
    print(f"Extracted {total_frames} frames across {len(tasks)} replicas.")


if __name__ == "__main__":
    main()
