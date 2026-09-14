#!/usr/bin/env python3
"""Convert B6 q0--q3 long trajectories into 25 ps first-event kernels.

The legacy product model was trained on a discrete competing-risk observable,
not on whole-trajectory contact occupancy.  This script chops each long MD
trajectory into non-overlapping 25 ps interval blocks and evaluates the same
distance-dependent first-event law (5 A center, 0.5 A width, kappa 0.01/ps).
Replica means are kept separate so downstream uncertainty can resample the
three independent trajectories rather than treating blocks as independent.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

import analyze_explicit_dcm_rare_event_cycles as hazard


PAIR = re.compile(r"^V(\d+)_V(\d+)_(?:同臂|跨臂)$")
REPLICA = re.compile(r"replica_(\d+)")


@dataclass(frozen=True)
class StateInput:
    state: str
    pattern: str
    trajectory_ns: int


DEFAULT_INPUTS = (
    StateInput(
        "B6_p_C6_q0",
        "results/b6_p_c6_paper_auto/stage_50ns/**/gromacs_dcm/runs/B6_p_C6_q0/replica_*/vinyl_distances.csv",
        50,
    ),
    StateInput(
        "B6_p_C6_q1_V3_V6_cis",
        "results/b6_q1_v3_v6_50ns_download/**/gromacs_dcm/runs/B6_p_C6_q1_V3_V6_cis/replica_*/vinyl_distances.csv",
        50,
    ),
    StateInput(
        "B6_p_C6_q2_V3_V6_V11_V12_cis",
        "results/b6_q2_v3_v6_v11_v12_50ns_download/**/gromacs_dcm/runs/B6_p_C6_q2_V3_V6_V11_V12_cis/replica_*/vinyl_distances.csv",
        50,
    ),
    StateInput(
        "B6_p_C6_q3_V3_V6_V11_V12_V8_V10_cis",
        "results/b6_q3_20ns_merged/**/gromacs_dcm/runs/B6_p_C6_q3_V3_V6_V11_V12_V8_V10_cis/replica_*/vinyl_distances.csv",
        20,
    ),
    StateInput(
        "B6_p_C6_q3_V3_V6_V11_V12_V7_V9_cis",
        "results/b6_q3_20ns_merged/**/gromacs_dcm/runs/B6_p_C6_q3_V3_V6_V11_V12_V7_V9_cis/replica_*/vinyl_distances.csv",
        20,
    ),
)


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def locate(root: Path, item: StateInput) -> list[Path]:
    paths = sorted(root.glob(item.pattern))
    by_replica: dict[int, list[Path]] = defaultdict(list)
    for path in paths:
        match = REPLICA.search(str(path))
        if match:
            by_replica[int(match.group(1))].append(path)
    selected: list[Path] = []
    for replica in (1, 2, 3):
        candidates = by_replica.get(replica, [])
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"{item.state} replica {replica}: expected one file, found {candidates}"
            )
        selected.append(candidates[0])
    return selected


def load_distance_csv(path: Path) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    if header[:3] != ["frame", "time_ps", "absolute_time_ps"]:
        raise ValueError(f"unexpected distance header: {path}")
    edges: list[tuple[int, int]] = []
    for name in header[3:]:
        match = PAIR.fullmatch(name)
        if match is None:
            raise ValueError(f"unrecognized pair column {name}: {path}")
        edges.append((int(match.group(1)), int(match.group(2))))
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if values.shape[1] != len(header) or not np.isfinite(values).all():
        raise ValueError(f"invalid numeric table: {path}")
    times = values[:, 1]
    distances = values[:, 3:]
    if np.any(distances <= 0.0):
        raise ValueError(f"non-positive distance: {path}")
    return edges, times, distances


def block_probabilities(
    times: np.ndarray,
    distances: np.ndarray,
    window_ps: float,
    center_A: float,
    width_A: float,
    kappa_per_ps: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    dt = float(np.median(np.diff(times)))
    if dt <= 0.0 or np.max(np.abs(np.diff(times) - dt)) > 1.0e-6:
        raise ValueError("trajectory time grid is not uniform")
    intervals = int(round(window_ps / dt))
    if not math.isclose(intervals * dt, window_ps, abs_tol=1.0e-8):
        raise ValueError(f"window {window_ps} ps is not divisible by dt={dt}")
    blocks = (len(times) - 1) // intervals
    if blocks < 1:
        raise ValueError("trajectory is shorter than one analysis window")
    probabilities: list[np.ndarray] = []
    survivals: list[float] = []
    multipliers = np.ones(distances.shape[1], dtype=float)
    for block in range(blocks):
        start = block * intervals
        stop = start + intervals + 1
        pair_probability, survival = hazard.competing_risk_probabilities(
            distances[start:stop], dt, center_A, width_A, kappa_per_ps, multipliers
        )
        probabilities.append(pair_probability)
        survivals.append(survival)
    return np.stack(probabilities), np.asarray(survivals), dt


def analyze(
    root: Path,
    output: Path,
    window_ps: float,
    center_A: float,
    width_A: float,
    kappa_per_ps: float,
) -> None:
    replica_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    state_vectors: dict[str, list[tuple[list[tuple[int, int]], np.ndarray]]] = defaultdict(list)
    inventory: list[dict[str, object]] = []
    for item in DEFAULT_INPUTS:
        for path in locate(root, item):
            replica = int(REPLICA.search(str(path)).group(1))
            edges, times, distances = load_distance_csv(path)
            blocks, survivals, dt = block_probabilities(
                times, distances, window_ps, center_A, width_A, kappa_per_ps
            )
            mean = np.mean(blocks, axis=0)
            conditioned = mean / float(np.sum(mean))
            state_vectors[item.state].append((edges, mean))
            inventory.append(
                {
                    "state": item.state,
                    "replica": replica,
                    "path": str(path.relative_to(root)),
                    "frames": len(times),
                    "dt_ps": dt,
                    "blocks": len(blocks),
                    "mean_event_probability": 1.0 - float(np.mean(survivals)),
                }
            )
            for (left, right), raw, cond in zip(edges, mean, conditioned):
                replica_rows.append(
                    {
                        "state": item.state,
                        "system": "B6_p_C6",
                        "replica": replica,
                        "pair": f"V{left}_V{right}",
                        "left": left,
                        "right": right,
                        "first_event_probability_unconditional_mean": float(raw),
                        "first_event_probability_conditioned": float(cond),
                        "windows": len(blocks),
                    }
                )

    for item in DEFAULT_INPUTS:
        records = state_vectors[item.state]
        if len(records) != 3:
            raise ValueError(f"{item.state} does not have three replica vectors")
        reference_edges = records[0][0]
        if any(edges != reference_edges for edges, _ in records[1:]):
            raise ValueError(f"replica pair schema mismatch: {item.state}")
        mean = np.mean(np.stack([vector for _, vector in records]), axis=0)
        conditioned = mean / float(np.sum(mean))
        for (left, right), raw, cond in zip(reference_edges, mean, conditioned):
            state_rows.append(
                {
                    "state": item.state,
                    "system": "B6_p_C6",
                    "pair": f"V{left}_V{right}",
                    "left": left,
                    "right": right,
                    "first_event_probability_unconditional_mean": float(raw),
                    "first_event_probability_conditioned": float(cond),
                    "replicas": 3,
                    "trajectory_ns": item.trajectory_ns,
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    write_csv(
        output / "pair_state_summary.csv", state_rows,
        ["state", "system", "pair", "left", "right",
         "first_event_probability_unconditional_mean",
         "first_event_probability_conditioned", "replicas", "trajectory_ns"],
    )
    write_csv(
        output / "pair_replica_summary.csv", replica_rows,
        ["state", "system", "replica", "pair", "left", "right",
         "first_event_probability_unconditional_mean",
         "first_event_probability_conditioned", "windows"],
    )
    write_csv(
        output / "trajectory_inventory.csv", inventory,
        ["state", "replica", "path", "frames", "dt_ps", "blocks",
         "mean_event_probability"],
    )
    contract = {
        "analysis": "nonoverlapping_long_md_blocks_to_competing_risk_first_event",
        "window_ps": window_ps,
        "center_A": center_A,
        "width_A": width_A,
        "kappa_per_ps": kappa_per_ps,
        "same_arm_multiplier": 1.0,
        "independence_unit_for_downstream_bootstrap": "replica",
        "warning": (
            "25 ps blocks within a replica are correlated and are used only to estimate "
            "that replica's mean kernel; they are not treated as independent replicas. "
            "The 0.5 ps saved interval is coarser than the legacy 0.05 ps cycles."
        ),
    }
    (output / "analysis_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote competing-risk kernels to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/b6_long_md_competing_risk_kernels"),
    )
    parser.add_argument("--window-ps", type=float, default=25.0)
    parser.add_argument("--center-A", type=float, default=5.0)
    parser.add_argument("--width-A", type=float, default=0.5)
    parser.add_argument("--kappa-per-ps", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output = root / args.output if not args.output.is_absolute() else args.output
    analyze(root, output, args.window_ps, args.center_A, args.width_A, args.kappa_per_ps)


if __name__ == "__main__":
    main()
