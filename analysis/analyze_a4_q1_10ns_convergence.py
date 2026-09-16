#!/usr/bin/env python3
"""Convergence diagnostics for A4 q1 NAC trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from analyze_explicit_dcm_postreaction_nac import (
    RT,
    geometry_arrays,
    gro_box_matrix_A,
    is_same_or_neighbour_arm_pair,
    read_coordinates,
    trajectory_boxes_A,
    write_csv,
)
from analyze_explicit_dcm_q0_geometry_gate import soft_lower, soft_upper


DISTANCE_A = 5.0
DISTANCE_WIDTH_A = 0.5
AXIS_DEG = 45.0
SIDE_DEG = 45.0
ANGLE_WIDTH_DEG = 7.5
BLOCK_PS = 1000.0


def group_probability(
    nac: np.ndarray,
    annotations: list[Mapping[str, object]],
    frame_mask: np.ndarray,
    grouping: str,
    group: str,
) -> float:
    indices = [i for i, row in enumerate(annotations) if row[grouping] == group]
    if not indices:
        raise ValueError(f"no {grouping}={group} candidates")
    return float(np.mean(nac[np.ix_(frame_mask, indices)]))


def ddg(numerator: float, denominator: float) -> float:
    return -RT * math.log(numerator / denominator)


def analyze_manifest(path: Path, neighbour_arms_only: bool = False) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    labels = [str(value) for value in manifest["ordered_vinyl_labels"]]
    times, coordinates = read_coordinates(
        path.parent / manifest["coordinate_file"],
        2 * len(labels),
        int(manifest["frames"]),
    )
    annotations = list(manifest["pair_annotations"])
    if neighbour_arms_only:
        annotations = [row for row in annotations if is_same_or_neighbour_arm_pair(row)]
    if manifest.get("box_file"):
        boxes = trajectory_boxes_A(path.parent / manifest["box_file"], len(times))
    else:
        boxes = gro_box_matrix_A(path.parent / "production_2ns.gro")
    distances, axes, sides, _, valid, _, _ = geometry_arrays(
        coordinates, labels, annotations, boxes
    )
    times = times[valid]
    contact = soft_upper(distances, DISTANCE_A, DISTANCE_WIDTH_A)
    orientation = soft_upper(axes, AXIS_DEG, ANGLE_WIDTH_DEG) * soft_lower(
        sides, SIDE_DEG, ANGLE_WIDTH_DEG
    )
    nac = contact * orientation
    state = str(manifest["state"])
    replica = int(manifest["replica"])
    definitions = (
        ("route_family", "reuse", "switch", "reuse_minus_switch"),
        ("topology", "cross_arm", "same_arm", "cross_minus_same"),
    )

    full_mask = np.ones(len(times), dtype=bool)
    replica_row: dict[str, object] = {
        "state": state,
        "replica": replica,
        "frames": len(times),
        "duration_ps": float(times[-1] - times[0]),
        "candidate_pairs": len(annotations),
    }
    for grouping, numerator, denominator, contrast in definitions:
        p_num = group_probability(nac, annotations, full_mask, grouping, numerator)
        p_den = group_probability(nac, annotations, full_mask, grouping, denominator)
        replica_row[f"{contrast}_numerator_nac"] = p_num
        replica_row[f"{contrast}_denominator_nac"] = p_den
        replica_row[f"{contrast}_odds"] = p_num / p_den
        replica_row[f"{contrast}_ddG_kcal_mol"] = ddg(p_num, p_den)

    block_rows: list[dict[str, object]] = []
    last_block = max(0, int(math.ceil((times[-1] - times[0]) / BLOCK_PS)) - 1)
    block_index = np.minimum(((times - times[0]) / BLOCK_PS).astype(int), last_block)
    for block in sorted(set(block_index)):
        mask = block_index == block
        row: dict[str, object] = {
            "state": state,
            "replica": replica,
            "block": int(block) + 1,
            "start_ps": float(np.min(times[mask])),
            "end_ps": float(np.max(times[mask])),
            "frames": int(np.sum(mask)),
        }
        for grouping, numerator, denominator, contrast in definitions:
            p_num = group_probability(nac, annotations, mask, grouping, numerator)
            p_den = group_probability(nac, annotations, mask, grouping, denominator)
            row[f"{contrast}_numerator_nac"] = p_num
            row[f"{contrast}_denominator_nac"] = p_den
            row[f"{contrast}_odds"] = p_num / p_den
            row[f"{contrast}_ddG_kcal_mol"] = ddg(p_num, p_den)
        block_rows.append(row)

    prefix_rows: list[dict[str, object]] = []
    for end_ps in (2000.0, 4000.0, 6000.0, 8000.0, 10000.0):
        mask = times <= end_ps + 1e-9
        row = {
            "state": state,
            "replica": replica,
            "start_ps": 0.0,
            "end_ps": end_ps,
            "frames": int(np.sum(mask)),
        }
        for grouping, numerator, denominator, contrast in definitions:
            p_num = group_probability(nac, annotations, mask, grouping, numerator)
            p_den = group_probability(nac, annotations, mask, grouping, denominator)
            row[f"{contrast}_numerator_nac"] = p_num
            row[f"{contrast}_denominator_nac"] = p_den
            row[f"{contrast}_odds"] = p_num / p_den
            row[f"{contrast}_ddG_kcal_mol"] = ddg(p_num, p_den)
        prefix_rows.append(row)
    return replica_row, block_rows, prefix_rows


def aggregate_prefix(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, float], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["state"]), float(row["end_ps"]))].append(row)
    output: list[dict[str, object]] = []
    for (state, end_ps), members in sorted(buckets.items()):
        record: dict[str, object] = {
            "state": state,
            "start_ps": 0.0,
            "end_ps": end_ps,
            "replicas": len(members),
        }
        for contrast in ("reuse_minus_switch", "cross_minus_same"):
            p_num = float(np.mean([float(row[f"{contrast}_numerator_nac"]) for row in members]))
            p_den = float(np.mean([float(row[f"{contrast}_denominator_nac"]) for row in members]))
            record[f"{contrast}_odds"] = p_num / p_den
            record[f"{contrast}_ddG_kcal_mol"] = ddg(p_num, p_den)
        output.append(record)
    return output


def aggregate_blocks(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["state"]), int(row["block"]))].append(row)
    output: list[dict[str, object]] = []
    for (state, block), members in sorted(buckets.items()):
        record: dict[str, object] = {
            "state": state,
            "block": block,
            "start_ps": min(float(row["start_ps"]) for row in members),
            "end_ps": max(float(row["end_ps"]) for row in members),
            "replicas": len(members),
        }
        for contrast in ("reuse_minus_switch", "cross_minus_same"):
            p_num = float(np.mean([float(row[f"{contrast}_numerator_nac"]) for row in members]))
            p_den = float(np.mean([float(row[f"{contrast}_denominator_nac"]) for row in members]))
            record[f"{contrast}_odds"] = p_num / p_den
            record[f"{contrast}_ddG_kcal_mol"] = ddg(p_num, p_den)
            record[f"{contrast}_replicas_favoring_numerator"] = sum(
                float(row[f"{contrast}_ddG_kcal_mol"]) < 0 for row in members
            )
        output.append(record)
    return output


def threshold_robustness(path: Path) -> list[dict[str, object]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    buckets: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row["state"], row["contrast"])].append(row)
    output: list[dict[str, object]] = []
    for (state, contrast), members in sorted(buckets.items()):
        points = np.asarray([float(row["ddG_total_nac_point_kcal_mol"]) for row in members])
        lows = np.asarray([float(row["ddG_total_nac_ci_low_kcal_mol"]) for row in members])
        highs = np.asarray([float(row["ddG_total_nac_ci_high_kcal_mol"]) for row in members])
        output.append({
            "state": state,
            "contrast": contrast,
            "settings": len(members),
            "negative_point_estimates": int(np.sum(points < 0)),
            "positive_point_estimates": int(np.sum(points > 0)),
            "ci_entirely_negative": int(np.sum(highs < 0)),
            "ci_entirely_positive": int(np.sum(lows > 0)),
            "point_min_kcal_mol": float(np.min(points)),
            "point_max_kcal_mol": float(np.max(points)),
        })
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--contrasts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--neighbour-arms-only", action="store_true",
        help="apply the B6-style neighbour-arm filter (not appropriate for tetrahedral A4)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifests = sorted(args.input.resolve().rglob("nac_manifest.json"))
    if not manifests:
        raise FileNotFoundError(f"no nac_manifest.json below {args.input}")
    replicas: list[dict[str, object]] = []
    blocks: list[dict[str, object]] = []
    prefixes: list[dict[str, object]] = []
    for manifest in manifests:
        replica, block_rows, prefix_rows = analyze_manifest(manifest, args.neighbour_arms_only)
        replicas.append(replica)
        blocks.extend(block_rows)
        prefixes.extend(prefix_rows)
    aggregate = aggregate_prefix(prefixes)
    block_aggregate = aggregate_blocks(blocks)
    robustness = threshold_robustness(args.contrasts)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "q1_replica_primary.csv", sorted(replicas, key=lambda row: (str(row["state"]), int(row["replica"]))))
    write_csv(args.output / "q1_1ns_blocks.csv", sorted(blocks, key=lambda row: (str(row["state"]), int(row["replica"]), int(row["block"]))))
    write_csv(args.output / "q1_1ns_blocks_aggregate.csv", block_aggregate)
    write_csv(args.output / "q1_prefix_by_replica.csv", sorted(prefixes, key=lambda row: (str(row["state"]), int(row["replica"]), float(row["end_ps"]))))
    write_csv(args.output / "q1_prefix_aggregate.csv", aggregate)
    write_csv(args.output / "q1_threshold_robustness.csv", robustness)
    print(f"manifests={len(manifests)}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
