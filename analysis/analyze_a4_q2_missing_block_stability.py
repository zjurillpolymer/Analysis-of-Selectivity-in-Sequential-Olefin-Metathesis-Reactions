#!/usr/bin/env python3
"""Diagnose 500 ps block stability of the A4 q2 missing-history NAC runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import analyze_explicit_dcm_postreaction_nac as nac
from analyze_explicit_dcm_q0_geometry_gate import soft_lower, soft_upper


PRIMARY = nac.Setting("d5_axis45_side45", 5.0, 0.5, 45.0, 45.0, 7.5)


def geometry(manifest_path: Path) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = [str(value) for value in manifest["ordered_vinyl_labels"]]
    times, coordinates = nac.read_coordinates(
        manifest_path.parent / str(manifest["coordinate_file"]),
        2 * len(labels),
        int(manifest["frames"]),
    )
    boxes = nac.trajectory_boxes_A(
        manifest_path.parent / str(manifest["box_file"]), len(times)
    )
    annotations = list(manifest["pair_annotations"])
    distances, axes, sides, _, valid, _, _ = nac.geometry_arrays(
        coordinates, labels, annotations, boxes
    )
    return manifest, times[valid], distances, axes, sides, annotations


def per_pair_group_mean(
    values: np.ndarray,
    annotations: Sequence[Mapping[str, object]],
    field: str,
    group: str,
) -> float:
    indices = [index for index, row in enumerate(annotations) if row[field] == group]
    if not indices:
        raise ValueError(f"no pairs for {field}={group}")
    return float(np.mean(values[:, indices]))


def analyze_manifest(manifest_path: Path, block_ps: float) -> list[dict[str, object]]:
    manifest, times, distances, axes, sides, annotations = geometry(manifest_path)
    contact = soft_upper(distances, PRIMARY.distance_center_A, PRIMARY.distance_width_A)
    orientation = soft_upper(axes, PRIMARY.axis_max_deg, PRIMARY.angle_width_deg) * soft_lower(
        sides, PRIMARY.side_min_deg, PRIMARY.angle_width_deg
    )
    values = contact * orientation
    definitions = (
        ("topology", "cross_arm", "same_arm", "cross_minus_same"),
        ("route_family", "reuse", "switch", "reuse_minus_switch"),
    )
    duration = float(times[-1] - times[0])
    boundaries = np.arange(float(times[0]), float(times[-1]) + block_ps, block_ps)
    if boundaries[-1] < times[-1]:
        boundaries = np.append(boundaries, times[-1])
    rows: list[dict[str, object]] = []
    for block_index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:]), 1):
        mask = (times >= start) & ((times < stop) if stop < times[-1] else (times <= stop))
        if int(np.sum(mask)) < 2:
            continue
        for field, numerator, denominator, contrast in definitions:
            available = {str(row[field]) for row in annotations}
            if numerator not in available or denominator not in available:
                continue
            numerator_value = per_pair_group_mean(values[mask], annotations, field, numerator)
            denominator_value = per_pair_group_mean(values[mask], annotations, field, denominator)
            ddg = -nac.RT * math.log(numerator_value / denominator_value)
            rows.append(
                {
                    "state": manifest["state"],
                    "system": manifest["system"],
                    "branch": manifest["branch"],
                    "replica": int(manifest["replica"]),
                    "contrast": contrast,
                    "numerator_group": numerator,
                    "denominator_group": denominator,
                    "block": block_index,
                    "start_ps": float(start),
                    "stop_ps": float(stop),
                    "frames": int(np.sum(mask)),
                    "ddG_total_nac_kcal_mol": ddg,
                    "numerator_per_pair_nac": numerator_value,
                    "denominator_per_pair_nac": denominator_value,
                    "trajectory_duration_ps": duration,
                }
            )
    return rows


def summarize(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str, int], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row["state"]), str(row["contrast"]), int(row["replica"]))].append(row)
    output = []
    for (state, contrast, replica), values in sorted(buckets.items()):
        values = sorted(values, key=lambda row: int(row["block"]))
        ddg = [float(row["ddG_total_nac_kcal_mol"]) for row in values]
        signs = [value < 0.0 for value in ddg]
        output.append(
            {
                "state": state,
                "contrast": contrast,
                "replica": replica,
                "blocks": len(ddg),
                "block_ddG_kcal_mol": ";".join(f"{value:+.6f}" for value in ddg),
                "minimum_block_ddG_kcal_mol": min(ddg),
                "maximum_block_ddG_kcal_mol": max(ddg),
                "block_range_kcal_mol": max(ddg) - min(ddg),
                "direction_consistent_across_blocks": len(set(signs)) == 1,
                "sign_changes": sum(left != right for left, right in zip(signs[:-1], signs[1:])),
            }
        )
    return output


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-ps", type=float, default=500.0)
    args = parser.parse_args()
    if args.block_ps <= 0:
        raise SystemExit("block-ps must be positive")
    manifests = sorted(
        {path.resolve() for root in args.inputs for path in root.rglob("nac_manifest.json")}
    )
    if not manifests:
        raise FileNotFoundError("no NAC manifests found")
    rows = [row for manifest in manifests for row in analyze_manifest(manifest, args.block_ps)]
    summary = summarize(rows)
    output = args.output.resolve()
    write_csv(output / "block_contrasts.csv", rows)
    write_csv(output / "block_stability_summary.csv", summary)
    print(f"exports={len(manifests)}")
    print(f"block_rows={len(rows)}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
