#!/usr/bin/env python3
"""Analyze compact q1/q2 vinyl-coordinate exports as near-attack proxies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analyze_explicit_dcm_q0_geometry_gate import soft_lower, soft_upper


R_KCAL = 0.00198720425864083
TEMPERATURE_K = 313.15
RT = R_KCAL * TEMPERATURE_K
BOOTSTRAP_REPLICATES = 10_000


@dataclass(frozen=True)
class Setting:
    setting_id: str
    distance_center_A: float
    distance_width_A: float
    axis_max_deg: float
    side_min_deg: float
    angle_width_deg: float


def settings_grid() -> tuple[Setting, ...]:
    return tuple(
        Setting(f"d{distance:g}_axis{axis:g}_side{side:g}", distance, 0.5, axis, side, 7.5)
        for distance in (4.5, 5.0, 5.5)
        for axis in (30.0, 45.0, 60.0)
        for side in (30.0, 45.0, 60.0)
    )


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str] | None = None) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"fields required for empty CSV: {path}")
        fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_coordinates(path: Path, atom_count: int, expected_frames: int) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#@&":
            continue
        rows.append([float(value) for value in stripped.split()])
    values = np.asarray(rows, dtype=float)
    expected_columns = 1 + atom_count * 3
    if values.ndim != 2 or values.shape[1] != expected_columns:
        raise ValueError(f"{path}: expected {expected_columns} columns, got {values.shape}")
    if len(values) != expected_frames:
        raise ValueError(f"{path}: expected {expected_frames} frames, got {len(values)}")
    if np.any(~np.isfinite(values)):
        raise ValueError(f"non-finite coordinates in {path}")
    times = values[:, 0]
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"non-increasing times in {path}")
    return times, values[:, 1:].reshape(len(values), atom_count, 3) * 10.0


def gro_box_matrix_A(path: Path) -> np.ndarray:
    fields = [float(value) for value in path.read_text(encoding="utf-8", errors="replace").splitlines()[-1].split()]
    if len(fields) == 3:
        matrix_nm = np.diag(fields)
    elif len(fields) == 9:
        matrix_nm = np.asarray([
            [fields[0], fields[3], fields[4]],
            [fields[5], fields[1], fields[6]],
            [fields[7], fields[8], fields[2]],
        ])
    else:
        raise ValueError(f"unexpected GRO box line in {path}: {fields}")
    if abs(float(np.linalg.det(matrix_nm))) < 1.0e-8:
        raise ValueError(f"singular periodic box in {path}")
    return matrix_nm * 10.0


def trajectory_boxes_A(path: Path, expected_frames: int) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and stripped[0] not in "#@&":
            rows.append([float(value) for value in stripped.split()])
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape != (expected_frames, 7):
        raise ValueError(f"unexpected gmx traj box table in {path}: {values.shape}")
    # The current protocol uses isotropic orthorhombic boxes; gmx traj writes
    # time, three edge lengths and three off-diagonal/tilt values.
    if np.max(np.abs(values[:, 4:])) > 1.0e-6:
        raise ValueError("triclinic per-frame box export is not supported")
    boxes = np.zeros((expected_frames, 3, 3), dtype=float)
    boxes[:, 0, 0] = values[:, 1] * 10.0
    boxes[:, 1, 1] = values[:, 2] * 10.0
    boxes[:, 2, 2] = values[:, 3] * 10.0
    return boxes


def minimum_image(displacements: np.ndarray, box_A: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(box_A)
    if box_A.ndim == 2:
        fractional = np.asarray(displacements, dtype=float) @ inverse
    elif box_A.ndim == 3:
        fractional = np.einsum("f...i,fij->f...j", np.asarray(displacements, dtype=float), inverse)
    else:
        raise ValueError(f"unexpected box rank: {box_A.ndim}")
    fractional -= np.round(fractional)
    if box_A.ndim == 2:
        return fractional @ box_A
    return np.einsum("f...i,fij->f...j", fractional, box_A)


def geometry_arrays(
    coordinates_A: np.ndarray,
    labels: Sequence[str],
    annotations: Sequence[Mapping[str, object]],
    box_A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], np.ndarray, float, float]:
    label_index = {int(label.removeprefix("V")): index for index, label in enumerate(labels)}
    first = coordinates_A[:, 0::2, :]
    second = coordinates_A[:, 1::2, :]
    vectors = minimum_image(second - first, box_A)
    lengths = np.linalg.norm(vectors, axis=2)
    valid_frames = np.all((lengths >= 1.15) & (lengths <= 1.55), axis=1)
    if float(np.mean(valid_frames)) < 0.90:
        raise ValueError(
            f"fewer than 90% frames have physical minimum-image vinyl bonds: "
            f"{np.mean(valid_frames):.3f}; range={np.min(lengths):.3f}..{np.max(lengths):.3f} A"
        )
    first = first[valid_frames]
    vectors = vectors[valid_frames]
    lengths = lengths[valid_frames]
    if box_A.ndim == 3:
        box_A = box_A[valid_frames]
    axes = vectors / lengths[:, :, None]
    centers = first + 0.5 * vectors
    left = np.asarray([label_index[int(row["left_vinyl"])] for row in annotations])
    right = np.asarray([label_index[int(row["right_vinyl"])] for row in annotations])
    connectors = minimum_image(centers[:, right, :] - centers[:, left, :], box_A)
    distances = np.linalg.norm(connectors, axis=2)
    connector_axes = connectors / distances[:, :, None]
    axis_cosine = np.clip(np.abs(np.sum(axes[:, left, :] * axes[:, right, :], axis=2)), 0.0, 1.0)
    axis_angles = np.degrees(np.arccos(axis_cosine))
    side_left = np.degrees(np.arccos(np.clip(
        np.abs(np.sum(axes[:, left, :] * connector_axes, axis=2)), 0.0, 1.0
    )))
    side_right = np.degrees(np.arccos(np.clip(
        np.abs(np.sum(axes[:, right, :] * connector_axes, axis=2)), 0.0, 1.0
    )))
    side_angles = np.minimum(side_left, side_right)
    return (
        distances, axis_angles, side_angles,
        tuple(str(row["pair"]) for row in annotations), valid_frames,
        float(np.min(lengths)), float(np.max(lengths)),
    )


def is_same_or_neighbour_arm_pair(annotation: Mapping[str, object]) -> bool:
    """Keep same-arm and cyclic nearest-neighbour arm pairs on the A4 core."""
    left = int(annotation["left_arm"])
    right = int(annotation["right_arm"])
    return left == right or (left - right) % 4 in {1, 3}


def analyze_export(
    manifest_path: Path,
    settings: Sequence[Setting],
    *,
    neighbour_arms_only: bool = False,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "explicit-dcm-postreaction-nac-coordinates-v1":
        raise ValueError(f"unsupported NAC export: {manifest_path}")
    labels = [str(value) for value in manifest["ordered_vinyl_labels"]]
    atom_count = 2 * len(labels)
    coordinate_path = manifest_path.parent / manifest["coordinate_file"]
    times, coordinates = read_coordinates(coordinate_path, atom_count, int(manifest["frames"]))
    annotations = list(manifest["pair_annotations"])
    input_pair_count = len(annotations)
    if neighbour_arms_only:
        annotations = [row for row in annotations if is_same_or_neighbour_arm_pair(row)]
    if not annotations:
        raise ValueError(f"no candidate pairs remain after model filtering: {manifest_path}")
    if manifest.get("box_file"):
        box_A = trajectory_boxes_A(manifest_path.parent / manifest["box_file"], len(times))
        pbc_correction = "minimum_image_using_per_frame_gmx_traj_boxes"
    else:
        final_gro = manifest_path.parent / "production_2ns.gro"
        if not final_gro.is_file():
            raise FileNotFoundError(f"periodic box source is missing: {final_gro}")
        box_A = gro_box_matrix_A(final_gro)
        pbc_correction = "minimum_image_using_final_gro_box_with_ambiguous_frames_excluded"
    distances, axes, sides, pair_names, valid_frames, minimum_bond, maximum_bond = geometry_arrays(
        coordinates, labels, annotations, box_A
    )
    input_frame_count = len(times)
    times = times[valid_frames]
    rows: list[dict[str, object]] = []
    for setting in settings:
        contact = soft_upper(distances, setting.distance_center_A, setting.distance_width_A)
        orientation = soft_upper(axes, setting.axis_max_deg, setting.angle_width_deg) * soft_lower(
            sides, setting.side_min_deg, setting.angle_width_deg
        )
        nac = contact * orientation
        for index, annotation in enumerate(annotations):
            contact_mass = float(np.sum(contact[:, index]))
            nac_mass = float(np.sum(nac[:, index]))
            rows.append({
                "state": manifest["state"],
                "system": manifest["system"],
                "stage": manifest["stage"],
                "branch": manifest["branch"],
                "replica": int(manifest["replica"]),
                "setting_id": setting.setting_id,
                "distance_center_A": setting.distance_center_A,
                "distance_width_A": setting.distance_width_A,
                "axis_max_deg": setting.axis_max_deg,
                "side_min_deg": setting.side_min_deg,
                "angle_width_deg": setting.angle_width_deg,
                "pair": pair_names[index],
                "topology": annotation["topology"],
                "history_class": annotation["history_class"],
                "route_family": annotation["route_family"],
                "left_arm": annotation["left_arm"],
                "right_arm": annotation["right_arm"],
                "frames": len(times),
                "duration_ps": float(times[-1] - times[0]),
                "mean_soft_contact": contact_mass / len(times),
                "conditional_orientation_gate": nac_mass / contact_mass,
                "mean_soft_nac": nac_mass / len(times),
                "contact_weighted_axis_angle_deg": float(np.sum(contact[:, index] * axes[:, index]) / contact_mass),
                "contact_weighted_side_angle_deg": float(np.sum(contact[:, index] * sides[:, index]) / contact_mass),
            })
    quality = {
        "manifest": str(manifest_path),
        "state": manifest["state"],
        "system": manifest["system"],
        "stage": manifest["stage"],
        "branch": manifest["branch"],
        "replica": int(manifest["replica"]),
        "frames": len(times),
        "input_frames": input_frame_count,
        "pbc_ambiguous_frames_excluded": input_frame_count - len(times),
        "retained_frame_fraction": len(times) / input_frame_count,
        "duration_ps": float(times[-1] - times[0]),
        "frame_dt_ps": float(np.median(np.diff(times))),
        "pair_count": len(annotations),
        "input_pair_count": input_pair_count,
        "candidate_scope": (
            "same_or_cyclic_neighbour_arms"
            if neighbour_arms_only
            else "all_active_vinyl_pairs"
        ),
        "minimum_retained_vinyl_bond_A": minimum_bond,
        "maximum_retained_vinyl_bond_A": maximum_bond,
        "periodic_box_A": box_A.tolist() if box_A.ndim == 2 else "per_frame",
        "pbc_correction": pbc_correction,
        "status": "PASS",
    }
    return rows, quality


def group_replica_rows(pair_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    groupings = (
        ("topology", "topology"),
        ("route_family", "route_family"),
        ("history_class", "history_class"),
    )
    for row in pair_rows:
        prefix = (
            row["state"], row["system"], row["stage"], row["branch"],
            row["replica"], row["setting_id"],
        )
        for grouping, field in groupings:
            buckets[prefix + (grouping, row[field])].append(row)
    result: list[dict[str, object]] = []
    for key, rows in sorted(buckets.items(), key=lambda item: tuple(str(x) for x in item[0])):
        state, system, stage, branch, replica, setting_id, grouping, group = key
        contact = float(np.mean([float(row["mean_soft_contact"]) for row in rows]))
        nac = float(np.mean([float(row["mean_soft_nac"]) for row in rows]))
        result.append({
            "state": state,
            "system": system,
            "stage": stage,
            "branch": branch,
            "replica": replica,
            "setting_id": setting_id,
            "grouping": grouping,
            "group": group,
            "pair_count": len(rows),
            "per_pair_contact": contact,
            "conditional_orientation": nac / contact,
            "per_pair_nac": nac,
            "total_group_nac_including_multiplicity": nac * len(rows),
        })
    return result


def summarize_group_rows(group_rows: Sequence[Mapping[str, object]], bootstrap: int) -> list[dict[str, object]]:
    buckets: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in group_rows:
        key = tuple(row[field] for field in (
            "state", "system", "stage", "branch", "setting_id", "grouping", "group"
        ))
        buckets[key].append(row)
    metrics = ("per_pair_contact", "conditional_orientation", "per_pair_nac", "total_group_nac_including_multiplicity")
    output: list[dict[str, object]] = []
    for key, rows in sorted(buckets.items(), key=lambda item: tuple(str(x) for x in item[0])):
        state, system, stage, branch, setting_id, grouping, group = key
        arrays = {metric: np.asarray([float(row[metric]) for row in rows]) for metric in metrics}
        rng = np.random.default_rng(stable_seed("group", *key))
        indices = rng.integers(0, len(rows), size=(bootstrap, len(rows)))
        record: dict[str, object] = {
            "state": state, "system": system, "stage": stage, "branch": branch,
            "setting_id": setting_id, "grouping": grouping, "group": group,
            "replicas": len(rows), "pair_count": int(rows[0]["pair_count"]),
        }
        for metric, values in arrays.items():
            point = float(np.mean(values))
            draws = np.mean(values[indices], axis=1)
            low, median, high = np.quantile(draws, (0.025, 0.5, 0.975))
            record[f"{metric}_point"] = point
            record[f"{metric}_median"] = float(median)
            record[f"{metric}_ci_low"] = float(low)
            record[f"{metric}_ci_high"] = float(high)
        output.append(record)
    return output


def contrast_rows(group_rows: Sequence[Mapping[str, object]], bootstrap: int) -> list[dict[str, object]]:
    by_state: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in group_rows:
        key = tuple(row[field] for field in ("state", "system", "stage", "branch", "setting_id", "grouping"))
        by_state[key].append(row)
    definitions = {
        "topology": ("cross_arm", "same_arm", "cross_minus_same"),
        "route_family": ("reuse", "switch", "reuse_minus_switch"),
    }
    results: list[dict[str, object]] = []
    for key, rows in sorted(by_state.items(), key=lambda item: tuple(str(x) for x in item[0])):
        state, system, stage, branch, setting_id, grouping = key
        if grouping not in definitions:
            continue
        numerator_name, denominator_name, contrast_name = definitions[str(grouping)]
        by_group: dict[str, dict[int, Mapping[str, object]]] = defaultdict(dict)
        for row in rows:
            by_group[str(row["group"])][int(row["replica"])] = row
        common = sorted(set(by_group[numerator_name]) & set(by_group[denominator_name]))
        if not common:
            continue
        contact_num = np.asarray([float(by_group[numerator_name][rep]["per_pair_contact"]) for rep in common])
        contact_den = np.asarray([float(by_group[denominator_name][rep]["per_pair_contact"]) for rep in common])
        nac_num = np.asarray([float(by_group[numerator_name][rep]["per_pair_nac"]) for rep in common])
        nac_den = np.asarray([float(by_group[denominator_name][rep]["per_pair_nac"]) for rep in common])

        def ddg(left: np.ndarray, right: np.ndarray) -> float:
            return -RT * math.log(float(np.mean(left)) / float(np.mean(right)))

        point_contact = ddg(contact_num, contact_den)
        point_total = ddg(nac_num, nac_den)
        point_orient = point_total - point_contact
        rng = np.random.default_rng(stable_seed("contrast", *key))
        indices = rng.integers(0, len(common), size=(bootstrap, len(common)))
        draws_contact = -RT * np.log(np.mean(contact_num[indices], axis=1) / np.mean(contact_den[indices], axis=1))
        draws_total = -RT * np.log(np.mean(nac_num[indices], axis=1) / np.mean(nac_den[indices], axis=1))
        draws_orient = draws_total - draws_contact
        record: dict[str, object] = {
            "state": state, "system": system, "stage": stage, "branch": branch,
            "setting_id": setting_id, "contrast": contrast_name,
            "numerator_group": numerator_name, "denominator_group": denominator_name,
            "replicas": len(common),
        }
        for component, point, draws in (
            ("contact", point_contact, draws_contact),
            ("orientation", point_orient, draws_orient),
            ("total_nac", point_total, draws_total),
        ):
            low, median, high = np.quantile(draws, (0.025, 0.5, 0.975))
            record[f"ddG_{component}_point_kcal_mol"] = point
            record[f"ddG_{component}_median_kcal_mol"] = float(median)
            record[f"ddG_{component}_ci_low_kcal_mol"] = float(low)
            record[f"ddG_{component}_ci_high_kcal_mol"] = float(high)
            record[f"{component}_odds_point"] = math.exp(-point / RT)
        results.append(record)
    return results


def write_report(output: Path, contrast: Sequence[Mapping[str, object]], quality: Sequence[Mapping[str, object]]) -> None:
    primary = [row for row in contrast if row["setting_id"] == "d5_axis45_side45"]
    retained = [float(row["retained_frame_fraction"]) for row in quality]
    fallback_count = sum(
        row["pbc_correction"] == "minimum_image_using_final_gro_box_with_ambiguous_frames_excluded"
        for row in quality
    )
    lines = [
        "# A4-m-C5–C8 q1/q2 near-attack geometry analysis", "",
        "Primary gate: midpoint distance 5.0 Å (width 0.5 Å), vinyl-axis angle ≤45° and side-on angle ≥45° (angular width 7.5°).",
        "All free energies are catalyst-free effective proxies at 313.15 K. Negative values favor the numerator group.", "",
        "| state | contrast | replicas | ΔΔG contact | ΔΔG orientation | ΔΔG total NAC | 95% replica-bootstrap CI | odds (numerator/denominator) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in primary:
        replicas = int(row["replicas"])
        ci_text = (
            "not estimable"
            if replicas < 2
            else (
                f"[{float(row['ddG_total_nac_ci_low_kcal_mol']):+.3f}, "
                f"{float(row['ddG_total_nac_ci_high_kcal_mol']):+.3f}]"
            )
        )
        lines.append(
            f"| {row['state']} | {row['contrast']} | {row['replicas']} | "
            f"{float(row['ddG_contact_point_kcal_mol']):+.3f} | "
            f"{float(row['ddG_orientation_point_kcal_mol']):+.3f} | "
            f"{float(row['ddG_total_nac_point_kcal_mol']):+.3f} | "
            f"{ci_text} | "
            f"{float(row['total_nac_odds_point']):.4g} |"
        )
    lines.extend([
        "", "## Primary result", "",
    ])
    for row in primary:
        point = float(row["ddG_total_nac_point_kcal_mol"])
        low = float(row["ddG_total_nac_ci_low_kcal_mol"])
        high = float(row["ddG_total_nac_ci_high_kcal_mol"])
        replicas = int(row["replicas"])
        if replicas < 3:
            favored = (
                row["numerator_group"] if point < 0.0 else row["denominator_group"]
            )
            verdict = (
                f"pilot point estimate favors {favored}; independent-replica "
                "uncertainty is unresolved"
            )
        elif high < 0.0:
            verdict = f"robustly favors {row['numerator_group']}"
        elif low > 0.0:
            verdict = f"robustly favors {row['denominator_group']}"
        else:
            verdict = "is unresolved across independent replicas"
        contact = abs(float(row["ddG_contact_point_kcal_mol"]))
        orientation = abs(float(row["ddG_orientation_point_kcal_mol"]))
        dominant = "contact accessibility" if contact >= orientation else "orientation gating"
        uncertainty = (
            "replica CI not estimable"
            if replicas < 2
            else f"95% CI [{low:+.3f}, {high:+.3f}]"
        )
        lines.append(
            f"- `{row['state']}` / `{row['contrast']}`: ΔΔG={point:+.3f} kcal mol⁻¹ "
            f"({uncertainty}); {verdict}. The larger decomposition term is {dominant}."
        )
    lines.extend([
        "", "## Interpretation boundary", "",
        "The per-pair comparison removes route multiplicity. `topology` compares cross-arm with same-arm candidates; `route_family` compares reuse of a previously reacted pair of molecular arms with alternative cross-arm switches.",
        f"Candidate scope: `{quality[0]['candidate_scope']}`. In `same_or_cyclic_neighbour_arms` mode, cross-arm pairs on opposite A4 arms are excluded by the model assumption and cannot contribute to reuse/switch probabilities.",
        "Replica bootstrap treats independent GROMACS replicas as the uncertainty unit. With fewer than three replicas, confidence intervals are only pilot diagnostics.",
        "These values quantify substrate preorganization and orientation, not a Ru-catalyzed activation barrier.", "",
        "## Periodic-boundary quality control", "",
        f"Quality-controlled exports: {len(quality)}. Retained frame fraction: {min(retained):.3%}–{max(retained):.3%}.",
    ])
    if fallback_count:
        lines.append(
            f"{fallback_count} export(s) did not contain per-frame box vectors, so minimum-image reconstruction used the final GRO box and discarded frames whose reconstructed vinyl bonds remained nonphysical."
        )
    else:
        lines.append("All exports used per-frame box vectors; no final-GRO periodic-box fallback was needed.")
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="directories containing nac_manifest.json exports")
    parser.add_argument("--output", type=Path, default=Path("results/a4_postreaction_nac"))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument(
        "--neighbour-arms-only",
        action="store_true",
        help="exclude cross pairs on opposite A4 arms; retain same-arm and cyclic neighbours",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifests = sorted({path.resolve() for root in args.inputs for path in root.rglob("nac_manifest.json")})
    if not manifests:
        raise FileNotFoundError("no nac_manifest.json files found")
    settings = settings_grid()
    pair_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    for manifest in manifests:
        rows, quality = analyze_export(
            manifest,
            settings,
            neighbour_arms_only=args.neighbour_arms_only,
        )
        pair_rows.extend(rows)
        quality_rows.append(quality)
    group_rows = group_replica_rows(pair_rows)
    summary_rows = summarize_group_rows(group_rows, args.bootstrap)
    contrasts = contrast_rows(group_rows, args.bootstrap)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "pair_nac_statistics.csv", pair_rows)
    write_csv(output / "replica_group_statistics.csv", group_rows)
    write_csv(output / "group_bootstrap_summary.csv", summary_rows)
    write_csv(output / "free_energy_contrasts.csv", contrasts, fields=(
        "state", "system", "stage", "branch", "setting_id", "contrast",
        "numerator_group", "denominator_group", "replicas",
        "ddG_contact_point_kcal_mol", "ddG_contact_median_kcal_mol",
        "ddG_contact_ci_low_kcal_mol", "ddG_contact_ci_high_kcal_mol", "contact_odds_point",
        "ddG_orientation_point_kcal_mol", "ddG_orientation_median_kcal_mol",
        "ddG_orientation_ci_low_kcal_mol", "ddG_orientation_ci_high_kcal_mol", "orientation_odds_point",
        "ddG_total_nac_point_kcal_mol", "ddG_total_nac_median_kcal_mol",
        "ddG_total_nac_ci_low_kcal_mol", "ddG_total_nac_ci_high_kcal_mol", "total_nac_odds_point",
    ))
    write_csv(output / "data_quality_checks.csv", quality_rows)
    write_report(output, contrasts, quality_rows)
    print(f"exports={len(manifests)}")
    print(f"states={len({row['state'] for row in quality_rows})}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
