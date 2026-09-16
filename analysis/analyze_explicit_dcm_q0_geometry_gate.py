#!/usr/bin/env python3
"""Estimate the same/cross-arm geometry gate from explicit-DCM q0 trajectories.

This is a catalyst-free, label-free physical proxy for ``epsilon0``.  For
candidate vinyl pair ``e`` and frame ``t`` it computes

    contact_e(t) = sigmoid((d0 - midpoint_distance) / sigma_d)

and a smooth cofacial near-attack proxy

    orient_e(t) = parallel_axis_gate * side_on_approach_gate.

The conditional orientation pass fraction is

    kappa_e = sum(contact * orient) / sum(contact).

Consequently the official distance-only q0 contribution is divided out.  The
primary same/cross multiplier is the equal-starting-cluster geometric mean of
the q0-mass-weighted conditional fractions:

    epsilon_geom = GM_clusters(kappa_same / kappa_cross).

At 313.15 K, ``-RT log(epsilon_geom)`` is reported as a geometry-gating free
energy proxy.  It is not a catalyst-aware metathesis activation barrier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


CACHE_VERSION = 1
DEFAULT_STATES = ("A4_m_C5_q0",)
DEFAULT_DISTANCE_CENTERS_A = (4.5, 5.0, 5.5)
DEFAULT_AXIS_MAXIMA_DEG = (30.0, 45.0, 60.0)
DEFAULT_SIDE_MINIMA_DEG = (30.0, 45.0, 60.0)
GAS_CONSTANT_KCAL_MOL_K = 1.98720425864083e-3


@dataclass(frozen=True)
class GateSetting:
    setting_id: str
    distance_center_A: float
    distance_width_A: float
    axis_max_deg: float
    angle_width_deg: float
    side_min_deg: float


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str] | None = None,
) -> None:
    if fields is None:
        if not rows:
            raise ValueError(f"field names are required for empty CSV: {path}")
        fields = tuple(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join(str(part) for part in (base_seed, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "little")


def soft_upper(values: np.ndarray, cutoff: float, width: float) -> np.ndarray:
    if width <= 0.0:
        raise ValueError("soft-cutoff width must be positive")
    z = np.clip((np.asarray(values, dtype=float) - cutoff) / width, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(z))


def soft_lower(values: np.ndarray, cutoff: float, width: float) -> np.ndarray:
    return soft_upper(-np.asarray(values, dtype=float), -cutoff, width)


def pair_geometry(
    positions_A: np.ndarray,
    mapping: Mapping[str, Sequence[int]],
    pair_schema: Sequence[Mapping[str, object]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return distance, acute axis angle, side-on angle, and vinyl lengths.

    The two mapped atoms define an undirected vinyl axis.  The side-on angle is
    the smaller acute angle between either vinyl axis and the midpoint connector;
    90 degrees is an ideal side-by-side approach and 0 degrees is end-on.
    """
    positions = np.asarray(positions_A, dtype=float)
    vinyls = sorted(mapping, key=lambda name: int(name.removeprefix("V")))
    atom_pairs = np.asarray(
        [[int(serial) - 1 for serial in mapping[name]] for name in vinyls], dtype=int
    )
    if atom_pairs.ndim != 2 or atom_pairs.shape[1] != 2:
        raise ValueError("every vinyl mapping must contain exactly two atoms")
    vectors = positions[atom_pairs[:, 1]] - positions[atom_pairs[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    if np.any(~np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("invalid mapped vinyl bond length")
    axes = vectors / lengths[:, None]
    centers = 0.5 * (positions[atom_pairs[:, 0]] + positions[atom_pairs[:, 1]])
    # Pair schemas retain the original vinyl labels after reaction.  Those labels
    # are consecutive for q0 (V1...Vn), but become sparse for q1/q2 states (for
    # example V1, V2, V4, V6).  Therefore a vinyl label is not an array index.
    # Resolve every schema endpoint through the active mapping explicitly.
    vinyl_index_by_label: dict[int, int] = {}
    for index, name in enumerate(vinyls):
        label = int(name.removeprefix("V"))
        if label in vinyl_index_by_label:
            raise ValueError(f"duplicate vinyl label V{label} in mapping")
        vinyl_index_by_label[label] = index

    def endpoint_indices(field: str) -> np.ndarray:
        labels = [int(pair[field]) for pair in pair_schema]
        missing = sorted({label for label in labels if label not in vinyl_index_by_label})
        if missing:
            missing_text = ", ".join(f"V{label}" for label in missing)
            raise ValueError(
                f"pair schema references vinyl labels absent from mapping: {missing_text}"
            )
        return np.asarray([vinyl_index_by_label[label] for label in labels], dtype=int)

    left = endpoint_indices("left")
    right = endpoint_indices("right")
    connectors = centers[right] - centers[left]
    distances = np.linalg.norm(connectors, axis=1)
    if np.any(~np.isfinite(distances)) or np.any(distances <= 0.0):
        raise ValueError("invalid vinyl-pair midpoint distance")
    connector_axes = connectors / distances[:, None]
    axis_cosine = np.clip(
        np.abs(np.sum(axes[left] * axes[right], axis=1)), 0.0, 1.0
    )
    axis_angle = np.degrees(np.arccos(axis_cosine))
    side_left = np.degrees(
        np.arccos(
            np.clip(np.abs(np.sum(axes[left] * connector_axes, axis=1)), 0.0, 1.0)
        )
    )
    side_right = np.degrees(
        np.arccos(
            np.clip(np.abs(np.sum(axes[right] * connector_axes, axis=1)), 0.0, 1.0)
        )
    )
    side_angle = np.minimum(side_left, side_right)
    return distances, axis_angle, side_angle, lengths


def build_settings(
    distance_centers: Sequence[float],
    distance_width: float,
    axis_maxima: Sequence[float],
    angle_width: float,
    side_minima: Sequence[float],
) -> tuple[GateSetting, ...]:
    if distance_width <= 0.0 or angle_width <= 0.0:
        raise ValueError("gate widths must be positive")
    settings = []
    for distance in distance_centers:
        for axis in axis_maxima:
            for side in side_minima:
                identifier = f"d{distance:g}_axis{axis:g}_side{side:g}"
                settings.append(
                    GateSetting(identifier, distance, distance_width, axis, angle_width, side)
                )
    if len({item.setting_id for item in settings}) != len(settings):
        raise ValueError("gate setting IDs are not unique")
    return tuple(settings)


def select_manifest_rows(
    rows: Sequence[dict[str, str]],
    states: Sequence[str],
    cycles: int,
    state_cycles: Mapping[str, int] | None = None,
) -> list[dict[str, str]]:
    requested = set(states)
    unknown = requested - {row["state"] for row in rows}
    if unknown:
        raise ValueError(f"states absent from cycle manifest: {sorted(unknown)}")
    targets = {state: cycles for state in states}
    if state_cycles:
        unexpected = set(state_cycles) - requested
        if unexpected:
            raise ValueError(f"state-cycle entries were not selected: {sorted(unexpected)}")
        targets.update(state_cycles)
    if any(value < 1 for value in targets.values()):
        raise ValueError("all requested cycle counts must be positive")
    selected = [
        row
        for row in rows
        if row["state"] in requested and int(row["cycle"]) <= targets[row["state"]]
    ]
    expected = sum(targets.values())
    if len(selected) != expected:
        raise ValueError(f"selected {len(selected)} cycles, expected {expected}")
    return sorted(selected, key=lambda row: (row["state"], int(row["cycle"])))


def parse_state_cycles(values: Sequence[str]) -> dict[str, int]:
    output: dict[str, int] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"state-cycle entry must be STATE=COUNT: {item}")
        state, count_text = item.rsplit("=", 1)
        if not state or state in output:
            raise ValueError(f"invalid or duplicate state-cycle entry: {item}")
        try:
            count = int(count_text)
        except ValueError as error:
            raise ValueError(f"invalid state-cycle count: {item}") from error
        if count < 1:
            raise ValueError(f"state-cycle count must be positive: {item}")
        output[state] = count
    return output


def complete_pair_schema(schema: Sequence[Mapping[str, object]], n_vinyl: int) -> bool:
    observed = {
        tuple(sorted((int(pair["left"]), int(pair["right"])))) for pair in schema
    }
    expected = {
        (left, right)
        for left in range(1, n_vinyl + 1)
        for right in range(left + 1, n_vinyl + 1)
    }
    return observed == expected and len(schema) == len(expected)


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def cache_metadata(
    row: Mapping[str, str],
    xtc: Path,
    topology: Path,
    expected_frames: int,
    settings: Sequence[GateSetting],
) -> dict[str, object]:
    return {
        "cache_version": CACHE_VERSION,
        "state": row["state"],
        "cycle": int(row["cycle"]),
        "expected_frames": expected_frames,
        "settings": [asdict(item) for item in settings],
        "xtc": file_signature(xtc),
        "topology": file_signature(topology),
    }


def cycle_cache_is_current(csv_path: Path, json_path: Path, expected: Mapping[str, object]) -> bool:
    if not csv_path.is_file() or not json_path.is_file():
        return False
    try:
        observed = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return observed.get("input") == expected and observed.get("status") == "complete"


def extract_cycle(
    root: Path,
    row: Mapping[str, str],
    settings: Sequence[GateSetting],
    expected_frames: int,
    force: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    try:
        import MDAnalysis as mda
        from MDAnalysis.transformations import unwrap
    except ModuleNotFoundError as error:
        raise RuntimeError("MDAnalysis is required for geometry extraction") from error

    cycle_dir = root / row["output_dir"]
    setup = root / "gromacs_dcm" / "runs" / row["state"] / "setup"
    metadata_path = setup / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"state metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    xtc = cycle_dir / "short_25ps.xtc"
    topology = setup / "solute_parameters" / "slu.mol2"
    if not xtc.is_file() or not topology.is_file():
        raise FileNotFoundError(f"cycle XTC or solute topology is missing in {cycle_dir}")
    output = cycle_dir / "vinyl_geometry_gate_statistics.csv"
    sidecar = cycle_dir / "vinyl_geometry_gate_statistics.json"
    expected_cache = cache_metadata(row, xtc, topology, expected_frames, settings)
    if not force and cycle_cache_is_current(output, sidecar, expected_cache):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        return read_csv(output), {**payload["quality"], "status": "SKIPPED"}

    universe = mda.Universe(str(topology), str(xtc))
    source_atom_count = int(metadata["source_atom_count"])
    if len(universe.atoms) != source_atom_count:
        raise ValueError(
            f"solute topology/XTC atom mismatch: {len(universe.atoms)} != {source_atom_count}"
        )
    solute = universe.atoms
    universe.trajectory.add_transformations(unwrap(solute))
    mapping = {
        str(name): [int(value) for value in values]
        for name, values in metadata["vinyl_mapping_gromacs_1based"].items()
    }
    pair_schema = list(metadata["pair_schema"])
    n_vinyl = len(mapping)
    if set(mapping) != {f"V{index}" for index in range(1, n_vinyl + 1)}:
        raise ValueError("vinyl mapping is not a complete V1..Vn sequence")
    if not complete_pair_schema(pair_schema, n_vinyl):
        raise ValueError("pair schema is not a complete labelled K_n table")
    pair_names = [str(pair["column"]) for pair in pair_schema]
    topology_labels = [str(pair["topology"]) for pair in pair_schema]
    if set(topology_labels) != {"same_arm", "cross_arm"}:
        raise ValueError("pair topology must contain same_arm and cross_arm")

    setting_count = len(settings)
    pair_count = len(pair_schema)
    contact_mass = np.zeros((setting_count, pair_count), dtype=float)
    reactive_mass = np.zeros_like(contact_mass)
    contact_axis_sum = np.zeros_like(contact_mass)
    contact_side_sum = np.zeros_like(contact_mass)
    frames = 0
    minimum_vinyl_length = math.inf
    maximum_vinyl_length = -math.inf
    for timestep in universe.trajectory:
        distances, axis_angles, side_angles, lengths = pair_geometry(
            solute.positions, mapping, pair_schema
        )
        minimum_vinyl_length = min(minimum_vinyl_length, float(np.min(lengths)))
        maximum_vinyl_length = max(maximum_vinyl_length, float(np.max(lengths)))
        for setting_index, setting in enumerate(settings):
            contact = soft_upper(
                distances, setting.distance_center_A, setting.distance_width_A
            )
            orientation = soft_upper(
                axis_angles, setting.axis_max_deg, setting.angle_width_deg
            ) * soft_lower(side_angles, setting.side_min_deg, setting.angle_width_deg)
            contact_mass[setting_index] += contact
            reactive_mass[setting_index] += contact * orientation
            contact_axis_sum[setting_index] += contact * axis_angles
            contact_side_sum[setting_index] += contact * side_angles
        frames += 1
    if frames != expected_frames:
        raise ValueError(f"{cycle_dir} has {frames} frames, expected {expected_frames}")
    if not np.all(contact_mass > 0.0) or not np.all(reactive_mass > 0.0):
        raise ValueError("soft gate unexpectedly reached a zero sufficient statistic")

    output_rows: list[dict[str, object]] = []
    for setting_index, setting in enumerate(settings):
        for pair_index, (pair_name, pair_topology) in enumerate(
            zip(pair_names, topology_labels)
        ):
            denominator = contact_mass[setting_index, pair_index]
            numerator = reactive_mass[setting_index, pair_index]
            output_rows.append(
                {
                    "state": row["state"],
                    "system": row["system"],
                    "cycle": int(row["cycle"]),
                    "cluster": row["cluster"],
                    "setting_id": setting.setting_id,
                    "distance_center_A": setting.distance_center_A,
                    "distance_width_A": setting.distance_width_A,
                    "axis_max_deg": setting.axis_max_deg,
                    "angle_width_deg": setting.angle_width_deg,
                    "side_min_deg": setting.side_min_deg,
                    "pair": pair_name,
                    "topology": pair_topology,
                    "frames": frames,
                    "soft_contact_mass": denominator,
                    "soft_reactive_mass": numerator,
                    "mean_soft_contact": denominator / frames,
                    "conditional_orientation_gate": numerator / denominator,
                    "contact_weighted_axis_angle_deg": (
                        contact_axis_sum[setting_index, pair_index] / denominator
                    ),
                    "contact_weighted_side_angle_deg": (
                        contact_side_sum[setting_index, pair_index] / denominator
                    ),
                }
            )
    write_csv(output, output_rows)
    quality = {
        "state": row["state"],
        "system": row["system"],
        "cycle": int(row["cycle"]),
        "cluster": row["cluster"],
        "frames": frames,
        "minimum_vinyl_bond_A": minimum_vinyl_length,
        "maximum_vinyl_bond_A": maximum_vinyl_length,
        "vinyl_bond_range_passed": (
            minimum_vinyl_length >= 1.15 and maximum_vinyl_length <= 1.55
        ),
        "status": "OK",
    }
    sidecar_payload = {"input": expected_cache, "quality": quality, "status": "complete"}
    temporary = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(sidecar_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(sidecar)
    return output_rows, quality


def extract_cycle_worker(
    root: Path,
    row: Mapping[str, str],
    settings: Sequence[GateSetting],
    expected_frames: int,
    force: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Pickle-friendly process-pool entry point."""
    return extract_cycle(root, row, settings, expected_frames, force)


def statistic_arrays(
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], np.ndarray, np.ndarray, np.ndarray
]:
    settings = tuple(sorted({str(row["setting_id"]) for row in rows}))
    cycles = tuple(
        sorted(
            {str(row["cycle"]) for row in rows},
            key=lambda value: int(value),
        )
    )
    pairs = tuple(sorted({str(row["pair"]) for row in rows}))
    setting_index = {value: index for index, value in enumerate(settings)}
    cycle_index = {value: index for index, value in enumerate(cycles)}
    pair_index = {value: index for index, value in enumerate(pairs)}
    numerator = np.full((len(settings), len(cycles), len(pairs)), np.nan)
    denominator = np.full_like(numerator, np.nan)
    topology = np.full(len(pairs), "", dtype=object)
    for row in rows:
        indices = (
            setting_index[str(row["setting_id"])],
            cycle_index[str(row["cycle"])],
            pair_index[str(row["pair"])],
        )
        if np.isfinite(numerator[indices]):
            raise ValueError(f"duplicate sufficient-statistic row at {indices}")
        numerator[indices] = float(row["soft_reactive_mass"])
        denominator[indices] = float(row["soft_contact_mass"])
        topology[indices[2]] = str(row["topology"])
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise ValueError("incomplete setting x cycle x pair sufficient-statistic cube")
    if np.any(numerator <= 0.0) or np.any(denominator <= 0.0):
        raise ValueError("nonpositive geometry sufficient statistic")
    return settings, cycles, pairs, numerator, denominator, topology


def epsilon_from_cycle_indices(
    numerator: np.ndarray,
    denominator: np.ndarray,
    topology: np.ndarray,
    cluster_cycle_indices: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    same = topology == "same_arm"
    cross = topology == "cross_arm"
    if not np.any(same) or not np.any(cross):
        raise ValueError("same/cross topology coverage is incomplete")
    log_epsilon_by_cluster = []
    same_by_cluster = []
    cross_by_cluster = []
    for indices in cluster_cycle_indices:
        same_kappa = np.sum(numerator[:, indices][:, :, same], axis=(1, 2)) / np.sum(
            denominator[:, indices][:, :, same], axis=(1, 2)
        )
        cross_kappa = np.sum(numerator[:, indices][:, :, cross], axis=(1, 2)) / np.sum(
            denominator[:, indices][:, :, cross], axis=(1, 2)
        )
        log_epsilon_by_cluster.append(np.log(same_kappa) - np.log(cross_kappa))
        same_by_cluster.append(np.log(same_kappa))
        cross_by_cluster.append(np.log(cross_kappa))
    return (
        np.mean(np.stack(log_epsilon_by_cluster), axis=0),
        np.exp(np.mean(np.stack(same_by_cluster), axis=0)),
        np.exp(np.mean(np.stack(cross_by_cluster), axis=0)),
    )


def hierarchical_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    topology: np.ndarray,
    cluster_for_cycle: Sequence[str],
    replicates: int,
    seed: int,
) -> np.ndarray:
    if replicates < 1:
        return np.empty((0, numerator.shape[0]))
    cluster_names = tuple(sorted(set(cluster_for_cycle)))
    indices_by_cluster = {
        name: np.asarray(
            [index for index, observed in enumerate(cluster_for_cycle) if observed == name],
            dtype=int,
        )
        for name in cluster_names
    }
    rng = np.random.default_rng(seed)
    output = np.empty((replicates, numerator.shape[0]), dtype=float)
    for replicate in range(replicates):
        selected_clusters = rng.integers(0, len(cluster_names), size=len(cluster_names))
        sampled_cycle_groups = []
        for selected in selected_clusters:
            available = indices_by_cluster[cluster_names[int(selected)]]
            sampled_cycle_groups.append(
                available[rng.integers(0, len(available), size=len(available))]
            )
        log_epsilon, _, _ = epsilon_from_cycle_indices(
            numerator, denominator, topology, sampled_cycle_groups
        )
        output[replicate] = log_epsilon
    return output


def setting_map(rows: Sequence[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        identifier = str(row["setting_id"])
        output.setdefault(
            identifier,
            {
                "distance_center_A": float(row["distance_center_A"]),
                "distance_width_A": float(row["distance_width_A"]),
                "axis_max_deg": float(row["axis_max_deg"]),
                "angle_width_deg": float(row["angle_width_deg"]),
                "side_min_deg": float(row["side_min_deg"]),
            },
        )
    return output


def sec_interval(root: Path, fit_source: Path, system: str) -> tuple[float, float] | None:
    path = fit_source / "identifiability_summary.csv"
    if not path.is_file():
        return None
    for row in read_csv(path):
        if row["system"] == system:
            return float(row["epsilon0_min"]), float(row["epsilon0_max"])
    return None


def analyze_state(
    root: Path,
    output: Path,
    state: str,
    manifest_rows: Sequence[Mapping[str, str]],
    statistic_rows: Sequence[Mapping[str, object]],
    bootstrap_replicates: int,
    seed: int,
    temperature_K: float,
    primary_setting: str,
    fit_source: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    system = str(manifest_rows[0]["system"])
    settings, cycles, pairs, numerator, denominator, topology = statistic_arrays(statistic_rows)
    frames_by_cycle: dict[str, int] = {}
    for row in statistic_rows:
        cycle = str(row["cycle"])
        frames = int(row["frames"])
        if cycle in frames_by_cycle and frames_by_cycle[cycle] != frames:
            raise ValueError(f"inconsistent frame counts within cycle {cycle}")
        frames_by_cycle[cycle] = frames
    if set(frames_by_cycle) != set(cycles):
        raise ValueError("frame-count coverage does not match the cycle cube")
    cluster_by_cycle = {
        str(row["cycle"]): str(row["cluster"]) for row in manifest_rows
    }
    cluster_for_cycle = [cluster_by_cycle[cycle] for cycle in cycles]
    clusters = tuple(sorted(set(cluster_for_cycle)))
    original_groups = [
        np.asarray(
            [index for index, value in enumerate(cluster_for_cycle) if value == cluster],
            dtype=int,
        )
        for cluster in clusters
    ]
    point_log_epsilon, point_same, point_cross = epsilon_from_cycle_indices(
        numerator, denominator, topology, original_groups
    )
    bootstrap_log_epsilon = hierarchical_bootstrap(
        numerator,
        denominator,
        topology,
        cluster_for_cycle,
        bootstrap_replicates,
        stable_seed(seed, state, "hierarchical_geometry_gate"),
    )
    parameters = setting_map(statistic_rows)
    rt = GAS_CONSTANT_KCAL_MOL_K * temperature_K
    scan_rows: list[dict[str, object]] = []
    for index, identifier in enumerate(settings):
        draws = bootstrap_log_epsilon[:, index]
        epsilon_draws = np.exp(draws)
        energy_draws = -rt * draws
        point_epsilon = math.exp(float(point_log_epsilon[index]))
        scan_rows.append(
            {
                "state": state,
                "system": system,
                "setting_id": identifier,
                **parameters[identifier],
                "clusters": len(clusters),
                "cycles": len(cycles),
                "same_arm_conditional_gate": float(point_same[index]),
                "cross_arm_conditional_gate": float(point_cross[index]),
                "epsilon_geom_point": point_epsilon,
                "epsilon_geom_bootstrap_median": float(np.median(epsilon_draws)),
                "epsilon_geom_ci95_low": float(np.quantile(epsilon_draws, 0.025)),
                "epsilon_geom_ci95_high": float(np.quantile(epsilon_draws, 0.975)),
                "probability_epsilon_below_one": float(np.mean(epsilon_draws < 1.0)),
                "delta_delta_G_gate_kcal_mol_point": -rt * float(point_log_epsilon[index]),
                "delta_delta_G_gate_bootstrap_median": float(np.median(energy_draws)),
                "delta_delta_G_gate_ci95_low": float(np.quantile(energy_draws, 0.025)),
                "delta_delta_G_gate_ci95_high": float(np.quantile(energy_draws, 0.975)),
                "bootstrap_replicates": bootstrap_replicates,
            }
        )
    if primary_setting not in settings:
        raise ValueError(
            f"primary setting {primary_setting} absent; available settings: {settings}"
        )
    primary = next(row for row in scan_rows if row["setting_id"] == primary_setting)
    epsilon_points = np.asarray([float(row["epsilon_geom_point"]) for row in scan_rows])
    directions = epsilon_points < 1.0
    majority_below = float(np.mean(directions)) >= 0.5
    direction_agreement = float(np.mean(directions == majority_below))
    interval = sec_interval(root, fit_source, system)
    if interval is None:
        sec_low = sec_high = math.nan
        overlap = False
    else:
        sec_low, sec_high = interval
        overlap = not (
            float(primary["epsilon_geom_ci95_high"]) < sec_low
            or float(primary["epsilon_geom_ci95_low"]) > sec_high
        )
    summary = {
        **primary,
        "temperature_K": temperature_K,
        "RT_kcal_mol": rt,
        "gate_settings": len(settings),
        "fraction_settings_epsilon_below_one": float(np.mean(directions)),
        "setting_direction_agreement": direction_agreement,
        "sec_epsilon_near_optimal_min": sec_low,
        "sec_epsilon_near_optimal_max": sec_high,
        "geometry_ci_overlaps_sec_near_optimal_interval": overlap,
        "physical_claim": "catalyst_free_geometry_gate_not_activation_barrier",
    }

    primary_index = settings.index(primary_setting)
    pair_rows = []
    for pair_index, pair in enumerate(pairs):
        cluster_log_kappa = []
        cluster_contact = []
        for indices in original_groups:
            pair_numerator = float(np.sum(numerator[primary_index, indices, pair_index]))
            pair_denominator = float(np.sum(denominator[primary_index, indices, pair_index]))
            frames_in_cluster = sum(frames_by_cycle[cycles[index]] for index in indices)
            cluster_log_kappa.append(math.log(pair_numerator / pair_denominator))
            cluster_contact.append(pair_denominator / frames_in_cluster)
        pair_rows.append(
            {
                "state": state,
                "system": system,
                "pair": pair,
                "topology": str(topology[pair_index]),
                "setting_id": primary_setting,
                "conditional_orientation_gate_geometric_cluster_mean": math.exp(
                    float(np.mean(cluster_log_kappa))
                ),
                "conditional_orientation_gate_min_cluster": math.exp(
                    float(np.min(cluster_log_kappa))
                ),
                "conditional_orientation_gate_max_cluster": math.exp(
                    float(np.max(cluster_log_kappa))
                ),
                "mean_soft_contact_equal_cluster": float(np.mean(cluster_contact)),
            }
        )
    write_csv(output / f"{state}_pair_gate_primary.csv", pair_rows)
    return scan_rows, summary


def make_scan_svg(path: Path, scan_rows: Sequence[Mapping[str, object]]) -> None:
    systems = sorted({str(row["system"]) for row in scan_rows})
    width = 920
    panel_height = 250
    height = 70 + panel_height * len(systems)
    colors = ("#2166ac", "#f7f7f7", "#b2182b")

    def color(value: float) -> str:
        clipped = max(-2.0, min(2.0, value)) / 2.0
        if clipped < 0:
            fraction = clipped + 1.0
            left, right = colors[0], colors[1]
        else:
            fraction = clipped
            left, right = colors[1], colors[2]
        rgb_left = tuple(int(left[index:index + 2], 16) for index in (1, 3, 5))
        rgb_right = tuple(int(right[index:index + 2], 16) for index in (1, 3, 5))
        rgb = tuple(round(a + fraction * (b - a)) for a, b in zip(rgb_left, rgb_right))
        return "#" + "".join(f"{component:02x}" for component in rgb)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="30" y="35" font-family="Arial,sans-serif" font-size="20" font-weight="700">Catalyst-free same/cross geometry gate</text>',
        '<text x="30" y="56" font-family="Arial,sans-serif" font-size="12" fill="#455a64">Cell value: log(epsilon_geom); blue = same-arm suppressed, red = same-arm favored</text>',
    ]
    for panel, system in enumerate(systems):
        rows = [row for row in scan_rows if row["system"] == system]
        distances = sorted({float(row["distance_center_A"]) for row in rows})
        axes = sorted({float(row["axis_max_deg"]) for row in rows})
        sides = sorted({float(row["side_min_deg"]) for row in rows})
        y0 = 80 + panel * panel_height
        svg.append(
            f'<text x="30" y="{y0}" font-family="Arial,sans-serif" font-size="16" font-weight="700">{system}</text>'
        )
        cell_w, cell_h = 70, 42
        for distance_index, distance in enumerate(distances):
            x0 = 150 + distance_index * 245
            svg.append(
                f'<text x="{x0 + 1.5 * cell_w}" y="{y0}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">d0={distance:g} Å</text>'
            )
            for axis_index, axis in enumerate(axes):
                svg.append(
                    f'<text x="{x0 + axis_index * cell_w + cell_w/2}" y="{y0+22}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10">axis≤{axis:g}°</text>'
                )
                for side_index, side in enumerate(sides):
                    row = next(
                        item
                        for item in rows
                        if float(item["distance_center_A"]) == distance
                        and float(item["axis_max_deg"]) == axis
                        and float(item["side_min_deg"]) == side
                    )
                    value = math.log(float(row["epsilon_geom_point"]))
                    x = x0 + axis_index * cell_w
                    y = y0 + 30 + side_index * cell_h
                    svg.extend(
                        [
                            f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" rx="3" fill="{color(value)}"/>',
                            f'<text x="{x+(cell_w-3)/2}" y="{y+18}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11">{value:+.2f}</text>',
                            f'<text x="{x+(cell_w-3)/2}" y="{y+32}" text-anchor="middle" font-family="Arial,sans-serif" font-size="9">ε={math.exp(value):.2f}</text>',
                        ]
                    )
            for side_index, side in enumerate(sides):
                svg.append(
                    f'<text x="{x0-8}" y="{y0+55+side_index*cell_h}" text-anchor="end" font-family="Arial,sans-serif" font-size="10">side≥{side:g}°</text>'
                )
    svg.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument(
        "--state-cycles",
        nargs="*",
        default=(),
        metavar="STATE=COUNT",
        help="per-state cycle-count overrides; useful when 100- and 150-cycle states are combined",
    )
    parser.add_argument("--expected-frames", type=int, default=501)
    parser.add_argument("--distance-centers-A", nargs="+", type=float, default=list(DEFAULT_DISTANCE_CENTERS_A))
    parser.add_argument("--distance-width-A", type=float, default=0.5)
    parser.add_argument("--axis-maxima-deg", nargs="+", type=float, default=list(DEFAULT_AXIS_MAXIMA_DEG))
    parser.add_argument("--side-minima-deg", nargs="+", type=float, default=list(DEFAULT_SIDE_MINIMA_DEG))
    parser.add_argument("--angle-width-deg", type=float, default=7.5)
    parser.add_argument("--primary-setting", default="d5_axis45_side45")
    parser.add_argument("--temperature-K", type=float, default=313.15)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--fit-source",
        default="results/explicit_dcm_per_system_three_parameters_gamma_signed",
    )
    parser.add_argument(
        "--output", default="results/explicit_dcm_q0_geometry_gate"
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if (
        args.cycles < 1
        or args.expected_frames < 2
        or args.bootstrap < 1
        or args.jobs < 1
        or args.temperature_K <= 0.0
    ):
        raise SystemExit("invalid cycles, frame count, bootstrap count, or temperature")
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    fit_source = Path(args.fit_source)
    if not fit_source.is_absolute():
        fit_source = root / fit_source
    settings = build_settings(
        args.distance_centers_A,
        args.distance_width_A,
        args.axis_maxima_deg,
        args.angle_width_deg,
        args.side_minima_deg,
    )
    state_cycles = parse_state_cycles(args.state_cycles)
    manifest_path = root / "gromacs_dcm" / "rare_event_cycles" / "cycle_manifest.csv"
    selected = select_manifest_rows(
        read_csv(manifest_path), args.states, args.cycles, state_cycles
    )
    print(
        f"Geometry gate: states={len(args.states)}; cycles={len(selected)}; "
        f"settings={len(settings)}; bootstrap={args.bootstrap}; jobs={args.jobs}",
        flush=True,
    )
    if args.dry_run:
        for row in selected:
            print(root / row["output_dir"] / "short_25ps.xtc")
        return

    statistics_by_state: dict[str, list[dict[str, object]]] = defaultdict(list)
    quality_rows: list[dict[str, object]] = []
    if args.jobs == 1:
        completed_cycles = []
        for completed, row in enumerate(selected, start=1):
            result = extract_cycle(root, row, settings, args.expected_frames, args.force)
            completed_cycles.append((row, result))
            print(
                f"[{completed}/{len(selected)}] {row['state']} "
                f"cycle {int(row['cycle']):03d} [{result[1]['status']}]",
                flush=True,
            )
    else:
        completed_cycles = []
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = {
                executor.submit(
                    extract_cycle_worker,
                    root,
                    row,
                    settings,
                    args.expected_frames,
                    args.force,
                ): row
                for row in selected
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                row = futures[future]
                result = future.result()
                completed_cycles.append((row, result))
                print(
                    f"[{completed}/{len(selected)}] {row['state']} "
                    f"cycle {int(row['cycle']):03d} [{result[1]['status']}]",
                    flush=True,
                )
    completed_cycles.sort(key=lambda item: (item[0]["state"], int(item[0]["cycle"])))
    for index, (row, (statistics, quality)) in enumerate(completed_cycles, start=1):
        statistics_by_state[row["state"]].extend(statistics)
        quality_rows.append(quality)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "cycle_data_quality.csv", quality_rows)
    all_scan_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for state in args.states:
        state_manifest = [row for row in selected if row["state"] == state]
        scan_rows, summary = analyze_state(
            root,
            output,
            state,
            state_manifest,
            statistics_by_state[state],
            args.bootstrap,
            args.seed,
            args.temperature_K,
            args.primary_setting,
            fit_source,
        )
        all_scan_rows.extend(scan_rows)
        summaries.append(summary)
    write_csv(output / "geometry_gate_scan.csv", all_scan_rows)
    write_csv(output / "primary_geometry_gate_summary.csv", summaries)
    make_scan_svg(output / "figures" / "geometry_gate_scan.svg", all_scan_rows)

    checks = [
        {
            "check": "all_selected_cycles_processed",
            "passed": len(quality_rows) == len(selected),
            "detail": f"{len(quality_rows)}/{len(selected)}",
        },
        {
            "check": "all_cycles_expected_frame_count",
            "passed": all(int(row["frames"]) == args.expected_frames for row in quality_rows),
            "detail": f"expected={args.expected_frames}",
        },
        {
            "check": "mapped_vinyl_bond_lengths_physical",
            "passed": all(bool(row["vinyl_bond_range_passed"]) for row in quality_rows),
            "detail": "required 1.15 <= mapped C=C <= 1.55 A in every frame",
        },
        {
            "check": "gate_probabilities_valid",
            "passed": all(
                0.0 < float(row["same_arm_conditional_gate"]) < 1.0
                and 0.0 < float(row["cross_arm_conditional_gate"]) < 1.0
                for row in all_scan_rows
            ),
            "detail": f"settings={len(all_scan_rows)}",
        },
        {
            "check": "threshold_grid_has_primary_setting",
            "passed": args.primary_setting in {item.setting_id for item in settings},
            "detail": args.primary_setting,
        },
    ]
    write_csv(output / "data_quality_checks.csv", checks)
    if not all(bool(row["passed"]) for row in checks):
        raise RuntimeError(f"geometry-gate data quality failed; see {output / 'data_quality_checks.csv'}")

    report = [
        "# q0 同臂/跨臂无催化剂几何门控",
        "",
        "本分析在已有 q0 软距离接触权重之上加入烯烃轴平行、侧向接近的平滑门控。"
        "分子为 `contact × orientation`，分母为完全相同的 `contact`，所以输出是已经接近条件下的取向合格率。",
        "",
        "该量是 `epsilon0` 的无催化剂几何代理，不是复分解活化能垒。",
        "",
        "![Geometry gate scan](figures/geometry_gate_scan.svg)",
        "",
        "| 体系 | epsilon_geom | 95% CI | P(epsilon<1) | ΔΔG_gate (kcal/mol) | 设置方向一致率 | 与 SEC 近最优区间重叠 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        report.append(
            f"| {row['system']} | {float(row['epsilon_geom_point']):.3g} | "
            f"[{float(row['epsilon_geom_ci95_low']):.3g}, {float(row['epsilon_geom_ci95_high']):.3g}] | "
            f"{100*float(row['probability_epsilon_below_one']):.1f}% | "
            f"{float(row['delta_delta_G_gate_kcal_mol_point']):+.3f} | "
            f"{100*float(row['setting_direction_agreement']):.1f}% | "
            f"{row['geometry_ci_overlaps_sec_near_optimal_interval']} |"
        )
    report.extend(
        [
            "",
            "## 解释边界",
            "",
            "- q0 距离贡献通过条件比值消除；同臂/跨臂原始接触频率不是本指标。",
            "- 主结果采用构象簇等权；bootstrap 先重采样构象簇，再在簇内重采样速度循环。",
            "- 两原子烯烃轴不能描述催化剂进攻面、Ru 配位或完整四面体几何。",
            "- 阈值网格用于判断方向稳健性，不能看到 SEC 后挑选误差最小阈值。",
        ]
    )
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary_payload = {
        "analysis": "catalyst-free conditional q0 vinyl geometry gate",
        "formula": "epsilon_geom=GM_cluster[(sum_same h*g/sum_same h)/(sum_cross h*g/sum_cross h)]",
        "orientation_proxy": "parallel vinyl axes plus side-on midpoint approach",
        "temperature_K": args.temperature_K,
        "settings": [asdict(item) for item in settings],
        "primary_setting": args.primary_setting,
        "bootstrap_replicates": args.bootstrap,
        "bootstrap_seed": args.seed,
        "cycles_by_state": {
            state: sum(row["state"] == state for row in selected) for state in args.states
        },
        "states": summaries,
        "uses_SEC_for_fitting": False,
        "SEC_use": "post_hoc interval-overlap annotation only when fit source exists",
        "physical_claim_boundary": "geometry gate proxy, not catalyst-aware activation barrier",
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for row in summaries:
        print(
            f"{row['system']}: epsilon_geom={float(row['epsilon_geom_point']):.4g} "
            f"[{float(row['epsilon_geom_ci95_low']):.4g}, "
            f"{float(row['epsilon_geom_ci95_high']):.4g}]; "
            f"delta_delta_G_gate={float(row['delta_delta_G_gate_kcal_mol_point']):+.3f} kcal/mol; "
            f"setting agreement={100*float(row['setting_direction_agreement']):.1f}%"
        )
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
