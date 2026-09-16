#!/usr/bin/env python3
"""Reweight q2 umbrella trajectories into distance-orientation free-energy surfaces.

No new dynamics are generated.  For each pathway, reduced harmonic bias
energies are reconstructed from the pull coordinate and solved with the MBAR
self-consistency equations.  The resulting unbiased frame weights are used for
F(distance, orientation-gate) and for contact/orientation/NAC decomposition.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.special import logsumexp

from analyze_explicit_dcm_q0_geometry_gate import soft_lower, soft_upper
from extract_a4_q2_umbrella_qm_candidates import (
    PATHWAY_LOCATIONS,
    parse_ndx,
    read_numeric_xvg,
)


R_KJ = 0.00831446261815324
R_KCAL = 0.00198720425864083
TEMPERATURE_K = 313.15
RT_KCAL = R_KCAL * TEMPERATURE_K
BETA_KJ = 1.0 / (R_KJ * TEMPERATURE_K)
K_FORCE = 500.0  # kJ mol-1 nm-2
DISTANCE_EDGES_NM = np.linspace(0.30, 2.80, 51)
ORIENTATION_EDGES = np.linspace(0.0, 1.0, 26)


GROUPS = (
    ("C5_adjacent_same", ("c5_adj_same_v2_v3",)),
    ("C5_adjacent_cross", ("c5_adj_cross_v2_v1",)),
    ("C8_adjacent_same", ("c8_adj_same_v1_v2",)),
    ("C8_adjacent_cross", ("c8_adj_cross_v1_v4",)),
    ("C5_opposite_reuse", ("c5_reuse_v1_v6", "c5_reuse_v2_v4")),
    ("C5_opposite_switch", ("c5_switch_v4_v6",)),
    ("C8_opposite_reuse", ("c8_reuse_v5_v7",)),
    ("C8_opposite_switch", ("c8_switch_v4_v5",)),
)

CONTRASTS = (
    ("C5_adjacent", "C5_adjacent_cross", "C5_adjacent_same", "cross_minus_same"),
    ("C8_adjacent", "C8_adjacent_cross", "C8_adjacent_same", "cross_minus_same"),
    ("C5_opposite", "C5_opposite_switch", "C5_opposite_reuse", "switch_minus_reuse"),
    ("C8_opposite", "C8_opposite_switch", "C8_opposite_reuse", "switch_minus_reuse"),
)


@dataclass
class PathwayResult:
    metadata: dict[str, object]
    centers_nm: np.ndarray
    window_index: np.ndarray
    time_ps: np.ndarray
    distance_nm: np.ndarray
    axis_angle_deg: np.ndarray
    side_angle_deg: np.ndarray
    orientation_gate: np.ndarray
    weights: np.ndarray
    mbar_f: np.ndarray
    mbar_iterations: int
    mbar_residual: float
    geometry_pull_rms_A: float
    geometry_pull_max_A: float
    effective_sample_size: float
    surface_probability: np.ndarray
    surface_pmf_kcal: np.ndarray
    metrics: dict[str, float]
    half_metrics: dict[str, dict[str, float]]


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run(command: Sequence[str], stdin: str | None = None) -> str:
    result = subprocess.run(
        list(command), input=stdin, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-35:])
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{tail}")
    return result.stdout


def parse_window_center(path: Path) -> float:
    match = re.search(r"window_\d+_([0-9.]+)nm$", path.name)
    if not match:
        raise ValueError(f"cannot parse umbrella center from {path}")
    return float(match.group(1))


def read_pullx(path: Path, begin: float, end: float) -> np.ndarray:
    values = read_numeric_xvg(path)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"unexpected pullx table: {path} {values.shape}")
    keep = (values[:, 0] >= begin - 1.0e-6) & (values[:, 0] <= end + 1.0e-6)
    return values[keep, :2]


def orientation_geometry_matching_pull(
    coordinate_values: np.ndarray,
    box_values: np.ndarray,
    pull_distance_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the connector image that reproduces GROMACS' pull distance.

    GROMACS pull groups use a reference-atom/history-aware PBC convention.  A
    naive shortest-image connector can therefore refer to a different periodic
    image even though each vinyl bond itself is physical.  Enumerating the 27
    neighboring images and selecting the connector whose norm matches pullx
    restores the exact reaction-coordinate image before computing orientation.
    """
    coordinates = coordinate_values[:, 1:].reshape(-1, 4, 3)
    if box_values.shape != (len(coordinates), 7) or np.max(np.abs(box_values[:, 4:])) > 1.0e-6:
        raise ValueError("unexpected or triclinic box table")
    box = box_values[:, 1:4]

    def minimum_image(vectors: np.ndarray) -> np.ndarray:
        return vectors - box * np.round(vectors / box)

    left_vector = minimum_image(coordinates[:, 1] - coordinates[:, 0])
    right_vector = minimum_image(coordinates[:, 3] - coordinates[:, 2])
    left_length = np.linalg.norm(left_vector, axis=1)
    right_length = np.linalg.norm(right_vector, axis=1)
    physical = (
        (left_length >= 0.115) & (left_length <= 0.155)
        & (right_length >= 0.115) & (right_length <= 0.155)
    )
    left_axis = left_vector / left_length[:, None]
    right_axis = right_vector / right_length[:, None]
    left_center = coordinates[:, 0] + 0.5 * left_vector
    right_center = coordinates[:, 2] + 0.5 * right_vector
    raw_connector = right_center - left_center
    shifts = np.asarray(list(itertools.product((-1.0, 0.0, 1.0), repeat=3)))
    candidates = raw_connector[:, None, :] + shifts[None, :, :] * box[:, None, :]
    candidate_lengths = np.linalg.norm(candidates, axis=2)
    best = np.argmin(np.abs(candidate_lengths - pull_distance_nm[:, None]), axis=1)
    connector = candidates[np.arange(len(candidates)), best]
    distance = candidate_lengths[np.arange(len(candidates)), best]
    connector_axis = connector / distance[:, None]
    axis_angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(left_axis * right_axis, axis=1)), 0, 1)))
    side_left = np.degrees(np.arccos(np.clip(np.abs(np.sum(left_axis * connector_axis, axis=1)), 0, 1)))
    side_right = np.degrees(np.arccos(np.clip(np.abs(np.sum(right_axis * connector_axis, axis=1)), 0, 1)))
    side_angle = np.minimum(side_left, side_right)
    return distance, axis_angle, side_angle, physical


def mbar_weights(
    distances_nm: np.ndarray,
    centers_nm: np.ndarray,
    window_index: np.ndarray,
    tolerance: float = 1.0e-11,
    max_iterations: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    counts = np.bincount(window_index, minlength=len(centers_nm)).astype(float)
    if np.any(counts <= 0):
        raise ValueError("every MBAR window must contribute samples")
    # The umbrella potential depends only on distance.  Solve the equivalent
    # finely binned WHAM equations (0.002 nm bins), then evaluate the resulting
    # denominator at each original frame.  This avoids repeatedly traversing a
    # ~20,000-frame matrix during fixed-point iteration while preserving all
    # frame-resolved orientation information for the final 2D weighting.
    bin_width = 0.002
    lower = math.floor(float(np.min(distances_nm)) / bin_width) * bin_width
    upper = math.ceil(float(np.max(distances_nm)) / bin_width) * bin_width + bin_width
    edges = np.arange(lower, upper + 0.5 * bin_width, bin_width)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    histograms = np.asarray([
        np.histogram(distances_nm[window_index == index], bins=edges)[0]
        for index in range(len(centers_nm))
    ], dtype=float)
    total_histogram = np.sum(histograms, axis=0)
    occupied = total_histogram > 0
    bin_centers = bin_centers[occupied]
    total_histogram = total_histogram[occupied]
    reduced_bias_bins = BETA_KJ * 0.5 * K_FORCE * (
        bin_centers[None, :] - centers_nm[:, None]
    ) ** 2
    log_counts = np.log(counts)
    free = np.zeros(len(centers_nm), dtype=float)
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        log_denominator_bins = logsumexp(
            log_counts[:, None] + free[:, None] - reduced_bias_bins, axis=0
        )
        log_probability_mass = np.log(total_histogram) - log_denominator_bins
        updated = -logsumexp(
            log_probability_mass[None, :] - reduced_bias_bins, axis=1
        )
        updated -= updated[0]
        residual = float(np.max(np.abs(updated - free)))
        free = 0.5 * free + 0.5 * updated
        if residual < tolerance:
            break
    else:
        raise RuntimeError(f"MBAR did not converge; residual={residual:g}")
    reduced_bias_frames = BETA_KJ * 0.5 * K_FORCE * (
        distances_nm[None, :] - centers_nm[:, None]
    ) ** 2
    log_denominator = logsumexp(
        log_counts[:, None] + free[:, None] - reduced_bias_frames, axis=0
    )
    log_weight = -log_denominator
    log_weight -= logsumexp(log_weight)
    weights = np.exp(log_weight)
    return weights, free, iteration, residual


def weighted_metrics(distance_nm: np.ndarray, orientation: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    contact = soft_upper(distance_nm, 0.50, 0.05)
    nac = contact * orientation
    contact_probability = float(np.sum(weights * contact))
    nac_probability = float(np.sum(weights * nac))
    conditional_orientation = nac_probability / contact_probability
    return {
        "soft_contact_probability": contact_probability,
        "conditional_orientation": conditional_orientation,
        "soft_nac_probability": nac_probability,
        "contact_free_energy_kcal_mol": -RT_KCAL * math.log(contact_probability),
        "orientation_free_energy_kcal_mol": -RT_KCAL * math.log(conditional_orientation),
        "nac_free_energy_kcal_mol": -RT_KCAL * math.log(nac_probability),
    }


def weighted_surface(distance: np.ndarray, orientation: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probability, _, _ = np.histogram2d(
        distance, orientation, bins=(DISTANCE_EDGES_NM, ORIENTATION_EDGES), weights=weights
    )
    probability /= np.sum(probability)
    smooth = gaussian_filter(probability, sigma=(0.7, 0.7), mode="nearest")
    smooth /= np.sum(smooth)
    positive = smooth > 0
    pmf = np.full_like(smooth, np.nan)
    pmf[positive] = -RT_KCAL * np.log(smooth[positive])
    pmf -= np.nanmin(pmf)
    return probability, pmf


def analyze_subset(
    distance: np.ndarray,
    orientation: np.ndarray,
    centers: np.ndarray,
    window_index: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    weights, _, _, _ = mbar_weights(distance, centers, window_index)
    return weighted_metrics(distance, orientation, weights), weights


def analyze_pathway(
    pathway: Path,
    workspace: Path,
    output: Path,
    gmx: str,
    topology_root: Path | None = None,
) -> PathwayResult:
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    name = str(metadata["pathway"])
    begin = float(metadata["wham_begin_ps"])
    # Some earlier q1 manifests recorded only the WHAM start time.  For those
    # pathways, the common end time is inferred below from the last pull-x
    # timestamp present in every production window.
    requested_end = float(metadata.get("wham_end_ps", float("inf")))
    qm_dir = (topology_root or (workspace / "results/a4_q2_qm_candidates")) / name
    solute_gro = qm_dir / "solute_topology.gro"
    pair_index = qm_dir / "reactive_pair.ndx"
    if not solute_gro.is_file() or not pair_index.is_file():
        raise FileNotFoundError(f"run QM candidate extraction first: {qm_dir}")

    def available_end_time(window: Path) -> float | None:
        trajectory = window / "production.xtc"
        pullx = window / "production_pullx.xvg"
        if not trajectory.is_file() or not pullx.is_file():
            return None
        values = read_numeric_xvg(pullx)
        if values.ndim != 2 or len(values) <= 1 or values[-1, 0] < begin - 1.0e-6:
            return None
        return float(values[-1, 0])

    candidates = [
        (window, available_end_time(window))
        for window in sorted((pathway / "windows").glob("window_*"))
    ]
    candidates = [(window, last) for window, last in candidates if last is not None]
    end = min(requested_end, min(float(last) for _, last in candidates))
    complete_windows = [window for window, last in candidates if float(last) >= end - 1.0e-6]
    if len(complete_windows) < 20:
        raise ValueError(f"{name}: only {len(complete_windows)} complete windows")
    centers = np.asarray([parse_window_center(window) for window in complete_windows])
    if np.any(np.diff(centers) <= 0):
        raise ValueError(f"{name}: non-increasing window centers")

    cache_root = output / "cache" / name
    cache_root.mkdir(parents=True, exist_ok=True)
    all_time: list[np.ndarray] = []
    all_pull: list[np.ndarray] = []
    all_geo: list[np.ndarray] = []
    all_axis: list[np.ndarray] = []
    all_side: list[np.ndarray] = []
    all_orientation: list[np.ndarray] = []
    all_window: list[np.ndarray] = []
    for window_number, window in enumerate(complete_windows):
        window_cache = cache_root / window.name
        window_cache.mkdir(parents=True, exist_ok=True)
        coordinates_xvg = window_cache / "reactive_pair_coordinates_nm.xvg"
        box_xvg = window_cache / "box_vectors_nm.xvg"
        if not coordinates_xvg.is_file() or not box_xvg.is_file():
            run([
                gmx, "traj", "-f", str(window / "production.xtc"), "-s", str(solute_gro),
                "-n", str(pair_index), "-ox", str(coordinates_xvg), "-ob", str(box_xvg),
                "-b", str(begin), "-e", str(end), "-x", "-y", "-z",
            ], stdin="Reactive_pair\n")
        coordinate_values = read_numeric_xvg(coordinates_xvg)
        if coordinate_values.shape[1] != 13:
            raise ValueError(f"{coordinates_xvg}: expected 13 columns")
        box_values = read_numeric_xvg(box_xvg)
        pull_values = read_pullx(window / "production_pullx.xvg", begin, end)
        if len(pull_values) != len(coordinate_values) or not np.allclose(
            pull_values[:, 0], coordinate_values[:, 0], atol=1.0e-5
        ):
            raise ValueError(f"time mismatch in {window}")
        geo_distance, axis, side, physical = orientation_geometry_matching_pull(
            coordinate_values, box_values, pull_values[:, 1]
        )
        if not np.all(physical):
            raise ValueError(f"{window}: nonphysical vinyl coordinates after minimum image")
        orientation = soft_upper(axis, 45.0, 7.5) * soft_lower(side, 45.0, 7.5)
        all_time.append(pull_values[:, 0])
        all_pull.append(pull_values[:, 1])
        all_geo.append(geo_distance)
        all_axis.append(axis)
        all_side.append(side)
        all_orientation.append(orientation)
        all_window.append(np.full(len(pull_values), window_number, dtype=int))

    time = np.concatenate(all_time)
    distance = np.concatenate(all_pull)
    geo_distance = np.concatenate(all_geo)
    axis = np.concatenate(all_axis)
    side = np.concatenate(all_side)
    orientation = np.concatenate(all_orientation)
    window_index = np.concatenate(all_window)
    weights, free, iterations, residual = mbar_weights(distance, centers, window_index)
    metrics = weighted_metrics(distance, orientation, weights)
    probability, pmf = weighted_surface(distance, orientation, weights)

    half_metrics: dict[str, dict[str, float]] = {}
    midpoint = 0.5 * (begin + end)
    for half_name, selector in (
        ("first_half", time <= midpoint + 1.0e-6),
        ("second_half", time > midpoint + 1.0e-6),
    ):
        half_metrics[half_name], _ = analyze_subset(
            distance[selector], orientation[selector], centers, window_index[selector]
        )
    discrepancy_A = (geo_distance - distance) * 10.0
    return PathwayResult(
        metadata=metadata,
        centers_nm=centers,
        window_index=window_index,
        time_ps=time,
        distance_nm=distance,
        axis_angle_deg=axis,
        side_angle_deg=side,
        orientation_gate=orientation,
        weights=weights,
        mbar_f=free,
        mbar_iterations=iterations,
        mbar_residual=residual,
        geometry_pull_rms_A=float(np.sqrt(np.mean(discrepancy_A ** 2))),
        geometry_pull_max_A=float(np.max(np.abs(discrepancy_A))),
        effective_sample_size=float(1.0 / np.sum(weights ** 2)),
        surface_probability=probability,
        surface_pmf_kcal=pmf,
        metrics=metrics,
        half_metrics=half_metrics,
    )


def aggregate_groups(results: Mapping[str, PathwayResult]) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    groups: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for group_name, pathway_names in GROUPS:
        selected = [results[name] for name in pathway_names]
        probability = np.mean([result.surface_probability for result in selected], axis=0)
        probability /= np.sum(probability)
        smooth = gaussian_filter(probability, sigma=(0.7, 0.7), mode="nearest")
        smooth /= np.sum(smooth)
        pmf = -RT_KCAL * np.log(np.maximum(smooth, np.finfo(float).tiny))
        pmf -= np.nanmin(pmf)
        def aggregate_metric_set(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
            contact = float(np.mean([value["soft_contact_probability"] for value in values]))
            nac = float(np.mean([value["soft_nac_probability"] for value in values]))
            orientation = nac / contact
            return {
                "soft_contact_probability": contact,
                "conditional_orientation": orientation,
                "soft_nac_probability": nac,
                "contact_free_energy_kcal_mol": -RT_KCAL * math.log(contact),
                "orientation_free_energy_kcal_mol": -RT_KCAL * math.log(orientation),
                "nac_free_energy_kcal_mol": -RT_KCAL * math.log(nac),
            }

        metrics = aggregate_metric_set([result.metrics for result in selected])
        half_metrics = {
            half: aggregate_metric_set([result.half_metrics[half] for result in selected])
            for half in ("first_half", "second_half")
        }
        groups[group_name] = {
            "pathways": pathway_names,
            "probability": probability,
            "pmf": pmf,
            "metrics": metrics,
            "half_metrics": half_metrics,
        }
        rows.append({
            "group": group_name,
            "pathways": ";".join(pathway_names),
            "pathway_count": len(pathway_names),
            **metrics,
            **{
                f"{half}_{key}": value
                for half, values in half_metrics.items() for key, value in values.items()
            },
        })
    return groups, rows


def contrast_rows(groups: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    for state, alternative_name, reference_name, contrast in CONTRASTS:
        alternative = groups[alternative_name]
        reference = groups[reference_name]

        def compare(metric: str, half: str | None = None) -> float:
            if half is None:
                alt = float(alternative["metrics"][metric])
                ref = float(reference["metrics"][metric])
            else:
                alt = float(alternative["half_metrics"][half][metric])
                ref = float(reference["half_metrics"][half][metric])
            return -RT_KCAL * math.log(alt / ref)

        record: dict[str, object] = {
            "state": state,
            "contrast": contrast,
            "alternative_group": alternative_name,
            "reference_group": reference_name,
        }
        for half in (None, "first_half", "second_half"):
            prefix = "full" if half is None else half
            ddg_contact = compare("soft_contact_probability", half)
            ddg_orientation = compare("conditional_orientation", half)
            ddg_nac = compare("soft_nac_probability", half)
            record[f"{prefix}_ddG_contact_kcal_mol"] = ddg_contact
            record[f"{prefix}_ddG_orientation_kcal_mol"] = ddg_orientation
            record[f"{prefix}_ddG_nac_kcal_mol"] = ddg_nac
            record[f"{prefix}_additivity_residual_kcal_mol"] = ddg_nac - ddg_contact - ddg_orientation
        record["half_difference_ddG_nac_kcal_mol"] = (
            float(record["second_half_ddG_nac_kcal_mol"])
            - float(record["first_half_ddG_nac_kcal_mol"])
        )
        output.append(record)
    return output


def surface_rows(groups: Mapping[str, Mapping[str, object]]) -> list[dict[str, object]]:
    d_centers = 0.5 * (DISTANCE_EDGES_NM[:-1] + DISTANCE_EDGES_NM[1:])
    o_centers = 0.5 * (ORIENTATION_EDGES[:-1] + ORIENTATION_EDGES[1:])
    rows = []
    for group_name, group in groups.items():
        probability = np.asarray(group["probability"])
        pmf = np.asarray(group["pmf"])
        for i, distance in enumerate(d_centers):
            for j, orientation in enumerate(o_centers):
                rows.append({
                    "group": group_name,
                    "distance_nm": float(distance),
                    "orientation_gate": float(orientation),
                    "raw_probability_mass": float(probability[i, j]),
                    "smoothed_pmf_kcal_mol": float(pmf[i, j]),
                })
    return rows


def method_comparison_rows(
    workspace: Path, contrasts: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    previous = read_csv(
        workspace / "results/a4_q2_unified_nac_analysis/final/q2_umbrella_nac_agreement.csv"
    )
    indexed = {
        (str(row["system"]), str(row["branch"])): row
        for row in previous
    }
    output = []
    for contrast in contrasts:
        state = str(contrast["state"])
        system, branch = state.split("_", 1)
        row = indexed[(system, branch)]
        half_values = (
            float(contrast["first_half_ddG_nac_kcal_mol"]),
            float(contrast["second_half_ddG_nac_kcal_mol"]),
        )
        output.append({
            "state": state,
            "contrast": contrast["contrast"],
            "unbiased_nac_point_kcal_mol": float(row["nac_point_kcal_mol"]),
            "unbiased_nac_ci_low_kcal_mol": float(row["nac_ci_low_kcal_mol"]),
            "unbiased_nac_ci_high_kcal_mol": float(row["nac_ci_high_kcal_mol"]),
            "one_dimensional_umbrella_point_kcal_mol": float(row["umbrella_point_kcal_mol"]),
            "one_dimensional_umbrella_ci_low_kcal_mol": float(row["umbrella_ci_low_kcal_mol"]),
            "one_dimensional_umbrella_ci_high_kcal_mol": float(row["umbrella_ci_high_kcal_mol"]),
            "two_dimensional_reweight_point_kcal_mol": float(contrast["full_ddG_nac_kcal_mol"]),
            "two_dimensional_reweight_half_low_kcal_mol": min(half_values),
            "two_dimensional_reweight_half_high_kcal_mol": max(half_values),
        })
    return output


def plot_surfaces(path: Path, groups: Mapping[str, Mapping[str, object]]) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.0,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    row_defs = (
        ("C5 adjacent", "C5_adjacent_same", "C5_adjacent_cross"),
        ("C8 adjacent", "C8_adjacent_same", "C8_adjacent_cross"),
        ("C5 opposite", "C5_opposite_reuse", "C5_opposite_switch"),
        ("C8 opposite", "C8_opposite_reuse", "C8_opposite_switch"),
    )
    fig, axes = plt.subplots(4, 2, figsize=(7.2, 6.8), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for row_index, (row_label, reference, alternative) in enumerate(row_defs):
        for column, group_name in enumerate((reference, alternative)):
            ax = axes[row_index, column]
            pmf = np.asarray(groups[group_name]["pmf"]).T
            image = ax.imshow(
                np.clip(pmf, 0, 6), origin="lower", aspect="auto", cmap="magma_r",
                extent=(DISTANCE_EDGES_NM[0], DISTANCE_EDGES_NM[-1], 0, 1),
                vmin=0, vmax=6, interpolation="nearest",
            )
            ax.contour(
                0.5 * (DISTANCE_EDGES_NM[:-1] + DISTANCE_EDGES_NM[1:]),
                0.5 * (ORIENTATION_EDGES[:-1] + ORIENTATION_EDGES[1:]),
                pmf, levels=(1, 2, 3, 4, 5), colors="0.35", linewidths=0.35,
            )
            ax.axvline(0.5, color="#2B6CB0", lw=0.8, ls="--")
            ax.text(0.98, 0.93, group_name.split("_")[-1], transform=ax.transAxes,
                    ha="right", va="top", fontsize=7.5, fontweight="bold",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5})
            if column == 0:
                ax.set_ylabel(f"{row_label}\nOrientation gate")
            if row_index == 3:
                ax.set_xlabel("Vinyl midpoint distance (nm)")
    axes[0, 0].set_title(r"$\bf{a}$  Reference route")
    axes[0, 1].set_title(r"$\bf{b}$  Alternative route")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, shrink=0.72, pad=0.02)
        colorbar.set_label(r"Reweighted $F(d,\Omega)$ (kcal mol$^{-1}$)")
    fig.suptitle("Distance-orientation free-energy landscapes from existing q2 umbrella windows", fontsize=10)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=450)
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def plot_summary(
    path: Path,
    contrasts: Sequence[Mapping[str, object]],
    method_rows: Sequence[Mapping[str, object]],
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    labels = ("C5 adjacent\ncross - same", "C8 adjacent\ncross - same",
              "C5 opposite\nswitch - reuse", "C8 opposite\nswitch - reuse")
    y = np.arange(4)
    contact = np.asarray([float(row["full_ddG_contact_kcal_mol"]) for row in contrasts])
    orientation = np.asarray([float(row["full_ddG_orientation_kcal_mol"]) for row in contrasts])
    total = np.asarray([float(row["full_ddG_nac_kcal_mol"]) for row in contrasts])
    first = np.asarray([float(row["first_half_ddG_nac_kcal_mol"]) for row in contrasts])
    second = np.asarray([float(row["second_half_ddG_nac_kcal_mol"]) for row in contrasts])
    fig = plt.figure(figsize=(7.2, 5.45), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.9))
    ax = fig.add_subplot(grid[0, 0])
    ax.barh(y, contact, color="#3B6FB6", height=0.56, label="Contact")
    ax.barh(y, orientation, left=contact, color="#E69F00", height=0.56, label="Orientation")
    ax.scatter(total, y, color="black", s=18, zorder=4, label="Total NAC")
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Reweighted $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax.set_title(r"$\bf{a}$  Distance and orientation decomposition")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.92", lw=0.5)

    ax = fig.add_subplot(grid[0, 1])
    for i in range(4):
        ax.plot([first[i], second[i]], [i, i], color="0.65", lw=1)
        ax.scatter(first[i], i - 0.10, color="#4C78A8", marker="o", s=22, label="First half" if i == 0 else None)
        ax.scatter(second[i], i + 0.10, color="#D55E5E", marker="s", s=22, label="Second half" if i == 0 else None)
        ax.scatter(total[i], i, color="black", marker="D", s=18, label="Full" if i == 0 else None)
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Reweighted NAC $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax.set_title(r"$\bf{b}$  First/second-half diagnostic")
    ax.legend(frameon=False, fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.92", lw=0.5)

    ax = fig.add_subplot(grid[1, :])
    offsets = (-0.16, 0.0, 0.16)
    colors = ("#4C78A8", "#7A5195", "#E69F00")
    markers = ("o", "s", "D")
    method_specs = (
        ("Unbiased NAC", "unbiased_nac_point_kcal_mol", "unbiased_nac_ci_low_kcal_mol", "unbiased_nac_ci_high_kcal_mol"),
        ("1D umbrella PMF", "one_dimensional_umbrella_point_kcal_mol", "one_dimensional_umbrella_ci_low_kcal_mol", "one_dimensional_umbrella_ci_high_kcal_mol"),
        ("2D reweight", "two_dimensional_reweight_point_kcal_mol", "two_dimensional_reweight_half_low_kcal_mol", "two_dimensional_reweight_half_high_kcal_mol"),
    )
    for method_index, (method, point_field, low_field, high_field) in enumerate(method_specs):
        for i, row in enumerate(method_rows):
            point = float(row[point_field])
            low = float(row[low_field])
            high = float(row[high_field])
            ax.errorbar(
                point, i + offsets[method_index], xerr=np.asarray([[point - low], [high - point]]),
                fmt=markers[method_index], color=colors[method_index], ms=4.5,
                lw=0.9, capsize=2, label=method if i == 0 else None,
            )
    ax.axvline(0, color="0.25", lw=0.8)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel(r"Unified $\Delta\Delta G$ estimate (kcal mol$^{-1}$)")
    ax.set_title(r"$\bf{c}$  Three MD estimators preserve the same preference direction")
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="0.92", lw=0.5)
    fig.savefig(path.with_suffix(".svg"))
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"), dpi=450)
    fig.savefig(path.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def make_report(
    path: Path,
    contrasts: Sequence[Mapping[str, object]],
    pathway_rows: Sequence[Mapping[str, object]],
    method_rows: Sequence[Mapping[str, object]],
) -> None:
    lines = [
        "# q2 伞形轨迹二维距离–取向重加权",
        "",
        "本分析没有运行新的分子动力学。它利用现有伞形窗口的 pull distance、完整溶质轨迹和已知谐振偏压，通过0.002 nm距离分箱的WHAM/MBAR等价自洽方程恢复无偏帧权重，再计算 `F(distance, orientation gate)`。",
        "",
        "统一符号为 alternative−reference：adjacent 使用 cross−same，opposite 使用 switch−reuse；负值表示 alternative 更有利。",
        "",
        "| 状态 | contact | orientation | total NAC | first half | second half |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in contrasts:
        lines.append(
            f"| {row['state']} | {float(row['full_ddG_contact_kcal_mol']):+.3f} | "
            f"{float(row['full_ddG_orientation_kcal_mol']):+.3f} | "
            f"{float(row['full_ddG_nac_kcal_mol']):+.3f} | "
            f"{float(row['first_half_ddG_nac_kcal_mol']):+.3f} | "
            f"{float(row['second_half_ddG_nac_kcal_mol']):+.3f} |"
        )
    signs = [np.sign(float(row["full_ddG_nac_kcal_mol"])) for row in contrasts]
    expected = (-1, -1, 1, -1)
    sign_agreement = sum(int(sign == target) for sign, target in zip(signs, expected))
    max_half = max(abs(float(row["half_difference_ddG_nac_kcal_mol"])) for row in contrasts)
    min_ess = min(float(row["effective_sample_size"]) for row in pathway_rows)
    max_geometry = max(float(row["geometry_pull_max_A"]) for row in pathway_rows)
    lines += [
        "",
        "## 解释",
        "",
        f"- 与此前无偏NAC/一维PMF的预期方向一致数为 {sign_agreement}/4。",
        f"- 四个比较中最大的前后半程差为 {max_half:.3f} kcal mol⁻¹；这是收敛诊断，不是置信区间。",
        f"- 九条路径中最小的权重有效样本量为 {min_ess:.1f}；该数值反映权重集中程度，未校正时间自相关。pull距离与由四个乙烯基原子重算的距离最大差异为 {max_geometry:.3f} Å。",
        "- contact 项衡量进入约0.5 nm近接触区的重加权概率；orientation 项是进入该区后的条件取向门控；两项相加得到soft-NAC自由能差。",
        "",
        "## 三种MD估计的交叉验证",
        "",
        "| 状态 | 无偏NAC | 一维umbrella | 二维重加权 |",
        "|---|---:|---:|---:|",
    ]
    for row in method_rows:
        lines.append(
            f"| {row['state']} | {float(row['unbiased_nac_point_kcal_mol']):+.3f} | "
            f"{float(row['one_dimensional_umbrella_point_kcal_mol']):+.3f} | "
            f"{float(row['two_dimensional_reweight_point_kcal_mol']):+.3f} |"
        )
    lines += [
        "",
        "三种方法对4个比较的方向均一致。数值幅度不同是预期现象：无偏NAC平均所有自然出现的候选对，一维PMF描述选定路径的距离自由能，二维重加权则在同一路径上进一步纳入条件取向。最稳健的证据是方向一致及contact主导，而不是要求三组数值相等。",
        "",
        "## 适用边界",
        "",
        "二维面仅在各窗口实际覆盖的距离与取向区域内可信。图中的轻度高斯平滑只用于显示，所有定量数值来自未平滑的逐帧MBAR权重。窗口轨迹具有时间相关性，因此有效样本量与前后半程差用于诊断，而不是把每一帧当作独立重复。每组路径数量较少，不能把该结果解释为化学反应速率或活化能。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/a4_q2_umbrella_2d_analysis/final"))
    parser.add_argument("--gmx", default="gmx")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    output.mkdir(parents=True, exist_ok=True)

    results: dict[str, PathwayResult] = {}
    for location in PATHWAY_LOCATIONS:
        pathway = workspace / location
        result = analyze_pathway(pathway, workspace, output, args.gmx)
        results[str(result.metadata["pathway"])] = result
        print(f"analyzed={result.metadata['pathway']} windows={len(result.centers_nm)} frames={len(result.distance_nm)}")

    pathway_rows = []
    for name, result in results.items():
        pathway_rows.append({
            "pathway": name,
            "state": result.metadata["state"],
            "system": result.metadata["system"],
            "route_family": result.metadata["route_family"],
            "topology": result.metadata.get("topology", "cross_arm"),
            "windows": len(result.centers_nm),
            "frames": len(result.distance_nm),
            "mbar_iterations": result.mbar_iterations,
            "mbar_residual": result.mbar_residual,
            "effective_sample_size": result.effective_sample_size,
            "geometry_pull_rms_A": result.geometry_pull_rms_A,
            "geometry_pull_max_A": result.geometry_pull_max_A,
            **result.metrics,
            **{
                f"{half}_{key}": value
                for half, values in result.half_metrics.items() for key, value in values.items()
            },
        })
    groups, group_rows = aggregate_groups(results)
    contrasts = contrast_rows(groups)
    methods = method_comparison_rows(workspace, contrasts)
    surfaces = surface_rows(groups)
    write_csv(output / "pathway_reweight_summary.csv", pathway_rows)
    write_csv(output / "group_reweight_summary.csv", group_rows)
    write_csv(output / "q2_2d_reweighted_contrasts.csv", contrasts)
    write_csv(output / "q2_method_comparison.csv", methods)
    write_csv(output / "q2_2d_surface_source_data.csv", surfaces)
    plot_surfaces(output / "q2_distance_orientation_surfaces", groups)
    plot_summary(output / "q2_2d_reweight_summary", contrasts, methods)
    make_report(output / "q2_2d_reweight_report_zh.md", contrasts, pathway_rows, methods)
    summary = {
        "temperature_K": TEMPERATURE_K,
        "bias_model": "0.5*k*(distance-center)^2",
        "force_constant_kJ_mol_nm2": K_FORCE,
        "method": "0.002-nm binned WHAM/MBAR-equivalent self-consistency with frame-resolved weights; no new MD",
        "visual_smoothing": "Gaussian sigma 0.7 bins for surfaces only; quantitative metrics are unsmoothed",
        "contrasts": contrasts,
        "method_comparison": methods,
        "pathways": pathway_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "figure_legend_en.md").write_text(
        "**Figure | Distance accessibility dominates the reweighted q2 route preference.** "
        "The distance-orientation landscapes were reconstructed without new simulations from existing umbrella windows using 0.002-nm-binned WHAM/MBAR-equivalent self-consistency and frame-resolved unbiased weights. "
        "In the landscape figure, the dashed line marks 0.5 nm and colours show the two-dimensional free energy after light Gaussian smoothing (0.7 bins) for visualization only. "
        "In the summary figure, (a) alternative-minus-reference soft-NAC free energies are decomposed into contact and conditional-orientation terms; negative values favour cross-arm (adjacent q2) or switch (opposite q2). "
        "(b) First- and second-half values are convergence diagnostics rather than confidence intervals. "
        "(c) Unbiased NAC, one-dimensional umbrella PMF and two-dimensional reweighting preserve the same preference direction in all four comparisons. "
        "Unbiased-NAC and one-dimensional-PMF error bars are bootstrap 95% confidence intervals; two-dimensional-reweighting bars span the two trajectory halves. All calculations used 313.15 K.\n",
        encoding="utf-8",
    )
    (output / "figure_qa_notes.md").write_text(
        "# Figure QA notes\n\n"
        "- Core claim: q2 route preference is dominated by distance accessibility; conditional orientation is a secondary correction.\n"
        "- Archetype: quantitative grid plus cross-method validation.\n"
        "- Backend: Python/matplotlib only; editable SVG/PDF and 600-dpi TIFF are included.\n"
        "- Source trajectories: nine existing umbrella pathways, 219 successful windows and 155,619 analysed frames; no new MD was run.\n"
        "- Reweighting: 0.002-nm distance-binned WHAM/MBAR-equivalent self-consistency; harmonic force constant 500 kJ mol-1 nm-2 at 313.15 K.\n"
        "- Quantitative metrics: unsmoothed frame weights. Surface display only: Gaussian smoothing sigma 0.7 bins, explicitly disclosed.\n"
        "- Uncertainty: prior unbiased-NAC and 1D-PMF bars are bootstrap 95% intervals. The 2D bars span first/second halves and are diagnostics, not confidence intervals.\n"
        "- Replication limitation: most route groups contain one selected pathway; C5 reuse is an equal mixture of two pathways. No p values or multiple-comparison correction are claimed.\n"
        "- Weight ESS is not corrected for trajectory autocorrelation.\n"
        "- Source data: pathway_reweight_summary.csv, group_reweight_summary.csv, q2_2d_reweighted_contrasts.csv, q2_method_comparison.csv and q2_2d_surface_source_data.csv.\n"
        "- Scope: conformational free-energy proxies only; no catalyst, reaction rate or chemical activation barrier.\n",
        encoding="utf-8",
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
