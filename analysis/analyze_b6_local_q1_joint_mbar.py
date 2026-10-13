#!/usr/bin/env python3
"""Jointly reweight A6/A2 umbrella families for B6 q1 within paired starts."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from analyze_explicit_dcm_q0_geometry_gate import soft_lower, soft_upper
from extract_a4_q2_umbrella_qm_candidates import read_numeric_xvg


PROJECT = Path(__file__).resolve().parents[1]
DOWNLOAD = PROJECT.parent / "bohrium_download/b6_local_q1_paired_umbrella_complete_20261001"
SOURCE_METADATA = PROJECT.parent / (
    "bohrium_download/b6_neighbor_md_extensions_2to10ns_20260918/20750428/"
    "extracted/workspace/gromacs_dcm/runs/B6_p_C6_q1_V1_V2_cis/setup/metadata.json"
)
SINGLE = PROJECT / "results/b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001"
OUTPUT = SINGLE / "joint_mbar"
JOBS = {
    1: {"A6": 20806766, "A2": 20806768},
    2: {"A6": 20806769, "A2": 20806770},
    3: {"A6": 20806772, "A2": 20806773},
}
TEMP_K = 313.15
R_KJ = 0.00831446261815324
R_KCAL = 0.00198720425864083
RT_KCAL = R_KCAL * TEMP_K
BETA_KJ = 1.0 / (R_KJ * TEMP_K)
K_FORCE = 500.0
BIN_NM = 0.02


def midpoint(first: np.ndarray, second: np.ndarray, box: np.ndarray) -> np.ndarray:
    return first + 0.5 * minimum_image(second - first, box)


def minimum_image(vector: np.ndarray, box: np.ndarray) -> np.ndarray:
    return vector - box * np.round(vector / box)


def geometry(coordinates: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, ...]:
    """Coordinates are V3, V10, V12; each vinyl is represented by two atoms."""
    atom = coordinates.reshape(-1, 6, 3)
    vinyl = [midpoint(atom[:, i], atom[:, i + 1], box) for i in (0, 2, 4)]
    axis = [minimum_image(atom[:, i + 1] - atom[:, i], box) for i in (0, 2, 4)]
    lengths = [np.linalg.norm(value, axis=1) for value in axis]
    if any(np.any((value < 0.115) | (value > 0.155)) for value in lengths):
        raise ValueError("nonphysical vinyl bond length in triad extraction")
    distances: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    for right in (2, 1):  # A6=V12, A2=V10
        connector = minimum_image(vinyl[right] - vinyl[0], box)
        distance = np.linalg.norm(connector, axis=1)
        left_axis = axis[0] / lengths[0][:, None]
        right_axis = axis[right] / lengths[right][:, None]
        connector_axis = connector / distance[:, None]
        axis_angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(left_axis * right_axis, axis=1)), 0, 1)))
        side_left = np.degrees(np.arccos(np.clip(np.abs(np.sum(left_axis * connector_axis, axis=1)), 0, 1)))
        side_right = np.degrees(np.arccos(np.clip(np.abs(np.sum(right_axis * connector_axis, axis=1)), 0, 1)))
        side_angle = np.minimum(side_left, side_right)
        distances.append(distance)
        gates.append(soft_upper(axis_angle, 45.0, 7.5) * soft_lower(side_angle, 45.0, 7.5))
    return distances[0], distances[1], gates[0], gates[1]


def run_traj(pathway: Path, job_id: int, window: Path, cache: Path) -> None:
    name = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))["pathway"]
    topology = SINGLE / "reactive_pair_topologies" / name / "solute_topology.gro"
    if not topology.is_file():
        raise FileNotFoundError(topology)
    mapping = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))["vinyl_mapping_gromacs_1based"]
    atom_ids = mapping["V3"] + mapping["V10"] + mapping["V12"]
    if len(atom_ids) != 6 or max(atom_ids) > 438:
        raise ValueError(f"invalid vinyl triad groups: {atom_ids}")
    cache.mkdir(parents=True, exist_ok=True)
    triad_index = cache / "triad.ndx"
    if not triad_index.is_file():
        triad_index.write_text("[ Triad ]\n" + " ".join(map(str, atom_ids)) + "\n", encoding="utf-8")
    coordinates = cache / "triad_coordinates.xvg"
    box = cache / "box_vectors.xvg"
    if coordinates.is_file() and box.is_file():
        return
    result = subprocess.run([
        "gmx", "traj", "-f", str(window / "production.xtc"), "-s", str(topology),
        "-n", str(triad_index), "-ox", str(coordinates), "-ob", str(box),
        "-b", "100", "-e", "500", "-x", "-y", "-z",
    ], input="Triad\n", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"gmx traj failed for {job_id}/{window.name}:\n{result.stdout[-2500:]}")


def window_data(pair_id: int, family: str, job_id: int) -> list[dict[str, object]]:
    pathway = DOWNLOAD / str(job_id) / "extracted/pathway"
    rows: list[dict[str, object]] = []
    for index, window in enumerate(sorted((pathway / "windows").glob("window_*"))):
        if index >= 25:
            raise ValueError("unexpected umbrella window count")
        cache = OUTPUT / "cache" / str(job_id) / window.name
        run_traj(pathway, job_id, window, cache)
        coords = read_numeric_xvg(cache / "triad_coordinates.xvg")
        box = read_numeric_xvg(cache / "box_vectors.xvg")
        pull = read_numeric_xvg(window / "production_pullx.xvg")
        pull = pull[(pull[:, 0] >= 100 - 1.0e-6) & (pull[:, 0] <= 500 + 1.0e-6)]
        if coords.shape != (801, 19) or box.shape != (801, 7) or pull.shape[0] != 801:
            raise ValueError(f"incomplete triad extraction: {job_id}/{window.name}")
        if not np.allclose(coords[:, 0], box[:, 0], atol=1e-5) or not np.allclose(coords[:, 0], pull[:, 0], atol=1e-5):
            raise ValueError(f"frame-time mismatch: {job_id}/{window.name}")
        if np.max(np.abs(box[:, 4:])) > 1.0e-6:
            raise ValueError("triclinic box not supported")
        x, y, omega_x, omega_y = geometry(coords[:, 1:], box[:, 1:4])
        biased = x if family == "A6" else y
        discrepancy = np.abs(biased - pull[:, 1])
        if float(np.max(discrepancy)) > 0.005:
            raise ValueError(f"biased pull/geometry disagreement {job_id}/{window.name}: {np.max(discrepancy):g} nm")
        rows.append({
            "job_id": job_id, "family": family, "window_index": index,
            "center_nm": 0.35 + 0.1 * index,
            "time_ps": pull[:, 0], "x_nm": x, "y_nm": y,
            "omega_x": omega_x, "omega_y": omega_y,
            "pull_max_abs_error_A": float(np.max(discrepancy) * 10),
        })
    return rows


def solve(states: list[dict[str, object]], selector: str) -> dict[str, object]:
    if selector == "full":
        keep = lambda time: np.ones(len(time), dtype=bool)
    elif selector == "first_half":
        keep = lambda time: time <= 300 + 1e-6
    else:
        keep = lambda time: time > 300 + 1e-6
    x = np.concatenate([row["x_nm"][keep(row["time_ps"])] for row in states])
    y = np.concatenate([row["y_nm"][keep(row["time_ps"])] for row in states])
    ox = np.concatenate([row["omega_x"][keep(row["time_ps"])] for row in states])
    oy = np.concatenate([row["omega_y"][keep(row["time_ps"])] for row in states])
    state_index = np.concatenate([
        np.full(np.count_nonzero(keep(row["time_ps"])), index, dtype=int)
        for index, row in enumerate(states)
    ])
    nstate = len(states)
    counts = np.bincount(state_index, minlength=nstate).astype(float)
    centers = np.asarray([row["center_nm"] for row in states])
    is_a6 = np.asarray([row["family"] == "A6" for row in states])
    # Binned 2D WHAM equations give window free energies efficiently. We then
    # evaluate every original frame's exact harmonic bias for the final weights.
    bin_index = np.floor(np.column_stack((x, y)) / BIN_NM).astype(int)
    unique, inverse, occupancy = np.unique(bin_index, axis=0, return_inverse=True, return_counts=True)
    bin_centers = (unique.astype(float) + 0.5) * BIN_NM
    chosen = np.where(is_a6[:, None], bin_centers[None, :, 0], bin_centers[None, :, 1])
    reduced_bias_bins = BETA_KJ * 0.5 * K_FORCE * (chosen - centers[:, None]) ** 2
    log_counts = np.log(counts)
    free = np.zeros(nstate)
    residual = math.inf
    for iteration in range(1, 30001):
        denominator = logsumexp(log_counts[:, None] + free[:, None] - reduced_bias_bins, axis=0)
        log_mass = np.log(occupancy) - denominator
        updated = -logsumexp(log_mass[None, :] - reduced_bias_bins, axis=1)
        updated -= updated[0]
        residual = float(np.max(np.abs(updated - free)))
        free = 0.5 * (free + updated)
        if residual < 1e-9:
            break
    else:
        raise RuntimeError(f"joint WHAM/MBAR did not converge: residual={residual:g}")
    chosen_frames = np.where(is_a6[:, None], x[None, :], y[None, :])
    reduced_bias_frames = BETA_KJ * 0.5 * K_FORCE * (chosen_frames - centers[:, None]) ** 2
    logits = log_counts[:, None] + free[:, None] - reduced_bias_frames
    denominator = logsumexp(logits, axis=0)
    log_weights = -denominator
    log_weights -= logsumexp(log_weights)
    weights = np.exp(log_weights)
    post_a6 = np.exp(logsumexp(logits[is_a6], axis=0) - denominator)
    post_a2 = 1.0 - post_a6
    contact_x = soft_upper(x, 0.50, 0.05)
    contact_y = soft_upper(y, 0.50, 0.05)
    nac_x = contact_x * ox
    nac_y = contact_y * oy
    p_x = float(np.sum(weights * nac_x))
    p_y = float(np.sum(weights * nac_y))
    contributions_x = weights * nac_x
    contributions_y = weights * nac_y
    per_window_x = np.bincount(state_index, weights=contributions_x, minlength=nstate)
    per_window_y = np.bincount(state_index, weights=contributions_y, minlength=nstate)
    coarse = np.floor(np.column_stack((x, y)) / 0.10).astype(int)
    _, coarse_inverse = np.unique(coarse, axis=0, return_inverse=True)
    coarse_a6 = np.bincount(coarse_inverse[state_index < 25], minlength=coarse_inverse.max() + 1)
    coarse_a2 = np.bincount(coarse_inverse[state_index >= 25], minlength=coarse_inverse.max() + 1)
    coarse_overlap = float(np.minimum(coarse_a6 / coarse_a6.sum(), coarse_a2 / coarse_a2.sum()).sum())
    return {
        "frames": len(x), "occupied_2d_bins": len(unique),
        "iteration_count": iteration, "residual": residual,
        "reweight_kish_ess": float(1.0 / np.sum(weights**2)),
        "family_bridge_A6_to_A2": float(np.mean(post_a2[state_index < 25])),
        "family_bridge_A2_to_A6": float(np.mean(post_a6[state_index >= 25])),
        "coarse_2d_family_histogram_overlap": coarse_overlap,
        "shared_0p1nm_cells_with_5_frames_each": int(np.count_nonzero((coarse_a6 >= 5) & (coarse_a2 >= 5))),
        "A6_NAC_fraction_from_A6_family_windows": float(np.sum(contributions_x[state_index < 25]) / p_x),
        "A2_NAC_fraction_from_A2_family_windows": float(np.sum(contributions_y[state_index >= 25]) / p_y),
        "A6_NAC_weighted_frame_ESS": float(p_x**2 / np.sum(contributions_x**2)),
        "A2_NAC_weighted_frame_ESS": float(p_y**2 / np.sum(contributions_y**2)),
        "A6_NAC_largest_single_window_fraction": float(per_window_x.max() / p_x),
        "A2_NAC_largest_single_window_fraction": float(per_window_y.max() / p_y),
        "contact_A6": float(np.sum(weights * contact_x)),
        "contact_A2_per_endpoint": float(np.sum(weights * contact_y)),
        "conditional_orientation_A6": p_x / float(np.sum(weights * contact_x)),
        "conditional_orientation_A2": p_y / float(np.sum(weights * contact_y)),
        "nac_A6": p_x, "nac_A2_per_endpoint": p_y,
        "ratio_A6_over_2A2": p_x / (2 * p_y),
        "deltaG_A6_minus_A2_kcal_mol": -RT_KCAL * math.log(p_x / (2 * p_y)),
        "P_A6_if_two_channels_only": p_x / (p_x + 2 * p_y),
        "P_A2_if_two_channels_only": 2 * p_y / (p_x + 2 * p_y),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=int, choices=(1, 2, 3), action="append")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = args.pair or [1, 2, 3]
    all_results = []
    for pair_id in selected:
        states = []
        for family, job_id in JOBS[pair_id].items():
            states.extend(window_data(pair_id, family, job_id))
            print(f"pair {pair_id} {family}: extracted 25 windows", flush=True)
        if len(states) != 50:
            raise ValueError("joint analysis requires exactly 50 windows")
        summary = {"pair_id": pair_id, "jobs": JOBS[pair_id],
                   "maximum_biased_pull_error_A": max(row["pull_max_abs_error_A"] for row in states)}
        for segment in ("full", "first_half", "second_half"):
            summary[segment] = solve(states, segment)
            print(f"pair {pair_id} {segment}: ratio={summary[segment]['ratio_A6_over_2A2']:.5g}, "
                  f"bridge={summary[segment]['family_bridge_A6_to_A2']:.3g}/"
                  f"{summary[segment]['family_bridge_A2_to_A6']:.3g}", flush=True)
        path = OUTPUT / f"pair_{pair_id:02d}.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_results.append(summary)
    (OUTPUT / "summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
