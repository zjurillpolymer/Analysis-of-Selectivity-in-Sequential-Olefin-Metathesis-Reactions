#!/usr/bin/env python3
"""Post-process the B6 q4 V1--V5 distance umbrella for orientation sampling.

No new dynamics are generated.  The script reconstructs unbiased frame weights
from the 25 distance windows and applies the same soft near-attack geometry gate
used for the A4 analyses: approximately parallel vinyl axes and a side-on
midpoint approach.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_a4_q2_umbrella_2d_reweight import (
    RT_KCAL,
    mbar_weights,
    orientation_geometry_matching_pull,
    read_numeric_xvg,
    soft_lower,
    soft_upper,
)


def parse_center(path: Path) -> float:
    match = re.fullmatch(r"window_\d+_([0-9.]+)nm", path.name)
    if match is None:
        raise ValueError(f"cannot parse window center: {path}")
    return float(match.group(1))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(command: list[str], stdin: str | None = None) -> None:
    result = subprocess.run(
        command,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-40:])
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{tail}")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(np.interp(quantile, cumulative, values[order]))


def summarize(
    distance: np.ndarray,
    axis: np.ndarray,
    side: np.ndarray,
    gate: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    contact = soft_upper(distance, 0.50, 0.05)
    contact_mass = float(np.sum(weights * contact))
    contact_weights = weights * contact
    contact_weights /= np.sum(contact_weights)
    conditional_gate = float(np.sum(contact_weights * gate))
    hard_contact = distance <= 0.50
    hard_reactive = hard_contact & (axis <= 45.0) & (side >= 45.0)
    hard_contact_mass = float(np.sum(weights[hard_contact]))
    hard_reactive_fraction = float(np.sum(weights[hard_reactive]) / hard_contact_mass)
    return {
        "soft_contact_probability": contact_mass,
        "conditional_orientation_gate": conditional_gate,
        "orientation_penalty_kcal_mol": -RT_KCAL * math.log(conditional_gate),
        "soft_nac_probability": contact_mass * conditional_gate,
        "hard_contact_probability": hard_contact_mass,
        "hard_reactive_fraction_given_contact": hard_reactive_fraction,
        "contact_axis_q10_deg": weighted_quantile(axis, contact_weights, 0.10),
        "contact_axis_median_deg": weighted_quantile(axis, contact_weights, 0.50),
        "contact_axis_q90_deg": weighted_quantile(axis, contact_weights, 0.90),
        "contact_side_q10_deg": weighted_quantile(side, contact_weights, 0.10),
        "contact_side_median_deg": weighted_quantile(side, contact_weights, 0.50),
        "contact_side_q90_deg": weighted_quantile(side, contact_weights, 0.90),
        "effective_sample_size": float(1.0 / np.sum(weights**2)),
    }


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=project / "bohrium_download/b6_q4_v1_v5_umbrella_20260922_fix3/20767009/results",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=project / (
            "b6_q4_parameterization_bundle_20260921/gromacs_dcm/runs/"
            "B6_p_C6_q4_V3_V6_V11_V12_V8_V10_V7_V9_cis/setup/metadata.json"
        ),
    )
    parser.add_argument(
        "--solute-pdb",
        type=Path,
        default=project / (
            "b6_q4_parameterization_bundle_20260921/gromacs_dcm/runs/"
            "B6_p_C6_q4_V3_V6_V11_V12_V8_V10_V7_V9_cis/setup/solute_parameters/slu.pdb"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project / "results/b6_q4_v1_v5_orientation_analysis",
    )
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--begin-ps", type=float, default=100.0)
    parser.add_argument("--end-ps", type=float, default=500.0)
    args = parser.parse_args()

    output = args.output.resolve()
    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    mapping = metadata["vinyl_mapping_gromacs_1based"]
    pair_atoms = [int(value) for label in ("V1", "V5") for value in mapping[label]]
    pair_index = cache / "reactive_pair.ndx"
    pair_index.write_text(
        "[ Reactive_pair ]\n" + " ".join(str(value) for value in pair_atoms) + "\n",
        encoding="utf-8",
    )

    windows = sorted((args.results_root / "windows").glob("window_*"), key=parse_center)
    if len(windows) != 25:
        raise ValueError(f"expected 25 windows, found {len(windows)}")

    centers = np.asarray([parse_center(window) for window in windows])
    all_time: list[np.ndarray] = []
    all_distance: list[np.ndarray] = []
    all_axis: list[np.ndarray] = []
    all_side: list[np.ndarray] = []
    all_gate: list[np.ndarray] = []
    all_window: list[np.ndarray] = []
    raw_window_rows: list[dict[str, object]] = []

    for window_index, window in enumerate(windows):
        window_cache = cache / window.name
        window_cache.mkdir(parents=True, exist_ok=True)
        coordinate_path = window_cache / "reactive_pair_coordinates_nm.xvg"
        box_path = window_cache / "box_vectors_nm.xvg"
        if not coordinate_path.is_file() or not box_path.is_file():
            run(
                [
                    args.gmx,
                    "traj",
                    "-f",
                    str(window / "production.xtc"),
                    "-s",
                    str(args.solute_pdb),
                    "-n",
                    str(pair_index),
                    "-ox",
                    str(coordinate_path),
                    "-ob",
                    str(box_path),
                    "-b",
                    str(args.begin_ps),
                    "-e",
                    str(args.end_ps),
                    "-x",
                    "-y",
                    "-z",
                ],
                stdin="Reactive_pair\n",
            )
        coordinates = read_numeric_xvg(coordinate_path)
        boxes = read_numeric_xvg(box_path)
        pull = read_numeric_xvg(window / "production_pullx.xvg")
        keep = (pull[:, 0] >= args.begin_ps - 1.0e-6) & (pull[:, 0] <= args.end_ps + 1.0e-6)
        pull = pull[keep, :2]
        if coordinates.shape[1] != 13 or len(coordinates) != len(pull):
            raise ValueError(f"coordinate/pull mismatch in {window}")
        if not np.allclose(coordinates[:, 0], pull[:, 0], atol=1.0e-5):
            raise ValueError(f"time mismatch in {window}")
        geometry_distance, axis, side, physical = orientation_geometry_matching_pull(
            coordinates, boxes, pull[:, 1]
        )
        if not np.all(physical):
            raise ValueError(f"nonphysical vinyl coordinates in {window}")
        mismatch_A = (geometry_distance - pull[:, 1]) * 10.0
        gate = soft_upper(axis, 45.0, 7.5) * soft_lower(side, 45.0, 7.5)
        midpoint = 0.5 * (args.begin_ps + args.end_ps)
        raw_window_rows.append(
            {
                "window": window.name,
                "center_nm": centers[window_index],
                "n_frames": len(pull),
                "distance_mean_nm": float(np.mean(pull[:, 1])),
                "axis_mean_deg": float(np.mean(axis)),
                "side_mean_deg": float(np.mean(side)),
                "gate_mean": float(np.mean(gate)),
                "gate_first_half": float(np.mean(gate[pull[:, 0] <= midpoint])),
                "gate_second_half": float(np.mean(gate[pull[:, 0] > midpoint])),
                "geometry_pull_rms_A": float(np.sqrt(np.mean(mismatch_A**2))),
                "geometry_pull_max_A": float(np.max(np.abs(mismatch_A))),
            }
        )
        all_time.append(pull[:, 0])
        all_distance.append(pull[:, 1])
        all_axis.append(axis)
        all_side.append(side)
        all_gate.append(gate)
        all_window.append(np.full(len(pull), window_index, dtype=int))

    time = np.concatenate(all_time)
    distance = np.concatenate(all_distance)
    axis = np.concatenate(all_axis)
    side = np.concatenate(all_side)
    gate = np.concatenate(all_gate)
    window_index = np.concatenate(all_window)
    weights, free, iterations, residual = mbar_weights(distance, centers, window_index)
    full = summarize(distance, axis, side, gate, weights)

    half_results: dict[str, dict[str, float]] = {}
    midpoint = 0.5 * (args.begin_ps + args.end_ps)
    for name, selector in (
        ("first_half", time <= midpoint + 1.0e-6),
        ("second_half", time > midpoint + 1.0e-6),
    ):
        subset_weights, _, _, _ = mbar_weights(
            distance[selector], centers, window_index[selector]
        )
        half_results[name] = summarize(
            distance[selector], axis[selector], side[selector], gate[selector], subset_weights
        )

    contact_weights = weights * soft_upper(distance, 0.50, 0.05)
    contact_weights /= np.sum(contact_weights)
    basin_masks = {
        "reactive_parallel_side_on": (axis <= 45.0) & (side >= 45.0),
        "parallel_end_on": (axis <= 45.0) & (side < 45.0),
        "nonparallel_side_on": (axis > 45.0) & (side >= 45.0),
        "nonparallel_end_on": (axis > 45.0) & (side < 45.0),
    }
    basin_probabilities = {
        name: float(np.sum(contact_weights[mask])) for name, mask in basin_masks.items()
    }

    summary = {
        "analysis": "B6 q4 V1-V5 orientation post-analysis of existing distance umbrella",
        "source_results": str(args.results_root.resolve()),
        "state": metadata["state"],
        "pair": "V1-V5",
        "orientation_definition": {
            "axis": "acute angle between undirected C=C axes",
            "side": "smaller acute angle between either C=C axis and midpoint connector",
            "soft_gate": "sigmoid(axis <= 45 deg, width 7.5 deg) * sigmoid(side >= 45 deg, width 7.5 deg)",
        },
        "window_count": len(windows),
        "production_interval_ps": [args.begin_ps, args.end_ps],
        "frame_count": len(distance),
        "mbar_iterations": iterations,
        "mbar_residual": residual,
        "mbar_free_offsets": free.tolist(),
        "full": full,
        "halves": half_results,
        "contact_basin_probabilities": basin_probabilities,
        "maximum_geometry_pull_mismatch_A": max(
            float(row["geometry_pull_max_A"]) for row in raw_window_rows
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(output / "window_orientation_diagnostics.csv", raw_window_rows)

    frame_table = np.column_stack(
        (time, window_index, distance, axis, side, gate, weights)
    )
    np.savez_compressed(
        output / "frame_orientation_weights.npz",
        time_ps=time,
        window_index=window_index,
        distance_nm=distance,
        axis_angle_deg=axis,
        side_angle_deg=side,
        orientation_gate=gate,
        unbiased_weight=weights,
    )

    distance_edges = np.linspace(0.30, 2.80, 51)
    gate_edges = np.linspace(0.0, 1.0, 31)
    probability, _, _ = np.histogram2d(
        distance, gate, bins=(distance_edges, gate_edges), weights=weights
    )
    probability /= np.sum(probability)
    probability = np.ma.masked_where(probability <= 0.0, probability)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True)
    mesh = axes[0].pcolormesh(
        distance_edges, gate_edges, probability.T, shading="auto", cmap="viridis"
    )
    axes[0].axvline(0.50, color="white", linestyle="--", linewidth=1.0)
    axes[0].set(xlabel="V1–V5 midpoint distance (nm)", ylabel="Orientation gate")
    axes[0].set_title("Unbiased distance–orientation population")
    fig.colorbar(mesh, ax=axes[0], label="Probability mass")

    contact = distance <= 0.50
    axes[1].hist(
        gate[contact], bins=np.linspace(0, 1, 31), weights=weights[contact],
        density=True, color="#4477AA", alpha=0.82,
    )
    axes[1].axvline(0.5, color="black", linestyle="--", linewidth=1.0)
    axes[1].set(xlabel="Orientation gate at distance ≤ 0.50 nm", ylabel="Weighted density")
    axes[1].set_title("Contact-region orientation distribution")
    fig.savefig(output / "b6_q4_v1_v5_orientation_diagnostic.png", dpi=220)
    fig.savefig(output / "b6_q4_v1_v5_orientation_diagnostic.pdf")
    plt.close(fig)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
