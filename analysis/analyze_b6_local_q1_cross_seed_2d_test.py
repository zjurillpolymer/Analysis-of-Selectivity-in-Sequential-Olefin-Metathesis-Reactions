#!/usr/bin/env python3
"""Analyze reciprocal-start B6 q1 two-distance restrained mixing tests."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audit_b6_local_q1_basin_exchange import matched_model
from analyze_b6_local_q1_joint_mbar import OUTPUT as JOINT, SINGLE, minimum_image
from diagnose_b6_local_q1_hidden_conformations import (
    DEST as DIAG, FEATURE_NAMES, SOURCE_METADATA, VINYLS, fingerprint,
)
from extract_a4_q2_umbrella_qm_candidates import read_numeric_xvg


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD = ROOT.parent / "bohrium_download/b6_local_q1_cross_seed_2d_mixing_20261001"
OUT = JOINT / "hidden_conformation_diagnostic/cross_seed_test_analysis"
JOBS = {"A6_origin": 20807020, "A2_origin": 20807021}
TIME_PS = 2000.0
BLOCK_PS = 100.0
ACTIVE_VINYLS = tuple(f"V{i}" for i in range(3, 13))
PAIR_INDICES = [(i, j) for i in range(10) for j in range(i + 1, 10)]


def extract(job: int, origin: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch = "A1_to_A6" if origin == "A6_origin" else "A1_to_A2"
    topology = SINGLE / "reactive_pair_topologies" / f"b6_basin02_{branch}" / "solute_topology.gro"
    work = DOWNLOAD / str(job) / "extracted/work"
    cache = OUT / "trajectory_cache" / str(job)
    cache.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))["vinyl_mapping_gromacs_1based"]
    atom_ids = [atom for name in ACTIVE_VINYLS for atom in mapping[name]]
    ndx = cache / "active20.ndx"
    ndx.write_text("[ Active20 ]\n" + " ".join(map(str, atom_ids)) + "\n", encoding="utf-8")
    coords_path, box_path = cache / "active20_coordinates.xvg", cache / "box_vectors.xvg"
    result = subprocess.run([
        "gmx", "traj", "-f", str(work / "production.xtc"), "-s", str(topology),
        "-n", str(ndx), "-ox", str(coords_path), "-ob", str(box_path),
        "-b", "0", "-e", str(TIME_PS), "-x", "-y", "-z",
    ], input="Active20\n", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"gmx traj failed for {job}:\n{result.stdout[-3000:]}")
    coords, box = read_numeric_xvg(coords_path), read_numeric_xvg(box_path)
    pull = read_numeric_xvg(work / "production_pullx.xvg")
    if coords.shape != (4001, 61) or box.shape != (4001, 7) or pull.shape != (4001, 3):
        raise ValueError(f"unexpected output dimensions: {coords.shape}, {box.shape}, {pull.shape}")
    if not (np.allclose(coords[:, 0], box[:, 0], atol=1e-5) and np.allclose(coords[:, 0], pull[:, 0], atol=1e-5)):
        raise ValueError("trajectory, box, and pull times do not align")
    return coords, box, pull


def block_mean(values: np.ndarray, times: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers, means = [], []
    for start in bins:
        mask = (times >= start) & (times < start + BLOCK_PS)
        if np.count_nonzero(mask) == 0:
            continue
        centers.append(start + BLOCK_PS / 2)
        means.append(values[mask].mean(axis=0))
    return np.asarray(centers), np.asarray(means)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mean, scale, pc1, _, pc_centers = matched_model(2)
    keep = [i for i, name in enumerate(FEATURE_NAMES) if name not in ("V3-V12", "V3-V10")]
    records, series = {}, {}
    for origin, job in JOBS.items():
        coords, box, pull = extract(job, origin)
        dist, radius = fingerprint(coords, box)
        x_idx, y_idx = FEATURE_NAMES.index("V3-V12"), FEATURE_NAMES.index("V3-V10")
        err_x = float(np.max(np.abs(dist[:, x_idx] - pull[:, 1])))
        err_y = float(np.max(np.abs(dist[:, y_idx] - pull[:, 2])))
        if max(err_x, err_y) > 0.005:
            raise ValueError(f"geometry/pull disagreement: {err_x}, {err_y} nm")
        feature = np.column_stack((dist[:, keep], radius))
        projected = ((feature - mean) / scale) @ pc1
        standardized = (feature - mean) / scale
        times = coords[:, 0]
        bins = np.arange(0, TIME_PS, BLOCK_PS)
        block_time, block_feature = block_mean(feature, times, bins)
        _, block_feature_z = block_mean(standardized, times, bins)
        _, block_pull = block_mean(pull[:, 1:], times, bins)
        _, block_pc = block_mean(projected[:, None], times, bins)
        series[origin] = {"times": times, "pull": pull[:, 1:], "pc": projected,
                          "block_time": block_time, "block_feature": block_feature,
                          "block_feature_z": block_feature_z,
                          "block_pull": block_pull, "block_pc": block_pc[:, 0]}
        records[origin] = {
            "bohr_job_id": job,
            "initial_mean_1ns_equilibration_not_in_xtc": True,
            "production_frames": int(len(times)), "production_duration_ns": float((times[-1] - times[0]) / 1000),
            "x_pull_vs_geometry_max_error_nm": err_x, "y_pull_vs_geometry_max_error_nm": err_y,
            "target_nm": 1.75,
            "x_mean_sd_nm": [float(pull[:, 1].mean()), float(pull[:, 1].std())],
            "y_mean_sd_nm": [float(pull[:, 2].mean()), float(pull[:, 2].std())],
            "x_first_last_500ps_means_nm": [float(pull[times < 500, 1].mean()), float(pull[times >= 1500, 1].mean())],
            "y_first_last_500ps_means_nm": [float(pull[times < 500, 2].mean()), float(pull[times >= 1500, 2].mean())],
            "pc1_first_last_500ps_means": [float(projected[times < 500].mean()), float(projected[times >= 1500].mean())],
            "pc1_first_last_500ps_sd": [float(projected[times < 500].std()), float(projected[times >= 1500].std())],
            "feature_block_count": int(len(block_time)),
        }
    left, right = series["A6_origin"], series["A2_origin"]
    if not np.allclose(left["block_time"], right["block_time"]):
        raise ValueError("the two simulations do not have aligned 100 ps blocks")
    block_distance = np.linalg.norm(left["block_feature_z"] - right["block_feature_z"], axis=1)
    pc_gap = left["block_pc"] - right["block_pc"]
    records["between_start_comparison"] = {
        "blocks_ps": BLOCK_PS,
        "fingerprint_mean_distance_first_500ps_z_euclidean": float(block_distance[:5].mean()),
        "fingerprint_mean_distance_last_500ps_z_euclidean": float(block_distance[-5:].mean()),
        "fingerprint_distance_by_100ps_block": block_distance.tolist(),
        "pc1_gap_first_500ps": float(pc_gap[:5].mean()),
        "pc1_gap_last_500ps": float(pc_gap[-5:].mean()),
        "pc1_gap_by_100ps_block": pc_gap.tolist(),
        "warning": "Blocks and trajectory frames are autocorrelated; these are descriptive contrasts, not independent-sample tests or PMF estimates.",
    }
    (OUT / "summary.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    feature_names = [FEATURE_NAMES[i] for i in keep] + ["vinyl_centroid_radius"]
    with (OUT / "block_feature_means.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["block_center_ps", "origin", "x_mean_nm", "y_mean_nm", "pc1_mean"] + feature_names)
        writer.writeheader()
        for origin, data in series.items():
            for i, t in enumerate(data["block_time"]):
                writer.writerow({"block_center_ps": t, "origin": origin,
                                 "x_mean_nm": data["block_pull"][i, 0], "y_mean_nm": data["block_pull"][i, 1],
                                 "pc1_mean": data["block_pc"][i],
                                 **dict(zip(feature_names, data["block_feature"][i]))})
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    colors = {"A6_origin": "#147d83", "A2_origin": "#b45309"}
    for origin, data in series.items():
        color = colors[origin]
        axes[0].plot(data["times"], data["pull"][:, 0], color=color, alpha=0.15, lw=0.5)
        axes[0].plot(data["block_time"], data["block_pull"][:, 0], color=color, label=f"{origin}: d(V3,V12)")
        axes[0].plot(data["block_time"], data["block_pull"][:, 1], color=color, ls="--", label=f"{origin}: d(V3,V10)")
        axes[1].plot(data["block_time"], data["block_pc"], color=color, marker="o", ms=3, label=origin)
    axes[0].axhline(1.75, color="black", ls=":", lw=1)
    axes[0].set_ylabel("restrained distances (nm)")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].axhline(pc_centers["A6"], color=colors["A6_origin"], ls=":", lw=1)
    axes[1].axhline(pc_centers["A2"], color=colors["A2_origin"], ls=":", lw=1)
    axes[1].set_ylabel("contact fingerprint PC1\n(100 ps block mean)")
    axes[1].legend(fontsize=8)
    axes[2].plot(left["block_time"], block_distance, color="#553c9a", marker="o", ms=3)
    axes[2].set_ylabel("between-start fingerprint\ndistance (z Euclidean)")
    axes[2].set_xlabel("production time (ps)")
    fig.savefig(OUT / "reciprocal_start_convergence.png", dpi=180)
    plt.close(fig)
    print(json.dumps(records["between_start_comparison"], indent=2))


if __name__ == "__main__":
    main()
