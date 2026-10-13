#!/usr/bin/env python3
"""Analyze the paired 5 ns B6 q1 2D-restraint follow-up trajectories."""

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
from analyze_b6_local_q1_joint_mbar import SINGLE
from diagnose_b6_local_q1_hidden_conformations import (
    FEATURE_NAMES, SOURCE_METADATA, VINYLS, fingerprint,
)
from extract_a4_q2_umbrella_qm_candidates import read_numeric_xvg


PROJECT = Path(__file__).resolve().parents[1]
DOWNLOAD = PROJECT.parent / "bohrium_download/b6_local_q1_cross_seed_2d_extension_20261001"
OUT = PROJECT / "results/b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001/joint_mbar/hidden_conformation_diagnostic/extension_5ns_analysis"
JOBS = {"A6_origin": 20807597, "A2_origin": 20807598}
END_PS = 5000.0
BLOCK_PS = 250.0
VINYL_PAIR_INDICES = [(i, j) for i in range(10) for j in range(i + 1, 10)]


def extract(job: int, origin: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    branch = "A1_to_A6" if origin == "A6_origin" else "A1_to_A2"
    topology = SINGLE / "reactive_pair_topologies" / f"b6_basin02_{branch}" / "solute_topology.gro"
    work = DOWNLOAD / str(job) / "extracted/work"
    cache = OUT / "trajectory_cache" / str(job)
    cache.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))["vinyl_mapping_gromacs_1based"]
    atom_ids = [atom for name in VINYLS for atom in mapping[name]]
    ndx = cache / "active20.ndx"
    ndx.write_text("[ Active20 ]\n" + " ".join(map(str, atom_ids)) + "\n", encoding="utf-8")
    coords_path, box_path = cache / "active20_coordinates.xvg", cache / "box_vectors.xvg"
    result = subprocess.run([
        "gmx", "traj", "-f", str(work / "extension.xtc"), "-s", str(topology),
        "-n", str(ndx), "-ox", str(coords_path), "-ob", str(box_path),
        "-b", "0", "-e", str(END_PS), "-x", "-y", "-z",
    ], input="Active20\n", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"gmx traj failed for {job}:\n{result.stdout[-2500:]}")
    coords, box = read_numeric_xvg(coords_path), read_numeric_xvg(box_path)
    pull = read_numeric_xvg(work / "extension_pullx.xvg")
    if coords.shape != (10001, 61) or box.shape != (10001, 7) or pull.shape != (10001, 3):
        raise ValueError(f"unexpected trajectory arrays {coords.shape} {box.shape} {pull.shape}")
    if not (np.allclose(coords[:, 0], box[:, 0], atol=1e-5) and np.allclose(coords[:, 0], pull[:, 0], atol=1e-5)):
        raise ValueError("coordinate, box and restraint frame times do not match")
    return coords, box, pull


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mean, scale, pc_axis, pc_threshold, pc_centers = matched_model(2)
    keep = [i for i, name in enumerate(FEATURE_NAMES) if name not in ("V3-V12", "V3-V10")]
    feature_names = [FEATURE_NAMES[i] for i in keep] + ["vinyl_centroid_radius"]
    trajectories, summary = {}, {}
    for origin, job in JOBS.items():
        coords, box, pull = extract(job, origin)
        distance, radius = fingerprint(coords, box)
        x_idx, y_idx = FEATURE_NAMES.index("V3-V12"), FEATURE_NAMES.index("V3-V10")
        ex = float(np.max(np.abs(distance[:, x_idx] - pull[:, 1])))
        ey = float(np.max(np.abs(distance[:, y_idx] - pull[:, 2])))
        if max(ex, ey) > 0.005:
            raise ValueError(f"pull/geometry mismatch {job}: {ex}, {ey} nm")
        raw = np.column_stack((distance[:, keep], radius))
        z = (raw - mean) / scale
        pc = z @ pc_axis
        time = coords[:, 0]
        centers, block_raw, block_z, block_pc, block_pull = [], [], [], [], []
        for left in np.arange(0, END_PS, BLOCK_PS):
            mask = (time >= left) & (time < left + BLOCK_PS)
            centers.append(left + BLOCK_PS / 2)
            block_raw.append(raw[mask].mean(axis=0))
            block_z.append(z[mask].mean(axis=0))
            block_pc.append(float(pc[mask].mean()))
            block_pull.append(pull[mask, 1:].mean(axis=0))
        trajectories[origin] = {
            "time": time, "pull": pull[:, 1:], "pc": pc,
            "block_time": np.asarray(centers), "block_raw": np.asarray(block_raw),
            "block_z": np.asarray(block_z), "block_pc": np.asarray(block_pc),
            "block_pull": np.asarray(block_pull),
        }
        summary[origin] = {
            "bohr_job_id": job, "exit_code": 0,
            "production_frames": len(time), "duration_ns": float((time[-1] - time[0]) / 1000),
            "pull_geometry_max_error_nm": {"x": ex, "y": ey},
            "restraint_mean_sd_nm": {
                "V3-V12": [float(pull[:, 1].mean()), float(pull[:, 1].std())],
                "V3-V10": [float(pull[:, 2].mean()), float(pull[:, 2].std())],
            },
            "first_last_1ns_distance_means_nm": {
                name: [float(pull[time < 1000, k].mean()), float(pull[time >= 4000, k].mean())]
                for k, name in ((1, "V3-V12"), (2, "V3-V10"))
            },
            "pc1_first_last_1ns_mean_sd": [
                [float(pc[time < 1000].mean()), float(pc[time < 1000].std())],
                [float(pc[time >= 4000].mean()), float(pc[time >= 4000].std())],
            ],
        }
    a, b = trajectories["A6_origin"], trajectories["A2_origin"]
    block_gap = np.linalg.norm(a["block_z"] - b["block_z"], axis=1)
    pc_gap = a["block_pc"] - b["block_pc"]
    summary["comparison"] = {
        "block_ps": BLOCK_PS,
        "standardized_contact_mean_gap_first_1ns": float(block_gap[:4].mean()),
        "standardized_contact_mean_gap_last_1ns": float(block_gap[-4:].mean()),
        "standardized_contact_mean_gap_all_blocks": block_gap.tolist(),
        "pc1_gap_first_1ns": float(pc_gap[:4].mean()),
        "pc1_gap_last_1ns": float(pc_gap[-4:].mean()),
        "pc1_gap_all_blocks": pc_gap.tolist(),
        "independent_sample_inference": False,
        "interpretation": "Descriptive trajectory comparison; no PMF or rate estimate.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (OUT / "block_means.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["block_center_ps", "origin", "x_nm", "y_nm", "pc1"] + feature_names
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for origin, data in trajectories.items():
            for i, t in enumerate(data["block_time"]):
                writer.writerow({"block_center_ps": t, "origin": origin,
                    "x_nm": data["block_pull"][i, 0], "y_nm": data["block_pull"][i, 1],
                    "pc1": data["block_pc"][i], **dict(zip(feature_names, data["block_raw"][i]))})
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    colors = {"A6_origin": "#147d83", "A2_origin": "#b45309"}
    for origin, data in trajectories.items():
        c = colors[origin]
        axes[0].plot(data["time"], data["pull"][:, 0], color=c, alpha=0.12, lw=0.45)
        axes[0].plot(data["block_time"], data["block_pull"][:, 0], color=c, label=f"{origin}: V3–V12")
        axes[0].plot(data["block_time"], data["block_pull"][:, 1], color=c, ls="--", label=f"{origin}: V3–V10")
        axes[1].plot(data["block_time"], data["block_pc"], color=c, marker="o", ms=3, label=origin)
    axes[0].axhline(1.75, color="black", ls=":", lw=1)
    axes[0].set_ylabel("restrained distances (nm)")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].axhline(pc_centers["A6"], color=colors["A6_origin"], ls=":", lw=1)
    axes[1].axhline(pc_centers["A2"], color=colors["A2_origin"], ls=":", lw=1)
    axes[1].set_ylabel("contact fingerprint PC1\n(250 ps block mean)")
    axes[1].legend(fontsize=8)
    axes[2].plot(a["block_time"], block_gap, color="#553c9a", marker="o", ms=3)
    axes[2].set_ylabel("between-start fingerprint\ngap (standardized Euclidean)")
    axes[2].set_xlabel("follow-up production time (ps)")
    fig.savefig(OUT / "extension_5ns_convergence.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary["comparison"], indent=2))


if __name__ == "__main__":
    main()
