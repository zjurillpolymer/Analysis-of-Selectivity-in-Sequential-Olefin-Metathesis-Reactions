#!/usr/bin/env python3
"""Audit within-window exchange of distance-matched B6 q1 contact fingerprints.

The projection and class boundary are diagnostic only. They are not reaction
coordinates and their crossing counts are not kinetic rates.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_b6_local_q1_joint_mbar import JOBS, window_data
from diagnose_b6_local_q1_hidden_conformations import (
    DEST, FEATURE_NAMES, HIGH_NM, LOW_NM, fingerprint,
)
from extract_a4_q2_umbrella_qm_candidates import read_numeric_xvg


OUT = DEST / "exchange_audit"
FEATURES = [name + "_nm" for name in FEATURE_NAMES if name not in ("V3-V12", "V3-V10")]
FEATURES.append("vinyl_centroid_radius_nm")


def matched_model(pair_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, float]]:
    path = DEST / "matched_frames_and_contacts.csv"
    with path.open(encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["pair"]) == pair_id]
    values = np.asarray([[float(row[key]) for key in FEATURES] for row in rows])
    groups = np.asarray([row["family"] for row in rows])
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std[std < 1e-8] = 1.0
    standardized = (values - mean) / std
    _, _, axes = np.linalg.svd(standardized, full_matrices=False)
    vector = axes[0]
    pc = standardized @ vector
    if pc[groups == "A6"].mean() < pc[groups == "A2"].mean():
        vector = -vector
        pc = -pc
    family_means = {family: float(pc[groups == family].mean()) for family in ("A6", "A2")}
    threshold = (family_means["A6"] + family_means["A2"]) / 2
    return mean, std, vector, threshold, family_means


def data_for_cached_window(cache: Path, row: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = read_numeric_xvg(cache / "active20_coordinates.xvg")
    box = read_numeric_xvg(cache / "box_vectors.xvg")
    if not np.allclose(coords[:, 0], row["time_ps"], atol=1e-5):
        raise ValueError(f"frame times differ in {cache}")
    distances, radius = fingerprint(coords, box)
    keep = [i for i, label in enumerate(FEATURE_NAMES) if label not in ("V3-V12", "V3-V10")]
    values = np.column_stack((distances[:, keep], radius))
    x, y = row["x_nm"], row["y_nm"]
    shell = (x >= LOW_NM) & (x < HIGH_NM) & (y >= LOW_NM) & (y < HIGH_NM)
    return values, shell, row["time_ps"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_windows = []
    representative = []
    for pair_id, jobs in JOBS.items():
        mean, std, vector, threshold, centroids = matched_model(pair_id)
        shell_points = []
        pair_windows = []
        for family, job_id in jobs.items():
            for row in window_data(pair_id, family, job_id):
                caches = sorted((DEST / "trajectory_cache" / str(job_id)).glob(f"window_{row['window_index']:03d}_*"))
                if not caches:
                    continue
                if len(caches) != 1:
                    raise ValueError(f"ambiguous window {job_id}/{row['window_index']}")
                values, shell, time = data_for_cached_window(caches[0], row)
                pc = ((values - mean) / std) @ vector
                idx = np.flatnonzero(shell)
                if not len(idx):
                    continue
                assigned_a6 = pc[idx] > threshold
                native = assigned_a6 if family == "A6" else ~assigned_a6
                adjacent = np.diff(idx) == 1
                flips = int(np.count_nonzero((np.diff(assigned_a6.astype(int)) != 0) & adjacent))
                # 25-ps blocks with at least five in-shell frames give a much
                # more conservative view than frame-to-frame sign changes.
                block_labels = []
                for left in np.arange(100, 500, 25):
                    block = shell & (time >= left) & (time < left + 25)
                    if np.count_nonzero(block) >= 5:
                        block_labels.append((left, float(np.median(pc[block])) > threshold))
                block_flips = sum(a[1] != b[1] for a, b in zip(block_labels, block_labels[1:]))
                record = {
                    "pair": pair_id, "family": family, "job": job_id,
                    "window": row["window_index"], "center_nm": row["center_nm"],
                    "shell_frames": len(idx), "shell_fraction": float(len(idx) / len(time)),
                    "pc1_mean_in_shell": float(pc[idx].mean()),
                    "pc1_first_half_mean": float(pc[idx[time[idx] <= 300]].mean()) if np.any(time[idx] <= 300) else float("nan"),
                    "pc1_second_half_mean": float(pc[idx[time[idx] > 300]].mean()) if np.any(time[idx] > 300) else float("nan"),
                    "opposite_family_fraction": float(1 - native.mean()),
                    "adjacent_frame_label_flips": flips,
                    "observed_25ps_blocks": len(block_labels),
                    "block_label_flips": block_flips,
                    "start_block_label": "A6-like" if block_labels and block_labels[0][1] else "A2-like" if block_labels else "unobserved",
                    "end_block_label": "A6-like" if block_labels and block_labels[-1][1] else "A2-like" if block_labels else "unobserved",
                }
                pair_windows.append(record)
                for i in idx:
                    shell_points.append((family, int(row["window_index"]), float(time[i]), float(pc[i]),
                                         float(row["x_nm"][i]), float(row["y_nm"][i])))
        if not pair_windows:
            raise ValueError(f"no cached windows for pair {pair_id}")
        all_windows.extend(pair_windows)
        for family in ("A6", "A2"):
            choice = max((row for row in pair_windows if row["family"] == family), key=lambda row: row["shell_frames"])
            representative.append(choice)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
        colors = {"A6": "#147d83", "A2": "#b45309"}
        for ax, choice in zip(axes, representative[-2:]):
            family, window = choice["family"], choice["window"]
            points = [p for p in shell_points if p[0] == family and p[1] == window]
            ax.scatter([p[2] for p in points], [p[3] for p in points], s=10, alpha=0.7, color=colors[family])
            ax.axhline(threshold, color="black", lw=1, ls="--", label="family-centroid midpoint")
            ax.axhline(centroids[family], color=colors[family], lw=1, ls=":")
            ax.set(ylabel="fingerprint PC1", title=f"{family} window {window:03d}, n={len(points)} in matched-distance shell")
            ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("production time (ps)")
        fig.savefig(OUT / f"pair_{pair_id:02d}_representative_time_series.png", dpi=180)
        plt.close(fig)
        report = {
            "pair": pair_id, "family_pc1_means": centroids, "pc1_boundary": threshold,
            "cached_windows": len(pair_windows),
            "windows_with_at_least_50_shell_frames": sum(row["shell_frames"] >= 50 for row in pair_windows),
            "windows_with_both_25ps_block_labels": sum(
                row["block_label_flips"] > 0 for row in pair_windows if row["observed_25ps_blocks"] >= 2
            ),
            "representative_windows": representative[-2:],
        }
        (OUT / f"pair_{pair_id:02d}_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"pair {pair_id}: {report['cached_windows']} windows; "
              f"{report['windows_with_both_25ps_block_labels']} with 25ps label flips", flush=True)
    with (OUT / "window_exchange_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_windows[0]))
        writer.writeheader()
        writer.writerows(all_windows)


if __name__ == "__main__":
    main()
