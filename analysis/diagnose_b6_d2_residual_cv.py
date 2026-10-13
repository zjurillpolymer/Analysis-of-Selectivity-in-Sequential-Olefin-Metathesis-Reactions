#!/usr/bin/env python3
"""Find residual structural differences after biasing V6-V8 and V8-V10."""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "results/b6_d2_targeted_2d_umbrella_results_20260920/raw"
OUT = PROJECT / "results/b6_d2_targeted_2d_umbrella_analysis_20260920/residual_cv"
JOBS = {20760833: 1, 20760834: 3}
VINYLS = tuple(range(3, 13))
ATOM_PAIRS = {v: (189 + 2 * (v - 3), 190 + 2 * (v - 3)) for v in VINYLS}
EXCLUDED = {(6, 8), (8, 10)}


def xvg(path: Path) -> np.ndarray:
    return np.asarray([[float(x) for x in line.split()] for line in path.read_text().splitlines()
                       if line.strip() and line.lstrip()[0] not in "@#&"], dtype=float)


def minimum_image(v: np.ndarray, box: np.ndarray) -> np.ndarray:
    return v - box[:, None, :] * np.round(v / box[:, None, :])


def extract(gmx: str, window: Path, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    coord, box = cache / "vinyl_coords.xvg", cache / "box.xvg"
    if not coord.exists():
        ndx = cache / "vinyl.ndx"
        ndx.write_text("[ All_vinyls ]\n" + " ".join(str(x) for pair in ATOM_PAIRS.values() for x in pair) + "\n")
        subprocess.run([gmx, "traj", "-f", str(window / "production.xtc"), "-s", str(window / "production.gro"),
                        "-n", str(ndx), "-ox", str(coord), "-ob", str(box), "-x", "-y", "-z"],
                       input="All_vinyls\n", text=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return xvg(coord), xvg(box)


def geometry(coord: np.ndarray, box_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = coord[:, 1:].reshape(-1, len(VINYLS), 2, 3); box = box_rows[:, 1:4]
    bond = minimum_image(xyz[:, :, 1] - xyz[:, :, 0], box)
    center = xyz[:, :, 0] + 0.5 * bond
    dv = center[:, None, :, :] - center[:, :, None, :]
    dv = minimum_image(dv.reshape(len(center), -1, 3), box).reshape(dv.shape)
    dist = np.linalg.norm(dv, axis=-1)
    mask = np.triu(np.ones((len(VINYLS), len(VINYLS)), dtype=bool), 1)
    rg2 = np.mean(dist[:, mask] ** 2, axis=1) * (len(VINYLS) - 1) / (2 * len(VINYLS))
    return dist, np.sqrt(np.maximum(rg2, 0))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--gmx", default="gmx"); parser.add_argument("--output", type=Path, default=OUT); args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    pairs = [p for p in itertools.combinations(VINYLS, 2) if p not in EXCLUDED]
    per_window = []
    for job, replica in JOBS.items():
        path = RAW / str(job) / "pathway"
        windows = list(csv.DictReader((path / "windows.csv").open()))[6:12]
        for row in windows:
            window = path / "windows" / row["window_name"]
            coord, box = extract(args.gmx, window, out / "cache" / str(job) / window.name)
            dist, rg = geometry(coord, box); idx = {v: i for i, v in enumerate(VINYLS)}
            for a, b in pairs:
                values = dist[:, idx[a], idx[b]]
                per_window.append({"job_id": job, "replica": replica, "window": window.name,
                    "d1_center_nm": float(row["d1_center_nm"]), "d2_center_nm": float(row["d2_center_nm"]),
                    "feature": f"V{a}-V{b}", "distance_mean_nm": float(np.mean(values)),
                    "distance_median_nm": float(np.median(values)), "distance_std_nm": float(np.std(values)),
                    "contact_0p8_probability": float(np.mean(values < 0.80))})
            per_window.append({"job_id": job, "replica": replica, "window": window.name,
                "d1_center_nm": float(row["d1_center_nm"]), "d2_center_nm": float(row["d2_center_nm"]),
                "feature": "vinyl_Rg", "distance_mean_nm": float(np.mean(rg)),
                "distance_median_nm": float(np.median(rg)), "distance_std_nm": float(np.std(rg)),
                "contact_0p8_probability": ""})
    with (out / "per_window_feature_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_window[0])); writer.writeheader(); writer.writerows(per_window)
    features = sorted({r["feature"] for r in per_window})
    comparison = []
    for feature in features:
        rows = [r for r in per_window if r["feature"] == feature]
        differences, effects, contact_diffs = [], [], []
        for d2 in sorted({float(r["d2_center_nm"]) for r in rows}):
            a = next(r for r in rows if r["replica"] == 1 and float(r["d2_center_nm"]) == d2)
            b = next(r for r in rows if r["replica"] == 3 and float(r["d2_center_nm"]) == d2)
            diff = float(a["distance_mean_nm"]) - float(b["distance_mean_nm"])
            pooled = np.sqrt(0.5 * (float(a["distance_std_nm"])**2 + float(b["distance_std_nm"])**2))
            differences.append(diff); effects.append(diff / pooled if pooled else np.nan)
            if a["contact_0p8_probability"] != "": contact_diffs.append(float(a["contact_0p8_probability"]) - float(b["contact_0p8_probability"]))
        comparison.append({"feature": feature, "mean_rep1_minus_rep3_nm": float(np.mean(differences)),
                           "mean_absolute_difference_nm": float(np.mean(np.abs(differences))),
                           "max_absolute_standardized_difference": float(np.nanmax(np.abs(effects))),
                           "mean_contact_probability_difference": float(np.mean(contact_diffs)) if contact_diffs else "",
                           "max_absolute_contact_probability_difference": float(np.max(np.abs(contact_diffs))) if contact_diffs else ""})
    comparison.sort(key=lambda r: float(r["mean_absolute_difference_nm"]), reverse=True)
    with (out / "replica_residual_feature_ranking.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)
    print("top residual coordinates")
    for row in comparison[:12]: print(row)


if __name__ == "__main__":
    main()
