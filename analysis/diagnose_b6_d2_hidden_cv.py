#!/usr/bin/env python3
"""Find structural CVs separating the two B6 full-switch-d2 replicas."""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "results/b6_q1_umbrella_extensions_2ns_analysis_20260920/prepared"
JOBS = {20759498: 1, 20759494: 3}
VINYLS = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
ATOM_PAIRS = {
    3: (189, 190), 4: (191, 192), 5: (193, 194), 6: (195, 196),
    7: (197, 198), 8: (199, 200), 9: (201, 202), 10: (203, 204),
    11: (205, 206), 12: (207, 208),
}
SAME_ARM = ((3, 4), (5, 6), (7, 8), (9, 10), (11, 12))
SELECTED = (6, 8)


def read_xvg(path: Path) -> np.ndarray:
    return np.asarray(
        [[float(v) for v in line.split()] for line in path.read_text().splitlines()
         if line.strip() and line.lstrip()[0] not in "#@&"], dtype=float,
    )


def minimum_image(vectors: np.ndarray, box: np.ndarray) -> np.ndarray:
    return vectors - box[:, None, :] * np.round(vectors / box[:, None, :])


def extract_window(gmx: str, window: Path, solute: Path, index: Path, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    coord = cache / "all_vinyl_coords.xvg"
    box = cache / "all_vinyl_box.xvg"
    if not coord.exists() or not box.exists():
        ndx = cache / "all_vinyl.ndx"
        atoms = " ".join(str(atom) for pair in ATOM_PAIRS.values() for atom in pair)
        ndx.write_text(f"[ All_vinyls ]\n{atoms}\n")
        subprocess.run(
            [gmx, "traj", "-f", str(window / "production.xtc"), "-s", str(solute),
             "-n", str(ndx), "-ox", str(coord), "-ob", str(box),
             "-b", "500", "-e", "2000", "-x", "-y", "-z"],
            input="All_vinyls\n", text=True, check=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
    return read_xvg(coord), read_xvg(box)


def features(coords: np.ndarray, box_rows: np.ndarray) -> dict[str, np.ndarray]:
    # coords: time + 20 xyz columns; box_rows: time + box lengths/angles.
    xyz = coords[:, 1:].reshape(-1, len(VINYLS), 2, 3)
    box = box_rows[:, 1:4]
    bond = minimum_image(xyz[:, :, 1] - xyz[:, :, 0], box)
    centers = xyz[:, :, 0] + 0.5 * bond
    pair_vectors = centers[:, None, :, :] - centers[:, :, None, :]
    pair_vectors = minimum_image(pair_vectors.reshape(len(centers), -1, 3), box).reshape(pair_vectors.shape)
    distances = np.linalg.norm(pair_vectors, axis=-1)
    idx = {v: i for i, v in enumerate(VINYLS)}
    hard = distances < 0.80
    mask = np.triu(np.ones((len(VINYLS), len(VINYLS)), dtype=bool), 1)
    mask[idx[6], idx[8]] = False
    mask[idx[8], idx[6]] = False
    remaining = [v for v in VINYLS if v not in SELECTED]
    remaining_idx = [idx[v] for v in remaining]
    remaining_mask = np.zeros_like(mask)
    for i, j in itertools.combinations(remaining_idx, 2):
        remaining_mask[i, j] = True
    same_mask = np.zeros_like(mask)
    for a, b in SAME_ARM:
        i, j = idx[a], idx[b]
        same_mask[min(i, j), max(i, j)] = True
    selected_remaining_mask = np.zeros_like(mask)
    for a in SELECTED:
        for b in remaining:
            i, j = sorted((idx[a], idx[b]))
            selected_remaining_mask[i, j] = True
    all_v = np.arange(len(VINYLS))
    rg2 = np.mean(distances[:, mask] ** 2, axis=1) * (len(VINYLS) - 1) / (2 * len(VINYLS))
    result = {
        "time_ps": coords[:, 0],
        "selected_distance_nm": distances[:, idx[6], idx[8]],
        "remaining_contact_count": np.sum(hard[:, remaining_mask], axis=1),
        "same_arm_contact_count": np.sum(hard[:, same_mask], axis=1),
        "selected_remaining_contact_count": np.sum(hard[:, selected_remaining_mask], axis=1),
        "all_other_contact_count": np.sum(hard[:, mask], axis=1),
        "remaining_min_distance_nm": np.min(
            np.where(np.eye(len(remaining_idx), dtype=bool)[None, :, :], np.inf,
                     distances[:, remaining_idx][:, :, remaining_idx]), axis=(1, 2)),
        "vinyl_midpoint_rg_nm": np.sqrt(np.maximum(rg2, 0.0)),
    }
    # Preserve the identity of each candidate contact.  Aggregate counts can
    # hide whether the replica difference is one specific competing arm pair
    # or a broad compactness effect.
    for a, b in itertools.combinations(VINYLS, 2):
        if (a, b) == SELECTED:
            continue
        result[f"contact_V{a}_V{b}"] = hard[:, idx[a], idx[b]]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/b6_d2_hidden_cv_diagnosis_20260920")
    parser.add_argument("--gmx", default="gmx")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    cache_root = args.output / "trajectory_cache"
    for job, replica in JOBS.items():
        pathway = args.root / str(job) / "pathway"
        topology = next((args.root.parent / "reactive_pair_topologies" / p.name / "solute_topology.gro"
                         for p in [pathway] if (args.root.parent / "reactive_pair_topologies" / p.name / "solute_topology.gro").exists()), None)
        if topology is None:
            # Names are stored in pathway.json and the topology directory uses
            # exactly that pathway name.
            import json
            name = json.loads((pathway / "pathway.json").read_text())["pathway"]
            topology = args.root.parent / "reactive_pair_topologies" / name / "solute_topology.gro"
        index = pathway / "index.ndx"
        for window in sorted((pathway / "windows").glob("window_*")):
            coord, box = extract_window(args.gmx, window, topology, index, cache_root / str(job) / window.name)
            f = features(coord, box)
            pull = read_xvg(window / "production_pullx.xvg")
            if len(pull) != len(coord):
                raise ValueError(f"time mismatch {window}: pull={len(pull)} coords={len(coord)}")
            select = (pull[:, 1] <= 0.75) & (pull[:, 1] >= 0.35)
            for i in np.flatnonzero(select):
                row = {"job_id": job, "replica": replica, "window": window.name,
                       "time_ps": float(pull[i, 0]), "pull_distance_nm": float(pull[i, 1])}
                row.update({key: float(value[i]) for key, value in f.items() if key != "time_ps"})
                rows.append(row)
    with (args.output / "matched_distance_frames.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    feature_names = [
        "remaining_contact_count", "same_arm_contact_count",
        "selected_remaining_contact_count", "all_other_contact_count",
        "remaining_min_distance_nm", "vinyl_midpoint_rg_nm",
    ]
    feature_names.extend(
        f"contact_V{a}_V{b}" for a, b in itertools.combinations(VINYLS, 2)
        if (a, b) != SELECTED
    )
    summary: list[dict[str, object]] = []
    for feature in feature_names:
        for lo, hi in ((0.35, 0.50), (0.50, 0.60), (0.60, 0.75)):
            values = {}
            for replica in (1, 3):
                values[replica] = np.asarray([
                    float(row[feature]) for row in rows
                    if int(row["replica"]) == replica and lo <= float(row["pull_distance_nm"]) < hi
                ])
            a, b = values[1], values[3]
            pooled = np.sqrt(0.5 * (np.var(a) + np.var(b))) if len(a) and len(b) else np.nan
            summary.append({"feature": feature, "distance_bin_nm": f"{lo:.2f}-{hi:.2f}",
                            "rep1_n": len(a), "rep3_n": len(b),
                            "rep1_mean": float(np.mean(a)) if len(a) else np.nan,
                            "rep3_mean": float(np.mean(b)) if len(b) else np.nan,
                            "rep1_median": float(np.median(a)) if len(a) else np.nan,
                            "rep3_median": float(np.median(b)) if len(b) else np.nan,
                            "standardized_difference": float((np.mean(a)-np.mean(b))/pooled) if pooled else np.nan})
    with (args.output / "matched_distance_feature_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    print(f"wrote {len(rows)} matched-distance frames and {len(summary)} summary rows")


if __name__ == "__main__":
    main()
