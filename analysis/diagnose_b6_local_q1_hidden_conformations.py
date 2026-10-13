#!/usr/bin/env python3
"""Distance-matched structural audit of the six paired B6 q1 umbrella runs.

This is a diagnostic, not another free-energy estimator. It compares early and
late frames from both umbrella families only in x/y cells sampled by all four
groups. Every group contributes the same number of frames from each cell.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from analyze_b6_local_q1_joint_mbar import (
    DOWNLOAD, JOBS, OUTPUT, SINGLE, SOURCE_METADATA, midpoint, minimum_image,
    window_data,
)
from extract_a4_q2_umbrella_qm_candidates import read_numeric_xvg


DEST = OUTPUT / "hidden_conformation_diagnostic"
LOW_NM = 1.5
HIGH_NM = 2.0
CELL_NM = 0.1
MIN_PER_GROUP_CELL = 5
CONTACT_NM = 0.6
VINYLS = tuple(f"V{i}" for i in range(3, 13))
PAIR_IDS = [(i, j) for i in range(10) for j in range(i + 1, 10)]
FEATURE_NAMES = [f"{VINYLS[i]}-{VINYLS[j]}" for i, j in PAIR_IDS]


def extract_active_vinyls(job_id: int, window_index: int, window_name: str) -> tuple[np.ndarray, np.ndarray]:
    pathway = DOWNLOAD / str(job_id) / "extracted/pathway"
    name = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))["pathway"]
    topology = SINGLE / "reactive_pair_topologies" / name / "solute_topology.gro"
    window = pathway / "windows" / window_name
    cache = DEST / "trajectory_cache" / str(job_id) / window_name
    cache.mkdir(parents=True, exist_ok=True)
    mapping = json.loads(SOURCE_METADATA.read_text(encoding="utf-8"))["vinyl_mapping_gromacs_1based"]
    atom_ids = [atom for label in VINYLS for atom in mapping[label]]
    if len(atom_ids) != 20 or len(set(atom_ids)) != 20 or max(atom_ids) > 438:
        raise ValueError("invalid active-vinyl atom mapping")
    index = cache / "active20.ndx"
    if not index.is_file():
        index.write_text("[ Active20 ]\n" + " ".join(map(str, atom_ids)) + "\n", encoding="utf-8")
    coords_path = cache / "active20_coordinates.xvg"
    box_path = cache / "box_vectors.xvg"
    if not (coords_path.is_file() and box_path.is_file()):
        result = subprocess.run([
            "gmx", "traj", "-f", str(window / "production.xtc"), "-s", str(topology),
            "-n", str(index), "-ox", str(coords_path), "-ob", str(box_path),
            "-b", "100", "-e", "500", "-x", "-y", "-z",
        ], input="Active20\n", text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"gmx traj {job_id}/{window_index}:\n{result.stdout[-3000:]}")
    coords = read_numeric_xvg(coords_path)
    box = read_numeric_xvg(box_path)
    if coords.shape != (801, 61) or box.shape != (801, 7):
        raise ValueError(f"bad extracted trajectory shape: {job_id}/{window_index} {coords.shape} {box.shape}")
    if not np.allclose(coords[:, 0], box[:, 0], atol=1e-5) or np.max(np.abs(box[:, 4:])) > 1e-6:
        raise ValueError(f"time/box mismatch: {job_id}/{window_index}")
    return coords, box


def fingerprint(coords: np.ndarray, box_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    atom = coords[:, 1:].reshape(-1, 10, 2, 3)
    box = box_vectors[:, 1:4]
    bond = minimum_image(atom[:, :, 1] - atom[:, :, 0], box[:, None, :])
    length = np.linalg.norm(bond, axis=2)
    if np.any((length < 0.115) | (length > 0.155)):
        raise ValueError("nonphysical vinyl bond in active20 coordinates")
    centers = midpoint(atom[:, :, 0], atom[:, :, 1], box[:, None, :])
    distances = np.column_stack([
        np.linalg.norm(minimum_image(centers[:, j] - centers[:, i], box), axis=1)
        for i, j in PAIR_IDS
    ])
    # PBC-unwrapped ten-vinyl-centroid radius, not whole-molecule Rg.
    unwrapped = centers[:, :1] + minimum_image(centers - centers[:, :1], box[:, None, :])
    radius = np.sqrt(np.mean(np.sum((unwrapped - unwrapped.mean(axis=1, keepdims=True)) ** 2, axis=2), axis=1))
    return distances, radius


def group_name(family: str, time_ps: float) -> str:
    return f"{family}_{'early' if time_ps <= 300 + 1e-6 else 'late'}"


def gather_pair(pair_id: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    candidates: list[dict[str, object]] = []
    extracted_windows = 0
    for family, job_id in JOBS[pair_id].items():
        for row in window_data(pair_id, family, job_id):
            x, y = row["x_nm"], row["y_nm"]
            select = (x >= LOW_NM) & (x < HIGH_NM) & (y >= LOW_NM) & (y < HIGH_NM)
            if not np.any(select):
                continue
            extracted_windows += 1
            pathway = DOWNLOAD / str(job_id) / "extracted/pathway"
            window = sorted((pathway / "windows").glob("window_*"))[row["window_index"]]
            coords, box = extract_active_vinyls(job_id, row["window_index"], window.name)
            if not np.allclose(coords[:, 0], row["time_ps"], atol=1e-5):
                raise ValueError("active20/triad frame time mismatch")
            distances, radius = fingerprint(coords, box)
            ix = FEATURE_NAMES.index("V3-V12")
            iy = FEATURE_NAMES.index("V3-V10")
            if max(np.max(np.abs(distances[:, ix] - x)), np.max(np.abs(distances[:, iy] - y))) > 0.005:
                raise ValueError(f"active20/triad geometry mismatch: {job_id}/{window.name}")
            for k in np.flatnonzero(select):
                cell = (int(np.floor((x[k] - LOW_NM) / CELL_NM)), int(np.floor((y[k] - LOW_NM) / CELL_NM)))
                candidates.append({
                    "pair": pair_id, "family": family, "group": group_name(family, row["time_ps"][k]),
                    "job": job_id, "window": row["window_index"], "frame": int(k),
                    "time_ps": float(row["time_ps"][k]), "x_nm": float(x[k]), "y_nm": float(y[k]),
                    "cell": cell, "distances": distances[k], "vinyl_radius_nm": float(radius[k]),
                })
    return candidates, {"candidate_frames": len(candidates), "extracted_windows": extracted_windows}


def match_cells(candidates: list[dict[str, object]], pair_id: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    groups = ("A6_early", "A6_late", "A2_early", "A2_late")
    by_cell: dict[tuple[int, int], dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in candidates:
        by_cell[row["cell"]][row["group"]].append(row)
    rng = np.random.default_rng(20261001 + pair_id)
    matched: list[dict[str, object]] = []
    kept_cells = []
    for cell in sorted(by_cell):
        count = min(len(by_cell[cell][group]) for group in groups)
        if count < MIN_PER_GROUP_CELL:
            continue
        kept_cells.append({"x_index": cell[0], "y_index": cell[1], "per_group_frames": count})
        for group in groups:
            pool = by_cell[cell][group]
            selected = rng.choice(len(pool), size=count, replace=False)
            matched.extend(pool[int(i)] for i in selected)
    if not matched:
        raise ValueError(f"no matched x/y cells in pair {pair_id}")
    return matched, {"matched_cells": kept_cells, "matched_frames_per_group": sum(c["per_group_frames"] for c in kept_cells)}


def analyze_pair(pair_id: int, rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], dict[str, object]]:
    # Exclude the two reaction-coordinate distances from hidden-variable ranking.
    keep = [i for i, label in enumerate(FEATURE_NAMES) if label not in ("V3-V12", "V3-V10")]
    names = [FEATURE_NAMES[i] for i in keep] + ["vinyl_centroid_radius"]
    data = np.vstack([np.r_[row["distances"][keep], row["vinyl_radius_nm"]] for row in rows])
    groups = np.asarray([row["group"] for row in rows])
    means = data.mean(axis=0)
    scale = data.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (data - means) / scale
    _, _, vh = np.linalg.svd(z - z.mean(axis=0), full_matrices=False)
    pcs = z @ vh[:2].T
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    palette = {"A6_early": "#147d83", "A6_late": "#75c8ca", "A2_early": "#b45309", "A2_late": "#f5b95a"}
    for group, color in palette.items():
        mask = groups == group
        axes[0].scatter(pcs[mask, 0], pcs[mask, 1], s=8, alpha=0.4, color=color, label=group)
    axes[0].set(xlabel="contact-fingerprint PC1", ylabel="PC2", title=f"Pair {pair_id}: matched x/y cells")
    axes[0].legend(fontsize=8, markerscale=2)
    score_rows: list[dict[str, object]] = []
    comparisons = (("A6_early", "A6_late"), ("A2_early", "A2_late"), ("A6_early", "A2_early"), ("A6_late", "A2_late"))
    for feature, name in enumerate(names):
        record: dict[str, object] = {"pair": pair_id, "feature": name}
        for first, second in comparisons:
            a, b = data[groups == first, feature], data[groups == second, feature]
            tag = f"{first}_vs_{second}"
            record[f"{tag}_delta_nm"] = float(b.mean() - a.mean())
            record[f"{tag}_wasserstein_nm"] = float(wasserstein_distance(a, b))
            record[f"{tag}_ks"] = float(ks_2samp(a, b).statistic)
            record[f"{tag}_contact_delta"] = float(np.mean(b < CONTACT_NM) - np.mean(a < CONTACT_NM))
        score_rows.append(record)
    score_rows.sort(key=lambda row: max(row["A6_early_vs_A6_late_wasserstein_nm"], row["A2_early_vs_A2_late_wasserstein_nm"]), reverse=True)
    top = score_rows[0]["feature"]
    top_index = names.index(top)
    right_edge = max(3.0, float(np.ceil(data[:, top_index].max() * 10) / 10))
    for group, color in palette.items():
        axes[1].hist(data[groups == group, top_index], bins=np.linspace(0, right_edge, 61), density=True,
                     histtype="step", lw=1.8, color=color, label=group)
    axes[1].set(xlabel=f"{top} (nm)", ylabel="density", title="Largest early/late distance shift")
    axes[1].legend(fontsize=8)
    fig.savefig(DEST / f"pair_{pair_id:02d}_matched_fingerprint.png", dpi=180)
    plt.close(fig)
    for row, pc in zip(rows, pcs):
        row["pc1"] = float(pc[0])
        row["pc2"] = float(pc[1])
    summary = {
        "top_early_late_feature": top,
        "top_early_late_shift": score_rows[0],
        "pc_group_centroids": {group: pcs[groups == group].mean(axis=0).tolist() for group in palette},
        "comparison_note": "Frame-level KS/Wasserstein values are descriptive; autocorrelated frames are not independent replicates.",
    }
    return score_rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    ranking: list[dict[str, object]] = []
    export: list[dict[str, object]] = []
    summaries = []
    for pair_id in JOBS:
        candidates, counts = gather_pair(pair_id)
        matched, matching = match_cells(candidates, pair_id)
        scores, structure = analyze_pair(pair_id, matched)
        ranking.extend(scores)
        for row in matched:
            export.append({
                "pair": row["pair"], "family": row["family"], "group": row["group"],
                "job": row["job"], "window": row["window"], "frame": row["frame"],
                "time_ps": row["time_ps"], "x_nm": row["x_nm"], "y_nm": row["y_nm"],
                "cell_x": row["cell"][0], "cell_y": row["cell"][1],
                "vinyl_centroid_radius_nm": row["vinyl_radius_nm"], "pc1": row["pc1"], "pc2": row["pc2"],
                **{label + "_nm": float(value) for label, value in zip(FEATURE_NAMES, row["distances"])},
            })
        summary = {"pair": pair_id, "jobs": JOBS[pair_id], **counts, **matching, **structure}
        summaries.append(summary)
        print(f"pair {pair_id}: {counts['extracted_windows']} windows, "
              f"{len(matching['matched_cells'])} cells, {matching['matched_frames_per_group']} frames/group, "
              f"top={structure['top_early_late_feature']}", flush=True)
    write_csv(DEST / "matched_frames_and_contacts.csv", export)
    write_csv(DEST / "feature_shift_ranking.csv", ranking)
    (DEST / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
