#!/usr/bin/env python3
"""Descriptive full-fingerprint comparison of B6 q1 candidate NPT trajectories.

Distances are standardized by the matched q1 umbrella reference distribution.
The output is a structural similarity diagnostic only; it does not estimate
equilibrium basin populations (the candidate NPT runs had reactive-distance
restraints).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = ROOT / (
    "Analysis-of-Selectivity-in-Sequential-Olefin-Metathesis-Reactions/results/"
    "b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001/"
    "joint_mbar/hidden_conformation_diagnostic/matched_frames_and_contacts.csv"
)
DEFAULT_INPUT = ROOT / "b6_q1_candidate_equilibration_results_20261003/analysis_basin_persistence"
DEFAULT_OUTPUT = ROOT / "results/b6_q1_candidate_fingerprint_clusters_20261003"
CANDIDATES = {
    "A6_00690": "A6_origin_00690ps_high_novelty_positive_pc1",
    "A6_01270": "A6_origin_01270ps_high_novelty_crossing_frame",
    "A2_05480": "A2_origin_05480ps_positive_pc1_return",
    "A2_09090": "A2_origin_09090ps_high_novelty_negative_pc1",
}


def rmsz_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return cdist(a, b, metric="euclidean") / np.sqrt(a.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(args.reference)
    ref = ref.loc[ref["pair"] == 2]
    features = [c for c in ref.columns if re.fullmatch(r"V\d+-V\d+_nm", c)
                and c not in {"V3-V12_nm", "V3-V10_nm"}]
    mean = ref[features].mean().to_numpy()
    scale = ref[features].std().to_numpy().copy()
    scale[scale == 0] = 1.0

    clouds: dict[str, np.ndarray] = {}
    blocks: dict[str, list[np.ndarray]] = {}
    trajectories: dict[str, np.ndarray] = {}
    for short, candidate in CANDIDATES.items():
        no_pbc = args.output / f"{short}_43_pair_distances_no_pbc.xvg"
        path = no_pbc if no_pbc.exists() else args.input / f"{candidate}_43_pair_distances.xvg"
        data = np.loadtxt(path)
        trajectories[short] = data
        sampled = data[::20]  # 10-ps spacing; reduce frame autocorrelation
        z = (sampled[:, 1:] - mean) / scale
        clouds[short] = z
        blocks[short] = [z[(sampled[:, 0] >= lo) & (sampled[:, 0] < hi)]
                         for lo, hi in ((0, 250), (250, 500), (500, 750), (750, 1000))]

    pair_rows = []
    for i, a in enumerate(CANDIDATES):
        for b in list(CANDIDATES)[i + 1:]:
            d = rmsz_distance(clouds[a], clouds[b])
            ab, ba = d.min(axis=1), d.min(axis=0)
            pair_rows.append({
                "candidate_a": a, "candidate_b": b,
                "a_to_b_nearest_median_rmsz": float(np.median(ab)),
                "a_to_b_nearest_q10_rmsz": float(np.quantile(ab, .1)),
                "a_to_b_nearest_q90_rmsz": float(np.quantile(ab, .9)),
                "b_to_a_nearest_median_rmsz": float(np.median(ba)),
                "b_to_a_nearest_q10_rmsz": float(np.quantile(ba, .1)),
                "b_to_a_nearest_q90_rmsz": float(np.quantile(ba, .9)),
            })
    pd.DataFrame(pair_rows).to_csv(args.output / "pairwise_cloud_nearest_distances.csv", index=False)

    block_rows = []
    medoid_rows = []
    for name, sets in blocks.items():
        for i in range(len(sets)):
            lo, hi = ((0, 250), (250, 500), (500, 750), (750, 1000))[i]
            candidate = CANDIDATES[name]
            data = trajectories[name]
            sampled = data[::20]
            in_block = (sampled[:, 0] >= lo) & (sampled[:, 0] < hi)
            block_frames = sets[i]
            center = block_frames.mean(axis=0)
            medoid_local = int(np.argmin(np.mean((block_frames - center) ** 2, axis=1)))
            medoid_row = int(np.flatnonzero(in_block)[medoid_local])
            medoid_rows.append({"candidate": name, "block": f"{lo}-{hi}ps",
                                "representative_time_ps": float(sampled[medoid_row, 0]),
                                "nearest_centroid_rmsz": float(np.sqrt(np.mean((block_frames[medoid_local] - center) ** 2)))})
            for j in range(i + 1, len(sets)):
                d = rmsz_distance(sets[i], sets[j])
                block_rows.append({
                    "candidate": name, "block_a": f"{i*250}-{(i+1)*250}ps",
                    "block_b": f"{j*250}-{(j+1)*250}ps",
                    "centroid_rmsz": float(np.sqrt(np.mean((sets[i].mean(0) - sets[j].mean(0))**2))),
                    "a_to_b_nearest_median_rmsz": float(np.median(d.min(axis=1))),
                    "b_to_a_nearest_median_rmsz": float(np.median(d.min(axis=0))),
                })
    pd.DataFrame(block_rows).to_csv(args.output / "within_candidate_block_distances.csv", index=False)
    pd.DataFrame(medoid_rows).to_csv(args.output / "representative_snapshot_times.csv", index=False)

    print(f"features={len(features)}; sampled frames={ {k: len(v) for k, v in clouds.items()} }")
    print(pd.DataFrame(pair_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nWithin-candidate block centroid RMS-z (smaller means more stable block centers):")
    print(pd.DataFrame(block_rows).groupby("candidate")["centroid_rmsz"].agg(["median", "max"]).to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nRepresentative block-centroid frames:")
    print(pd.DataFrame(medoid_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nWrote diagnostics to {args.output}")


if __name__ == "__main__":
    main()
