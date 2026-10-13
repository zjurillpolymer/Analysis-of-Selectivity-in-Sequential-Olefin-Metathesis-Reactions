#!/usr/bin/env python3
"""Audit existing unrestrained B6 post-q0 trajectories for q1 conformational sampling.

The simulations are full-solvent, but their archived XTCs contain the 438-atom
solute only. Intramolecular distances are therefore recomputed with GROMACS
using the full topology to make the solute whole, then ``-nopbc`` to prevent
minimum-image switching between atoms of that same molecule.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = ROOT / (
    "bohrium_download/b6_neighbor_md_extensions_2to10ns_20260918/20750428/"
    "extracted/workspace/gromacs_dcm/runs/B6_p_C6_q1_V1_V2_cis"
)
DEFAULT_REFERENCE = ROOT / (
    "Analysis-of-Selectivity-in-Sequential-Olefin-Metathesis-Reactions/results/"
    "b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001/"
    "joint_mbar/hidden_conformation_diagnostic/matched_frames_and_contacts.csv"
)
DEFAULT_OUTPUT = ROOT / "results/b6_postq0_unbiased_ensemble_audit_20261003"
VINYLS = list(range(3, 13))
ALL_PAIRS = list(itertools.combinations(VINYLS, 2))
REACTIVE_PAIRS = {(3, 10), (3, 12)}
FEATURE_PAIRS = [pair for pair in ALL_PAIRS if pair not in REACTIVE_PAIRS]


def atom_pair(vinyl: int) -> tuple[int, int]:
    # This mapping was verified against the existing q1 midpoint groups:
    # V3 midpoint = GROMACS atoms 189/190; V12 = 207/208.
    start = 189 + 2 * (vinyl - 3)
    return start, start + 1


def selection(pair: tuple[int, int]) -> str:
    a1, a2 = atom_pair(pair[0])
    b1, b2 = atom_pair(pair[1])
    return f"com of (atomnr {a1} or atomnr {a2}) plus com of (atomnr {b1} or atomnr {b2})"


def rmsz_centroid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a.mean(axis=0) - b.mean(axis=0)) ** 2)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stride", type=int, default=20, help="frame stride; default 20 = 10 ps")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    ref = pd.read_csv(args.reference)
    ref = ref.loc[ref["pair"] == 2]
    feature_names = [f"V{a}-V{b}_nm" for a, b in FEATURE_PAIRS]
    mean = ref[feature_names].mean().to_numpy()
    scale = ref[feature_names].std().to_numpy().copy()
    scale[scale == 0] = 1.0

    selection_arg = "; ".join(selection(pair) for pair in ALL_PAIRS)
    raw: dict[str, np.ndarray] = {}
    full_rows = []
    for replica in (1, 2, 3):
        rep = f"replica_{replica:02d}"
        xtc = args.runs / rep / "production_2ns.xtc"  # contains the concatenated 0–10 ns trajectory
        tpr = args.runs / rep / "production_10ns.tpr"
        out_xvg = args.output / f"{rep}_45_pair_distances_whole_no_pbc.xvg"
        cmd = [args.gmx, "distance", "-f", str(xtc), "-s", str(tpr), "-select", selection_arg,
               "-oall", str(out_xvg), "-xvg", "none", "-nopbc"]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        data = np.loadtxt(out_xvg)
        if data.shape[1] != 1 + len(ALL_PAIRS):
            raise RuntimeError(f"{rep}: expected time + 45 distances, got {data.shape}")
        raw[rep] = data
        columns = ["time_ps"] + [f"V{a}-V{b}_nm" for a, b in ALL_PAIRS]
        frame_df = pd.DataFrame(data, columns=columns)
        frame_df.iloc[::args.stride].to_csv(args.output / f"{rep}_features_10ps.csv.gz", index=False,
                                                   compression="gzip")
        full_rows.append({"replica": replica, "n_frames": int(len(data)),
                          "first_time_ps": float(data[0, 0]), "last_time_ps": float(data[-1, 0]),
                          "dt_ps": float(np.median(np.diff(data[:, 0]))),
                          "xtc_atoms": 438, "tpr_atoms": 9219,
                          "pull_enabled": False, "distance_pbc": "whole-solute + no minimum image"})

    clouds: dict[str, np.ndarray] = {}
    blocks: dict[str, list[np.ndarray]] = {}
    stride = args.stride
    for rep, data in raw.items():
        sampled = data[::stride]
        cols = [ALL_PAIRS.index(pair) + 1 for pair in FEATURE_PAIRS]
        z = (sampled[:, cols] - mean) / scale
        clouds[rep] = z
        blocks[rep] = [z[(sampled[:, 0] >= lo) & (sampled[:, 0] < hi)]
                       for lo, hi in [(i * 1000, (i + 1) * 1000) for i in range(10)]]

    pair_rows = []
    reps = list(raw)
    for i, rep_a in enumerate(reps):
        for rep_b in reps[i + 1:]:
            d = cdist(clouds[rep_a], clouds[rep_b]) / np.sqrt(len(FEATURE_PAIRS))
            pair_rows.append({"replica_a": rep_a, "replica_b": rep_b,
                              "a_to_b_nearest_median_rmsz": float(np.median(d.min(axis=1))),
                              "a_to_b_nearest_q10_rmsz": float(np.quantile(d.min(axis=1), .1)),
                              "b_to_a_nearest_median_rmsz": float(np.median(d.min(axis=0))),
                              "b_to_a_nearest_q10_rmsz": float(np.quantile(d.min(axis=0), .1)),
                              "centroid_rmsz": rmsz_centroid(clouds[rep_a], clouds[rep_b])})
    pd.DataFrame(pair_rows).to_csv(args.output / "replica_cloud_overlap.csv", index=False)

    block_rows = []
    contact_rows = []
    for rep, data in raw.items():
        sampled = data[::stride]
        cols = [ALL_PAIRS.index(pair) + 1 for pair in FEATURE_PAIRS]
        z = (sampled[:, cols] - mean) / scale
        for b in range(10):
            lo, hi = 1000 * b, 1000 * (b + 1)
            take = (sampled[:, 0] >= lo) & (sampled[:, 0] < hi)
            block_rows.append({"replica": rep, "block_ps": f"{lo}-{hi}",
                               "n_frames": int(take.sum()),
                               "centroid_rmsz_from_replica_mean": float(np.sqrt(np.mean((z[take].mean(0) - z.mean(0)) ** 2))),
                               "V3_V12_mean_nm": float(np.mean(sampled[take, 1 + ALL_PAIRS.index((3, 12))])),
                               "V3_V10_mean_nm": float(np.mean(sampled[take, 1 + ALL_PAIRS.index((3, 10))])),
                               "V3_V12_contact_lt_0p6nm_fraction": float(np.mean(sampled[take, 1 + ALL_PAIRS.index((3, 12))] < .6)),
                               "V3_V10_contact_lt_0p6nm_fraction": float(np.mean(sampled[take, 1 + ALL_PAIRS.index((3, 10))] < .6))})
        for pair in ((3, 12), (3, 10)):
            values = data[:, 1 + ALL_PAIRS.index(pair)]
            contact = values < .6
            starts = np.flatnonzero(contact & np.r_[True, ~contact[:-1]])
            ends = np.flatnonzero(contact & np.r_[~contact[1:], True])
            contact_rows.append({"replica": rep, "pair": f"V{pair[0]}-V{pair[1]}",
                                 "mean_nm": float(values.mean()), "median_nm": float(np.median(values)),
                                 "q10_nm": float(np.quantile(values, .1)), "minimum_nm": float(values.min()),
                                 "fraction_below_0p6nm": float(contact.mean()),
                                 "episode_count_below_0p6nm": int(len(starts)),
                                 "total_contact_time_ps": float(contact.sum() * .5),
                                 "max_episode_ps": float((ends - starts + 1).max() * .5) if len(starts) else 0.0})
    block_df = pd.DataFrame(block_rows)
    block_df.to_csv(args.output / "replica_1ns_block_diagnostics.csv", index=False)
    contact_df = pd.DataFrame(contact_rows)
    contact_df.to_csv(args.output / "reactive_distance_contact_diagnostics.csv", index=False)

    # Exploratory TICA on the 43 whole-solute features. Fit each lag using
    # independent replica time series (never concatenate across replica ends).
    lag_summaries = []
    sampled_by_rep = {rep: arr[::stride] for rep, arr in raw.items()}
    pooled_scaled = np.vstack([
        (data[:, [ALL_PAIRS.index(pair) + 1 for pair in FEATURE_PAIRS]] - mean) / scale
        for data in sampled_by_rep.values()
    ])
    pooled_feature_mean = pooled_scaled.mean(axis=0)
    for lag_ps in (20.0, 50.0, 100.0):
        lag_steps = int(round(lag_ps / (stride * .5)))
        c0_num = np.zeros((len(FEATURE_PAIRS), len(FEATURE_PAIRS)))
        ct_num = np.zeros_like(c0_num)
        n_pairs = 0
        centered = {}
        for rep, data in sampled_by_rep.items():
            cols = [ALL_PAIRS.index(pair) + 1 for pair in FEATURE_PAIRS]
            x = (data[:, cols] - mean) / scale - pooled_feature_mean
            centered[rep] = x
            x0, xt = x[:-lag_steps], x[lag_steps:]
            c0_num += x0.T @ x0 + xt.T @ xt
            ct_num += x0.T @ xt + xt.T @ x0
            n_pairs += len(x0)
        c0 = c0_num / (2 * n_pairs)
        ct = ct_num / (2 * n_pairs)
        ridge = 1e-7 * float(np.trace(c0) / len(FEATURE_PAIRS))
        eigenvalues, eigenvectors = eigh(ct, c0 + ridge * np.eye(len(FEATURE_PAIRS)))
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        proj_rows = []
        for rep, x in centered.items():
            projected = x @ eigenvectors[:, :3]
            times = sampled_by_rep[rep][:, 0]
            for idx in range(len(times)):
                proj_rows.append({"replica": rep, "time_ps": float(times[idx]),
                                  "tIC1": float(projected[idx, 0]), "tIC2": float(projected[idx, 1]),
                                  "tIC3": float(projected[idx, 2])})
        pd.DataFrame(proj_rows).to_csv(args.output / f"tica_lag_{int(lag_ps)}ps_projection.csv.gz",
                                       index=False, compression="gzip")
        lag_summaries.append({"lag_ps": lag_ps, "lag_steps_10ps_sampled": lag_steps,
                              "eigenvalues_top10": [float(x) for x in eigenvalues[:10]],
                              "ridge": ridge})

    summary = {"system": "q0 product topology / q1 reactant", "state": "B6_p_C6_q1_V1_V2_cis",
               "source_bohrium_job": 20750428, "temperature_K": 313.15,
               "n_replicas": 3, "target_distances_unrestrained": True,
               "xtc_atom_count": 438, "full_system_atom_count": 9219,
               "trajectory_solvent_coordinates_archived": False,
               "features": [f"V{a}-V{b}" for a, b in FEATURE_PAIRS],
               "reactive_distances_monitored_separately": ["V3-V10", "V3-V12"],
               "replicas": full_rows,
               "replica_pair_comparison": pair_rows,
               "block_diagnostics": block_rows,
               "reactive_distance_contact_diagnostics": contact_rows,
               "exploratory_tica": lag_summaries,
               "interpretation_guardrail": "No basin populations inferred until stationarity and reversible transitions are established."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("Replica pair full-fingerprint overlap (RMS-z):")
    print(pd.DataFrame(pair_rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n1-ns block drift/contact summaries:")
    print(block_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nUnbiased reactive-distance contact diagnostics (<0.6 nm; not reaction probabilities):")
    print(contact_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nExploratory TICA top eigenvalues:")
    for item in lag_summaries:
        print(item["lag_ps"], np.round(item["eigenvalues_top10"][:5], 3))
    print(f"\nWrote analysis to {args.output}")


if __name__ == "__main__":
    main()
