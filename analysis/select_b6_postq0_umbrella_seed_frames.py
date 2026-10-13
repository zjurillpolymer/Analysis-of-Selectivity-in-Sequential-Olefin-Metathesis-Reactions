#!/usr/bin/env python3
"""Select decorrelated post-q0 B6 frames for multi-start umbrella tests."""
from __future__ import annotations
import itertools
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/b6_postq0_unbiased_30ns_20261011"
OUT = BASE / "timescale_analysis/umbrella_seed_selection"
JOBS = {"20855001": "seed_05000", "20855003": "seed_12500", "20855005": "seed_19500"}
# Corrected post-q0 mapping from corrected_target_to_hrex_atom_mapping.tsv.
VINYL_ATOMS = {
    "V2": (207, 208), "V3": (189, 190), "V4": (191, 192),
    "V5": (193, 194), "V6": (195, 196), "V7": (197, 198),
    "V8": (199, 200), "V9": (201, 202), "V10": (203, 204), "V11": (205, 206),
}

def read_xvg(path: Path) -> np.ndarray:
    return np.asarray([[float(x) for x in line.split()] for line in path.read_text().splitlines() if line and not line.startswith(("#", "@"))])

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = list(VINYL_ATOMS)
    pairs = list(itertools.combinations(labels, 2))
    selection = "; ".join(f"com of (atomnr {VINYL_ATOMS[a][0]} or atomnr {VINYL_ATOMS[a][1]}) plus com of (atomnr {VINYL_ATOMS[b][0]} or atomnr {VINYL_ATOMS[b][1]})" for a, b in pairs)
    frame_tables = []
    for job, label in JOBS.items():
        job_out = OUT / job
        job_out.mkdir(exist_ok=True)
        tpr = BASE / job / "solute.tpr"
        xtc = BASE.parent.parent / "bohrium_download/b6_postq0_unbiased_30ns_20261011" / job / "extracted/production.xtc"
        xvg = job_out / "vinyl_pair_fingerprint.xvg"
        subprocess.run(["gmx", "distance", "-f", str(xtc), "-s", str(tpr), "-select", selection, "-oall", str(xvg), "-xvg", "none", "-nopbc"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        d = read_xvg(xvg)
        cols = ["time_ps"] + [f"{a}-{b}_nm" for a, b in pairs]
        df = pd.DataFrame(d, columns=cols)
        obs = pd.read_csv(BASE / "timescale_analysis" / f"{label}_observables_2ps.csv.gz")
        df["V2-V11_nm"] = obs["V2-V11_nm"]
        df["V2-V3_nm"] = obs["V2-V3_nm"]
        df["V2-V4_nm"] = obs["V2-V4_nm"]
        df["Rg_nm"] = obs["Rg_nm"]
        # 10 ps sampling is much finer than the slowest observed correlation time.
        df = df.iloc[::5].reset_index(drop=True)
        df.insert(0, "job", job)
        df.insert(1, "replica", label)
        df.to_csv(job_out / "fingerprint_10ps.csv.gz", index=False, compression="gzip")
        frame_tables.append(df)
    all_df = pd.concat(frame_tables, ignore_index=True)
    feature_cols = [c for c in all_df.columns if c.endswith("_nm")]
    # Give the three reaction-relevant distances modest extra weight without
    # allowing them to erase the global fingerprint.
    raw_X = all_df[feature_cols].to_numpy(float)
    X = (raw_X - raw_X.mean(axis=0)) / np.where(raw_X.std(axis=0) == 0, 1.0, raw_X.std(axis=0))
    for c in ("V2-V11_nm", "V2-V3_nm", "V2-V4_nm"):
        X[:, feature_cols.index(c)] *= 1.5
    rng = np.random.default_rng(20261011)
    centers, labels = kmeans2(X, 12, minit="++", iter=100, thresh=1e-6, seed=rng)
    all_df["cluster"] = labels
    all_df["cluster_distance"] = np.linalg.norm(X - centers[labels], axis=1)
    # First choose the best cluster medoid from each replica; then enforce a
    # four-ns separation within each replica and fill any missing clusters.
    medoids = all_df.sort_values("cluster_distance").drop_duplicates(["replica", "cluster"]).copy()
    selected = []
    for replica, group in medoids.groupby("replica", sort=True):
        chosen = []
        for _, row in group.sort_values("cluster_distance").iterrows():
            if all(abs(float(row.time_ps) - float(prev.time_ps)) >= 4000 for prev in chosen):
                chosen.append(row)
        selected.extend(chosen)
    selected_df = pd.DataFrame(selected).sort_values(["replica", "time_ps"]).reset_index(drop=True)
    selected_df.insert(0, "candidate_id", [f"{r.replica}_{int(r.time_ps):06d}ps" for r in selected_df.itertuples()])
    selected_df.to_csv(OUT / "selected_umbrella_seed_frames.csv", index=False)
    extrema_rows = []
    for replica, group in all_df.groupby("replica", sort=True):
        for observable in ("V2-V11_nm", "V2-V3_nm", "V2-V4_nm"):
            row = group.loc[group[observable].idxmin()].copy()
            row["extremum"] = observable
            row["candidate_id"] = f"{replica}_{int(row.time_ps):06d}ps_min_{observable.replace('-', '_')}"
            extrema_rows.append(row)
    pd.DataFrame(extrema_rows).to_csv(OUT / "extrema_umbrella_seed_frames.csv", index=False)
    all_df.to_csv(OUT / "all_fingerprint_frames_10ps.csv.gz", index=False, compression="gzip")
    manifest = {"n_clusters": 12, "n_selected": int(len(selected_df)), "n_extrema_candidates": len(extrema_rows), "minimum_same_replica_separation_ps": 4000, "sample_stride_ps": 10, "features": feature_cols, "vinyl_atom_mapping": VINYL_ATOMS, "selection_note": "Cluster medoids are decorrelated candidates; extrema candidates retain rare near-contact frames. Neither set is an equilibrium basin weight. Full-solvent coordinates must be rebuilt before umbrella MD because production XTC stores solute only."}
    (OUT / "selection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(selected_df[["candidate_id", "replica", "time_ps", "cluster", "V2-V11_nm", "V2-V3_nm", "V2-V4_nm", "Rg_nm"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nSelected {len(selected_df)} candidates; wrote {OUT}")

if __name__ == "__main__":
    main()
