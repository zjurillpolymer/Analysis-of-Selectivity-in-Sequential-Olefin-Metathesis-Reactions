#!/usr/bin/env python3
"""Estimate motion timescales from the three unbiased post-q0 B6 trajectories."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/b6_postq0_unbiased_30ns_20261011"
JOBS = ("20855001", "20855003", "20855005")
LABELS = {"20855001": "seed_05000", "20855003": "seed_12500", "20855005": "seed_19500"}
DT_PS = 2.0


def read_xvg(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text().splitlines():
        if line and not line.startswith(("#", "@")):
            rows.append([float(x) for x in line.split()])
    return np.asarray(rows)


def acf_tau(x: np.ndarray, dt_ps: float) -> tuple[float, float, float]:
    """Return integrated positive ACF, 1/e crossing, and first zero crossing."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    if n < 4 or np.var(x) == 0:
        return 0.0, 0.0, 0.0
    f = np.fft.rfft(np.r_[x, np.zeros_like(x)])
    ac = np.fft.irfft(f * np.conjugate(f))[:n]
    ac /= np.arange(n, 0, -1)
    ac /= ac[0]
    positive = np.where(ac[1:] <= 0)[0]
    stop = int(positive[0] + 1) if len(positive) else min(n - 1, n // 2)
    tau_int = dt_ps * (0.5 + np.sum(ac[1:stop]))
    cross1 = np.where(ac <= np.exp(-1))[0]
    cross0 = np.where(ac <= 0)[0]
    return float(max(0.0, tau_int)), float(cross1[0] * dt_ps if len(cross1) else np.nan), float(cross0[0] * dt_ps if len(cross0) else np.nan)


def episodes(values: np.ndarray, low: float = 0.6, high: float = 0.8) -> tuple[int, float, float, float]:
    """Hysteretic low-contact episodes: enter <= low, exit >= high."""
    state = False
    starts, ends = [], []
    for i, val in enumerate(values):
        if not state and val <= low:
            state = True
            starts.append(i)
        elif state and val >= high:
            state = False
            ends.append(i - 1)
    if state:
        ends.append(len(values) - 1)
    if not starts:
        return 0, 0.0, 0.0, 0.0
    ends = ends[:len(starts)]
    lengths = np.asarray(ends) - np.asarray(starts) + 1
    return int(len(lengths)), float(np.mean(lengths) * DT_PS), float(np.median(lengths) * DT_PS), float(np.max(lengths) * DT_PS)


def main() -> None:
    out = BASE / "timescale_analysis"
    out.mkdir(parents=True, exist_ok=True)
    acf_rows, contact_rows, block_rows, endpoint_rows = [], [], [], []
    traces = {}
    for job in JOBS:
        label = LABELS[job]
        dist = read_xvg(BASE / job / "reaction_distances.xvg")
        rg = read_xvg(BASE / job / "rg.xvg")
        if len(dist) != 15001 or len(rg) != 15001:
            raise RuntimeError(f"{job}: unexpected frame count {len(dist)}, {len(rg)}")
        frame = pd.DataFrame({"time_ps": dist[:, 0], "V2-V11_nm": dist[:, 1], "V2-V3_nm": dist[:, 2], "V2-V4_nm": dist[:, 3], "Rg_nm": rg[:, 1]})
        frame.to_csv(out / f"{label}_observables_2ps.csv.gz", index=False, compression="gzip")
        traces[label] = frame
        for col in frame.columns[1:]:
            tau, cross1, cross0 = acf_tau(frame[col].to_numpy(), DT_PS)
            acf_rows.append({"replica": label, "observable": col, "mean": frame[col].mean(), "sd": frame[col].std(ddof=1), "tau_integrated_ps": tau, "acf_1e_ps": cross1, "acf_first_zero_ps": cross0})
        for col in ("V2-V11_nm", "V2-V3_nm", "V2-V4_nm"):
            vals = frame[col].to_numpy()
            n, mean_res, median_res, max_res = episodes(vals)
            contact_rows.append({"replica": label, "observable": col, "fraction_le_0p6_nm": float(np.mean(vals <= .6)), "fraction_le_0p8_nm": float(np.mean(vals <= .8)), "hysteretic_episode_count": n, "mean_episode_ps": mean_res, "median_episode_ps": median_res, "max_episode_ps": max_res, "minimum_nm": vals.min()})
        for lo, hi in ((0, 5000), (5000, 10000), (10000, 15000), (15000, 20000), (20000, 25000), (25000, 30000)):
            take = (frame.time_ps >= lo) & (frame.time_ps < hi if hi < 30000 else frame.time_ps <= hi)
            row = {"replica": label, "block_ps": f"{lo}-{hi}"}
            for col in frame.columns[1:]:
                row[f"{col}_mean"] = frame.loc[take, col].mean()
                row[f"{col}_sd"] = frame.loc[take, col].std(ddof=1)
            block_rows.append(row)
        for lo, hi in ((0, 5000), (25000, 30000)):
            take = (frame.time_ps >= lo) & (frame.time_ps <= hi)
            endpoint_rows.append({"replica": label, "block_ps": f"{lo}-{hi}", **{f"{c}_mean": frame.loc[take, c].mean() for c in frame.columns[1:]}})
    acf = pd.DataFrame(acf_rows); contact = pd.DataFrame(contact_rows); blocks = pd.DataFrame(block_rows); endpoints = pd.DataFrame(endpoint_rows)
    acf.to_csv(out / "autocorrelation_timescales.csv", index=False)
    contact.to_csv(out / "contact_episode_timescales.csv", index=False)
    blocks.to_csv(out / "5ns_block_means.csv", index=False)
    endpoints.to_csv(out / "early_late_5ns_means.csv", index=False)
    summary = {"dt_ps": DT_PS, "n_frames_each": 15001, "trajectory_ns_each": 30.0, "contact_definition": "hysteretic enter <=0.6 nm, exit >=0.8 nm", "mapping": {"V2": [207, 208], "V11": [205, 206], "V3": [189, 190], "V4": [191, 192]}, "observables": ["V2-V11", "V2-V3", "V2-V4", "Rg"], "notes": ["Distances are computed with a solute-only TPR and -nopbc, so the full solute is not split by box images.", "Contact fractions describe geometric accessibility, not reaction probabilities.", "Three starts are not enough to infer equilibrium basin weights by themselves."]}
    (out / "analysis_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Autocorrelation times (ps):")
    print(acf.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\nContact episodes:")
    print(contact.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
