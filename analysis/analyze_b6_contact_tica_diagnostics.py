#!/usr/bin/env python3
"""Resolve B6 post-q0 contact recurrence, replica stationarity and TICA relation.

Uses the audited whole-solute/no-minimum-image distance matrices at their native
0.5-ps sampling. TICA is fitted on the existing 10-ps subsample (50-ps lag),
then the same linear projection is applied to all native-rate frames so brief
contacts are not lost in the TICA/contact map.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/b6_postq0_unbiased_ensemble_audit_20261003"
REF = ROOT / (
    "Analysis-of-Selectivity-in-Sequential-Olefin-Metathesis-Reactions/results/"
    "b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001/"
    "joint_mbar/hidden_conformation_diagnostic/matched_frames_and_contacts.csv"
)
VINYLS = range(3, 13)
PAIRS = list(itertools.combinations(VINYLS, 2))
FEATURE_PAIRS = [p for p in PAIRS if p not in {(3, 10), (3, 12)}]
CONTACT_CUTOFF_NM = 0.6
SAMPLE_STEP_PS = 10.0
LAG_PS = 50.0


def xvg(rep: int) -> np.ndarray:
    path = OUT / f"replica_{rep:02d}_45_pair_distances_whole_no_pbc.xvg"
    return np.loadtxt(path)


def episodes(time: np.ndarray, contact: np.ndarray, dt: float) -> list[dict]:
    starts = np.flatnonzero(contact & np.r_[True, ~contact[:-1]])
    ends = np.flatnonzero(contact & np.r_[~contact[1:], True])
    rows = []
    for start, end in zip(starts, ends):
        rows.append({
            "start_ps": float(time[start]), "end_ps": float(time[end]),
            "duration_ps": float((end - start + 1) * dt),
            "left_censored": bool(start == 0),
            "right_censored": bool(end == len(time) - 1),
        })
    return rows


def fit_tica_and_project(raw: dict[int, np.ndarray]):
    ref = pd.read_csv(REF)
    ref = ref.loc[ref["pair"] == 2]
    names = [f"V{a}-V{b}_nm" for a, b in FEATURE_PAIRS]
    means = ref[names].mean().to_numpy()
    scales = ref[names].std().to_numpy().copy()
    scales[scales == 0] = 1.0
    cols = [1 + PAIRS.index(pair) for pair in FEATURE_PAIRS]
    sample_stride = round(SAMPLE_STEP_PS / np.median(np.diff(raw[1][:, 0])))
    lag_steps = round(LAG_PS / SAMPLE_STEP_PS)
    sampled = {r: a[::sample_stride] for r, a in raw.items()}
    pooled = np.vstack([(a[:, cols] - means) / scales for a in sampled.values()])
    pooled_mean = pooled.mean(axis=0)
    c0_num = np.zeros((len(cols), len(cols)))
    ct_num = np.zeros_like(c0_num)
    n = 0
    for a in sampled.values():
        z = (a[:, cols] - means) / scales - pooled_mean
        x0, xt = z[:-lag_steps], z[lag_steps:]
        c0_num += x0.T @ x0 + xt.T @ xt
        ct_num += x0.T @ xt + xt.T @ x0
        n += len(x0)
    c0 = c0_num / (2 * n)
    ct = ct_num / (2 * n)
    ridge = 1e-7 * float(np.trace(c0) / len(cols))
    vals, vecs = eigh(ct, c0 + ridge * np.eye(len(cols)))
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    projections = {}
    for r, a in raw.items():
        z = (a[:, cols] - means) / scales - pooled_mean
        projections[r] = z @ vecs[:, :2]
    return vals, projections


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = {r: xvg(r) for r in (1, 2, 3)}
    dt = float(np.median(np.diff(raw[1][:, 0])))
    if any(not np.isclose(np.median(np.diff(a[:, 0])), dt) for a in raw.values()):
        raise RuntimeError("Replica native frame intervals differ")

    event_rows, split_rows, details = [], [], {}
    for r, a in raw.items():
        time = a[:, 0]
        details[r] = {}
        for pair in ((3, 12), (3, 10)):
            d = a[:, 1 + PAIRS.index(pair)]
            c = d < CONTACT_CUTOFF_NM
            eps = episodes(time, c, dt)
            internal = [e for e in eps if not (e["left_censored"] or e["right_censored"])]
            details[r][f"V{pair[0]}-V{pair[1]}"] = eps
            event_rows.append({
                "replica": r, "pair": f"V{pair[0]}-V{pair[1]}",
                "n_frames": len(time), "dt_ps": dt,
                "contact_fraction": float(c.mean()),
                "contact_episodes_total": len(eps),
                "left_censored_episodes": sum(e["left_censored"] for e in eps),
                "right_censored_episodes": sum(e["right_censored"] for e in eps),
                "complete_entries": len(internal), "complete_exits": len(internal),
                "mean_episode_ps_all": float(np.mean([e["duration_ps"] for e in eps])) if eps else 0.0,
                "median_episode_ps_all": float(np.median([e["duration_ps"] for e in eps])) if eps else 0.0,
                "q90_episode_ps_all": float(np.quantile([e["duration_ps"] for e in eps], .9)) if eps else 0.0,
                "max_episode_ps": max((e["duration_ps"] for e in eps), default=0.0),
                "total_contact_ps": float(c.sum() * dt),
            })
            for half, mask in (("first_0_5ns", time < 5000), ("second_5_10ns", time >= 5000)):
                half_contact = c[mask]
                half_starts = np.flatnonzero(half_contact & np.r_[True, ~half_contact[:-1]])
                split_rows.append({
                    "replica": r, "pair": f"V{pair[0]}-V{pair[1]}", "half": half,
                    "n_frames": int(mask.sum()),
                    "contact_fraction": float(half_contact.mean()) if mask.any() else np.nan,
                    "contact_time_ps": float(half_contact.sum() * dt),
                    "episodes_starting_in_half": int(len(half_starts)),
                    "mean_distance_nm": float(d[mask].mean()),
                    "minimum_distance_nm": float(d[mask].min()),
                })
    event_df = pd.DataFrame(event_rows)
    split_df = pd.DataFrame(split_rows)
    event_df.to_csv(OUT / "contact_episode_recurrence.csv", index=False)
    split_df.to_csv(OUT / "contact_half_trajectory_stationarity.csv", index=False)
    (OUT / "contact_episode_timeseries.json").write_text(json.dumps(details, indent=2) + "\n")

    approach_rows = []
    pair_v10_col = 1 + PAIRS.index((3, 10))
    for r, a in raw.items():
        time, dist = a[:, 0], a[:, pair_v10_col]
        for threshold in (0.6, 0.8, 1.0, 1.2, 1.5):
            contact = dist < threshold
            starts = np.flatnonzero(contact & np.r_[True, ~contact[:-1]])
            ends = np.flatnonzero(contact & np.r_[~contact[1:], True])
            approach_rows.append({
                "replica": r, "pair": "V3-V10", "threshold_nm": threshold,
                "fraction": float(contact.mean()), "episodes": int(len(starts)),
                "total_time_ps": float(contact.sum() * dt),
                "max_episode_ps": float((ends - starts + 1).max() * dt) if len(starts) else 0.0,
                "minimum_distance_nm": float(dist.min()),
            })
    pd.DataFrame(approach_rows).to_csv(OUT / "v3_v10_approach_threshold_recurrence.csv", index=False)

    eigvals, proj = fit_tica_and_project(raw)
    tica_rows = []
    for r, a in raw.items():
        for i in range(len(a)):
            tica_rows.append({"replica": r, "time_ps": float(a[i, 0]),
                              "tIC1": float(proj[r][i, 0]), "tIC2": float(proj[r][i, 1]),
                              "V3_V12_nm": float(a[i, 1 + PAIRS.index((3, 12))]),
                              "V3_V10_nm": float(a[i, 1 + PAIRS.index((3, 10))]),
                              "V3_V12_contact": bool(a[i, 1 + PAIRS.index((3, 12))] < CONTACT_CUTOFF_NM),
                              "V3_V10_contact": bool(a[i, 1 + PAIRS.index((3, 10))] < CONTACT_CUTOFF_NM)})
    tica_df = pd.DataFrame(tica_rows)
    tica_df.to_csv(OUT / "tica50ps_native_rate_contact_projection.csv.gz", index=False, compression="gzip")

    # Occupancy landscapes and per-cell distance minima/means. These are descriptive
    # projections, not unbiased free energies because stationarity is unestablished.
    bins = 35
    xedges = np.linspace(tica_df.tIC1.quantile(.005), tica_df.tIC1.quantile(.995), bins + 1)
    yedges = np.linspace(tica_df.tIC2.quantile(.005), tica_df.tIC2.quantile(.995), bins + 1)
    landscape_rows = []
    for pair in ("V3_V12", "V3_V10"):
        values = f"{pair}_nm"
        ids_x = np.digitize(tica_df.tIC1, xedges) - 1
        ids_y = np.digitize(tica_df.tIC2, yedges) - 1
        work = tica_df.assign(ix=ids_x, iy=ids_y)
        for (ix, iy), g in work.groupby(["ix", "iy"]):
            if 0 <= ix < bins and 0 <= iy < bins:
                landscape_rows.append({"pair": pair, "ix": int(ix), "iy": int(iy),
                                       "n_frames": len(g), "mean_distance_nm": float(g[values].mean()),
                                       "minimum_distance_nm": float(g[values].min()),
                                       "contact_fraction": float(g[f"{pair}_contact"].mean()),
                                       "replicas_present": int(g.replica.nunique())})
    landscape = pd.DataFrame(landscape_rows)
    landscape.to_csv(OUT / "tica50ps_contact_landscape_bins.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    for ax, pair, label in zip(axes, ("V3_V12", "V3_V10"), ("V3–V12", "V3–V10")):
        sc = ax.scatter(tica_df.tIC1, tica_df.tIC2, c=tica_df[f"{pair}_nm"], s=2,
                        alpha=.35, cmap=cmap, vmin=0.3, vmax=3.0, rasterized=True)
        contacts = tica_df[tica_df[f"{pair}_contact"]]
        if len(contacts):
            ax.scatter(contacts.tIC1, contacts.tIC2, s=7, facecolors="none", edgecolors="red",
                       linewidths=.45, label="d < 0.6 nm")
            ax.legend(loc="best", frameon=True, fontsize=8)
        ax.set(title=f"{label} distance over TICA projection (50-ps lag)", xlabel="tIC1", ylabel="tIC2")
        fig.colorbar(sc, ax=ax, label=f"{label} distance (nm)")
    fig.savefig(OUT / "tica50ps_reactive_distances.png", dpi=220)
    plt.close(fig)

    # Half-trajectory plot makes replica-specific drift visible.
    order = ["first_0_5ns", "second_5_10ns"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    colors = {"V3-V12": "#1565c0", "V3-V10": "#c62828"}
    x = np.arange(3)
    offsets = {"first_0_5ns": -.12, "second_5_10ns": .12}
    for pair in ("V3-V12", "V3-V10"):
        sub = split_df[split_df.pair == pair]
        for half in order:
            y = [float(sub[(sub.replica == r) & (sub.half == half)].contact_fraction.iloc[0]) for r in (1,2,3)]
            ax.scatter(x + offsets[half], y, label=f"{pair}, {half.replace('_',' ')}", color=colors[pair],
                       marker="o" if half == order[0] else "s", s=65,
                       facecolors=colors[pair] if half == order[0] else "none", linewidths=1.5)
    ax.set(xticks=x, xticklabels=["replica 1", "replica 2", "replica 3"], ylabel="fraction of frames with d < 0.6 nm",
           title="Contact fraction by replica and trajectory half")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=.25)
    fig.savefig(OUT / "contact_fraction_first_vs_second_half.png", dpi=220)
    plt.close(fig)

    summary = {
        "native_frame_dt_ps": dt, "contact_cutoff_nm": CONTACT_CUTOFF_NM,
        "episodes": event_rows, "half_trajectory": split_rows,
        "tica_fit": {"lag_ps": LAG_PS, "fit_stride_ps": SAMPLE_STEP_PS,
                     "top_eigenvalues": [float(v) for v in eigvals[:5]],
                     "note": "Descriptive projected occupancy only; not equilibrium free energy."},
        "v3_v10_interpretation_guardrail": "No <0.6 nm contact observed; finite trajectories cannot prove inaccessibility.",
    }
    (OUT / "contact_tica_diagnostic_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(event_df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nHalf trajectory:")
    print(split_df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nTICA eigenvalues:", np.round(eigvals[:5], 4))
    print(f"Wrote diagnostics to {OUT}")


if __name__ == "__main__":
    main()
