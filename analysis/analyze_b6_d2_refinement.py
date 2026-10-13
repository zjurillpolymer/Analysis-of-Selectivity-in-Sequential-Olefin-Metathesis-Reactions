#!/usr/bin/env python3
"""Analyze the dense V8-V10 grid plus 1.5 ns continuation of B6 d2 paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp


PROJECT = Path(__file__).resolve().parents[1]
OLD_RAW = PROJECT / "results/b6_d2_targeted_2d_umbrella_results_20260920/raw"
NEW_RAW = PROJECT / "results/b6_d2_refinement_results_20260921/raw"
OUT = PROJECT / "results/b6_d2_refinement_analysis_20260921"
RT_KCAL = 0.008314462618 * 313.15 / 4.184
BETA = 1.0 / (0.008314462618 * 313.15)
K = 500.0
SEEDS = {"compact": (20760833, 20761407, 20761408), "extended": (20760834, 20761409, 20761410)}


def read_xvg(path: Path) -> np.ndarray:
    return np.asarray([[float(v) for v in line.split()] for line in path.read_text().splitlines()
                       if line.strip() and line.lstrip()[0] not in "@#&"], dtype=float)


def rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader((path / "windows.csv").open()))


def load_seed(seed: str) -> dict[str, object]:
    old_job, refine_job, extend_job = SEEDS[seed]
    old = OLD_RAW / str(old_job) / "pathway"; refine = NEW_RAW / str(refine_job) / "pathway"; extend = NEW_RAW / str(extend_job) / "pathway"
    centers: list[tuple[float, float]] = []; samples: list[np.ndarray] = []; states: list[np.ndarray] = []; segments: list[np.ndarray] = []; audit = []
    center_to_state: dict[tuple[float, float], int] = {}

    def add(center: tuple[float, float], data: np.ndarray, segment: str, source: str) -> None:
        if center not in center_to_state:
            center_to_state[center] = len(centers); centers.append(center)
        state = center_to_state[center]; samples.append(data[:, 1:3]); states.append(np.full(len(data), state)); segments.append(np.full(len(data), segment, dtype=object))
        audit.append({"seed": seed, "state": state, "d1_center_nm": center[0], "d2_center_nm": center[1], "segment": segment, "source": source, "frames": len(data),
                      "d1_mean_nm": float(np.mean(data[:, 1])), "d1_std_nm": float(np.std(data[:, 1])), "d2_mean_nm": float(np.mean(data[:, 2])), "d2_std_nm": float(np.std(data[:, 2]))})

    old_rows = rows(old)
    for row in old_rows:
        center = (float(row["d1_center_nm"]), float(row["d2_center_nm"]))
        add(center, read_xvg(old / "windows" / row["window_name"] / "production_pullx.xvg"), "original", f"{old_job}/{row['window_name']}")
        add(center, read_xvg(extend / "windows" / row["window_name"] / "extension_pullx.xvg"), "extension", f"{extend_job}/{row['window_name']}")
    for row in rows(refine):
        center = (float(row["d1_center_nm"]), float(row["d2_center_nm"]))
        add(center, read_xvg(refine / "windows" / row["window_name"] / "production_pullx.xvg"), "refine", f"{refine_job}/{row['window_name']}")
    return {"seed": seed, "centers": np.asarray(centers), "samples": np.concatenate(samples), "state": np.concatenate(states),
            "segments": np.concatenate(segments), "audit": audit}


def solve(samples: np.ndarray, centers: np.ndarray, state: np.ndarray) -> tuple[np.ndarray, int, float]:
    counts = np.bincount(state, minlength=len(centers)).astype(float)
    delta = samples[None, :, :] - centers[:, None, :]
    u = BETA * 0.5 * K * np.sum(delta * delta, axis=2)
    log_n = np.log(counts); f = np.zeros(len(centers)); residual = math.inf
    for iteration in range(1, 100000):
        denom = logsumexp(log_n[:, None] + f[:, None] - u, axis=0)
        new = -logsumexp(-u - denom[None, :], axis=1); new -= new[0]
        residual = float(np.max(np.abs(new - f))); f = 0.5 * (f + new)
        if residual < 1e-8: break
    denom = logsumexp(log_n[:, None] + f[:, None] - u, axis=0); log_w = -denom; log_w -= logsumexp(log_w)
    return np.exp(log_w), iteration, residual


def metrics(samples: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    d2 = samples[:, 1]; compact = d2 < 1.10; bridge = (d2 >= 1.10) & (d2 <= 1.40); extended = d2 > 1.40
    pc, pb, pe = float(weights[compact].sum()), float(weights[bridge].sum()), float(weights[extended].sum())
    return {"compact_probability": pc, "bridge_probability": pb, "extended_probability": pe, "compact_over_extended": pc / pe,
            "compact_minus_extended_kcal_mol": -RT_KCAL * math.log(pc / pe), "effective_sample_size": float(1 / np.sum(weights ** 2))}


def overlap(samples: np.ndarray, state: np.ndarray, centers: np.ndarray) -> list[dict[str, float]]:
    out = []; lookup = {tuple(x): i for i, x in enumerate(centers)}
    for d1, d2 in map(tuple, centers):
        for axis, other, col, edges in [("d1", (round(d1 + .10, 2), d2), 0, np.linspace(.3, .85, 111)),
                                        ("d2", (d1, round(d2 + .15, 2)), 1, np.linspace(.25, 2.25, 201))]:
            if other not in lookup: continue
            i, j = lookup[(d1, d2)], lookup[other]; a = np.histogram(samples[state == i, col], edges)[0].astype(float); b = np.histogram(samples[state == j, col], edges)[0].astype(float); a /= a.sum(); b /= b.sum()
            out.append({"axis": axis, "from_d1_nm": d1, "from_d2_nm": d2, "to_d1_nm": other[0], "to_d2_nm": other[1], "bhattacharyya_overlap": float(np.sum(np.sqrt(a * b)))})
    return out


def write_csv(path: Path, data: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUT); args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    datasets = {seed: load_seed(seed) for seed in SEEDS}
    # Pull coordinates are sampled every 0.5 ps and are strongly correlated.
    # A stride-5 representation preserves each window's time fraction while
    # making the 33-state MBAR solve tractable; ESS is still reported on this
    # conservative decorrelated representation.
    for data in datasets.values():
        data["samples"] = data["samples"][::5]
        data["state"] = data["state"][::5]
        data["segments"] = data["segments"][::5]
    write_csv(out / "state_segment_audit.csv", [x for d in datasets.values() for x in d["audit"]])
    summary = []; surfaces = {}
    for seed, data in datasets.items():
        weights, it, res = solve(data["samples"], data["centers"], data["state"]); m = metrics(data["samples"], weights)
        summary.append({"seed": seed, "states": len(data["centers"]), "frames": len(data["samples"]), "mbar_iterations": it, "mbar_residual": res, **m})
        surfaces[seed] = (data, weights)
    write_csv(out / "basin_reweight_summary.csv", summary)
    # Cross-seed comparison uses the same 33-state grid and is a diagnostic only.
    pooled_samples = np.concatenate([d["samples"] for d in datasets.values()]); pooled_state = np.concatenate([d["state"] for d in datasets.values()]); pooled_centers = datasets["compact"]["centers"]
    pw, pit, pres = solve(pooled_samples, pooled_centers, pooled_state); pm = metrics(pooled_samples, pw)
    write_csv(out / "pooled_summary.csv", [{"states": len(pooled_centers), "frames": len(pooled_samples), "mbar_iterations": pit, "mbar_residual": pres, **pm}])
    overlaps = []
    for seed, data in datasets.items():
        for row in overlap(data["samples"], data["state"], data["centers"]): row["seed"] = seed; overlaps.append(row)
    write_csv(out / "adjacent_overlap.csv", overlaps)
    xedges = np.arange(.34, .82, .02); yedges = np.arange(.30, 2.21, .05)
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.6), constrained_layout=True, sharex=True, sharey=True)
    for ax, seed in zip(axes, ("compact", "extended")):
        data, weights = surfaces[seed]; prob, _, _ = np.histogram2d(data["samples"][:, 0], data["samples"][:, 1], bins=(xedges, yedges), weights=weights); mask = prob > 0; pmf = np.full_like(prob, np.nan); pmf[mask] = -RT_KCAL * np.log(prob[mask]); pmf -= np.nanmin(pmf); mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(pmf.T), shading="auto", cmap="viridis", vmin=0, vmax=5); ax.axhline(1.1, color="white", ls="--", lw=.8); ax.axhline(1.4, color="white", ls="--", lw=.8); ax.set_title(seed); ax.set_xlabel("V6–V8 distance (nm)")
    axes[0].set_ylabel("V8–V10 distance (nm)"); fig.colorbar(mesh, ax=axes, label="Local PMF (kcal mol⁻¹)"); fig.savefig(out / "dense_2d_pmf.png", dpi=300); fig.savefig(out / "dense_2d_pmf.pdf")
    d2 = [float(r["bhattacharyya_overlap"]) for r in overlaps if r["axis"] == "d2"]; d1 = [float(r["bhattacharyya_overlap"]) for r in overlaps if r["axis"] == "d1"]
    lines = ["# B6 d2 dense-grid and continuation analysis", "", "All 4 jobs completed successfully. The refined grid adds five V8–V10 centers per V6–V8 layer; the continuation adds 1.5 ns to each of the original 18 windows from each basin seed.", "", "| seed | states | frames | compact | bridge | extended | compact/extended | ΔG compact−extended (kcal/mol) | ESS |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in summary: lines.append(f"| {r['seed']} | {r['states']} | {r['frames']} | {r['compact_probability']:.5f} | {r['bridge_probability']:.5f} | {r['extended_probability']:.5f} | {r['compact_over_extended']:.4f} | {r['compact_minus_extended_kcal_mol']:.3f} | {r['effective_sample_size']:.0f} |")
    lines += ["", f"Dense-grid adjacent overlap: V6–V8 minimum {min(d1):.3f}; V8–V10 minimum {min(d2):.3f}, median {float(np.median(d2)):.3f}.", "", "The compact and extended seeds are compared on the same 33-state grid. If their basin probabilities remain separated after the 1.5 ns continuation, the discrepancy is not explained by the former coarse d2 spacing and the next method should use a contact-map/path CV rather than another isolated pair distance.", "", "Files: `basin_reweight_summary.csv`, `pooled_summary.csv`, `adjacent_overlap.csv`, `state_segment_audit.csv`, and `dense_2d_pmf.png/pdf`."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(out), "summary": summary, "pooled": pm, "min_d1_overlap": min(d1), "min_d2_overlap": min(d2), "median_d2_overlap": float(np.median(d2))}, indent=2))


if __name__ == "__main__": main()
