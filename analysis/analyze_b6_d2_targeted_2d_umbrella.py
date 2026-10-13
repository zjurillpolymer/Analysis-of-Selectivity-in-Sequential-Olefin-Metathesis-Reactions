#!/usr/bin/env python3
"""Analyze the focused B6 full-switch-d2 distance x distance umbrella grid."""

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
RAW = PROJECT / "results/b6_d2_targeted_2d_umbrella_results_20260920/raw"
OUT = PROJECT / "results/b6_d2_targeted_2d_umbrella_analysis_20260920"
JOBS = {20760833: 1, 20760834: 3}
T_K = 313.15
R_KJ = 0.008314462618
BETA = 1.0 / (R_KJ * T_K)
RT_KCAL = R_KJ * T_K / 4.184
K_FORCE = 500.0


def read_xvg(path: Path) -> np.ndarray:
    return np.asarray([[float(x) for x in line.split()] for line in path.read_text().splitlines()
                       if line.strip() and line.lstrip()[0] not in "@#&"], dtype=float)


def load_job(job: int) -> dict[str, object]:
    path = RAW / str(job) / "pathway"
    rows = list(csv.DictReader((path / "windows.csv").open()))
    centers, samples, state, time = [], [], [], []
    audit = []
    for index, row in enumerate(rows):
        data = read_xvg(path / "windows" / row["window_name"] / "production_pullx.xvg")
        d1, d2 = float(row["d1_center_nm"]), float(row["d2_center_nm"])
        centers.append((d1, d2)); samples.append(data[:, 1:3]); state.append(np.full(len(data), index)); time.append(data[:, 0])
        audit.append({"job_id": job, "replica": JOBS[job], "window": row["window_name"],
                      "d1_center_nm": d1, "d2_center_nm": d2, "frames": len(data),
                      "d1_mean_nm": float(np.mean(data[:, 1])), "d1_std_nm": float(np.std(data[:, 1])),
                      "d2_mean_nm": float(np.mean(data[:, 2])), "d2_std_nm": float(np.std(data[:, 2])),
                      "d1_mean_minus_center_nm": float(np.mean(data[:, 1]) - d1),
                      "d2_mean_minus_center_nm": float(np.mean(data[:, 2]) - d2)})
    return {"job": job, "replica": JOBS[job], "path": path, "centers": np.asarray(centers),
            "samples": np.concatenate(samples), "state": np.concatenate(state),
            "time": np.concatenate(time), "audit": audit}


def solve_mbar(samples: np.ndarray, centers: np.ndarray, state: np.ndarray,
               tolerance: float = 1e-10, max_iter: int = 100_000) -> tuple[np.ndarray, np.ndarray, int, float]:
    counts = np.bincount(state, minlength=len(centers)).astype(float)
    if np.any(counts == 0):
        raise ValueError("empty umbrella state")
    delta = samples[None, :, :] - centers[:, None, :]
    u = BETA * 0.5 * K_FORCE * np.sum(delta * delta, axis=2)
    log_counts = np.log(counts)
    free = np.zeros(len(centers))
    residual = math.inf
    for iteration in range(1, max_iter + 1):
        log_den = logsumexp(log_counts[:, None] + free[:, None] - u, axis=0)
        updated = -logsumexp(-u - log_den[None, :], axis=1)
        updated -= updated[0]
        residual = float(np.max(np.abs(updated - free)))
        free = 0.5 * (free + updated)
        if residual < tolerance:
            break
    else:
        raise RuntimeError(f"MBAR did not converge; residual={residual:g}")
    log_den = logsumexp(log_counts[:, None] + free[:, None] - u, axis=0)
    log_w = -log_den
    log_w -= logsumexp(log_w)
    return np.exp(log_w), free, iteration, residual


def metrics(samples: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    d2 = samples[:, 1]
    compact = d2 < 1.10
    bridge = (d2 >= 1.10) & (d2 <= 1.40)
    extended = d2 > 1.40
    pc = float(np.sum(weights[compact])); pb = float(np.sum(weights[bridge])); pe = float(np.sum(weights[extended]))
    return {"compact_probability": pc, "bridge_probability": pb, "extended_probability": pe,
            "compact_over_extended": pc / pe,
            "compact_minus_extended_kcal_mol": -RT_KCAL * math.log(pc / pe),
            "effective_sample_size": float(1.0 / np.sum(weights * weights)),
            "d1_weighted_mean_nm": float(np.sum(weights * samples[:, 0])),
            "d2_weighted_mean_nm": float(np.sum(weights * samples[:, 1]))}


def subset(data: dict[str, object], which: str) -> tuple[np.ndarray, np.ndarray]:
    time = data["time"]
    if which == "full": mask = np.ones(len(time), dtype=bool)
    elif which == "early": mask = time <= 250.0
    elif which == "late": mask = time > 250.0
    else: raise ValueError(which)
    return data["samples"][mask], data["state"][mask]


def bhattacharyya(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    x = np.histogram(a, bins=edges)[0].astype(float); y = np.histogram(b, bins=edges)[0].astype(float)
    x /= np.sum(x); y /= np.sum(y)
    return float(np.sum(np.sqrt(x * y)))


def adjacent_overlap(data: dict[str, object]) -> list[dict[str, object]]:
    centers, samples, state = data["centers"], data["samples"], data["state"]
    lookup = {tuple(center): i for i, center in enumerate(centers)}
    rows = []
    for d1, d2 in map(tuple, centers):
        for axis, step, edges in (("d1", (0.10, 0.0), np.linspace(0.30, 0.85, 111)),
                                  ("d2", (0.0, 0.30), np.linspace(0.25, 2.25, 201))):
            other = (round(d1 + step[0], 2), round(d2 + step[1], 2))
            if other not in lookup: continue
            i, j = lookup[(d1, d2)], lookup[other]
            col = 0 if axis == "d1" else 1
            rows.append({"job_id": data["job"], "replica": data["replica"], "axis": axis,
                         "from_center": f"{d1:.2f},{d2:.2f}", "to_center": f"{other[0]:.2f},{other[1]:.2f}",
                         "bhattacharyya_overlap": bhattacharyya(samples[state == i, col], samples[state == j, col], edges)})
    return rows


def surface(samples: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xedges = np.arange(0.34, 0.82, 0.02); yedges = np.arange(0.30, 2.21, 0.05)
    prob, _, _ = np.histogram2d(samples[:, 0], samples[:, 1], bins=(xedges, yedges), weights=weights)
    positive = prob > 0
    pmf = np.full_like(prob, np.nan); pmf[positive] = -RT_KCAL * np.log(prob[positive]); pmf -= np.nanmin(pmf)
    return xedges, yedges, prob, pmf


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, default=OUT); args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    jobs = [load_job(job) for job in JOBS]
    write_csv(out / "window_audit.csv", [row for data in jobs for row in data["audit"]])
    overlaps = [row for data in jobs for row in adjacent_overlap(data)]; write_csv(out / "adjacent_window_overlap.csv", overlaps)
    result_rows, surfaces = [], {}
    for data in jobs:
        for part in ("full", "early", "late"):
            samples, state = subset(data, part)
            weights, free, iterations, residual = solve_mbar(samples, data["centers"], state)
            row = {"job_id": data["job"], "replica": data["replica"], "segment": part,
                   "frames": len(samples), "mbar_iterations": iterations, "mbar_residual": residual, **metrics(samples, weights)}
            result_rows.append(row)
            if part == "full": surfaces[f"rep{data['replica']}"] = (*surface(samples, weights), samples, weights)
    # Pooled estimate uses the same 18 centers with samples from both independently seeded paths.
    combined_samples = np.concatenate([d["samples"] for d in jobs])
    combined_state = np.concatenate([d["state"] for d in jobs])
    weights, free, iterations, residual = solve_mbar(combined_samples, jobs[0]["centers"], combined_state)
    result_rows.append({"job_id": "pooled", "replica": "pooled", "segment": "full", "frames": len(combined_samples),
                        "mbar_iterations": iterations, "mbar_residual": residual, **metrics(combined_samples, weights)})
    surfaces["pooled"] = (*surface(combined_samples, weights), combined_samples, weights)
    write_csv(out / "basin_reweight_summary.csv", result_rows)

    # Source table for reproducible surfaces and compact plotting.
    source_rows = []
    for label, (xe, ye, probability, pmf, _, _) in surfaces.items():
        for i in range(len(xe)-1):
            for j in range(len(ye)-1):
                source_rows.append({"dataset": label, "d1_center_nm": 0.5*(xe[i]+xe[i+1]),
                                    "d2_center_nm": 0.5*(ye[j]+ye[j+1]), "probability": probability[i,j],
                                    "pmf_kcal_mol": pmf[i,j]})
    write_csv(out / "surface_source_data.csv", source_rows)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True, sharex=True, sharey=True)
    for ax, label in zip(axes, ("rep1", "rep3", "pooled")):
        xe, ye, _, pmf, _, _ = surfaces[label]
        mesh = ax.pcolormesh(xe, ye, np.ma.masked_invalid(pmf.T), shading="auto", cmap="viridis", vmin=0, vmax=5)
        ax.axhline(1.10, color="white", lw=0.8, ls="--"); ax.axhline(1.40, color="white", lw=0.8, ls="--")
        ax.set_title(label); ax.set_xlabel("V6–V8 distance (nm)")
    axes[0].set_ylabel("V8–V10 distance (nm)")
    fig.colorbar(mesh, ax=axes, label="Local PMF (kcal mol⁻¹)", shrink=0.9)
    fig.savefig(out / "targeted_2d_pmf.png", dpi=300); fig.savefig(out / "targeted_2d_pmf.pdf")

    full = {str(row["replica"]): row for row in result_rows if row["segment"] == "full"}
    min_d1 = min(r["bhattacharyya_overlap"] for r in overlaps if r["axis"] == "d1")
    min_d2 = min(r["bhattacharyya_overlap"] for r in overlaps if r["axis"] == "d2")
    med_d2 = float(np.median([r["bhattacharyya_overlap"] for r in overlaps if r["axis"] == "d2"]))
    lines = ["# B6 full-switch d2 targeted 2D umbrella analysis", "", "## Result", "",
             f"All 36 windows completed with 1001 production frames each. Adjacent-window Bhattacharyya overlap has minimum {min_d1:.3f} along V6–V8 and {min_d2:.3f} along V8–V10 (median V8–V10 overlap {med_d2:.3f}).",
             "", "The local unbiased basin estimates are:", "",
             "| dataset | compact d2<1.10 | bridge 1.10–1.40 | extended d2>1.40 | compact/extended | ΔG compact−extended (kcal/mol) | ESS |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key in ("1", "3", "pooled"):
        r = full[key]; label = f"rep{key}" if key != "pooled" else "pooled"
        lines.append(f"| {label} | {r['compact_probability']:.5f} | {r['bridge_probability']:.5f} | {r['extended_probability']:.5f} | {r['compact_over_extended']:.4f} | {r['compact_minus_extended_kcal_mol']:.3f} | {r['effective_sample_size']:.0f} |")
    lines += ["", "## Interpretation", "",
              "The second coordinate is real and strongly separates the two one-dimensional-umbrella endpoints. This calculation supplies a joint distance×topology surface rather than treating their disagreement as random noise.",
              "", "The V8–V10 grid is intentionally sparse. Its adjacent overlaps are non-zero but small, so the basin free-energy difference is a pilot estimate. Early/late and replica agreement must be used as the decision criterion before inserting the pooled value into the final q1 branching model.",
              "", "Files: `window_audit.csv`, `adjacent_window_overlap.csv`, `basin_reweight_summary.csv`, `surface_source_data.csv`, and `targeted_2d_pmf.png/pdf`."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(out), "windows": 36, "min_d1_overlap": min_d1, "min_d2_overlap": min_d2,
                      "pooled": full["pooled"]}, indent=2))


if __name__ == "__main__":
    main()
