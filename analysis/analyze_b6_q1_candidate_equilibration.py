#!/usr/bin/env python3
"""Check whether selected B6 contact-PC1 candidate fingerprints persist in NPT.

This is a restrained, 1-ns basin-persistence diagnostic, not an equilibrium
population or free-energy estimator. It reproduces the 43-distance PC1 used
for WT-MetaD and compares each NPT frame to the umbrella reference cloud.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = ROOT / "b6_q1_candidate_equilibration_results_20261003"
DEFAULT_PACKAGE = ROOT / "b6_q1_contact_pc1_wtmetad_server_20261002"
DEFAULT_REFERENCE = ROOT / (
    "Analysis-of-Selectivity-in-Sequential-Olefin-Metathesis-Reactions/results/"
    "b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001/"
    "joint_mbar/hidden_conformation_diagnostic/matched_frames_and_contacts.csv"
)
CANDIDATES = (
    ("A6_origin_00690ps_high_novelty_positive_pc1", "npt", "A6_origin", 690),
    ("A6_origin_01270ps_high_novelty_crossing_frame", "npt_full", "A6_origin", 1270),
    ("A2_origin_05480ps_positive_pc1_return", "npt_full", "A2_origin", 5480),
    ("A2_origin_09090ps_high_novelty_negative_pc1", "npt_full", "A2_origin", 9090),
)


def parse_features(cvdef: dict) -> list[tuple[int, int, str]]:
    parsed = []
    for name in cvdef["features"]:
        match = re.fullmatch(r"V(\d+)-V(\d+)_nm", name)
        if not match:
            raise ValueError(f"Unrecognized CV feature: {name}")
        parsed.append((int(match.group(1)), int(match.group(2)), name))
    return parsed


def selection_for_pair(a: int, b: int) -> str:
    # Vinyl-center atoms 3..12 occupy atom numbers 189..208 in this topology.
    aa, ab = 189 + 2 * (a - 3), 190 + 2 * (a - 3)
    ba, bb = 189 + 2 * (b - 3), 190 + 2 * (b - 3)
    return f"com of (atomnr {aa} or atomnr {ab}) plus com of (atomnr {ba} or atomnr {bb})"


def load_xvg(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments=("#", "@"))


def extract(gmx: str, result_root: Path, candidate: str, prefix: str,
            features: list[tuple[int, int, str]], outdir: Path) -> np.ndarray:
    selections = "; ".join(selection_for_pair(a, b) for a, b, _ in features)
    output = outdir / f"{candidate}_43_pair_distances.xvg"
    cmd = [gmx, "distance", "-f", str(result_root / candidate / f"{prefix}.xtc"),
           "-s", str(result_root / candidate / f"{prefix}.tpr"),
           "-select", selections, "-oall", str(output), "-xvg", "none"]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    data = load_xvg(output)
    if data.ndim != 2 or data.shape[1] != len(features) + 1:
        raise RuntimeError(f"Expected time + {len(features)} CV distances; found {data.shape}")
    return data


def heldout_reference_baseline(zref: np.ndarray, windows: np.ndarray) -> tuple[float, float]:
    nearest = np.empty(len(zref), dtype=float)
    for start in range(0, len(zref), 50):
        stop = min(start + 50, len(zref))
        delta = zref[start:stop, None, :] - zref[None, :, :]
        d2 = np.mean(delta * delta, axis=2)
        same_window = windows[start:stop, None] == windows[None, :]
        d2[same_window] = np.inf
        nearest[start:stop] = np.sqrt(np.min(d2, axis=1))
    return float(np.quantile(nearest, .95)), float(np.quantile(nearest, .99))


def nearest_reference(x: np.ndarray, ref_z: np.ndarray, mean: np.ndarray,
                      sd: np.ndarray, chunk: int = 100) -> np.ndarray:
    z = (x - mean) / sd
    out = np.empty(len(z), dtype=float)
    for start in range(0, len(z), chunk):
        delta = z[start:start + chunk, None, :] - ref_z[None, :, :]
        d2 = np.mean(delta * delta, axis=2)
        out[start:start + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def nearest_between_clouds(za: np.ndarray, zb: np.ndarray, chunk: int = 30) -> np.ndarray:
    out = np.empty(len(za), dtype=float)
    for start in range(0, len(za), chunk):
        delta = za[start:start + chunk, None, :] - zb[None, :, :]
        d2 = np.mean(delta * delta, axis=2)
        out[start:start + chunk] = np.sqrt(np.min(d2, axis=1))
    return out


def interval_stats(values: np.ndarray, times: np.ndarray, edges: list[tuple[float, float]]) -> list[dict]:
    stats = []
    for lo, hi in edges:
        mask = (times >= lo) & ((times < hi) if hi < times[-1] else (times <= hi))
        vals = values[mask]
        stats.append({"interval_ps": f"{lo:g}-{hi:g}", "n_frames": int(mask.sum()),
                      "mean": float(vals.mean()), "sd": float(vals.std()),
                      "q05": float(np.quantile(vals, .05)), "q95": float(np.quantile(vals, .95))})
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmx", default="gmx")
    ap.add_argument("--result-root", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--metad-result-root", type=Path, default=ROOT / "b6_q1_contact_pc1_wtmetad_results_20261002")
    args = ap.parse_args()

    outdir = args.result_root / "analysis_basin_persistence"
    outdir.mkdir(parents=True, exist_ok=True)
    cvdef = json.loads((args.package_root / "cv_definition.json").read_text())
    features = parse_features(cvdef)
    plumed = (args.package_root / "A6_origin" / "plumed.dat").read_text()
    cvline = next(line for line in plumed.splitlines() if line.startswith("pc1: COMBINE"))
    coeff_text = re.search(r"COEFFICIENTS=([^ ]+)", cvline).group(1)
    coeff = np.array([float(v) for v in coeff_text.split(",")])
    if len(coeff) != len(features):
        raise ValueError("PLUMED coefficient count does not match the 43 CV features")

    ref = pd.read_csv(args.reference)
    ref = ref.loc[ref["pair"] == 2].copy()
    names = [feature[2] for feature in features]
    ref_x = ref[names].to_numpy(dtype=float)
    ref_mean = ref_x.mean(axis=0)
    ref_sd = ref_x.std(axis=0)
    ref_sd[ref_sd == 0] = 1.0
    ref_z = (ref_x - ref_mean) / ref_sd
    ref_pc = ref_x @ coeff
    ref_summary = {}
    for family in ("A6", "A2"):
        vals = ref_pc[ref["family"].to_numpy() == family]
        ref_summary[family] = {"n": int(len(vals)), "mean": float(vals.mean()),
                               "sd": float(vals.std()), "q05": float(np.quantile(vals, .05)),
                               "q95": float(np.quantile(vals, .95))}
    q95, q99 = heldout_reference_baseline(ref_z, ref["window"].to_numpy())
    metad_z = {}
    for origin in ("A6_origin", "A2_origin"):
        metad_csv = pd.read_csv(args.metad_result_root / "analysis" / origin / "all_pair_distances.csv")
        metad_x = metad_csv[names].to_numpy(dtype=float)[::20]
        metad_z[origin] = (metad_x - ref_mean) / ref_sd
    source_manifest = pd.read_csv(args.result_root / "candidate_box_manifest.csv")
    source_by_name = source_manifest.set_index("name").to_dict(orient="index")

    all_rows = []
    summaries = []
    candidate_z = {}
    edges = [(0, 250), (250, 500), (500, 750), (750, 1000)]
    for candidate, prefix, family, source_time in CANDIDATES:
        data = extract(args.gmx, args.result_root, candidate, prefix, features, outdir)
        times = data[:, 0]
        distances = data[:, 1:]
        pc = distances @ coeff
        novelty = nearest_reference(distances, ref_z, ref_mean, ref_sd)
        source = source_by_name[candidate]
        # Subsample each 0.5-ps trajectory every 10 ps for cross-seed matching,
        # reducing the influence of highly correlated adjacent frames.
        candidate_z[candidate] = ((distances - ref_mean) / ref_sd)[::20]
        opposite_origin = "A2_origin" if family == "A6_origin" else "A6_origin"
        opposite_nn = nearest_between_clouds(candidate_z[candidate], metad_z[opposite_origin])
        pd.DataFrame({"time_ps": times[::20], "nearest_opposite_origin_metad_rms_z": opposite_nn}).to_csv(
            outdir / f"{candidate}_opposite_origin_metad_match.csv", index=False)
        table = pd.DataFrame({"time_ps": times, "pc1": pc, "nearest_ref_rms_z": novelty})
        table.to_csv(outdir / f"{candidate}_fingerprint.csv", index=False)
        summary = {
            "candidate": candidate, "family": family, "source_time_ps": source_time,
            "source_pc1": float(source["candidate_pc1"]),
            "source_nearest_ref_rms_z": float(source["nearest_reference_rms_z"]),
            "frames": int(len(times)), "time_ps": [float(times[0]), float(times[-1])],
            "frame_interval_ps": float(np.median(np.diff(times))),
            "pc1": {"mean": float(pc.mean()), "sd": float(pc.std()),
                    "min": float(pc.min()), "max": float(pc.max()),
                    "first_frame": float(pc[0]), "last_frame": float(pc[-1]),
                    "blocks": interval_stats(pc, times, edges)},
            "nearest_reference_rms_z": {
                "median": float(np.median(novelty)), "q95": float(np.quantile(novelty, .95)),
                "max": float(novelty.max()), "fraction_gt_heldout_q95": float(np.mean(novelty > q95)),
                "fraction_gt_heldout_q99": float(np.mean(novelty > q99)),
                "blocks": interval_stats(novelty, times, edges)},
            "opposite_origin_metad_match": {
                "origin_compared": opposite_origin, "median": float(np.median(opposite_nn)),
                "q10": float(np.quantile(opposite_nn, .1)), "q90": float(np.quantile(opposite_nn, .9)),
                "fraction_within_reference_q95": float(np.mean(opposite_nn <= q95)),
                "fraction_within_reference_q99": float(np.mean(opposite_nn <= q99)),
            },
        }
        summaries.append(summary)
        all_rows.append((candidate, family, times, pc, novelty, source_time, float(source["candidate_pc1"])))

    report = [
        "# B6 q1 candidate NPT basin-persistence diagnostic", "",
        "This is a 1-ns NPT diagnostic with both reactive distances harmonically restrained near 1.75 nm. It is not an unbiased population estimate, PMF, or product-selectivity estimate.", "",
        f"All four trajectories are projected onto the deployed 43-distance PC1 (coefficients from PLUMED). The nearest-reference novelty score is standardized RMS distance to pair-2 matched umbrella frames. Held-out-window reference thresholds: q95={q95:.3f}, q99={q99:.3f}.", "",
        "## Candidate results", "",
        "| Candidate | frames / ps | source PC1 → NPT 0 ps → 1000 ps | PC1 mean ± SD | nearest-ref median / q95 | frames above ref q95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        pc, nv = s["pc1"], s["nearest_reference_rms_z"]
        report.append(f"| {s['candidate']} | {s['frames']} / {s['time_ps'][0]:.1f}–{s['time_ps'][1]:.1f} | {s['source_pc1']:.2f} → {pc['first_frame']:.2f} → {pc['last_frame']:.2f} | {pc['mean']:.2f} ± {pc['sd']:.2f} | {nv['median']:.2f} / {nv['q95']:.2f} | {nv['fraction_gt_heldout_q95']:.1%} |")
    report += ["", "## 250-ps block summaries", "", "PC1 block means and SDs; novelty block means are RMS-z distances to the umbrella reference cloud.", ""]
    for s in summaries:
        report += [f"### {s['candidate']}", "",
                   "| Interval (ps) | PC1 mean ± SD | novelty mean ± SD | novelty fraction above reference q95 |",
                   "|---:|---:|---:|---:|"]
        for pc_block, nv_block in zip(s["pc1"]["blocks"], s["nearest_reference_rms_z"]["blocks"]):
            mask_lo, mask_hi = map(float, pc_block["interval_ps"].split("-"))
            source_data = next(row for row in all_rows if row[0] == s["candidate"])
            mask = (source_data[2] >= mask_lo) & ((source_data[2] < mask_hi) if mask_hi < source_data[2][-1] else (source_data[2] <= mask_hi))
            frac = float(np.mean(source_data[4][mask] > q95))
            report.append(f"| {pc_block['interval_ps']} | {pc_block['mean']:.2f} ± {pc_block['sd']:.2f} | {nv_block['mean']:.2f} ± {nv_block['sd']:.2f} | {frac:.1%} |")
        report.append("")
    report += [
        "## Cross-candidate fingerprint similarity", "",
        "For each pair, 10-ps-subsampled frames from candidate A were matched to their nearest frame from candidate B in the same standardized 43-distance metric. Low nearest-neighbor distances indicate that the two preparations visit similar fingerprints; this is descriptive overlap, not evidence of equilibrium exchange.", "",
        "| Candidate A | Candidate B | nearest-frame RMS-z median (q10–q90) | fraction of A frames within reference held-out q95 |",
        "|---|---|---:|---:|",
    ]
    cross_similarity = {}
    candidate_names = [row[0] for row in all_rows]
    for i, name_a in enumerate(candidate_names):
        za = candidate_z[name_a]
        for name_b in candidate_names[i + 1:]:
            zb = candidate_z[name_b]
            nearest = np.empty(len(za), dtype=float)
            for start in range(0, len(za), 100):
                delta = za[start:start + 100, None, :] - zb[None, :, :]
                d2 = np.mean(delta * delta, axis=2)
                nearest[start:start + 100] = np.sqrt(np.min(d2, axis=1))
            pair_key = f"{name_a}__{name_b}"
            cross_similarity[pair_key] = {
                "n_sampled_frames_a": int(len(za)), "n_sampled_frames_b": int(len(zb)),
                "median": float(np.median(nearest)), "q10": float(np.quantile(nearest, .1)),
                "q90": float(np.quantile(nearest, .9)),
                "fraction_a_within_reference_q95": float(np.mean(nearest <= q95)),
            }
            report.append(f"| {name_a} | {name_b} | {np.median(nearest):.3f} ({np.quantile(nearest, .1):.3f}–{np.quantile(nearest, .9):.3f}) | {np.mean(nearest <= q95):.1%} |")
    report.append("")
    report += [
        "## Candidate match to the opposite WT-MetaD start", "",
        "Each equilibrated candidate was compared against the entire 10-ns WT-MetaD trajectory launched from the other reciprocal origin, subsampled every 10 ps. This asks whether the candidate fingerprint appears in the opposite biased trajectory; it does not establish unbiased equilibrium exchange. The held-out umbrella q95/q99 thresholds are shown only as scale references.", "",
        "| Candidate NPT trajectory | compared WT-MetaD origin | nearest RMS-z median (q10–q90) | within reference q95 / q99 |",
        "|---|---|---:|---:|",
    ]
    for s in summaries:
        m = s["opposite_origin_metad_match"]
        report.append(f"| {s['candidate']} | {m['origin_compared']} | {m['median']:.3f} ({m['q10']:.3f}–{m['q90']:.3f}) | {m['fraction_within_reference_q95']:.1%} / {m['fraction_within_reference_q99']:.1%} |")
    report.append("")
    report += [
        "## Interpretation", "",
        "Compare each 250-ps block with the source fingerprint, the matched-umbrella reference family, and the held-out novelty threshold. Retention of elevated novelty over late blocks supports persistence of an unusual geometry under the restraint protocol; relaxation toward the reference cloud means the original MetaD snapshot was not retained. A projected PC1 alone is not a basin identity, so inspect the full 43-distance fingerprints and representative structures before deciding whether separate umbrella-start families are needed.", "",
        "The candidates were each re-solvated independently and are one trajectory per snapshot. Their biased/restraint-conditioned dynamics do not establish equilibrium basin weights or reaction probabilities. Use the diagnostic to choose candidate umbrella starting basins, then calculate matched PMFs with consistent PBC settings and windows.", "",
    ]
    (outdir / "REPORT.md").write_text("\n".join(report))
    (outdir / "summary.json").write_text(json.dumps({"reference_pc1": ref_summary,
                                                       "novelty_reference_q95": q95,
                                                       "novelty_reference_q99": q99,
                                                       "candidates": summaries,
                                                       "cross_candidate_fingerprint_similarity": cross_similarity}, indent=2) + "\n")

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    colors = {"A6_origin": "#2374AB", "A2_origin": "#D1495B"}
    for candidate, family, times, pc, novelty, source_time, source_pc1 in all_rows:
        label = candidate.replace("_high_novelty", "").replace("_positive_pc1", "").replace("_negative_pc1", "").replace("_crossing_frame", "").replace("_return", "")
        axes[0].plot(times / 1000, pd.Series(pc).rolling(101, center=True, min_periods=1).mean(),
                     lw=1.0, color=colors[family], alpha=.9, label=label)
        axes[1].plot(times / 1000, pd.Series(novelty).rolling(101, center=True, min_periods=1).mean(),
                     lw=1.0, color=colors[family], alpha=.9, label=label)
    axes[1].axhline(q95, color="black", ls="--", lw=1, label=f"reference held-out q95={q95:.2f}")
    axes[0].set_ylabel("43-distance PC1 (50 ps rolling mean)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_ylabel("nearest-reference RMS z-distance\n(50 ps rolling mean)")
    axes[1].set_xlabel("NPT time (ns)")
    axes[1].legend(fontsize=7, ncol=2)
    fig.savefig(outdir / "basin_persistence.png", dpi=180)
    plt.close(fig)
    print(outdir / "REPORT.md")
    print(outdir / "basin_persistence.png")


if __name__ == "__main__":
    main()
