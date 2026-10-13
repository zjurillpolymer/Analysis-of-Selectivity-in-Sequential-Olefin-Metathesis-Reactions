#!/usr/bin/env python3
"""Analyze reciprocal-start B6 contact-PC1 WT-MetaD trajectories.

This is a basin-discovery diagnostic, not a free-energy estimator. It extracts
the 45 vinyl-center pair distances from each XTC with GROMACS, verifies the
43-feature PLUMED PC1 against COLVAR, and compares sampled distance fingerprints
with the umbrella frames used to define the CV.
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
DEFAULT_RESULT_ROOT = ROOT / "b6_q1_contact_pc1_wtmetad_results_20261002"
DEFAULT_PACKAGE_ROOT = ROOT / "b6_q1_contact_pc1_wtmetad_server_20261002"
DEFAULT_REFERENCE = ROOT / (
    "Analysis-of-Selectivity-in-Sequential-Olefin-Metathesis-Reactions/results/"
    "b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001/"
    "joint_mbar/hidden_conformation_diagnostic/matched_frames_and_contacts.csv"
)
ORIGINS = ("A6_origin", "A2_origin")
CANDIDATE_TIMES_PS = {
    # Mix high novelty, both signs of the exploration CV, and well-separated
    # late frames rather than choosing adjacent snapshots from one excursion.
    "A6_origin": [690, 1270, 6940, 8540],
    "A2_origin": [2030, 3890, 5480, 9090],
}


def parse_features(cv_json: dict) -> list[tuple[int, int, str]]:
    features = []
    for name in cv_json["features"]:
        match = re.fullmatch(r"V(\d+)-V(\d+)_nm", name)
        if not match:
            raise ValueError(f"Unrecognized feature name: {name}")
        features.append((int(match.group(1)), int(match.group(2)), name))
    return features


def selection_for_pair(a: int, b: int) -> str:
    # PLUMED atom indices in this topology start the ten vinyl centers at 189.
    aa, bb = 189 + 2 * (a - 3), 189 + 2 * (a - 3) + 1
    ba, bbi = 189 + 2 * (b - 3), 189 + 2 * (b - 3) + 1
    return (
        f"com of (atomnr {aa} or atomnr {bb}) plus "
        f"com of (atomnr {ba} or atomnr {bbi})"
    )


def load_xvg(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments=("#", "@"))


def extract_distances(gmx: str, origin_dir: Path, features: list[tuple[int, int, str]], outdir: Path) -> tuple[np.ndarray, list[str]]:
    # The exact same atom pairs as the 43-feature CV plus the two restrained
    # reaction distances, to audit that both restraints remained near 1.75 nm.
    all_pairs = [(a, b, name) for a, b, name in features]
    all_pairs += [(3, 10, "V3-V10_nm"), (3, 12, "V3-V12_nm")]
    sel_file = outdir / "distance_selections.txt"
    dist_file = outdir / "all_vinyl_pair_distances.xvg"
    sel_file.write_text("\n".join(selection_for_pair(a, b) for a, b, _ in all_pairs) + "\n")
    selection_string = "; ".join(selection_for_pair(a, b) for a, b, _ in all_pairs)
    cmd = [
        gmx, "distance", "-f", str(origin_dir / "wtmeta.xtc"),
        "-s", str(origin_dir / "wtmeta.tpr"), "-select", selection_string,
        "-oall", str(dist_file), "-xvg", "none",
    ]
    subprocess.run(cmd, check=True, cwd=outdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    data = load_xvg(dist_file)
    if data.shape[1] != len(all_pairs) + 1:
        raise RuntimeError(f"Expected {len(all_pairs)} distances, got {data.shape[1]-1} in {dist_file}")
    return data, [x[2] for x in all_pairs]


def block_summary(values: np.ndarray, times_ps: np.ndarray, boundaries_ns: list[tuple[int, int]]) -> list[dict]:
    out = []
    for lo, hi in boundaries_ns:
        mask = (times_ps >= lo * 1000) & (times_ps < hi * 1000)
        x = values[mask]
        out.append({
            "interval_ns": f"{lo}-{hi}", "n_frames": int(len(x)),
            "mean": float(np.mean(x)), "sd": float(np.std(x)),
            "min": float(np.min(x)), "max": float(np.max(x)),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gmx", default="gmx")
    ap.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    ap.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    args = ap.parse_args()
    outroot = args.result_root / "analysis"
    outroot.mkdir(parents=True, exist_ok=True)
    cvdef = json.loads((args.package_root / "cv_definition.json").read_text())
    features = parse_features(cvdef)
    ref = pd.read_csv(args.reference)
    # The PCA was fit specifically on pair-2 matched umbrella frames.
    ref = ref.loc[ref["pair"] == 2].copy()
    names = [x[2] for x in features]
    for name in names:
        if name not in ref:
            raise ValueError(f"Reference CSV lacks {name}")
    # Coefficients are read directly from the exact deployed PLUMED definition.
    plumed = (args.package_root / "A6_origin/plumed.dat").read_text()
    cvline = next(line for line in plumed.splitlines() if line.startswith("pc1: COMBINE"))
    coeff_text = re.search(r"COEFFICIENTS=([^ ]+)", cvline).group(1)
    coeff = np.array([float(x) for x in coeff_text.split(",")])
    if len(coeff) != len(features):
        raise ValueError("PLUMED coefficient count does not match feature count")
    ref_x = ref[names].to_numpy(float)
    ref_pc = ref_x @ coeff
    ref["deployed_pc1_reconstructed"] = ref_pc
    ref_summary = {}
    for fam in ("A6", "A2"):
        x = ref.loc[ref["family"] == fam, "deployed_pc1_reconstructed"].to_numpy()
        ref_summary[fam] = {
            "n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std()),
            "q05": float(np.quantile(x, .05)), "q50": float(np.quantile(x, .50)),
            "q95": float(np.quantile(x, .95)),
        }
    ref_mean = ref_x.mean(axis=0)
    ref_sd = ref_x.std(axis=0)
    ref_sd[ref_sd == 0] = 1.0
    zref = (ref_x - ref_mean) / ref_sd
    # Calibrate fingerprint novelty against held-out umbrella windows rather
    # than frame-wise leave-one-out, which is artificially small because
    # adjacent MD frames are highly correlated.
    ref_windows = ref["window"].to_numpy()
    ref_loo = np.empty(len(zref))
    for start in range(0, len(zref), 50):
        idx = np.arange(start, min(start + 50, len(zref)))
        d2 = ((zref[idx, None, :] - zref[None, :, :]) ** 2).mean(axis=2)
        d2[ref_windows[idx, None] == ref_windows[None, :]] = np.inf
        ref_loo[idx] = np.sqrt(d2.min(axis=1))
    ref_loo_q95 = float(np.quantile(ref_loo, .95))
    ref_loo_q99 = float(np.quantile(ref_loo, .99))

    all_out = {}
    candidate_rows = []
    for origin in ORIGINS:
        origin_dir = args.result_root / origin
        od = outroot / origin
        od.mkdir(parents=True, exist_ok=True)
        dist, dist_names = extract_distances(args.gmx, origin_dir, features, od)
        times = dist[:, 0]
        x = dist[:, 1:1 + len(features)]
        pc = x @ coeff
        colvar = load_xvg(origin_dir / "COLVAR")
        if len(colvar) != len(pc):
            raise RuntimeError(f"COLVAR/trajectory frame mismatch for {origin}: {len(colvar)} vs {len(pc)}")
        pc_error = float(np.max(np.abs(pc - colvar[:, 1])))
        # Check reference trained CV direction: class centers separate the two
        # conformational families, while WT-MetaD itself is not reweighted here.
        standardized = (x - ref_mean) / ref_sd
        # RMS distance to the nearest matched umbrella fingerprint; summarize
        # modest frame subsampling to keep this diagnostic interpretable.
        sample_idx = np.arange(0, len(x), 20)
        nearest = np.empty(len(sample_idx))
        for start in range(0, len(sample_idx), 100):
            idx = sample_idx[start:start + 100]
            # Euclidean RMS across standardized feature dimensions.
            d2 = ((standardized[idx, None, :] - zref[None, :, :]) ** 2).mean(axis=2)
            nearest[start:start + len(idx)] = np.sqrt(d2.min(axis=1))
        # Put the nearest-neighbor score back onto the subsampled time grid.
        rows = []
        for j, i in enumerate(sample_idx):
            rows.append({"time_ps": float(times[i]), "pc1_reconstructed": float(pc[i]),
                         "pc1_colvar": float(colvar[i, 1]), "nearest_ref_rms_z": float(nearest[j])})
        pd.DataFrame(rows).to_csv(od / "sampled_cv_and_reference_novelty.csv", index=False)
        dist_frame = pd.DataFrame(dist[:, 1:], columns=dist_names)
        dist_frame.insert(0, "time_ps", times)
        dist_frame.to_csv(od / "all_pair_distances.csv", index=False)
        sampled = pd.DataFrame(rows)
        distance_lookup = dist_frame.set_index("time_ps")
        for time_ps in CANDIDATE_TIMES_PS[origin]:
            # The chosen times are on both the 0.5 ps trajectory and 10 ps
            # novelty-sampling grids.
            gro = outroot / "candidate_structures" / f"{origin}_{time_ps:05d}ps.gro"
            gro.parent.mkdir(parents=True, exist_ok=True)
            cmd = [args.gmx, "trjconv", "-f", str(origin_dir / "wtmeta.xtc"),
                   "-s", str(origin_dir / "wtmeta.tpr"), "-dump", str(time_ps),
                   "-pbc", "mol", "-center", "-o", str(gro)]
            subprocess.run(cmd, input="2\n2\n", check=True, cwd=gro.parent,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            text = gro.read_text().splitlines()
            atom_count = int(text[1].strip())
            xyz = np.array([[float(line[20:28]), float(line[28:36]), float(line[36:44])]
                            for line in text[2:2 + atom_count]])
            if atom_count != 438 or not np.isfinite(xyz).all():
                raise RuntimeError(f"Unexpected/corrupt solute structure at {gro}")
            row = sampled.loc[np.isclose(sampled["time_ps"], time_ps)].iloc[0]
            drow = distance_lookup.loc[float(time_ps)]
            candidate_rows.append({
                "origin": origin, "time_ps": time_ps,
                "pc1": float(row["pc1_colvar"]),
                "nearest_ref_rms_z": float(row["nearest_ref_rms_z"]),
                "V3-V12_nm": float(drow["V3-V12_nm"]),
                "V3-V10_nm": float(drow["V3-V10_nm"]),
                "atoms": atom_count, "structure": str(gro),
                "solvent_coordinates_available": False,
            })
        # Pull output has both restrained distances and should be cross-checked.
        pull = load_xvg(origin_dir / "wtmeta_pullx.xvg")
        restraint_stats = []
        for i, label in enumerate(("V3-V12", "V3-V10"), start=1):
            vals = pull[:, i]
            restraint_stats.append({"distance": label, "mean_nm": float(vals.mean()),
                                    "sd_nm": float(vals.std()), "min_nm": float(vals.min()),
                                    "max_nm": float(vals.max())})
        hills = load_xvg(origin_dir / "HILLS")
        # HILLS file has five fields: time, pc1, sigma, height, biasfactor.
        cv_summary = {
            "time_ps": [float(times[0]), float(times[-1])],
            "n_frames": int(len(times)), "median_frame_interval_ps": float(np.median(np.diff(times))),
            "pc1_min": float(pc.min()), "pc1_max": float(pc.max()),
            "pc1_mean": float(pc.mean()), "pc1_sd": float(pc.std()),
            "pc1_blocks_1ns": block_summary(pc, times, [(i, i + 1) for i in range(10)]),
            "pc1_colvar_max_abs_error": pc_error,
            "hills_count": int(len(hills)), "hill_first_time_ps": float(hills[0, 0]),
            "hill_last_time_ps": float(hills[-1, 0]),
            "hill_height_first_kj_mol": float(hills[0, 3]),
            "hill_height_last_kj_mol": float(hills[-1, 3]),
            "reference_nearest_rms_z_quantiles": [float(q) for q in np.quantile(nearest, [.5, .9, .95, .99, 1])],
            "reference_held_out_window_nearest_rms_z_q95": ref_loo_q95,
            "reference_held_out_window_nearest_rms_z_q99": ref_loo_q99,
            "fraction_meta_samples_above_reference_held_out_window_q95": float(np.mean(nearest > ref_loo_q95)),
            "fraction_meta_samples_above_reference_held_out_window_q99": float(np.mean(nearest > ref_loo_q99)),
            "reference_nearest_rms_z_top_examples": sorted(rows, key=lambda r: r["nearest_ref_rms_z"], reverse=True)[:10],
            "restrained_distances": restraint_stats,
            "reference_pc1_family_summary": ref_summary,
            "interpretation_note": "WT-MetaD samples are biased; no population/free-energy inference is made.",
        }
        all_out[origin] = cv_summary
        (od / "summary.json").write_text(json.dumps(cv_summary, indent=2) + "\n")

    # Compact human-readable report; characterize novelty conservatively.
    lines = [
        "# B6 q1 contact-PC1 WT-MetaD: reciprocal-start diagnostic",
        "",
        "This is a basin-discovery supplement only. Biased time fractions are not equilibrium populations and are not converted into free energies or product yields.",
        "",
        "## Reference family separation on the deployed CV",
        "",
        "The deployed PC1 is reconstructed as the exact linear combination from PLUMED. Reference statistics come from the matched umbrella frames used to define this CV.",
        "",
        "| Family | n | Mean PC1 | SD | 5th–95th percentile |",
        "|---|---:|---:|---:|---:|",
    ]
    for fam in ("A6", "A2"):
        s = ref_summary[fam]
        lines.append(f"| {fam} | {s['n']} | {s['mean']:.3f} | {s['sd']:.3f} | {s['q05']:.3f}–{s['q95']:.3f} |")
    lines += ["", "## Reciprocal-start results", ""]
    for origin, s in all_out.items():
        q = s["reference_nearest_rms_z_quantiles"]
        lines += [
            f"### {origin}", "",
            f"- Production frames: {s['n_frames']:,}; sampled time {s['time_ps'][0]:.1f}–{s['time_ps'][1]:.1f} ps; HILLS: {s['hills_count']:,} (last at {s['hill_last_time_ps']:.1f} ps).",
            f"- PC1 range {s['pc1_min']:.2f} to {s['pc1_max']:.2f}; mean {s['pc1_mean']:.2f}, SD {s['pc1_sd']:.2f}.",
            f"- Recomputed PC1 vs PLUMED COLVAR max absolute discrepancy: {s['pc1_colvar_max_abs_error']:.4g}.",
            f"- Nearest matched-umbrella fingerprint RMS distance (standardized over 43 features), median / 95th / max: {q[0]:.2f} / {q[2]:.2f} / {q[4]:.2f}.",
            f"- Against a held-out-window reference baseline (95th percentile {s['reference_held_out_window_nearest_rms_z_q95']:.2f}), {s['fraction_meta_samples_above_reference_held_out_window_q95']:.0%} of MetaD samples lie farther from the umbrella reference cloud; this flags candidate novelty, not proof of a distinct basin.",
            "- Restrained reactive distances: " + "; ".join(f"{r['distance']} {r['mean_nm']:.3f}±{r['sd_nm']:.3f} nm" for r in s["restrained_distances"]) + ".",
            "",
            "1-ns PC1 means: " + ", ".join(f"{b['interval_ns']} ns {b['mean']:.2f}±{b['sd']:.2f}" for b in s["pc1_blocks_1ns"]),
            "",
        ]
    lines += [
        "## Interpretation and limits", "",
        "The key positive result is whether the trajectories move through PC1/contact-fingerprint regions outside the matched-umbrella reference cloud. Large excursions along PC1 alone are not sufficient to claim a new basin: WT-MetaD explicitly biases this coordinate, and a one-dimensional projection can merge distinct structures. The nearest-reference score is exploratory and depends on the umbrella reference set; visually inspect the listed high-novelty frames before calling them new basins.",
        "",
        "These results do not establish a probability ratio for A1→A6 versus A1→A2 and cannot replace the umbrella PMF. Any candidate new basin should be extracted, checked for structural integrity and PBC artifacts, and then seeded into matched umbrella windows with the original two reaction-distance restraints.",
        "",
        "Eight representative solute-only structures (438 atoms) were extracted with molecule-whole PBC treatment. The XTC contains only the solute, not the 8,781 solvent atoms, so these structures are not directly ready to serve as full-box umbrella starts; solvent coordinates must be reconstructed and equilibrated before such a test.",
        "",
    ]
    (outroot / "REPORT.md").write_text("\n".join(lines))
    (outroot / "summary.json").write_text(json.dumps(all_out, indent=2) + "\n")
    pd.DataFrame(candidate_rows).to_csv(outroot / "candidate_structure_manifest.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    colors = {"A6_origin": "#2374AB", "A2_origin": "#D1495B"}
    threshold = (ref_summary["A6"]["mean"] + ref_summary["A2"]["mean"]) / 2
    for origin in ORIGINS:
        cv = load_xvg(args.result_root / origin / "COLVAR")
        sample = pd.read_csv(outroot / origin / "sampled_cv_and_reference_novelty.csv")
        # 50 ps rolling average suppresses rapid thermal recrossings.
        smooth = pd.Series(cv[:, 1]).rolling(100, center=True, min_periods=1).mean().to_numpy()
        axes[0].plot(cv[:, 0] / 1000, smooth, lw=0.8, alpha=0.85,
                     color=colors[origin], label=origin)
        axes[1].plot(sample.time_ps / 1000, sample.nearest_ref_rms_z, lw=0.8,
                     color=colors[origin], label=origin)
    axes[0].axhline(threshold, color="black", ls="--", lw=1,
                    label=f"reference-family midpoint ({threshold:.2f})")
    axes[0].axhspan(ref_summary["A2"]["q05"], ref_summary["A2"]["q95"],
                    color=colors["A2_origin"], alpha=0.07, label="A2 reference 5–95%")
    axes[0].axhspan(ref_summary["A6"]["q05"], ref_summary["A6"]["q95"],
                    color=colors["A6_origin"], alpha=0.07, label="A6 reference 5–95%")
    axes[0].set_ylabel("deployed contact PC1 (50 ps mean)")
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].axhline(ref_loo_q95, color="black", ls="--", lw=1,
                    label=f"held-out umbrella-window 95th percentile ({ref_loo_q95:.2f})")
    axes[1].set_ylabel("nearest umbrella fingerprint\nRMS z-distance")
    axes[1].set_xlabel("WT-MetaD time (ns)")
    axes[1].legend(fontsize=8)
    fig.savefig(outroot / "reciprocal_start_diagnostics.png", dpi=180)
    plt.close(fig)
    print(outroot / "reciprocal_start_diagnostics.png")
    print(outroot / "REPORT.md")


if __name__ == "__main__":
    main()
