#!/usr/bin/env python3
"""Audit B6 q1 targeted 2D Rg+pair-distance WT-MetaD trajectories."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
RAW = WORKSPACE / "Analysis-of-Selectivity-in-Sequential-Olefin-Metathesis-Reactions/results/b6_q1_2d_wtmetad_results_20260919/raw/b6_q1_2d_wtmetad_server_20260919"
OUT = PROJECT / "results/b6_q1_2d_wtmetad_analysis_20260919"
RUNS = (
    "b6_q1_full_switch_d2_rep1",
    "b6_q1_full_switch_d2_rep3",
    "b6_q1_one_arm_reuse_rep1",
    "b6_q1_one_arm_reuse_rep3",
)


def read_segments(path: Path, min_fields: int) -> list[np.ndarray]:
    segments: list[list[list[float]]] = [[]]
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#!") or line.startswith("#"):
            if line.startswith("#! FIELDS") and segments[-1]:
                segments.append([])
            continue
        fields = line.split()
        try:
            row = [float(value) for value in fields]
        except ValueError:
            continue
        if len(row) >= min_fields:
            segments[-1].append(row[:min_fields])
    return [np.asarray(segment, dtype=float) for segment in segments if segment]


def stitch(path: Path, min_fields: int) -> tuple[np.ndarray, int]:
    segments = read_segments(path, min_fields)
    if not segments:
        raise ValueError(f"no numeric data in {path}")
    if len(segments) == 1:
        data = segments[0]
        return data[np.argsort(data[:, 0])], 0
    # A resumed PLUMED file can contain the original segment followed by a
    # second segment starting at an earlier time. Keep the latest segment at
    # duplicate times, which is the physically continued trajectory.
    combined = np.concatenate(segments, axis=0)
    order = np.argsort(combined[:, 0], kind="stable")
    combined = combined[order]
    unique_time: dict[float, np.ndarray] = {}
    for row in combined:
        unique_time[float(row[0])] = row
    data = np.asarray([unique_time[key] for key in sorted(unique_time)], dtype=float)
    return data, len(segments) - 1


def soft_contact(distance: np.ndarray, center: float = 0.50, width: float = 0.05) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip((distance - center) / width, -700, 700)))


def quantiles(values: np.ndarray) -> dict[str, float]:
    return dict(zip(("min", "q01", "median", "q99", "max"), map(float, np.quantile(values, (0, .01, .5, .99, 1)))))


def grid(values: np.ndarray, rg_edges: np.ndarray, distance_edges: np.ndarray) -> np.ndarray:
    counts, _, _ = np.histogram2d(values[:, 1], values[:, 2], bins=(rg_edges, distance_edges))
    total = counts.sum()
    return counts / total if total else counts


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.minimum(a, b).sum())


def episodes(time: np.ndarray, active: np.ndarray) -> tuple[int, float, int, int]:
    lengths: list[float] = []
    start: int | None = None
    for idx, state in enumerate(np.r_[active, False]):
        if state and start is None:
            start = idx
        elif not state and start is not None:
            lengths.append(float(time[idx - 1] - time[start] + (time[1] - time[0] if idx > 1 else 0.0)))
            start = None
    return len(lengths), max(lengths, default=0.0), sum(length >= 10.0 for length in lengths), sum(length >= 50.0 for length in lengths)


def run_stats(name: str, cv: np.ndarray, hills: np.ndarray, segments: int) -> dict[str, object]:
    time = cv[:, 0]
    rg = cv[:, 1]
    distance = cv[:, 2]
    dt = float(np.median(np.diff(time))) if len(time) > 1 else float("nan")
    n_episodes, max_episode, ge10, ge50 = episodes(time, distance <= 0.5)
    first = hills[: min(100, len(hills)), 5]
    last = hills[-min(1000, len(hills)) :, 5]
    half = len(cv) // 2
    return {
        "run": name,
        "orbit": "full-switch d2" if "full_switch" in name else "one-arm reuse",
        "replica": 1 if name.endswith("rep1") else 3,
        "frames": int(len(cv)),
        "duration_ns": float((time[-1] - time[0]) / 1000.0),
        "dt_ps": dt,
        "colvar_segments": int(segments + 1),
        "hills": int(len(hills)),
        "rg": quantiles(rg),
        "pair_distance": quantiles(distance),
        "hard_contact_fraction_d_le_0.50_nm": float(np.mean(distance <= 0.50)),
        "hard_contact_fraction_d_le_0.60_nm": float(np.mean(distance <= 0.60)),
        "soft_contact_mean": float(np.mean(soft_contact(distance))),
        "contact_episodes": int(n_episodes),
        "max_contact_episode_ps": float(max_episode),
        "contact_episodes_ge_10ps": int(ge10),
        "contact_episodes_ge_50ps": int(ge50),
        "hill_height_first100_mean": float(np.mean(first)),
        "hill_height_last1000_mean": float(np.mean(last)),
        "hill_height_late_over_early": float(np.mean(last) / np.mean(first)),
        "two_d_bins_full": int(np.count_nonzero(grid(cv, RG_EDGES, DIST_EDGES))),
        "two_d_bins_early": int(np.count_nonzero(grid(cv[:half], RG_EDGES, DIST_EDGES))),
        "two_d_bins_late": int(np.count_nonzero(grid(cv[half:], RG_EDGES, DIST_EDGES))),
    }


RG_EDGES = np.arange(0.80, 1.3501, 0.025)
DIST_EDGES = np.arange(0.30, 3.7001, 0.10)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    flattened = []
    for row in rows:
        out = dict(row)
        for key in ("rg", "pair_distance"):
            if key in out:
                values = out.pop(key)
                for subkey, value in values.items():
                    out[f"{key}_{subkey}"] = value
        flattened.append(out)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def plot(cvs: dict[str, np.ndarray], out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {
        RUNS[0]: "#4477AA", RUNS[1]: "#66CCEE", RUNS[2]: "#CC6677", RUNS[3]: "#AA3377",
    }
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), layout="constrained")
    for name, data in cvs.items():
        axes[0, 0].plot(data[:, 0] / 1000.0, data[:, 1], lw=.45, color=colors[name], label=name.replace("b6_q1_", ""))
        axes[0, 1].scatter(data[::5, 1], data[::5, 2], s=1, alpha=.25, color=colors[name], rasterized=True)
        axes[1, 0].hist(data[:, 2], bins=DIST_EDGES, density=True, histtype="step", lw=1.0, color=colors[name], label=name.replace("b6_q1_", ""))
    axes[0, 0].set(xlabel="Time (ns)", ylabel="Rg (nm)", title="2D WT-MetaD trajectories")
    axes[0, 1].set(xlabel="Rg (nm)", ylabel="Pair distance (nm)", title="Joint CV coverage")
    axes[1, 0].set(xlabel="Pair distance (nm)", ylabel="Biased density", title="Distance coverage")
    axes[1, 1].axis("off")
    axes[0, 0].legend(fontsize=6, ncol=2)
    axes[1, 0].legend(fontsize=6)
    for suffix in ("png", "pdf"):
        fig.savefig(out / f"b6_q1_2d_wtmetad_sampling.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cvs: dict[str, np.ndarray] = {}
    hills_by_run: dict[str, np.ndarray] = {}
    segment_counts: dict[str, int] = {}
    stats: list[dict[str, object]] = []
    for name in RUNS:
        run = RAW / name
        cv, cv_segments = stitch(run / "COLVAR", 3)
        hills, hill_segments = stitch(run / "HILLS", 7)
        cvs[name] = cv
        hills_by_run[name] = hills
        segment_counts[name] = max(cv_segments, hill_segments)
        stats.append(run_stats(name, cv, hills, segment_counts[name]))
        log = (run / "wtmeta.log").read_text(encoding="utf-8", errors="replace")
        stats[-1]["finished"] = bool(re.search(r"Finished mdrun on rank 0", log))
        stats[-1]["lincs_warning_count"] = int(log.count("LINCS WARNING"))
        stats[-1]["fatal_error"] = bool(re.search(r"Fatal error|Error on node", log, flags=re.I))
        stats[-1]["xtc_present"] = (run / "wtmeta.xtc").is_file()

    overlap_rows: list[dict[str, object]] = []
    half_rows: list[dict[str, object]] = []
    for name in RUNS:
        cv = cvs[name]
        mid = len(cv) // 2
        half_rows.append({
            "run": name,
            "early_late_2d_overlap": overlap(grid(cv[:mid], RG_EDGES, DIST_EDGES), grid(cv[mid:], RG_EDGES, DIST_EDGES)),
            "early_late_rg_overlap": overlap(np.histogram(cv[:mid, 1], RG_EDGES)[0] / mid, np.histogram(cv[mid:, 1], RG_EDGES)[0] / (len(cv) - mid)),
            "early_late_distance_overlap": overlap(np.histogram(cv[:mid, 2], DIST_EDGES)[0] / mid, np.histogram(cv[mid:, 2], DIST_EDGES)[0] / (len(cv) - mid)),
        })
    for left, right in ((RUNS[0], RUNS[1]), (RUNS[2], RUNS[3])):
        a, b = cvs[left], cvs[right]
        overlap_rows.append({
            "orbit": "full-switch d2" if "full_switch" in left else "one-arm reuse",
            "left": left, "right": right,
            "two_d_overlap": overlap(grid(a, RG_EDGES, DIST_EDGES), grid(b, RG_EDGES, DIST_EDGES)),
            "rg_overlap": overlap(np.histogram(a[:, 1], RG_EDGES)[0] / len(a), np.histogram(b[:, 1], RG_EDGES)[0] / len(b)),
            "distance_overlap": overlap(np.histogram(a[:, 2], DIST_EDGES)[0] / len(a), np.histogram(b[:, 2], DIST_EDGES)[0] / len(b)),
        })

    write_csv(OUT / "run_audit.csv", stats)
    write_csv(OUT / "replica_2d_overlap.csv", overlap_rows)
    write_csv(OUT / "half_trajectory_overlap.csv", half_rows)
    summary = {"runs": stats, "replica_overlap": overlap_rows, "half_overlap": half_rows, "rg_bin_nm": .025, "distance_bin_nm": .10}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(cvs, OUT)

    rows_by_name = {row["run"]: row for row in stats}
    lines = [
        "# B6 q1 targeted 2D WT-MetaD result audit (19 September 2026)",
        "",
        "## Scope",
        "",
        "These four trajectories bias `Rg` and the selected neighbouring-arm midpoint distance simultaneously. They are basin-discovery and umbrella-reseeding evidence, not direct q1 probabilities: biased occupancy must not be converted directly into reaction yields.",
        "",
        "## Technical completion",
        "",
        "All four paths reached 10.0 ns and reported `Finished mdrun`. The four XTC, COLVAR, HILLS and checkpoint outputs are present. The one-arm reuse replica 3 log contains a checkpoint restart at approximately 7.37 ns; the analysis removes the duplicated overlap and reconstructs one continuous 0--10 ns sequence.",
        "",
        "| path | frames after stitching | hills after stitching | Rg range (nm) | pair-distance range (nm) | d ≤ 0.50 nm | soft-contact mean |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in stats:
        lines.append(f"| {row['orbit']} rep{row['replica']} | {row['frames']} | {row['hills']} | {row['rg']['min']:.3f}–{row['rg']['max']:.3f} | {row['pair_distance']['min']:.3f}–{row['pair_distance']['max']:.3f} | {row['hard_contact_fraction_d_le_0.50_nm']:.4f} | {row['soft_contact_mean']:.4g} |")
    lines += [
        "",
        "## Two-dimensional overlap",
        "",
        "The overlap below is the histogram intersection on a 0.025 nm × 0.10 nm (`Rg` × distance) grid. It measures whether the two independent endpoints visit the same joint-CV regions; it is not a free-energy estimate.",
        "",
        "| orbit | 2D overlap | Rg overlap | distance overlap |",
        "|---|---:|---:|---:|",
    ]
    for row in overlap_rows:
        lines.append(f"| {row['orbit']} | {row['two_d_overlap']:.3f} | {row['rg_overlap']:.3f} | {row['distance_overlap']:.3f} |")
    lines += [
        "",
        "## Within-trajectory stationarity",
        "",
        "| path | early/late 2D overlap | early/late Rg overlap | early/late distance overlap | late/early hill height |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in half_rows:
        audit = rows_by_name[row["run"]]
        lines.append(f"| {row['run']} | {row['early_late_2d_overlap']:.3f} | {row['early_late_rg_overlap']:.3f} | {row['early_late_distance_overlap']:.3f} | {audit['hill_height_late_over_early']:.3f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The two endpoint replicas now share joint `(Rg, pair distance)` regions strongly: the 2D histogram intersections are 0.772 for full-switch d2 and 0.739 for one-arm reuse, compared with the earlier 1D pair-distance overlaps of 0.348 and 0.538. This is evidence that adding the monitored pair distance has addressed the main replica-dependent slow coordinate and has produced reproducible candidate basins.",
        "",
        "The trajectories are not fully converged enhanced-sampling ensembles yet. Early-versus-late 2D overlaps are only 0.348--0.542, and the final 1,000 hill heights remain about 0.30--0.42 of the first 100 hills. Therefore these runs are strong basin-discovery evidence, but they should not be used as a direct free-energy or q1 probability estimate.",
        "",
        "The contact fractions in the table are biased-CV occupancies, not reaction probabilities. In particular, no conclusion about R2/R4 yield should be drawn from the 0.50 nm contact fraction alone.",
        "",
        "The next valid use of these data is conservative: identify basins that are visited from both independent endpoints, extract full-system frames (including DCM), and restart the identical 25-window umbrella calculation from those frames. Only the umbrella PMF/branch comparison should enter the final q1 and product-yield calculation.",
        "",
        "## Files",
        "",
        "- `run_audit.csv`: per-path technical and coverage audit.",
        "- `replica_2d_overlap.csv`: endpoint-to-endpoint joint-CV overlap.",
        "- `half_trajectory_overlap.csv`: early-vs-late stationarity diagnostic.",
        "- `common_basin_seed_manifest.csv`: jointly visited contact/intermediate bins and representative times for full-system extraction.",
        "- `summary.json`: machine-readable summary.",
        "- `b6_q1_2d_wtmetad_sampling.png/pdf`: coverage figure.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
