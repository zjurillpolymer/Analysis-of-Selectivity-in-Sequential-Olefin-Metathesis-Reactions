#!/usr/bin/env python3
"""Screen targeted C8 q1 full-reuse WT-MetaD for previously unseen CV basins.

This is a discovery/robustness analysis only.  It does not estimate a primary
reaction-path PMF or replace the distance umbrella WHAM/MBAR calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def read_colvar(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fields = None
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("#! FIELDS"):
            fields = line.split()[2:]
        elif line and not line.startswith(("#", "@")):
            rows.append([float(value) for value in line.split()])
    if not fields or not rows:
        raise ValueError(f"no PLUMED FIELDS/data found in {path}")
    index = {name: i for i, name in enumerate(fields)}
    for name in ("time", "rg", "k2"):
        if name not in index:
            raise ValueError(f"{path}: missing {name}; fields={fields}")
    data = np.asarray(rows, dtype=float)
    return data[:, index["time"]], data[:, index["rg"]], data[:, index["k2"]]


def read_reference(paths: Iterable[Path]) -> tuple[np.ndarray, np.ndarray]:
    rg, k2 = [], []
    for path in paths:
        _, one_rg, one_k2 = read_colvar(path)
        rg.append(one_rg)
        k2.append(one_k2)
    if not rg:
        raise ValueError("at least one reference COLVAR is required")
    return np.concatenate(rg), np.concatenate(k2)


def occupancy(rg: np.ndarray, k2: np.ndarray, rg_edges: np.ndarray, k2_edges: np.ndarray) -> np.ndarray:
    hist, _, _ = np.histogram2d(rg, k2, bins=(rg_edges, k2_edges))
    return hist


def connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    components: list[list[tuple[int, int]]] = []
    nrow, ncol = mask.shape
    for i in range(nrow):
        for j in range(ncol):
            if not mask[i, j] or (i, j) in seen:
                continue
            stack = [(i, j)]
            seen.add((i, j))
            component: list[tuple[int, int]] = []
            while stack:
                ii, jj = stack.pop()
                component.append((ii, jj))
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if not (di or dj):
                            continue
                        ni, nj = ii + di, jj + dj
                        if 0 <= ni < nrow and 0 <= nj < ncol and mask[ni, nj] and (ni, nj) not in seen:
                            seen.add((ni, nj))
                            stack.append((ni, nj))
            components.append(component)
    return components


def analyze(meta_paths: list[Path], reference_paths: list[Path], output: Path, burn_ps: float) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    ref_rg, ref_k2 = read_reference(reference_paths)
    rg_edges = np.linspace(0.30, 4.50, 169)
    k2_edges = np.linspace(0.0, 1.0, 101)
    ref_hist = occupancy(ref_rg, ref_k2, rg_edges, k2_edges)
    ref_mask = ref_hist > 0
    rows: list[dict[str, object]] = []
    overall_new = []
    for path in sorted(meta_paths):
        time, rg, k2 = read_colvar(path)
        keep = np.isfinite(time) & np.isfinite(rg) & np.isfinite(k2) & (time >= burn_ps)
        if keep.sum() < 100:
            raise ValueError(f"{path}: only {int(keep.sum())} post-burn frames")
        post_rg, post_k2 = rg[keep], k2[keep]
        hist = occupancy(post_rg, post_k2, rg_edges, k2_edges)
        novel_mask = (hist > 0) & ~ref_mask
        novel_frames = hist[novel_mask].sum()
        # A candidate component must be persistently populated, not a single bin.
        candidate_components = []
        for component in connected_components(novel_mask):
            count = sum(hist[i, j] for i, j in component)
            if count / len(post_rg) >= 0.01 and len(component) >= 3:
                candidate_components.append({
                    "bins": len(component),
                    "frames": int(count),
                    "fraction": float(count / len(post_rg)),
                })
        half = len(post_rg) // 2
        h1 = occupancy(post_rg[:half], post_k2[:half], rg_edges, k2_edges) > 0
        h2 = occupancy(post_rg[half:], post_k2[half:], rg_edges, k2_edges) > 0
        union = np.logical_or(h1, h2).sum()
        jaccard = float(np.logical_and(h1, h2).sum() / union) if union else 0.0
        row = {
            "replica": path.parent.name,
            "colvar": str(path),
            "frames_total": int(len(time)),
            "frames_post_burn": int(len(post_rg)),
            "rg_q01": float(np.quantile(post_rg, 0.01)),
            "rg_q50": float(np.quantile(post_rg, 0.50)),
            "rg_q99": float(np.quantile(post_rg, 0.99)),
            "k2_q01": float(np.quantile(post_k2, 0.01)),
            "k2_q50": float(np.quantile(post_k2, 0.50)),
            "k2_q99": float(np.quantile(post_k2, 0.99)),
            "novel_frame_fraction_vs_reference": float(novel_frames / len(post_rg)),
            "candidate_new_components": json.dumps(candidate_components, separators=(",", ":")),
            "half_occupancy_jaccard": jaccard,
        }
        rows.append(row)
        overall_new.append(float(novel_frames / len(post_rg)))

    with (output / "replica_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = [
        "# C8 q1 full-reuse targeted WT-MetaD basin screen",
        "",
        "This report is a discovery check only. WT-MetaD is not used to replace the distance umbrella PMF or to estimate the product yield.",
        "",
        f"Reference trajectories: {len(reference_paths)} COLVAR files; MetaD trajectories: {len(meta_paths)}; post-burn time starts at {burn_ps:g} ps.",
        "",
        "| replica | post-burn frames | novel frame fraction | candidate persistent new components | half occupancy Jaccard |",
        "|---|---:|---:|---|---:|",
    ]
    for row in rows:
        report.append(
            f"| {row['replica']} | {row['frames_post_burn']} | {float(row['novel_frame_fraction_vs_reference']):.3f} | "
            f"{row['candidate_new_components']} | {float(row['half_occupancy_jaccard']):.3f} |"
        )
    report += [
        "",
        "Interpretation rule: a new basin is only a candidate when it forms a persistent 2D connected component (at least 3 adjacent CV bins and at least 1% of post-burn frames) that is absent from the ordinary-MD/umbrella reference occupancy. A high novel fraction without a persistent component is treated as broad biased exploration, not proof of a thermodynamic basin.",
        "",
        "The bias is global (solute Rg and KAPPA2); the V5–V7 reaction distance is not biased. Any candidate basin must therefore be returned to the existing distance umbrella protocol before it can affect the main free-energy result.",
        "",
    ]
    (output / "BASIN_SCREEN_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    summary = {
        "meta_trajectories": len(meta_paths),
        "reference_trajectories": len(reference_paths),
        "burn_ps": burn_ps,
        "novel_fraction_mean": float(np.mean(overall_new)),
        "novel_fraction_max": float(np.max(overall_new)),
        "persistent_new_component_present": any(json.loads(row["candidate_new_components"]) for row in rows),
        "interpretation": "candidate-basin screen only; follow-up with distance umbrella required",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metad-root", type=Path, required=True)
    parser.add_argument("--reference-colvar", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--burn-ps", type=float, default=2000.0)
    args = parser.parse_args()
    meta_paths = sorted(args.metad_root.glob("**/COLVAR"))
    if not meta_paths:
        raise SystemExit(f"no COLVAR files under {args.metad_root}")
    print(json.dumps(analyze(meta_paths, args.reference_colvar, args.output, args.burn_ps), indent=2))


if __name__ == "__main__":
    main()
