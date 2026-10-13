#!/usr/bin/env python3
"""Select jointly visited 2D WT-MetaD basins for umbrella re-seeding."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from analyze_b6_q1_2d_wtmetad import DIST_EDGES, RG_EDGES, RAW, RUNS, grid, stitch


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "results/b6_q1_2d_wtmetad_analysis_20260919"


def candidates(left: str, right: str, distance_max: float, limit: int) -> list[dict[str, object]]:
    a, _ = stitch(RAW / left / "COLVAR", 3)
    b, _ = stitch(RAW / right / "COLVAR", 3)
    ga, gb = grid(a, RG_EDGES, DIST_EDGES), grid(b, RG_EDGES, DIST_EDGES)
    common = np.minimum(ga, gb)
    rows: list[dict[str, object]] = []
    for i, j in zip(*np.where((common > 0) & (DIST_EDGES[:-1] < distance_max))):
        a_mask = (a[:, 1] >= RG_EDGES[i]) & (a[:, 1] < RG_EDGES[i + 1]) & (a[:, 2] >= DIST_EDGES[j]) & (a[:, 2] < DIST_EDGES[j + 1])
        b_mask = (b[:, 1] >= RG_EDGES[i]) & (b[:, 1] < RG_EDGES[i + 1]) & (b[:, 2] >= DIST_EDGES[j]) & (b[:, 2] < DIST_EDGES[j + 1])
        if not a_mask.any() or not b_mask.any():
            continue
        ac, bc = a[a_mask], b[b_mask]
        rows.append({
            "orbit": "full-switch d2" if "full_switch" in left else "one-arm reuse",
            "basin_type": "contact" if distance_max <= 0.60 else "intermediate",
            "rg_low_nm": float(RG_EDGES[i]), "rg_high_nm": float(RG_EDGES[i + 1]),
            "distance_low_nm": float(DIST_EDGES[j]), "distance_high_nm": float(DIST_EDGES[j + 1]),
            "common_mass": float(common[i, j]), "rep1_mass": float(ga[i, j]), "rep3_mass": float(gb[i, j]),
            "rep1_frames": int(len(ac)), "rep3_frames": int(len(bc)),
            "rep1_time_ps": float(ac[np.argmin(np.abs(ac[:, 0] - np.median(ac[:, 0]))), 0]),
            "rep3_time_ps": float(bc[np.argmin(np.abs(bc[:, 0] - np.median(bc[:, 0]))), 0]),
        })
    rows.sort(key=lambda row: float(row["common_mass"]), reverse=True)
    return rows[:limit]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for left, right in ((RUNS[0], RUNS[1]), (RUNS[2], RUNS[3])):
        rows.extend(candidates(left, right, 0.60, 8))
        rows.extend(candidates(left, right, 1.20, 5))
    path = OUT / "common_basin_seed_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(path)


if __name__ == "__main__":
    main()
