#!/usr/bin/env python3
"""Audit global-conformation exchange in existing C8 q1 full-reuse umbrellas.

This is a no-new-MD diagnostic.  It recomputes the solute Rg, shape anisotropy,
distance, and soft-NAC observables for each existing distance-biased window,
then reports adjacent-window overlap in Rg.  The result is deliberately not a
PMF or a reaction probability: it only tests whether the old distance umbrella
can move between the normal and compact global-conformation regions.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.analyze_c8_q1_colvars_wtmetad import extract, topology  # noqa: E402


PATH_SPECS = (
    ("rep1", PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_download/20730624/extracted/pathway"),
    ("rep2", PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_download/20730625/extracted/pathway"),
    ("rep3", PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_rep3_download/20730989/extracted/pathway"),
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def window_center(window: Path) -> float:
    # window_004_0.750nm
    return float(window.name.rsplit("_", 1)[1].removesuffix("nm"))


def distribution(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=edges)
    hist = hist.astype(float)
    total = hist.sum()
    return hist / total if total else hist


def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.sqrt(a * b)))


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    # Natural-log Jensen-Shannon divergence; zero bins are handled explicitly.
    m = 0.5 * (a + b)
    terms = []
    for x in (a, b):
        mask = x > 0
        terms.append(float(np.sum(x[mask] * np.log(x[mask] / m[mask]))))
    return 0.5 * sum(terms)


def load_reference_meta(path: Path) -> np.ndarray:
    with np.load(path) as z:
        data = np.asarray(z["data"], dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"invalid MetaD source data: {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "results/c8_q1_full_reuse_exchange_diagnostic_20260914")
    parser.add_argument("--metad-source", type=Path,
                        default=PROJECT / "results/c8_q1_colvars_wtmetad_analysis_20260914")
    parser.add_argument("--gmx", default="gmx", help="kept for a stable CLI; no gmx call is needed")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"

    all_window_rows: list[dict[str, object]] = []
    all_data: dict[str, list[np.ndarray]] = {name: [] for name, _ in PATH_SPECS}
    all_centers: dict[str, list[float]] = {name: [] for name, _ in PATH_SPECS}
    masses = bonds = names = None

    for replica, pathway in PATH_SPECS:
        if not pathway.is_dir():
            raise FileNotFoundError(pathway)
        masses, bonds, names = topology(pathway / "system.top")
        windows = sorted((pathway / "windows").glob("window_*"), key=window_center)
        if not windows:
            raise FileNotFoundError(f"no windows under {pathway}")
        for window in windows:
            xtc, gro = window / "production.xtc", window / "production.gro"
            if not xtc.is_file() or not gro.is_file():
                raise FileNotFoundError(f"incomplete window: {window}")
            data, _ = extract(xtc, gro, masses, bonds, names, cache)
            center = window_center(window)
            all_data[replica].append(data)
            all_centers[replica].append(center)
            all_window_rows.append({
                "replica": replica,
                "window": window.name,
                "center_nm": center,
                "frames": len(data),
                "distance_mean_nm": float(np.mean(data[:, 3])),
                "distance_sd_nm": float(np.std(data[:, 3])),
                "rg_mean_nm": float(np.mean(data[:, 1])),
                "rg_sd_nm": float(np.std(data[:, 1])),
                "rg_min_nm": float(np.min(data[:, 1])),
                "rg_max_nm": float(np.max(data[:, 1])),
                "fraction_rg_lt_0_80": float(np.mean(data[:, 1] < 0.80)),
                "fraction_rg_lt_0_75": float(np.mean(data[:, 1] < 0.75)),
                "kappa2_mean": float(np.mean(data[:, 2])),
                "soft_nac_mean": float(np.mean(data[:, 7])),
                "hard_contact_frames": int(np.sum(data[:, 3] <= 0.50)),
                "hard_nac_frames": int(np.sum((data[:, 3] <= 0.50) & (data[:, 4] <= 45.0) & (data[:, 5] >= 45.0))),
            })

    write_csv(output / "window_summary.csv", all_window_rows)

    # Rg exchange is assessed on a common physical range.  The compact cutoff
    # is the minimum seen in the old reference set (0.8004 nm), rounded down
    # only for a transparent diagnostic threshold.
    rg_edges = np.arange(0.55, 1.251, 0.025)
    overlap_rows: list[dict[str, object]] = []
    for replica, _ in PATH_SPECS:
        arrays = all_data[replica]
        centers = all_centers[replica]
        for i in range(len(arrays) - 1):
            a = distribution(arrays[i][:, 1], rg_edges)
            b = distribution(arrays[i + 1][:, 1], rg_edges)
            overlap_rows.append({
                "replica": replica,
                "left_window": f"{centers[i]:.3f}",
                "right_window": f"{centers[i + 1]:.3f}",
                "left_frames": len(arrays[i]),
                "right_frames": len(arrays[i + 1]),
                "rg_bhattacharyya": bhattacharyya(a, b),
                "rg_js_divergence_nats": js_divergence(a, b),
                "shared_rg_bins": int(np.sum((a > 0) & (b > 0))),
                "left_compact_fraction": float(np.mean(arrays[i][:, 1] < 0.80)),
                "right_compact_fraction": float(np.mean(arrays[i + 1][:, 1] < 0.80)),
            })
    write_csv(output / "adjacent_rg_overlap.csv", overlap_rows)

    # Compare the old umbrella compact coverage with the MetaD supplement.
    metad_parts = []
    for replica in ("rep1", "rep2", "rep3"):
        source = args.metad_source.resolve() / f"{replica}_source_data.npz"
        if source.is_file():
            metad_parts.append(load_reference_meta(source))
    if not metad_parts:
        raise FileNotFoundError("no MetaD source_data.npz files found")
    metad = np.concatenate(metad_parts)
    old = np.concatenate([x for arrays in all_data.values() for x in arrays])
    comparison = {
        "method": "existing C8 q1 full-reuse distance umbrellas; no new MD",
        "old_umbrella": {
            "frames": int(len(old)),
            "rg_min_nm": float(np.min(old[:, 1])),
            "rg_max_nm": float(np.max(old[:, 1])),
            "rg_median_nm": float(np.median(old[:, 1])),
            "fraction_rg_lt_0_80": float(np.mean(old[:, 1] < 0.80)),
            "fraction_rg_lt_0_75": float(np.mean(old[:, 1] < 0.75)),
            "hard_nac_frames": int(np.sum((old[:, 3] <= 0.50) & (old[:, 4] <= 45.0) & (old[:, 5] >= 45.0))),
        },
        "metad_all_10ns": {
            "frames": int(len(metad)),
            "rg_min_nm": float(np.min(metad[:, 1])),
            "rg_max_nm": float(np.max(metad[:, 1])),
            "rg_median_nm": float(np.median(metad[:, 1])),
            "fraction_rg_lt_0_80": float(np.mean(metad[:, 1] < 0.80)),
            "fraction_rg_lt_0_75": float(np.mean(metad[:, 1] < 0.75)),
            "hard_nac_frames": int(np.sum((metad[:, 3] <= 0.50) & (metad[:, 4] <= 45.0) & (metad[:, 5] >= 45.0))),
        },
        "interpretation": {
            "compact_cutoff_nm": 0.80,
            "population_warning": "raw biased-frame fractions are coverage diagnostics, not equilibrium populations",
            "decision_rule": "if old windows lack compact coverage or adjacent-window Rg overlap is poor, use joint distance×Rg umbrella",
        },
    }
    (output / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # A compact human-readable report keeps the result auditable without
    # converting this diagnostic into a yield estimate.
    overlap = np.asarray([float(r["rg_bhattacharyya"]) for r in overlap_rows])
    lines = [
        "# C8 q1 full-reuse：全局构象交换诊断",
        "",
        "本分析只重算已有距离 umbrella 和已完成 MetaD 的几何量，没有提交或运行新的 MD。",
        "Rg 重叠用于判断距离 umbrella 是否覆盖相同的全局构象；不把有偏帧数转换成平衡人口或产率。",
        "",
        "## 结论",
        "",
        f"旧 full-reuse 距离 umbrella 的 Rg 范围为 {comparison['old_umbrella']['rg_min_nm']:.4f}–{comparison['old_umbrella']['rg_max_nm']:.4f} nm，Rg<0.80 nm 帧比例为 {100*comparison['old_umbrella']['fraction_rg_lt_0_80']:.3f}%。",
        f"MetaD 全10 ns 的 Rg 范围为 {comparison['metad_all_10ns']['rg_min_nm']:.4f}–{comparison['metad_all_10ns']['rg_max_nm']:.4f} nm，Rg<0.80 nm 帧比例为 {100*comparison['metad_all_10ns']['fraction_rg_lt_0_80']:.3f}%。",
        f"相邻距离窗口的 Rg Bhattacharyya 重叠系数范围为 {float(np.min(overlap)):.3f}–{float(np.max(overlap)):.3f}，中位数 {float(np.median(overlap)):.3f}。",
        "",
        "## 证据边界",
        "",
        "若旧窗口几乎没有 Rg<0.80 nm，而 MetaD 可以重复进入该区，则旧距离 umbrella 对全局构象覆盖不足；这支持补做距离×Rg 联合 umbrella。即便相邻窗口 Rg 重叠较好，也只能说明交换/连续性较好，不能证明低 Rg 区域具有更高平衡人口。",
        "本诊断不改变原 full-reuse umbrella 的 q1 概率、C8 R2/R4 结果或 MetaD 报告。",
        "",
        "## 文件",
        "",
        "- `window_summary.csv`：逐重复、逐距离窗口的 Rg/形状/目标对统计。",
        "- `adjacent_rg_overlap.csv`：相邻距离窗口的 Rg 分布重叠。",
        "- `comparison.json`：旧 umbrella 与 MetaD 的覆盖对照。",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
