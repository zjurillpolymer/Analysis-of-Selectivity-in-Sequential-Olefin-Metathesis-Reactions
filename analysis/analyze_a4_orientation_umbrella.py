#!/usr/bin/env python3
"""Analyse the targeted axis-angle umbrella pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

PROJECT = Path(__file__).resolve().parents[1]
EXTRACTED = PROJECT / "results/a4_orientation_umbrella_pilot_extracted"
TEMPERATURE_K = 313.15
R_KJ = 0.00831446261815324
R_KCAL = 0.00198720425864083
RT_KCAL = R_KCAL * TEMPERATURE_K
BETA_KJ = 1.0 / (R_KJ * TEMPERATURE_K)
ANGLE_FORCE_KJ_MOL_RAD2 = 50.0
CENTERS_DEG = np.asarray((0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0))


def read_xvg(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "@")):
            continue
        rows.append([float(value) for value in line.split()])
    return np.asarray(rows, dtype=float)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty rows: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def angle_wham(values_deg: np.ndarray, centers_deg: np.ndarray, window_index: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    counts = np.bincount(window_index, minlength=len(centers_deg)).astype(float)
    if np.any(counts <= 0):
        raise ValueError("every angular window must contribute frames")
    width_deg = 0.5
    edges = np.arange(-0.25, 180.2501, width_deg)
    centers_bins = 0.5 * (edges[:-1] + edges[1:])
    histograms = np.asarray([
        np.histogram(values_deg[window_index == i], bins=edges)[0]
        for i in range(len(centers_deg))
    ], dtype=float)
    total_histogram = np.sum(histograms, axis=0)
    occupied = total_histogram > 0
    centers_bins = centers_bins[occupied]
    total_histogram = total_histogram[occupied]
    values_rad = np.radians(values_deg)
    centers_rad = np.radians(centers_deg)
    bias_bins = BETA_KJ * 0.5 * ANGLE_FORCE_KJ_MOL_RAD2 * (
        centers_bins[None, :] * np.pi / 180.0 - centers_rad[:, None]
    ) ** 2
    log_counts = np.log(counts)
    free = np.zeros(len(centers_deg), dtype=float)
    residual = math.inf
    for iteration in range(1, 50001):
        log_denominator = logsumexp(log_counts[:, None] + free[:, None] - bias_bins, axis=0)
        log_probability_mass = np.log(total_histogram) - log_denominator
        updated = -logsumexp(log_probability_mass[None, :] - bias_bins, axis=1)
        updated -= updated[0]
        residual = float(np.max(np.abs(updated - free)))
        free = 0.5 * free + 0.5 * updated
        if residual < 1.0e-11:
            break
    else:
        raise RuntimeError(f"angle WHAM did not converge: {residual}")
    bias_frames = BETA_KJ * 0.5 * ANGLE_FORCE_KJ_MOL_RAD2 * (
        values_rad[None, :] - centers_rad[:, None]
    ) ** 2
    log_denominator = logsumexp(log_counts[:, None] + free[:, None] - bias_frames, axis=0)
    log_weights = -log_denominator
    log_weights -= logsumexp(log_weights)
    return np.exp(log_weights), free, iteration, residual


def folded_axis_angle(angle_deg: np.ndarray) -> np.ndarray:
    return np.minimum(angle_deg, 180.0 - angle_deg)


def orientation_gate(angle_deg: np.ndarray) -> np.ndarray:
    folded = folded_axis_angle(angle_deg)
    return 1.0 / (1.0 + np.exp((folded - 45.0) / 7.5))


def metrics(values_deg: np.ndarray, distance_nm: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    gate = orientation_gate(values_deg)
    return {
        "mean_axis_angle_deg": float(np.sum(weights * folded_axis_angle(values_deg))),
        "orientation_gate_probability": float(np.sum(weights * gate)),
        "mean_distance_nm": float(np.sum(weights * distance_nm)),
        "distance_contact_probability": float(np.sum(weights * (1.0 / (1.0 + np.exp((distance_nm - 0.50) / 0.05))))),
        "orientation_free_energy_kcal_mol": float(-RT_KCAL * math.log(np.sum(weights * gate))),
        "effective_sample_size": float(1.0 / np.sum(weights ** 2)),
    }


def transitions(values_deg: np.ndarray) -> int:
    state = folded_axis_angle(values_deg) <= 45.0
    return int(np.count_nonzero(state[1:] != state[:-1]))


def analyse_path(task_dirs: list[Path], output: Path) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    tasks = [task_dir / "task.json" for task_dir in task_dirs]
    tasks.sort(key=lambda task: float(json.loads(task.read_text(encoding="utf-8"))["angle_center_deg"]))
    if len(tasks) != 7:
        raise ValueError(f"expected seven task metadata files, found {len(tasks)}")
    metadata = [json.loads(task.read_text(encoding="utf-8")) for task in tasks]
    metadata.sort(key=lambda row: float(row["angle_center_deg"]))
    values_all, distance_all, time_all, window_all = [], [], [], []
    raw_rows = []
    for window_index, row in enumerate(metadata):
        task_dir = tasks[window_index].parent
        pull = read_xvg(task_dir / "production_pullx.xvg")
        if pull.shape[1] < 3:
            raise ValueError(f"{task_dir}: expected time, distance, angle columns")
        time, distance, angle = pull[:, 0], pull[:, 1], pull[:, 2]
        values_all.append(angle)
        distance_all.append(distance)
        time_all.append(time)
        window_all.append(np.full(len(angle), window_index, dtype=int))
        raw_rows.append({
            "pathway": row["name"], "system": row["system"], "route": row["route"],
            "angle_center_deg": row["angle_center_deg"], "frames": len(angle),
            "observed_angle_mean_deg": float(np.mean(angle)),
            "observed_angle_std_deg": float(np.std(angle)),
            "observed_distance_mean_nm": float(np.mean(distance)),
            "observed_distance_std_nm": float(np.std(distance)),
            "transitions_gate": transitions(angle),
        })
    values = np.concatenate(values_all)
    distance = np.concatenate(distance_all)
    time = np.concatenate(time_all)
    window_index = np.concatenate(window_all)
    centers = CENTERS_DEG.copy()
    weights, free, iterations, residual = angle_wham(values, centers, window_index)
    full = metrics(values, distance, weights)
    half_metrics = {}
    for name, selector in (("first_half", time <= 500.0), ("second_half", time > 500.0)):
        half_weights, _, _, _ = angle_wham(values[selector], centers, window_index[selector])
        half_metrics[name] = metrics(values[selector], distance[selector], half_weights)
    # Adjacent-window overlap is the Bhattacharyya coefficient of normalized
    # observed histograms. It is a direct diagnostic, not a PMF estimate.
    bins = np.linspace(0.0, 180.0, 73)
    histograms = []
    for i in range(len(centers)):
        hist, _ = np.histogram(values[window_index == i], bins=bins, density=True)
        histograms.append(hist / np.sum(hist))
    overlaps = [
        float(np.sum(np.sqrt(histograms[i] * histograms[i + 1])))
        for i in range(len(centers) - 1)
    ]
    pathway_row = {
        "pathway": metadata[0]["name"], "system": metadata[0]["system"], "route": metadata[0]["route"],
        "windows": len(centers), "frames": len(values),
        "wham_iterations": iterations, "wham_residual": residual,
        "min_adjacent_overlap": min(overlaps), "mean_adjacent_overlap": float(np.mean(overlaps)),
        "transitions_gate_total": int(sum(r["transitions_gate"] for r in raw_rows)),
        **full,
        **{f"{half}_{key}": value for half, vals in half_metrics.items() for key, value in vals.items()},
    }
    pmf_bins = np.linspace(0.0, 180.0, 181)
    probability, _ = np.histogram(values, bins=pmf_bins, weights=weights)
    probability /= np.sum(probability)
    pmf = np.full(len(probability), np.nan)
    positive = probability > 0
    pmf[positive] = -RT_KCAL * np.log(probability[positive])
    pmf[positive] -= np.nanmin(pmf)
    pmf_rows = [
        {"pathway": metadata[0]["name"], "angle_bin_deg": float(0.5 * (pmf_bins[i] + pmf_bins[i + 1])),
         "probability": float(probability[i]), "pmf_kcal_mol": float(pmf[i])}
        for i in range(len(probability))
    ]
    diagnostics = {
        "pathway": metadata[0]["name"], "system": metadata[0]["system"], "route": metadata[0]["route"],
        "centers_deg": centers.tolist(), "free_energy_dimensionless": free.tolist(),
        "adjacent_overlap": overlaps, "pathway_summary": pathway_row,
        "half_metrics": half_metrics,
    }
    return pathway_row, pmf_rows, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=EXTRACTED)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/a4_orientation_umbrella_pilot_analysis")
    args = parser.parse_args()
    input_root, output = args.input.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    pathways = {}
    for task_meta in sorted(input_root.glob("*/task.json")):
        row = json.loads(task_meta.read_text(encoding="utf-8"))
        pathways.setdefault(row["name"], []).append(task_meta.parent)
    if len(pathways) != 4:
        raise RuntimeError(f"expected four pathways, found {len(pathways)}")
    summary_rows, pmf_rows, diagnostics = [], [], {}
    for name, task_dirs in sorted(pathways.items()):
        summary, pmf, detail = analyse_path(task_dirs, output)
        summary_rows.append(summary)
        pmf_rows.extend(pmf)
        diagnostics[name] = detail
        print(f"analysed={name} frames={summary['frames']} overlap={summary['min_adjacent_overlap']:.3f}", flush=True)
    write_csv(output / "pathway_orientation_summary.csv", summary_rows)
    write_csv(output / "axis_angle_pmf.csv", pmf_rows)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    route_rows = []
    for system in ("C5", "C8"):
        for route in ("reuse", "switch"):
            row = next(r for r in summary_rows if r["system"] == system and r["route"] == route)
            first = float(row["first_half_orientation_gate_probability"])
            second = float(row["second_half_orientation_gate_probability"])
            full = float(row["orientation_gate_probability"])
            route_rows.append({
                "system": system, "route": route,
                "orientation_gate_probability": full,
                "first_half_orientation_gate_probability": first,
                "second_half_orientation_gate_probability": second,
                "half_difference": second - first,
                "orientation_free_energy_kcal_mol": row["orientation_free_energy_kcal_mol"],
                "reuse_switch_ddG_kcal_mol": 0.0,
                "min_adjacent_overlap": row["min_adjacent_overlap"],
                "effective_sample_size": row["effective_sample_size"],
            })
    for system in ("C5", "C8"):
        reuse = next(r for r in route_rows if r["system"] == system and r["route"] == "reuse")
        switch = next(r for r in route_rows if r["system"] == system and r["route"] == "switch")
        reuse_prob = float(reuse["orientation_gate_probability"])
        switch_prob = float(switch["orientation_gate_probability"])
        ddg = -RT_KCAL * math.log(reuse_prob / switch_prob)
        first_ddg = -RT_KCAL * math.log(
            float(reuse["first_half_orientation_gate_probability"])
            / float(switch["first_half_orientation_gate_probability"])
        )
        second_ddg = -RT_KCAL * math.log(
            float(reuse["second_half_orientation_gate_probability"])
            / float(switch["second_half_orientation_gate_probability"])
        )
        for row in route_rows:
            if row["system"] == system:
                row["reuse_switch_ddG_kcal_mol"] = ddg
                row["first_half_reuse_switch_ddG_kcal_mol"] = first_ddg
                row["second_half_reuse_switch_ddG_kcal_mol"] = second_ddg
    write_csv(output / "route_orientation_summary.csv", route_rows)

    report = [
        "# A4 C5/C8 取向 umbrella 采样充分性验证",
        "",
        "本批次不是完整距离×取向二维网格，而是在距离 0.50 nm 接触区固定距离后，对两个乙烯基键向量的夹角做 0–180° 的七个 umbrella 窗口。每个窗口为 100 ps 平衡 + 1 ns 生产。取向门控仍按 folded axis angle ≤45° 的平滑版本计算。",
        "",
        "## 取向门控与窗口诊断",
        "",
        "| 体系 | 路径 | gate 概率 | 前半段 | 后半段 | 差值 | 最小相邻重叠 | ESS | 转换次数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in route_rows:
        report.append(
            f"| {row['system']} | {row['route']} | {float(row['orientation_gate_probability']):.4f} | "
            f"{float(row['first_half_orientation_gate_probability']):.4f} | "
            f"{float(row['second_half_orientation_gate_probability']):.4f} | "
            f"{float(row['half_difference']):+.4f} | "
            f"{float(row['min_adjacent_overlap']):.3f} | "
            f"{float(row['effective_sample_size']):.1f} | "
            f"{next(int(r['transitions_gate_total']) for r in summary_rows if r['system']==row['system'] and r['route']==row['route'])} |"
        )
    report += [
        "",
        "## reuse–switch 取向自由能差",
        "",
        "| 体系 | full | first half | second half |",
        "|---|---:|---:|---:|",
    ]
    for system in ("C5", "C8"):
        row = next(r for r in route_rows if r["system"] == system and r["route"] == "reuse")
        report.append(
            f"| {system} | {float(row['reuse_switch_ddG_kcal_mol']):+.3f} | "
            f"{float(row['first_half_reuse_switch_ddG_kcal_mol']):+.3f} | "
            f"{float(row['second_half_reuse_switch_ddG_kcal_mol']):+.3f} kcal/mol |"
        )
    report += [
        "",
        "## 判据",
        "",
        "- 相邻窗口重叠用于判断角度 PMF 是否能被连续连接；它不是独立样本数。",
        "- 前后半段差异用于检查取向门控是否漂移；它不是置信区间。",
        "- ESS 仍未校正时间自相关；如果 ESS 很低，即使 WHAM 数值收敛，也不能把取向项视为稳健。",
        "- 本批次只直接约束 axis angle，没有单独偏置 side angle。因此它可以检验主要轴向取向是否因距离 umbrella 而严重欠采样，但不能单独证明完整 axis×side NAC 取向已经充分采样。",
    ]
    (output / "ORIENTATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps({
        "method": "one-dimensional axis-angle umbrella at fixed 0.50 nm distance",
        "temperature_K": TEMPERATURE_K,
        "pathways": summary_rows,
        "routes": route_rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "pathways": len(summary_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
