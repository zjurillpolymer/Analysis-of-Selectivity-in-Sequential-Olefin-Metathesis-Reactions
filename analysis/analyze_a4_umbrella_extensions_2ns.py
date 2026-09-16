#!/usr/bin/env python3
"""Analyse the 500 ps + 1.5 ns continuation of the A4 C5/C8 umbrellas.

The continuation jobs extend the existing production trajectories from 500 to
2000 ps.  This script keeps the original cached geometry for the old segment,
extracts the reactive-pair geometry from each continuation XTC, removes the
duplicated 500 ps frame, and applies the same distance-bias WHAM/MBAR-equivalent
weights plus frame-resolved orientation gate used in the validated analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
if str(PROJECT / "analysis") not in sys.path:
    sys.path.insert(0, str(PROJECT / "analysis"))

from analysis.analyze_a4_m_c6_c7_q1_umbrella_2d import prepare_topology  # noqa: E402
from analysis.analyze_a4_q2_umbrella_2d_reweight import (  # noqa: E402
    RT_KCAL,
    mbar_weights,
    orientation_geometry_matching_pull,
    parse_window_center,
    read_pullx,
    read_numeric_xvg,
    soft_lower,
    soft_upper,
    weighted_metrics,
    weighted_surface,
)


MULTIPLICITY = {"full_reuse": 1, "partial_reuse": 8, "full_switch": 4}
Q1_CATEGORIES = ("full_reuse", "partial_reuse", "full_switch")
Q2_ROUTE = ("reuse", "switch")

Q1_CACHE = PROJECT / "results/a4_m_c5_c8_q1_umbrella_2d_3rep_analysis/cache"
Q2_CACHE = PROJECT / "results/a4_q2_umbrella_2d_analysis/final/cache"
EXT_ROOT = PROJECT / "results/a4_umbrella_extensions_fix1_extracted"


@dataclass
class Result:
    metadata: dict[str, object]
    centers: np.ndarray
    time: np.ndarray
    distance: np.ndarray
    orientation: np.ndarray
    weights: np.ndarray
    metrics: dict[str, float]
    half_metrics: dict[str, dict[str, float]]
    surface_probability: np.ndarray
    surface_pmf: np.ndarray
    windows: int
    frames: int
    effective_sample_size: float
    mbar_iterations: int
    mbar_residual: float
    extension_start_ps: float
    extension_end_ps: float


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run(command: Sequence[str], stdin: str | None = None) -> str:
    completed = subprocess.run(
        list(command), input=stdin, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if completed.returncode:
        tail = "\n".join(completed.stdout.splitlines()[-30:])
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{tail}")
    return completed.stdout


def metadata_paths(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for metadata_path in sorted(root.glob("*/extracted/pathway/pathway.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        output[str(metadata["pathway"])] = metadata_path.parent
    return output


def discover_old_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    paths.update(metadata_paths(PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_download"))
    paths.update(metadata_paths(PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_rep3_download"))
    # Prefer the resumed C8 reuse path and the complete partial C8 switch path.
    candidates: dict[str, list[Path]] = {}
    for root in (
        PROJECT / "results/a4_q2_umbrella_bohrium/download",
        PROJECT / "results/a4_q2_umbrella_bohrium/resumed_download",
        PROJECT / "results/a4_q2_umbrella_bohrium/partial",
    ):
        for metadata_path in root.glob("*/pathway/pathway.json"):
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            candidates.setdefault(str(metadata["pathway"]), []).append(metadata_path.parent)
    for name, values in candidates.items():
        # Select the source with the largest number of complete pull-x windows.
        # This resolves the duplicated C8 reuse download deterministically.
        def score(path: Path) -> tuple[int, int]:
            windows = list((path / "windows").glob("window_*/production_pullx.xvg"))
            complete = 0
            for pullx in windows:
                data = read_numeric_xvg(pullx)
                if len(data) and float(data[-1, 0]) >= 499.9:
                    complete += 1
            preferred = int("resumed_download" in str(path)) + int("partial" in str(path))
            return complete, preferred
        paths[name] = max(values, key=score)
    return paths


def discover_extensions() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for metadata_path in sorted(EXT_ROOT.glob("*/pathway/pathway.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths[str(metadata["pathway"])] = metadata_path.parent
    return paths


def window_center(path: Path) -> float:
    match = re.search(r"window_\d+_([0-9.]+)nm$", path.name)
    if not match:
        raise ValueError(f"cannot parse window center: {path}")
    return float(match.group(1))


def filtered_rows(path: Path, start: float, end: float, strict_start: bool = False) -> np.ndarray:
    values = read_numeric_xvg(path)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"unexpected XVG shape {path}: {values.shape}")
    if strict_start:
        keep = (values[:, 0] > start + 1.0e-6) & (values[:, 0] <= end + 1.0e-6)
    else:
        keep = (values[:, 0] >= start - 1.0e-6) & (values[:, 0] <= end + 1.0e-6)
    return values[keep]


def cache_geometry(
    xtc: Path, topology: Path, index: Path, coordinates: Path, box: Path,
    gmx: str,
) -> None:
    if coordinates.is_file() and box.is_file():
        return
    coordinates.parent.mkdir(parents=True, exist_ok=True)
    run([
        gmx, "traj", "-f", str(xtc), "-s", str(topology), "-n", str(index),
        "-ox", str(coordinates), "-ob", str(box), "-b", "500", "-e", "2000",
        "-x", "-y", "-z",
    ], stdin="Reactive_pair\n")


def choose_cache(alias: str, window: str) -> Path:
    for root in (Q1_CACHE, Q2_CACHE):
        candidate = root / alias / window / "reactive_pair_coordinates_nm.xvg"
        if candidate.is_file():
            return candidate.parent
    raise FileNotFoundError(f"old geometry cache missing for {alias}/{window}")


def analyse_path(
    alias: str,
    old_path: Path,
    extension_path: Path,
    topology_root: Path,
    output: Path,
    gmx: str,
) -> Result:
    metadata = json.loads((old_path / "pathway.json").read_text(encoding="utf-8"))
    begin = float(metadata.get("wham_begin_ps", 100.0))
    old_end = min(500.0, min(
        float(read_numeric_xvg(p)[-1, 0])
        for p in (old_path / "windows").glob("window_*/production_pullx.xvg")
        if len(read_numeric_xvg(p))
    ))
    topology = topology_root / alias / "solute_topology.gro"
    index = topology_root / alias / "reactive_pair.ndx"
    complete_windows: list[tuple[Path, Path]] = []
    for old_window in sorted((old_path / "windows").glob("window_*"), key=window_center):
        extension_window = extension_path / "windows" / old_window.name
        if not (old_window / "production_pullx.xvg").is_file():
            continue
        if not (extension_window / "extension.xtc").is_file() or not (extension_window / "extension_pullx.xvg").is_file():
            continue
        ext_pull = read_numeric_xvg(extension_window / "extension_pullx.xvg")
        if len(ext_pull) and float(ext_pull[-1, 0]) >= 1999.9:
            complete_windows.append((old_window, extension_window))
    if len(complete_windows) < 20:
        raise ValueError(f"{alias}: only {len(complete_windows)} combined windows")
    centers = np.asarray([window_center(old) for old, _ in complete_windows], dtype=float)

    all_time: list[np.ndarray] = []
    all_distance: list[np.ndarray] = []
    all_orientation: list[np.ndarray] = []
    all_window: list[np.ndarray] = []
    cache_root = output / "cache" / alias
    for window_index, (old_window, extension_window) in enumerate(complete_windows):
        old_cache = choose_cache(alias, old_window.name)
        old_coordinates = read_numeric_xvg(old_cache / "reactive_pair_coordinates_nm.xvg")
        old_box = read_numeric_xvg(old_cache / "box_vectors_nm.xvg")
        old_pull = filtered_rows(old_window / "production_pullx.xvg", begin, old_end)
        old_keep = np.isin(np.round(old_coordinates[:, 0], 5), np.round(old_pull[:, 0], 5))
        old_coordinates = old_coordinates[old_keep]
        old_box = old_box[np.isin(np.round(old_box[:, 0], 5), np.round(old_pull[:, 0], 5))]
        if len(old_pull) != len(old_coordinates) or len(old_pull) != len(old_box):
            raise ValueError(f"{alias}/{old_window.name}: old time mismatch")

        ext_cache = cache_root / extension_window.name
        ext_coords_path = ext_cache / "extension_reactive_pair_coordinates_nm.xvg"
        ext_box_path = ext_cache / "extension_box_vectors_nm.xvg"
        cache_geometry(
            extension_window / "extension.xtc", topology, index,
            ext_coords_path, ext_box_path, gmx,
        )
        ext_coordinates = read_numeric_xvg(ext_coords_path)
        ext_box = read_numeric_xvg(ext_box_path)
        ext_pull = filtered_rows(extension_window / "extension_pullx.xvg", old_end, 2000.0, strict_start=True)
        ext_keep = np.isin(np.round(ext_coordinates[:, 0], 5), np.round(ext_pull[:, 0], 5))
        ext_coordinates = ext_coordinates[ext_keep]
        ext_box = ext_box[np.isin(np.round(ext_box[:, 0], 5), np.round(ext_pull[:, 0], 5))]
        if len(ext_pull) != len(ext_coordinates) or len(ext_pull) != len(ext_box):
            raise ValueError(f"{alias}/{extension_window.name}: extension time mismatch")

        pull = np.vstack((old_pull[:, :2], ext_pull[:, :2]))
        coords = np.vstack((old_coordinates, ext_coordinates))
        boxes = np.vstack((old_box, ext_box))
        distance, axis, side, physical = orientation_geometry_matching_pull(coords, boxes, pull[:, 1])
        if not np.all(physical):
            raise ValueError(f"{alias}/{old_window.name}: nonphysical vinyl geometry")
        orientation = soft_upper(axis, 45.0, 7.5) * soft_lower(side, 45.0, 7.5)
        all_time.append(pull[:, 0])
        all_distance.append(pull[:, 1])
        all_orientation.append(orientation)
        all_window.append(np.full(len(pull), window_index, dtype=int))

    time = np.concatenate(all_time)
    distance = np.concatenate(all_distance)
    orientation = np.concatenate(all_orientation)
    window_index = np.concatenate(all_window)
    weights, _, iterations, residual = mbar_weights(distance, centers, window_index)
    metrics = weighted_metrics(distance, orientation, weights)
    midpoint = 0.5 * (begin + 2000.0)
    half_metrics: dict[str, dict[str, float]] = {}
    for name, selector in (
        ("first_half", time <= midpoint + 1.0e-6),
        ("second_half", time > midpoint + 1.0e-6),
    ):
        half_weights, _, _, _ = mbar_weights(distance[selector], centers, window_index[selector])
        half_metrics[name] = weighted_metrics(distance[selector], orientation[selector], half_weights)
    probability, pmf = weighted_surface(distance, orientation, weights)
    return Result(
        metadata=metadata, centers=centers, time=time, distance=distance,
        orientation=orientation, weights=weights, metrics=metrics,
        half_metrics=half_metrics, surface_probability=probability,
        surface_pmf=pmf, windows=len(centers), frames=len(time),
        effective_sample_size=float(1.0 / np.sum(weights ** 2)),
        mbar_iterations=iterations, mbar_residual=residual,
        extension_start_ps=float(old_end), extension_end_ps=float(np.max(time)),
    )


def aggregate(results: Sequence[Result]) -> dict[str, object]:
    def one(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
        contact = float(np.mean([v["soft_contact_probability"] for v in values]))
        nac = float(np.mean([v["soft_nac_probability"] for v in values]))
        orient = nac / contact
        return {
            "soft_contact_probability": contact,
            "conditional_orientation": orient,
            "soft_nac_probability": nac,
            "contact_free_energy_kcal_mol": -RT_KCAL * math.log(contact),
            "orientation_free_energy_kcal_mol": -RT_KCAL * math.log(orient),
            "nac_free_energy_kcal_mol": -RT_KCAL * math.log(nac),
        }
    return {
        "metrics": one([r.metrics for r in results]),
        "half_metrics": {
            h: one([r.half_metrics[h] for r in results])
            for h in ("first_half", "second_half")
        },
        "surface_probability": np.mean([r.surface_probability for r in results], axis=0),
    }


def ddg(left: Mapping[str, float], right: Mapping[str, float], key: str) -> float:
    return -RT_KCAL * math.log(float(left[key]) / float(right[key]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/a4_umbrella_extensions_fix1_analysis")
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    topology_root = output / "reactive_pair_topologies"
    old_paths = discover_old_paths()
    extension_paths = discover_extensions()
    aliases = sorted(set(old_paths) & set(extension_paths))
    expected = 23
    if len(aliases) != expected:
        raise RuntimeError(f"expected {expected} continuation pathways, found {len(aliases)}")
    for alias in aliases:
        prepare_topology(old_paths[alias], topology_root)

    results: dict[str, Result] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                analyse_path, alias, old_paths[alias], extension_paths[alias],
                topology_root, output, args.gmx,
            ): alias for alias in aliases
        }
        for future in as_completed(futures):
            alias = futures[future]
            try:
                results[alias] = future.result()
                result = results[alias]
                print(f"analysed={alias} windows={result.windows} frames={result.frames}", flush=True)
            except Exception as exc:  # keep all failures visible in one run
                failures[alias] = f"{type(exc).__name__}: {exc}"
                print(f"FAILED={alias} {failures[alias]}", flush=True)
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"{len(failures)} pathways failed; see {output / 'failures.json'}")

    pathway_rows: list[dict[str, object]] = []
    for alias in aliases:
        result = results[alias]
        pathway_rows.append({
            "pathway": alias,
            "system": result.metadata["system"].replace("A4_m_", ""),
            "route_family": result.metadata["route_family"],
            "stage": result.metadata.get("stage", "q2"),
            "replica": result.metadata.get("replica", ""),
            "windows": result.windows,
            "frames": result.frames,
            "extension_start_ps": result.extension_start_ps,
            "extension_end_ps": result.extension_end_ps,
            "effective_sample_size": result.effective_sample_size,
            "mbar_iterations": result.mbar_iterations,
            "mbar_residual": result.mbar_residual,
            **result.metrics,
            **{f"{h}_{k}": v for h, values in result.half_metrics.items() for k, v in values.items()},
        })
    write_csv(output / "pathway_reweight_summary.csv", pathway_rows)

    q1_results: dict[str, dict[str, list[Result]]] = {"C5": {c: [] for c in Q1_CATEGORIES}, "C8": {c: [] for c in Q1_CATEGORIES}}
    q2_results: dict[str, dict[str, list[Result]]] = {"C5": {r: [] for r in Q2_ROUTE}, "C8": {r: [] for r in Q2_ROUTE}}
    for alias, result in results.items():
        system = str(result.metadata["system"]).replace("A4_m_", "")
        route = str(result.metadata["route_family"])
        stage = str(result.metadata.get("stage", "q2"))
        if stage == "q1":
            q1_results[system][route].append(result)
        else:
            q2_results[system][route].append(result)

    category_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    yield_rows: list[dict[str, object]] = []
    half_yield_rows: list[dict[str, object]] = []
    q2_rows: list[dict[str, object]] = []
    for system in ("C5", "C8"):
        q1_agg = {category: aggregate(q1_results[system][category]) for category in Q1_CATEGORIES}
        for category in Q1_CATEGORIES:
            pooled = q1_agg[category]["metrics"]
            category_rows.append({
                "system": system, "stage": "q1", "category": category,
                "pathways": ";".join(r.metadata["pathway"] for r in q1_results[system][category]),
                **pooled,
                **{f"{h}_{k}": v for h, values in q1_agg[category]["half_metrics"].items() for k, v in values.items()},
            })
        raw = {c: MULTIPLICITY[c] * q1_agg[c]["metrics"]["soft_nac_probability"] for c in Q1_CATEGORIES}
        total = sum(raw.values())
        q1_prob = {c: raw[c] / total for c in Q1_CATEGORIES}
        for category in Q1_CATEGORIES:
            branch_rows.append({
                "system": system, "category": category, "probability": q1_prob[category],
                "soft_nac_probability": q1_agg[category]["metrics"]["soft_nac_probability"],
                "candidate_multiplicity": MULTIPLICITY[category],
            })

        q2_agg = {route: aggregate(q2_results[system][route]) for route in Q2_ROUTE}
        for route in Q2_ROUTE:
            pooled = q2_agg[route]["metrics"]
            q2_rows.append({
                "system": system, "stage": "q2", "route": route,
                "pathways": ";".join(r.metadata["pathway"] for r in q2_results[system][route]),
                **pooled,
                **{f"{h}_{k}": v for h, values in q2_agg[route]["half_metrics"].items() for k, v in values.items()},
            })
        q2_raw = {
            "R2x2": 2.0 * q2_agg["reuse"]["metrics"]["soft_nac_probability"],
            "R4": 4.0 * q2_agg["switch"]["metrics"]["soft_nac_probability"],
        }
        q2_total = sum(q2_raw.values())
        q2_prob = {k: value / q2_total for k, value in q2_raw.items()}
        r2 = q1_prob["full_reuse"] + q1_prob["full_switch"] * q2_prob["R2x2"]
        yield_rows.append({
            "system": system,
            "q1_full_reuse": q1_prob["full_reuse"],
            "q1_partial_reuse": q1_prob["partial_reuse"],
            "q1_full_switch": q1_prob["full_switch"],
            "q2_R2x2": q2_prob["R2x2"], "q2_R4": q2_prob["R4"],
            "R2_mass_fraction": r2, "R4_mass_fraction": 1.0 - r2,
        })
        for half in ("first_half", "second_half"):
            half_q1 = {
                category: q1_agg[category]["half_metrics"][half]["soft_nac_probability"]
                for category in Q1_CATEGORIES
            }
            half_raw = {c: MULTIPLICITY[c] * half_q1[c] for c in Q1_CATEGORIES}
            half_total = sum(half_raw.values())
            half_prob = {c: half_raw[c] / half_total for c in Q1_CATEGORIES}
            half_q2_reuse = q2_agg["reuse"]["half_metrics"][half]["soft_nac_probability"]
            half_q2_switch = q2_agg["switch"]["half_metrics"][half]["soft_nac_probability"]
            half_q2_r2 = (2.0 * half_q2_reuse) / (2.0 * half_q2_reuse + 4.0 * half_q2_switch)
            half_r2 = half_prob["full_reuse"] + half_prob["full_switch"] * half_q2_r2
            half_yield_rows.append({
                "system": system, "half": half,
                "q1_full_reuse": half_prob["full_reuse"],
                "q1_partial_reuse": half_prob["partial_reuse"],
                "q1_full_switch": half_prob["full_switch"],
                "q2_R2x2": half_q2_r2, "q2_R4": 1.0 - half_q2_r2,
                "R2_mass_fraction": half_r2, "R4_mass_fraction": 1.0 - half_r2,
            })
    write_csv(output / "q1_category_reweight_summary.csv", category_rows)
    write_csv(output / "q1_branch_probabilities.csv", branch_rows)
    write_csv(output / "q2_route_reweight_summary.csv", q2_rows)
    write_csv(output / "q1_q2_yield_prediction.csv", yield_rows)
    write_csv(output / "q1_q2_yield_half_diagnostics.csv", half_yield_rows)

    previous_q1 = PROJECT / "results/a4_m_c5_c8_q1_umbrella_2d_3rep_analysis/q1_2d_yield_prediction.csv"
    previous_q2 = PROJECT / "results/a4_q2_umbrella_2d_analysis/final/group_reweight_summary.csv"
    old_yields = {row["system"]: row for row in csv.DictReader(previous_q1.open())} if previous_q1.is_file() else {}
    old_q2 = {row["group"]: row for row in csv.DictReader(previous_q2.open())} if previous_q2.is_file() else {}
    experiment = {"C5": (75.0 / 99.0), "C6": 39.0 / 94.0, "C7": 20.0 / 91.0, "C8": 13.0 / 76.0}
    report = [
        "# A4-m-C5/C8 umbrella 延长至 2 ns 的完整分析",
        "",
        "本报告使用 23 条 Bohrium 延长任务的全部成功输出：18 条 q1 路径和 5 条 q2 路径，共 573 个完整窗口。每个窗口把原 0.5 ns production 段与 0.5–2.0 ns continuation 段拼接，重复的 500 ps 帧只保留一次。q1 的 18 条路径和 q2 的 5 条路径均采用与此前一致的距离偏置 WHAM/MBAR 等价重加权，并在逐帧层面加入乙烯基轴向/侧向取向门控。这里的“二维”仍指距离偏置加取向后处理，不是同时对两个坐标施加独立伞形偏置。",
        "",
        "## q1 分支概率",
        "",
        "| 体系 | full reuse | partial reuse | full switch |",
        "|---|---:|---:|---:|",
    ]
    for system in ("C5", "C8"):
        rows = {r["category"]: r for r in branch_rows if r["system"] == system}
        report.append(f"| {system} | {float(rows['full_reuse']['probability']):.4f} | {float(rows['partial_reuse']['probability']):.4f} | {float(rows['full_switch']['probability']):.4f} |")
    report += [
        "",
        "## q2 路由概率（候选多重性已计入）",
        "",
        "| 体系 | reuse→2×R2 | switch→R4 |",
        "|---|---:|---:|",
    ]
    for row in yield_rows:
        report.append(f"| {row['system']} | {float(row['q2_R2x2']):.4f} | {float(row['q2_R4']):.4f} |")
    report += [
        "",
        "## 最终 R2/R4 传播",
        "",
        "| 体系 | 模拟 R2 | 模拟 R4 | 实验 R2 | 模拟−实验 (R2) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in yield_rows:
        system = str(row["system"])
        exp = experiment.get(system, float("nan"))
        pred = float(row["R2_mass_fraction"])
        report.append(f"| {system} | {pred:.4f} | {1.0-pred:.4f} | {exp:.4f} | {pred-exp:+.4f} |")
    report += [
        "",
        "## 延长前后变化",
        "",
        "| 体系 | 旧模拟 R2 | 2 ns 模拟 R2 | 变化 |",
        "|---|---:|---:|---:|",
    ]
    for row in yield_rows:
        system = str(row["system"])
        old = float(old_yields[system]["R2_mass_fraction"]) if system in old_yields else float("nan")
        new = float(row["R2_mass_fraction"])
        report.append(f"| {system} | {old:.4f} | {new:.4f} | {new-old:+.4f} |")
    report += [
        "",
        "## 前后半段稳定性诊断",
        "",
        "| 体系 | first-half R2 | second-half R2 | 差值 |",
        "|---|---:|---:|---:|",
    ]
    for system in ("C5", "C8"):
        halves = {r["half"]: r for r in half_yield_rows if r["system"] == system}
        first = float(halves["first_half"]["R2_mass_fraction"])
        second = float(halves["second_half"]["R2_mass_fraction"])
        report.append(f"| {system} | {first:.4f} | {second:.4f} | {second-first:+.4f} |")
    for system in ("C5", "C8"):
        ess = [float(r["effective_sample_size"]) for r in pathway_rows if r["system"] == system]
        report.append(f"- {system} 路径权重 ESS 范围为 {min(ess):.1f}–{max(ess):.1f}；ESS 未校正时间自相关。")
    report += [
        "",
        "## 解释与边界",
        "",
        "- q1 概率先按 full reuse:partial reuse:full switch 的候选多重性 1:8:4 归一化；q2 的 reuse 分支按一次生成 2 个 R2 计权，switch 分支按 1 个 R4 计权。",
        "- R2 传播公式为 `R2 = q1_full_reuse + q1_full_switch × q2_R2x2`；partial reuse 在本模型中直接进入 R4。",
        "- 延长段增加了每个窗口约 3 倍的时间样本，因此前后半段差异和权重 ESS 是判断稳定性的重点；它们不是独立重复的统计置信区间。",
        "- C8 switch 只有 24 个完整窗口（最高中心 2.65 nm），因为 2.75 nm 终端窗口在原任务中不完整；该窗口没有被补成数据。",
        "- 这些结果是构象可及性/soft-NAC 的自由能代理，不是化学活化自由能，也没有把分子间副反应、小分子产物或间位臂纳入模型。",
    ]
    (output / "EXTENDED_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = {
        "pathway_count": len(results), "window_count": int(sum(r.windows for r in results.values())),
        "frame_count": int(sum(r.frames for r in results.values())),
        "method": "500 ps original + 1500 ps continuation; distance-bias WHAM/MBAR-equivalent with frame-resolved orientation gate",
        "yield_predictions": yield_rows, "half_yield_diagnostics": half_yield_rows,
        "q1_branch_probabilities": branch_rows,
        "q1_categories": category_rows, "q2_routes": q2_rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "pathways": len(results), "windows": int(sum(r.windows for r in results.values())), "yield_predictions": yield_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
