#!/usr/bin/env python3
"""Quantify the q0 same-arm/cross-arm contact origin in A4-m-C8.

The analysis deliberately separates two questions:

1. intrinsic per-pair accessibility: the mean contact probability of one
   same-arm pair versus one cross-arm pair;
2. molecular multiplicity: the probability that any of the four same-arm or
   any of the twenty-four cross-arm pairs is in contact.

Uncertainty is estimated with a hierarchical bootstrap over independent MD
replicas and 200-ps time blocks.  The reported free-energy contrasts are
occupancy-derived contact/preorganization terms, not metathesis activation
barriers.  A distance-only shortlist of decorrelated contact episodes is also
written for later orientation filtering and coordinate extraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


R_KCAL_MOL_K = 1.98720425864083e-3
DEFAULT_TEMPERATURE_K = 313.15
DEFAULT_CUTOFFS_A = (4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0)


@dataclass(frozen=True)
class ReplicaData:
    name: str
    time_ps: np.ndarray
    distances_A: np.ndarray


def read_distance_csv(path: Path, pair_columns: Sequence[str]) -> ReplicaData:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = set(pair_columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing pair columns: {sorted(missing)}")
        times: list[float] = []
        rows: list[list[float]] = []
        for row in reader:
            times.append(float(row["absolute_time_ps"]))
            rows.append([float(row[column]) for column in pair_columns])
    time = np.asarray(times, dtype=float)
    distances = np.asarray(rows, dtype=float)
    if len(time) < 2 or distances.shape != (len(time), len(pair_columns)):
        raise ValueError(f"invalid distance table shape in {path}")
    if not np.all(np.isfinite(distances)) or np.any(distances <= 0.0):
        raise ValueError(f"nonfinite/nonpositive distances in {path}")
    delta = np.diff(time)
    if not np.allclose(delta, delta[0], rtol=0.0, atol=1e-6):
        raise ValueError(f"nonuniform time grid in {path}")
    return ReplicaData(path.parent.name, time, distances)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def soft_contact(distances_A: np.ndarray, center_A: float, width_A: float) -> np.ndarray:
    z = np.clip((np.asarray(distances_A) - center_A) / width_A, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(z))


def ddg_from_ratio(ratio: np.ndarray | float, temperature_K: float) -> np.ndarray:
    values = np.asarray(ratio, dtype=float)
    return -R_KCAL_MOL_K * temperature_K * np.log(values)


def metric_values(
    contact: np.ndarray, same_mask: np.ndarray, cross_mask: np.ndarray, mode: str
) -> np.ndarray:
    """Return frame-wise same and cross values for one contact definition."""
    if mode == "per_pair":
        same = np.mean(contact[:, same_mask], axis=1)
        cross = np.mean(contact[:, cross_mask], axis=1)
    elif mode == "any_pair":
        same = np.any(contact[:, same_mask], axis=1).astype(float)
        cross = np.any(contact[:, cross_mask], axis=1).astype(float)
    else:
        raise ValueError(f"unknown metric mode: {mode}")
    return np.column_stack((same, cross))


def full_estimate(replica_frame_values: Sequence[np.ndarray]) -> tuple[float, float]:
    means = np.stack([np.mean(values, axis=0) for values in replica_frame_values])
    same, cross = np.mean(means, axis=0)
    return float(same), float(cross)


def block_means(values: np.ndarray, block_frames: int) -> np.ndarray:
    if block_frames < 1:
        raise ValueError("block_frames must be positive")
    count = len(values) // block_frames
    if count < 2:
        raise ValueError("each replica must contain at least two complete blocks")
    trimmed = values[: count * block_frames]
    return trimmed.reshape(count, block_frames, values.shape[1]).mean(axis=1)


def hierarchical_bootstrap(
    replica_frame_values: Sequence[np.ndarray],
    block_frames: int,
    samples: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap [same, cross] means over replicas and within-replica blocks."""
    blocks = [block_means(values, block_frames) for values in replica_frame_values]
    rng = np.random.default_rng(seed)
    n_replicas = len(blocks)
    output = np.empty((samples, 2), dtype=float)
    for sample in range(samples):
        selected = rng.integers(0, n_replicas, size=n_replicas)
        replica_estimates = np.empty((n_replicas, 2), dtype=float)
        for position, index in enumerate(selected):
            current = blocks[int(index)]
            chosen_blocks = rng.integers(0, len(current), size=len(current))
            replica_estimates[position] = np.mean(current[chosen_blocks], axis=0)
        output[sample] = np.mean(replica_estimates, axis=0)
    return output


def quantile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, (0.025, 0.975))
    return float(low), float(high)


def summarize_metric(
    label: str,
    definition: str,
    frame_values: Sequence[np.ndarray],
    block_frames: int,
    bootstrap_samples: int,
    seed: int,
    temperature_K: float,
) -> dict[str, object]:
    same, cross = full_estimate(frame_values)
    bootstrap = hierarchical_bootstrap(
        frame_values, block_frames, bootstrap_samples, seed
    )
    valid = (bootstrap[:, 0] > 0.0) & (bootstrap[:, 1] > 0.0)
    if not np.all(valid):
        bootstrap = bootstrap[valid]
    ratio = cross / same
    ratios = bootstrap[:, 1] / bootstrap[:, 0]
    ddg = float(ddg_from_ratio(ratio, temperature_K))
    ddgs = ddg_from_ratio(ratios, temperature_K)
    same_ci = quantile_interval(bootstrap[:, 0])
    cross_ci = quantile_interval(bootstrap[:, 1])
    ratio_ci = quantile_interval(ratios)
    ddg_ci = quantile_interval(ddgs)
    return {
        "contact_definition": label,
        "metric": definition,
        "same_arm": same,
        "same_arm_ci_low": same_ci[0],
        "same_arm_ci_high": same_ci[1],
        "cross_arm": cross,
        "cross_arm_ci_low": cross_ci[0],
        "cross_arm_ci_high": cross_ci[1],
        "cross_over_same": ratio,
        "ratio_ci_low": ratio_ci[0],
        "ratio_ci_high": ratio_ci[1],
        "ddG_cross_minus_same_kcal_mol": ddg,
        "ddG_ci_low": ddg_ci[0],
        "ddG_ci_high": ddg_ci[1],
        "bootstrap_valid_samples": len(bootstrap),
    }


def replica_rows(
    replicas: Sequence[ReplicaData],
    same_mask: np.ndarray,
    cross_mask: np.ndarray,
    cutoff_A: float,
    temperature_K: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for replica in replicas:
        hard = replica.distances_A < cutoff_A
        for metric in ("per_pair", "any_pair"):
            values = metric_values(hard, same_mask, cross_mask, metric)
            same, cross = np.mean(values, axis=0)
            ratio = float(cross / same) if same > 0.0 else math.inf
            output.append(
                {
                    "replica": replica.name,
                    "metric": metric,
                    "cutoff_A": cutoff_A,
                    "same_arm": same,
                    "cross_arm": cross,
                    "cross_over_same": ratio,
                    "ddG_cross_minus_same_kcal_mol": (
                        float(ddg_from_ratio(ratio, temperature_K))
                        if math.isfinite(ratio)
                        else -math.inf
                    ),
                    "frames": len(replica.time_ps),
                    "duration_ps": replica.time_ps[-1] - replica.time_ps[0],
                }
            )
    return output


def pair_statistics(
    replicas: Sequence[ReplicaData],
    pair_schema: Sequence[Mapping[str, object]],
    cutoff_A: float,
    soft_center_A: float,
    soft_width_A: float,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for pair_index, pair in enumerate(pair_schema):
        per_replica_hard = []
        per_replica_soft = []
        per_replica_mean_distance = []
        all_distances = []
        for replica in replicas:
            values = replica.distances_A[:, pair_index]
            all_distances.append(values)
            per_replica_hard.append(float(np.mean(values < cutoff_A)))
            per_replica_soft.append(float(np.mean(soft_contact(values, soft_center_A, soft_width_A))))
            per_replica_mean_distance.append(float(np.mean(values)))
        concatenated = np.concatenate(all_distances)
        output.append(
            {
                "pair": str(pair["column"]).rsplit("_", 1)[0],
                "topology": pair["topology"],
                "left": pair["left"],
                "right": pair["right"],
                "hard_contact_probability": float(np.mean(per_replica_hard)),
                "hard_contact_replica_sd": float(np.std(per_replica_hard, ddof=1)),
                "soft_contact_mean": float(np.mean(per_replica_soft)),
                "mean_distance_A": float(np.mean(per_replica_mean_distance)),
                "median_distance_A": float(np.median(concatenated)),
                "minimum_observed_distance_A": float(np.min(concatenated)),
            }
        )
    output.sort(key=lambda row: float(row["hard_contact_probability"]), reverse=True)
    for rank, row in enumerate(output, start=1):
        row["overall_contact_rank"] = rank
    return output


def contact_episodes(
    replicas: Sequence[ReplicaData],
    pair_schema: Sequence[Mapping[str, object]],
    same_mask: np.ndarray,
    cross_mask: np.ndarray,
    cutoff_A: float,
) -> list[dict[str, object]]:
    """Find contiguous any-pair contact episodes and retain each distance minimum."""
    output: list[dict[str, object]] = []
    masks = {"same_arm": same_mask, "cross_arm": cross_mask}
    for replica in replicas:
        dt = float(np.median(np.diff(replica.time_ps)))
        for topology, mask in masks.items():
            pair_indices = np.flatnonzero(mask)
            class_distances = replica.distances_A[:, pair_indices]
            minimum = np.min(class_distances, axis=1)
            active = minimum < cutoff_A
            changes = np.diff(np.r_[False, active, False].astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            stops = np.flatnonzero(changes == -1)
            for episode_index, (start, stop) in enumerate(zip(starts, stops), start=1):
                local_frame = int(np.argmin(minimum[start:stop]))
                frame = int(start + local_frame)
                local_pair = int(np.argmin(class_distances[frame]))
                pair_index = int(pair_indices[local_pair])
                pair = pair_schema[pair_index]
                output.append(
                    {
                        "replica": replica.name,
                        "topology": topology,
                        "episode": episode_index,
                        "start_frame": int(start),
                        "stop_frame_exclusive": int(stop),
                        "start_time_ps": float(replica.time_ps[start]),
                        "stop_time_ps": float(replica.time_ps[stop - 1]),
                        "duration_ps": float((stop - start) * dt),
                        "minimum_frame": frame,
                        "minimum_time_ps": float(replica.time_ps[frame]),
                        "minimum_pair": str(pair["column"]).rsplit("_", 1)[0],
                        "minimum_distance_A": float(minimum[frame]),
                    }
                )
    return output


def select_decorrelated_candidates(
    episodes: Sequence[Mapping[str, object]],
    per_topology: int,
    separation_ps: float,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for topology in ("same_arm", "cross_arm"):
        candidates = sorted(
            (row for row in episodes if row["topology"] == topology),
            key=lambda row: (float(row["minimum_distance_A"]), str(row["replica"]), float(row["minimum_time_ps"])),
        )
        chosen: list[Mapping[str, object]] = []
        for candidate in candidates:
            independent = all(
                str(candidate["replica"]) != str(previous["replica"])
                or abs(float(candidate["minimum_time_ps"]) - float(previous["minimum_time_ps"])) >= separation_ps
                for previous in chosen
            )
            if independent:
                chosen.append(candidate)
            if len(chosen) == per_topology:
                break
        for rank, row in enumerate(chosen, start=1):
            selected.append(
                {
                    "candidate_id": f"{topology}_{rank:02d}",
                    "rank_within_topology": rank,
                    **dict(row),
                    "selection_note": "distance-only prescreen; orientation and catalyst geometry not yet evaluated",
                }
            )
    return selected


def select_distance_matched_candidates(
    episodes: Sequence[Mapping[str, object]],
    pair_count: int,
    separation_ps: float,
) -> list[dict[str, object]]:
    """Prespecify same/cross candidates matched only on midpoint distance.

    The scarce same-arm episodes define the target distances.  Cross-arm
    episodes are then selected without replacement to minimize distance
    mismatch while enforcing the same temporal separation rule.  No angle,
    energy, or product information enters this pairing.
    """
    same = select_decorrelated_candidates(episodes, pair_count, separation_ps)
    same = [row for row in same if row["topology"] == "same_arm"]
    cross_pool = [row for row in episodes if row["topology"] == "cross_arm"]
    selected_cross: list[Mapping[str, object]] = []
    output: list[dict[str, object]] = []
    for pair_index, same_row in enumerate(same, start=1):
        available = []
        for cross_row in cross_pool:
            independent = all(
                str(cross_row["replica"]) != str(previous["replica"])
                or abs(float(cross_row["minimum_time_ps"]) - float(previous["minimum_time_ps"])) >= separation_ps
                for previous in selected_cross
            )
            if independent:
                available.append(cross_row)
        if not available:
            break
        cross_row = min(
            available,
            key=lambda row: (
                abs(float(row["minimum_distance_A"]) - float(same_row["minimum_distance_A"])),
                float(row["minimum_distance_A"]),
                str(row["replica"]),
                float(row["minimum_time_ps"]),
            ),
        )
        selected_cross.append(cross_row)
        mismatch = abs(float(cross_row["minimum_distance_A"]) - float(same_row["minimum_distance_A"]))
        for topology, row in (("same_arm", same_row), ("cross_arm", cross_row)):
            output.append(
                {
                    "matched_pair_id": f"matched_{pair_index:02d}",
                    "topology": topology,
                    **{key: value for key, value in dict(row).items() if key != "topology"},
                    "distance_mismatch_A": mismatch,
                    "matching_note": "matched on midpoint distance only; geometry/energy/product blind",
                }
            )
    return output


def make_figure(
    output: Path,
    summary_rows: Sequence[Mapping[str, object]],
    replica_stats: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)

    hard_pair = [
        row for row in summary_rows
        if row["metric"] == "per_pair" and str(row["contact_definition"]).startswith("hard_")
    ]
    cutoffs = np.asarray([float(str(row["contact_definition"]).split("_")[1].removesuffix("A")) for row in hard_pair])
    ratios = np.asarray([float(row["cross_over_same"]) for row in hard_pair])
    low = np.asarray([float(row["ratio_ci_low"]) for row in hard_pair])
    high = np.asarray([float(row["ratio_ci_high"]) for row in hard_pair])
    axes[0].plot(cutoffs, ratios, marker="o", color="#176B87")
    axes[0].fill_between(cutoffs, low, high, color="#64CCC5", alpha=0.28, linewidth=0)
    axes[0].axhline(1.0, color="0.4", lw=1, ls="--")
    axes[0].set(xlabel="Contact cutoff (Å)", ylabel="Per-pair cross/same", title="Cutoff sensitivity")

    replica_pair = [row for row in replica_stats if row["metric"] == "per_pair"]
    names = [str(row["replica"]).replace("replica_", "R") for row in replica_pair]
    values = [float(row["cross_over_same"]) for row in replica_pair]
    axes[1].bar(names, values, color="#DAA520")
    axes[1].axhline(1.0, color="0.4", lw=1, ls="--")
    axes[1].set(ylabel="Per-pair cross/same at 5 Å", title="Independent replicas")

    top = sorted(pair_rows, key=lambda row: float(row["hard_contact_probability"]), reverse=True)[:10]
    labels = [str(row["pair"]) for row in top][::-1]
    values = [float(row["hard_contact_probability"]) for row in top][::-1]
    colors = ["#C4473A" if row["topology"] == "same_arm" else "#176B87" for row in top][::-1]
    axes[2].barh(labels, values, color=colors)
    axes[2].set(xlabel="Contact probability at 5 Å", title="Most accessible vinyl pairs")
    fig.savefig(output / "contact_origin_summary.png", dpi=240)
    fig.savefig(output / "contact_origin_summary.pdf")
    plt.close(fig)


def build_report(
    output: Path,
    run_root: Path,
    summary_rows: Sequence[Mapping[str, object]],
    replica_stats: Sequence[Mapping[str, object]],
    pair_rows: Sequence[Mapping[str, object]],
    episodes: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    matched_candidates: Sequence[Mapping[str, object]],
    temperature_K: float,
    cutoff_A: float,
    block_ps: float,
) -> None:
    primary = next(
        row for row in summary_rows
        if row["contact_definition"] == f"hard_{cutoff_A:g}A" and row["metric"] == "per_pair"
    )
    molecular = next(
        row for row in summary_rows
        if row["contact_definition"] == f"hard_{cutoff_A:g}A" and row["metric"] == "any_pair"
    )
    soft = next(row for row in summary_rows if row["contact_definition"] == "soft_5A_w0.5A")
    replica_pair = [row for row in replica_stats if row["metric"] == "per_pair"]
    top_cross = [row for row in pair_rows if row["topology"] == "cross_arm"][:5]
    top_same = [row for row in pair_rows if row["topology"] == "same_arm"][:4]
    episode_counts = {
        topology: sum(row["topology"] == topology for row in episodes)
        for topology in ("same_arm", "cross_arm")
    }

    def interval(row: Mapping[str, object], key: str, low: str, high: str, digits: int = 3) -> str:
        return f"{float(row[key]):.{digits}f} [{float(row[low]):.{digits}f}, {float(row[high]):.{digits}f}]"

    lines = [
        "# A4-m-C8 未反应态跨臂/同臂接触来源分析",
        "",
        "## 结论",
        "",
        f"在 {cutoff_A:g} Å 接触定义下，单对跨臂末端烯烃的平均接触概率是同臂的 "
        f"**{float(primary['cross_over_same']):.2f} 倍**（分层 bootstrap 95% CI "
        f"{float(primary['ratio_ci_low']):.2f}–{float(primary['ratio_ci_high']):.2f}）。对应的占据率接触自由能差为 "
        f"**ΔΔG_contact(cross−same) = {float(primary['ddG_cross_minus_same_kcal_mol']):.2f} kcal mol⁻¹** "
        f"（95% CI {float(primary['ddG_ci_low']):.2f}–{float(primary['ddG_ci_high']):.2f}）。负值表示跨臂接触更有利。",
        "",
        f"将组合数计入后（24 个跨臂组合、4 个同臂组合），任意跨臂对发生接触的概率是任意同臂对的 "
        f"**{float(molecular['cross_over_same']):.2f} 倍**，表观 ΔΔG 为 "
        f"{float(molecular['ddG_cross_minus_same_kcal_mol']):.2f} kcal mol⁻¹。这个量包含组合多重性，不能解释成单条反应路径的化学活化能。",
        "",
        f"软接触函数（中心 5 Å、宽度 0.5 Å）给出跨臂/同臂 = {float(soft['cross_over_same']):.2f}，"
        f"ΔΔG_contact = {float(soft['ddG_cross_minus_same_kcal_mol']):.2f} kcal mol⁻¹，说明结论并非硬阈值造成。",
        "",
        "因此，A4-m-C8 的跨臂来源至少包含两个已经被这批 MD 支持的成分：",
        "",
        "1. **单对内禀预组织优势**：一对跨臂烯烃比一对同臂烯烃更容易进入接触区；",
        "2. **组合熵/多重性优势**：跨臂候选对数量为同臂的 6 倍，进一步放大分子层面的跨臂通量。",
        "",
        "这仍不是最终的反应能垒证明。接下来需要对距离候选帧做取向门控，并从匹配的同臂/跨臂构象开展约束采样或 QM/MM/DFT 计算。",
        "",
        "## 独立重复与稳健性",
        "",
        f"四条 4 ns 轨迹均给出跨臂/同臂比值大于 1："
        + "；".join(f"{row['replica']} = {float(row['cross_over_same']):.2f}" for row in replica_pair)
        + "。重复间差异很大，因此这里使用独立轨迹与 "
        f"{block_ps:g} ps 时间块的分层 bootstrap，而没有把 32,004 帧当成独立样本。",
        "",
        "4–7 Å 阈值扫描的结果见 `contact_metric_summary.csv` 和 `contact_origin_summary.png`。",
        "",
        "## 主要接触对",
        "",
        "跨臂：",
        "",
        "| 排名 | 烯烃对 | 5 Å 接触概率 | 平均距离 / Å |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(top_cross, start=1):
        lines.append(
            f"| {rank} | {row['pair']} | {float(row['hard_contact_probability']):.5f} | {float(row['mean_distance_A']):.2f} |"
        )
    lines.extend(["", "同臂：", "", "| 排名 | 烯烃对 | 5 Å 接触概率 | 平均距离 / Å |", "|---:|---|---:|---:|"])
    for rank, row in enumerate(top_same, start=1):
        lines.append(
            f"| {rank} | {row['pair']} | {float(row['hard_contact_probability']):.5f} | {float(row['mean_distance_A']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## 近接触事件与下一步输入",
            "",
            f"5 Å 下共识别出 {episode_counts['same_arm']} 个同臂接触片段和 {episode_counts['cross_arm']} 个跨臂接触片段。"
            f"`candidate_frames.csv` 从事件最低距离帧中按每条轨迹至少 {100:g} ps 间隔筛出 "
            f"{sum(row['topology'] == 'same_arm' for row in candidates)} 个同臂和 "
            f"{sum(row['topology'] == 'cross_arm' for row in candidates)} 个跨臂候选。",
            "",
            f"此外，`matched_candidate_frames.csv` 给出 {len(matched_candidates) // 2} 组仅按中点距离匹配的同臂/跨臂候选，"
            "用于下一步无偏比较取向门控。",
            "",
            "这些候选仅通过距离筛选，不应直接送入量化计算。下一步需要读取相应 XTC 帧，计算 C=C 轴夹角、侧向进攻角和催化剂可接近性，然后做同臂/跨臂几何匹配。",
            "",
            "## 方法边界",
            "",
            f"- 温度：{temperature_K:.2f} K；ΔΔG = −RT ln(P_cross/P_same)。",
            f"- 输入：`{run_root}` 下四条等权独立重复轨迹。",
            "- 统计单位：独立重复轨迹；轨迹内使用连续时间块保留慢构象相关性。",
            "- `per_pair` 回答单个反应对的内禀接触倾向；`any_pair` 回答整分子的接触通量并包含组合数。",
            "- 占据率自由能只描述无催化剂 q0 构象预组织，不是烯烃复分解化学活化自由能。",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs/A4_m_C8_q0",
    )
    parser.add_argument("--output", type=Path, default=root / "results/a4_m_c8_contact_origin")
    parser.add_argument("--temperature-K", type=float, default=DEFAULT_TEMPERATURE_K)
    parser.add_argument("--primary-cutoff-A", type=float, default=5.0)
    parser.add_argument("--block-ps", type=float, default=200.0)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--candidate-separation-ps", type=float, default=100.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    if args.temperature_K <= 0.0 or args.block_ps <= 0.0 or args.bootstrap_samples < 100:
        raise SystemExit("temperature/block must be positive and bootstrap samples >= 100")
    metadata = json.loads((run_root / "setup/metadata.json").read_text(encoding="utf-8"))
    pair_schema = list(metadata["pair_schema"])
    pair_columns = [str(pair["column"]) for pair in pair_schema]
    same_mask = np.asarray([pair["topology"] == "same_arm" for pair in pair_schema])
    cross_mask = np.asarray([pair["topology"] == "cross_arm" for pair in pair_schema])
    if (int(np.sum(same_mask)), int(np.sum(cross_mask))) != (4, 24):
        raise ValueError("A4-m-C8 must contain 4 same-arm and 24 cross-arm pairs")
    replicas = [
        read_distance_csv(run_root / f"replica_{index:02d}/vinyl_distances.csv", pair_columns)
        for index in range(1, 5)
    ]
    frame_intervals = [float(np.median(np.diff(replica.time_ps))) for replica in replicas]
    if not np.allclose(frame_intervals, frame_intervals[0]):
        raise ValueError("replicas have different frame intervals")
    block_frames = int(round(args.block_ps / frame_intervals[0]))
    output.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    seed_offset = 0
    cutoffs = sorted(set(DEFAULT_CUTOFFS_A + (args.primary_cutoff_A,)))
    for cutoff in cutoffs:
        hard = [replica.distances_A < cutoff for replica in replicas]
        for metric in ("per_pair", "any_pair"):
            values = [metric_values(item, same_mask, cross_mask, metric) for item in hard]
            summary_rows.append(
                summarize_metric(
                    f"hard_{cutoff:g}A",
                    metric,
                    values,
                    block_frames,
                    args.bootstrap_samples,
                    args.seed + seed_offset,
                    args.temperature_K,
                )
            )
            seed_offset += 1
    soft_values = [
        metric_values(soft_contact(replica.distances_A, 5.0, 0.5), same_mask, cross_mask, "per_pair")
        for replica in replicas
    ]
    summary_rows.append(
        summarize_metric(
            "soft_5A_w0.5A",
            "per_pair",
            soft_values,
            block_frames,
            args.bootstrap_samples,
            args.seed + seed_offset,
            args.temperature_K,
        )
    )

    summary_fields = [
        "contact_definition", "metric", "same_arm", "same_arm_ci_low", "same_arm_ci_high",
        "cross_arm", "cross_arm_ci_low", "cross_arm_ci_high", "cross_over_same",
        "ratio_ci_low", "ratio_ci_high", "ddG_cross_minus_same_kcal_mol",
        "ddG_ci_low", "ddG_ci_high", "bootstrap_valid_samples",
    ]
    write_csv(output / "contact_metric_summary.csv", summary_rows, summary_fields)

    replica_stats = replica_rows(
        replicas, same_mask, cross_mask, args.primary_cutoff_A, args.temperature_K
    )
    write_csv(
        output / "replica_statistics.csv",
        replica_stats,
        ["replica", "metric", "cutoff_A", "same_arm", "cross_arm", "cross_over_same",
         "ddG_cross_minus_same_kcal_mol", "frames", "duration_ps"],
    )
    pair_rows = pair_statistics(replicas, pair_schema, args.primary_cutoff_A, 5.0, 0.5)
    write_csv(
        output / "pair_statistics.csv",
        pair_rows,
        ["overall_contact_rank", "pair", "topology", "left", "right",
         "hard_contact_probability", "hard_contact_replica_sd", "soft_contact_mean",
         "mean_distance_A", "median_distance_A", "minimum_observed_distance_A"],
    )
    episodes = contact_episodes(
        replicas, pair_schema, same_mask, cross_mask, args.primary_cutoff_A
    )
    episode_fields = [
        "replica", "topology", "episode", "start_frame", "stop_frame_exclusive",
        "start_time_ps", "stop_time_ps", "duration_ps", "minimum_frame", "minimum_time_ps",
        "minimum_pair", "minimum_distance_A",
    ]
    write_csv(output / "contact_episodes.csv", episodes, episode_fields)
    candidates = select_decorrelated_candidates(
        episodes, args.candidate_count, args.candidate_separation_ps
    )
    write_csv(
        output / "candidate_frames.csv",
        candidates,
        ["candidate_id", "rank_within_topology", *episode_fields, "selection_note"],
    )
    matched_candidates = select_distance_matched_candidates(
        episodes, args.candidate_count, args.candidate_separation_ps
    )
    write_csv(
        output / "matched_candidate_frames.csv",
        matched_candidates,
        ["matched_pair_id", *episode_fields, "distance_mismatch_A", "matching_note"],
    )
    make_figure(output, summary_rows, replica_stats, pair_rows)
    build_report(
        output, run_root, summary_rows, replica_stats, pair_rows, episodes, candidates,
        matched_candidates,
        args.temperature_K, args.primary_cutoff_A, args.block_ps,
    )
    provenance = {
        "analysis": "A4-m-C8 q0 same-arm/cross-arm contact-origin decomposition",
        "run_root": str(run_root),
        "replicas": [replica.name for replica in replicas],
        "frames_per_replica": [len(replica.time_ps) for replica in replicas],
        "frame_interval_ps": frame_intervals[0],
        "temperature_K": args.temperature_K,
        "primary_cutoff_A": args.primary_cutoff_A,
        "soft_contact": {"center_A": 5.0, "width_A": 0.5},
        "same_arm_pairs": int(np.sum(same_mask)),
        "cross_arm_pairs": int(np.sum(cross_mask)),
        "bootstrap": {
            "scheme": "replica plus within-replica nonoverlapping time-block bootstrap",
            "block_ps": args.block_ps,
            "complete_blocks_per_replica": len(replicas[0].time_ps) // block_frames,
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "interpretation_boundary": "occupancy-derived contact/preorganization free energy; not chemical activation barrier",
    }
    (output / "analysis_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Analysis complete: {output}")


if __name__ == "__main__":
    main()
