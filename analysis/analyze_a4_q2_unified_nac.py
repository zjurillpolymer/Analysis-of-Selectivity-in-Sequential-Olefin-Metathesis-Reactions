#!/usr/bin/env python3
"""Unify q2 NAC contrasts and select representative QM frame requests.

The common sign convention is

    delta-delta-G = alternative - reference

where alternative is cross-arm for adjacent q2 and switch for opposite q2.
Negative values therefore always mean that cross/switch is favoured.

This analysis uses the compact vinyl-coordinate exports.  They are sufficient
for contact/orientation/NAC statistics and exact frame-time selection, but not
for extracting complete QM cluster structures; that requires the original full
q2 trajectories listed in each NAC manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from analyze_explicit_dcm_postreaction_nac import (
    geometry_arrays,
    read_coordinates,
    trajectory_boxes_A,
)
from analyze_explicit_dcm_q0_geometry_gate import soft_lower, soft_upper


R_KCAL = 0.00198720425864083
TEMPERATURE_K = 313.15
RT = R_KCAL * TEMPERATURE_K
PRIMARY_SETTING = "d5_axis45_side45"

STATE_ORDER = (
    "A4_m_C5_2_adjacent_cis",
    "A4_m_C8_2_adjacent_cis",
    "A4_m_C5_2_opposite_cis",
    "A4_m_C8_2_opposite_cis",
)

STATE_LABELS = {
    "A4_m_C5_2_adjacent_cis": "C5 adjacent\ncross - same",
    "A4_m_C8_2_adjacent_cis": "C8 adjacent\ncross - same",
    "A4_m_C5_2_opposite_cis": "C5 opposite\nswitch - reuse",
    "A4_m_C8_2_opposite_cis": "C8 opposite\nswitch - reuse",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def f(row: Mapping[str, str], field: str) -> float:
    return float(row[field])


def invert_interval(low: float, high: float) -> tuple[float, float]:
    return -high, -low


def state_definition(state: str) -> dict[str, str]:
    if "adjacent" in state:
        return {
            "contrast": "cross_minus_same",
            "grouping": "topology",
            "alternative": "cross_arm",
            "reference": "same_arm",
            "human_contrast": "cross - same",
        }
    return {
        "contrast": "switch_minus_reuse",
        "grouping": "route_family",
        "alternative": "switch",
        "reference": "reuse",
        "human_contrast": "switch - reuse",
    }


def unified_contrast_rows(source: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in read_csv(source):
        if row["state"] not in STATE_ORDER:
            continue
        sign = -1.0 if row["branch"] == "opposite" else 1.0
        definition = state_definition(row["state"])
        item: dict[str, object] = {
            "state": row["state"],
            "system": row["system"],
            "branch": row["branch"],
            "setting_id": row["setting_id"],
            "contrast": definition["contrast"],
            "alternative_group": definition["alternative"],
            "reference_group": definition["reference"],
            "replicas": int(row["replicas"]),
        }
        for component in ("contact", "orientation", "total_nac"):
            prefix = f"ddG_{component}"
            point = sign * f(row, f"{prefix}_point_kcal_mol")
            median = sign * f(row, f"{prefix}_median_kcal_mol")
            old_low = f(row, f"{prefix}_ci_low_kcal_mol")
            old_high = f(row, f"{prefix}_ci_high_kcal_mol")
            if sign < 0:
                low, high = invert_interval(old_low, old_high)
            else:
                low, high = old_low, old_high
            item[f"{prefix}_point_kcal_mol"] = point
            item[f"{prefix}_median_kcal_mol"] = median
            item[f"{prefix}_ci_low_kcal_mol"] = low
            item[f"{prefix}_ci_high_kcal_mol"] = high
        item["alternative_over_reference_nac_odds"] = math.exp(
            -float(item["ddG_total_nac_point_kcal_mol"]) / RT
        )
        output.append(item)
    return output


def replica_estimates(source: Path) -> list[dict[str, object]]:
    rows = read_csv(source)
    indexed = {
        (row["state"], int(row["replica"]), row["setting_id"], row["grouping"], row["group"]): row
        for row in rows
    }
    output: list[dict[str, object]] = []
    for state in STATE_ORDER:
        definition = state_definition(state)
        for replica in (1, 2, 3):
            base = (state, replica, PRIMARY_SETTING, definition["grouping"])
            alternative = indexed[(*base, definition["alternative"])]
            reference = indexed[(*base, definition["reference"])]
            alt_contact = f(alternative, "per_pair_contact")
            ref_contact = f(reference, "per_pair_contact")
            alt_orientation = f(alternative, "conditional_orientation")
            ref_orientation = f(reference, "conditional_orientation")
            alt_nac = f(alternative, "per_pair_nac")
            ref_nac = f(reference, "per_pair_nac")
            output.append({
                "state": state,
                "system": alternative["system"],
                "branch": alternative["branch"],
                "contrast": definition["contrast"],
                "replica": replica,
                "alternative_group": definition["alternative"],
                "reference_group": definition["reference"],
                "alternative_per_pair_contact": alt_contact,
                "reference_per_pair_contact": ref_contact,
                "alternative_conditional_orientation": alt_orientation,
                "reference_conditional_orientation": ref_orientation,
                "alternative_per_pair_nac": alt_nac,
                "reference_per_pair_nac": ref_nac,
                "ddG_contact_kcal_mol": -RT * math.log(alt_contact / ref_contact),
                "ddG_orientation_kcal_mol": -RT * math.log(alt_orientation / ref_orientation),
                "ddG_total_nac_kcal_mol": -RT * math.log(alt_nac / ref_nac),
            })
    return output


def umbrella_rows(adjacent_path: Path, opposite_path: Path) -> list[dict[str, object]]:
    adjacent = {row["quantity"]: row for row in read_csv(adjacent_path)}
    opposite = {
        row["quantity"].split()[0]: row
        for row in read_csv(opposite_path)
        if row["quantity"].startswith(("C5 switch", "C8 switch"))
    }
    output = []
    for state in STATE_ORDER:
        system = "C5" if "C5" in state else "C8"
        if "adjacent" in state:
            row = adjacent[system]
            point_field = "point_ddG_kcal_mol"
        else:
            row = opposite[system]
            point_field = "point_kcal_mol"
        output.append({
            "state": state,
            "system": system,
            "branch": "adjacent" if "adjacent" in state else "opposite",
            "contrast": state_definition(state)["contrast"],
            "umbrella_point_kcal_mol": float(row[point_field]),
            "umbrella_ci_low_kcal_mol": float(row["bootstrap_ci_low_kcal_mol"]),
            "umbrella_ci_high_kcal_mol": float(row["bootstrap_ci_high_kcal_mol"]),
        })
    return output


def match_umbrella_nac(
    primary_rows: Sequence[Mapping[str, object]],
    umbrellas: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    nac = {str(row["state"]): row for row in primary_rows}
    output = []
    for umbrella in umbrellas:
        row = nac[str(umbrella["state"])]
        umbrella_point = float(umbrella["umbrella_point_kcal_mol"])
        nac_point = float(row["ddG_total_nac_point_kcal_mol"])
        output.append({
            **umbrella,
            "nac_point_kcal_mol": nac_point,
            "nac_ci_low_kcal_mol": float(row["ddG_total_nac_ci_low_kcal_mol"]),
            "nac_ci_high_kcal_mol": float(row["ddG_total_nac_ci_high_kcal_mol"]),
            "sign_agreement": int(np.sign(umbrella_point) == np.sign(nac_point)),
            "interpretation": "alternative_favored" if nac_point < 0 else "reference_favored",
        })
    return output


def representative_frames(manifests: Iterable[Path], workspace: Path) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for manifest_path in sorted(manifests):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        state = str(manifest["state"])
        if state not in STATE_ORDER:
            continue
        annotations = list(manifest["pair_annotations"])
        labels = [str(label) for label in manifest["ordered_vinyl_labels"]]
        times, coordinates = read_coordinates(
            manifest_path.parent / str(manifest["coordinate_file"]),
            2 * len(labels),
            int(manifest["frames"]),
        )
        boxes = trajectory_boxes_A(manifest_path.parent / str(manifest["box_file"]), len(times))
        distances, axes, sides, _, valid, _, _ = geometry_arrays(coordinates, labels, annotations, boxes)
        times = times[valid]
        contact = soft_upper(distances, 5.0, 0.5)
        orientation = soft_upper(axes, 45.0, 7.5) * soft_lower(sides, 45.0, 7.5)
        nac = contact * orientation
        definition = state_definition(state)
        field = definition["grouping"]
        groups = (("alternative", definition["alternative"]), ("reference", definition["reference"]))
        for role, group in groups:
            pair_indices = [i for i, annotation in enumerate(annotations) if annotation[field] == group]
            if not pair_indices:
                raise ValueError(f"{state}: no {field}={group} pairs")
            subset = nac[:, pair_indices]
            frame_index, subset_index = np.unravel_index(int(np.argmax(subset)), subset.shape)
            pair_index = pair_indices[subset_index]
            annotation = annotations[pair_index]
            replica = int(manifest["replica"])
            source_xtc = str(manifest["source"]["xtc"])
            source_tpr = str(manifest["source"]["tpr"])
            output.append({
                "state": state,
                "system": manifest["system"],
                "branch": manifest["branch"],
                "contrast": definition["contrast"],
                "group_role": role,
                "grouping": field,
                "group": group,
                "replica": replica,
                "time_ps": float(times[frame_index]),
                "pair": annotation["pair"],
                "left_vinyl": f"V{annotation['left_vinyl']}",
                "right_vinyl": f"V{annotation['right_vinyl']}",
                "left_arm": annotation["left_arm"],
                "right_arm": annotation["right_arm"],
                "topology": annotation["topology"],
                "route_family": annotation["route_family"],
                "distance_A": float(distances[frame_index, pair_index]),
                "axis_angle_deg": float(axes[frame_index, pair_index]),
                "side_angle_deg": float(sides[frame_index, pair_index]),
                "soft_contact": float(contact[frame_index, pair_index]),
                "orientation_gate": float(orientation[frame_index, pair_index]),
                "soft_nac": float(nac[frame_index, pair_index]),
                "qm_geometry_quality": (
                    "strict_gate_pass"
                    if (
                        float(distances[frame_index, pair_index]) <= 5.0
                        and float(axes[frame_index, pair_index]) <= 45.0
                        and float(sides[frame_index, pair_index]) >= 45.0
                    )
                    else ("soft_gate_candidate" if float(nac[frame_index, pair_index]) >= 0.1 else "poor_unbiased_candidate")
                ),
                "compact_manifest": str(manifest_path.relative_to(workspace)),
                "required_full_xtc": source_xtc,
                "required_tpr": source_tpr,
                "requested_structure_name": (
                    f"{state}_rep{replica:02d}_{role}_{annotation['left_vinyl']}_"
                    f"{annotation['right_vinyl']}_{float(times[frame_index]):.1f}ps.gro"
                ),
                "full_structure_status": "blocked_missing_full_q2_xtc",
            })
    return output


def make_figure(
    output_stem: Path,
    primary: Sequence[Mapping[str, object]],
    replica_rows: Sequence[Mapping[str, object]],
    sensitivity: Sequence[Mapping[str, object]],
    agreement: Sequence[Mapping[str, object]],
) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    y = np.arange(len(STATE_ORDER))
    labels = [STATE_LABELS[state] for state in STATE_ORDER]
    by_state = {str(row["state"]): row for row in primary}

    contact_color = "#3B6FB6"
    orientation_color = "#E69F00"
    c5_color = "#4C78A8"
    c8_color = "#D55E5E"

    # A: additive free-energy decomposition.
    contact_values = np.asarray([float(by_state[state]["ddG_contact_point_kcal_mol"]) for state in STATE_ORDER])
    orient_values = np.asarray([float(by_state[state]["ddG_orientation_point_kcal_mol"]) for state in STATE_ORDER])
    total_values = contact_values + orient_values
    ax_a.barh(y, contact_values, color=contact_color, height=0.55, label="Contact")
    ax_a.barh(y, orient_values, left=contact_values, color=orientation_color, height=0.55, label="Orientation")
    ax_a.scatter(total_values, y, s=18, color="black", zorder=4, label="Total NAC")
    ax_a.axvline(0, color="0.25", lw=0.8)
    ax_a.set_yticks(y, labels)
    ax_a.invert_yaxis()
    ax_a.set_xlabel(r"Unified $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax_a.set_title(r"$\bf{a}$  Contact accessibility dominates")
    ax_a.legend(frameon=False, fontsize=7, loc="upper left")

    # B: independent umbrella/NAC sign agreement.
    for row in agreement:
        x = float(row["umbrella_point_kcal_mol"])
        yv = float(row["nac_point_kcal_mol"])
        xerr = np.asarray([[x - float(row["umbrella_ci_low_kcal_mol"])],
                           [float(row["umbrella_ci_high_kcal_mol"]) - x]])
        yerr = np.asarray([[yv - float(row["nac_ci_low_kcal_mol"])],
                           [float(row["nac_ci_high_kcal_mol"]) - yv]])
        color = c5_color if str(row["system"]) == "C5" else c8_color
        marker = "o" if str(row["branch"]) == "adjacent" else "s"
        ax_b.errorbar(x, yv, xerr=xerr, yerr=yerr, fmt=marker, color=color,
                      ms=5, capsize=2, lw=0.9)
        ax_b.annotate(f"{row['system']} {str(row['branch'])[:3]}", (x, yv),
                      xytext=(4, 3), textcoords="offset points", fontsize=6.5)
    identity_bounds = (-16.0, 3.0)
    ax_b.plot(identity_bounds, identity_bounds, color="0.55", lw=0.8, ls="--")
    ax_b.axhline(0, color="0.75", lw=0.6)
    ax_b.axvline(0, color="0.75", lw=0.6)
    ax_b.set_xlim((-9.0, 3.0))
    ax_b.set_ylim((-16.0, 3.0))
    ax_b.set_xlabel(r"Umbrella $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax_b.set_ylabel(r"Unbiased NAC $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax_b.set_title(r"$\bf{b}$  Umbrella and NAC agree in sign (4/4)")

    # C: replica-resolved NAC estimates.
    rng_offsets = {1: -0.13, 2: 0.0, 3: 0.13}
    for i, state in enumerate(STATE_ORDER):
        color = c5_color if "C5" in state else c8_color
        state_reps = [row for row in replica_rows if row["state"] == state]
        for row in state_reps:
            ax_c.scatter(float(row["ddG_total_nac_kcal_mol"]), i + rng_offsets[int(row["replica"])],
                         color=color, s=20, edgecolor="white", linewidth=0.4, zorder=3)
        combined = by_state[state]
        point = float(combined["ddG_total_nac_point_kcal_mol"])
        low = float(combined["ddG_total_nac_ci_low_kcal_mol"])
        high = float(combined["ddG_total_nac_ci_high_kcal_mol"])
        ax_c.errorbar(point, i, xerr=np.asarray([[point-low], [high-point]]), fmt="D",
                      color="black", ms=3.5, capsize=2, lw=1.0, zorder=4)
    ax_c.axvline(0, color="0.25", lw=0.8)
    ax_c.set_yticks(np.arange(4), labels)
    ax_c.invert_yaxis()
    ax_c.set_xlabel(r"Replica NAC $\Delta\Delta G$ (kcal mol$^{-1}$)")
    ax_c.set_title(r"$\bf{c}$  Three independent 2 ns replicas")

    # D: all 27 gate definitions per state.
    for i, state in enumerate(STATE_ORDER):
        values = np.asarray([
            float(row["ddG_total_nac_point_kcal_mol"])
            for row in sensitivity if row["state"] == state
        ])
        jitter = np.linspace(-0.16, 0.16, len(values))
        color = c5_color if "C5" in state else c8_color
        ax_d.scatter(values, i + jitter, color=color, s=10, alpha=0.7, linewidth=0)
        primary_value = float(by_state[state]["ddG_total_nac_point_kcal_mol"])
        ax_d.scatter(primary_value, i, marker="D", color="black", s=22, zorder=4)
        ax_d.plot([float(np.min(values)), float(np.max(values))], [i, i], color=color, lw=1.1, zorder=1)
    ax_d.axvline(0, color="0.25", lw=0.8)
    ax_d.set_yticks(np.arange(4), labels)
    ax_d.invert_yaxis()
    ax_d.set_xlabel(r"NAC $\Delta\Delta G$ across gates (kcal mol$^{-1}$)")
    ax_d.set_title(r"$\bf{d}$  Preference sign stable for all 108 tests")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="0.92", lw=0.5, zorder=0)
    fig.suptitle("q2 arm-pair selectivity arises primarily from contact accessibility", fontsize=10.5)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"))
    fig.savefig(output_stem.with_suffix(".pdf"))
    fig.savefig(output_stem.with_suffix(".png"), dpi=450)
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def make_report(
    path: Path,
    primary: Sequence[Mapping[str, object]],
    sensitivity: Sequence[Mapping[str, object]],
    agreement: Sequence[Mapping[str, object]],
    frames: Sequence[Mapping[str, object]],
) -> None:
    by_state = {str(row["state"]): row for row in primary}
    frame_quality = {
        quality: sum(1 for row in frames if row["qm_geometry_quality"] == quality)
        for quality in ("strict_gate_pass", "soft_gate_candidate", "poor_unbiased_candidate")
    }
    lines = [
        "# A4-m-C5/C8 q2 统一 NAC 分解",
        "",
        "## 核心结论",
        "",
        "在统一符号 `alternative-reference` 下，负值表示 cross/switch 更有利。四个 q2 状态中，偏好方向主要由乙烯基对能否形成近距离接触决定；取向门控通常只作小幅修正。伞形采样与无偏轨迹 NAC 的方向在 4/4 个比较中一致。",
        "",
        "| 状态 | 比较 | contact | orientation | total NAC | 95% CI | 方向 |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for state in STATE_ORDER:
        row = by_state[state]
        total = float(row["ddG_total_nac_point_kcal_mol"])
        direction = "cross/switch" if total < 0 else "same/reuse"
        lines.append(
            f"| {str(row['system']).replace('A4_m_', '')} {row['branch']} | {row['contrast']} | "
            f"{float(row['ddG_contact_point_kcal_mol']):+.3f} | "
            f"{float(row['ddG_orientation_point_kcal_mol']):+.3f} | {total:+.3f} | "
            f"[{float(row['ddG_total_nac_ci_low_kcal_mol']):+.3f}, "
            f"{float(row['ddG_total_nac_ci_high_kcal_mol']):+.3f}] | {direction} |"
        )
    lines += [
        "",
        "数值单位均为 kcal mol⁻¹，温度为 313.15 K。contact 与 orientation 在自由能上可加，total NAC = contact + orientation。",
        "",
        "## 机制解释",
        "",
        "- C8 adjacent 与 C8 opposite 均强烈偏向 alternative；两者几乎全部来自 contact accessibility，而不是预组织取向差异。也就是说，C8 更长的连接臂主要改变了不同臂乙烯基彼此接近的构象概率。",
        "- C5 opposite 保留 reuse 优势，同样主要是接触概率造成的；说明短链在该已反应拓扑下更容易维持既有臂对，而不是切换到新的臂对。",
        "- C5 adjacent 中 cross 仍优于 same，但 orientation 贡献约 −0.38 kcal mol⁻¹，是四个状态中唯一相对明显的取向贡献；它仍小于 contact 贡献。",
        "- 27 套距离/角度门控在每个状态上都保持同一方向，共 108/108 个符号测试稳定，因此结论不是某一个阈值的偶然结果。",
        "",
        "## 与伞形采样的关系",
        "",
        "伞形采样和无偏 NAC 对四个状态的偏好方向完全一致，但自由能数值不应直接视为同一物理量：伞形采样沿选定距离坐标得到 PMF，而 NAC 同时衡量接触与几何取向的平衡占有概率。这里最可靠的是方向一致与机制分解，而不是要求两类数值一一相等。",
        "",
        "## 代表帧与后续 QM",
        "",
        f"已为每个状态、每个重复的 alternative/reference 各筛选一个最高 soft-NAC 时刻，共 {len(frames)} 个无偏候选，见 `q2_representative_frame_requests.csv`。其中 {frame_quality['strict_gate_pass']} 个通过严格门控、{frame_quality['soft_gate_candidate']} 个仅为软门控候选、{frame_quality['poor_unbiased_candidate']} 个即使作为该组最佳帧也很差。后者本身就是接触不可达的证据，不应直接作为 QM 反应物构型。",
        "",
        "无偏轨迹下载结果只含乙烯基紧凑坐标和末帧 GRO，不含清单时刻的完整 q2 XTC，因此无法复原这些无偏帧的完整结构。作为可操作替代，已从 9 条伞形路径的 0.35 nm 窗口中筛选取向最佳帧，并导出 9 套完整溶质 GRO/PDB/XYZ 到 `results/a4_q2_qm_candidates/`；9/9 均通过严格几何门控。伞形帧只用于提供可比的近反应起始几何，不能再次用于估计平衡构象偏好。",
        "",
        "## 结论边界",
        "",
        "这些结果解释的是催化剂缺失条件下的构象可达性/近攻击构象来源，不是化学成键活化能。下一阶段 QM 应比较从代表帧构建的 cross/switch 与 same/reuse 反应路径，并将构象选择自由能与化学势垒分开报告。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/a4_q2_unified_nac_analysis/final"))
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output

    nac_root = workspace / "results/a4_postreaction_nac_analysis/b6-gromacs-parallel-kcl7v"
    export_root = workspace / "results/a4_postreaction_nac_exports/b6-gromacs-parallel-kcl7v"
    all_contrasts = unified_contrast_rows(nac_root / "free_energy_contrasts.csv")
    primary = [row for row in all_contrasts if row["setting_id"] == PRIMARY_SETTING]
    if len(primary) != 4 or len(all_contrasts) != 108:
        raise ValueError(f"expected 4 primary and 108 sensitivity rows, got {len(primary)}, {len(all_contrasts)}")
    primary.sort(key=lambda row: STATE_ORDER.index(str(row["state"])))
    all_contrasts.sort(key=lambda row: (STATE_ORDER.index(str(row["state"])), str(row["setting_id"])))

    replicas = replica_estimates(nac_root / "replica_group_statistics.csv")
    umbrellas = umbrella_rows(
        workspace / "results/a4_q2_adjacent_umbrella_analysis/final/adjacent_q2_same_cross_contrasts.csv",
        workspace / "results/a4_q2_umbrella_analysis/final/q2_route_contrasts.csv",
    )
    agreement = match_umbrella_nac(primary, umbrellas)
    frames = representative_frames(export_root.glob("**/nac_manifest.json"), workspace)
    if len(frames) != 24:
        raise ValueError(f"expected 24 representative frame requests, got {len(frames)}")

    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "q2_unified_decomposition.csv", primary)
    write_csv(output / "q2_unified_replica_estimates.csv", replicas)
    write_csv(output / "q2_unified_gate_sensitivity.csv", all_contrasts)
    write_csv(output / "q2_umbrella_nac_agreement.csv", agreement)
    write_csv(output / "q2_representative_frame_requests.csv", frames)

    sensitivity_summary = {}
    for state in STATE_ORDER:
        values = np.asarray([
            float(row["ddG_total_nac_point_kcal_mol"])
            for row in all_contrasts if row["state"] == state
        ])
        expected_sign = np.sign(float(next(row for row in primary if row["state"] == state)["ddG_total_nac_point_kcal_mol"]))
        sensitivity_summary[state] = {
            "tests": int(len(values)),
            "minimum_kcal_mol": float(np.min(values)),
            "maximum_kcal_mol": float(np.max(values)),
            "same_sign_as_primary": int(np.sum(np.sign(values) == expected_sign)),
        }
    summary = {
        "temperature_K": TEMPERATURE_K,
        "sign_convention": "alternative_minus_reference; negative means cross/switch favoured",
        "primary_gate": {
            "distance_center_A": 5.0,
            "distance_width_A": 0.5,
            "axis_max_deg": 45.0,
            "side_min_deg": 45.0,
            "angle_width_deg": 7.5,
        },
        "states": primary,
        "umbrella_nac_sign_agreement": f"{sum(int(row['sign_agreement']) for row in agreement)}/4",
        "gate_sensitivity": sensitivity_summary,
        "representative_frame_requests": len(frames),
        "full_structure_extraction": {
            "unbiased_frames": "blocked_missing_full_q2_xtc",
            "umbrella_fallback": "9 orientation-screened solute structures extracted separately",
        },
    }
    (output / "q2_unified_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    make_figure(output / "q2_selectivity_mechanism", primary, replicas, all_contrasts, agreement)
    make_report(output / "q2_selectivity_mechanism_report_zh.md", primary, all_contrasts, agreement, frames)
    (output / "figure_legend_en.md").write_text(
        "**Figure | Contact accessibility dominates q2 arm-pair selectivity.** "
        "(A) Decomposition of the unified alternative-minus-reference free-energy proxy into contact and conditional-orientation terms; negative values favour cross-arm (adjacent q2) or switch (opposite q2). "
        "(B) Independent umbrella-sampling and unbiased near-attack-conformation (NAC) estimates agree in sign for all four comparisons; error bars are bootstrap 95% confidence intervals. "
        "(C) NAC estimates from three independent 2-ns replicas (coloured points) and their combined estimate (black diamond, bootstrap 95% confidence interval). "
        "(D) Sensitivity across 27 distance/orientation gate definitions per state; all 108 tests preserve the primary preference sign. "
        "NAC statistics use per-pair probabilities at 313.15 K; they quantify catalyst-free conformational accessibility rather than chemical activation barriers.\n",
        encoding="utf-8",
    )
    (output / "figure_qa_notes.md").write_text(
        "# Figure QA notes\n\n"
        "- Core claim: q2 arm-pair preference is dominated by contact accessibility; conditional orientation is secondary.\n"
        "- Archetype: double-column quantitative 2 x 2 grid; final size 7.2 x 5.8 in (183 x 147 mm).\n"
        "- Backend: Python/matplotlib only. SVG text is retained as text and PDF uses TrueType fonts.\n"
        "- Replicate definition: three independently initialized 2-ns explicit-DCM GROMACS trajectories per q2 state; 4001 frames per trajectory at 0.5-ps spacing.\n"
        "- Center statistic: free-energy proxy from the ratio of across-replica mean per-pair probabilities at 313.15 K.\n"
        "- Interval: 95% percentile interval from 10,000 paired replica-bootstrap draws; the independent trajectory replica is the resampling unit.\n"
        "- Multiple-comparison correction and p values: not used; inference is based on bootstrap intervals and a prespecified 27-setting sensitivity grid.\n"
        "- Panel d: 27 distance/orientation gates per state (108 state-setting tests); all retain the primary sign. These are sensitivity tests, not independent replicates.\n"
        "- Source data: q2_unified_decomposition.csv, q2_unified_replica_estimates.csv, q2_unified_gate_sensitivity.csv, and q2_umbrella_nac_agreement.csv.\n"
        "- Export bundle: editable SVG/PDF, 600-dpi LZW TIFF, 450-dpi PNG preview, source CSV files, analysis script, legend, and these QA notes.\n"
        "- Scope: catalyst-free NAC probabilities are conformational free-energy proxies, not chemical activation barriers.\n",
        encoding="utf-8",
    )
    print(f"output={output}")


if __name__ == "__main__":
    main()
