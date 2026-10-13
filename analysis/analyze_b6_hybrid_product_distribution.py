#!/usr/bin/env python3
"""Propagate B6_p_C6 product distributions with sparse state-specific MD kernels.

The complete labelled partial-matching DAG supplies the reaction network.  At
states for which explicit-DCM trajectories exist, the q0 contact prior and the
global gamma response are replaced by that state's measured soft-contact
vector.  The same-arm multiplier and arm-pair reuse coefficient are retained.
Unmeasured states continue to use the original three-parameter response law.

This is deliberately a hybrid sensitivity model, not a claim that contact
occupancy is an absolute reaction rate.  It quantifies what the currently
measured q1--q3 conditional geometries do to the full R1--R6 distribution and
which unmeasured partial matchings can most strongly affect the result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

import fit_three_parameter_model as shared
import mtmr_baseline as baseline


DEFAULT_Q0 = "results/b6_p_c6_paper_auto/audit_50ns/pair_state_summary.csv"
DEFAULT_Q1 = "results/b6_q1_v3_v6_50ns_analysis/audit_50ns/pair_state_summary.csv"
DEFAULT_Q2 = "results/b6_q2_v3_v6_v11_v12_50ns_analysis/audit/pair_state_summary.csv"
DEFAULT_Q3 = "results/b6_q3_20ns_analysis/pair_state_summary.csv"
DEFAULT_OUTPUT = "results/b6_hybrid_product_distribution"

MEASURED_STATE_EDGES: dict[str, tuple[tuple[int, int], ...]] = {
    "B6_p_C6_q0": (),
    "B6_p_C6_q1_V3_V6_cis": ((3, 6),),
    "B6_p_C6_q2_V3_V6_V11_V12_cis": ((3, 6), (11, 12)),
    "B6_p_C6_q3_V3_V6_V11_V12_V8_V10_cis": (
        (3, 6), (8, 10), (11, 12)
    ),
    "B6_p_C6_q3_V3_V6_V11_V12_V7_V9_cis": (
        (3, 6), (7, 9), (11, 12)
    ),
}


@dataclass(frozen=True)
class KernelSource:
    state_name: str
    trajectory_ns: int
    summary_path: Path


@dataclass
class B6Model:
    system: shared.PreparedSystem
    edge_to_index: dict[tuple[int, int], int]
    edge_masks: tuple[int, ...]
    observed_mean: dict[tuple[int, ...], np.ndarray]
    observed_replicas: dict[tuple[int, ...], dict[int, np.ndarray]]
    observed_sources: dict[tuple[int, ...], KernelSource]


@dataclass
class Propagation:
    forward_layers: list[dict[tuple[int, ...], float]]
    backward_layers: list[dict[tuple[int, ...], np.ndarray]]
    distribution: np.ndarray
    priority_rows: list[dict[str, object]]
    coverage_rows: list[dict[str, object]]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames required for empty CSV: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_edges(edges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    output = tuple(sorted((min(int(a), int(b)), max(int(a), int(b))) for a, b in edges))
    vertices = [value for edge in output for value in edge]
    if len(vertices) != len(set(vertices)):
        raise ValueError(f"not a matching: {output}")
    return output


def load_manifest(root: Path) -> tuple[list[tuple[int, int]], list[str], tuple[int, ...]]:
    rows = read_csv(root / "gromacs_dcm/state_manifest.csv")
    row = next(item for item in rows if item["state"] == "B6_p_C6_q0")
    schema = json.loads(row["pair_schema_json"])
    edges = [(int(item["left"]) - 1, int(item["right"]) - 1) for item in schema]
    kinds = [str(item["topology"]) for item in schema]
    expected = {(i, j) for i in range(12) for j in range(i + 1, 12)}
    if set(edges) != expected:
        raise ValueError("B6_p_C6 q0 manifest does not contain K12")
    same = [edge for edge, kind in zip(edges, kinds) if kind == "same_arm"]
    if len(same) != 6:
        raise ValueError(f"expected six same-arm pairs, found {len(same)}")
    terminal_to_monomer = [-1] * 12
    for arm, (left, right) in enumerate(sorted(same)):
        terminal_to_monomer[left] = arm
        terminal_to_monomer[right] = arm
    if min(terminal_to_monomer) < 0:
        raise ValueError("same-arm pairs do not cover all B6 termini")
    return edges, kinds, tuple(terminal_to_monomer)


def pair_vectors(
    path: Path,
    edge_to_index: Mapping[tuple[int, int], int],
    value_column: str,
) -> dict[str, np.ndarray]:
    grouped: dict[str, np.ndarray] = {}
    for row in read_csv(path):
        if row.get("system") != "B6_p_C6":
            continue
        state = row["state"]
        vector = grouped.setdefault(state, np.full(len(edge_to_index), np.nan, dtype=float))
        edge = tuple(sorted((int(row["left"]) - 1, int(row["right"]) - 1)))
        value = float(row[value_column])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"invalid {value_column} for {state} {edge}: {value}")
        vector[edge_to_index[edge]] = value
    return grouped


def replica_vectors(
    path: Path,
    edge_to_index: Mapping[tuple[int, int], int],
    value_column: str,
) -> dict[str, dict[int, np.ndarray]]:
    if not path.exists():
        return {}
    grouped: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for row in read_csv(path):
        if row.get("system") != "B6_p_C6":
            continue
        state = row["state"]
        replica = int(row["replica"])
        vector = grouped[state].setdefault(
            replica, np.full(len(edge_to_index), np.nan, dtype=float)
        )
        edge = tuple(sorted((int(row["left"]) - 1, int(row["right"]) - 1)))
        vector[edge_to_index[edge]] = float(row[value_column])
    return dict(grouped)


def state_indices(
    edges_one_based: Sequence[tuple[int, int]],
    edge_to_index: Mapping[tuple[int, int], int],
) -> tuple[int, ...]:
    edges = canonical_edges((a - 1, b - 1) for a, b in edges_one_based)
    return tuple(sorted(edge_to_index[edge] for edge in edges))


def build_model(
    root: Path,
    summary_paths: Sequence[Path],
    value_column: str = "soft_q_mean",
) -> B6Model:
    pair_vertices, pair_types, terminal_to_monomer = load_manifest(root)
    edge_to_index = {edge: index for index, edge in enumerate(pair_vertices)}
    if len(edge_to_index) != 66:
        raise ValueError("B6 edge table is not unique")

    sources_by_name: dict[str, KernelSource] = {}
    mean_by_name: dict[str, np.ndarray] = {}
    replica_by_name: dict[str, dict[int, np.ndarray]] = {}
    for path in summary_paths:
        for name, vector in pair_vectors(path, edge_to_index, value_column).items():
            if name in MEASURED_STATE_EDGES:
                mean_by_name[name] = vector
                sources_by_name[name] = KernelSource(
                    name, 20 if "q3" in name else 50, path
                )
        replica_path = path.with_name("pair_replica_summary.csv")
        replica_by_name.update(replica_vectors(replica_path, edge_to_index, value_column))

    missing = sorted(set(MEASURED_STATE_EDGES) - set(mean_by_name))
    if missing:
        raise FileNotFoundError(f"missing measured state summaries: {missing}")

    configs = shared.load_core_configs(root)
    td = np.asarray(configs["B6_p_C6"]["td"], dtype=float)
    left = np.asarray([edge[0] for edge in pair_vertices], dtype=np.int16)
    right = np.asarray([edge[1] for edge in pair_vertices], dtype=np.int16)
    same = np.asarray([kind == "same_arm" for kind in pair_types], dtype=bool)
    arm = np.asarray(terminal_to_monomer, dtype=np.int16)
    maximum = float(np.max(td))
    gamma_increment = np.empty((66, 12), dtype=float)
    alpha_increment = np.zeros((66, 66), dtype=np.uint8)
    pair_arm_left, pair_arm_right = arm[left], arm[right]
    for selected, (i, j) in enumerate(zip(left, right)):
        gamma_increment[selected] = maximum - np.minimum(td[i], td[j])
        arm_i, arm_j = int(arm[i]), int(arm[j])
        if arm_i != arm_j:
            alpha_increment[selected] = (
                ((pair_arm_left == arm_i) & (pair_arm_right == arm_j))
                | ((pair_arm_left == arm_j) & (pair_arm_right == arm_i))
            ).astype(np.uint8)

    data = baseline.SystemData(
        name="B6_p_C6",
        n_monomers=6,
        n_vinyl=12,
        pair_names=[f"V{a + 1}_V{b + 1}" for a, b in pair_vertices],
        pair_vertices=pair_vertices,
        pair_types=pair_types,
        same_edges=tuple(edge for edge, flag in zip(pair_vertices, same) if flag),
        terminal_to_monomer=terminal_to_monomer,
        cycle_ids=[], cycle_q={}, cycle_q_first={}, cycle_q_second={},
        cycle_soft_q=np.empty((0, 66)),
        cycle_soft_q_first=np.empty((0, 66)),
        cycle_soft_q_second=np.empty((0, 66)),
        distances=np.empty((0, 66)), frame_count=0, pdb_atoms=0, issues=[],
    )
    q0 = mean_by_name["B6_p_C6_q0"].copy()
    if np.isnan(q0).any():
        raise ValueError("q0 summary is missing one or more K12 edges")
    system = shared.PreparedSystem(
        data=data,
        topological_distances=td,
        base_pair_weights=q0,
        pair_left=left,
        pair_right=right,
        pair_same=same,
        gamma_increment=gamma_increment,
        alpha_increment=alpha_increment,
        target_rings=(2, 3, 4, 6),
        target_values=np.asarray([12, 12, 19, 53], dtype=float) / 96.0,
        target_ring=6,
    )

    observed_mean: dict[tuple[int, ...], np.ndarray] = {}
    observed_replicas: dict[tuple[int, ...], dict[int, np.ndarray]] = {}
    observed_sources: dict[tuple[int, ...], KernelSource] = {}
    for name, edges in MEASURED_STATE_EDGES.items():
        state = state_indices(edges, edge_to_index)
        vector = mean_by_name[name]
        consumed = {value - 1 for edge in edges for value in edge}
        active_indices = [
            index for index, (left_v, right_v) in enumerate(pair_vertices)
            if left_v not in consumed and right_v not in consumed
        ]
        if np.isnan(vector[active_indices]).any():
            raise ValueError(f"{name} is missing active pair values")
        observed_mean[state] = vector
        observed_replicas[state] = replica_by_name.get(name, {})
        observed_sources[state] = sources_by_name[name]

    edge_masks = tuple((1 << left_v) | (1 << right_v) for left_v, right_v in pair_vertices)
    return B6Model(
        system=system,
        edge_to_index=edge_to_index,
        edge_masks=edge_masks,
        observed_mean=observed_mean,
        observed_replicas=observed_replicas,
        observed_sources=observed_sources,
    )


def used_mask(state: Sequence[int], edge_masks: Sequence[int]) -> int:
    mask = 0
    for edge_index in state:
        mask |= int(edge_masks[int(edge_index)])
    return mask


def candidates_for(state: Sequence[int], model: B6Model) -> np.ndarray:
    mask = used_mask(state, model.edge_masks)
    return np.asarray(
        [index for index, edge_mask in enumerate(model.edge_masks) if not (mask & edge_mask)],
        dtype=np.int16,
    )


def transition_probabilities(
    model: B6Model,
    params: shared.Parameters,
    state: tuple[int, ...],
    candidates: np.ndarray,
    observed: Mapping[tuple[int, ...], np.ndarray],
    max_observed_depth: int,
) -> tuple[np.ndarray, str]:
    system = model.system
    selected = np.asarray(state, dtype=np.int16)
    if state:
        reuse = np.sum(system.alpha_increment[selected], axis=0, dtype=np.uint16)
    else:
        reuse = np.zeros(66, dtype=np.uint16)
    use_observed = state in observed and len(state) <= max_observed_depth
    if use_observed:
        base = np.asarray(observed[state][candidates], dtype=float).copy()
        if np.isnan(base).any():
            raise ValueError(f"observed kernel lacks a candidate at state {state}")
        log_weights = np.log(np.maximum(base, 1.0e-300))
        log_weights += math.log(params.epsilon0) * system.pair_same[candidates]
        log_weights -= params.alpha * reuse[candidates]
        source = "state_specific_md"
    else:
        exposure = (
            np.sum(system.gamma_increment[selected], axis=0, dtype=float)
            if state else np.zeros(12, dtype=float)
        )
        pair_exposure = (
            exposure[system.pair_left[candidates]]
            + exposure[system.pair_right[candidates]]
        )
        log_weights = np.log(
            np.maximum(system.base_pair_weights[candidates], 1.0e-300)
        )
        log_weights += math.log(params.epsilon0) * system.pair_same[candidates]
        log_weights -= params.gamma * pair_exposure
        log_weights -= params.alpha * reuse[candidates]
        source = "three_parameter_prior"
    log_weights -= float(np.max(log_weights))
    weights = np.exp(np.clip(log_weights, -700.0, 0.0))
    probabilities = weights / float(np.sum(weights))
    return probabilities, source


def child_state(state: tuple[int, ...], edge_index: int) -> tuple[int, ...]:
    return tuple(sorted((*state, int(edge_index))))


def state_text(state: Sequence[int], model: B6Model) -> str:
    edges = [model.system.data.pair_vertices[int(index)] for index in state]
    return ";".join(f"V{left + 1}-V{right + 1}" for left, right in sorted(edges)) or "q0"


def pair_text(edge_index: int, model: B6Model) -> str:
    left, right = model.system.data.pair_vertices[int(edge_index)]
    return f"V{left + 1}-V{right + 1}"


def propagate(
    model: B6Model,
    params: shared.Parameters,
    observed: Mapping[tuple[int, ...], np.ndarray],
    max_observed_depth: int,
) -> Propagation:
    forward_layers: list[dict[tuple[int, ...], float]] = [{(): 1.0}]
    for step in range(6):
        next_layer: defaultdict[tuple[int, ...], float] = defaultdict(float)
        for state, state_mass in forward_layers[-1].items():
            candidates = candidates_for(state, model)
            probabilities, _ = transition_probabilities(
                model, params, state, candidates, observed, max_observed_depth
            )
            for edge_index, probability in zip(candidates, probabilities):
                next_layer[child_state(state, int(edge_index))] += (
                    state_mass * float(probability)
                )
        mass = float(sum(next_layer.values()))
        if not math.isclose(mass, 1.0, abs_tol=1.0e-11):
            raise RuntimeError(f"probability loss at q{step + 1}: {mass}")
        forward_layers.append(dict(next_layer))

    terminal: dict[tuple[int, ...], np.ndarray] = {}
    distribution = np.zeros(7, dtype=float)
    for state, probability in forward_layers[6].items():
        matching = [model.system.data.pair_vertices[index] for index in state]
        product = baseline.component_mass_fractions(
            matching, model.system.data.terminal_to_monomer, 6
        )
        terminal[state] = product
        distribution += probability * product
    if not math.isclose(float(distribution.sum()), 1.0, abs_tol=1.0e-11):
        raise RuntimeError("product distribution does not normalize")

    backward_layers: list[dict[tuple[int, ...], np.ndarray]] = [{} for _ in range(7)]
    backward_layers[6] = terminal
    priority_rows: list[dict[str, object]] = []
    for step in range(5, -1, -1):
        current: dict[tuple[int, ...], np.ndarray] = {}
        for state, state_mass in forward_layers[step].items():
            candidates = candidates_for(state, model)
            probabilities, source = transition_probabilities(
                model, params, state, candidates, observed, max_observed_depth
            )
            children = [backward_layers[step + 1][child_state(state, int(edge))] for edge in candidates]
            conditional = np.sum(
                np.stack(children) * probabilities[:, None], axis=0
            )
            current[state] = conditional
            tv = np.asarray(
                [0.5 * np.abs(child - conditional).sum() for child in children],
                dtype=float,
            )
            positive = probabilities[probabilities > 0.0]
            entropy = -float(np.sum(positive * np.log(positive)))
            normalized_entropy = entropy / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
            top = int(np.argmax(probabilities))
            priority_rows.append(
                {
                    "q": step,
                    "state": state_text(state, model),
                    "state_edge_indices": ";".join(str(index) for index in state),
                    "kernel_source": source,
                    "direct_md": source == "state_specific_md",
                    "reach_probability": state_mass,
                    "outgoing_candidates": len(candidates),
                    "normalized_outgoing_entropy": normalized_entropy,
                    "top_next_pair": pair_text(int(candidates[top]), model),
                    "top_next_probability": float(probabilities[top]),
                    "conditional_R2": float(conditional[2]),
                    "conditional_R3": float(conditional[3]),
                    "conditional_R4": float(conditional[4]),
                    "conditional_R6": float(conditional[6]),
                    "expected_child_tv": float(np.sum(probabilities * tv)),
                    "max_child_tv": float(np.max(tv)),
                    "priority_expected_tv": state_mass * float(np.sum(probabilities * tv)),
                    "priority_max_bound": state_mass * float(np.max(tv)),
                }
            )
        backward_layers[step] = current

    coverage_rows: list[dict[str, object]] = []
    for step, layer in enumerate(forward_layers[:-1]):
        measured = [
            state for state in layer
            if state in observed and len(state) <= max_observed_depth
        ]
        coverage_rows.append(
            {
                "q": step,
                "reachable_states": len(layer),
                "direct_md_states": len(measured),
                "direct_md_state_mass": sum(layer[state] for state in measured),
                "prior_state_mass": 1.0 - sum(layer[state] for state in measured),
            }
        )
    return Propagation(
        forward_layers=forward_layers,
        backward_layers=backward_layers,
        distribution=distribution,
        priority_rows=priority_rows,
        coverage_rows=coverage_rows,
    )


def propagate_distribution_only(
    model: B6Model,
    params: shared.Parameters,
    observed: Mapping[tuple[int, ...], np.ndarray],
    max_observed_depth: int,
) -> np.ndarray:
    """Fast forward-only variant used for replica bootstrap."""
    layer: dict[tuple[int, ...], float] = {(): 1.0}
    for _step in range(6):
        next_layer: defaultdict[tuple[int, ...], float] = defaultdict(float)
        for state, state_mass in layer.items():
            candidates = candidates_for(state, model)
            probabilities, _ = transition_probabilities(
                model, params, state, candidates, observed, max_observed_depth
            )
            for edge_index, probability in zip(candidates, probabilities):
                next_layer[child_state(state, int(edge_index))] += (
                    state_mass * float(probability)
                )
        layer = dict(next_layer)
    distribution = np.zeros(7, dtype=float)
    for state, probability in layer.items():
        matching = [model.system.data.pair_vertices[index] for index in state]
        distribution += probability * baseline.component_mass_fractions(
            matching, model.system.data.terminal_to_monomer, 6
        )
    if not math.isclose(float(distribution.sum()), 1.0, abs_tol=1.0e-11):
        raise RuntimeError("bootstrap product distribution does not normalize")
    return distribution


def distribution_row(
    scenario: str,
    parameter_source: str,
    max_depth: int,
    params: shared.Parameters,
    distribution: np.ndarray,
) -> list[dict[str, object]]:
    reported = float(distribution[[2, 3, 4, 6]].sum())
    experimental = {2: 12 / 96, 3: 12 / 96, 4: 19 / 96, 6: 53 / 96}
    rows: list[dict[str, object]] = []
    for ring in range(1, 7):
        conditioned = float(distribution[ring] / reported) if ring in experimental else math.nan
        rows.append(
            {
                "scenario": scenario,
                "parameter_source": parameter_source,
                "max_md_q": max_depth,
                "epsilon0": params.epsilon0,
                "gamma_prior_only": params.gamma,
                "alpha": params.alpha,
                "ring": f"R{ring}",
                "all_product_mass": float(distribution[ring]),
                "conditioned_reported_fraction": conditioned,
                "experimental_conditioned_fraction": experimental.get(ring, math.nan),
                "absolute_error_pp": (
                    100.0 * abs(conditioned - experimental[ring])
                    if ring in experimental else math.nan
                ),
                "mass_outside_R2_R3_R4_R6": 1.0 - reported,
            }
        )
    return rows


def bootstrap_observed(
    model: B6Model,
    rng: np.random.Generator,
) -> dict[tuple[int, ...], np.ndarray]:
    output: dict[tuple[int, ...], np.ndarray] = {}
    for state, mean in model.observed_mean.items():
        replicas = model.observed_replicas.get(state, {})
        if len(replicas) < 2:
            output[state] = mean
            continue
        keys = sorted(replicas)
        chosen = rng.choice(keys, size=len(keys), replace=True)
        stack = np.stack([replicas[int(key)] for key in chosen])
        valid = np.isfinite(stack)
        counts = np.sum(valid, axis=0)
        vector = np.full(stack.shape[1], np.nan, dtype=float)
        np.divide(
            np.nansum(stack, axis=0), counts,
            out=vector, where=counts > 0,
        )
        output[state] = vector
    return output


def write_report(
    path: Path,
    distributions: Sequence[Mapping[str, object]],
    coverages: Mapping[str, Sequence[Mapping[str, object]]],
    priorities: Mapping[str, Sequence[Mapping[str, object]]],
    bootstrap_rows: Sequence[Mapping[str, object]],
    kernel_value_column: str,
) -> None:
    by_scenario: dict[str, dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in distributions:
        by_scenario[str(row["scenario"])][str(row["ring"])] = row
    lines = [
        "# B6_p_C6：混合 MD/参数模型的完整产物分布审计",
        "",
        "## 结论先行",
        "",
        "本报告在完整的 12 端点部分匹配 DAG 上传播全部六次成键，而不是只跟随一条主路径。已模拟节点使用状态特异 MD 条件核；未模拟节点使用三参数先验。MD 接触是构象机会，不是绝对速率，因此本结果属于模型敏感性与计算优先级审计。",
        "",
        "## 产物分布",
        "",
        "下表均对实验报告的 R2/R3/R4/R6 四类重新归一化。",
        "",
        "| 场景 | R2 | R3 | R4 | R6 | 未报告 R1+R5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for scenario, rows in by_scenario.items():
        lines.append(
            f"| {scenario} | "
            f"{100*float(rows['R2']['conditioned_reported_fraction']):.2f}% | "
            f"{100*float(rows['R3']['conditioned_reported_fraction']):.2f}% | "
            f"{100*float(rows['R4']['conditioned_reported_fraction']):.2f}% | "
            f"{100*float(rows['R6']['conditioned_reported_fraction']):.2f}% | "
            f"{100*float(rows['R6']['mass_outside_R2_R3_R4_R6']):.2f}% |"
        )
    lines.extend([
        "| 实验 | 12.50% | 12.50% | 19.79% | 55.21% | 未报告 |",
        "",
        "## 直接 MD 覆盖率",
        "",
    ])
    for scenario, rows in coverages.items():
        lines.extend([f"### {scenario}", "", "| q | 可达节点 | MD 节点 | MD 覆盖概率质量 |", "|---:|---:|---:|---:|"])
        for row in rows:
            lines.append(
                f"| {row['q']} | {row['reachable_states']} | {row['direct_md_states']} | "
                f"{100*float(row['direct_md_state_mass']):.4f}% |"
            )
        lines.append("")
    lines.extend([
        "## 下一批状态的选择原则",
        "",
        "优先级不是简单的接触排名，而是 `节点到达概率 × 该节点不同下一步对最终产物分布的分歧`。下列为每个参数场景中尚未模拟节点的前十名。",
        "",
    ])
    for scenario, rows in priorities.items():
        lines.extend([f"### {scenario}", "", "| 排名 | q | 状态 | 到达概率 | top 下一步 | 局部影响分数 |", "|---:|---:|---|---:|---|---:|"])
        selected = sorted(
            (row for row in rows if not bool(row["direct_md"]) and 1 <= int(row["q"]) <= 4),
            key=lambda row: float(row["priority_expected_tv"]), reverse=True,
        )[:10]
        for rank, row in enumerate(selected, 1):
            lines.append(
                f"| {rank} | {row['q']} | `{row['state']}` | "
                f"{100*float(row['reach_probability']):.3f}% | {row['top_next_pair']} | "
                f"{float(row['priority_expected_tv']):.6f} |"
            )
        lines.append("")
    q1_rankings: dict[str, list[Mapping[str, object]]] = {}
    for scenario, rows in priorities.items():
        q1_rankings[scenario] = sorted(
            (row for row in rows if int(row["q"]) == 1 and not bool(row["direct_md"])),
            key=lambda row: float(row["priority_expected_tv"]), reverse=True,
        )
    if len(q1_rankings) >= 2:
        rank_maps = {
            scenario: {str(row["state"]): rank for rank, row in enumerate(rows, 1)}
            for scenario, rows in q1_rankings.items()
        }
        all_states = set().union(*(set(ranks) for ranks in rank_maps.values()))
        consensus = sorted(
            all_states,
            key=lambda state: (
                sum(ranks.get(state, len(ranks) + 1) for ranks in rank_maps.values()),
                state,
            ),
        )[:6]
        existing_mass = float(next(iter(coverages.values()))[1]["direct_md_state_mass"])
        cumulative = existing_mass
        lines.extend([
            "## 自适应最小计算批次",
            "",
            "两个参数情景共同认为，当前最该补的是 q1 覆盖，而不是继续沿现有路径构建 q4。下表按两个情景的 q1 影响排名和排序；第一批只做前两项，得到新条件核后重新传播，再决定是否启动第二批。",
            "",
            "| 顺序 | q1 状态 | legacy 排名 | frozen 排名 | 平均到达概率 | 加上现有 V3–V6 后累计 q1 覆盖 | 建议 |",
            "|---:|---|---:|---:|---:|---:|---|",
        ])
        for order, state in enumerate(consensus, 1):
            records = [
                next(row for row in rows if str(row["state"]) == state)
                for rows in q1_rankings.values() if any(str(row["state"]) == state for row in rows)
            ]
            reach = float(np.mean([float(row["reach_probability"]) for row in records]))
            cumulative += reach
            ranks = [rank_maps[scenario].get(state, len(rank_maps[scenario]) + 1) for scenario in rank_maps]
            advice = "第一批" if order <= 2 else ("第二批候选" if order <= 4 else "暂缓")
            lines.append(
                f"| {order} | `{state}` | {ranks[0]} | {ranks[1]} | "
                f"{100*reach:.2f}% | {100*cumulative:.2f}% | {advice} |"
            )
        lines.extend([
            "",
            "第一批建议为 `q1 V11–V12` 和 `q1 V9–V10`，各做 3 × 20 ns。完成后先重算首事件核和完整产物分布；若 R2/R3/R4/R6 仍对先验敏感，再考虑 `q1 V6–V8` 与 `q1 V3–V8`。",
            "",
        ])
    if bootstrap_rows:
        lines.extend([
            "## Replica bootstrap",
            "",
            "固定参数不重新拟合；对每个已模拟状态的三条 replica 进行有放回重采样，再传播完整反应图。区间只反映当前 replica 接触谱的不确定性。",
            "",
            "| ring | 2.5% | 中位数 | 97.5% |",
            "|---|---:|---:|---:|",
        ])
        for row in bootstrap_rows:
            lines.append(
                f"| {row['ring']} | {100*float(row['ci95_low']):.2f}% | "
                f"{100*float(row['median']):.2f}% | {100*float(row['ci95_high']):.2f}% |"
            )
    lines.extend([
        "",
        "## 方法边界",
        "",
        (
            "- `legacy` 参数原本配合 25 ps 短循环、多起始构象簇的首事件权重逐体系拟合；"
            + ("这里使用相同首事件定义，但采样来自三条长轨迹的 25 ps 区间，且存帧更稀，因此仍只是参数转移情景，不是原模型复现。" if "first_event" in kernel_value_column else "这里改用了长轨迹平均软接触谱，因此只能作为参数转移敏感性情景。")
        ),
        (
            "- `frozen external` 参数不读取 B6_p_C6 标签；"
            + ("其首事件定义一致，但从短循环构象簇迁移到长轨迹区间后的可迁移性仍未经验证。" if "first_event" in kernel_value_column else "它原本配合短循环首事件权重，转移到平均软接触谱后的可迁移性未经验证。")
        ),
        "- 在 MD 节点移除 gamma 是为了避免把已显式观测到的历史重排重复计算；alpha 仍作为臂对复用的化学/拓扑先验保留。",
        "- 目前只有一个 q1、一个 q2 和两个 q3 节点有反应后 MD，因此不能把混合模型称为无模型的产率预测。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--q0", type=Path, default=Path(DEFAULT_Q0))
    parser.add_argument("--q1", type=Path, default=Path(DEFAULT_Q1))
    parser.add_argument("--q2", type=Path, default=Path(DEFAULT_Q2))
    parser.add_argument("--q3", type=Path, default=Path(DEFAULT_Q3))
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument("--bootstrap", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--kernel-value-column", default="soft_q_mean")
    parser.add_argument("--priority-top-per-q", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    paths = [root / path if not path.is_absolute() else path for path in (args.q0, args.q1, args.q2, args.q3)]
    output = root / args.output if not args.output.is_absolute() else args.output
    output.mkdir(parents=True, exist_ok=True)
    model = build_model(root, paths, args.kernel_value_column)

    parameter_sets = {
        "legacy_fit_transferred_to_long_md": shared.Parameters(0.002316, 0.7, 8.1),
        "frozen_external_transferred_to_long_md": shared.Parameters(0.3, 0.5, -1.0),
    }
    distribution_rows: list[dict[str, object]] = []
    coverages: dict[str, list[dict[str, object]]] = {}
    priorities: dict[str, list[dict[str, object]]] = {}
    propagations: dict[str, Propagation] = {}
    for parameter_source, params in parameter_sets.items():
        for depth in (-1, 1, 2, 3):
            scenario = f"{parameter_source}__{'prior_only' if depth < 0 else f'md_through_q{depth}'}"
            propagation = propagate(model, params, model.observed_mean, depth)
            propagations[scenario] = propagation
            distribution_rows.extend(
                distribution_row(scenario, parameter_source, depth, params, propagation.distribution)
            )
            if depth == 3:
                coverages[scenario] = propagation.coverage_rows
                priorities[scenario] = propagation.priority_rows

    bootstrap_rows: list[dict[str, object]] = []
    if args.bootstrap > 0:
        rng = np.random.default_rng(args.seed)
        samples: list[np.ndarray] = []
        params = parameter_sets["legacy_fit_transferred_to_long_md"]
        original_q0 = model.system.base_pair_weights.copy()
        for _ in range(args.bootstrap):
            observed = bootstrap_observed(model, rng)
            model.system.base_pair_weights = observed[()]
            result = propagate_distribution_only(model, params, observed, 3)
            reported = float(result[[2, 3, 4, 6]].sum())
            samples.append(result / reported)
        model.system.base_pair_weights = original_q0
        array = np.stack(samples)
        for ring in (2, 3, 4, 6):
            low, median, high = np.quantile(array[:, ring], [0.025, 0.5, 0.975])
            bootstrap_rows.append(
                {"ring": f"R{ring}", "ci95_low": low, "median": median, "ci95_high": high, "replicates": args.bootstrap}
            )

    write_csv(output / "product_distributions.csv", distribution_rows)
    coverage_rows = [dict(row, scenario=scenario) for scenario, rows in coverages.items() for row in rows]
    write_csv(output / "md_coverage.csv", coverage_rows)
    priority_rows: list[dict[str, object]] = []
    for scenario, rows in priorities.items():
        direct = [row for row in rows if bool(row["direct_md"])]
        selected = list(direct)
        for q in range(6):
            layer = [row for row in rows if int(row["q"]) == q and not bool(row["direct_md"])]
            selected.extend(sorted(
                layer, key=lambda row: float(row["priority_expected_tv"]), reverse=True
            )[:args.priority_top_per_q])
        priority_rows.extend(dict(row, scenario=scenario) for row in selected)
    write_csv(output / "state_priorities.csv", priority_rows)
    if bootstrap_rows:
        write_csv(output / "replica_bootstrap.csv", bootstrap_rows)
    write_report(
        output / "B6_p_C6_完整产物分布与分支覆盖审计.md",
        distribution_rows, coverages, priorities, bootstrap_rows,
        args.kernel_value_column,
    )
    metadata = {
        "model": "hybrid_state_specific_md_on_exact_partial_matching_dag",
        "measured_states": {
            state_text(state, model): {
                "state_name": source.state_name,
                "trajectory_ns": source.trajectory_ns,
                "summary_path": str(source.summary_path),
                "replicas": len(model.observed_replicas.get(state, {})),
            }
            for state, source in model.observed_sources.items()
        },
        "parameter_sets": {
            name: {"epsilon0": p.epsilon0, "gamma": p.gamma, "alpha": p.alpha}
            for name, p in parameter_sets.items()
        },
        "bootstrap_replicates": args.bootstrap,
        "kernel_value_column": args.kernel_value_column,
        "reported_classes": [2, 3, 4, 6],
        "experimental_conditioned": {"R2": 12/96, "R3": 12/96, "R4": 19/96, "R6": 53/96},
    }
    (output / "model_contract.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote hybrid B6 distribution audit to {output}")


if __name__ == "__main__":
    main()
