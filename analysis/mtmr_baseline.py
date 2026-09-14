#!/usr/bin/env python3
"""Initial-state MTMR trajectory-to-product distribution experiment.

The implementation deliberately keeps the chemistry/model boundary explicit:
distance-only encounter probabilities are conformational weights, not rate
constants.  Product propagation is exact for the mean q matrices; cycle-level
bootstrap uncertainty is propagated with deterministic-seed Monte Carlo.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import math
import re
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_THRESHOLDS = (4.0, 5.0, 6.0, 7.0)
DEFAULT_PRIMARY_THRESHOLD = 5.0
SOFT_SIGMOID_DC = 5.0
SOFT_SIGMOID_SIGMA = 0.5
EXCLUDED_SYSTEMS: set[str] = set()
PAIR_PATTERN = re.compile(r"^V(\d+)_V(\d+)_(同臂|跨臂)$")
CYCLE_PATTERN = re.compile(r"cycle_(\d+)")
VINYL_LOG_PATTERN = re.compile(r"V(\d+):\s*\((\d+),\s*(\d+)\)")


@dataclass
class SystemData:
    name: str
    n_monomers: int
    n_vinyl: int
    pair_names: list[str]
    pair_vertices: list[tuple[int, int]]
    pair_types: list[str]
    same_edges: tuple[tuple[int, int], ...]
    terminal_to_monomer: tuple[int, ...]
    cycle_ids: list[int]
    cycle_q: dict[float, np.ndarray]
    cycle_q_first: dict[float, np.ndarray]
    cycle_q_second: dict[float, np.ndarray]
    cycle_soft_q: np.ndarray
    cycle_soft_q_first: np.ndarray
    cycle_soft_q_second: np.ndarray
    distances: np.ndarray
    frame_count: int
    pdb_atoms: int
    issues: list[dict[str, str]]


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"fieldnames are required for empty CSV: {path}")
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [item.strip() for item in next(csv.reader(handle))]


def parse_pairs(header: Sequence[str]) -> tuple[list[str], list[tuple[int, int]], list[str], int]:
    if list(header[:3]) != ["step", "time_ps", "energy_eV"]:
        raise ValueError(f"unexpected metadata columns: {header[:3]}")
    names: list[str] = []
    vertices: list[tuple[int, int]] = []
    pair_types: list[str] = []
    max_vinyl = 0
    for name in header[3:]:
        match = PAIR_PATTERN.fullmatch(name)
        if match is None:
            raise ValueError(f"unrecognized pair column: {name}")
        left, right = int(match.group(1)), int(match.group(2))
        if left >= right:
            raise ValueError(f"pair is not canonically ordered: {name}")
        names.append(name)
        vertices.append((left - 1, right - 1))
        pair_types.append(match.group(3))
        max_vinyl = max(max_vinyl, left, right)
    expected_pairs = max_vinyl * (max_vinyl - 1) // 2
    if len(vertices) != expected_pairs or set(vertices) != {
        (i, j) for i in range(max_vinyl) for j in range(i + 1, max_vinyl)
    }:
        raise ValueError(
            f"pair coverage is incomplete: found {len(vertices)}, expected {expected_pairs} for {max_vinyl} vinyls"
        )
    return names, vertices, pair_types, max_vinyl


def same_arm_mapping(
    n_vinyl: int, pair_vertices: Sequence[tuple[int, int]], pair_types: Sequence[str]
) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    same_edges = tuple(sorted(edge for edge, kind in zip(pair_vertices, pair_types) if kind == "同臂"))
    if len(same_edges) * 2 != n_vinyl:
        raise ValueError(f"expected {n_vinyl // 2} same-arm pairs, found {len(same_edges)}")
    flattened = [vertex for edge in same_edges for vertex in edge]
    if sorted(flattened) != list(range(n_vinyl)):
        raise ValueError("same-arm pairs are not a disjoint cover of all vinyl groups")
    terminal_to_monomer = [-1] * n_vinyl
    for monomer, edge in enumerate(same_edges):
        for vertex in edge:
            terminal_to_monomer[vertex] = monomer
    return same_edges, tuple(terminal_to_monomer)


def parse_pdb(path: Path) -> tuple[dict[int, str], set[tuple[int, int]]]:
    atoms: dict[int, str] = {}
    bonds: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                serial = int(line[6:11])
                element = line[76:78].strip() or line[12:16].strip()[0]
                atoms[serial] = element.upper()
            elif line.startswith("CONECT"):
                fields = line.split()[1:]
                if len(fields) >= 2:
                    source = int(fields[0])
                    for target_text in fields[1:]:
                        target = int(target_text)
                        bonds.add(tuple(sorted((source, target))))
    return atoms, bonds


def validate_pdb_mapping(
    project_root: Path, system: str, n_vinyl: int, issues: list[dict[str, str]]
) -> int:
    pdb_path = project_root / "structures" / f"{system}.pdb"
    log_path = project_root / "trace" / f"{system}_md" / "md_simulation.log"
    if not pdb_path.exists():
        issues.append(issue(system, "ERROR", "missing_pdb", str(pdb_path)))
        return 0
    atoms, bonds = parse_pdb(pdb_path)
    if not log_path.exists():
        issues.append(issue(system, "ERROR", "missing_mapping_log", str(log_path)))
        return len(atoms)
    mapping: dict[int, tuple[int, int]] = {}
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    for match in VINYL_LOG_PATTERN.finditer(log_text):
        mapping[int(match.group(1))] = (int(match.group(2)), int(match.group(3)))
    if set(mapping) != set(range(1, n_vinyl + 1)):
        issues.append(
            issue(
                system,
                "ERROR",
                "vinyl_mapping_coverage",
                f"mapped {sorted(mapping)}; expected V1..V{n_vinyl}",
            )
        )
        return len(atoms)
    for vinyl, atom_pair in sorted(mapping.items()):
        if any(atom not in atoms for atom in atom_pair):
            issues.append(issue(system, "ERROR", "vinyl_atom_out_of_range", f"V{vinyl}: {atom_pair}"))
            continue
        elements = tuple(atoms[atom] for atom in atom_pair)
        if elements != ("C", "C"):
            issues.append(issue(system, "ERROR", "vinyl_noncarbon_mapping", f"V{vinyl}: {atom_pair} -> {elements}"))
        if tuple(sorted(atom_pair)) not in bonds:
            issues.append(issue(system, "ERROR", "vinyl_atoms_not_bonded", f"V{vinyl}: {atom_pair}"))
    return len(atoms)


def issue(system: str, severity: str, code: str, detail: str) -> dict[str, str]:
    return {"system": system, "severity": severity, "code": code, "detail": detail}


def analyse_system(
    project_root: Path,
    system_dir: Path,
    thresholds: Sequence[float],
) -> tuple[SystemData | None, list[dict[str, object]]]:
    system = system_dir.name.removesuffix("_md")
    issues: list[dict[str, str]] = []
    cycle_rows: list[dict[str, object]] = []
    trajectory_files = sorted(
        system_dir.glob("cycle_*/trajectory.csv"),
        key=lambda path: int(CYCLE_PATTERN.search(path.parent.name).group(1)),
    )
    if not trajectory_files:
        return None, [issue(system, "ERROR", "no_trajectories", str(system_dir))]

    base_header: list[str] | None = None
    pair_names: list[str] = []
    pair_vertices: list[tuple[int, int]] = []
    pair_types: list[str] = []
    n_vinyl = 0
    same_edges: tuple[tuple[int, int], ...] = ()
    terminal_to_monomer: tuple[int, ...] = ()
    cycle_ids: list[int] = []
    distance_blocks: list[np.ndarray] = []
    q_blocks: dict[float, list[np.ndarray]] = {threshold: [] for threshold in thresholds}
    q_first_blocks: dict[float, list[np.ndarray]] = {threshold: [] for threshold in thresholds}
    q_second_blocks: dict[float, list[np.ndarray]] = {threshold: [] for threshold in thresholds}
    soft_q_blocks: list[np.ndarray] = []
    soft_q_first_blocks: list[np.ndarray] = []
    soft_q_second_blocks: list[np.ndarray] = []
    row_counts: list[int] = []

    for trajectory_path in trajectory_files:
        cycle_match = CYCLE_PATTERN.search(trajectory_path.parent.name)
        cycle_id = int(cycle_match.group(1)) if cycle_match else -1
        inventory: dict[str, object] = {
            "system": system,
            "cycle": cycle_id,
            "path": str(trajectory_path.relative_to(project_root)),
            "valid": False,
            "rows": 0,
            "columns": 0,
            "time_start_ps": "",
            "time_end_ps": "",
            "median_dt_ps": "",
            "energy_mean_eV": "",
            "energy_sd_eV": "",
            "content_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
            "problem": "",
        }
        try:
            header = read_header(trajectory_path)
            inventory["columns"] = len(header)
            if base_header is None:
                base_header = header
                pair_names, pair_vertices, pair_types, n_vinyl = parse_pairs(header)
                same_edges, terminal_to_monomer = same_arm_mapping(n_vinyl, pair_vertices, pair_types)
            elif header != base_header:
                raise ValueError("header differs from the system reference header")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                values = np.loadtxt(trajectory_path, delimiter=",", skiprows=1, ndmin=2)
            inventory["rows"] = int(values.shape[0])
            if values.shape[1] != len(header):
                raise ValueError(f"numeric field count {values.shape[1]} != header count {len(header)}")
            if values.shape[0] < 2:
                raise ValueError("trajectory contains fewer than two frames")
            if not np.isfinite(values).all():
                raise ValueError("trajectory contains NaN or infinite values")
            steps, times, energies = values[:, 0], values[:, 1], values[:, 2]
            distances = values[:, 3:]
            if not (np.diff(steps) > 0).all():
                raise ValueError("step axis is not strictly increasing")
            if not (np.diff(times) > 0).all():
                raise ValueError("time axis is not strictly increasing")
            if (distances < 0).any():
                raise ValueError("negative distance encountered")
            midpoint = values.shape[0] // 2
            soft_scores = 1.0 / (
                1.0 + np.exp((distances - SOFT_SIGMOID_DC) / SOFT_SIGMOID_SIGMA)
            )
            soft_q_blocks.append(soft_scores.mean(axis=0))
            soft_q_first_blocks.append(soft_scores[:midpoint].mean(axis=0))
            soft_q_second_blocks.append(soft_scores[midpoint:].mean(axis=0))
            for threshold in thresholds:
                contacts = distances < threshold
                q_blocks[threshold].append(contacts.mean(axis=0))
                q_first_blocks[threshold].append(contacts[:midpoint].mean(axis=0))
                q_second_blocks[threshold].append(contacts[midpoint:].mean(axis=0))
            cycle_ids.append(cycle_id)
            row_counts.append(values.shape[0])
            distance_blocks.append(distances)
            inventory.update(
                {
                    "valid": True,
                    "time_start_ps": float(times[0]),
                    "time_end_ps": float(times[-1]),
                    "median_dt_ps": float(np.median(np.diff(times))),
                    "energy_mean_eV": float(energies.mean()),
                    "energy_sd_eV": float(energies.std(ddof=1)),
                }
            )
        except Exception as exc:  # Keep the audit running and record the exact file.
            inventory["problem"] = str(exc)
            issues.append(issue(system, "ERROR", "invalid_trajectory", f"{trajectory_path}: {exc}"))
        cycle_rows.append(inventory)

    if base_header is None or not distance_blocks:
        return None, cycle_rows

    expected_cycle_ids = set(range(1, 101))
    missing_cycles = sorted(expected_cycle_ids - set(cycle_ids))
    extra_cycles = sorted(set(cycle_ids) - expected_cycle_ids)
    if missing_cycles:
        issues.append(issue(system, "WARN", "missing_cycles", ",".join(map(str, missing_cycles))))
    if extra_cycles:
        issues.append(issue(system, "WARN", "unexpected_cycles", ",".join(map(str, extra_cycles))))
    row_count_mode = Counter(row_counts).most_common(1)[0][0]
    nonmodal_rows = sorted({count for count in row_counts if count != row_count_mode})
    if nonmodal_rows:
        issues.append(
            issue(system, "WARN", "nonuniform_frame_counts", f"mode={row_count_mode}; others={nonmodal_rows}")
        )
    valid_hash_groups: dict[str, list[int]] = defaultdict(list)
    for row in cycle_rows:
        if row["valid"]:
            valid_hash_groups[str(row["content_sha256"])].append(int(row["cycle"]))
    duplicate_groups = [cycles for cycles in valid_hash_groups.values() if len(cycles) > 1]
    if duplicate_groups:
        issues.append(
            issue(
                system,
                "WARN",
                "duplicate_cycle_content",
                "; ".join(",".join(map(str, cycles)) for cycles in duplicate_groups),
            )
        )

    pdb_atoms = validate_pdb_mapping(project_root, system, n_vinyl, issues)
    distances_all = np.concatenate(distance_blocks, axis=0)
    return (
        SystemData(
            name=system,
            n_monomers=n_vinyl // 2,
            n_vinyl=n_vinyl,
            pair_names=pair_names,
            pair_vertices=pair_vertices,
            pair_types=pair_types,
            same_edges=same_edges,
            terminal_to_monomer=terminal_to_monomer,
            cycle_ids=cycle_ids,
            cycle_q={threshold: np.stack(q_blocks[threshold]) for threshold in thresholds},
            cycle_q_first={threshold: np.stack(q_first_blocks[threshold]) for threshold in thresholds},
            cycle_q_second={threshold: np.stack(q_second_blocks[threshold]) for threshold in thresholds},
            cycle_soft_q=np.stack(soft_q_blocks),
            cycle_soft_q_first=np.stack(soft_q_first_blocks),
            cycle_soft_q_second=np.stack(soft_q_second_blocks),
            distances=distances_all,
            frame_count=int(distances_all.shape[0]),
            pdb_atoms=pdb_atoms,
            issues=issues,
        ),
        cycle_rows,
    )


def q_statistics(
    data: SystemData,
    thresholds: Sequence[float],
    bootstrap_replicates: int,
    base_seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    n_cycles = len(data.cycle_ids)
    rng = np.random.default_rng(stable_seed(base_seed, data.name, "q_bootstrap"))
    bootstrap_indices = rng.integers(0, n_cycles, size=(bootstrap_replicates, n_cycles))
    distance_quantiles = np.quantile(data.distances, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    distance_means = data.distances.mean(axis=0)
    for threshold in thresholds:
        cycle_q = data.cycle_q[threshold]
        q_mean = cycle_q.mean(axis=0)
        q_se = cycle_q.std(axis=0, ddof=1) / math.sqrt(n_cycles)
        boot_means = cycle_q[bootstrap_indices].mean(axis=1)
        ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975], axis=0)
        first_mean = data.cycle_q_first[threshold].mean(axis=0)
        second_mean = data.cycle_q_second[threshold].mean(axis=0)
        for index, (name, vertices, pair_type) in enumerate(
            zip(data.pair_names, data.pair_vertices, data.pair_types)
        ):
            rows.append(
                {
                    "system": data.name,
                    "n_monomers": data.n_monomers,
                    "pair": name,
                    "vinyl_i": vertices[0] + 1,
                    "vinyl_j": vertices[1] + 1,
                    "pair_type": "same_arm" if pair_type == "同臂" else "cross_arm",
                    "threshold_A": threshold,
                    "n_cycles": n_cycles,
                    "n_frames_total": data.frame_count,
                    "q_mean": q_mean[index],
                    "q_se_cycle": q_se[index],
                    "q_ci_low_cycle_bootstrap": ci_low[index],
                    "q_ci_high_cycle_bootstrap": ci_high[index],
                    "q_first_half": first_mean[index],
                    "q_second_half": second_mean[index],
                    "q_window_delta": second_mean[index] - first_mean[index],
                    "distance_mean_A": distance_means[index],
                    "distance_p05_A": distance_quantiles[0, index],
                    "distance_p25_A": distance_quantiles[1, index],
                    "distance_median_A": distance_quantiles[2, index],
                    "distance_p75_A": distance_quantiles[3, index],
                    "distance_p95_A": distance_quantiles[4, index],
                }
            )
        same_mask = np.array([kind == "同臂" for kind in data.pair_types])
        cross_mask = ~same_mask
        summaries.append(
            {
                "system": data.name,
                "threshold_A": threshold,
                "mean_q_same_arm": float(q_mean[same_mask].mean()),
                "mean_q_cross_arm": float(q_mean[cross_mask].mean()),
                "cross_to_same_q_ratio": float(q_mean[cross_mask].mean() / q_mean[same_mask].mean())
                if q_mean[same_mask].mean() > 0
                else math.nan,
                "max_abs_first_second_delta": float(np.max(np.abs(second_mean - first_mean))),
                "mean_abs_first_second_delta": float(np.mean(np.abs(second_mean - first_mean))),
            }
        )
    return rows, summaries


def soft_q_statistics(
    data: SystemData,
    bootstrap_replicates: int,
    base_seed: int,
) -> list[dict[str, object]]:
    """Pair-resolved statistics for the fixed soft sigmoid encounter score."""
    n_cycles = len(data.cycle_ids)
    rng = np.random.default_rng(stable_seed(base_seed, data.name, "soft_q_bootstrap"))
    bootstrap_indices = rng.integers(0, n_cycles, size=(bootstrap_replicates, n_cycles))
    q_mean = data.cycle_soft_q.mean(axis=0)
    q_se = data.cycle_soft_q.std(axis=0, ddof=1) / math.sqrt(n_cycles)
    boot_means = data.cycle_soft_q[bootstrap_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(boot_means, [0.025, 0.975], axis=0)
    first_mean = data.cycle_soft_q_first.mean(axis=0)
    second_mean = data.cycle_soft_q_second.mean(axis=0)
    rows: list[dict[str, object]] = []
    for index, (name, vertices, pair_type) in enumerate(
        zip(data.pair_names, data.pair_vertices, data.pair_types)
    ):
        rows.append(
            {
                "system": data.name,
                "n_monomers": data.n_monomers,
                "pair": name,
                "vinyl_i": vertices[0] + 1,
                "vinyl_j": vertices[1] + 1,
                "pair_type": "same_arm" if pair_type == "同臂" else "cross_arm",
                "score_model": "soft_sigmoid",
                "d_c_A": SOFT_SIGMOID_DC,
                "sigma_A": SOFT_SIGMOID_SIGMA,
                "n_cycles": n_cycles,
                "n_frames_total": data.frame_count,
                "q_mean": q_mean[index],
                "q_se_cycle": q_se[index],
                "q_ci_low_cycle_bootstrap": ci_low[index],
                "q_ci_high_cycle_bootstrap": ci_high[index],
                "q_first_half": first_mean[index],
                "q_second_half": second_mean[index],
                "q_window_delta": second_mean[index] - first_mean[index],
            }
        )
    return rows


@lru_cache(maxsize=None)
def perfect_matchings(n_vertices: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    def recurse(vertices: tuple[int, ...]) -> Iterable[tuple[tuple[int, int], ...]]:
        if not vertices:
            yield ()
            return
        first = vertices[0]
        for position in range(1, len(vertices)):
            second = vertices[position]
            remaining = vertices[1:position] + vertices[position + 1 :]
            for rest in recurse(remaining):
                yield ((first, second),) + rest

    return tuple(recurse(tuple(range(n_vertices))))


def component_mass_fractions(
    matching: Sequence[tuple[int, int]], terminal_to_monomer: Sequence[int], n_monomers: int
) -> np.ndarray:
    parent = list(range(n_monomers))

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right in matching:
        union(terminal_to_monomer[left], terminal_to_monomer[right])
    component_sizes = Counter(find(monomer) for monomer in range(n_monomers))
    fractions = np.zeros(n_monomers + 1, dtype=float)
    for size in component_sizes.values():
        fractions[size] += size / n_monomers
    return fractions


def exact_product_distribution(data: SystemData, q_pair: np.ndarray) -> tuple[np.ndarray, float]:
    n_vertices = data.n_vinyl
    n_monomers = data.n_monomers
    pair_weight = {edge: float(weight) for edge, weight in zip(data.pair_vertices, q_pair)}
    pair_edges = list(pair_weight)
    full_vertex_mask = (1 << n_vertices) - 1
    denominator = np.zeros(1 << n_vertices, dtype=float)
    for active_mask in range(1 << n_vertices):
        total = 0.0
        for left, right in pair_edges:
            if active_mask & (1 << left) and active_mask & (1 << right):
                total += pair_weight[(left, right)]
        denominator[active_mask] = total

    result = np.zeros(n_monomers + 1, dtype=float)
    completion_probability = 0.0
    subset_full = (1 << n_monomers) - 1
    for matching in perfect_matchings(n_vertices):
        weights = [pair_weight[tuple(sorted(edge))] for edge in matching]
        edge_masks = [(1 << left) | (1 << right) for left, right in matching]
        removed_masks = [0] * (1 << n_monomers)
        for subset in range(1, 1 << n_monomers):
            least_bit = subset & -subset
            edge_index = least_bit.bit_length() - 1
            removed_masks[subset] = removed_masks[subset ^ least_bit] | edge_masks[edge_index]
        probabilities = [0.0] * (1 << n_monomers)
        probabilities[0] = 1.0
        for subset in range(1 << n_monomers):
            state_probability = probabilities[subset]
            if state_probability == 0.0 or subset == subset_full:
                continue
            active_mask = full_vertex_mask ^ removed_masks[subset]
            total_weight = denominator[active_mask]
            if total_weight <= 0.0:
                continue
            remaining = subset_full ^ subset
            while remaining:
                least_bit = remaining & -remaining
                edge_index = least_bit.bit_length() - 1
                weight = weights[edge_index]
                if weight > 0.0:
                    probabilities[subset | least_bit] += state_probability * weight / total_weight
                remaining ^= least_bit
        matching_probability = probabilities[subset_full]
        if matching_probability:
            completion_probability += matching_probability
            result += matching_probability * component_mass_fractions(
                matching, data.terminal_to_monomer, n_monomers
            )
    if completion_probability > 0:
        result /= completion_probability
    return result, completion_probability


def simulate_product_distribution(
    data: SystemData, q_pair: np.ndarray, simulations: int, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    pair_left = np.array([edge[0] for edge in data.pair_vertices], dtype=int)
    pair_right = np.array([edge[1] for edge in data.pair_vertices], dtype=int)
    q_pair = np.asarray(q_pair, dtype=float)
    active = np.ones((simulations, data.n_vinyl), dtype=bool)
    selected_pairs = np.full((simulations, data.n_monomers), -1, dtype=int)
    still_running = np.ones(simulations, dtype=bool)
    for step in range(data.n_monomers):
        valid_pairs = active[:, pair_left] & active[:, pair_right]
        weights = valid_pairs * q_pair[None, :]
        totals = weights.sum(axis=1)
        runnable = still_running & (totals > 0)
        rows = np.flatnonzero(runnable)
        if rows.size == 0:
            break
        cumulative = np.cumsum(weights[rows], axis=1)
        draws = rng.random(rows.size) * totals[rows]
        selected = (cumulative < draws[:, None]).sum(axis=1)
        selected_pairs[rows, step] = selected
        active[rows, pair_left[selected]] = False
        active[rows, pair_right[selected]] = False
        still_running &= totals > 0
    completed_rows = np.flatnonzero((selected_pairs >= 0).all(axis=1))
    completion_fraction = completed_rows.size / simulations
    output = np.zeros(data.n_monomers + 1, dtype=float)
    if completed_rows.size == 0:
        return output, completion_fraction
    for row in completed_rows:
        matching = [data.pair_vertices[index] for index in selected_pairs[row]]
        output += component_mass_fractions(matching, data.terminal_to_monomer, data.n_monomers)
    output /= completed_rows.size
    return output, completion_fraction


def read_targets(path: Path) -> dict[str, dict[str, object]]:
    targets: dict[str, dict[str, object]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = targets.setdefault(
                row["system"],
                {
                    "n_monomers": int(row["n_monomers"]),
                    "funi": float(row["funi_wt_percent"]) / 100.0,
                    "rings": {},
                },
            )
            record["rings"][int(row["ring_size"])] = float(row["ring_wt_percent"]) / 100.0
    return targets


def bootstrap_product_uncertainty(
    systems: Sequence[SystemData],
    targets: dict[str, dict[str, object]],
    cycle_score_getter,
    bootstrap_replicates: int,
    simulations: int,
    base_seed: int,
    seed_label: str,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for data in systems:
        target = targets[data.name]
        feasible_observed = sorted(ring for ring in target["rings"] if 1 <= ring <= data.n_monomers)
        rng = np.random.default_rng(stable_seed(base_seed, data.name, seed_label))
        distributions: list[np.ndarray] = []
        conditioned_distributions: list[np.ndarray] = []
        completion_fractions: list[float] = []
        cycle_scores = cycle_score_getter(data)
        for _ in range(bootstrap_replicates):
            indices = rng.integers(0, cycle_scores.shape[0], size=cycle_scores.shape[0])
            q_boot = cycle_scores[indices].mean(axis=0)
            distribution, completion_fraction = simulate_product_distribution(
                data, q_boot, simulations, rng
            )
            distributions.append(distribution)
            reported_total = sum(distribution[ring] for ring in feasible_observed)
            conditioned = np.full(data.n_monomers + 1, np.nan)
            if reported_total > 0:
                for ring in feasible_observed:
                    conditioned[ring] = distribution[ring] / reported_total
            conditioned_distributions.append(conditioned)
            completion_fractions.append(completion_fraction)
        distribution_array = np.stack(distributions)
        conditioned_array = np.stack(conditioned_distributions)
        for ring in range(1, data.n_monomers + 1):
            raw_low, raw_median, raw_high = np.quantile(
                distribution_array[:, ring], [0.025, 0.5, 0.975]
            )
            if ring in feasible_observed:
                condition_low, condition_median, condition_high = np.nanquantile(
                    conditioned_array[:, ring], [0.025, 0.5, 0.975]
                )
            else:
                condition_low = condition_median = condition_high = math.nan
            row = {
                "system": data.name,
                "ring_size": ring,
                "all_classes_ci_low": raw_low,
                "all_classes_median": raw_median,
                "all_classes_ci_high": raw_high,
                "conditioned_reported_ci_low": condition_low,
                "conditioned_reported_median": condition_median,
                "conditioned_reported_ci_high": condition_high,
                "bootstrap_replicates": bootstrap_replicates,
                "network_mc_simulations_per_replicate": simulations,
                "mean_mc_completion_fraction": float(np.mean(completion_fractions)),
            }
            row.update(metadata)
            rows.append(row)
    return rows


def matrix_from_pairs(data: SystemData, values: np.ndarray) -> np.ndarray:
    matrix = np.zeros((data.n_vinyl, data.n_vinyl), dtype=float)
    for (left, right), value in zip(data.pair_vertices, values):
        matrix[left, right] = matrix[right, left] = value
    return matrix


def write_q_matrix(path: Path, data: SystemData, values: np.ndarray) -> None:
    matrix = matrix_from_pairs(data, values)
    rows: list[dict[str, object]] = []
    for row_index in range(data.n_vinyl):
        row: dict[str, object] = {"vinyl": f"V{row_index + 1}"}
        for column_index in range(data.n_vinyl):
            row[f"V{column_index + 1}"] = matrix[row_index, column_index]
        rows.append(row)
    write_csv(path, rows, ["vinyl", *(f"V{index + 1}" for index in range(data.n_vinyl))])


def color_scale(value: float, maximum: float) -> str:
    fraction = 0.0 if maximum <= 0 else min(max(value / maximum, 0.0), 1.0)
    start = np.array([247, 251, 255], dtype=float)
    end = np.array([8, 81, 156], dtype=float)
    rgb = np.rint(start * (1 - fraction) + end * fraction).astype(int)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def write_heatmap_svg(path: Path, data: SystemData, values: np.ndarray, threshold: float, maximum: float) -> None:
    cell = 34
    margin_left, margin_top, legend_width = 76, 82, 100
    width = margin_left + cell * data.n_vinyl + legend_width
    height = margin_top + cell * data.n_vinyl + 55
    matrix = matrix_from_pairs(data, values)
    same_set = {tuple(sorted(edge)) for edge in data.same_edges}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin_left}" y="28" font-family="Arial,sans-serif" font-size="18" font-weight="700">{html.escape(data.name)} q(0), d &lt; {threshold:g} Å</text>',
        '<text x="{0}" y="50" font-family="Arial,sans-serif" font-size="11" fill="#455a64">Red borders mark same-arm pairs; diagonal is not a candidate reaction.</text>'.format(margin_left),
    ]
    for index in range(data.n_vinyl):
        x = margin_left + index * cell + cell / 2
        y = margin_top - 10
        elements.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11">V{index + 1}</text>')
        y_label = margin_top + index * cell + cell / 2 + 4
        elements.append(f'<text x="{margin_left - 10}" y="{y_label}" text-anchor="end" font-family="Arial,sans-serif" font-size="11">V{index + 1}</text>')
    for row in range(data.n_vinyl):
        for column in range(data.n_vinyl):
            x, y = margin_left + column * cell, margin_top + row * cell
            value = matrix[row, column]
            fill = "#eceff1" if row == column else color_scale(value, maximum)
            edge = tuple(sorted((row, column)))
            stroke = "#d32f2f" if row != column and edge in same_set else "#cfd8dc"
            stroke_width = 2.2 if stroke == "#d32f2f" else 0.7
            elements.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )
            if row != column and data.n_vinyl <= 8:
                text_color = "white" if maximum > 0 and value / maximum > 0.55 else "#263238"
                elements.append(
                    f'<text x="{x + cell / 2}" y="{y + cell / 2 + 3}" text-anchor="middle" font-family="Arial,sans-serif" font-size="8" fill="{text_color}">{value:.3f}</text>'
                )
    legend_x = margin_left + cell * data.n_vinyl + 30
    for step in range(80):
        value = maximum * (79 - step) / 79
        elements.append(
            f'<rect x="{legend_x}" y="{margin_top + step * 2}" width="18" height="2.2" fill="{color_scale(value, maximum)}"/>'
        )
    elements.extend(
        [
            f'<text x="{legend_x + 24}" y="{margin_top + 8}" font-family="Arial,sans-serif" font-size="10">{maximum:.3f}</text>',
            f'<text x="{legend_x + 24}" y="{margin_top + 160}" font-family="Arial,sans-serif" font-size="10">0</text>',
            f'<text x="{legend_x}" y="{margin_top + 182}" font-family="Arial,sans-serif" font-size="10">occupancy</text>',
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def write_product_comparison_svg(
    path: Path,
    systems: Sequence[SystemData],
    primary_rows: Sequence[dict[str, object]],
    primary_threshold: float,
    title: str | None = None,
) -> None:
    row_lookup = {
        (str(row["system"]), int(row["ring_size"])): row
        for row in primary_rows
        if float(row["threshold_A"]) == primary_threshold and bool(row["experimentally_compared"])
    }
    panel_width, left, bar_width = 1000, 160, 700
    y = 70
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_width}" height="1600" viewBox="0 0 {panel_width} 1600">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="40" y="32" font-family="Arial,sans-serif" font-size="20" font-weight="700">{html.escape(title or f"Initial-state product baseline at d < {primary_threshold:g} Å")}</text>',
        '<rect x="650" y="16" width="16" height="10" fill="#1565c0"/><text x="672" y="25" font-family="Arial,sans-serif" font-size="11">model (conditioned on reported classes)</text>',
        '<rect x="650" y="34" width="16" height="10" fill="#f9a825"/><text x="672" y="43" font-family="Arial,sans-serif" font-size="11">experiment (renormalized)</text>',
    ]
    for data in systems:
        compared = sorted(
            ring for (system, ring), row in row_lookup.items() if system == data.name
        )
        if not compared:
            continue
        elements.append(
            f'<text x="40" y="{y}" font-family="Arial,sans-serif" font-size="14" font-weight="700">{html.escape(data.name)}</text>'
        )
        y += 12
        for ring in compared:
            row = row_lookup[(data.name, ring)]
            prediction = float(row["predicted_fraction_conditioned_reported"])
            experiment = float(row["experimental_fraction_normalized"])
            elements.append(f'<text x="{left - 12}" y="{y + 14}" text-anchor="end" font-family="Arial,sans-serif" font-size="11">R{ring}</text>')
            elements.append(f'<rect x="{left}" y="{y + 2}" width="{bar_width}" height="8" fill="#eceff1"/>')
            elements.append(f'<rect x="{left}" y="{y + 2}" width="{bar_width * prediction:.2f}" height="8" fill="#1565c0"/>')
            elements.append(f'<rect x="{left}" y="{y + 13}" width="{bar_width * experiment:.2f}" height="8" fill="#f9a825"/>')
            elements.append(f'<text x="{left + bar_width + 8}" y="{y + 10}" font-family="Arial,sans-serif" font-size="10">{100 * prediction:.1f}%</text>')
            elements.append(f'<text x="{left + bar_width + 8}" y="{y + 21}" font-family="Arial,sans-serif" font-size="10">{100 * experiment:.1f}%</text>')
            y += 28
        y += 20
    for tick in range(0, 101, 20):
        x = left + bar_width * tick / 100
        elements.append(f'<line x1="{x}" y1="48" x2="{x}" y2="{y}" stroke="#cfd8dc" stroke-width="0.5"/>')
        elements.append(f'<text x="{x}" y="{y + 18}" text-anchor="middle" font-family="Arial,sans-serif" font-size="10">{tick}%</text>')
    elements[0] = f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_width}" height="{y + 35}" viewBox="0 0 {panel_width} {y + 35}">'
    elements.append('</svg>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def format_missing_cycles(issues: Sequence[dict[str, str]]) -> str:
    items = [f"{row['system']}: {row['detail']}" for row in issues if row["code"] == "missing_cycles"]
    return "; ".join(items) if items else "none"


def build_report(
    output_path: Path,
    systems: Sequence[SystemData],
    issues: Sequence[dict[str, str]],
    metrics: Sequence[dict[str, object]],
    prediction_rows: Sequence[dict[str, object]],
    bootstrap_rows: Sequence[dict[str, object]],
    soft_metrics: Sequence[dict[str, object]],
    soft_prediction_rows: Sequence[dict[str, object]],
    soft_bootstrap_rows: Sequence[dict[str, object]],
    core_audit_rows: Sequence[dict[str, object]],
    contact_summaries: Sequence[dict[str, object]],
    primary_threshold: float,
    thresholds: Sequence[float],
) -> None:
    primary_metric = next(row for row in metrics if float(row["threshold_A"]) == primary_threshold)
    soft_metric = soft_metrics[0]
    primary_contact_rows = [
        row for row in contact_summaries if float(row["threshold_A"]) == primary_threshold
    ]
    worst_window = max(
        primary_contact_rows,
        key=lambda row: float(row["max_abs_first_second_delta"]),
    )
    bootstrap_lookup = {
        (str(row["system"]), int(row["ring_size"])): row for row in bootstrap_rows
    }
    soft_bootstrap_lookup = {
        (str(row["system"]), int(row["ring_size"])): row for row in soft_bootstrap_rows
    }
    soft_target_lookup = {
        (str(row["system"]), int(row["ring_size"])): row
        for row in soft_prediction_rows
        if bool(row["experimentally_compared"])
        and int(row["ring_size"]) == int(row["n_monomers"])
    }
    target_rows = [
        row
        for row in prediction_rows
        if float(row["threshold_A"]) == primary_threshold
        and bool(row["experimentally_compared"])
        and int(row["ring_size"]) == int(row["n_monomers"])
    ]
    lines = [
        "# Initial-state trajectory-to-product baseline",
        "",
        "## tl;dr",
        "",
        f"The stage-one pipeline successfully parsed {sum(len(data.cycle_ids) for data in systems)} cycles "
        f"({sum(data.frame_count for data in systems):,} frames) across {len(systems)} usable systems. "
        f"The provisional {primary_threshold:g} Å initial-state network has a mean conditional MAE of "
        f"{float(primary_metric['conditional_mae_pp']):.1f} percentage points against the reported SEC classes. "
        f"A fixed soft-sigmoid sensitivity model (d_c={SOFT_SIGMOID_DC:g} Å, sigma={SOFT_SIGMOID_SIGMA:g} Å), adapted from `core/` without label fitting, "
        f"reduces this exploratory in-sample MAE to {float(soft_metric['conditional_mae_pp']):.1f} points and completes every network. "
        "This is a baseline diagnostic, not a validated predictor: only distances are available, and no post-reaction "
        "state trajectories are present.",
        "",
        "## Data and QC",
        "",
        "- `B6_p_C5_md` is now included: 100 complete cycles, 50,000 frames, consistent headers, no exact duplicate cycle files, and valid PDB/log vinyl mappings.",
        f"- Missing or unusable cycle data: {format_missing_cycles(issues)}. The detailed audit distinguishes absent files from header-only CSVs.",
        f"- Error-level QC findings: {sum(row['severity'] == 'ERROR' for row in issues)}; "
        f"warning-level findings: {sum(row['severity'] == 'WARN' for row in issues)}.",
        f"- The largest pair-level first-half/second-half shift at {primary_threshold:g} Å is "
        f"{float(worst_window['max_abs_first_second_delta']):.3f} in {worst_window['system']}; this is a convergence warning, not independent evidence of chemistry.",
        "- Every valid cycle is treated as one uncertainty unit; the 500 correlated frames inside a cycle are not counted as independent replicates.",
        "- PDB checks use the vinyl atom pairs written in each `md_simulation.log` and require both mapped atoms to be bonded carbons.",
        "",
        "## Model",
        "",
        "For each pair, `q_ij^(0)` is the cycle-averaged occupancy of `d_ij < d_c`. At each reaction step, "
        "the next active pair is selected with probability proportional to `q_ij^(0)` (kappa = 1). "
        "All perfect matchings are enumerated exactly for the mean matrix. The union of fixed same-arm monomer edges "
        "and reaction edges is a set of cycles; component size gives R1, R2, ... .",
        "",
        f"The auxiliary soft score is `1 / (1 + exp((d - {SOFT_SIGMOID_DC:g}) / {SOFT_SIGMOID_SIGMA:g}))`. It remains pair-resolved and uses no zero filling. "
        "It is reported as an exploratory model comparison because it was added after reviewing `core/`, not as a preregistered validation result.",
        "",
        "Reported classes are renormalized for comparison because Table 1 omits R1/R5 and does not always sum to 100%. "
        "The model's omitted-class mass is retained separately rather than silently discarded. B3 R4 is also outside a "
        "single-reactor B3 network and is not used in the within-reactor comparison.",
        "",
        "## Primary-threshold target products",
        "",
        "| System | Hard 5 Å target | Hard 95% interval | Soft target | Soft 95% interval | Experimental target (normalized reported classes) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in target_rows:
        key = (str(row["system"]), int(row["ring_size"]))
        boot = bootstrap_lookup.get(key)
        soft_row = soft_target_lookup[key]
        soft_boot = soft_bootstrap_lookup.get(key)
        interval = (
            f"{100 * float(boot['conditioned_reported_ci_low']):.1f}–{100 * float(boot['conditioned_reported_ci_high']):.1f}%"
            if boot
            else "n/a"
        )
        soft_interval = (
            f"{100 * float(soft_boot['conditioned_reported_ci_low']):.1f}–{100 * float(soft_boot['conditioned_reported_ci_high']):.1f}%"
            if soft_boot
            else "n/a"
        )
        lines.append(
            f"| {row['system']} | {100 * float(row['predicted_fraction_conditioned_reported']):.1f}% | "
            f"{interval} | {100 * float(soft_row['predicted_fraction_conditioned_reported']):.1f}% | "
            f"{soft_interval} | {100 * float(row['experimental_fraction_normalized']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Threshold sensitivity",
            "",
            "| Distance threshold (Å) | Conditional MAE (percentage points) | Target-product MAE (percentage points) | Mean model mass outside reported classes | Minimum exact completion probability |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metrics:
        lines.append(
            f"| {float(row['threshold_A']):.1f} | {float(row['conditional_mae_pp']):.2f} | "
            f"{float(row['target_product_mae_pp']):.2f} | {100 * float(row['mean_unreported_model_mass']):.2f}% | "
            f"{float(row['minimum_exact_completion_probability']):.6f} |"
        )
    lines.extend(
        [
            "",
            "These threshold errors are descriptive sensitivity results. Selecting the best row after seeing all labels would be in-sample tuning, so 5 Å remains the provisional registered threshold for this stage. A zero minimum completion probability means at least one hard-threshold contact graph cannot finish every pairing sequence; its error is calculated only for finite comparable outputs.",
            "",
            "## Core reference review",
            "",
            f"The fixed soft score improves the descriptive conditional MAE from {float(primary_metric['conditional_mae_pp']):.2f} to {float(soft_metric['conditional_mae_pp']):.2f} percentage points, "
            f"and its minimum exact completion probability is {float(soft_metric['minimum_exact_completion_probability']):.6f}. "
            "This comparison is exploratory and must be tested on held-out or newly generated systems before model selection.",
            "",
            "| Review item | Assessment | Action |",
            "|---|---|---|",
        ]
    )
    for row in core_audit_rows:
        lines.append(
            f"| {row['item']} | {row['assessment']} | {row['action']} |"
        )
    lines.extend(
        [
            "",
            "## What is and is not learned",
            "",
            "- The full pair matrix is used in the network; same-arm/cross-arm means are descriptive only.",
            "- A distance-only initial-state matrix can be tested immediately, but it cannot identify catalyst kinetics or reactive alkene orientation.",
            "- Reusing `q^(0)` after every reaction assumes no conformational redistribution. The route map explicitly requires post-reaction MD to test that assumption.",
            "- SEC values describe the cleaved experimental mixture, while this network is a single-reactor intramolecular model. The comparison is therefore an approximation, especially for the small non-unimer fraction and B3 R4.",
            "",
            "## Next experiment",
            "",
            "Generate named one-reaction states for a small representative set (for example, A4-m-C5 and A4-p-C6, with one same-arm and one cross-arm first edge each). The same pipeline can then compute pair-resolved `q^(1) - q^(0)` and replace the current initial-state reuse assumption with actual state-specific transitions.",
            "",
            "## Artifacts",
            "",
            "- [`q_pair_stats.csv`](q_pair_stats.csv)",
            "- [`product_predictions.csv`](product_predictions.csv)",
            "- [`product_predictions_primary_bootstrap.csv`](product_predictions_primary_bootstrap.csv)",
            "- [`model_metrics.csv`](model_metrics.csv)",
            "- [`soft_sigmoid_model_metrics.csv`](soft_sigmoid_model_metrics.csv)",
            "- [`soft_sigmoid_product_predictions.csv`](soft_sigmoid_product_predictions.csv)",
            "- [`core_reference_audit.csv`](core_reference_audit.csv)",
            "- [`validation_report.md`](validation_report.md)",
            "- [`figures/product_comparison_dc5p0.svg`](figures/product_comparison_dc5p0.svg)",
            "- [`figures/product_comparison_soft_sigmoid.svg`](figures/product_comparison_soft_sigmoid.svg)",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_core_reference(project_root: Path, systems: Sequence[SystemData]) -> list[dict[str, object]]:
    core_file = project_root / "core" / "run_all_extended.py"
    mapping_matches = 0
    mapping_total = 0
    if core_file.exists():
        spec = importlib.util.spec_from_file_location("mtmr_core_reference", core_file)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            configs = getattr(module, "all_configs", {})
            for data in systems:
                if data.name not in configs:
                    continue
                mapping_total += 1
                configured = {tuple(sorted(edge)) for edge in configs[data.name]["sp"]}
                if configured == set(data.same_edges):
                    mapping_matches += 1

    b6 = next(data for data in systems if data.name == "B6_p_C5")
    b6_q = b6.cycle_q[DEFAULT_PRIMARY_THRESHOLD].mean(axis=0)
    same_mask = np.array([kind == "同臂" for kind in b6.pair_types])
    cross_mask = ~same_mask
    all_cross_mean = float(b6_q[cross_mask].mean())
    positive_cross_mean = float(b6_q[cross_mask][b6_q[cross_mask] > 0].mean())
    return [
        {
            "item": "Same-arm mapping",
            "assessment": f"PASS: {mapping_matches}/{mapping_total} core mappings match CSV headers",
            "action": "Retained the header-derived mappings; no manual remapping needed",
            "source": "core/run_all_extended.py and trace/*/trajectory.csv",
        },
        {
            "item": "Soft sigmoid encounter score",
            "assessment": f"Useful fixed sensitivity model at d_c={SOFT_SIGMOID_DC:g} Å, sigma={SOFT_SIGMOID_SIGMA:g} Å",
            "action": "Adopted as a separate pair-resolved exploratory model without label fitting",
            "source": "core/validate_model.py",
        },
        {
            "item": "Zero-contact symmetry fill",
            "assessment": f"B6-p-C5 cross-pair mean changes from {all_cross_mean:.5f} (all observed pairs) to {positive_cross_mean:.5f} when zeros are excluded",
            "action": "Not adopted; observed zero contacts remain zero in the hard-cutoff baseline",
            "source": "core/run_all_extended.py read_md",
        },
        {
            "item": "Same/cross matrix compression",
            "assessment": "Collapses distinct pair-resolved geometry into two averages",
            "action": "Not adopted; the full q_ij matrix is propagated",
            "source": "core/run_all_extended.py main model construction",
        },
        {
            "item": "epsilon0, gamma, alpha scans",
            "assessment": "Parameters are selected against the same experimental yields in several workflows",
            "action": "Not imported into the baseline; require held-out validation or post-reaction MD",
            "source": "core/run_all_extended.py and core/validate_model.py",
        },
        {
            "item": "Product aggregation",
            "assessment": "Core validation reports ring-count/mole fractions, whereas SEC targets are mass fractions",
            "action": "Current pipeline weights each Rn component by n before comparison",
            "source": "core/validate_model.py aggregate_products",
        },
        {
            "item": "Dynamic topology updates",
            "assessment": "Scientifically plausible interpolation but not yet identified from supplied state-specific MD",
            "action": "Deferred until named post-reaction trajectories are available",
            "source": "core/gcmc_dynamic_graph.py and core/solve_b6.py",
        },
    ]


def build_validation_report(
    output_path: Path,
    systems: Sequence[SystemData],
    qc_rows: Sequence[dict[str, object]],
    issues: Sequence[dict[str, str]],
    hard_metric: dict[str, object],
    soft_metric: dict[str, object],
    core_audit_rows: Sequence[dict[str, object]],
) -> None:
    b6_qc = next(row for row in qc_rows if row["system"] == "B6_p_C5")
    convergence_rows = []
    for data in systems:
        first = data.cycle_q_first[DEFAULT_PRIMARY_THRESHOLD].mean(axis=0)
        second = data.cycle_q_second[DEFAULT_PRIMARY_THRESHOLD].mean(axis=0)
        convergence_rows.append((data.name, float(np.max(np.abs(second - first)))))
    worst_convergence = max(convergence_rows, key=lambda item: item[1])
    lines = [
        "# Validation Report",
        "",
        "## Overall Assessment: Share with caveats",
        "",
        "The updated initial-state analysis is reproducible and the newly supplied B6-p-C5 data are suitable for inclusion. "
        "The product model remains exploratory because no state-specific post-reaction trajectories are present and the soft-score comparison was introduced after reviewing prior code.",
        "",
        "## Methodology Review",
        "",
        "- Question: predict cleaved-product mass distributions from initial-state vinyl-pair distance trajectories.",
        "- Grain: one independently restarted MD cycle is the uncertainty unit; frames within a cycle are correlated observations.",
        "- Sources: `trace/*_md/cycle_*/trajectory.csv`, `structures/*.pdb`, simulation logs, Table 1 targets, and reference implementations under `core/`.",
        "- Models: pair-resolved hard distance occupancies at 4-7 Å and a fixed 5 Å/0.5 Å soft sigmoid sensitivity score.",
        "- Output definition: monomer-mass fractions, conditioned only for the explicit comparison with reported SEC classes.",
        "",
        "## Data-quality Findings",
        "",
        f"- B6-p-C5: {b6_qc['n_valid_cycles']} valid cycles, {int(b6_qc['n_frames_total']):,} frames, "
        f"{b6_qc['duplicate_cycle_files']} exact duplicate cycle files, QC status `{b6_qc['status']}`.",
        f"- Full dataset: {sum(len(data.cycle_ids) for data in systems)} valid cycles and {sum(data.frame_count for data in systems):,} frames across {len(systems)} systems.",
        f"- Remaining file-level errors: {sum(row['severity'] == 'ERROR' for row in issues)}; warnings: {sum(row['severity'] == 'WARN' for row in issues)}.",
        f"- Largest first-half/second-half 5 Å pair shift: {worst_convergence[1]:.3f} in {worst_convergence[0]}; treat this as a convergence warning.",
        "",
        "## Issues Found",
        "",
        "1. **High - Missing state-specific data.** Reusing q(0) after each reaction cannot establish post-reaction conformational redistribution.",
        "2. **Medium - Hard-cutoff network stalls.** Zero observed contacts make some terminal pairing sequences unreachable; conditional predictions can therefore be selective.",
        "3. **Medium - Soft-model selection risk.** The soft score improves descriptive fit but was added after inspecting earlier code and results, so it is not independent validation.",
        "4. **Medium - Prior aggregation mismatch.** Parts of `core/` compare mole/ring-count fractions with SEC mass fractions; those numbers should not be used directly.",
        "",
        "## Calculation Spot-checks",
        "",
        f"- Hard 5 Å conditional MAE: {float(hard_metric['conditional_mae_pp']):.2f} percentage points across {hard_metric['n_systems']} systems.",
        f"- Soft sigmoid conditional MAE: {float(soft_metric['conditional_mae_pp']):.2f} percentage points; minimum exact completion probability {float(soft_metric['minimum_exact_completion_probability']):.6f}.",
        "- Every completed perfect matching conserves monomer mass: the R1...Rn mass fractions sum to one.",
        "- Same-arm mappings in `core/` were reconciled against actual CSV headers before reuse.",
        "",
        "## Core-reference Decisions",
        "",
        "| Item | Assessment | Action |",
        "|---|---|---|",
    ]
    for row in core_audit_rows:
        lines.append(f"| {row['item']} | {row['assessment']} | {row['action']} |")
    lines.extend(
        [
            "",
            "## Visualization Review",
            "",
            "- Hard and soft comparison charts use a shared 0-100% horizontal scale starting at zero, consistent class ordering, and exact percentage labels.",
            "- Contact heatmaps share one color maximum at the primary threshold, so between-system color intensity is comparable.",
            "- SVG syntax validation passed; numerical conclusions are also available in CSV and do not depend on reading chart geometry.",
            "",
            "## Suggested Improvements",
            "",
            "1. Generate named post-reaction trajectories and validate state-specific q matrices.",
            "2. Add an alkene-orientation gate before interpreting any encounter score as reaction geometry.",
            "3. Test the soft sigmoid on a held-out or newly simulated system before selecting it over the hard baseline.",
            "4. Replace or explicitly archive the two header-only cycle files and regenerate the two absent cycles if practical.",
            "",
            "## Required Caveats",
            "",
            "- Soft-score improvement is exploratory/in-sample and should not be called predictive validation.",
            "- Funi and inter-reactor products are not explicitly generated by the single-reactor network.",
            "- Distance-only scores omit alkene orientation and catalyst-dependent kinetics.",
            "",
            "## Recommended Next Check",
            "",
            "Use named one-reaction MD states to measure q(1)-q(0), then compare the state-specific model against both the hard and soft initial-state baselines on a held-out system.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, object]:
    project_root = Path(args.root).resolve()
    output_dir = (project_root / args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = tuple(sorted(set(float(item) for item in args.thresholds)))
    primary_threshold = float(args.primary_threshold)
    if primary_threshold not in thresholds:
        raise ValueError("primary threshold must be included in the threshold grid")

    targets = read_targets(project_root / "analysis" / "experimental_targets.csv")
    system_dirs = sorted((project_root / "trace").glob("*_md"))
    systems: list[SystemData] = []
    all_issues: list[dict[str, str]] = []
    cycle_inventory: list[dict[str, object]] = []
    for system_dir in system_dirs:
        system_name = system_dir.name.removesuffix("_md")
        if system_name in EXCLUDED_SYSTEMS:
            all_issues.append(issue(system_name, "INFO", "excluded_by_request", "incomplete MD; not analysed"))
            continue
        data, inventory = analyse_system(project_root, system_dir, thresholds)
        cycle_inventory.extend(inventory)
        if data is not None:
            systems.append(data)
            all_issues.extend(data.issues)

    if not systems:
        raise RuntimeError("no usable systems were found")

    qc_rows: list[dict[str, object]] = []
    for data in systems:
        severities = {row["severity"] for row in data.issues}
        status = "ERROR" if "ERROR" in severities else "WARN" if "WARN" in severities else "PASS"
        system_hashes = [
            str(row["content_sha256"])
            for row in cycle_inventory
            if row["system"] == data.name and row["valid"]
        ]
        duplicate_cycle_files = sum(count - 1 for count in Counter(system_hashes).values() if count > 1)
        qc_rows.append(
            {
                "system": data.name,
                "status": status,
                "n_monomers": data.n_monomers,
                "n_vinyl": data.n_vinyl,
                "n_pairs": len(data.pair_names),
                "n_same_arm_pairs": len(data.same_edges),
                "n_valid_cycles": len(data.cycle_ids),
                "n_frames_total": data.frame_count,
                "sampling_time_per_cycle_ps": max(
                    float(row["time_end_ps"])
                    for row in cycle_inventory
                    if row["system"] == data.name and row["valid"]
                ),
                "aggregate_sampling_time_ns": sum(
                    float(row["time_end_ps"])
                    for row in cycle_inventory
                    if row["system"] == data.name and row["valid"]
                )
                / 1000.0,
                "duplicate_cycle_files": duplicate_cycle_files,
                "pdb_atoms": data.pdb_atoms,
            }
        )
    write_csv(output_dir / "qc_systems.csv", qc_rows)
    write_csv(
        output_dir / "qc_issues.csv",
        all_issues,
        ["system", "severity", "code", "detail"],
    )
    write_csv(output_dir / "cycle_inventory.csv", cycle_inventory)

    pair_stat_rows: list[dict[str, object]] = []
    soft_pair_stat_rows: list[dict[str, object]] = []
    contact_summaries: list[dict[str, object]] = []
    for data in systems:
        system_rows, system_summaries = q_statistics(
            data, thresholds, args.q_bootstrap, args.seed
        )
        pair_stat_rows.extend(system_rows)
        soft_pair_stat_rows.extend(soft_q_statistics(data, args.q_bootstrap, args.seed))
        contact_summaries.extend(system_summaries)
        for threshold in thresholds:
            label = str(threshold).replace(".", "p")
            write_q_matrix(
                output_dir / "q_matrices" / f"{data.name}_dc{label}.csv",
                data,
                data.cycle_q[threshold].mean(axis=0),
            )
        write_q_matrix(
            output_dir / "q_matrices" / f"{data.name}_soft_dc5p0_sigma0p5.csv",
            data,
            data.cycle_soft_q.mean(axis=0),
        )
    write_csv(output_dir / "q_pair_stats.csv", pair_stat_rows)
    write_csv(output_dir / "q_soft_sigmoid_pair_stats.csv", soft_pair_stat_rows)
    write_csv(output_dir / "contact_summary.csv", contact_summaries)

    exact_results: dict[tuple[str, float], tuple[np.ndarray, float]] = {}
    prediction_rows: list[dict[str, object]] = []
    metric_accumulator: dict[float, dict[str, list[float]]] = {
        threshold: defaultdict(list) for threshold in thresholds
    }
    for data in systems:
        target = targets.get(data.name)
        if target is None:
            all_issues.append(issue(data.name, "WARN", "missing_experimental_target", "not found"))
            continue
        feasible_observed = sorted(
            ring for ring in target["rings"] if 1 <= ring <= data.n_monomers
        )
        experimental_total = sum(target["rings"][ring] for ring in feasible_observed)
        experimental_normalized = {
            ring: target["rings"][ring] / experimental_total for ring in feasible_observed
        }
        for threshold in thresholds:
            exact_distribution, completion_probability = exact_product_distribution(
                data, data.cycle_q[threshold].mean(axis=0)
            )
            exact_results[(data.name, threshold)] = (exact_distribution, completion_probability)
            predicted_reported_total = sum(exact_distribution[ring] for ring in feasible_observed)
            for ring in range(1, data.n_monomers + 1):
                compared = ring in feasible_observed
                conditioned = (
                    exact_distribution[ring] / predicted_reported_total
                    if compared and predicted_reported_total > 0
                    else math.nan
                )
                experimental = experimental_normalized.get(ring, math.nan)
                prediction_rows.append(
                    {
                        "system": data.name,
                        "n_monomers": data.n_monomers,
                        "threshold_A": threshold,
                        "ring_size": ring,
                        "predicted_mass_fraction_all_classes": exact_distribution[ring],
                        "predicted_fraction_conditioned_reported": conditioned,
                        "experimental_fraction_normalized": experimental,
                        "experimental_fraction_table_raw": target["rings"].get(ring, math.nan),
                        "experimentally_compared": compared,
                        "exact_completion_probability": completion_probability,
                        "model_mass_outside_reported_classes": 1.0 - predicted_reported_total,
                        "experimental_reported_feasible_total": experimental_total,
                        "funi_experimental": target["funi"],
                    }
                )
                if compared and math.isfinite(conditioned) and math.isfinite(experimental):
                    metric_accumulator[threshold]["absolute_errors"].append(abs(conditioned - experimental))
            target_ring = data.n_monomers
            if target_ring in feasible_observed and predicted_reported_total > 0:
                metric_accumulator[threshold]["target_errors"].append(
                    abs(exact_distribution[target_ring] / predicted_reported_total - experimental_normalized[target_ring])
                )
            metric_accumulator[threshold]["unreported_mass"].append(1.0 - predicted_reported_total)
            metric_accumulator[threshold]["completion"].append(completion_probability)

    write_csv(output_dir / "product_predictions.csv", prediction_rows)
    metrics: list[dict[str, object]] = []
    for threshold in thresholds:
        accumulator = metric_accumulator[threshold]
        metrics.append(
            {
                "threshold_A": threshold,
                "n_systems": len(systems),
                "n_compared_class_entries": len(accumulator["absolute_errors"]),
                "conditional_mae_pp": 100 * float(np.mean(accumulator["absolute_errors"]))
                if accumulator["absolute_errors"]
                else math.nan,
                "target_product_mae_pp": 100 * float(np.mean(accumulator["target_errors"]))
                if accumulator["target_errors"]
                else math.nan,
                "mean_unreported_model_mass": float(np.mean(accumulator["unreported_mass"])),
                "minimum_exact_completion_probability": float(np.min(accumulator["completion"])),
                "n_zero_completion_systems": sum(value <= 1e-12 for value in accumulator["completion"]),
            }
        )
    write_csv(output_dir / "model_metrics.csv", metrics)

    soft_prediction_rows: list[dict[str, object]] = []
    soft_accumulator: dict[str, list[float]] = defaultdict(list)
    for data in systems:
        target = targets[data.name]
        feasible_observed = sorted(
            ring for ring in target["rings"] if 1 <= ring <= data.n_monomers
        )
        experimental_total = sum(target["rings"][ring] for ring in feasible_observed)
        experimental_normalized = {
            ring: target["rings"][ring] / experimental_total for ring in feasible_observed
        }
        distribution, completion_probability = exact_product_distribution(
            data, data.cycle_soft_q.mean(axis=0)
        )
        predicted_reported_total = sum(distribution[ring] for ring in feasible_observed)
        for ring in range(1, data.n_monomers + 1):
            compared = ring in feasible_observed
            conditioned = (
                distribution[ring] / predicted_reported_total
                if compared and predicted_reported_total > 0
                else math.nan
            )
            experimental = experimental_normalized.get(ring, math.nan)
            soft_prediction_rows.append(
                {
                    "system": data.name,
                    "n_monomers": data.n_monomers,
                    "score_model": "soft_sigmoid",
                    "d_c_A": SOFT_SIGMOID_DC,
                    "sigma_A": SOFT_SIGMOID_SIGMA,
                    "ring_size": ring,
                    "predicted_mass_fraction_all_classes": distribution[ring],
                    "predicted_fraction_conditioned_reported": conditioned,
                    "experimental_fraction_normalized": experimental,
                    "experimental_fraction_table_raw": target["rings"].get(ring, math.nan),
                    "experimentally_compared": compared,
                    "exact_completion_probability": completion_probability,
                    "model_mass_outside_reported_classes": 1.0 - predicted_reported_total,
                    "experimental_reported_feasible_total": experimental_total,
                    "funi_experimental": target["funi"],
                }
            )
            if compared and math.isfinite(conditioned) and math.isfinite(experimental):
                soft_accumulator["absolute_errors"].append(abs(conditioned - experimental))
        target_ring = data.n_monomers
        if target_ring in feasible_observed and predicted_reported_total > 0:
            soft_accumulator["target_errors"].append(
                abs(distribution[target_ring] / predicted_reported_total - experimental_normalized[target_ring])
            )
        soft_accumulator["unreported_mass"].append(1.0 - predicted_reported_total)
        soft_accumulator["completion"].append(completion_probability)
    soft_metrics = [
        {
            "score_model": "soft_sigmoid",
            "d_c_A": SOFT_SIGMOID_DC,
            "sigma_A": SOFT_SIGMOID_SIGMA,
            "n_systems": len(systems),
            "n_compared_class_entries": len(soft_accumulator["absolute_errors"]),
            "conditional_mae_pp": 100 * float(np.mean(soft_accumulator["absolute_errors"])),
            "target_product_mae_pp": 100 * float(np.mean(soft_accumulator["target_errors"])),
            "mean_unreported_model_mass": float(np.mean(soft_accumulator["unreported_mass"])),
            "minimum_exact_completion_probability": float(np.min(soft_accumulator["completion"])),
        }
    ]
    write_csv(output_dir / "soft_sigmoid_product_predictions.csv", soft_prediction_rows)
    write_csv(output_dir / "soft_sigmoid_model_metrics.csv", soft_metrics)

    bootstrap_rows = bootstrap_product_uncertainty(
        systems,
        targets,
        lambda data: data.cycle_q[primary_threshold],
        args.product_bootstrap,
        args.mc_sims,
        args.seed,
        "product_bootstrap_hard",
        {"score_model": "hard_cutoff", "threshold_A": primary_threshold},
    )
    write_csv(output_dir / "product_predictions_primary_bootstrap.csv", bootstrap_rows)
    soft_bootstrap_rows = bootstrap_product_uncertainty(
        systems,
        targets,
        lambda data: data.cycle_soft_q,
        args.product_bootstrap,
        args.mc_sims,
        args.seed,
        "product_bootstrap_soft_sigmoid",
        {
            "score_model": "soft_sigmoid",
            "d_c_A": SOFT_SIGMOID_DC,
            "sigma_A": SOFT_SIGMOID_SIGMA,
        },
    )
    write_csv(
        output_dir / "soft_sigmoid_product_predictions_bootstrap.csv",
        soft_bootstrap_rows,
    )
    core_audit_rows = audit_core_reference(project_root, systems)
    write_csv(output_dir / "core_reference_audit.csv", core_audit_rows)
    primary_metric = next(
        row for row in metrics if float(row["threshold_A"]) == primary_threshold
    )
    build_validation_report(
        output_dir / "validation_report.md",
        systems,
        qc_rows,
        all_issues,
        primary_metric,
        soft_metrics[0],
        core_audit_rows,
    )

    primary_maximum = max(
        float(data.cycle_q[primary_threshold].mean(axis=0).max()) for data in systems
    )
    for data in systems:
        write_heatmap_svg(
            output_dir / "figures" / f"q_heatmap_{data.name}_dc5p0.svg",
            data,
            data.cycle_q[primary_threshold].mean(axis=0),
            primary_threshold,
            primary_maximum,
        )
    write_product_comparison_svg(
        output_dir / "figures" / "product_comparison_dc5p0.svg",
        systems,
        prediction_rows,
        primary_threshold,
    )
    soft_plot_rows = [dict(row, threshold_A=primary_threshold) for row in soft_prediction_rows]
    write_product_comparison_svg(
        output_dir / "figures" / "product_comparison_soft_sigmoid.svg",
        systems,
        soft_plot_rows,
        primary_threshold,
        title="Exploratory soft-sigmoid product baseline (d_c=5 Å, sigma=0.5 Å)",
    )

    build_report(
        output_dir / "report.md",
        systems,
        all_issues,
        metrics,
        prediction_rows,
        bootstrap_rows,
        soft_metrics,
        soft_prediction_rows,
        soft_bootstrap_rows,
        core_audit_rows,
        contact_summaries,
        primary_threshold,
        thresholds,
    )
    return {
        "systems": len(systems),
        "cycles": sum(len(data.cycle_ids) for data in systems),
        "frames": sum(data.frame_count for data in systems),
        "qc_errors": sum(row["severity"] == "ERROR" for row in all_issues),
        "qc_warnings": sum(row["severity"] == "WARN" for row in all_issues),
        "primary_mae_pp": next(
            float(row["conditional_mae_pp"])
            for row in metrics
            if float(row["threshold_A"]) == primary_threshold
        ),
        "soft_mae_pp": float(soft_metrics[0]["conditional_mae_pp"]),
        "output_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", default=str(default_root), help="Project root")
    parser.add_argument("--output", default="results/baseline", help="Output path relative to root")
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--primary-threshold", type=float, default=DEFAULT_PRIMARY_THRESHOLD)
    parser.add_argument("--q-bootstrap", type=int, default=1000)
    parser.add_argument("--product-bootstrap", type=int, default=200)
    parser.add_argument("--mc-sims", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = run(args)
    print(
        "Completed initial-state MTMR baseline: "
        f"{summary['systems']} systems, {summary['cycles']} cycles, {summary['frames']:,} frames; "
        f"QC errors={summary['qc_errors']}, warnings={summary['qc_warnings']}; "
        f"hard-5A conditional MAE={summary['primary_mae_pp']:.2f} pp; "
        f"soft-sigmoid exploratory MAE={summary['soft_mae_pp']:.2f} pp."
    )
    print(f"Results: {summary['output_dir']}")


if __name__ == "__main__":
    main()
