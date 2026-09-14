#!/usr/bin/env python3
"""Audit and summarize replicated explicit-DCM vinyl-distance trajectories.

The replica, rather than an individual trajectory frame, is the uncertainty
unit.  Frame-wise autocorrelation is retained through an integrated
autocorrelation-time/effective-sample-size calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DEFAULT_STATES = ("A4_m_C5_q0", "A4_m_C6_q0", "A4_m_C7_q0", "A4_m_C8_q0")
HARD_THRESHOLDS_A = (4.0, 5.0, 6.0, 7.0)
T_CRITICAL_95 = {
    2: 12.706205,
    3: 4.302653,
    4: 3.182446,
    5: 2.776445,
    6: 2.570582,
    7: 2.446912,
    8: 2.364624,
    9: 2.306004,
    10: 2.262157,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def soft_contact(distance_A: np.ndarray, center_A: float = 5.0, width_A: float = 0.5) -> np.ndarray:
    """Fixed sigmoid encounter score used by the earlier trajectory model."""
    z = np.clip((np.asarray(distance_A, dtype=float) - center_A) / width_A, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(z))


def statistical_inefficiency(values: Sequence[float]) -> float:
    """Geyer initial-positive-sequence statistical inefficiency (>= 1)."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 4:
        return 1.0
    x = x - float(np.mean(x))
    variance = float(np.dot(x, x) / n)
    if not math.isfinite(variance) or variance <= np.finfo(float).eps:
        return 1.0
    n_fft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=n_fft)
    autocovariance = np.fft.irfft(spectrum * np.conjugate(spectrum), n=n_fft)[:n]
    autocovariance /= np.arange(n, 0, -1, dtype=float)
    rho = autocovariance / autocovariance[0]
    positive_sum = 0.0
    # Paired sums rho_1+rho_2, rho_3+rho_4, ... are truncated at the
    # first non-positive pair (Geyer's initial positive sequence).
    for lag in range(1, n - 1, 2):
        pair_sum = float(rho[lag] + rho[lag + 1])
        if not math.isfinite(pair_sum) or pair_sum <= 0.0:
            break
        positive_sum += pair_sum
    return float(max(1.0, 1.0 + 2.0 * positive_sum))


def split_rhat(chains: Sequence[Sequence[float]]) -> float:
    """Classical split-Rhat across equal-length chain halves."""
    arrays = [np.asarray(chain, dtype=float) for chain in chains]
    if len(arrays) < 2 or min((array.size for array in arrays), default=0) < 4:
        return math.nan
    half = min(array.size // 2 for array in arrays)
    split = np.asarray(
        [piece for array in arrays for piece in (array[:half], array[-half:])],
        dtype=float,
    )
    n = split.shape[1]
    chain_means = np.mean(split, axis=1)
    within = float(np.mean(np.var(split, axis=1, ddof=1)))
    between = float(n * np.var(chain_means, ddof=1))
    if within <= np.finfo(float).eps:
        return 1.0 if between <= np.finfo(float).eps else math.inf
    variance_hat = ((n - 1.0) / n) * within + between / n
    return float(max(1.0, math.sqrt(max(0.0, variance_hat / within))))


def contact_events(mask: Sequence[bool]) -> int:
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return 0
    return int(values[0]) + int(np.sum(values[1:] & ~values[:-1]))


def mean_ci95_replica(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    if array.size < 2:
        return mean, math.nan, math.nan, math.nan, math.nan
    sd = float(np.std(array, ddof=1))
    se = sd / math.sqrt(array.size)
    critical = T_CRITICAL_95.get(int(array.size), 1.96)
    return mean, sd, se, mean - critical * se, mean + critical * se


def load_distance_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] != len(header):
        raise ValueError(f"column mismatch in {path}: {data.shape[1]} != {len(header)}")
    return header, data


def finite_or_blank(value: float) -> object:
    return "" if math.isnan(value) else value


def convergence_class(rhat: float, combined_ess: float) -> str:
    if math.isfinite(rhat) and rhat <= 1.05 and combined_ess >= 400.0:
        return "pass"
    if math.isfinite(rhat) and rhat <= 1.10 and combined_ess >= 200.0:
        return "caution"
    return "extend"


def extension_duration_ps(
    root: Path,
    output_dir: str,
    replicas: int,
) -> float | None:
    """Read a uniform completed extension target from per-replica markers."""
    durations: list[float | None] = []
    for replica in range(1, replicas + 1):
        marker = root / output_dir / f"replica_{replica:02d}" / "extension_metadata.json"
        if not marker.is_file():
            durations.append(None)
            continue
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            durations.append(None)
            continue
        durations.append(float(metadata["target_total_ns"]) * 1000.0)
    present = [value for value in durations if value is not None]
    if not present:
        return None
    if len(present) != replicas:
        raise ValueError(
            f"extension markers are incomplete in {output_dir}: "
            f"{len(present)}/{replicas} replicas complete"
        )
    if max(present) - min(present) > 1e-6:
        raise ValueError(f"extension targets disagree in {output_dir}: {present}")
    return present[0]


def parse_state_durations(values: Sequence[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected STATE=PS, got {value!r}")
        state, duration = value.split("=", 1)
        duration_ps = float(duration)
        if not state or duration_ps <= 0.0:
            raise ValueError(f"invalid state duration: {value!r}")
        parsed[state] = duration_ps
    return parsed


def analyze(
    root: Path,
    output_dir: Path,
    states: Sequence[str] = DEFAULT_STATES,
    replicas: int = 4,
    center_A: float = 5.0,
    width_A: float = 0.5,
    expected_frames: int = 4001,
    expected_dt_ps: float = 0.5,
    expected_duration_ps: float = 2000.0,
    state_duration_ps: dict[str, float] | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {row["state"]: row for row in read_csv(root / "gromacs_dcm" / "state_manifest.csv")}
    unknown = set(states) - set(manifest)
    if unknown:
        raise ValueError(f"states absent from manifest: {sorted(unknown)}")
    duration_overrides = state_duration_ps or {}
    unknown_duration_states = set(duration_overrides) - set(states)
    if unknown_duration_states:
        raise ValueError(
            f"duration overrides do not belong to selected states: {sorted(unknown_duration_states)}"
        )
    expected_by_state: dict[str, tuple[int, float]] = {}
    for state in states:
        row = manifest[state]
        marker_duration = extension_duration_ps(root, row["output_dir"], replicas)
        duration = duration_overrides.get(state, marker_duration or expected_duration_ps)
        frames = (
            expected_frames
            if marker_duration is None and state not in duration_overrides
            else int(round(duration / expected_dt_ps)) + 1
        )
        if not math.isclose((frames - 1) * expected_dt_ps, duration, abs_tol=1e-6):
            raise ValueError(
                f"duration {duration:g} ps is incompatible with {expected_dt_ps:g} ps output grid"
            )
        expected_by_state[state] = (frames, duration)

    quality_rows: list[dict[str, object]] = []
    replica_rows: list[dict[str, object]] = []
    score_series: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    replica_topology_scores: dict[tuple[str, int, str], list[np.ndarray]] = defaultdict(list)
    pair_metadata: dict[tuple[str, str], dict[str, object]] = {}
    total_frames = 0
    fatal_issues: list[str] = []

    for state in states:
        row = manifest[state]
        state_expected_frames, state_expected_duration_ps = expected_by_state[state]
        pair_schema = list(json.loads(row["pair_schema_json"]))
        expected_columns = [str(pair["column"]) for pair in pair_schema]
        for replica in range(1, replicas + 1):
            path = root / row["output_dir"] / f"replica_{replica:02d}" / "vinyl_distances.csv"
            base_quality: dict[str, object] = {
                "state": state,
                "system": row["system"],
                "replica": replica,
                "expected_frames": state_expected_frames,
                "expected_duration_ps": state_expected_duration_ps,
                "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
            }
            if not path.is_file():
                issue = f"missing distance file: {path}"
                fatal_issues.append(issue)
                quality_rows.append({**base_quality, "status": "fail", "issue": issue})
                continue
            try:
                header, data = load_distance_csv(path)
                if header[:3] != ["frame", "time_ps", "absolute_time_ps"]:
                    raise ValueError(f"unexpected leading columns: {header[:3]}")
                if header[3:] != expected_columns:
                    raise ValueError("distance columns do not exactly match metadata pair_schema")
            except Exception as error:
                issue = f"{type(error).__name__}: {error}"
                fatal_issues.append(f"{path}: {issue}")
                quality_rows.append({**base_quality, "status": "fail", "issue": issue})
                continue

            frames = data[:, 0]
            time_ps = data[:, 1]
            absolute_time_ps = data[:, 2]
            distances = data[:, 3:]
            n_frames = int(data.shape[0])
            total_frames += n_frames
            time_steps = np.diff(time_ps)
            finite_distances = np.isfinite(distances)
            nonfinite = int(distances.size - np.count_nonzero(finite_distances))
            nonpositive = int(np.count_nonzero(distances <= 0.0))
            duplicate_frames = int(n_frames - np.unique(frames).size)
            duplicate_times = int(n_frames - np.unique(time_ps).size)
            median_dt = float(np.median(time_steps)) if time_steps.size else math.nan
            max_dt_error = (
                float(np.max(np.abs(time_steps - expected_dt_ps))) if time_steps.size else math.nan
            )
            duration = float(time_ps[-1] - time_ps[0]) if n_frames else math.nan
            frame_sequence_ok = bool(np.array_equal(frames, np.arange(n_frames, dtype=float)))
            checks = {
                "frame_count_ok": n_frames == state_expected_frames,
                "frame_sequence_ok": frame_sequence_ok,
                "duration_ok": math.isclose(duration, state_expected_duration_ps, abs_tol=1e-6),
                "dt_ok": math.isfinite(max_dt_error) and max_dt_error <= 1e-6,
                "finite_ok": nonfinite == 0,
                "positive_ok": nonpositive == 0,
                "unique_ok": duplicate_frames == 0 and duplicate_times == 0,
            }
            status = "pass" if all(checks.values()) else "fail"
            failed = [name for name, passed in checks.items() if not passed]
            if failed:
                fatal_issues.append(f"{path}: failed {', '.join(failed)}")
            quality_rows.append(
                {
                    **base_quality,
                    "frames": n_frames,
                    "pair_columns": distances.shape[1],
                    "time_start_ps": float(time_ps[0]),
                    "time_end_ps": float(time_ps[-1]),
                    "absolute_time_start_ps": float(absolute_time_ps[0]),
                    "absolute_time_end_ps": float(absolute_time_ps[-1]),
                    "duration_ps": duration,
                    "median_dt_ps": median_dt,
                    "max_dt_error_ps": max_dt_error,
                    "duplicate_frames": duplicate_frames,
                    "duplicate_times": duplicate_times,
                    "nonfinite_distances": nonfinite,
                    "nonpositive_distances": nonpositive,
                    **checks,
                    "status": status,
                    "issue": ";".join(failed),
                }
            )
            if status != "pass":
                continue

            for pair_index, pair in enumerate(pair_schema):
                pair_name = str(pair["column"])
                topology = str(pair["topology"])
                pair_metadata[(state, pair_name)] = pair
                distance = distances[:, pair_index]
                q = soft_contact(distance, center_A, width_A)
                inefficiency = statistical_inefficiency(q)
                ess = n_frames / inefficiency
                dt = median_dt if math.isfinite(median_dt) else expected_dt_ps
                score_series[(state, pair_name)].append(q)
                replica_topology_scores[(state, replica, topology)].append(q)
                hard = {threshold: distance < threshold for threshold in HARD_THRESHOLDS_A}
                replica_rows.append(
                    {
                        "state": state,
                        "system": row["system"],
                        "replica": replica,
                        "pair": pair_name,
                        "left": pair["left"],
                        "right": pair["right"],
                        "topology": topology,
                        "frames": n_frames,
                        "distance_mean_A": float(np.mean(distance)),
                        "distance_sd_A": float(np.std(distance, ddof=1)),
                        "distance_min_A": float(np.min(distance)),
                        "distance_q05_A": float(np.quantile(distance, 0.05)),
                        "distance_median_A": float(np.median(distance)),
                        "distance_q95_A": float(np.quantile(distance, 0.95)),
                        "soft_q_mean": float(np.mean(q)),
                        "soft_q_sd": float(np.std(q, ddof=1)),
                        "statistical_inefficiency": inefficiency,
                        "tau_integrated_ps": 0.5 * inefficiency * dt,
                        "effective_sample_size": ess,
                        "first_half_soft_q": float(np.mean(q[: n_frames // 2])),
                        "second_half_soft_q": float(np.mean(q[-(n_frames // 2) :])),
                        **{
                            f"hard_{threshold:g}A_fraction": float(np.mean(hard[threshold]))
                            for threshold in HARD_THRESHOLDS_A
                        },
                        **{
                            f"hard_{threshold:g}A_events": contact_events(hard[threshold])
                            for threshold in HARD_THRESHOLDS_A
                        },
                    }
                )

    quality_fields = [
        "state", "system", "replica", "expected_frames", "expected_duration_ps", "path",
        "frames", "pair_columns", "time_start_ps",
        "time_end_ps", "absolute_time_start_ps", "absolute_time_end_ps", "duration_ps",
        "median_dt_ps", "max_dt_error_ps", "duplicate_frames", "duplicate_times",
        "nonfinite_distances", "nonpositive_distances", "frame_count_ok", "frame_sequence_ok",
        "duration_ok", "dt_ok", "finite_ok", "positive_ok", "unique_ok", "status", "issue",
    ]
    write_csv(output_dir / "trajectory_quality.csv", quality_rows, quality_fields)

    replica_fields = [
        "state", "system", "replica", "pair", "left", "right", "topology", "frames",
        "distance_mean_A", "distance_sd_A", "distance_min_A", "distance_q05_A",
        "distance_median_A", "distance_q95_A", "soft_q_mean", "soft_q_sd",
        "statistical_inefficiency", "tau_integrated_ps", "effective_sample_size",
        "first_half_soft_q", "second_half_soft_q",
        *(f"hard_{threshold:g}A_fraction" for threshold in HARD_THRESHOLDS_A),
        *(f"hard_{threshold:g}A_events" for threshold in HARD_THRESHOLDS_A),
    ]
    write_csv(output_dir / "pair_replica_summary.csv", replica_rows, replica_fields)

    by_pair_replica: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for summary_row in replica_rows:
        by_pair_replica[(str(summary_row["state"]), str(summary_row["pair"]))].append(summary_row)

    state_pair_rows: list[dict[str, object]] = []
    for key in sorted(by_pair_replica):
        state, pair_name = key
        summaries = by_pair_replica[key]
        q_means = [float(row["soft_q_mean"]) for row in summaries]
        q_mean, q_sd, q_se, q_low, q_high = mean_ci95_replica(q_means)
        rhat = split_rhat(score_series[key])
        combined_ess = float(sum(float(row["effective_sample_size"]) for row in summaries))
        first = float(np.mean([float(row["first_half_soft_q"]) for row in summaries]))
        second = float(np.mean([float(row["second_half_soft_q"]) for row in summaries]))
        meta = pair_metadata[key]
        state_pair_rows.append(
            {
                "state": state,
                "system": manifest[state]["system"],
                "pair": pair_name,
                "left": meta["left"],
                "right": meta["right"],
                "topology": meta["topology"],
                "replicas": len(summaries),
                "soft_q_mean": q_mean,
                "soft_q_replica_sd": q_sd,
                "soft_q_replica_se": q_se,
                "soft_q_ci95_low": q_low,
                "soft_q_ci95_high": q_high,
                "first_half_soft_q": first,
                "second_half_soft_q": second,
                "half_shift": second - first,
                "split_rhat": finite_or_blank(rhat),
                "combined_effective_sample_size": combined_ess,
                "max_tau_integrated_ps": max(float(row["tau_integrated_ps"]) for row in summaries),
                "min_replica_effective_sample_size": min(
                    float(row["effective_sample_size"]) for row in summaries
                ),
                **{
                    f"hard_{threshold:g}A_fraction": float(
                        np.mean([float(row[f"hard_{threshold:g}A_fraction"]) for row in summaries])
                    )
                    for threshold in HARD_THRESHOLDS_A
                },
                **{
                    f"hard_{threshold:g}A_events_total": int(
                        sum(int(row[f"hard_{threshold:g}A_events"]) for row in summaries)
                    )
                    for threshold in HARD_THRESHOLDS_A
                },
                "convergence": convergence_class(rhat, combined_ess),
            }
        )

    # A pair's importance is its share of the state's total mean encounter
    # score.  The 2% threshold is only a triage rule for deciding what to
    # extend; all pair-resolved values remain in the output.
    for state in states:
        selected = [row for row in state_pair_rows if row["state"] == state]
        total_q = sum(float(row["soft_q_mean"]) for row in selected)
        for row in selected:
            fraction = float(row["soft_q_mean"]) / total_q if total_q > 0.0 else 0.0
            row["soft_q_mass_fraction"] = fraction
            row["material_pair_2pct"] = fraction >= 0.02

    state_pair_fields = [
        "state", "system", "pair", "left", "right", "topology", "replicas", "soft_q_mean",
        "soft_q_replica_sd", "soft_q_replica_se", "soft_q_ci95_low", "soft_q_ci95_high",
        "soft_q_mass_fraction", "material_pair_2pct", "first_half_soft_q", "second_half_soft_q",
        "half_shift", "split_rhat", "combined_effective_sample_size", "max_tau_integrated_ps",
        "min_replica_effective_sample_size",
        *(f"hard_{threshold:g}A_fraction" for threshold in HARD_THRESHOLDS_A),
        *(f"hard_{threshold:g}A_events_total" for threshold in HARD_THRESHOLDS_A),
        "convergence",
    ]
    write_csv(output_dir / "pair_state_summary.csv", state_pair_rows, state_pair_fields)

    topology_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    for state in states:
        for topology in ("same_arm", "cross_arm", "all"):
            replica_means: list[float] = []
            for replica in range(1, replicas + 1):
                if topology == "all":
                    pieces = [
                        values
                        for (candidate_state, candidate_replica, _), values in replica_topology_scores.items()
                        if candidate_state == state and candidate_replica == replica
                    ]
                    flat = [value for group in pieces for value in group]
                else:
                    flat = replica_topology_scores.get((state, replica, topology), [])
                if flat:
                    replica_means.append(float(np.mean(np.asarray(flat))))
            if not replica_means:
                continue
            mean, sd, se, low, high = mean_ci95_replica(replica_means)
            topology_rows.append(
                {
                    "state": state,
                    "system": manifest[state]["system"],
                    "topology": topology,
                    "replicas": len(replica_means),
                    "soft_q_mean": mean,
                    "soft_q_replica_sd": sd,
                    "soft_q_replica_se": se,
                    "soft_q_ci95_low": low,
                    "soft_q_ci95_high": high,
                }
            )
        pair_rows = [row for row in state_pair_rows if row["state"] == state]
        material = [row for row in pair_rows if bool(row["material_pair_2pct"])]
        convergence_counts = {
            label: sum(row["convergence"] == label for row in pair_rows)
            for label in ("pass", "caution", "extend")
        }
        material_extend = [row for row in material if row["convergence"] == "extend"]
        material_caution = [row for row in material if row["convergence"] == "caution"]
        if material_extend:
            recommendation = "extend_selected_pairs"
        elif material_caution:
            recommendation = "usable_with_caution"
        else:
            recommendation = "sufficient_for_current_contact_observable"
        state_rows.append(
            {
                "state": state,
                "system": manifest[state]["system"],
                "trajectory_duration_ps": expected_by_state[state][1],
                "pairs": len(pair_rows),
                "material_pairs_2pct": len(material),
                "pairs_pass": convergence_counts["pass"],
                "pairs_caution": convergence_counts["caution"],
                "pairs_extend": convergence_counts["extend"],
                "material_pairs_extend": len(material_extend),
                "max_material_split_rhat": max(
                    (float(row["split_rhat"]) for row in material if row["split_rhat"] != ""),
                    default=math.nan,
                ),
                "min_material_combined_ess": min(
                    (float(row["combined_effective_sample_size"]) for row in material),
                    default=math.nan,
                ),
                "max_material_tau_ps": max(
                    (float(row["max_tau_integrated_ps"]) for row in material),
                    default=math.nan,
                ),
                "recommendation": recommendation,
            }
        )

    topology_fields = [
        "state", "system", "topology", "replicas", "soft_q_mean", "soft_q_replica_sd",
        "soft_q_replica_se", "soft_q_ci95_low", "soft_q_ci95_high",
    ]
    write_csv(output_dir / "topology_state_summary.csv", topology_rows, topology_fields)
    state_fields = [
        "state", "system", "trajectory_duration_ps", "pairs", "material_pairs_2pct",
        "pairs_pass", "pairs_caution",
        "pairs_extend", "material_pairs_extend", "max_material_split_rhat",
        "min_material_combined_ess", "max_material_tau_ps", "recommendation",
    ]
    write_csv(output_dir / "state_convergence_summary.csv", state_rows, state_fields)

    quality_passed = len(fatal_issues) == 0 and len(quality_rows) == len(states) * replicas
    report = {
        "analysis": "replicated explicit-DCM distance-trajectory audit",
        "states": list(states),
        "replicas_per_state": replicas,
        "trajectory_files_expected": len(states) * replicas,
        "trajectory_files_read": sum(row.get("status") == "pass" for row in quality_rows),
        "frames_read": total_frames,
        "expected_frames_total": sum(expected_by_state[state][0] * replicas for state in states),
        "expected_duration_ps_by_state": {
            state: expected_by_state[state][1] for state in states
        },
        "pair_columns_by_state": {
            state: len(json.loads(manifest[state]["pair_schema_json"])) for state in states
        },
        "quality_passed": quality_passed,
        "fatal_issues": fatal_issues,
        "contact_definition": {
            "soft_sigmoid_center_A": center_A,
            "soft_sigmoid_width_A": width_A,
            "hard_sensitivity_thresholds_A": list(HARD_THRESHOLDS_A),
        },
        "uncertainty_unit": "independent replica",
        "autocorrelation_method": "Geyer initial-positive-sequence statistical inefficiency",
        "convergence_rule": {
            "pass": "split_Rhat <= 1.05 and combined ESS >= 400",
            "caution": "split_Rhat <= 1.10 and combined ESS >= 200",
            "extend": "otherwise",
            "extension_triage": "only pairs carrying >=2% of state soft-contact mass control the state recommendation",
        },
        "state_recommendations": {row["state"]: row["recommendation"] for row in state_rows},
    }
    (output_dir / "data_quality.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown_report(output_dir / "README.md", report, state_rows, state_pair_rows)
    make_figures(output_dir, state_pair_rows, topology_rows)
    return report


def write_markdown_report(
    path: Path,
    report: dict[str, object],
    state_rows: Sequence[dict[str, object]],
    pair_rows: Sequence[dict[str, object]],
) -> None:
    lines = [
        "# Replicated explicit-DCM trajectory audit",
        "",
        f"- Files: {report['trajectory_files_read']}/{report['trajectory_files_expected']} passed structural QC.",
        f"- Frames: {report['frames_read']}/{report['expected_frames_total']} read.",
        "- Uncertainty unit: one independently reseeded replica; frame-level independence is not assumed.",
        "- Primary score: sigmoid contact with center 5 Å and width 0.5 Å; hard 4–7 Å occupancies are sensitivity checks.",
        "- Convergence: split-Rhat plus autocorrelation-corrected effective sample size (ESS).",
        "",
        "## State decisions",
        "",
        "| State | duration (ps) | material pairs | max Rhat | min ESS | max tau (ps) | decision |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in state_rows:
        lines.append(
            f"| {row['state']} | {float(row['trajectory_duration_ps']):.0f} | "
            f"{row['material_pairs_2pct']} | "
            f"{float(row['max_material_split_rhat']):.4g} | "
            f"{float(row['min_material_combined_ess']):.1f} | "
            f"{float(row['max_material_tau_ps']):.1f} | {row['recommendation']} |"
        )
    lines.extend(
        [
            "",
            "A material pair is one carrying at least 2% of the state's total mean soft-contact score. "
            "This threshold is a simulation-extension triage rule, not part of the chemical model.",
            "",
            "## Pairs that control an extension decision",
            "",
            "| State | pair | topology | q mass | Rhat | ESS | status |",
            "|---|---|---|---:|---:|---:|---|",
        ]
    )
    selected = [
        row for row in pair_rows
        if bool(row["material_pair_2pct"]) and row["convergence"] != "pass"
    ]
    for row in sorted(selected, key=lambda item: (str(item["state"]), -float(item["soft_q_mass_fraction"]))):
        lines.append(
            f"| {row['state']} | {row['pair']} | {row['topology']} | "
            f"{100.0 * float(row['soft_q_mass_fraction']):.2f}% | {row['split_rhat']} | "
            f"{float(row['combined_effective_sample_size']):.1f} | {row['convergence']} |"
        )
    if not selected:
        lines.append("| — | — | — | — | — | — | no material pair requires extension |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `trajectory_quality.csv`: completeness, time grid, and numeric integrity.",
            "- `pair_replica_summary.csv`: distance/contact statistics and ESS for each independent replica.",
            "- `pair_state_summary.csv`: replica-level uncertainty, split-Rhat, and pair-resolved contact features.",
            "- `topology_state_summary.csv`: same-arm/cross-arm/all-pair summaries.",
            "- `state_convergence_summary.csv`: compact current-duration sufficiency decision.",
            "- `figures/`: pair scores and convergence diagnostics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def make_figures(
    output_dir: Path,
    pair_rows: Sequence[dict[str, object]],
    topology_rows: Sequence[dict[str, object]],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    states = list(dict.fromkeys(str(row["state"]) for row in pair_rows))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, state in zip(axes.flat, states):
        selected = sorted(
            (row for row in pair_rows if row["state"] == state),
            key=lambda row: float(row["soft_q_mean"]),
            reverse=True,
        )
        labels = [str(row["pair"]).split("_")[0] + "-" + str(row["pair"]).split("_")[1] for row in selected]
        means = np.asarray([float(row["soft_q_mean"]) for row in selected])
        errors = np.asarray([float(row["soft_q_replica_se"]) for row in selected])
        colors = ["#d95f02" if row["topology"] == "same_arm" else "#1b9e77" for row in selected]
        x = np.arange(len(selected))
        axis.bar(x, means, yerr=errors, color=colors, alpha=0.85, capsize=2)
        axis.set_title(state)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=90, fontsize=7)
        axis.set_ylabel("mean soft-contact score")
    fig.suptitle("Pair-resolved contact scores (error bars: replica SE)")
    fig.savefig(figure_dir / "pair_soft_contact.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for state, marker in zip(states, ("o", "s", "^", "D")):
        selected = [row for row in pair_rows if row["state"] == state]
        axis.scatter(
            [
                min(2.0, float(row["split_rhat"]))
                if row["split_rhat"] != "" else 2.0
                for row in selected
            ],
            [float(row["combined_effective_sample_size"]) for row in selected],
            s=[30.0 + 800.0 * float(row["soft_q_mass_fraction"]) for row in selected],
            alpha=0.7,
            marker=marker,
            label=state,
        )
    axis.axvline(1.05, color="#666666", linestyle="--", linewidth=1)
    axis.axhline(400.0, color="#666666", linestyle="--", linewidth=1)
    axis.set_xlabel("split-Rhat")
    axis.set_ylabel("combined effective sample size")
    axis.set_yscale("log")
    axis.legend(frameon=False, fontsize=8)
    axis.set_title("Convergence diagnostic (point area reflects q mass)")
    fig.savefig(figure_dir / "convergence_rhat_ess.svg")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    topology_names = ("same_arm", "cross_arm")
    x = np.arange(len(states))
    width = 0.34
    for offset, topology in enumerate(topology_names):
        selected = [
            next(row for row in topology_rows if row["state"] == state and row["topology"] == topology)
            for state in states
        ]
        axis.bar(
            x + (offset - 0.5) * width,
            [float(row["soft_q_mean"]) for row in selected],
            width,
            yerr=[float(row["soft_q_replica_se"]) for row in selected],
            capsize=3,
            label=topology.replace("_", " "),
        )
    axis.set_xticks(x)
    axis.set_xticklabels([state.removeprefix("A4_m_").removesuffix("_q0") for state in states])
    axis.set_ylabel("mean soft-contact score per pair")
    axis.set_title("Topology-resolved encounter scores")
    axis.legend(frameon=False)
    fig.savefig(figure_dir / "topology_contact.svg")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default="results/explicit_dcm_q0")
    parser.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--center-A", type=float, default=5.0)
    parser.add_argument("--width-A", type=float, default=0.5)
    parser.add_argument("--expected-frames", type=int, default=4001)
    parser.add_argument("--expected-dt-ps", type=float, default=0.5)
    parser.add_argument("--expected-duration-ps", type=float, default=2000.0)
    parser.add_argument(
        "--state-duration-ps",
        action="append",
        default=[],
        metavar="STATE=PS",
        help="override one selected state's expected duration; completed extension markers are auto-detected",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.replicas < 2:
        raise SystemExit("at least two replicas are required for convergence analysis")
    if args.width_A <= 0.0:
        raise SystemExit("--width-A must be positive")
    root = Path(args.root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    report = analyze(
        root,
        output,
        states=args.states,
        replicas=args.replicas,
        center_A=args.center_A,
        width_A=args.width_A,
        expected_frames=args.expected_frames,
        expected_dt_ps=args.expected_dt_ps,
        expected_duration_ps=args.expected_duration_ps,
        state_duration_ps=parse_state_durations(args.state_duration_ps),
    )
    print(
        f"Audited {report['frames_read']}/{report['expected_frames_total']} frames across "
        f"{report['trajectory_files_read']}/{report['trajectory_files_expected']} valid replicas."
    )
    print(f"Structural data-quality gate passed: {report['quality_passed']}")
    for state, decision in report["state_recommendations"].items():
        print(f"{state}: {decision}")
    print(f"Results: {output}")
    if not report["quality_passed"]:
        raise SystemExit("structural data-quality gate failed; inspect data_quality.json")


if __name__ == "__main__":
    main()
