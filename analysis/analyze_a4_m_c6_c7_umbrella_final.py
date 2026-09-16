#!/usr/bin/env python3
"""Recompute and analyze the complete A4-m-C6/C7 umbrella data set.

The implementation deliberately follows the validated C5/C8 analysis:
WHAM is recomputed on each pathway's observed coordinate range, contact odds
use all histogram-supported non-contact distances, and uncertainty comes from
the 200 trajectory-bootstrap WHAM profiles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis.analyze_a4_m_c5_c8_umbrella_comparison import (
    CONTACT_SPLIT_NM,
    RT,
    contact_free_energy,
    q025_50_975,
    read_bootstrap_profiles,
    read_xvg,
)


TEMPERATURE_K = 313.15
BOOTSTRAPS = 200
SEED = 20260911


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_coordinate(path: Path, begin_ps: float, end_ps: float) -> np.ndarray:
    rows: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#@&":
            continue
        time_ps, coordinate_nm = map(float, stripped.split()[:2])
        if begin_ps <= time_ps <= end_ps:
            rows.append(coordinate_nm)
    if not rows:
        raise ValueError(f"no pull-coordinate data in {begin_ps}..{end_ps}: {path}")
    return np.asarray(rows, dtype=float)


def run(command: list[str], log: Path) -> None:
    env = os.environ.copy()
    env.setdefault("GMX_MAXBACKUP", "-1")
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def recompute_wham(pathway: Path, output: Path, gmx: str) -> dict[str, object]:
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    begin_ps = float(metadata["wham_begin_ps"])
    windows = sorted((pathway / "windows").glob("window_*"))
    tprs = [window / "production.tpr" for window in windows]
    pullx = [window / "production_pullx.xvg" for window in windows]
    if len(windows) != int(metadata["window_count"]) or not all(
        path.is_file() for path in (*tprs, *pullx)
    ):
        raise FileNotFoundError(f"incomplete umbrella windows: {pathway}")

    # Older pathway manifests recorded only the WHAM start time.  In that
    # case infer the production end time from the last timestamp in a pull-x
    # file (the C5/C8 q1 jobs use 0--500 ps production trajectories).
    if "wham_end_ps" in metadata:
        end_ps = float(metadata["wham_end_ps"])
    else:
        last_times = []
        for source in pullx:
            for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if stripped and stripped[0] not in "#@&":
                    last_times.append(float(stripped.split()[0]))
        if not last_times:
            raise ValueError(f"no timestamps in umbrella pull-x files: {pathway}")
        end_ps = max(last_times)
    midpoint_ps = 0.5 * (begin_ps + end_ps)

    coordinates = np.concatenate([
        read_coordinate(path, begin_ps, end_ps) for path in pullx
    ])
    minimum_nm = min(0.31, math.floor((float(coordinates.min()) - 0.01) * 100) / 100)
    maximum_nm = math.ceil((float(coordinates.max()) + 0.01) * 100) / 100
    reference_nm = float(np.quantile(coordinates, 0.995))
    bins = max(180, math.ceil((maximum_nm - minimum_nm) / 0.008))
    output.mkdir(parents=True, exist_ok=True)
    tpr_list, pullx_list = output / "tpr-files.dat", output / "pullx-files.dat"
    tpr_list.write_text("\n".join(map(str, tprs)) + "\n", encoding="utf-8")
    pullx_list.write_text("\n".join(map(str, pullx)) + "\n", encoding="utf-8")

    for label, begin, end, bootstrap in (
        ("full", begin_ps, end_ps, True),
        ("first_half", begin_ps, midpoint_ps, False),
        ("second_half", midpoint_ps, end_ps, False),
    ):
        suffix = "" if label == "full" else f"_{label}"
        command = [
            gmx, "wham", "-it", str(tpr_list), "-ix", str(pullx_list),
            "-temp", f"{TEMPERATURE_K:.2f}", "-unit", "kCal",
            "-min", f"{minimum_nm:.4f}", "-max", f"{maximum_nm:.4f}",
            "-bins", str(bins), "-zprof0", f"{reference_nm:.4f}",
            "-b", f"{begin:.1f}", "-e", f"{end:.1f}",
            "-o", str(output / f"pmf{suffix}.xvg"),
            "-hist", str(output / f"histograms{suffix}.xvg"),
        ]
        if bootstrap:
            command.extend([
                "-oiact", str(output / "iact.xvg"), "-ac",
                "-nBootstrap", str(BOOTSTRAPS), "-bs-method", "traj",
                "-bs-seed", str(SEED),
                "-bsres", str(output / "bootstrap_result.xvg"),
                "-bsprof", str(output / "bootstrap_profiles.xvg"),
            ])
        run(command, output / f"wham_{label}.stdout.log")

    record = {
        "pathway": metadata["pathway"],
        "window_count": len(windows),
        "begin_ps": begin_ps,
        "end_ps": end_ps,
        "observed_min_nm": float(coordinates.min()),
        "observed_max_nm": float(coordinates.max()),
        "wham_min_nm": minimum_nm,
        "wham_max_nm": maximum_nm,
        "wham_reference_nm": reference_nm,
        "bins": bins,
        "bootstrap_profiles": BOOTSTRAPS,
    }
    (output / "recompute.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def load_pathway(pathway: Path, wham: Path, minimum_count: float = 5.0,
                 split_nm: float = CONTACT_SPLIT_NM) -> dict[str, object]:
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    pmf = read_xvg(wham / "pmf.xvg")[:, :2]
    first = read_xvg(wham / "pmf_first_half.xvg")[:, :2]
    second = read_xvg(wham / "pmf_second_half.xvg")[:, :2]
    histogram = read_xvg(wham / "histograms.xvg")
    if not np.allclose(pmf[:, 0], histogram[:, 0]):
        raise ValueError(f"PMF/histogram grids differ: {wham}")
    supported = np.sum(histogram[:, 1:], axis=1) >= minimum_count
    bootstrap = read_bootstrap_profiles(wham / "bootstrap_profiles.xvg")
    bootstrap_dg = np.asarray([
        contact_free_energy(profile, supported, split_nm) for profile in bootstrap
    ])
    with (pathway / "window_diagnostics.csv").open(encoding="utf-8", newline="") as handle:
        diagnostics = list(csv.DictReader(handle))
    overlaps = [float(row["left_neighbor_gaussian_overlap"]) for row in diagnostics
                if row.get("left_neighbor_gaussian_overlap", "").strip()]
    support_x = pmf[supported, 0]
    return {
        "metadata": metadata,
        "dg": contact_free_energy(pmf, supported, split_nm),
        "dg_first": contact_free_energy(first, supported, split_nm),
        "dg_second": contact_free_energy(second, supported, split_nm),
        "bootstrap_dg": bootstrap_dg,
        "bootstrap_ci": q025_50_975(bootstrap_dg),
        "support_min_nm": float(support_x.min()),
        "support_max_nm": float(support_x.max()),
        "minimum_neighbor_overlap": min(overlaps),
        "maximum_mean_center_error_nm": max(
            abs(float(row["mean_minus_center_nm"])) for row in diagnostics
        ),
        "window_count": len(diagnostics),
    }


def difference(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (left[:, None] - right[None, :]).ravel()


def contrast_row(label: str, definition: str, left: dict[str, object],
                 right: dict[str, object]) -> dict[str, object]:
    point = float(left["dg"] - right["dg"])
    distribution = difference(left["bootstrap_dg"], right["bootstrap_dg"])
    low, median, high = q025_50_975(distribution)
    return {
        "quantity": label,
        "definition": definition,
        "point_ddG_kcal_mol": point,
        "bootstrap_median_kcal_mol": median,
        "bootstrap_ci_low_kcal_mol": low,
        "bootstrap_ci_high_kcal_mol": high,
        "left_over_right_contact_odds": math.exp(-point / RT),
        "probability_ddG_below_zero": float(np.mean(distribution < 0)),
        "first_half_ddG_kcal_mol": float(left["dg_first"] - right["dg_first"]),
        "second_half_ddG_kcal_mol": float(left["dg_second"] - right["dg_second"]),
    }


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gmx", default=os.environ.get("GMX", "gmx"))
    parser.add_argument(
        "--output", type=Path,
        default=project / "results/a4_m_c6_c7_umbrella_analysis",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    wham_root = output / "wham_recomputed"
    output.mkdir(parents=True, exist_ok=True)

    raw_roots = (
        project / "results/a4_m_c6_c7_q0_extracted",
        project / "results/a4_m_c6_c7_q2_extracted",
    )
    pathways: dict[str, Path] = {}
    for raw_root in raw_roots:
        for path in raw_root.glob("*/pathway"):
            metadata = json.loads((path / "pathway.json").read_text(encoding="utf-8"))
            pathways[str(metadata["pathway"])] = path
    if len(pathways) != 16:
        raise RuntimeError(f"expected 16 unique pathways, found {len(pathways)}")

    recompute_records = []
    for name, pathway in sorted(pathways.items()):
        recompute_records.append(recompute_wham(pathway, wham_root / name, args.gmx))
        print(f"recomputed {name}", flush=True)
    (output / "recompute_manifest.json").write_text(
        json.dumps(recompute_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    loaded = {
        name: load_pathway(pathway, wham_root / name)
        for name, pathway in pathways.items()
    }
    pathway_rows = []
    for name, item in sorted(loaded.items()):
        metadata = item["metadata"]
        low, median, high = item["bootstrap_ci"]
        pathway_rows.append({
            "pathway": name,
            "system": metadata["system"],
            "stage": metadata["stage"],
            "background": metadata.get("background", "q0"),
            "anchor": metadata.get("anchor", ""),
            "route_family": metadata.get("route_family", ""),
            "topology": metadata.get("topology", ""),
            "pair": f"V{metadata['left']}-V{metadata['right']}",
            "contact_dG_kcal_mol": item["dg"],
            "bootstrap_median_kcal_mol": median,
            "bootstrap_ci_low_kcal_mol": low,
            "bootstrap_ci_high_kcal_mol": high,
            "first_half_contact_dG_kcal_mol": item["dg_first"],
            "second_half_contact_dG_kcal_mol": item["dg_second"],
            "support_min_nm": item["support_min_nm"],
            "support_max_nm": item["support_max_nm"],
            "minimum_neighbor_overlap": item["minimum_neighbor_overlap"],
            "maximum_mean_center_error_nm": item["maximum_mean_center_error_nm"],
            "window_count": item["window_count"],
        })

    q0_pairs = {
        "C6_V3": ("c6_q0_anchor_v3_cross_v3_v8", "c6_q0_anchor_v3_same_v3_v5"),
        "C6_V8": ("c6_q0_anchor_v8_cross_v1_v8", "c6_q0_anchor_v8_same_v7_v8"),
        "C7_V2": ("c7_q0_anchor_v2_cross_v2_v4", "c7_q0_anchor_v2_same_v2_v3"),
        "C7_V5": ("c7_q0_anchor_v5_cross_v2_v5", "c7_q0_anchor_v5_same_v4_v5"),
    }
    contrasts = [
        contrast_row(label, "cross minus same", loaded[cross], loaded[same])
        for label, (cross, same) in q0_pairs.items()
    ]
    contrasts.extend([
        contrast_row(
            "C6_q2_adjacent", "cross minus same",
            loaded["c6_q2_adjacent_cross_v4_v7"],
            loaded["c6_q2_adjacent_same_v4_v6"],
        ),
        contrast_row(
            "C7_q2_adjacent", "cross minus same",
            loaded["c7_q2_adjacent_cross_v2_v8"],
            loaded["c7_q2_adjacent_same_v7_v8"],
        ),
        contrast_row(
            "C6_q2_opposite", "switch minus reuse",
            loaded["c6_q2_opposite_switch_v3_v7"],
            loaded["c6_q2_opposite_reuse_v3_v4"],
        ),
        contrast_row(
            "C7_q2_opposite", "switch minus reuse",
            loaded["c7_q2_opposite_switch_v2_v6"],
            loaded["c7_q2_opposite_reuse_v2_v5"],
        ),
    ])

    sensitivity = []
    comparison_specs = [
        (label, "cross minus same", cross, same)
        for label, (cross, same) in q0_pairs.items()
    ] + [
        ("C6_q2_adjacent", "cross minus same", "c6_q2_adjacent_cross_v4_v7", "c6_q2_adjacent_same_v4_v6"),
        ("C7_q2_adjacent", "cross minus same", "c7_q2_adjacent_cross_v2_v8", "c7_q2_adjacent_same_v7_v8"),
        ("C6_q2_opposite", "switch minus reuse", "c6_q2_opposite_switch_v3_v7", "c6_q2_opposite_reuse_v3_v4"),
        ("C7_q2_opposite", "switch minus reuse", "c7_q2_opposite_switch_v2_v6", "c7_q2_opposite_reuse_v2_v5"),
    ]
    for minimum_count in (1.0, 5.0, 20.0):
        for split_nm in (0.45, 0.50, 0.55):
            values = {
                name: load_pathway(pathways[name], wham_root / name, minimum_count, split_nm)["dg"]
                for name in pathways
            }
            for label, definition, left, right in comparison_specs:
                ddg = float(values[left] - values[right])
                sensitivity.append({
                    "quantity": label,
                    "definition": definition,
                    "minimum_histogram_count": minimum_count,
                    "contact_split_nm": split_nm,
                    "ddG_kcal_mol": ddg,
                    "left_over_right_contact_odds": math.exp(-ddg / RT),
                })

    write_csv(output / "pathway_contact_free_energies.csv", pathway_rows)
    write_csv(output / "selectivity_contrasts.csv", contrasts)
    write_csv(output / "contact_definition_sensitivity.csv", sensitivity)
    summary = {
        "temperature_K": TEMPERATURE_K,
        "contact_split_nm": CONTACT_SPLIT_NM,
        "minimum_histogram_count": 5.0,
        "pathways": pathway_rows,
        "contrasts": contrasts,
        "sensitivity": sensitivity,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pathways": len(pathway_rows), "contrasts": contrasts,
                      "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
