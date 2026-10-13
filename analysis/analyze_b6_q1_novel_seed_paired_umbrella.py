#!/usr/bin/env python3
"""Jointly reweight the six B6 q1 paired umbrellas from novel candidate seeds."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "Analysis-of-Selectivity-in-Sequential-Olefin-Metathesis-Reactions"
for path in (REPO / "analysis", PROJECT / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from analyze_a4_m_c6_c7_q1_umbrella_2d import prepare_topology  # noqa: E402

JOINT_SCRIPT = REPO / "analysis/analyze_b6_local_q1_joint_mbar.py"
SPEC = importlib.util.spec_from_file_location("prior_b6_joint_mbar", JOINT_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import validated B6 joint-MBAR implementation: {JOINT_SCRIPT}")
JOINT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = JOINT
SPEC.loader.exec_module(JOINT)

SEEDS = {
    1: {
        "name": "A2 origin 05480 ps",
        "key": "A2_origin_05480ps_positive_pc1_return",
        "A6": 20814445,
        "A2": 20814444,
    },
    2: {
        "name": "A6 origin 00690 ps",
        "key": "A6_origin_00690ps_high_novelty_positive_pc1",
        "A6": 20814451,
        "A2": 20814450,
    },
    3: {
        "name": "A6 origin 01270 ps",
        "key": "A6_origin_01270ps_high_novelty_crossing_frame",
        "A6": 20814453,
        "A2": 20814452,
    },
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_path(path: Path, expected_seed: str, expected_family: str) -> None:
    metadata = json.loads((path / "pathway.json").read_text(encoding="utf-8"))
    if metadata["seed"] != expected_seed:
        raise ValueError(f"seed mismatch in {path}: {metadata['seed']} != {expected_seed}")
    if metadata["route_family"] != expected_family:
        raise ValueError(f"route mismatch in {path}: {metadata['route_family']} != {expected_family}")
    windows = sorted((path / "windows").glob("window_*"))
    if len(windows) != 25:
        raise ValueError(f"expected 25 windows in {path}, found {len(windows)}")
    for window in windows:
        for filename in ("production.tpr", "production.xtc", "production_pullx.xvg", "production.gro"):
            item = window / filename
            if not item.is_file() or item.stat().st_size == 0:
                raise FileNotFoundError(item)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-root", type=Path,
        default=PROJECT / "bohrium_download/b6_q1_novel_seed_umbrella_20261003",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "results/b6_q1_novel_seed_umbrella_20261003_analysis",
    )
    parser.add_argument("--gmx", default="gmx")
    args = parser.parse_args()
    download_root, output = args.download_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {output}")
    output.mkdir(parents=True)
    topology_root = output / "reactive_pair_topologies"
    JOINT.DOWNLOAD = download_root
    JOINT.OUTPUT = output
    JOINT.SINGLE = output
    JOINT.SOURCE_METADATA = REPO / "results/b6_local_q1_arm1_competition_20260923/source_metadata.json"
    if not JOINT.SOURCE_METADATA.is_file():
        # Retain the canonical metadata location used by the original analysis.
        JOINT.SOURCE_METADATA = PROJECT / (
            "bohrium_download/b6_neighbor_md_extensions_2to10ns_20260918/20750428/"
            "extracted/workspace/gromacs_dcm/runs/B6_p_C6_q1_V1_V2_cis/setup/metadata.json"
        )
    if not JOINT.SOURCE_METADATA.is_file():
        raise FileNotFoundError(f"B6 vinyl atom mapping metadata not found: {JOINT.SOURCE_METADATA}")

    summary_rows: list[dict[str, object]] = []
    all_results: list[dict[str, object]] = []
    for pair_id, seed in SEEDS.items():
        states: list[dict[str, object]] = []
        path_rows = {}
        for family in ("A6", "A2"):
            job_id = int(seed[family])
            pathway = download_root / str(job_id) / "extracted/pathway"
            expected_seed = str(seed["key"])
            expected_route = "A6_full_reuse" if family == "A6" else "A2_one_arm_reuse"
            validate_path(pathway, expected_seed, expected_route)
            prepare_topology(pathway, topology_root)
            rows = JOINT.window_data(pair_id, family, job_id)
            states.extend(rows)
            path_rows[family] = {
                "job_id": job_id,
                "pathway": json.loads((pathway / "pathway.json").read_text(encoding="utf-8")),
                "window_count": len(rows),
                "samples_after_100ps_discard": sum(len(row["time_ps"]) for row in rows),
                "maximum_pull_geometry_error_A": max(float(row["pull_max_abs_error_A"]) for row in rows),
            }
            print(f"seed {pair_id} {family}: extracted {len(rows)} windows", flush=True)
        if len(states) != 50:
            raise ValueError(f"seed {pair_id}: joint comparison needs 50 windows, got {len(states)}")
        pair = {
            "pair_id": pair_id,
            "seed": seed["name"],
            "jobs": {"A6": seed["A6"], "A2": seed["A2"]},
            "pathways": path_rows,
        }
        for segment in ("full", "first_half", "second_half"):
            pair[segment] = JOINT.solve(states, segment)
            solved = pair[segment]
            print(
                f"seed {pair_id} {segment}: A6/(2*A2)="
                f"{solved['ratio_A6_over_2A2']:.5g}; bridge="
                f"{solved['family_bridge_A6_to_A2']:.3g}/"
                f"{solved['family_bridge_A2_to_A6']:.3g}; 2D overlap="
                f"{solved['coarse_2d_family_histogram_overlap']:.3g}",
                flush=True,
            )
        full, first, second = pair["full"], pair["first_half"], pair["second_half"]
        pair["half_shift_deltaG_kcal_mol"] = (
            float(second["deltaG_A6_minus_A2_kcal_mol"])
            - float(first["deltaG_A6_minus_A2_kcal_mol"])
        )
        pair["interpretation"] = (
            "conditional two-channel ratio for this candidate seed; not a seed population or product yield"
        )
        all_results.append(pair)
        summary_rows.append({
            "seed": seed["name"],
            "A6_job_id": seed["A6"],
            "A2_job_id": seed["A2"],
            "frames": full["frames"],
            "ratio_A6_over_2A2": full["ratio_A6_over_2A2"],
            "deltaG_A6_minus_A2_kcal_mol": full["deltaG_A6_minus_A2_kcal_mol"],
            "P_A6_two_channels_only": full["P_A6_if_two_channels_only"],
            "P_A2_two_channels_only": full["P_A2_if_two_channels_only"],
            "first_half_ratio": first["ratio_A6_over_2A2"],
            "second_half_ratio": second["ratio_A6_over_2A2"],
            "half_shift_deltaG_kcal_mol": pair["half_shift_deltaG_kcal_mol"],
            "coarse_2d_histogram_overlap": full["coarse_2d_family_histogram_overlap"],
            "shared_0p1nm_cells_5plus_frames_each": full["shared_0p1nm_cells_with_5_frames_each"],
            "family_bridge_A6_to_A2": full["family_bridge_A6_to_A2"],
            "family_bridge_A2_to_A6": full["family_bridge_A2_to_A6"],
            "reweight_kish_ess": full["reweight_kish_ess"],
            "A6_NAC_weighted_frame_ESS": full["A6_NAC_weighted_frame_ESS"],
            "A2_NAC_weighted_frame_ESS": full["A2_NAC_weighted_frame_ESS"],
        })

    summary = {
        "temperature_K": 313.15,
        "analysis": "joint 2D distance-biased WHAM/MBAR-equivalent reweighting, with orientation-resolved soft NAC",
        "coordinate_pairs": {"A6": "V3-V12", "A2": "V3-V10"},
        "multiplicity": {"A6": 1, "A2": 2},
        "windows_per_family_per_seed": 25,
        "frames_per_window_used": 801,
        "discarded_production_ps": 100.0,
        "seeds": all_results,
        "aggregation_warning": "Seed-specific conditional estimates are reported separately; candidate-selection counts are not equilibrium basin populations and must not be pooled as pathway probabilities.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output / "paired_seed_comparison.csv", summary_rows)

    table = [
        "# B6 q1 novel-seed paired umbrella analysis",
        "",
        "The six completed paths were analyzed in three paired seed sets. Within each set, 25 A6 windows bias V3–V12 and 25 A2 windows bias V3–V10. The first 100 ps of each 500 ps production segment were excluded. Vinyl-axis and side-angle soft gates were retained; the A2 endpoint multiplicity was applied as ×2.",
        "",
        "| Candidate seed | A6/(2×A2) NAC weight | ΔG A6−A2 (kcal/mol) | A6 share* | first → second half ratio | 2D overlap | ESS (Kish) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        table.append(
            f"| {row['seed']} | {float(row['ratio_A6_over_2A2']):.4g} | "
            f"{float(row['deltaG_A6_minus_A2_kcal_mol']):+.3f} | "
            f"{float(row['P_A6_two_channels_only']):.3f} | "
            f"{float(row['first_half_ratio']):.3g} → {float(row['second_half_ratio']):.3g} | "
            f"{float(row['coarse_2d_histogram_overlap']):.3f} | "
            f"{float(row['reweight_kish_ess']):.0f} |"
        )
    table.extend([
        "",
        "*A6 share is normalized only over the two modeled channels, including the A2 ×2 endpoint multiplicity. It is not a prediction of experimental product yield.",
        "",
        "## Interpretation guardrails",
        "",
        "- The table reports a **conditional path ratio within each selected candidate seed**. The three seeds were deliberately selected for novelty/persistence; their count is not a thermodynamic population, so no pooled global q1 probability is reported.",
        "- The decisive robustness checks are the direction and magnitude consistency across three seed families, first/second-half stability, 2D family histogram overlap, and bridge/ESS diagnostics. Weak overlap or strong half-shifts make a nominally extreme ratio non-quantitative.",
        "- `summary.json` retains detailed per-seed full/half results and diagnostics; `paired_seed_comparison.csv` is the compact comparison table. The raw window PMFs and all trajectories remain in the downloaded/extracted Bohr outputs.",
        "",
    ])
    (output / "REPORT.md").write_text("\n".join(table), encoding="utf-8")
    print(f"REPORT={output / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
