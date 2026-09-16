#!/usr/bin/env python3
"""Recompute q2-adjacent WHAM profiles on pathway-specific sampled ranges."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np


PATHWAYS = (
    "c5_adj_same_v2_v3",
    "c5_adj_cross_v2_v1",
    "c8_adj_same_v1_v2",
    "c8_adj_cross_v1_v4",
)
BEGIN_PS = 100.0
END_PS = 500.0
MIDPOINT_PS = 300.0
TEMPERATURE_K = 313.15
BOOTSTRAPS = 200
SEED = 20260829


def read_pull_coordinate(path: Path, begin_ps: float, end_ps: float) -> np.ndarray:
    rows: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped[0] in "#@&":
            continue
        values = stripped.split()
        time_ps, coordinate_nm = float(values[0]), float(values[1])
        if begin_ps <= time_ps <= end_ps:
            rows.append(coordinate_nm)
    if not rows:
        raise ValueError(f"no pull-coordinate data in {begin_ps}..{end_ps} ps: {path}")
    return np.asarray(rows, dtype=float)


def run(command: list[str], log_path: Path) -> None:
    env = os.environ.copy()
    env.setdefault("GMX_MAXBACKUP", "-1")
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"WHAM failed ({result.returncode}); see {log_path}")


def wham_command(
    gmx: str,
    tpr_list: Path,
    pullx_list: Path,
    output: Path,
    minimum_nm: float,
    maximum_nm: float,
    bins: int,
    reference_nm: float,
    begin_ps: float,
    end_ps: float,
    label: str,
    bootstrap: bool,
) -> list[str]:
    suffix = "" if label == "full" else f"_{label}"
    command = [
        gmx, "wham", "-it", str(tpr_list), "-ix", str(pullx_list),
        "-temp", f"{TEMPERATURE_K:.2f}", "-unit", "kCal",
        "-min", f"{minimum_nm:.4f}", "-max", f"{maximum_nm:.4f}",
        "-bins", str(bins), "-zprof0", f"{reference_nm:.4f}",
        "-b", f"{begin_ps:.1f}", "-e", f"{end_ps:.1f}",
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
    return command


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path,
        default=project_root / "results/a4_q2_adjacent_umbrella_bohrium/download",
    )
    parser.add_argument(
        "--output", type=Path,
        default=project_root / "results/a4_q2_adjacent_umbrella_analysis/wham_recomputed",
    )
    parser.add_argument("--gmx", default=os.environ.get("GMX", "gmx"))
    args = parser.parse_args()
    input_root, output_root = args.input.resolve(), args.output.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for pathway_name in PATHWAYS:
        pathway = input_root / pathway_name / "pathway"
        windows = sorted((pathway / "windows").glob("window_*"))
        tprs = [window / "production.tpr" for window in windows]
        pullx = [window / "production_pullx.xvg" for window in windows]
        if not windows or not all(path.is_file() for path in (*tprs, *pullx)):
            raise FileNotFoundError(f"incomplete umbrella windows: {pathway}")
        coordinates = np.concatenate([
            read_pull_coordinate(path, BEGIN_PS, END_PS) for path in pullx
        ])
        minimum_nm = min(0.31, math.floor((float(coordinates.min()) - 0.01) * 100) / 100)
        maximum_nm = math.ceil((float(coordinates.max()) + 0.01) * 100) / 100
        reference_nm = float(np.quantile(coordinates, 0.995))
        bins = max(180, math.ceil((maximum_nm - minimum_nm) / 0.008))
        output = output_root / pathway_name
        output.mkdir(parents=True, exist_ok=True)
        tpr_list, pullx_list = output / "tpr-files.dat", output / "pullx-files.dat"
        tpr_list.write_text("\n".join(map(str, tprs)) + "\n", encoding="utf-8")
        pullx_list.write_text("\n".join(map(str, pullx)) + "\n", encoding="utf-8")

        for label, begin_ps, end_ps, bootstrap in (
            ("full", BEGIN_PS, END_PS, True),
            ("first_half", BEGIN_PS, MIDPOINT_PS, False),
            ("second_half", MIDPOINT_PS, END_PS, False),
        ):
            command = wham_command(
                args.gmx, tpr_list, pullx_list, output, minimum_nm, maximum_nm,
                bins, reference_nm, begin_ps, end_ps, label, bootstrap,
            )
            run(command, output / f"wham_{label}.stdout.log")

        row = {
            "pathway": pathway_name,
            "window_count": len(windows),
            "observed_min_nm": float(coordinates.min()),
            "observed_max_nm": float(coordinates.max()),
            "wham_min_nm": minimum_nm,
            "wham_max_nm": maximum_nm,
            "wham_reference_nm": reference_nm,
            "bins": bins,
            "begin_ps": BEGIN_PS,
            "end_ps": END_PS,
            "bootstrap_profiles": BOOTSTRAPS,
            "gromacs_version_policy": "local GROMACS reads archived 2024.2 TPR files",
        }
        (output / "recompute.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        manifest.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    (output_root / "recompute_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(f"output={output_root}")


if __name__ == "__main__":
    main()
