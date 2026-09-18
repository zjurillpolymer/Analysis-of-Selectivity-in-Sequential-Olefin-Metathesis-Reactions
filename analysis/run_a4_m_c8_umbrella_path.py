#!/usr/bin/env python3
"""Run one prepared A4-m-C8 GROMACS umbrella pathway and perform WHAM.

The runner is resumable: completed steering legs and windows are skipped.  It
is intentionally pathway-local so four independent GPU jobs can run in
parallel without sharing output files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
from pathlib import Path


def run(
    command: list[str],
    *,
    cwd: Path,
    stdin: str | None = None,
    log_path: Path | None = None,
) -> None:
    printable = shlex.join(command)
    print(f"[{cwd.name}] {printable}", flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise RuntimeError(f"command failed ({result.returncode}): {printable}\n{tail}")


def mdrun_command(gmx: str, deffnm: Path, ntomp: int, gpu: bool) -> list[str]:
    command = [gmx, "mdrun", "-deffnm", str(deffnm), "-ntomp", str(ntomp), "-pin", "on"]
    if gpu:
        # CPU update/bonded tasks also work for topologies whose constrained
        # atoms are interspersed and therefore cannot form GPU update groups.
        command.extend(["-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "cpu"])
    return command


def read_windows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_pullx(path: Path, stride: int = 1) -> list[tuple[float, float]]:
    values: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line or line[0] in "#@":
            continue
        fields = line.split()
        if len(fields) < 2:
            continue
        values.append((float(fields[0]), float(fields[-1])))
    return values[::stride]


def steering_leg_covers_windows(pathway: Path, direction: str) -> bool:
    """Return true when a complete or interrupted steering leg spans its needed edge."""
    steering = pathway / "steering"
    pullx = steering / f"{direction}_pullx.xvg"
    trajectory = steering / f"{direction}.xtc"
    tpr = steering / f"{direction}.tpr"
    if not all(path.is_file() and path.stat().st_size > 0 for path in (pullx, trajectory, tpr)):
        return False
    samples = read_pullx(pullx, 10)
    if not samples:
        return False
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    edge = min(value for _, value in samples) if direction == "in" else max(value for _, value in samples)
    required = float(metadata["minimum_nm"] if direction == "in" else metadata["maximum_nm"])
    tolerance = 0.005
    return edge <= required + tolerance if direction == "in" else edge >= required - tolerance


def prepare_steering(pathway: Path, gmx: str, ntomp: int, gpu: bool) -> None:
    steering = pathway / "steering"
    steering.mkdir(exist_ok=True)
    for direction in ("in", "out"):
        deffnm = steering / direction
        if not deffnm.with_suffix(".tpr").exists():
            grompp = [
                gmx, "grompp", "-f", str(pathway / f"steer_{direction}.mdp"),
                "-c", str(pathway / "start.gro"),
            ]
            checkpoint = pathway / "start.cpt"
            if checkpoint.is_file():
                grompp.extend(["-t", str(checkpoint)])
            grompp.extend(
                [
                    "-p", str(pathway / "system.top"), "-n", str(pathway / "index.ndx"),
                    "-o", str(deffnm.with_suffix(".tpr")),
                    "-po", str(steering / f"{direction}_processed.mdp"), "-maxwarn", "0",
                ]
            )
            run(
                grompp,
                cwd=pathway,
                log_path=steering / f"{direction}_grompp.stdout.log",
            )
        if not deffnm.with_suffix(".gro").exists() and steering_leg_covers_windows(pathway, direction):
            print(
                f"[{pathway.name}] reusing interrupted {direction} steering leg; "
                "it already covers all requested window centers",
                flush=True,
            )
            continue
        if not deffnm.with_suffix(".gro").exists():
            command = mdrun_command(gmx, deffnm, ntomp, gpu)
            command.extend(
                ["-px", str(steering / f"{direction}_pullx.xvg"),
                 "-pf", str(steering / f"{direction}_pullf.xvg")]
            )
            try:
                run(command, cwd=pathway, log_path=steering / f"{direction}_mdrun.stdout.log")
            except RuntimeError:
                if steering_leg_covers_windows(pathway, direction):
                    print(
                        f"[{pathway.name}] {direction} steering terminated after covering all "
                        "window centers; continuing with the usable partial trajectory",
                        flush=True,
                    )
                else:
                    raise


def select_window_sources(pathway: Path, windows: list[dict[str, str]]) -> list[dict[str, object]]:
    candidates: list[tuple[str, float, float]] = []
    # pull coordinates are written every 0.5 ps and the full-system steering
    # trajectory every 5 ps, so every tenth pull row maps to an XTC frame.
    for direction in ("in", "out"):
        pullx = pathway / "steering" / f"{direction}_pullx.xvg"
        candidates.extend((direction, time_ps, distance_nm) for time_ps, distance_nm in read_pullx(pullx, 10))
    if not candidates:
        raise RuntimeError("no steering coordinates were found")
    selected: list[dict[str, object]] = []
    for window in windows:
        center = float(window["center_nm"])
        direction, time_ps, actual = min(candidates, key=lambda item: abs(item[2] - center))
        selected.append(
            {
                **window,
                "source_direction": direction,
                "source_time_ps": time_ps,
                "source_distance_nm": actual,
                "absolute_error_nm": abs(actual - center),
            }
        )
    return selected


def extract_window_starts(
    pathway: Path,
    gmx: str,
    windows: list[dict[str, object]],
) -> None:
    with (pathway / "window_sources.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(windows[0].keys()))
        writer.writeheader()
        writer.writerows(windows)
    for window in windows:
        window_root = pathway / "windows" / str(window["window_name"])
        window_root.mkdir(parents=True, exist_ok=True)
        start = window_root / "start.gro"
        if start.exists():
            continue
        direction = str(window["source_direction"])
        run(
            [
                gmx, "trjconv", "-f", str(pathway / "steering" / f"{direction}.xtc"),
                "-s", str(pathway / "steering" / f"{direction}.tpr"),
                "-n", str(pathway / "index.ndx"), "-dump", f"{float(window['source_time_ps']):.6f}",
                "-pbc", "mol", "-o", str(start),
            ],
            cwd=pathway,
            stdin="System\n",
            log_path=window_root / "extract.stdout.log",
        )


def run_windows(pathway: Path, gmx: str, ntomp: int, gpu: bool, windows: list[dict[str, object]]) -> None:
    for number, window in enumerate(windows, start=1):
        name = str(window["window_name"])
        root = pathway / "windows" / name
        print(f"[{pathway.name}] window {number}/{len(windows)}: {name}", flush=True)
        equil = root / "equil"
        production = root / "production"
        if not equil.with_suffix(".tpr").exists():
            run(
                [
                    gmx, "grompp", "-f", str(pathway / "window_mdps" / f"{name}_equil.mdp"),
                    "-c", str(root / "start.gro"), "-p", str(pathway / "system.top"),
                    "-n", str(pathway / "index.ndx"), "-o", str(equil.with_suffix(".tpr")),
                    "-po", str(root / "equil_processed.mdp"), "-maxwarn", "0",
                ], cwd=pathway, log_path=root / "equil_grompp.stdout.log",
            )
        if not equil.with_suffix(".gro").exists():
            command = mdrun_command(gmx, equil, ntomp, gpu)
            command.extend(["-px", str(root / "equil_pullx.xvg"), "-pf", str(root / "equil_pullf.xvg")])
            run(command, cwd=pathway, log_path=root / "equil_mdrun.stdout.log")
        if not production.with_suffix(".tpr").exists():
            run(
                [
                    gmx, "grompp", "-f", str(pathway / "window_mdps" / f"{name}_production.mdp"),
                    "-c", str(equil.with_suffix(".gro")), "-t", str(equil.with_suffix(".cpt")),
                    "-p", str(pathway / "system.top"), "-n", str(pathway / "index.ndx"),
                    "-o", str(production.with_suffix(".tpr")),
                    "-po", str(root / "production_processed.mdp"), "-maxwarn", "0",
                ], cwd=pathway, log_path=root / "production_grompp.stdout.log",
            )
        if not production.with_suffix(".gro").exists():
            command = mdrun_command(gmx, production, ntomp, gpu)
            command.extend(
                ["-px", str(root / "production_pullx.xvg"),
                 "-pf", str(root / "production_pullf.xvg")]
            )
            run(command, cwd=pathway, log_path=root / "production_mdrun.stdout.log")


def normal_overlap(mean_a: float, sd_a: float, mean_b: float, sd_b: float) -> float:
    pooled = math.sqrt((sd_a * sd_a + sd_b * sd_b) / 2.0)
    if pooled == 0.0:
        return float(mean_a == mean_b)
    return math.erfc(abs(mean_a - mean_b) / (2.0 * math.sqrt(2.0) * pooled))


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def write_diagnostics(pathway: Path, windows: list[dict[str, object]], begin_ps: float) -> None:
    summaries: list[dict[str, float | str]] = []
    raw_values: list[list[float]] = []
    for window in windows:
        root = pathway / "windows" / str(window["window_name"])
        samples = [value for time, value in read_pullx(root / "production_pullx.xvg") if time >= begin_ps]
        if not samples:
            raise RuntimeError(f"no post-equilibration pull samples for {window['window_name']}")
        mean, sd = mean_sd(samples)
        raw_values.append(samples)
        summaries.append(
            {
                "window_name": str(window["window_name"]),
                "center_nm": float(window["center_nm"]),
                "sample_count": len(samples),
                "mean_nm": mean,
                "sd_nm": sd,
                "mean_minus_center_nm": mean - float(window["center_nm"]),
                "left_neighbor_gaussian_overlap": "",
            }
        )
    for index in range(1, len(summaries)):
        summaries[index]["left_neighbor_gaussian_overlap"] = normal_overlap(
            float(summaries[index - 1]["mean_nm"]), float(summaries[index - 1]["sd_nm"]),
            float(summaries[index]["mean_nm"]), float(summaries[index]["sd_nm"]),
        )
    with (pathway / "window_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def run_wham(
    pathway: Path,
    gmx: str,
    windows: list[dict[str, object]],
    begin_ps: float,
    end_ps: float,
) -> None:
    wham = pathway / "wham"
    wham.mkdir(exist_ok=True)
    tprs = [pathway / "windows" / str(w["window_name"]) / "production.tpr" for w in windows]
    pullx = [pathway / "windows" / str(w["window_name"]) / "production_pullx.xvg" for w in windows]
    (wham / "tpr-files.dat").write_text("\n".join(map(str, tprs)) + "\n", encoding="utf-8")
    (wham / "pullx-files.dat").write_text("\n".join(map(str, pullx)) + "\n", encoding="utf-8")
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    sampled_minimum = min(float(window["center_nm"]) for window in windows)
    sampled_maximum = max(float(window["center_nm"]) for window in windows)
    minimum = sampled_minimum - 0.03
    maximum = sampled_maximum + 0.03
    common = [
        gmx, "wham", "-it", str(wham / "tpr-files.dat"), "-ix", str(wham / "pullx-files.dat"),
        "-temp", "313.15", "-unit", "kCal", "-min", f"{minimum:.4f}", "-max", f"{maximum:.4f}",
        # GROMACS 2024.x treats an explicit ``-noauto`` as if the ``-auto``
        # option was user-set and then rejects it together with -min/-max.
        # Supplying both bounds without mentioning -auto makes WHAM disable
        # automatic bounds internally, which is the intended behavior.
        "-bins", "300", "-zprof0", f"{sampled_maximum:.4f}",
    ]
    if not (wham / "pmf.xvg").exists():
        run(
            common + [
                "-b", f"{begin_ps:.1f}", "-o", str(wham / "pmf.xvg"),
                "-hist", str(wham / "histograms.xvg"), "-oiact", str(wham / "iact.xvg"),
                "-ac", "-nBootstrap", "200", "-bs-method", "traj", "-bs-seed", "20260827",
                "-bsres", str(wham / "bootstrap_result.xvg"),
                "-bsprof", str(wham / "bootstrap_profiles.xvg"),
            ], cwd=pathway, log_path=wham / "wham.stdout.log",
        )
    midpoint = (begin_ps + end_ps) / 2.0
    for label, begin, end in (("first_half", begin_ps, midpoint), ("second_half", midpoint, end_ps)):
        output = wham / f"pmf_{label}.xvg"
        if output.exists():
            continue
        run(
            common + ["-b", f"{begin:.1f}", "-e", f"{end:.1f}", "-o", str(output),
                      "-hist", str(wham / f"histograms_{label}.xvg")],
            cwd=pathway, log_path=wham / f"wham_{label}.stdout.log",
        )


def preflight(pathway: Path, gmx: str) -> None:
    output = pathway / "preflight"
    output.mkdir(exist_ok=True)
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    steering_grompp = [
        gmx, "grompp", "-f", str(pathway / "steer_in.mdp"), "-c", str(pathway / "start.gro"),
    ]
    checkpoint = pathway / "start.cpt"
    if checkpoint.is_file():
        steering_grompp.extend(["-t", str(checkpoint)])
    steering_grompp.extend(
        [
            "-p", str(pathway / "system.top"), "-n", str(pathway / "index.ndx"),
            "-o", str(output / "steer_in.tpr"),
            "-po", str(output / "steer_in_processed.mdp"), "-maxwarn", "0",
        ]
    )
    run(steering_grompp, cwd=pathway, log_path=output / "steer_in_grompp.stdout.log")
    first = read_windows(pathway / "windows.csv")[0]
    # Window preflight uses the valid full-system endpoint as coordinates.  It
    # tests topology/index/MDP compatibility, not proximity to the window center.
    run(
        [
            gmx, "grompp", "-f", str(pathway / "window_mdps" / f"{first['window_name']}_equil.mdp"),
            "-c", str(pathway / "start.gro"), "-p", str(pathway / "system.top"),
            "-n", str(pathway / "index.ndx"), "-o", str(output / "window_equil.tpr"),
            "-po", str(output / "window_equil_processed.mdp"), "-maxwarn", "0",
        ], cwd=pathway, log_path=output / "window_equil_grompp.stdout.log",
    )
    (output / "PASS.json").write_text(
        json.dumps({"pathway": pathway.name, "pair": [metadata["left"], metadata["right"]]}, indent=2) + "\n",
        encoding="utf-8",
    )


def completed_wham_windows(
    pathway: Path,
    windows: list[dict[str, object]],
    end_ps: float,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    completed: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    for window in windows:
        root = pathway / "windows" / str(window["window_name"])
        tpr = root / "production.tpr"
        pullx = root / "production_pullx.xvg"
        reason = ""
        final_time_ps = float("nan")
        if not tpr.is_file() or tpr.stat().st_size == 0:
            reason = "missing production.tpr"
        elif not pullx.is_file() or pullx.stat().st_size == 0:
            reason = "missing production_pullx.xvg"
        else:
            samples = read_pullx(pullx)
            if not samples:
                reason = "empty production_pullx.xvg"
            else:
                final_time_ps = samples[-1][0]
                if final_time_ps < end_ps - 0.51:
                    reason = f"incomplete pull trajectory ({final_time_ps:.3f} < {end_ps:.3f} ps)"
        if reason:
            excluded.append({**window, "final_time_ps": final_time_ps, "exclusion_reason": reason})
        else:
            completed.append(window)
    return completed, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathway", type=Path, required=True)
    parser.add_argument("--gmx", default=os.environ.get("GMX", "gmx"))
    parser.add_argument("--ntomp", type=int, default=4)
    parser.add_argument("--gpu", action="store_true", help="Use GPU nonbonded and PME; CPU bonded and update for topology compatibility")
    parser.add_argument("--stage", choices=("preflight", "steering", "windows", "wham", "all"), default="all")
    parser.add_argument(
        "--wham-begin-ps", type=float,
        help="Trajectory time at which WHAM begins; defaults to pathway metadata",
    )
    args = parser.parse_args()
    pathway = args.pathway.resolve()
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    equilibration_ps = float(metadata.get("equilibration_ps_per_window", 200.0))
    production_ps = float(metadata.get("production_ps_per_window", 2000.0))
    begin_ps = (
        float(args.wham_begin_ps)
        if args.wham_begin_ps is not None
        else float(metadata.get("wham_begin_ps", equilibration_ps + min(200.0, 0.2 * production_ps)))
    )
    # Window production is grompp'ed as a new run and its pull output starts at
    # 0 ps even though velocities are read from the equilibration checkpoint.
    # Older q2 metadata recorded equilibration + production here; clamp that
    # value to the actual production trajectory endpoint.
    end_ps = min(float(metadata.get("wham_end_ps", production_ps)), production_ps)
    if not 0.0 <= begin_ps < end_ps:
        raise ValueError(f"invalid WHAM interval: {begin_ps}..{end_ps} ps")
    if args.stage == "preflight":
        preflight(pathway, args.gmx)
        return
    windows_csv = read_windows(pathway / "windows.csv")
    if args.stage in ("steering", "all"):
        prepare_steering(pathway, args.gmx, args.ntomp, args.gpu)
    if args.stage in ("windows", "all"):
        selected = select_window_sources(pathway, windows_csv)
        extract_window_starts(pathway, args.gmx, selected)
        run_windows(pathway, args.gmx, args.ntomp, args.gpu, selected)
    if args.stage in ("wham", "all"):
        selected_path = pathway / "window_sources.csv"
        selected: list[dict[str, object]] = read_windows(selected_path) if selected_path.exists() else windows_csv
        selected, excluded = completed_wham_windows(pathway, selected, end_ps)
        if excluded:
            exclusion_path = pathway / "wham_excluded_windows.csv"
            with exclusion_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=tuple(excluded[0].keys()))
                writer.writeheader()
                writer.writerows(excluded)
            print(
                f"[{pathway.name}] WHAM excludes {len(excluded)} incomplete window(s); "
                f"details={exclusion_path}",
                flush=True,
            )
        if len(selected) < 2:
            raise RuntimeError(f"only {len(selected)} complete umbrella window(s) are available")
        write_diagnostics(pathway, selected, begin_ps)
        run_wham(pathway, args.gmx, selected, begin_ps, end_ps)


if __name__ == "__main__":
    main()
