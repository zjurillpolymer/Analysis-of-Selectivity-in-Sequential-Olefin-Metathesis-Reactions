#!/usr/bin/env python3
"""Continue completed distance-umbrella windows from their checkpoints.

The input pathway contains the completed 500 ps ``production.tpr`` and
``production.cpt`` for every window. ``gmx convert-tpr`` extends each tpr by
the requested amount and a separate ``extension`` deffnm is used, so the
original production files are never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path, log: Path) -> None:
    env = os.environ.copy()
    env.setdefault("GMX_MAXBACKUP", "-1")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"$ {shlex.join(command)}\n")
        print(f"$ {shlex.join(command)}", flush=True)
        process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, env=env)
        assert process.stdout is not None
        for line in process.stdout:
            handle.write(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"command failed ({returncode}); see {log}")


def read_xvg(path: Path) -> tuple[list[str], list[tuple[float, str]]]:
    headers, values = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "@")):
            headers.append(line)
        else:
            values.append((float(line.split()[0]), line))
    return headers, values


def pull_parts(window: Path, label: str) -> list[Path]:
    # Explicit -px/-pf and default deffnm place the part suffix differently.
    return sorted(set(window.glob(f"extension_{label}.part*.xvg"))
                  | set(window.glob(f"extension.part*_{label}.xvg")))


def last_pull_time(window: Path) -> float:
    paths = pull_parts(window, "pullx")
    if not paths:
        return float("-inf")
    return max((time for path in paths for time, _ in read_xvg(path)[1]), default=float("-inf"))


def materialize_outputs(window: Path, gmx: str, log: Path) -> None:
    """Keep immutable part files and expose canonical extension-only outputs."""
    for label in ("pullx", "pullf"):
        paths = pull_parts(window, label)
        if not paths:
            raise FileNotFoundError(f"no {label} parts in {window}")
        headers = read_xvg(paths[0])[0]
        samples = {}
        for path in paths:
            for time, line in read_xvg(path)[1]:
                samples[round(time, 6)] = line
        (window / f"extension_{label}.xvg").write_text(
            "\n".join(headers + [samples[time] for time in sorted(samples)]) + "\n",
            encoding="utf-8",
        )
    trajectories = sorted(window.glob("extension.part*.xtc"))
    if len(trajectories) == 1:
        shutil.copy2(trajectories[0], window / "extension.xtc")
    elif trajectories:
        run([gmx, "trjcat", "-f", *map(str, trajectories),
             "-o", str(window / "extension.xtc")], window, log)
    else:
        raise FileNotFoundError(f"no extension trajectory parts in {window}")
    structures = sorted(window.glob("extension.part*.gro"))
    if not structures:
        raise FileNotFoundError(f"no final extension coordinates in {window}")
    shutil.copy2(structures[-1], window / "extension.gro")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pathway", type=Path, required=True)
    parser.add_argument("--extend-ps", type=float, default=1500.0)
    parser.add_argument("--ntomp", type=int, default=4)
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--windows", nargs="+", help="Optional window names for a bounded verification run")
    args = parser.parse_args()
    pathway = args.pathway.resolve()
    windows = sorted(pathway.joinpath("windows").glob("window_*"))
    if args.windows:
        windows = [pathway / "windows" / name for name in args.windows]
    if not windows:
        raise FileNotFoundError(f"no windows under {pathway}")
    log = pathway / "extension_commands.log"
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    end_ps = float(metadata["production_ps_per_window"]) + args.extend_ps
    results: list[dict[str, object]] = []
    for window in windows:
        old_tpr = window / "production.tpr"
        old_cpt = window / "production.cpt"
        if not old_tpr.is_file() or not old_cpt.is_file():
            raise FileNotFoundError(f"incomplete production checkpoint in {window}")
        extended_tpr = window / "extension.tpr"
        if not extended_tpr.is_file():
            run(
                [args.gmx, "convert-tpr", "-s", str(old_tpr),
                 "-extend", f"{args.extend_ps:g}", "-o", str(extended_tpr)],
                window, log,
            )
        prefix = window / "extension"
        if last_pull_time(window) < end_ps - 1e-4 or not list(window.glob("extension.part*.gro")):
            checkpoint = prefix.with_suffix(".cpt")
            if not checkpoint.is_file():
                checkpoint = old_cpt
            command = [args.gmx, "mdrun", "-s", str(extended_tpr),
                       "-cpi", str(checkpoint), "-cpo", str(prefix.with_suffix(".cpt")),
                       "-deffnm", str(prefix), "-noappend",
                       "-ntmpi", "1", "-ntomp", str(args.ntomp), "-pin", "on",
                       "-px", str(window / "extension_pullx.xvg"),
                       "-pf", str(window / "extension_pullf.xvg")]
            if args.gpu:
                command += ["-nb", "gpu", "-pme", "gpu", "-bonded", "gpu", "-update", "gpu"]
            run(command, window, log)
        if last_pull_time(window) < end_ps - 1e-4:
            raise RuntimeError(f"{window.name}: stopped at {last_pull_time(window)} ps; target {end_ps} ps")
        materialize_outputs(window, args.gmx, log)
        results.append({
            "window": window.name,
            "old_production_tpr": old_tpr.name,
            "old_production_cpt": old_cpt.name,
            "extension_tpr": extended_tpr.name,
            "extension_ps": args.extend_ps,
            "observed_end_ps": last_pull_time(window),
            "status": "complete",
        })
        print(f"complete={window.name}", flush=True)
    (pathway / "extension_manifest.json").write_text(
        json.dumps({"pathway": pathway.name, "extension_ps": args.extend_ps,
                    "windows": results}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pathway": pathway.name, "windows": len(results),
                      "extension_ps": args.extend_ps}, indent=2), flush=True)


if __name__ == "__main__":
    main()
