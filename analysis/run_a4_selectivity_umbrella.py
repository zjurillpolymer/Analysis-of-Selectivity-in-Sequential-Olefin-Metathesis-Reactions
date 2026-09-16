#!/usr/bin/env python3
"""Run matched A4-p distance-umbrella paths sequentially on one GPU.

The preparation script writes two steering starts and independent equilibration
and production MDPs per window.  This runner is resumable and never launches
two GROMACS jobs concurrently, which is important for a single vGPU.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


def artifact(prefix: Path, suffix: str) -> Path:
    """Append a suffix without treating window decimals as file extensions."""
    return prefix.parent / f"{prefix.name}{suffix}"


def command(args: list[str], cwd: Path, log: Path, ntomp: int = 8, stdin: str | None = None) -> None:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(ntomp)
    env["HWLOC_COMPONENTS"] = "linux,stop"
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(args) + "\n")
        result = subprocess.run(args, cwd=cwd, input=stdin, text=True, stdout=handle, stderr=subprocess.STDOUT, env=env)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def grompp(path: Path, mdp: Path, coordinates: Path, top: Path, index: Path, tpr: Path, log: Path, checkpoint: Path | None = None) -> None:
    args = ["gmx", "grompp", "-f", str(mdp), "-c", str(coordinates), "-p", str(top), "-n", str(index), "-o", str(tpr), "-po", str(artifact(tpr, ".expanded.mdp")), "-maxwarn", "0"]
    if checkpoint is not None:
        args += ["-t", str(checkpoint)]
    command(args, path, log)


def mdrun(path: Path, prefix: Path, log: Path, ntomp: int) -> None:
    if artifact(prefix, ".gro").is_file():
        return
    command(["gmx", "mdrun", "-deffnm", str(prefix), "-ntmpi", "1", "-ntomp", str(ntomp), "-pin", "off", "-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "cpu"], path, log, ntomp)


def run_path(path: Path, ntomp: int) -> None:
    metadata = json.loads((path / "pathway.json").read_text(encoding="utf-8"))
    top, index, start = path / "system.top", path / "index.ndx", path / "start.gro"
    log = path / "run_commands.log"
    for required in (top, index, start, path / "steer_in.mdp", path / "steer_out.mdp"):
        if not required.is_file():
            raise FileNotFoundError(required)
    for name, mdp in (("steer_in", "steer_in.mdp"), ("steer_out", "steer_out.mdp")):
        tpr, prefix = path / f"{name}.tpr", path / name
        if not tpr.is_file():
            grompp(path, path / mdp, start, top, index, tpr, log)
        mdrun(path, prefix, log, ntomp)
    initial = float(metadata["initial_distance_nm"])
    in_gro, out_gro = path / "steer_in.gro", path / "steer_out.gro"
    windows = sorted((path / "window_mdps").glob("*_equil.mdp"))
    for equil_mdp in windows:
        window = equil_mdp.name.removesuffix("_equil.mdp")
        production_mdp = path / "window_mdps" / f"{window}_production.mdp"
        equil_tpr, equil_prefix = path / f"{window}_equil.tpr", path / f"{window}_equil"
        prod_tpr, prod_prefix = path / f"{window}_production.tpr", path / f"{window}_production"
        center = float(window.split("_")[-1].removesuffix("nm"))
        seed = in_gro if center <= initial else out_gro
        if not equil_tpr.is_file():
            grompp(path, equil_mdp, seed, top, index, equil_tpr, log)
        mdrun(path, equil_prefix, log, ntomp)
        if not prod_tpr.is_file():
            grompp(path, production_mdp, artifact(equil_prefix, ".gro"), top, index, prod_tpr, log, artifact(equil_prefix, ".cpt"))
        mdrun(path, prod_prefix, log, ntomp)
    print(f"[OK] {path.name}: {len(windows)} windows")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umbrella-root", type=Path, default=project / "umbrella/a4_p_c6_c8_selectivity")
    parser.add_argument("--ntomp", type=int, default=8)
    parser.add_argument("--paths", nargs="*", help="path directory names; default: all prepared paths")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    root = args.umbrella_root.resolve()
    paths = [root / name for name in args.paths] if args.paths else sorted(path for path in root.iterdir() if path.is_dir() and (path / "pathway.json").is_file())
    if not paths:
        raise SystemExit(f"no prepared pathway directories under {root}")
    if shutil.which("gmx") is None and not args.dry_run:
        raise RuntimeError("gmx is not on PATH")
    for path in paths:
        metadata = json.loads((path / "pathway.json").read_text(encoding="utf-8"))
        windows = len(list((path / "window_mdps").glob("*_equil.mdp")))
        print(f"{path.name}: state={metadata['state']} V{metadata['left']}-V{metadata['right']} class={metadata.get('route_class') or metadata.get('topology_class')} windows={windows}")
        if not args.dry_run:
            run_path(path, args.ntomp)


if __name__ == "__main__":
    main()
