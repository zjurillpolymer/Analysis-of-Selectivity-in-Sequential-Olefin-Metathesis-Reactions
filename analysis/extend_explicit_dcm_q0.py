#!/usr/bin/env python3
"""Continue selected completed explicit-DCM replicas from 2 ns to a longer time.

The default is the convergence-driven minimum experiment: continue all four
replicas of A4_m_C5_q0 and A4_m_C8_q0 from their existing checkpoints to
4 ns total.  Velocities are not regenerated.  Before the first append, the
complete original 2 ns output set is copied to ``pre_extension_2ns``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from run_explicit_dcm_gromacs import run_command


DEFAULT_STATES = ("A4_m_C5_q0", "A4_m_C8_q0")
ORIGINAL_TOTAL_NS = 2.0
BACKUP_SUFFIXES = (".tpr", ".cpt", ".xtc", ".edr", ".log", ".gro")


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_marker(replica_dir: Path, target_total_ns: float) -> dict[str, object] | None:
    marker = replica_dir / "extension_metadata.json"
    if not marker.is_file():
        return None
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if (
        metadata.get("status") == "complete"
        and float(metadata.get("target_total_ns", -1.0)) >= target_total_ns
    ):
        return metadata
    return None


def backup_original_outputs(replica_dir: Path) -> Path:
    prefix = replica_dir / "production_2ns"
    backup = replica_dir / "pre_extension_2ns"
    backup.mkdir(exist_ok=True)
    inventory: list[dict[str, object]] = []
    for suffix in BACKUP_SUFFIXES:
        source = prefix.with_suffix(suffix)
        if not source.is_file():
            if suffix in (".tpr", ".cpt", ".xtc", ".gro"):
                raise FileNotFoundError(f"required completed 2 ns output is missing: {source}")
            continue
        destination = backup / source.name
        if not destination.is_file():
            shutil.copy2(source, destination)
        inventory.append(
            {
                "file": source.name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    manifest = backup / "backup_manifest.json"
    if not manifest.is_file():
        manifest.write_text(
            json.dumps(
                {
                    "source": str(replica_dir),
                    "original_total_ns": ORIGINAL_TOTAL_NS,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "files": inventory,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return backup


def gromacs_environment(ntomp: int) -> dict[str, str]:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(ntomp)
    environment["HWLOC_COMPONENTS"] = "linux,stop"
    environment.pop("GMX_DISABLE_GPU_DETECTION", None)
    return environment


def part_files(production: Path, suffix: str) -> list[Path]:
    """Return GROMACS -noappend part files in simulation order."""
    return sorted(production.parent.glob(f"{production.name}.part*{suffix}"))


def newest_checkpoint(production: Path) -> Path:
    """Prefer a recovered partial-run checkpoint over the canonical one."""
    candidates = [production.with_suffix(".cpt"), *part_files(production, ".cpt")]
    candidates = [path for path in candidates if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"no continuation checkpoint below {production.parent}")
    return max(candidates, key=lambda path: (path.stat().st_mtime_ns, path.name))


def consolidate_noappend_outputs(
    production: Path,
    target_total_ns: float,
    log: Path,
) -> tuple[Path, list[Path], Path]:
    """Merge portable continuation parts and promote the newest state.

    Checkpoints retain absolute output paths, so appending after a workspace is
    uploaded to a different Bohrium directory is unsafe.  ``-noappend`` writes
    numbered part files without requiring those old paths.  Once mdrun
    succeeds, the trajectory parts are merged into the canonical XTC and the
    newest checkpoint/final structure are promoted for the next stage.
    """
    trajectories = part_files(production, ".xtc")
    structures = part_files(production, ".gro")
    canonical_checkpoint = production.with_suffix(".cpt")
    if not trajectories or not canonical_checkpoint.is_file() or not structures:
        raise RuntimeError(
            "portable continuation completed without part XTC, checkpoint, or part GRO: "
            f"xtc={trajectories}, cpt={canonical_checkpoint}, gro={structures}"
        )

    merged = production.parent / f".production_{target_total_ns:g}ns_merged.xtc"
    if merged.exists():
        merged.unlink()
    run_command(
        [
            "gmx", "trjcat",
            "-f", str(production.with_suffix(".xtc")),
            *[str(path) for path in trajectories],
            "-o", str(merged),
        ],
        production.parent,
        log,
    )
    merged.replace(production.with_suffix(".xtc"))

    latest_structure = max(
        structures, key=lambda path: (path.stat().st_mtime_ns, path.name)
    )
    promoted_structure = production.with_suffix(".gro.tmp")
    shutil.copy2(latest_structure, promoted_structure)
    promoted_structure.replace(production.with_suffix(".gro"))

    archive = production.parent / "extension_segments" / f"to_{target_total_ns:g}ns"
    archive.mkdir(parents=True, exist_ok=True)
    archived: list[Path] = []
    for path in sorted(production.parent.glob(f"{production.name}.part*")):
        destination = archive / path.name
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite archived segment: {destination}")
        shutil.move(str(path), destination)
        archived.append(destination)
    return canonical_checkpoint, archived, archive


def extend_replica(
    replica_dir: Path,
    state: str,
    replica: int,
    target_total_ns: float,
    ntomp: int,
) -> str:
    if completed_marker(replica_dir, target_total_ns) is not None:
        return "SKIPPED"
    if target_total_ns <= ORIGINAL_TOTAL_NS:
        raise ValueError(f"target total time must exceed {ORIGINAL_TOTAL_NS:g} ns")

    backup = backup_original_outputs(replica_dir)
    production = replica_dir / "production_2ns"
    extended_tpr = replica_dir / f"production_{target_total_ns:g}ns.tpr"
    extension_ps = (target_total_ns - ORIGINAL_TOTAL_NS) * 1000.0
    log = replica_dir / "extension_commands.log"
    if not extended_tpr.is_file():
        run_command(
            [
                "gmx", "convert-tpr",
                "-s", str(production.with_suffix(".tpr")),
                "-extend", f"{extension_ps:g}",
                "-o", str(extended_tpr),
            ],
            replica_dir,
            log,
        )

    continuation_checkpoint = newest_checkpoint(production)
    # Checkpoints record absolute output paths.  The Bohrium staging directory
    # changes between workflows, so -append cannot be used portably.  With
    # -noappend GROMACS writes numbered part files and safely resumes solely
    # from the checkpoint state; those parts are consolidated after success.
    command = [
        "gmx", "mdrun",
        "-s", str(extended_tpr),
        "-deffnm", str(production),
        "-cpi", str(continuation_checkpoint),
        "-noappend",
        "-ntmpi", "1",
        "-ntomp", str(ntomp),
        "-pin", "off",
        "-nb", "gpu",
        "-pme", "gpu",
        "-bonded", "cpu",
        "-update", "cpu",
    ]
    run_command(
        command,
        replica_dir,
        log,
        environment=gromacs_environment(ntomp),
    )
    promoted_checkpoint, archived_segments, segment_archive = (
        consolidate_noappend_outputs(production, target_total_ns, log)
    )
    required = [production.with_suffix(suffix) for suffix in (".cpt", ".xtc", ".gro")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"extension completed without required outputs: {missing}")
    (replica_dir / "extension_metadata.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "state": state,
                "replica": replica,
                "original_total_ns": ORIGINAL_TOTAL_NS,
                "target_total_ns": target_total_ns,
                "continuation_mode": "checkpoint_noappend_portable_no_velocity_reinitialization",
                "checkpoint_input": str(continuation_checkpoint),
                "promoted_checkpoint": str(promoted_checkpoint),
                "output_prefix": str(production),
                "extended_tpr": str(extended_tpr),
                "backup_dir": str(backup),
                "segment_archive": str(segment_archive),
                "archived_segments": [str(path) for path in archived_segments],
                "completed_utc": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return "OK"


def select_states(rows: Sequence[dict[str, str]], states: Sequence[str]) -> list[dict[str, str]]:
    by_state = {row["state"]: row for row in rows}
    unknown = set(states) - set(by_state)
    if unknown:
        raise ValueError(f"states absent from manifest: {sorted(unknown)}")
    selected = [by_state[state] for state in states]
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument(
        "--replica-indices",
        nargs="+",
        type=int,
        help="extend only these replica numbers (within 1..--replicas)",
    )
    parser.add_argument("--target-total-ns", type=float, default=4.0)
    parser.add_argument("--ntomp", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.replicas < 1 or args.ntomp < 1:
        raise SystemExit("--replicas and --ntomp must be positive")
    replica_indices = (
        list(range(1, args.replicas + 1))
        if args.replica_indices is None
        else args.replica_indices
    )
    if (
        not replica_indices
        or len(set(replica_indices)) != len(replica_indices)
        or any(index < 1 or index > args.replicas for index in replica_indices)
    ):
        raise SystemExit("--replica-indices must be unique values within 1..--replicas")
    if args.target_total_ns <= ORIGINAL_TOTAL_NS:
        raise SystemExit(f"--target-total-ns must exceed {ORIGINAL_TOTAL_NS:g}")
    root = Path(args.root).resolve()
    selected = select_states(
        read_manifest(root / "gromacs_dcm" / "state_manifest.csv"),
        args.states,
    )
    tasks = [
        (row, replica, root / row["output_dir"] / f"replica_{replica:02d}")
        for row in selected
        for replica in replica_indices
    ]
    print(
        f"Selected {len(selected)} states and {len(tasks)} replicas; "
        f"checkpoint continuation {ORIGINAL_TOTAL_NS:g} -> {args.target_total_ns:g} ns."
    )
    if args.dry_run:
        for row, replica, replica_dir in tasks:
            print(f"[DRY-RUN] {row['state']} replica {replica:02d}: {replica_dir}")
        return
    for index, (row, replica, replica_dir) in enumerate(tasks, start=1):
        print(f"[{index}/{len(tasks)}] {row['state']} replica {replica:02d}", flush=True)
        status = extend_replica(
            replica_dir,
            row["state"],
            replica,
            args.target_total_ns,
            args.ntomp,
        )
        print(f"[{status}] {row['state']} replica {replica:02d}", flush=True)
    print("Selected explicit-DCM checkpoint extensions completed.")
    print("Next: extract distances and run the analysis appropriate for the selected state(s).")


if __name__ == "__main__":
    main()
