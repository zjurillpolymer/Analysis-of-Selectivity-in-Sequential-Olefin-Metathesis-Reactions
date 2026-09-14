#!/usr/bin/env python3
"""Run pure-solvent validation and the explicit-DCM GROMACS pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Sequence


GEN_SEED_PATTERN = re.compile(r"^(gen-seed\s*=\s*)-?\d+\s*$", re.MULTILINE | re.IGNORECASE)
TARGET_DCM_DENSITY_G_CM3 = 1.29
DENSITY_RELATIVE_TOLERANCE = 0.03
DENSITY_STATIONARITY_TOLERANCE = 0.01
DEFAULT_DCM_MODEL = "ch2cl2_dc"
DCM_MODELS = ("ch2cl2_dc", "gaff2_am1bcc")
SUPPORTED_REACTION_STAGES = tuple(f"q{index}" for index in range(7))


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_seed(label: str) -> int:
    value = int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little")
    return 1 + value % (2**31 - 2)


def render_seeded_mdp(source: Path, destination: Path, seed: int) -> None:
    text = source.read_text(encoding="utf-8")
    updated, count = GEN_SEED_PATTERN.subn(lambda match: f"{match.group(1)}{seed}", text)
    if count != 1:
        raise ValueError(f"expected exactly one gen-seed in {source}, found {count}")
    destination.write_text(updated, encoding="utf-8")


def run_command(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    stdin: str | None = None,
    environment: dict[str, str] | None = None,
) -> None:
    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(command)}\n")
        log.flush()
        result = subprocess.run(
            list(command),
            cwd=cwd,
            input=stdin,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
            env=environment,
        )
        log.write(f"exit={result.returncode}; elapsed_s={time.perf_counter() - started:.1f}\n")
    if result.returncode != 0:
        # The remote console otherwise contains only this Python traceback,
        # while the useful GROMACS fatal error remains inside the workspace.
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-80:])
        except OSError as error:
            tail = f"<could not read log tail: {error}>"
        raise RuntimeError(
            f"command failed with exit code {result.returncode}: "
            f"{' '.join(command)}; see {log_path}\n"
            f"--- last 80 log lines ---\n{tail}\n--- end log tail ---"
        )


def grompp(
    *,
    mdp: Path,
    coordinates: Path,
    topology: Path,
    output_tpr: Path,
    log: Path,
    checkpoint: Path | None = None,
    index: Path | None = None,
) -> None:
    command = [
        "gmx", "grompp",
        "-f", str(mdp),
        "-c", str(coordinates),
        "-p", str(topology),
        "-o", str(output_tpr),
        "-po", str(output_tpr.with_suffix(".expanded.mdp")),
        "-maxwarn", "0",
    ]
    if checkpoint is not None:
        command.extend(["-t", str(checkpoint)])
    if index is not None:
        command.extend(["-n", str(index)])
    run_command(command, output_tpr.parent, log)


def ensure_em_tpr(setup: Path, mdp_dir: Path) -> bool:
    """Create the deferred EM input in the GROMACS runtime image."""
    output = setup / "em.tpr"
    if output.is_file():
        return False
    coordinates = setup / "system.gro"
    topology = setup / "system.top"
    for required in (coordinates, topology):
        if not required.is_file():
            raise FileNotFoundError(f"deferred GROMACS input is missing: {required}")
    grompp(
        mdp=mdp_dir / "em.mdp",
        coordinates=coordinates,
        topology=topology,
        output_tpr=output,
        log=setup / "setup.log",
    )
    return True


def mdrun(prefix: Path, ntomp: int, energy_minimization: bool = False) -> None:
    final_gro = prefix.with_suffix(".gro")
    if final_gro.is_file():
        return
    command = [
        "gmx", "mdrun",
        "-deffnm", str(prefix),
        "-ntmpi", "1",
        "-ntomp", str(ntomp),
        # Forced affinity can conflict with a cloud container's assigned
        # cpuset, which need not match the visible host CPU topology.
        "-pin", "off",
    ]
    environment = os.environ.copy()
    # The server image exports OMP_NUM_THREADS=16.  GROMACS treats a mismatch
    # with -ntomp as fatal, so keep the two sources of truth identical.
    environment["OMP_NUM_THREADS"] = str(ntomp)
    # Restrict embedded hwloc to the Linux CPU backend.  Scanning additional
    # topology components can block in vGPU containers before mdrun opens its
    # log; CUDA device selection remains GROMACS' responsibility.
    environment["HWLOC_COMPONENTS"] = "linux,stop"
    if energy_minimization:
        # EM is CPU-only in this protocol.  Avoid an unnecessary CUDA hardware
        # probe, which can block before GROMACS even creates em.log on some
        # virtual-GPU/container driver combinations.
        command.extend(
            ["-nb", "cpu", "-pme", "cpu", "-bonded", "cpu", "-update", "cpu"]
        )
        environment["GMX_DISABLE_GPU_DETECTION"] = "1"
    else:
        # Do not inherit a previous CPU-only debugging override.
        environment.pop("GMX_DISABLE_GPU_DETECTION", None)
        # CH2Cl2-DC is rigid: all three intramolecular distances are explicit
        # constraints, so the pure-solvent topology has no ordinary bonded
        # interaction type to offload.  GROMACS treats a forced
        # ``-bonded gpu`` in that situation as a fatal input inconsistency.
        # Keep the small bonded workload on CPU for every system.  GROMACS'
        # GPU update path also rejects the three mutually coupled constraints
        # of this rigid triangular solvent, so constraints/update remain on
        # CPU while the expensive nonbonded and PME work stays on GPU.
        command.extend(
            ["-nb", "gpu", "-pme", "gpu", "-bonded", "cpu", "-update", "cpu"]
        )
    checkpoint = prefix.with_suffix(".cpt")
    if checkpoint.is_file():
        command.extend(["-cpi", str(checkpoint), "-append"])
    run_command(
        command,
        prefix.parent,
        prefix.parent / "run_commands.log",
        environment=environment,
    )
    if not final_gro.is_file():
        raise RuntimeError(f"mdrun did not produce {final_gro}")


def run_base_equilibration(
    setup: Path,
    run_dir: Path,
    mdp_dir: Path,
    label: str,
    ntomp: int,
) -> Path:
    base = run_dir / "base_equilibration"
    base.mkdir(parents=True, exist_ok=True)
    topology = setup / "system.top"
    log = base / "run_commands.log"

    em = base / "em"
    if not em.with_suffix(".tpr").is_file():
        shutil.copy2(setup / "em.tpr", em.with_suffix(".tpr"))
    mdrun(em, ntomp, energy_minimization=True)

    nvt_mdp = base / "nvt_seeded.mdp"
    render_seeded_mdp(mdp_dir / "nvt.mdp", nvt_mdp, stable_seed(f"{label}|base_nvt"))
    nvt = base / "nvt"
    if not nvt.with_suffix(".tpr").is_file():
        grompp(
            mdp=nvt_mdp,
            coordinates=em.with_suffix(".gro"),
            topology=topology,
            output_tpr=nvt.with_suffix(".tpr"),
            log=log,
        )
    mdrun(nvt, ntomp)

    npt = base / "npt"
    if not npt.with_suffix(".tpr").is_file():
        grompp(
            mdp=mdp_dir / "npt.mdp",
            coordinates=nvt.with_suffix(".gro"),
            topology=topology,
            output_tpr=npt.with_suffix(".tpr"),
            checkpoint=nvt.with_suffix(".cpt"),
            log=log,
        )
    mdrun(npt, ntomp)
    return npt


def parse_xvg(path: Path) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith(("#", "@")):
            continue
        fields = line.split()
        rows.append((float(fields[0]), float(fields[1])))
    if not rows:
        raise ValueError(f"no data rows in {path}")
    return rows


def read_index_groups(path: Path) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped.strip("[] ")
            groups[current] = []
        elif stripped and not stripped.startswith(";"):
            if current is None:
                raise ValueError(f"index data appears before a group in {path}")
            groups[current].extend(map(int, stripped.split()))
    return groups


def ensure_simulation_index(setup: Path, solute_atom_count: int) -> bool:
    """Ensure a custom index retains both required production groups.

    Supplying ``grompp -n`` replaces the automatically generated default
    groups.  Production needs ``System`` for temperature coupling and
    ``Solute`` for the reduced XTC, so repair legacy setup indices that only
    contained the latter.
    """
    gro = setup / "system.gro"
    lines = gro.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise ValueError(f"invalid GRO file: {gro}")
    system_atom_count = int(lines[1].strip())
    if not 0 < solute_atom_count <= system_atom_count:
        raise ValueError(
            f"invalid solute/system atom counts: {solute_atom_count}/{system_atom_count}"
        )
    index = setup / "index.ndx"
    expected = {
        "System": list(range(1, system_atom_count + 1)),
        "Solute": list(range(1, solute_atom_count + 1)),
    }
    existing = read_index_groups(index) if index.is_file() else {}
    if all(existing.get(name) == values for name, values in expected.items()):
        return False
    output: list[str] = []
    for name, values in expected.items():
        output.append(f"[ {name} ]")
        for start in range(0, len(values), 15):
            output.append(" ".join(map(str, values[start : start + 15])))
        output.append("")
    index.write_text("\n".join(output) + "\n", encoding="utf-8")
    return True


def summarize_density_rows(rows: Sequence[tuple[float, float]]) -> dict[str, object]:
    """Summarize accuracy and stationarity over the final four nanoseconds."""
    maximum_time = max(time_ps for time_ps, _ in rows)
    analysis_start = maximum_time - 4000.0
    selected_rows = [row for row in rows if row[0] >= analysis_start]
    if len(selected_rows) < 100:
        raise ValueError("too few density samples in the final 4 ns")
    selected_g_cm3 = [(time_ps, density / 1000.0) for time_ps, density in selected_rows]
    mean_g_cm3 = sum(density for _, density in selected_g_cm3) / len(selected_g_cm3)
    relative_error = abs(mean_g_cm3 - TARGET_DCM_DENSITY_G_CM3) / TARGET_DCM_DENSITY_G_CM3

    block_means: list[float] = []
    for block in range(4):
        lower = analysis_start + block * 1000.0
        upper = lower + 1000.0
        values = [
            density
            for time_ps, density in selected_g_cm3
            if time_ps >= lower and (time_ps < upper or block == 3)
        ]
        if not values:
            raise ValueError(f"no density samples in block {block + 1}")
        block_means.append(sum(values) / len(values))
    block_mean = sum(block_means) / len(block_means)
    block_sd = math.sqrt(
        sum((value - block_mean) ** 2 for value in block_means) / (len(block_means) - 1)
    )
    block_sem = block_sd / math.sqrt(len(block_means))
    block_range_relative = (max(block_means) - min(block_means)) / TARGET_DCM_DENSITY_G_CM3

    times_ns = [(time_ps - analysis_start) / 1000.0 for time_ps, _ in selected_g_cm3]
    time_mean = sum(times_ns) / len(times_ns)
    denominator = sum((value - time_mean) ** 2 for value in times_ns)
    slope_g_cm3_per_ns = sum(
        (time_ns - time_mean) * (density - mean_g_cm3)
        for time_ns, (_, density) in zip(times_ns, selected_g_cm3)
    ) / denominator
    slope_relative_per_ns = slope_g_cm3_per_ns / TARGET_DCM_DENSITY_G_CM3
    stationary = (
        block_range_relative <= DENSITY_STATIONARITY_TOLERANCE
        and abs(slope_relative_per_ns) <= DENSITY_STATIONARITY_TOLERANCE / 2.0
    )
    return {
        "passed": relative_error <= DENSITY_RELATIVE_TOLERANCE and stationary,
        "density_accuracy_passed": relative_error <= DENSITY_RELATIVE_TOLERANCE,
        "density_stationarity_passed": stationary,
        "mean_density_g_cm3": mean_g_cm3,
        "target_density_g_cm3": TARGET_DCM_DENSITY_G_CM3,
        "relative_error": relative_error,
        "relative_tolerance": DENSITY_RELATIVE_TOLERANCE,
        # Stable text/JSON output avoids exposing binary summation noise such
        # as 1.2899999999999816 for an exactly constant 1.29 g cm^-3 input.
        "block_means_g_cm3": [round(value, 12) for value in block_means],
        "block_sem_g_cm3": block_sem,
        "block_range_relative": block_range_relative,
        "stationarity_tolerance": DENSITY_STATIONARITY_TOLERANCE,
        "linear_slope_g_cm3_per_ns": slope_g_cm3_per_ns,
        "linear_slope_relative_per_ns": slope_relative_per_ns,
        "samples_in_final_4ns": len(selected_rows),
        "analysis_start_ps": analysis_start,
        "analysis_end_ps": maximum_time,
    }


def density_gate(edr: Path, output_dir: Path, dcm_model: str) -> dict[str, object]:
    xvg = output_dir / "density.xvg"
    run_command(
        ["gmx", "energy", "-f", str(edr), "-o", str(xvg), "-xvg", "none"],
        output_dir,
        output_dir / "run_commands.log",
        stdin="Density\n0\n",
    )
    rows = parse_xvg(xvg)
    result = {"dcm_model": dcm_model, **summarize_density_rows(rows)}
    (output_dir / "density_gate.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_solvent_validation(root: Path, ntomp: int, dcm_model: str) -> dict[str, object]:
    setup = root / "gromacs_dcm" / "solvent_validation" / dcm_model / "setup"
    ensure_em_tpr(setup, root / "gromacs_dcm" / "mdp")
    run_dir = root / "gromacs_dcm" / "solvent_validation" / dcm_model / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    mdp_dir = root / "gromacs_dcm" / "mdp"
    npt = run_base_equilibration(setup, run_dir, mdp_dir, f"pure_dcm|{dcm_model}", ntomp)
    production = run_dir / "solvent_validation_5ns"
    if not production.with_suffix(".tpr").is_file():
        grompp(
            mdp=mdp_dir / "solvent_validation_5ns.mdp",
            coordinates=npt.with_suffix(".gro"),
            topology=setup / "system.top",
            output_tpr=production.with_suffix(".tpr"),
            checkpoint=npt.with_suffix(".cpt"),
            log=run_dir / "run_commands.log",
        )
    mdrun(production, ntomp)
    return density_gate(production.with_suffix(".edr"), run_dir, dcm_model)


def load_passed_density_gate(root: Path, dcm_model: str) -> dict[str, object]:
    path = root / "gromacs_dcm" / "solvent_validation" / dcm_model / "run" / "density_gate.json"
    if not path.is_file():
        raise RuntimeError("pure DCM density gate has not been run")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("dcm_model") != dcm_model:
        raise RuntimeError(f"density gate model mismatch: {result}")
    if not result.get("passed"):
        raise RuntimeError(f"pure DCM density gate failed: {result}")
    return result


def replica_initial_setups(setup: Path, replicas: int) -> list[Path] | None:
    """Resolve independently packed replica setups when the bundle provides them."""
    directory = setup / "initial_conformers"
    if not directory.is_dir():
        return None
    available = sorted(path for path in directory.glob("replica_*" ) if path.is_dir())
    expected_names = [f"replica_{index:02d}" for index in range(1, len(available) + 1)]
    if [path.name for path in available] != expected_names:
        raise ValueError(
            f"initial conformer setups must be consecutive replica_01..replica_N: "
            f"{[path.name for path in available]}"
        )
    if replicas > len(available):
        raise ValueError(
            f"requested {replicas} replicas but only {len(available)} independent "
            f"initial conformer setups are available in {directory}"
        )
    selected = available[:replicas]
    for path in selected:
        for name in ("system.gro", "system.top", "metadata.json"):
            required = path / name
            if not required.is_file():
                raise FileNotFoundError(
                    f"replica initial conformer setup is incomplete: {required}"
                )
    return selected


def run_state(
    root: Path,
    row: dict[str, str],
    replicas: int,
    ntomp: int,
    dcm_model: str,
    replica_indices: Sequence[int] | None = None,
) -> None:
    state = row["state"]
    state_dir = root / row["output_dir"]
    setup = state_dir / "setup"
    mdp_dir = root / "gromacs_dcm" / "mdp"
    ensure_em_tpr(setup, mdp_dir)
    metadata_path = setup / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"state setup metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dcm_model") != dcm_model:
        raise RuntimeError(
            f"state {state} uses DCM model {metadata.get('dcm_model')!r}, "
            f"not requested {dcm_model!r}; rerun setup for this state"
        )
    if ensure_simulation_index(setup, int(metadata["source_atom_count"])):
        print(f"[REPAIRED] {state} index.ndx: added System + Solute groups", flush=True)
    selected_replicas = (
        list(range(1, replicas + 1))
        if replica_indices is None
        else list(replica_indices)
    )
    independent_setups = replica_initial_setups(setup, max(selected_replicas))
    shared_base_npt = None
    if independent_setups is None:
        shared_base_npt = run_base_equilibration(
            setup, state_dir, mdp_dir, state, ntomp
        )
    else:
        print(
            f"[INDEPENDENT] {state}: using {len(independent_setups)} "
            "replica-specific packed conformers",
            flush=True,
        )
    for replica in selected_replicas:
        replica_dir = state_dir / f"replica_{replica:02d}"
        replica_dir.mkdir(parents=True, exist_ok=True)
        replica_setup = (
            setup if independent_setups is None else independent_setups[replica - 1]
        )
        if independent_setups is None:
            assert shared_base_npt is not None
            base_npt = shared_base_npt
        else:
            replica_metadata = json.loads(
                (replica_setup / "metadata.json").read_text(encoding="utf-8")
            )
            if replica_metadata.get("dcm_model") != dcm_model:
                raise RuntimeError(
                    f"state {state} replica {replica:02d} uses DCM model "
                    f"{replica_metadata.get('dcm_model')!r}, not {dcm_model!r}"
                )
            solute_atoms = int(replica_metadata["source_atom_count"])
            ensure_em_tpr(replica_setup, mdp_dir)
            if ensure_simulation_index(replica_setup, solute_atoms):
                print(
                    f"[REPAIRED] {state} replica {replica:02d} index.ndx",
                    flush=True,
                )
            base_npt = run_base_equilibration(
                replica_setup,
                replica_dir,
                mdp_dir,
                f"{state}|initial_conformer|replica_{replica:02d}",
                ntomp,
            )
        log = replica_dir / "run_commands.log"
        reseed_mdp = replica_dir / "reseed_nvt_seeded.mdp"
        render_seeded_mdp(
            mdp_dir / "reseed_nvt.mdp",
            reseed_mdp,
            stable_seed(f"{state}|replica|{replica}"),
        )
        reseed = replica_dir / "reseed_nvt"
        if not reseed.with_suffix(".tpr").is_file():
            grompp(
                mdp=reseed_mdp,
                coordinates=base_npt.with_suffix(".gro"),
                topology=replica_setup / "system.top",
                output_tpr=reseed.with_suffix(".tpr"),
                log=log,
            )
        mdrun(reseed, ntomp)
        production = replica_dir / "production_2ns"
        if not production.with_suffix(".tpr").is_file():
            grompp(
                mdp=mdp_dir / "production_2ns.mdp",
                coordinates=reseed.with_suffix(".gro"),
                topology=replica_setup / "system.top",
                output_tpr=production.with_suffix(".tpr"),
                checkpoint=reseed.with_suffix(".cpt"),
                index=replica_setup / "index.ndx",
                log=log,
            )
        mdrun(production, ntomp)
        print(f"[OK] {state} replica {replica:02d}", flush=True)


def select_rows(
    rows: Sequence[dict[str, str]], states: Sequence[str], stages: Sequence[str]
) -> list[dict[str, str]]:
    selected = [row for row in rows if row["stage"] in set(stages)]
    if states != ["all"]:
        requested = set(states)
        unknown = requested - {row["state"] for row in rows}
        if unknown:
            raise ValueError(f"unknown states: {sorted(unknown)}")
        selected = [row for row in selected if row["state"] in requested]
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--solvent-only", action="store_true")
    parser.add_argument("--states", nargs="+", default=["all"])
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=SUPPORTED_REACTION_STAGES,
        default=list(SUPPORTED_REACTION_STAGES),
    )
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument(
        "--replica-indices",
        nargs="+",
        type=int,
        help="run only these replica numbers (within 1..--replicas)",
    )
    parser.add_argument("--ntomp", type=int, default=8)
    parser.add_argument("--dcm-model", choices=DCM_MODELS, default=DEFAULT_DCM_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.replicas < 1 or args.ntomp < 1:
        raise SystemExit("replicas and ntomp must be positive")
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
    root = Path(args.root).resolve()
    rows = read_manifest(root / "gromacs_dcm" / "state_manifest.csv")
    selected = select_rows(rows, args.states, args.stages)
    print(
        f"Selected {len(selected)} states; replica indices {replica_indices}; "
        f"dcm_model={args.dcm_model}; GPU mdrun uses 1 rank x {args.ntomp} OpenMP threads."
    )
    if args.dry_run:
        print(f"gmx={shutil.which('gmx') or 'MISSING'}")
        for row in selected:
            print(f"  {row['state']} ({row['stage']})")
        return
    if shutil.which("gmx") is None:
        raise RuntimeError("gmx is not on PATH")
    if args.solvent_only:
        result = run_solvent_validation(root, args.ntomp, args.dcm_model)
        print(f"Pure DCM density gate: {result}")
        if not result["passed"]:
            raise SystemExit(3)
        return
    gate = load_passed_density_gate(root, args.dcm_model)
    print(f"Pure DCM density gate passed: {gate}")
    for index, row in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] Starting {row['state']}", flush=True)
        run_state(
            root,
            row,
            args.replicas,
            args.ntomp,
            args.dcm_model,
            replica_indices,
        )
    print("All selected explicit-DCM pilot trajectories completed.")


if __name__ == "__main__":
    main()
