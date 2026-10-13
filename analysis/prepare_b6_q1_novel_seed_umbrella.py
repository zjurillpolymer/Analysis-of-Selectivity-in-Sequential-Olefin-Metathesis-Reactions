#!/usr/bin/env python3
"""Prepare paired B6 q1 umbrellas from persistent WT-MetaD candidate boxes.

The prepared jobs compare V3-V12 (A6/full reuse) against V3-V10
(A2/one-arm reuse) from the same full-solvent seed.  This script only creates
input directories; it does not submit or execute simulations.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import tarfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis import prepare_a4_q2_umbrella_pilot as protocol  # noqa: E402

SEEDS = (
    "A6_origin_00690ps_high_novelty_positive_pc1",
    "A6_origin_01270ps_high_novelty_crossing_frame",
    "A2_origin_05480ps_positive_pc1_return",
)
PATHS = (
    {"name": "A6_reuse_V3_V12", "left": 3, "right": 12,
     "route_family": "A6_full_reuse", "site_multiplicity": 1},
    {"name": "A2_one_arm_reuse_V3_V10", "left": 3, "right": 10,
     "route_family": "A2_one_arm_reuse", "site_multiplicity": 2},
)
MINIMUM_NM = 0.35
MAXIMUM_NM = 2.75


def make_path(seed_root: Path, output_root: Path, seed_name: str, route: dict[str, object]) -> dict[str, object]:
    prefix = "npt" if (seed_root / "npt.gro").is_file() else "npt_full"
    source_gro = seed_root / f"{prefix}.gro"
    source_top = seed_root / "system.top"
    source_index = seed_root / "index.ndx"
    for required in (source_gro, source_top, source_index):
        if not required.is_file():
            raise FileNotFoundError(required)

    left, right = int(route["left"]), int(route["right"])
    source_index_text = source_index.read_text(encoding="utf-8")
    mapping: dict[str, list[int]] = {}
    lines = source_index_text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("[ V") and line.endswith("_midpoint ]"):
            vinyl = line[2:-11]
            if index + 1 >= len(lines):
                continue
            atoms = [int(value) for value in lines[index + 1].split()]
            if len(atoms) == 2:
                mapping[vinyl] = atoms
    if f"V{left}" not in mapping or f"V{right}" not in mapping:
        raise ValueError(f"could not recover midpoint atom indices from {source_index}: {mapping}")
    initial_nm, half_box_nm = protocol.pair_distance_nm(source_gro, mapping, left, right)
    if MAXIMUM_NM >= 0.49 * 2.0 * half_box_nm:
        raise ValueError(f"{seed_name}/{route['name']}: 2.75 nm exceeds safe half-box pull limit")

    path_name = f"{seed_name}__{route['name']}"
    destination = output_root / path_name
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_gro, destination / "start.gro")
    shutil.copy2(source_top, destination / "system.top")
    shutil.copy2(source_index, destination / "index.ndx")

    # The downloaded index already contains the V3/V10/V12 midpoint groups.
    index_text = (destination / "index.ndx").read_text(encoding="utf-8")
    for vinyl in (left, right):
        if f"[ V{vinyl}_midpoint ]" not in index_text:
            raise ValueError(f"missing [ V{vinyl}_midpoint ] group in {source_index}")

    (destination / "steer_in.mdp").write_text(
        protocol.steering_mdp(left, right, mapping, initial_nm, MINIMUM_NM - 0.10), encoding="utf-8"
    )
    (destination / "steer_out.mdp").write_text(
        protocol.steering_mdp(left, right, mapping, initial_nm, MAXIMUM_NM), encoding="utf-8"
    )
    window_root = destination / "window_mdps"
    window_root.mkdir()
    windows = protocol.centers(MAXIMUM_NM)
    with (destination / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("window_index", "window_name", "center_nm"))
        writer.writeheader()
        for index, center in enumerate(windows):
            writer.writerow({
                "window_index": index,
                "window_name": f"window_{index:03d}_{center:.3f}nm",
                "center_nm": f"{center:.3f}",
            })
    for index, center in enumerate(windows):
        name = f"window_{index:03d}_{center:.3f}nm"
        (window_root / f"{name}_equil.mdp").write_text(
            protocol.umbrella_mdp(left, right, mapping, center, False), encoding="utf-8"
        )
        (window_root / f"{name}_production.mdp").write_text(
            protocol.umbrella_mdp(left, right, mapping, center, True), encoding="utf-8"
        )
    manifest = {
        "pathway": path_name,
        "stage": "q1 conditional seed-specific paired umbrella",
        "seed": seed_name,
        "source_coordinate": str(source_gro.relative_to(PROJECT)),
        "source_has_full_solvent": True,
        "route_family": route["route_family"],
        "left": left,
        "right": right,
        "left_vinyl": left,
        "right_vinyl": right,
        "site_multiplicity_for_later_route_weighting_only": route["site_multiplicity"],
        "initial_distance_nm": initial_nm,
        "half_box_limit_nm": half_box_nm,
        "minimum_nm": MINIMUM_NM,
        "maximum_nm": MAXIMUM_NM,
        "window_spacing_nm": protocol.WINDOW_SPACING_NM,
        "window_count": len(windows),
        "force_constant_kJ_mol_nm2": protocol.FORCE_CONSTANT,
        "steering_rate_nm_ps": protocol.PULL_RATE_NM_PS,
        "equilibration_ps_per_window": protocol.EQUILIBRATION_PS,
        "production_ps_per_window": protocol.PRODUCTION_PS,
        "wham_begin_ps": protocol.WHAM_BEGIN_PS,
        "interpretation": "conditional PMF within this seed family; do not treat across-seed count as basin population",
    }
    (destination / "pathway.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-root", type=Path,
        default=PROJECT / "b6_q1_candidate_equilibration_results_20261003",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "b6_q1_novel_seed_umbrella_inputs_final_20261003",
    )
    args = parser.parse_args()
    candidate_root, output = args.candidate_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output}")
    output.mkdir(parents=True)
    records = []
    for seed in SEEDS:
        seed_root = candidate_root / seed
        for route in PATHS:
            records.append(make_path(seed_root, output, seed, route))

    runner_source = PROJECT / "analysis/run_a4_m_c8_umbrella_path.py"
    shutil.copy2(runner_source, output / "run_umbrella_path.py")
    (output / "run_all_serial.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "export GMX_MAXBACKUP=-1\n"
        "for p in " + " ".join(record["pathway"] for record in records) + "; do\n"
        "  echo \"===== $p =====\"\n"
        "  python3 -u run_umbrella_path.py --pathway \"$p\" --ntomp 16 --gpu --stage all 2>&1 | tee \"${p}.log\"\n"
        "done\n",
        encoding="utf-8",
    )
    (output / "run_all_serial.sh").chmod(0o755)
    summary = {
        "scope": "B6 q1 competing channels from three persistent WT-MetaD candidate seeds",
        "submission_or_execution_performed": False,
        "seed_count": len(SEEDS),
        "pathway_count": len(records),
        "windows_per_pathway": len(protocol.centers(MAXIMUM_NM)),
        "total_windows": len(records) * len(protocol.centers(MAXIMUM_NM)),
        "total_sampling_ns": len(records) * len(protocol.centers(MAXIMUM_NM)) *
            (protocol.EQUILIBRATION_PS + protocol.PRODUCTION_PS) / 1000.0,
        "pathways": records,
    }
    (output / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README_SERVER.md").write_text(
        "# B6 q1 persistent-seed paired umbrellas\n\n"
        "This bundle contains six pathway directories: three candidate seeds, each with the paired "
        "V3–V12 (A6 full reuse) and V3–V10 (A2 one-arm reuse) pathways. Each pathway has 25 "
        "windows from 0.35 to 2.75 nm at 0.10 nm spacing; each window uses 100 ps equilibration "
        "and 500 ps production. A2 site multiplicity ×2 is recorded for later route weighting only.\n\n"
        "The start structures are full 9,219-atom boxes. Steering writes the full System to XTC so "
        "window extraction preserves solvent coordinates. The runner is the same validated resumable "
        "runner used by the prior B6/A4 umbrella workflow.\n\n"
        "## Run on the GROMACS 2026.3 Colvars/PLUMED-capable server\n\n"
        "Upload/extract this directory, activate the environment that provides the intended GROMACS, "
        "then run a single-path preflight first:\n\n"
        "```bash\n"
        "python3 -u run_umbrella_path.py --pathway A6_origin_00690ps_high_novelty_positive_pc1__A6_reuse_V3_V12 --ntomp 16 --gpu --stage preflight\n"
        "```\n\n"
        "After confirming preflight succeeds, run all six pathways serially on the one GPU:\n\n"
        "```bash\n"
        "nohup bash run_all_serial.sh > run_all_serial.log 2>&1 &\n"
        "tail -f run_all_serial.log\n"
        "```\n\n"
        "The estimated total sampling is 90 ns (36 ns window equilibration + 54 ns production), "
        "excluding steering and WHAM. This bundle only prepares inputs; it has not submitted or "
        "started any job. Preserve the six seed-specific PMFs separately; do not pool their counts "
        "as equilibrium basin populations.\n",
        encoding="utf-8",
    )
    archive = output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(output, arcname=output.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"ARCHIVE={archive}")


if __name__ == "__main__":
    main()
