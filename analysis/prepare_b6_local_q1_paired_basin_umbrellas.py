#!/usr/bin/env python3
"""Select three persistent q1 conformational strata and package paired umbrellas."""
from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from scipy.cluster.vq import kmeans2

import prepare_a4_q2_umbrella_pilot as protocol

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_JOB = 20750428
SOURCE_ROOT = PROJECT.parent / (
    f"bohrium_download/b6_neighbor_md_extensions_2to10ns_20260918/"
    f"{SOURCE_JOB}/extracted/workspace"
)
OUTPUT = PROJECT.parent / "bohrium_submit/b6_local_q1_paired_basin_umbrella_20260928"
PROJECT_ID = 4377225
STATE = "B6_p_C6_q1_V1_V2_cis"
SOURCE_REPLICAS = (1, 2, 3)
PAIR_PATHS = (
    {
        "name": "A1_to_A6",
        "pull_pair": (3, 12),  # old labels; symmetry-mapped to current V2-V11
        "current_pair": "V2-V11",
        "multiplicity": 1,
        "family": "full_reuse",
    },
    {
        "name": "A1_to_A2",
        "pull_pair": (3, 10),  # old labels; symmetry-mapped to current V2-V3/V2-V4
        "current_pair": "V2-V3 (represents V2-V4)",
        "multiplicity": 2,
        "family": "one_arm_reuse",
    },
)
MIN_DWELL_FRAMES = 40  # 20 ps at 0.5 ps/frame


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def cluster_and_select() -> tuple[list[dict[str, object]], dict[str, object]]:
    tables: list[list[dict[str, str]]] = []
    arrays: list[np.ndarray] = []
    columns: list[str] | None = None
    for replica in SOURCE_REPLICAS:
        path = SOURCE_ROOT / "gromacs_dcm/runs" / STATE / f"replica_{replica:02d}/vinyl_distances.csv"
        rows = read_csv(path)
        if not rows:
            raise ValueError(f"empty distance table: {path}")
        pair_columns = [key for key in rows[0] if key.startswith("V")]
        if columns is None:
            columns = pair_columns
        if pair_columns != columns:
            raise ValueError("vinyl distance schemas differ across replicas")
        values = np.asarray([[float(row[key]) for key in columns] for row in rows], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite pair distance found in {path}")
        arrays.append(values)
        tables.append(rows)
    assert columns is not None
    if len(columns) != 45:
        raise ValueError(f"expected 45 pairwise distances for 10 active vinyls, got {len(columns)}")

    pooled = np.vstack(arrays)
    mean = pooled.mean(axis=0)
    scale = pooled.std(axis=0)
    keep = scale > 0.15
    if int(keep.sum()) != len(columns):
        raise ValueError("unexpected low-variance distance features; inspect schema")
    z = (pooled[:, keep] - mean[keep]) / scale[keep]
    centroids, labels = kmeans2(z, 3, minit="++", iter=100, seed=10)
    offsets = np.cumsum([0] + [len(values) for values in arrays])
    occupancy: dict[str, dict[str, int]] = {}
    selections: list[dict[str, object]] = []

    for index, replica in enumerate(SOURCE_REPLICAS):
        local_labels = labels[offsets[index]:offsets[index + 1]]
        occupancy_by_cluster = {str(cluster): int(np.count_nonzero(local_labels == cluster)) for cluster in range(3)}
        occupancy[str(replica)] = occupancy_by_cluster
        local_z = z[offsets[index]:offsets[index + 1]]
        # Use each replica's full-system final GRO. The downloaded XTC is solute-only,
        # so extracting an arbitrary XTC frame would mismatch it with solvent coordinates.
        # Assign the terminal solute conformation to its nearest pooled stratum.
        terminal_frame = len(local_labels) - 1
        terminal_distance_by_cluster = np.linalg.norm(centroids - local_z[terminal_frame], axis=1)
        cluster = int(np.argmin(terminal_distance_by_cluster))
        preceding = terminal_frame
        while preceding > 0 and local_labels[preceding - 1] == cluster:
            preceding -= 1
        row = tables[index][terminal_frame]
        selections.append({
            "basin_id": index + 1,
            "cluster_id": cluster,
            "source_replica": replica,
            "frame": int(row["frame"]),
            "time_ps": float(row["absolute_time_ps"]),
            "terminal_stratum_dwell_ps": float((terminal_frame - preceding + 1) * 0.5),
            "terminal_centroid_distance_standardized": float(terminal_distance_by_cluster[cluster]),
            "A6_old_pair_distance_A": float(row["V3_V12_跨臂"]),
            "A2_old_pair_distance_A": float(row["V3_V10_跨臂"]),
        })
    metadata = {
        "method": "k-means on pooled standardized pairwise distances among 10 active vinyl groups",
        "features": columns,
        "feature_count": len(columns),
        "frames_per_replica": [len(value) for value in arrays],
        "frame_interval_ps": 0.5,
        "k": 3,
        "minimum_persistent_dwell_ps": MIN_DWELL_FRAMES * 0.5,
        "cluster_occupancy_frames_by_replica": occupancy,
        "selection_rule": "use the physically complete final full-system GRO from each independent 10 ns replica; classify its terminal solute geometry by nearest pooled pair-distance centroid",
        "limitations": "The XTC files are solute-only, so arbitrary XTC frames cannot be paired with the saved solvent box. The three full-system terminal snapshots are independent paired starts, classified into overlapping conformational strata; cluster labels are not evidence of distinct equilibrium free-energy basins.",
        "seeds": selections,
    }
    return selections, metadata


def extract_frame(seed: dict[str, object], gro: Path, gmx: str) -> None:
    replica = int(seed["source_replica"])
    source_run = SOURCE_ROOT / "gromacs_dcm/runs" / STATE
    full_system_gro = source_run / f"replica_{replica:02d}/production_2ns.gro"
    shutil.copy2(full_system_gro, gro)
    if not gro.is_file() or gro.stat().st_size < 1000:
        raise RuntimeError(f"missing full-system terminal coordinates for replica {replica}")


def write_pathway(seed: dict[str, object], pair: dict[str, object], start_gro: Path,
                  mapping: dict[str, list[int]], setup: Path, task: Path) -> dict[str, object]:
    left, right = pair["pull_pair"]
    pathway = task / "pathway"
    (pathway / "window_mdps").mkdir(parents=True)
    shutil.copy2(setup / "system.top", pathway / "system.top")
    shutil.copy2(start_gro, pathway / "start.gro")
    index_text = (setup / "index.ndx").read_text(encoding="utf-8")
    (pathway / "index.ndx").write_text(protocol.append_pair_groups(index_text, left, right, mapping), encoding="utf-8")
    initial_nm, half_box_nm = protocol.pair_distance_nm(start_gro, mapping, left, right)
    maximum_nm = 2.75
    if maximum_nm >= 0.98 * half_box_nm:
        raise ValueError(f"umbrella extent violates box limit for basin {seed['basin_id']}")
    (pathway / "steer_in.mdp").write_text(protocol.steering_mdp(
        left, right, mapping, initial_nm, protocol.MINIMUM_NM - 0.10), encoding="utf-8")
    (pathway / "steer_out.mdp").write_text(protocol.steering_mdp(
        left, right, mapping, initial_nm, maximum_nm + 0.10), encoding="utf-8")
    windows = []
    for index, center in enumerate(protocol.centers(maximum_nm)):
        name = f"window_{index:03d}_{center:.3f}nm"
        (pathway / "window_mdps" / f"{name}_equil.mdp").write_text(
            protocol.umbrella_mdp(left, right, mapping, center, production=False), encoding="utf-8")
        (pathway / "window_mdps" / f"{name}_production.mdp").write_text(
            protocol.umbrella_mdp(left, right, mapping, center, production=True), encoding="utf-8")
        windows.append({"window_index": index, "window_name": name, "center_nm": f"{center:.3f}"})
    with (pathway / "windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
        writer.writeheader(); writer.writerows(windows)
    name = f"b6_basin{int(seed['basin_id']):02d}_{pair['name']}"
    record = {
        "pathway": name, "system": "B6_p_C6", "state": STATE, "stage": "q1",
        "route_family": pair["family"], "local_branch": pair["name"],
        "pull_pair_old_labels": f"V{left}-V{right}", "current_mapping": pair["current_pair"],
        "left": left, "right": right, "topology": "cross_arm",
        "replica": int(seed["source_replica"]), "basin_id": int(seed["basin_id"]),
        "cluster_id": int(seed["cluster_id"]), "orbit_multiplicity": int(pair["multiplicity"]),
        "source_job": SOURCE_JOB, "source_frame": int(seed["frame"]),
        "source_time_ps": float(seed["time_ps"]), "source_replica": int(seed["source_replica"]),
        "source_terminal_stratum_dwell_ps": float(seed["terminal_stratum_dwell_ps"]),
        "start_mode": "full-system coordinate snapshot from 10 ns equilibrium q1 trajectory; steering initializes Maxwell velocities at 313.15 K",
        "paired_comparison_group": f"B6_q1_basin{int(seed['basin_id']):02d}",
        "initial_distance_nm": initial_nm, "half_box_limit_nm": half_box_nm,
        "minimum_nm": protocol.MINIMUM_NM, "maximum_nm": maximum_nm,
        "window_spacing_nm": protocol.WINDOW_SPACING_NM, "window_count": len(windows),
        "force_constant_kJ_mol_nm2": protocol.FORCE_CONSTANT,
        "steering_rate_nm_ps": protocol.PULL_RATE_NM_PS,
        "equilibration_ps_per_window": protocol.EQUILIBRATION_PS,
        "production_ps_per_window": protocol.PRODUCTION_PS,
        "wham_begin_ps": protocol.WHAM_BEGIN_PS, "wham_end_ps": protocol.PRODUCTION_PS,
        "analysis_plan": "paired within-stratum distance PMF and orientation-aware NAC; A1-to-A2 degeneracy applied as weight 2",
    }
    (pathway / "pathway.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(PROJECT / "analysis/run_a4_m_c8_umbrella_path.py", task / "run_umbrella_path.py")
    archive = f"{name}_results.tar.gz"
    shell = f'''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${{BASH_SOURCE[0]}}")"
mkdir -p results
export GMX_MAXBACKUP=-1
export OMP_NUM_THREADS=4
archive_results() {{
  status=$?
  tar -czf results/{archive} pathway results/umbrella_job.log 2>/dev/null || true
  exit "$status"
}}
trap archive_results EXIT
python3 -u run_umbrella_path.py --pathway pathway --ntomp 4 --gpu --stage all 2>&1 | tee results/umbrella_job.log
'''
    run = task / "run_umbrella.sh"
    run.write_text(shell, encoding="utf-8"); run.chmod(0o755)
    job = {
        "job_name": f"B6 local q1 basin {int(seed['basin_id'])} {pair['name']} paired umbrella",
        "command": "bash run_umbrella.sh", "log_file": "results/umbrella_job.log",
        "backward_files": [f"results/{archive}", "results/umbrella_job.log"],
        "project_id": PROJECT_ID, "machine_type": "c4_m15_1 * NVIDIA T4",
        "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
        "input_directory": str(task), "max_reschedule_times": 2,
        "max_run_time": 4320, "nnode": 1,
    }
    (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    for replica in SOURCE_REPLICAS:
        metadata_path = SOURCE_ROOT / "gromacs_dcm/runs" / STATE / f"replica_{replica:02d}/extension_metadata.json"
        status = json.loads(metadata_path.read_text(encoding="utf-8"))
        if status.get("status") != "complete" or float(status.get("target_total_ns", 0)) != 10.0:
            raise ValueError(f"source replica {replica} is not a completed 10 ns trajectory")
    seeds, clustering = cluster_and_select()
    staging = PROJECT / "results/b6_local_q1_arm1_competition_20260923/basin_screen"
    staging.mkdir(parents=True, exist_ok=True)
    (staging / "paired_basin_clustering.json").write_text(
        json.dumps(clustering, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (staging / "selected_basin_seeds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(seeds[0])); writer.writeheader(); writer.writerows(seeds)

    OUTPUT.mkdir(parents=True)
    setup = SOURCE_ROOT / "gromacs_dcm/runs" / STATE / "setup"
    mapping = json.loads((setup / "metadata.json").read_text(encoding="utf-8"))["vinyl_mapping_gromacs_1based"]
    gmx = shutil.which("gmx")
    if not gmx:
        raise RuntimeError("GROMACS gmx executable is required to extract snapshots")
    records = []
    for seed in seeds:
        seed_gro = OUTPUT / f"basin{int(seed['basin_id']):02d}_seed.gro"
        extract_frame(seed, seed_gro, gmx)
        for pair in PAIR_PATHS:
            task_name = f"b6_basin{int(seed['basin_id']):02d}_{pair['name']}"
            task = OUTPUT / task_name
            task.mkdir()
            records.append(write_pathway(seed, pair, seed_gro, mapping, setup, task))
    summary = {
        "status": "prepared_not_submitted", "project_id": PROJECT_ID,
        "source_job": SOURCE_JOB, "task_count": len(records),
        "paired_comparison_groups": 3,
        "paths": [{"pathway": row["pathway"], "basin_id": row["basin_id"],
                   "source_replica": row["source_replica"], "source_time_ps": row["source_time_ps"],
                   "branch": row["local_branch"], "pull_pair_old_labels": row["pull_pair_old_labels"],
                   "current_mapping": row["current_mapping"], "degeneracy": row["orbit_multiplicity"]}
                  for row in records],
        "protocol": "25 windows 0.35-2.75 nm; 100 ps equilibration + 500 ps production per window; paired paths share exact starting coordinates; A1-to-A2 multiplicity=2",
        "basin_selection": clustering,
    }
    (OUTPUT / "jobs.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "output": str(OUTPUT), "task_count": len(records),
                      "selected_seeds": seeds, "protocol": summary["protocol"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
