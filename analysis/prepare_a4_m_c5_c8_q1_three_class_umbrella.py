#!/usr/bin/env python3
"""Prepare C5/C8 q1 full-reuse, partial-reuse, and full-switch umbrella paths."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from analysis import prepare_a4_q2_umbrella_pilot as protocol
from analysis.prepare_a4_m_c6_c7_missing_umbrella import package_job


PATHWAYS = (
    {"pathway": "c5_q1_full_reuse_v1_v6_rep1", "system": "A4_m_C5", "state": "A4_m_C5_1_cis",
     "route_family": "full_reuse", "history_class": "reuse_arm_pair", "left": 1, "right": 6, "replica": 1,
     "selection_reason": "unique active pair that reuses both q0-reacted arms"},
    {"pathway": "c5_q1_full_reuse_v1_v6_rep2", "system": "A4_m_C5", "state": "A4_m_C5_1_cis",
     "route_family": "full_reuse", "history_class": "reuse_arm_pair", "left": 1, "right": 6, "replica": 2,
     "selection_reason": "unique active pair that reuses both q0-reacted arms"},
    {"pathway": "c5_q1_partial_reuse_v1_v5_rep1", "system": "A4_m_C5", "state": "A4_m_C5_1_cis",
     "route_family": "partial_reuse", "history_class": "one_reacted_arm", "left": 1, "right": 5, "replica": 1,
     "selection_reason": "most reproducibly sampled C5 partial-reuse pair across all three 10 ns replicas"},
    {"pathway": "c5_q1_partial_reuse_v1_v5_rep2", "system": "A4_m_C5", "state": "A4_m_C5_1_cis",
     "route_family": "partial_reuse", "history_class": "one_reacted_arm", "left": 1, "right": 5, "replica": 2,
     "selection_reason": "same pair as replica 1; independent endpoint"},
    {"pathway": "c5_q1_full_switch_v2_v4_rep1", "system": "A4_m_C5", "state": "A4_m_C5_1_cis",
     "route_family": "full_switch", "history_class": "untouched_arm_pair", "left": 2, "right": 4, "replica": 1,
     "selection_reason": "high and reproducibly sampled C5 full-switch pair across all three 10 ns replicas"},
    {"pathway": "c5_q1_full_switch_v2_v4_rep2", "system": "A4_m_C5", "state": "A4_m_C5_1_cis",
     "route_family": "full_switch", "history_class": "untouched_arm_pair", "left": 2, "right": 4, "replica": 2,
     "selection_reason": "same pair as replica 1; independent endpoint"},
    {"pathway": "c8_q1_full_reuse_v5_v7_rep1", "system": "A4_m_C8", "state": "A4_m_C8_1_cis",
     "route_family": "full_reuse", "history_class": "reuse_arm_pair", "left": 5, "right": 7, "replica": 1,
     "selection_reason": "unique active pair that reuses both q0-reacted arms"},
    {"pathway": "c8_q1_full_reuse_v5_v7_rep2", "system": "A4_m_C8", "state": "A4_m_C8_1_cis",
     "route_family": "full_reuse", "history_class": "reuse_arm_pair", "left": 5, "right": 7, "replica": 2,
     "selection_reason": "unique active pair that reuses both q0-reacted arms"},
    {"pathway": "c8_q1_partial_reuse_v1_v5_rep1", "system": "A4_m_C8", "state": "A4_m_C8_1_cis",
     "route_family": "partial_reuse", "history_class": "one_reacted_arm", "left": 1, "right": 5, "replica": 1,
     "selection_reason": "highest pooled C8 partial-reuse soft-NAC across the three 10 ns replicas"},
    {"pathway": "c8_q1_partial_reuse_v1_v5_rep2", "system": "A4_m_C8", "state": "A4_m_C8_1_cis",
     "route_family": "partial_reuse", "history_class": "one_reacted_arm", "left": 1, "right": 5, "replica": 2,
     "selection_reason": "same pair as replica 1; independent endpoint"},
    {"pathway": "c8_q1_full_switch_v1_v3_rep1", "system": "A4_m_C8", "state": "A4_m_C8_1_cis",
     "route_family": "full_switch", "history_class": "untouched_arm_pair", "left": 1, "right": 3, "replica": 1,
     "selection_reason": "highest pooled and consistently sampled C8 full-switch pair"},
    {"pathway": "c8_q1_full_switch_v1_v3_rep2", "system": "A4_m_C8", "state": "A4_m_C8_1_cis",
     "route_family": "full_switch", "history_class": "untouched_arm_pair", "left": 1, "right": 3, "replica": 2,
     "selection_reason": "same pair as replica 1; independent endpoint"},
)

MAXIMUM_NM = 2.75


def metadata_mapping(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(key): [int(value) for value in values]
        for key, values in payload["vinyl_mapping_gromacs_1based"].items()
    }


def find_replica(download_root: Path, state: str, replica: int) -> Path:
    matches = sorted(download_root.glob(
        f"*/extracted/workspace/gromacs_dcm/runs/{state}/replica_{replica:02d}"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one downloaded endpoint for {state} replica {replica}, found {matches}"
        )
    return matches[0]


def prepare_pathway(
    output: Path,
    download_root: Path,
    run_root: Path,
    config: dict[str, object],
) -> dict[str, object]:
    state = str(config["state"])
    replica = int(config["replica"])
    source_replica = find_replica(download_root, state, replica)
    setup = run_root / state / "setup"
    start = source_replica / "production_2ns.gro"
    checkpoint = source_replica / "production_2ns.cpt"
    if not start.is_file() or not checkpoint.is_file():
        raise FileNotFoundError(f"10 ns endpoint is incomplete: {source_replica}")
    mapping = metadata_mapping(setup / "metadata.json")
    left, right = int(config["left"]), int(config["right"])
    initial_nm, half_box_nm = protocol.pair_distance_nm(start, mapping, left, right)
    if MAXIMUM_NM >= 0.98 * half_box_nm:
        raise ValueError(f"{config['pathway']}: maximum violates half-box guard ({half_box_nm:.3f} nm)")
    pathway = output / str(config["pathway"])
    pathway.mkdir(parents=True, exist_ok=True)
    shutil.copy2(setup / "system.top", pathway / "system.top")
    (pathway / "index.ndx").write_text(
        protocol.append_pair_groups(
            (setup / "index.ndx").read_text(encoding="utf-8"), left, right, mapping
        ),
        encoding="utf-8",
    )
    shutil.copy2(start, pathway / "start.gro")
    shutil.copy2(checkpoint, pathway / "start.cpt")
    (pathway / "steer_in.mdp").write_text(
        protocol.steering_mdp(left, right, mapping, initial_nm, protocol.MINIMUM_NM - 0.10),
        encoding="utf-8",
    )
    (pathway / "steer_out.mdp").write_text(
        protocol.steering_mdp(left, right, mapping, initial_nm, MAXIMUM_NM),
        encoding="utf-8",
    )
    mdp_root = pathway / "window_mdps"
    mdp_root.mkdir(exist_ok=True)
    windows = []
    for index, center in enumerate(protocol.centers(MAXIMUM_NM)):
        name = f"window_{index:03d}_{center:.3f}nm"
        (mdp_root / f"{name}_equil.mdp").write_text(
            protocol.umbrella_mdp(left, right, mapping, center, False), encoding="utf-8"
        )
        (mdp_root / f"{name}_production.mdp").write_text(
            protocol.umbrella_mdp(left, right, mapping, center, True), encoding="utf-8"
        )
        windows.append({"window_index": index, "window_name": name, "center_nm": f"{center:.3f}"})
    with (pathway / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("window_index", "window_name", "center_nm"))
        writer.writeheader()
        writer.writerows(windows)
    record = {
        **config,
        "stage": "q1",
        "comparison": "three_class_q1",
        "initial_distance_nm": initial_nm,
        "half_box_limit_nm": half_box_nm,
        "minimum_nm": protocol.MINIMUM_NM,
        "maximum_nm": MAXIMUM_NM,
        "window_spacing_nm": protocol.WINDOW_SPACING_NM,
        "window_count": len(windows),
        "force_constant_kJ_mol_nm2": protocol.FORCE_CONSTANT,
        "equilibration_ps_per_window": protocol.EQUILIBRATION_PS,
        "production_ps_per_window": protocol.PRODUCTION_PS,
        "wham_begin_ps": protocol.WHAM_BEGIN_PS,
        "source_endpoint": str(start),
        "start_mode": "q1 endpoint from completed 10 ns unbiased trajectory",
    }
    (pathway / "pathway.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-root", type=Path,
                        default=PROJECT / "results/a4_m_c5_c8_q1_10ns_download")
    parser.add_argument(
        "--run-root", type=Path,
        default=PROJECT / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs",
        help="validated local state setups containing system.top, index.ndx, and metadata.json",
    )
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "umbrella/a4_m_c5_c8_q1_three_class_umbrella")
    parser.add_argument("--job-root", type=Path,
                        default=PROJECT / "bohrium_submit/a4_m_c5_c8_q1_three_class_umbrella")
    args = parser.parse_args()
    download_root, run_root, output, job_root = (
        path.resolve() for path in (args.download_root, args.run_root, args.output, args.job_root)
    )
    if output.exists() or job_root.exists():
        raise FileExistsError(f"refusing to overwrite {output} or {job_root}")
    output.mkdir(parents=True)
    job_root.mkdir(parents=True)
    records = []
    for config in PATHWAYS:
        record = prepare_pathway(output, download_root, run_root, dict(config))
        package_job(job_root, output, record)
        records.append(record)
    manifest = {
        "format": "a4-m-c5-c8-q1-three-class-umbrella-v1",
        "project_id": 4377225,
        "dispatch": "one NVIDIA T4 job per independent q1 endpoint",
        "pathway_count": len(records),
        "paths_per_system": {"full_reuse": 2, "partial_reuse": 2, "full_switch": 2},
        "total_window_sampling_ns": sum(
            int(row["window_count"])
            * (float(row["equilibration_ps_per_window"]) + float(row["production_ps_per_window"]))
            / 1000.0
            for row in records
        ),
        "pathways": records,
    }
    (output / "submission_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (job_root / "jobs.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
