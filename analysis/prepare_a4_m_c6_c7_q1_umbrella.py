#!/usr/bin/env python3
"""Prepare q1 reuse/switch umbrella paths for A4-m-C6 and A4-m-C7."""

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

from analysis import prepare_a4_q2_umbrella_pilot as protocol  # noqa: E402
from analysis.prepare_a4_m_c6_c7_missing_umbrella import package_job  # noqa: E402


PATHWAYS = (
    {
        "pathway": "c6_q1_reuse_v1_v7",
        "system": "A4_m_C6",
        "state": "A4_m_C6_1_cis",
        "comparison": "reuse_minus_switch",
        "route_family": "reuse",
        "topology": "cross_arm",
        "left": 1,
        "right": 7,
        "replica": 1,
        "bohr_job_id_source": 20725517,
    },
    {
        "pathway": "c6_q1_switch_v3_v4",
        "system": "A4_m_C6",
        "state": "A4_m_C6_1_cis",
        "comparison": "reuse_minus_switch",
        "route_family": "switch",
        "topology": "cross_arm",
        "left": 3,
        "right": 4,
        "replica": 1,
        "bohr_job_id_source": 20725517,
    },
    {
        "pathway": "c7_q1_reuse_v2_v5",
        "system": "A4_m_C7",
        "state": "A4_m_C7_1_cis",
        "comparison": "reuse_minus_switch",
        "route_family": "reuse",
        "topology": "cross_arm",
        "left": 2,
        "right": 5,
        "replica": 2,
        "bohr_job_id_source": 20725520,
    },
    {
        "pathway": "c7_q1_switch_v1_v8",
        "system": "A4_m_C7",
        "state": "A4_m_C7_1_cis",
        "comparison": "reuse_minus_switch",
        "route_family": "switch",
        "topology": "cross_arm",
        "left": 1,
        "right": 8,
        "replica": 2,
        "bohr_job_id_source": 20725520,
    },
)

MAXIMUM_NM = 2.75


def metadata_mapping(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(key): [int(value) for value in values]
        for key, values in payload["vinyl_mapping_gromacs_1based"].items()
    }


def find_downloaded_replica(download_root: Path, state: str, replica: int) -> Path:
    matches = sorted(download_root.glob(
        f"*/extracted/workspace/gromacs_dcm/runs/{state}/replica_{replica:02d}"
    ))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one downloaded endpoint for {state} replica {replica}, found {matches}")
    return matches[0]


def prepare_pathway(output: Path, run_root: Path, download_root: Path, config: dict[str, object]) -> dict[str, object]:
    state = str(config["state"])
    replica = int(config["replica"])
    setup = run_root / state / "setup"
    source_replica = find_downloaded_replica(download_root, state, replica)
    start = source_replica / "production_2ns.gro"
    checkpoint = source_replica / "production_2ns.cpt"
    mapping = metadata_mapping(setup / "metadata.json")
    left, right = int(config["left"]), int(config["right"])
    initial_nm, half_box_nm = protocol.pair_distance_nm(start, mapping, left, right)
    if MAXIMUM_NM >= 0.98 * half_box_nm:
        raise ValueError(f"{config['pathway']}: maximum violates half-box guard ({half_box_nm:.3f} nm)")
    pathway = output / str(config["pathway"])
    pathway.mkdir(parents=True, exist_ok=True)
    shutil.copy2(setup / "system.top", pathway / "system.top")
    # The umbrella MDP pulls the two custom midpoint groups.  Append them to
    # the setup index; copying the setup index alone makes grompp fail before
    # any sampling starts.
    (pathway / "index.ndx").write_text(
        protocol.append_pair_groups(
            (setup / "index.ndx").read_text(encoding="utf-8"), left, right, mapping
        ),
        encoding="utf-8",
    )
    shutil.copy2(start, pathway / "start.gro")
    if checkpoint.is_file():
        shutil.copy2(checkpoint, pathway / "start.cpt")
    (pathway / "steer_in.mdp").write_text(
        protocol.steering_mdp(left, right, mapping, initial_nm, protocol.MINIMUM_NM - 0.10),
        encoding="utf-8",
    )
    (pathway / "steer_out.mdp").write_text(
        protocol.steering_mdp(left, right, mapping, initial_nm, MAXIMUM_NM),
        encoding="utf-8",
    )
    centers = protocol.centers(MAXIMUM_NM)
    mdp_root = pathway / "window_mdps"
    mdp_root.mkdir(exist_ok=True)
    windows = []
    for index, center in enumerate(centers):
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
        "initial_distance_nm": initial_nm,
        "half_box_limit_nm": half_box_nm,
        "minimum_nm": protocol.MINIMUM_NM,
        "maximum_nm": MAXIMUM_NM,
        "window_spacing_nm": protocol.WINDOW_SPACING_NM,
        "window_count": len(windows),
        "force_constant_kJ_mol_nm2": protocol.FORCE_CONSTANT,
        "steering_rate_nm_ps": protocol.PULL_RATE_NM_PS,
        "equilibration_ps_per_window": protocol.EQUILIBRATION_PS,
        "production_ps_per_window": protocol.PRODUCTION_PS,
        "wham_begin_ps": protocol.WHAM_BEGIN_PS,
        "wham_end_ps": protocol.PRODUCTION_PS,
        "source_endpoint": str(start.relative_to(PROJECT)),
        "start_mode": "full-system q1 endpoint from completed unbiased 10 ns batch",
    }
    (pathway / "pathway.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=PROJECT / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs")
    parser.add_argument("--download-root", type=Path, default=PROJECT / "results/a4_m_c6_c7_q1_10ns_download")
    parser.add_argument("--output", type=Path, default=PROJECT / "umbrella/a4_m_c6_c7_q1_umbrella")
    parser.add_argument("--job-root", type=Path, default=PROJECT / "bohrium_submit/a4_m_c6_c7_q1_umbrella")
    args = parser.parse_args()
    output, job_root = args.output.resolve(), args.job_root.resolve()
    if output.exists() or job_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output} or {job_root}")
    output.mkdir(parents=True)
    job_root.mkdir(parents=True)
    records = [prepare_pathway(output, args.run_root.resolve(), args.download_root.resolve(), dict(config)) for config in PATHWAYS]
    for record in records:
        package_job(job_root, output, record)
    manifest = {
        "format": "a4-m-c6-c7-q1-umbrella-v2",
        "project_id": 4377225,
        "dispatch": "one NVIDIA T4 job per q1 representative pathway",
        "model_scope": "all active same-arm and cross-arm pairs; A4 tetrahedral symmetry gives no opposite-arm exclusion",
        "pathway_count": len(records),
        "total_window_sampling_ns": sum(
            int(row["window_count"]) * (float(row["equilibration_ps_per_window"]) + float(row["production_ps_per_window"])) / 1000.0
            for row in records
        ),
        "pathways": records,
    }
    (output / "submission_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (job_root / "jobs.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
