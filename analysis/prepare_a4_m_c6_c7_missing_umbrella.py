#!/usr/bin/env python3
"""Prepare the missing A4-m-C6/C7 q0 and q2 umbrella evidence.

The design intentionally mirrors the completed A4-m-C5/C8 workflow:

* q0: two anchor-matched same-arm/cross-arm comparisons per molecule;
* q2 adjacent: one anchor-matched same-arm/cross-arm comparison per molecule;
* q2 opposite: one reuse/switch comparison per molecule.

Each pathway is packaged as one independent Bohrium T4 job.  Pair and replica
choices are frozen below from the completed unbiased-pilot prescreen, rather
than selected after seeing an umbrella result.
"""

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

from analysis import prepare_a4_m_c8_umbrella_pilot as q0_protocol  # noqa: E402
from analysis import prepare_a4_q2_umbrella_pilot as q2_protocol  # noqa: E402


Q0_MAXIMUM_NM = 2.600
Q2_MAXIMUM_NM = 2.650

Q0_PATHWAYS = (
    {
        "pathway": "c6_q0_anchor_v3_same_v3_v5", "system": "A4_m_C6",
        "state": "A4_m_C6_q0", "topology": "same_arm", "anchor": "V3",
        "left": 3, "right": 5, "replica": 4,
        "selection_reason": "highest q0 same-arm soft-contact pair; closest suitable endpoint",
    },
    {
        "pathway": "c6_q0_anchor_v3_cross_v3_v8", "system": "A4_m_C6",
        "state": "A4_m_C6_q0", "topology": "cross_arm", "anchor": "V3",
        "left": 3, "right": 8, "replica": 1,
        "selection_reason": "highest q0 cross-arm soft contact sharing anchor V3",
    },
    {
        "pathway": "c6_q0_anchor_v8_same_v7_v8", "system": "A4_m_C6",
        "state": "A4_m_C6_q0", "topology": "same_arm", "anchor": "V8",
        "left": 7, "right": 8, "replica": 1,
        "selection_reason": "second independent same-arm anchor; closest suitable endpoint",
    },
    {
        "pathway": "c6_q0_anchor_v8_cross_v1_v8", "system": "A4_m_C6",
        "state": "A4_m_C6_q0", "topology": "cross_arm", "anchor": "V8",
        "left": 1, "right": 8, "replica": 2,
        "selection_reason": "high-contact cross-arm pair sharing anchor V8; closest endpoint",
    },
    {
        "pathway": "c7_q0_anchor_v5_same_v4_v5", "system": "A4_m_C7",
        "state": "A4_m_C7_q0", "topology": "same_arm", "anchor": "V5",
        "left": 4, "right": 5, "replica": 3,
        "selection_reason": "highest q0 same-arm soft-contact pair; closest endpoint",
    },
    {
        "pathway": "c7_q0_anchor_v5_cross_v2_v5", "system": "A4_m_C7",
        "state": "A4_m_C7_q0", "topology": "cross_arm", "anchor": "V5",
        "left": 2, "right": 5, "replica": 2,
        "selection_reason": "highest q0 cross-arm soft contact sharing anchor V5",
    },
    {
        "pathway": "c7_q0_anchor_v2_same_v2_v3", "system": "A4_m_C7",
        "state": "A4_m_C7_q0", "topology": "same_arm", "anchor": "V2",
        "left": 2, "right": 3, "replica": 2,
        "selection_reason": "second independent same-arm anchor; closest endpoint",
    },
    {
        "pathway": "c7_q0_anchor_v2_cross_v2_v4", "system": "A4_m_C7",
        "state": "A4_m_C7_q0", "topology": "cross_arm", "anchor": "V2",
        "left": 2, "right": 4, "replica": 1,
        "selection_reason": "high-contact cross-arm pair sharing anchor V2; closest endpoint",
    },
)

Q2_PATHWAYS = (
    {
        "pathway": "c6_q2_adjacent_same_v4_v6", "system": "A4_m_C6",
        "state": "A4_m_C6_2_adjacent_cis", "background": "adjacent",
        "comparison": "cross_minus_same", "route_family": "same_arm",
        "topology": "same_arm", "anchor": "V4", "left": 4, "right": 6, "replica": 2,
        "selection_reason": "best jointly sampled anchor-matched replica for adjacent same/cross",
    },
    {
        "pathway": "c6_q2_adjacent_cross_v4_v7", "system": "A4_m_C6",
        "state": "A4_m_C6_2_adjacent_cis", "background": "adjacent",
        "comparison": "cross_minus_same", "route_family": "switch",
        "topology": "cross_arm", "anchor": "V4", "left": 4, "right": 7, "replica": 2,
        "selection_reason": "same replica and anchor as C6 adjacent same-arm path",
    },
    {
        "pathway": "c7_q2_adjacent_same_v7_v8", "system": "A4_m_C7",
        "state": "A4_m_C7_2_adjacent_cis", "background": "adjacent",
        "comparison": "cross_minus_same", "route_family": "same_arm",
        "topology": "same_arm", "anchor": "V8", "left": 7, "right": 8, "replica": 3,
        "selection_reason": "best jointly sampled anchor-matched replica for adjacent same/cross",
    },
    {
        "pathway": "c7_q2_adjacent_cross_v2_v8", "system": "A4_m_C7",
        "state": "A4_m_C7_2_adjacent_cis", "background": "adjacent",
        "comparison": "cross_minus_same", "route_family": "switch",
        "topology": "cross_arm", "anchor": "V8", "left": 2, "right": 8, "replica": 3,
        "maximum_nm": 2.550,
        "selection_reason": "same replica and anchor as C7 adjacent same-arm path",
    },
    {
        "pathway": "c6_q2_opposite_reuse_v3_v4", "system": "A4_m_C6",
        "state": "A4_m_C6_2_opposite_cis", "background": "opposite",
        "comparison": "reuse_minus_switch", "route_family": "reuse",
        "topology": "cross_arm", "left": 3, "right": 4, "replica": 2,
        "selection_reason": "highest nominal reuse NAC in the C6 opposite pilot",
    },
    {
        "pathway": "c6_q2_opposite_switch_v3_v7", "system": "A4_m_C6",
        "state": "A4_m_C6_2_opposite_cis", "background": "opposite",
        "comparison": "reuse_minus_switch", "route_family": "switch",
        "topology": "cross_arm", "left": 3, "right": 7, "replica": 2,
        "selection_reason": "highest nominal switch NAC in the same C6 replica",
    },
    {
        "pathway": "c7_q2_opposite_reuse_v2_v5", "system": "A4_m_C7",
        "state": "A4_m_C7_2_opposite_cis", "background": "opposite",
        "comparison": "reuse_minus_switch", "route_family": "reuse",
        "topology": "cross_arm", "left": 2, "right": 5, "replica": 2,
        # This route twice crossed GROMACS' 0.49*box pull-distance guard in
        # the terminal window under NPT (first at 2.650 nm, then 2.550 nm).
        # Stop at the last common, safely sampled center instead.
        "maximum_nm": 2.450,
        "selection_reason": "reuse had zero pilot NAC; this pair/replica had the lowest mean distance",
    },
    {
        "pathway": "c7_q2_opposite_switch_v2_v6", "system": "A4_m_C7",
        "state": "A4_m_C7_2_opposite_cis", "background": "opposite",
        "comparison": "reuse_minus_switch", "route_family": "switch",
        "topology": "cross_arm", "left": 2, "right": 6, "replica": 3,
        "maximum_nm": 2.550,
        "selection_reason": "highest nominal switch NAC in the C7 opposite pilot",
    },
)


def metadata_mapping(path: Path) -> dict[str, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(key): [int(value) for value in values]
        for key, values in payload["vinyl_mapping_gromacs_1based"].items()
    }


def q0_initial_distance(csv_path: Path, left: int, right: int, topology: str) -> float:
    suffix = "同臂" if topology == "same_arm" else "跨臂"
    column = f"V{left}_V{right}_{suffix}"
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty q0 distance file: {csv_path}")
    return float(rows[-1][column]) / 10.0


def write_windows(pathway: Path, centers: list[float], writer) -> int:
    mdp_root = pathway / "window_mdps"
    mdp_root.mkdir(exist_ok=True)
    rows = []
    for index, center in enumerate(centers):
        name = f"window_{index:03d}_{center:.3f}nm"
        equil, production = writer(center)
        (mdp_root / f"{name}_equil.mdp").write_text(equil, encoding="utf-8")
        (mdp_root / f"{name}_production.mdp").write_text(production, encoding="utf-8")
        rows.append({"window_index": index, "window_name": name, "center_nm": f"{center:.3f}"})
    with (pathway / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=("window_index", "window_name", "center_nm"))
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    return len(rows)


def prepare_q0(output: Path, run_root: Path, config: dict[str, object]) -> dict[str, object]:
    state = str(config["state"])
    source = run_root / state
    setup = source / "setup"
    replica = int(config["replica"])
    replica_root = source / f"replica_{replica:02d}"
    left, right = int(config["left"]), int(config["right"])
    mapping = metadata_mapping(setup / "metadata.json")
    pathway = output / str(config["pathway"])
    pathway.mkdir(parents=True, exist_ok=True)
    start = replica_root / "production_2ns.gro"
    initial_nm, half_box_nm = q2_protocol.pair_distance_nm(start, mapping, left, right)
    csv_initial_nm = q0_initial_distance(
        replica_root / "vinyl_distances.csv", left, right, str(config["topology"])
    )
    if abs(initial_nm - csv_initial_nm) > 0.01:
        raise ValueError(f"{pathway.name}: endpoint distance mismatch")
    if Q0_MAXIMUM_NM >= 0.98 * half_box_nm:
        raise ValueError(f"{pathway.name}: q0 maximum violates half-box guard")
    shutil.copy2(setup / "system.top", pathway / "system.top")
    shutil.copy2(start, pathway / "start.gro")
    checkpoint = replica_root / "production_2ns.cpt"
    if checkpoint.is_file():
        shutil.copy2(checkpoint, pathway / "start.cpt")
    (pathway / "index.ndx").write_text(
        q0_protocol.append_pair_groups(
            (setup / "index.ndx").read_text(encoding="utf-8"), left, right, mapping
        ),
        encoding="utf-8",
    )
    (pathway / "steer_in.mdp").write_text(
        q0_protocol.steering_mdp(left, right, mapping, initial_nm, q0_protocol.MINIMUM_NM - 0.10),
        encoding="utf-8",
    )
    (pathway / "steer_out.mdp").write_text(
        q0_protocol.steering_mdp(left, right, mapping, initial_nm, Q0_MAXIMUM_NM + 0.10),
        encoding="utf-8",
    )
    count = write_windows(
        pathway,
        q0_protocol.window_centers(Q0_MAXIMUM_NM),
        lambda center: (
            q0_protocol.umbrella_mdp(left, right, mapping, center, production=False),
            q0_protocol.umbrella_mdp(left, right, mapping, center, production=True),
        ),
    )
    record = {
        **config,
        "stage": "q0",
        "initial_distance_nm": initial_nm,
        "half_box_limit_nm": half_box_nm,
        "minimum_nm": q0_protocol.MINIMUM_NM,
        "maximum_nm": Q0_MAXIMUM_NM,
        "window_spacing_nm": q0_protocol.WINDOW_SPACING_NM,
        "window_count": count,
        "force_constant_kJ_mol_nm2": q0_protocol.FORCE_CONSTANT,
        "equilibration_ps_per_window": 200.0,
        "production_ps_per_window": 2000.0,
        "wham_begin_ps": 400.0,
        "wham_end_ps": 2000.0,
        "source_endpoint": str(start.relative_to(PROJECT)),
    }
    (pathway / "pathway.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def prepare_q2(output: Path, run_root: Path, endpoint_root: Path, config: dict[str, object]) -> dict[str, object]:
    state = str(config["state"])
    replica = int(config["replica"])
    setup = run_root / state / "setup"
    endpoint = endpoint_root / state / f"replica_{replica:02d}"
    left, right = int(config["left"]), int(config["right"])
    mapping = metadata_mapping(setup / "metadata.json")
    pathway = output / str(config["pathway"])
    pathway.mkdir(parents=True, exist_ok=True)
    start = endpoint / "production_2ns.gro"
    initial_nm, half_box_nm = q2_protocol.pair_distance_nm(start, mapping, left, right)
    maximum_nm = float(config.get("maximum_nm", Q2_MAXIMUM_NM))
    if maximum_nm >= 0.98 * half_box_nm:
        raise ValueError(f"{pathway.name}: q2 maximum violates half-box guard")
    shutil.copy2(setup / "system.top", pathway / "system.top")
    shutil.copy2(start, pathway / "start.gro")
    (pathway / "index.ndx").write_text(
        q2_protocol.append_pair_groups(
            (setup / "index.ndx").read_text(encoding="utf-8"), left, right, mapping
        ),
        encoding="utf-8",
    )
    (pathway / "steer_in.mdp").write_text(
        q2_protocol.steering_mdp(left, right, mapping, initial_nm, q2_protocol.MINIMUM_NM - 0.10),
        encoding="utf-8",
    )
    (pathway / "steer_out.mdp").write_text(
        q2_protocol.steering_mdp(left, right, mapping, initial_nm, maximum_nm),
        encoding="utf-8",
    )
    count = write_windows(
        pathway,
        q2_protocol.centers(maximum_nm),
        lambda center: (
            q2_protocol.umbrella_mdp(left, right, mapping, center, False),
            q2_protocol.umbrella_mdp(left, right, mapping, center, True),
        ),
    )
    record = {
        **config,
        "stage": "q2",
        "initial_distance_nm": initial_nm,
        "half_box_limit_nm": half_box_nm,
        "minimum_nm": q2_protocol.MINIMUM_NM,
        "maximum_nm": maximum_nm,
        "window_spacing_nm": q2_protocol.WINDOW_SPACING_NM,
        "window_count": count,
        "force_constant_kJ_mol_nm2": q2_protocol.FORCE_CONSTANT,
        "equilibration_ps_per_window": q2_protocol.EQUILIBRATION_PS,
        "production_ps_per_window": q2_protocol.PRODUCTION_PS,
        "wham_begin_ps": q2_protocol.WHAM_BEGIN_PS,
        "wham_end_ps": q2_protocol.PRODUCTION_PS,
        "source_endpoint": str(start.relative_to(PROJECT)),
    }
    (pathway / "pathway.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return record


def package_job(job_root: Path, umbrella_root: Path, record: dict[str, object]) -> None:
    name = str(record["pathway"])
    task = job_root / name
    task.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT / "analysis/run_a4_m_c8_umbrella_path.py", task / "run_umbrella_path.py")
    source = umbrella_root / name
    destination = task / "pathway"
    destination.mkdir(exist_ok=True)
    for filename in (
        "pathway.json", "windows.csv", "system.top", "index.ndx", "start.gro",
        "steer_in.mdp", "steer_out.mdp",
    ):
        shutil.copy2(source / filename, destination / filename)
    if (source / "start.cpt").is_file():
        shutil.copy2(source / "start.cpt", destination / "start.cpt")
    shutil.copytree(source / "window_mdps", destination / "window_mdps", dirs_exist_ok=True)
    archive = f"{name}_results.tar.gz"
    shell = f"""#!/usr/bin/env bash
set -euo pipefail
export GMX_MAXBACKUP=-1
mkdir -p results
python3 -u run_umbrella_path.py --pathway pathway --ntomp 4 --gpu --stage all 2>&1 | tee results/umbrella_job.log
tar -czf results/{archive} pathway
"""
    runner = task / "run_umbrella.sh"
    runner.write_text(shell, encoding="utf-8")
    runner.chmod(0o755)
    job = {
        "job_name": f"A4-m umbrella {name}",
        "command": "bash run_umbrella.sh",
        "log_file": "results/umbrella_job.log",
        "backward_files": [f"results/{archive}", "results/umbrella_job.log"],
        "project_id": 4377225,
        "machine_type": "c4_m15_1 * NVIDIA T4",
        "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
        "input_directory": str(task.resolve()),
        "max_reschedule_times": 2,
        "max_run_time": 4320,
        "nnode": 1,
    }
    (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root", type=Path,
        default=PROJECT / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs",
    )
    parser.add_argument(
        "--q2-endpoint-root", type=Path,
        default=PROJECT / "results/a4_m_c6_c7_pilot_collected/nac",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "umbrella/a4_m_c6_c7_missing_evidence",
    )
    parser.add_argument(
        "--job-root", type=Path,
        default=PROJECT / "bohrium_submit/a4_m_c6_c7_missing_umbrella",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    job_root = args.job_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    job_root.mkdir(parents=True, exist_ok=True)
    records = [prepare_q0(output, args.run_root.resolve(), dict(row)) for row in Q0_PATHWAYS]
    records.extend(
        prepare_q2(output, args.run_root.resolve(), args.q2_endpoint_root.resolve(), dict(row))
        for row in Q2_PATHWAYS
    )
    for record in records:
        package_job(job_root, output, record)
    total_ns = sum(
        int(row["window_count"])
        * (float(row["equilibration_ps_per_window"]) + float(row["production_ps_per_window"]))
        / 1000.0
        for row in records
    )
    manifest = {
        "format": "a4-m-c6-c7-missing-umbrella-v1",
        "project_id": 4377225,
        "dispatch": "one NVIDIA T4 job per independent pathway",
        "pathway_count": len(records),
        "total_window_sampling_ns": total_ns,
        "protocol_note": "q0 follows C5/C8 q0 timing; q2 follows C5/C8 q2 timing",
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
