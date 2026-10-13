#!/usr/bin/env python3
"""Prepare, but do not submit, umbrella paths from common 2D-MetaD basin seeds."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import prepare_a4_q2_umbrella_pilot as protocol


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
TEMPLATE_ROOT = WORKSPACE / "bohrium_submit/b6_neighbor_q1_umbrella_rep13_20260918"
SEED_ROOT = PROJECT / "results/b6_q1_2d_wtmetad_analysis_20260919/common_basin_seeds"
MAPPING_SOURCE = WORKSPACE / "bohrium_download/b6_neighbor_md_extensions_2to10ns_20260918/20750428/extracted/workspace/gromacs_dcm/runs/B6_p_C6_q1_V1_V2_cis/setup/metadata.json"
DEFAULT_OUTPUT = WORKSPACE / "bohrium_submit/b6_q1_2d_reseed_umbrella_20260919"

PATHS = (
    ("b6_q1_2d_reseed_full_switch_d2_rep1", "b6_q1_full_switch_d2_v6_v8_rep1", "d2_rep1_contact_seed.gro", 6, 8, 1, 2960),
    ("b6_q1_2d_reseed_full_switch_d2_rep3", "b6_q1_full_switch_d2_v6_v8_rep3", "d2_rep3_contact_seed.gro", 6, 8, 3, 7672),
    ("b6_q1_2d_reseed_one_arm_reuse_rep1", "b6_q1_one_arm_reuse_v3_v10_rep1", "reuse_rep1_contact_seed.gro", 3, 10, 1, 2220),
    ("b6_q1_2d_reseed_one_arm_reuse_rep3", "b6_q1_one_arm_reuse_v3_v10_rep3", "reuse_rep3_contact_seed.gro", 3, 10, 3, 6058),
)


def copy_inputs(template: Path, destination: Path, seed: Path) -> None:
    source_pathway = template / "pathway"
    pathway = destination / "pathway"
    pathway.mkdir(parents=True)
    for filename in ("system.top", "index.ndx", "steer_in.mdp", "steer_out.mdp", "windows.csv"):
        shutil.copy2(source_pathway / filename, pathway / filename)
    shutil.copytree(source_pathway / "window_mdps", pathway / "window_mdps")
    shutil.copy2(seed, pathway / "start.gro")
    shutil.copy2(template / "run_umbrella_path.py", destination / "run_umbrella_path.py")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    mapping = json.loads(MAPPING_SOURCE.read_text(encoding="utf-8"))["vinyl_mapping_gromacs_1based"]
    output.mkdir(parents=True)
    records: list[dict[str, object]] = []
    for name, template_name, seed_name, left, right, replica, seed_time in PATHS:
        template = TEMPLATE_ROOT / template_name
        destination = output / name
        destination.mkdir()
        seed = SEED_ROOT / seed_name
        copy_inputs(template, destination, seed)
        pathway = destination / "pathway"
        initial_distance, half_box = protocol.pair_distance_nm(pathway / "start.gro", mapping, left, right)
        metadata = json.loads((template / "pathway/pathway.json").read_text(encoding="utf-8"))
        metadata.update({
            "pathway": name,
            "route_family": "full_switch_d2" if "full_switch" in name else "one_arm_reuse",
            "replica": replica,
            "source_endpoint": str(seed),
            "start_mode": "full-system common basin seed from targeted 2D Rg+pair-distance WT-MetaD",
            "seed_time_ps": seed_time,
            "seed_basin": "highest-common-mass contact bin shared by endpoint replicas",
            "initial_distance_nm": initial_distance,
            "half_box_limit_nm": half_box,
            "umbrella_reseed_parent": "b6_q1_2d_wtmetad_results_20260919",
            "analysis_plan": "compare reseeded umbrella PMF with original q1 umbrella; do not use MetaD occupancy as q1 probability",
        })
        (pathway / "pathway.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        run_script = (template / "run_umbrella.sh").read_text(encoding="utf-8")
        old_name = template_name
        run_script = run_script.replace(old_name, name)
        (destination / "run_umbrella.sh").write_text(run_script, encoding="utf-8")
        (destination / "run_umbrella.sh").chmod(0o755)
        job = json.loads((template / "job.json").read_text(encoding="utf-8"))
        job.update({
            "job_name": f"B6 q1 2D-MetaD reseed umbrella {name}",
            "input_directory": str(destination),
            "backward_files": [f"results/{name}_results.tar.gz", "results/umbrella_job.log"],
        })
        (destination / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        records.append({"path": name, "seed": seed.name, "pair": f"V{left}-V{right}", "replica": replica, "initial_distance_nm": initial_distance, "job": job})
    summary = {
        "status": "prepared_not_submitted",
        "project_id": 4377225,
        "paths": len(records),
        "protocol": "25 windows; 0.35-2.75 nm; 0.10 nm spacing; 100 ps equilibration + 500 ps production",
        "records": records,
    }
    (output / "jobs.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# B6 q1 2D-MetaD reseeded umbrella\n\n"
        "Prepared but not submitted. Four paths use the highest-common-mass contact seed from the two endpoint replicas. "
        "The umbrella protocol is unchanged: 25 windows from 0.35 to 2.75 nm, 100 ps equilibration and 500 ps production. "
        "MetaD occupancy is not used as a probability.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": summary["status"], "paths": summary["paths"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
