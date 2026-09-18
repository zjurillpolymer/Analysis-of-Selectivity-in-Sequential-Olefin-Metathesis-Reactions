#!/usr/bin/env python3
"""Package matched B6 q1 neighbour-orbit umbrella paths from 10 ns endpoints."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import prepare_a4_q2_umbrella_pilot as protocol


STATE = "B6_p_C6_q1_V1_V2_cis"
SOURCE_JOB = 20750428
REPLICAS = (1, 3)
MAXIMUM_NM = 2.75
PROJECT_ID = 4377225


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=project.parent /
                        f"bohrium_download/b6_neighbor_md_extensions_2to10ns_20260918/{SOURCE_JOB}/extracted/workspace")
    parser.add_argument("--output", type=Path, default=project.parent /
                        "bohrium_submit/b6_neighbor_q1_umbrella_rep13_20260918")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source_run = source / "gromacs_dcm/runs" / STATE
    setup = source_run / "setup"
    mapping = json.loads((setup / "metadata.json").read_text(encoding="utf-8"))[
        "vinyl_mapping_gromacs_1based"
    ]
    orbit_manifest = json.loads((project / "results/b6_neighbor_orbit_table/manifest.json").read_text())
    representatives = orbit_manifest["representative_q1_pairs"]
    counts = orbit_manifest["orbit_counts"]
    if set(representatives) != {"full_reuse", "one_arm_reuse", "full_switch_d1", "full_switch_d2"}:
        raise ValueError("unexpected q1 orbit definitions")

    output.mkdir(parents=True)
    records = []
    for orbit, pair_text in representatives.items():
        left, right = (int(label.removeprefix("V")) for label in pair_text.split("-"))
        if f"V{left}" not in mapping or f"V{right}" not in mapping:
            raise ValueError(f"inactive representative: {orbit} {pair_text}")
        for replica in REPLICAS:
            name = f"b6_q1_{orbit}_v{left}_v{right}_rep{replica}"
            task = output / name
            pathway = task / "pathway"
            pathway.mkdir(parents=True)
            repdir = source_run / f"replica_{replica:02d}"
            start = repdir / "production_2ns.gro"
            checkpoint = repdir / "production_2ns.cpt"
            extension = json.loads((repdir / "extension_metadata.json").read_text())
            if extension.get("status") != "complete" or float(extension["target_total_ns"]) != 10.0:
                raise ValueError(f"not a completed 10 ns source: {repdir}")
            initial_nm, half_box_nm = protocol.pair_distance_nm(start, mapping, left, right)
            if MAXIMUM_NM >= 0.98 * half_box_nm:
                raise ValueError(f"{name}: maximum violates half-box guard {half_box_nm:.3f} nm")
            shutil.copy2(setup / "system.top", pathway / "system.top")
            shutil.copy2(start, pathway / "start.gro")
            shutil.copy2(checkpoint, pathway / "start.cpt")
            (pathway / "index.ndx").write_text(protocol.append_pair_groups(
                (setup / "index.ndx").read_text(), left, right, mapping), encoding="utf-8")
            (pathway / "steer_in.mdp").write_text(protocol.steering_mdp(
                left, right, mapping, initial_nm, protocol.MINIMUM_NM - 0.10), encoding="utf-8")
            (pathway / "steer_out.mdp").write_text(protocol.steering_mdp(
                left, right, mapping, initial_nm, MAXIMUM_NM + 0.10), encoding="utf-8")
            mdps = pathway / "window_mdps"
            mdps.mkdir()
            windows = []
            for index, center in enumerate(protocol.centers(MAXIMUM_NM)):
                window = f"window_{index:03d}_{center:.3f}nm"
                (mdps / f"{window}_equil.mdp").write_text(protocol.umbrella_mdp(
                    left, right, mapping, center, production=False), encoding="utf-8")
                (mdps / f"{window}_production.mdp").write_text(protocol.umbrella_mdp(
                    left, right, mapping, center, production=True), encoding="utf-8")
                windows.append({"window_index": index, "window_name": window, "center_nm": f"{center:.3f}"})
            with (pathway / "windows.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(windows[0]))
                writer.writeheader()
                writer.writerows(windows)
            record = {
                "pathway": name, "system": "B6_p_C6", "state": STATE, "stage": "q1",
                "route_family": orbit, "left": left, "right": right, "topology": "cross_arm",
                "replica": replica, "orbit_multiplicity": int(counts[orbit]),
                "source_job": SOURCE_JOB, "source_endpoint": str(start),
                "start_mode": "full-system 10 ns q1 checkpoint endpoint",
                "initial_distance_nm": initial_nm, "half_box_limit_nm": half_box_nm,
                "minimum_nm": protocol.MINIMUM_NM, "maximum_nm": MAXIMUM_NM,
                "window_spacing_nm": protocol.WINDOW_SPACING_NM, "window_count": len(windows),
                "force_constant_kJ_mol_nm2": protocol.FORCE_CONSTANT,
                "steering_rate_nm_ps": protocol.PULL_RATE_NM_PS,
                "equilibration_ps_per_window": protocol.EQUILIBRATION_PS,
                "production_ps_per_window": protocol.PRODUCTION_PS,
                "wham_begin_ps": protocol.WHAM_BEGIN_PS,
                "wham_end_ps": protocol.PRODUCTION_PS,
                "analysis_plan": "distance-biased PMF plus orientation-aware 2D reweighting and sampling audit",
            }
            (pathway / "pathway.json").write_text(json.dumps(record, indent=2) + "\n")
            shutil.copy2(project / "analysis/run_a4_m_c8_umbrella_path.py", task / "run_umbrella_path.py")
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
            (task / "run_umbrella.sh").write_text(shell)
            (task / "run_umbrella.sh").chmod(0o755)
            job = {
                "job_name": f"B6 q1 umbrella {name}", "command": "bash run_umbrella.sh",
                "log_file": "results/umbrella_job.log",
                "backward_files": [f"results/{archive}", "results/umbrella_job.log"],
                "project_id": PROJECT_ID, "machine_type": "c4_m15_1 * NVIDIA T4",
                "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
                "input_directory": str(task), "max_reschedule_times": 2,
                "max_run_time": 4320, "nnode": 1,
            }
            (task / "job.json").write_text(json.dumps(job, indent=2) + "\n")
            records.append(record)
    summary = {"status": "prepared_not_submitted", "project_id": PROJECT_ID,
               "paths": len(records), "orbits": counts, "replicas": list(REPLICAS),
               "window_protocol": "25 windows, 0.35–2.75 nm, 0.10 nm spacing, 100 ps equilibration + 500 ps production",
               "analysis_plan": "orientation-aware 2D reweighting; verify orientation mixing before inference",
               "records": records}
    (output / "jobs.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in ("status", "paths", "orbits", "replicas", "window_protocol")}, indent=2))


if __name__ == "__main__":
    main()
