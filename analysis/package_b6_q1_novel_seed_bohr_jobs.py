#!/usr/bin/env python3
"""Package the six prepared B6 q1 seed-specific umbrella pathways for Bohrium."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = PROJECT / "b6_q1_novel_seed_umbrella_inputs_final_20261003"
DEFAULT_OUTPUT = PROJECT / "bohrium_submit/b6_q1_novel_seed_umbrella_20261003"
RUNNER = PROJECT / "analysis/run_a4_m_c8_umbrella_path.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs, output = args.inputs.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing job bundle: {output}")
    output.mkdir(parents=True)

    pathways = sorted(path for path in inputs.iterdir() if path.is_dir() and "__" in path.name)
    if len(pathways) != 6:
        raise RuntimeError(f"expected six prepared pathways, found {len(pathways)}")
    for source in pathways:
        task = output / source.name
        task.mkdir()
        shutil.copy2(RUNNER, task / "run_umbrella_path.py")
        shutil.copytree(
            source, task / "pathway",
            ignore=shutil.ignore_patterns("preflight"),
        )
        result_archive = f"{source.name}_results.tar.gz"
        (task / "run_umbrella.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            "export GMX_MAXBACKUP=-1\n"
            "mkdir -p results\n"
            "python3 -u run_umbrella_path.py --pathway pathway --ntomp 4 --gpu --stage all "
            "2>&1 | tee results/umbrella_job.log\n"
            f"tar -czf results/{result_archive} pathway\n",
            encoding="utf-8",
        )
        (task / "run_umbrella.sh").chmod(0o755)
        job = {
            "job_name": f"B6 q1 seeded umbrella {source.name}",
            "command": "bash run_umbrella.sh",
            "log_file": "results/umbrella_job.log",
            "backward_files": [f"results/{result_archive}", "results/umbrella_job.log"],
            "project_id": 4377225,
            "machine_type": "c4_m15_1 * NVIDIA T4",
            "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
            "input_directory": str(task),
            "max_reschedule_times": 2,
            "max_run_time": 4320,
            "nnode": 1,
        }
        (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "job_count": len(pathways),
        "jobs": [path.name for path in pathways],
        "project_id": 4377225,
        "machine_type": "c4_m15_1 * NVIDIA T4",
        "image": "registry.dp.tech/dptech/gromacs:2024.2",
        "max_runtime_minutes_per_job": 4320,
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
