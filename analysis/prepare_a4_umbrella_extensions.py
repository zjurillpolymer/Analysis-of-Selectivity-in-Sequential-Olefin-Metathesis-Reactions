#!/usr/bin/env python3
"""Package equal 1.5 ns continuations for the A4 C5/C8 q1 and q2 paths."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
EXTEND_PS = 1500.0

Q1_ROOTS = (
    PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_download",
    PROJECT / "results/a4_m_c5_c8_q1_three_class_umbrella_rep3_download",
)
Q2_PATHS = (
    PROJECT / "results/a4_q2_umbrella_bohrium/download/c5_reuse_v1_v6/pathway",
    PROJECT / "results/a4_q2_umbrella_bohrium/download/c5_reuse_v2_v4/pathway",
    PROJECT / "results/a4_q2_umbrella_bohrium/download/c5_switch_v4_v6/pathway",
    PROJECT / "results/a4_q2_umbrella_bohrium/resumed_download/c8_reuse_v5_v7/pathway",
    PROJECT / "results/a4_q2_umbrella_bohrium/partial/c8_switch_v4_v5/pathway",
)


def discover_q1() -> dict[str, Path]:
    found: dict[str, Path] = {}
    for root in Q1_ROOTS:
        for metadata in root.glob("*/extracted/pathway/pathway.json"):
            record = json.loads(metadata.read_text(encoding="utf-8"))
            name = str(record["pathway"])
            if name in found and found[name] != metadata.parent:
                raise ValueError(f"duplicate pathway {name}")
            found[name] = metadata.parent
    return found


def validate_source(name: str, source: Path) -> dict[str, object]:
    metadata_path = source / "pathway.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    windows = sorted((source / "windows").glob("window_*"))
    if not windows:
        raise FileNotFoundError(f"no windows in {source}")
    complete_windows = []
    incomplete_windows = []
    for window in windows:
        missing = [filename for filename in ("production.tpr", "production.cpt", "production.gro")
                   if not (window / filename).is_file() or (window / filename).stat().st_size == 0]
        if missing:
            if name != "c8_switch_v4_v5" or window.name != "window_024_2.750nm":
                raise FileNotFoundError(f"{window}: missing {missing}")
            incomplete_windows.append(window.name)
        else:
            complete_windows.append(window)
    if not complete_windows:
        raise FileNotFoundError(f"no complete production windows in {source}")
    record["extension_ps"] = EXTEND_PS
    record["extension_mode"] = "gmx convert-tpr + mdrun -cpi -noappend; original production files preserved"
    record["source_pathway_name"] = name
    record["source_window_count"] = len(complete_windows)
    record["incomplete_source_windows_excluded"] = incomplete_windows
    record["original_source_maximum_nm"] = record["maximum_nm"]
    record["window_count"] = len(complete_windows)
    record["maximum_nm"] = max(float(window.name.rsplit("_", 1)[-1].removesuffix("nm"))
                               for window in complete_windows)
    for window in complete_windows:
        numeric = [line for line in (window / "production_pullx.xvg").read_text().splitlines()
                   if line.strip() and not line.lstrip().startswith(("#", "@"))]
        if not numeric or abs(float(numeric[-1].split()[0]) - 500.0) > 1e-4:
            raise ValueError(f"{window}: production endpoint is not 500 ps")
    return record


def package_task(job_root: Path, source: Path, record: dict[str, object]) -> None:
    name = str(record["pathway"])
    task = job_root / name
    task.mkdir(parents=True, exist_ok=True)
    destination = task / "pathway"
    destination.mkdir()
    (destination / "pathway.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = (source / "windows.csv").read_text().splitlines()
    excluded = set(record["incomplete_source_windows_excluded"])
    (destination / "windows.csv").write_text(
        "\n".join(line for line in lines if not any(name in line for name in excluded)) + "\n",
        encoding="utf-8",
    )
    for source_window in sorted((source / "windows").glob("window_*")):
        if any(not (source_window / filename).is_file() or (source_window / filename).stat().st_size == 0
               for filename in ("production.tpr", "production.cpt", "production.gro")):
            continue
        target_window = destination / "windows" / source_window.name
        target_window.mkdir(parents=True, exist_ok=True)
        for filename in ("production.tpr", "production.cpt", "production.gro"):
            shutil.copy2(source_window / filename, target_window / filename)
    (destination / "extension_request.json").write_text(
        json.dumps({"extend_ps": EXTEND_PS, "source": str(source)}, indent=2) + "\n",
        encoding="utf-8",
    )
    archive = f"{name}_extension_results.tar.gz"
    shell = "\n".join([
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export GMX_MAXBACKUP=-1",
        "export OMP_NUM_THREADS=4",
        "mkdir -p results",
        f"trap 'task_rc=$?; tar -czf results/{archive} pathway; exit \"$task_rc\"' EXIT",
        f"python3 -u run_extension.py --pathway pathway --extend-ps {EXTEND_PS:g} --ntomp 4 --gpu 2>&1 | tee results/extension_job.log",
        "",
    ])
    runner = task / "run_extension.sh"
    runner.write_text(shell, encoding="utf-8")
    runner.chmod(0o755)
    job = {
        "job_name": f"A4-m umbrella extension {name}",
        "command": "bash run_extension.sh",
        "log_file": "results/extension_job.log",
        "backward_files": [f"results/{archive}", "results/extension_job.log"],
        "project_id": 4377225,
        "machine_type": "c4_m15_1 * NVIDIA T4",
        "image_address": "registry.dp.tech/dptech/gromacs:2024.2",
        "input_directory": str(task.resolve()),
        "max_reschedule_times": 2,
        "max_run_time": 4320,
        "nnode": 1,
    }
    shutil.copy2(PROJECT / "analysis/run_a4_umbrella_extension.py", task / "run_extension.py")
    (task / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "bohrium_submit/a4_umbrella_extensions_2ns_fix1")
    args = parser.parse_args()
    job_root = args.output.resolve()
    if job_root.exists():
        raise FileExistsError(f"refusing to overwrite {job_root}")
    sources = discover_q1()
    for path in Q2_PATHS:
        metadata = json.loads((path / "pathway.json").read_text(encoding="utf-8"))
        name = str(metadata["pathway"])
        if name in sources and sources[name] != path:
            raise ValueError(f"duplicate pathway {name}")
        sources[name] = path
    if len(sources) != 23:
        raise RuntimeError(f"expected 23 C5/C8 q1+q2 paths, found {len(sources)}")
    job_root.mkdir(parents=True)
    records = []
    for name, source in sorted(sources.items()):
        record = validate_source(name, source)
        package_task(job_root, source, record)
        records.append(record)
    manifest = {
        "format": "a4-c5-c8-umbrella-extension-fix1-noappend",
        "project_id": 4377225,
        "extend_ps": EXTEND_PS,
        "target_total_production_ps": 2000.0,
        "pathway_count": len(records),
        "q1_pathway_count": 18,
        "q2_pathway_count": 5,
        "dispatch": "one NVIDIA T4 job per pathway",
        "pathways": records,
    }
    (job_root / "extensions_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(job_root), "pathways": len(records),
                      "target_total_production_ps": 2000.0}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
