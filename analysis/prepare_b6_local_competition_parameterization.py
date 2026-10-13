#!/usr/bin/env python3
"""Stage the corrected q0=A1-A6 B6 state for Bohrium Amber parameterization."""
from pathlib import Path
import csv, json, shutil

ROOT = Path(__file__).resolve().parents[2]
STATE = "B6_p_C6_q1_V1_V12_cis"
OUT = ROOT / "bohrium_submit/b6_local_q1_competition_parameterize_20260923_v4"

def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    src = ROOT / "gromacs_dcm_server_bundle"
    ws = OUT / "workspace"
    (ws / "analysis").mkdir(parents=True)
    (ws / "structures").mkdir()
    (ws / "gromacs_dcm").mkdir()
    for rel in ["mdp", "models"]:
        shutil.copytree(src / "gromacs_dcm" / rel, ws / "gromacs_dcm" / rel)
    shutil.copy2(src / "gromacs_dcm" / "dcm.pdb", ws / "gromacs_dcm" / "dcm.pdb")
    shutil.copy2(ROOT / "analysis/setup_explicit_dcm_gromacs.py", ws / "analysis/setup_explicit_dcm_gromacs.py")
    shutil.copy2(src / "structures" / f"{STATE}.pdb", ws / "structures" / f"{STATE}.pdb")
    shutil.copy2(src / "structures" / f"{STATE}_antechamber.mol2", ws / "structures" / f"{STATE}_antechamber.mol2")
    rows = list(csv.DictReader((src / "gromacs_dcm/state_manifest.csv").open(encoding="utf-8-sig", newline="")))
    row = next(r for r in rows if r["state"] == STATE)
    with (ws / "gromacs_dcm/state_manifest.csv").open("w", encoding="utf-8", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(row)); w.writeheader(); w.writerow(row)
    run = OUT / "run_parameterize.sh"
    run.write_text("""#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec > >(tee -a parameterize.log) 2>&1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python3 workspace/analysis/setup_explicit_dcm_gromacs.py \\
  --root workspace --parameterize-only --jobs 1 --sqm-grms-tol 0.05 \\
  --states B6_p_C6_q1_V1_V12_cis
tar -czf b6_local_q1_competition_parameterized.tar.gz \\
  workspace/gromacs_dcm/runs/B6_p_C6_q1_V1_V12_cis/setup/solute_parameters \\
  workspace/gromacs_dcm/state_manifest.csv parameterize.log
""", encoding="utf-8")
    run.chmod(0o755)
    job = {
        "job_name": "B6 local q1 A1-A6 Amber parameterization",
        "command": "bash run_parameterize.sh", "log_file": "parameterize.log",
        "backward_files": ["b6_local_q1_competition_parameterized.tar.gz", "parameterize.log"],
        "project_id": 4377225, "machine_type": "c8_m16_cpu",
        "image_address": "registry.dp.tech/dptech/amber:24_rc2",
        "input_directory": str(OUT), "max_reschedule_times": 2,
        "max_run_time": 720, "nnode": 1,
    }
    (OUT / "job.json").write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "job": job}, indent=2))

if __name__ == "__main__": main()
