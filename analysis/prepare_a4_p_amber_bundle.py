#!/usr/bin/env python3
"""Create a compact upload bundle for A4_p-C6/C8 Amber + optional MD runs.

The bundle deliberately excludes q2-mixed.  It contains one q1 cross-arm
state and q2 mixed/adjacent/opposite metadata only insofar as the present
request needs adjacent and opposite; the production state list below is q1,
q2-adjacent, and q2-opposite for both molecules.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from itertools import combinations
from pathlib import Path


STATES = (
    ("A4_p_C6_1_cis", "q1", "cross"),
    ("A4_p_C6_2_adjacent_cis", "q2", "adjacent"),
    ("A4_p_C6_2_opposite_cis", "q2", "opposite"),
    ("A4_p_C8_1_cis", "q1", "cross"),
    ("A4_p_C8_2_adjacent_cis", "q2", "adjacent"),
    ("A4_p_C8_2_opposite_cis", "q2", "opposite"),
)


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def pair_schema(active: list[int], arms: dict[int, int]) -> list[dict[str, object]]:
    return [
        {
            "left": left,
            "right": right,
            "topology": "same_arm" if arms[left] == arms[right] else "cross_arm",
            "column": f"V{left}_V{right}_{'同臂' if arms[left] == arms[right] else '跨臂'}",
        }
        for left, right in combinations(active, 2)
    ]


def build_manifest(
    root: Path, output: Path, audit: dict[str, object]
) -> list[dict[str, object]]:
    by_system = {str(item["system"]): item for item in audit["systems"]}
    rows: list[dict[str, object]] = []
    for state, stage, branch in STATES:
        system = state.removesuffix("_1_cis").removesuffix("_2_adjacent_cis").removesuffix("_2_opposite_cis")
        record = by_system[system]
        source = output / "structures" / f"{state}.pdb"
        if stage == "q1":
            info = record["q1"]
        else:
            info = record["q2"][branch]
        active = [int(value.removeprefix("V")) for value in info["active_vinyls"]]
        arms = {
            int(value.removeprefix("V")): int(arm.removeprefix("A"))
            for value, arm in record["arm_mapping_1based"].items()
        }
        mapping = {str(key.removeprefix("V")): value for key, value in info["mapping"].items()}
        rows.append(
            {
                "state": state,
                "stage": stage,
                "system": system,
                "branch": branch,
                "source_pdb": f"structures/{state}.pdb",
                "net_charge": 0,
                "active_vinyl_count": len(active),
                "active_pair_count": len(active) * (len(active) - 1) // 2,
                "vinyl_mapping_json": json.dumps({f"V{key}": value for key, value in mapping.items()}, separators=(",", ":")),
                "pair_schema_json": json.dumps(pair_schema(active, arms), ensure_ascii=False, separators=(",", ":")),
                "output_dir": f"gromacs_dcm/runs/{state}",
            }
        )
        if not source.is_file():
            raise FileNotFoundError(source)
    return rows


def write_server_readme(output: Path) -> None:
    text = r'''# A4-p-C6/C8 Amber + CPU workflow

This bundle contains only six production states: one q1 cross-arm state and q2-adjacent/q2-opposite for A4-p-C6 and A4-p-C8. q2-mixed is intentionally excluded from the Amber/MD upload.

## 0. Environment check

```bash
cd ~/a4_p_c6_c8_amber_md_bundle
conda activate md_tools
which antechamber parmchk2 tleap packmol
python -c "import parmed, MDAnalysis; print('ParmEd/MDAnalysis OK')"
```

`gmx` is optional for the Amber-only stage. If it is absent, use `--defer-grompp`; that finishes Antechamber/AM1-BCC, ParmChk, Packmol, tleap and ParmEd conversion without creating a GROMACS TPR.

## 1. Fast Amber parameterization on 16 vCPU

Run four independent SQM/AM1-BCC jobs, four OpenMP threads each. This uses all 16 vCPUs without launching sixteen competing SQM jobs:

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
python analysis/setup_explicit_dcm_gromacs.py \
  --parameterize-only --jobs 4 --sqm-grms-tol 0.05 \
  --states A4_p_C6_1_cis A4_p_C6_2_adjacent_cis A4_p_C6_2_opposite_cis \
           A4_p_C8_1_cis A4_p_C8_2_adjacent_cis A4_p_C8_2_opposite_cis
```

The explicit-bond `*_antechamber.mol2` files are already beside each PDB, so Antechamber does not need to guess the large-macrocycle bond graph.

## 2. Complete Amber/Packmol systems

Use two setup workers for Packmol/tleap and leave the four AM1-BCC results cached:

```bash
export OMP_NUM_THREADS=8
python analysis/setup_explicit_dcm_gromacs.py \
  --jobs 2 --defer-grompp --dcm-model ch2cl2_dc --sqm-grms-tol 0.05 \
  --states A4_p_C6_1_cis A4_p_C6_2_adjacent_cis A4_p_C6_2_opposite_cis \
           A4_p_C8_1_cis A4_p_C8_2_adjacent_cis A4_p_C8_2_opposite_cis
```

Each state should then contain `setup/solute_parameters/{slu.mol2,slu.frcmod,slu.pdb}`, `setup/system.prmtop`, `setup/system.rst7`, `setup/system.top`, and `setup/system.gro`. The explicit CH2Cl2-DC density gate must be run before production MD.

## 3. GPU GROMACS production

Your server exposes one 32-GB CUDA vGPU. For these solvent boxes the fastest stable setting is one GROMACS rank with eight OpenMP threads; nonbonded and PME work go to the GPU, while bonded/update/constraints stay on CPU as required by the rigid CH2Cl2-DC model:

```bash
nvidia-smi
python analysis/run_explicit_dcm_gromacs.py --solvent-only --dcm-model ch2cl2_dc --ntomp 8
python analysis/run_explicit_dcm_gromacs.py --dcm-model ch2cl2_dc --replicas 4 --ntomp 8 \
  --states A4_p_C6_1_cis A4_p_C6_2_adjacent_cis A4_p_C6_2_opposite_cis \
           A4_p_C8_1_cis A4_p_C8_2_adjacent_cis A4_p_C8_2_opposite_cis
```

Run states sequentially on the single GPU; do not launch six simultaneous `mdrun` processes. Do not mix these explicit-DCM trajectories with the old vacuum/GFN-FF runs. Extract distances afterward with `python analysis/extract_explicit_dcm_distances.py --replicas 4`.
'''
    (output / "README_SERVER.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_default = Path(__file__).resolve().parents[1]
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--output", type=Path, default=root_default / "results" / "a4_p_c6_c8_amber_md_bundle")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    review = root / "results" / "a4_p_c6_c8_q1_q2_three_branches_rdkit_review"
    audit = json.loads((review / "audit.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)

    for relative in (
        "gromacs_dcm/mdp",
        "gromacs_dcm/models",
        "gromacs_dcm/protocol.json",
        "gromacs_dcm/environment.yml",
        "gromacs_dcm/README.md",
        "analysis/setup_explicit_dcm_gromacs.py",
        "analysis/run_explicit_dcm_gromacs.py",
        "analysis/extract_explicit_dcm_distances.py",
        "analysis/analyze_explicit_dcm_postreaction_nac.py",
        "analysis/analyze_explicit_dcm_q0_geometry_gate.py",
        "analysis/build_antechamber_bonded_mol2.py",
    ):
        source = root / relative
        if source.is_dir():
            shutil.copytree(source, output / relative, dirs_exist_ok=True)
        else:
            copy_file(source, output / relative)

    for state, _, _ in STATES:
        for suffix in (".pdb", "_antechamber.mol2"):
            copy_file(review / f"{state}{suffix}", output / "structures" / f"{state}{suffix}")

    rows = build_manifest(root, output, audit)
    with (output / "gromacs_dcm" / "state_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    copy_file(review / "audit.json", output / "review" / "audit.json")
    copy_file(review / "review.md", output / "review" / "review.md")
    write_server_readme(output)
    (output / "review" / "selected_states.json").write_text(
        json.dumps({"states": [state for state, _, _ in STATES], "mixed_excluded": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "output": str(output), "states": [row["state"] for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
