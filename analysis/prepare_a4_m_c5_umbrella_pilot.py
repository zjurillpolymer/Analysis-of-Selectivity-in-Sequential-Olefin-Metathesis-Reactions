#!/usr/bin/env python3
"""Prepare the matched A4-m-C5 q0 same-arm/cross-arm umbrella comparison.

The two comparisons use symmetry-separated anchors and the strongest unbiased
cross-arm partner that shares each anchor: V1-V8 vs V1-V5 and V4-V5 vs V4-V8.
The restraint settings match the completed A4-m-C8 calculation.  Only the
maximum distance is shortened to the physically sampled support of A4-m-C5.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

from prepare_a4_m_c8_umbrella_pilot import (
    FORCE_CONSTANT,
    MINIMUM_NM,
    PULL_RATE_NM_PS,
    WINDOW_SPACING_NM,
    append_pair_groups,
    last_distance_nm,
    steering_mdp,
    umbrella_mdp,
    window_centers,
)


PAIR_CONFIGS = (
    {
        "pathway": "anchor_v1_same_v1_v8",
        "topology": "same_arm",
        "left": 1,
        "right": 8,
        "replica": 2,
        "maximum_nm": 1.625,
    },
    {
        "pathway": "anchor_v1_cross_v1_v5",
        "topology": "cross_arm",
        "left": 1,
        "right": 5,
        "replica": 2,
        "maximum_nm": 2.525,
    },
    {
        "pathway": "anchor_v4_same_v4_v5",
        "topology": "same_arm",
        "left": 4,
        "right": 5,
        "replica": 1,
        "maximum_nm": 1.625,
    },
    {
        "pathway": "anchor_v4_cross_v4_v8",
        "topology": "cross_arm",
        "left": 4,
        "right": 8,
        "replica": 1,
        "maximum_nm": 2.525,
    },
)


def pair_column(metadata: dict[str, object], left: int, right: int, topology: str) -> str:
    matches = [
        str(row["column"])
        for row in metadata["pair_schema"]
        if int(row["left"]) == left
        and int(row["right"]) == right
        and str(row["topology"]) == topology
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one {topology} schema entry for V{left}-V{right}; found {matches}"
        )
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--run-root",
        type=Path,
        default=root / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs/A4_m_C5_q0",
    )
    parser.add_argument("--output", type=Path, default=root / "umbrella/a4_m_c5_q0_pilot")
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((run_root / "setup/metadata.json").read_text(encoding="utf-8"))
    if metadata.get("state") != "A4_m_C5_q0":
        raise ValueError(f"unexpected source state: {metadata.get('state')}")
    mapping = {
        str(name): [int(value) for value in values]
        for name, values in metadata["vinyl_mapping_gromacs_1based"].items()
    }
    original_index = (run_root / "setup/index.ndx").read_text(encoding="utf-8")
    manifest: list[dict[str, object]] = []

    for config in PAIR_CONFIGS:
        pathway = output / str(config["pathway"])
        pathway.mkdir(parents=True, exist_ok=True)
        replica = int(config["replica"])
        replica_root = run_root / f"replica_{replica:02d}"
        left, right = int(config["left"]), int(config["right"])
        topology = str(config["topology"])
        column = pair_column(metadata, left, right, topology)
        initial_nm = last_distance_nm(replica_root / "vinyl_distances.csv", column)
        maximum_nm = float(config["maximum_nm"])
        centers = window_centers(maximum_nm)

        shutil.copy2(run_root / "setup/system.top", pathway / "system.top")
        shutil.copy2(replica_root / "production_2ns.gro", pathway / "start.gro")
        shutil.copy2(replica_root / "production_2ns.cpt", pathway / "start.cpt")
        (pathway / "index.ndx").write_text(
            append_pair_groups(original_index, left, right, mapping), encoding="utf-8"
        )
        inward_target = MINIMUM_NM - 0.10
        outward_target = maximum_nm + 0.10
        (pathway / "steer_in.mdp").write_text(
            steering_mdp(left, right, mapping, initial_nm, inward_target), encoding="utf-8"
        )
        (pathway / "steer_out.mdp").write_text(
            steering_mdp(left, right, mapping, initial_nm, outward_target), encoding="utf-8"
        )

        windows = pathway / "window_mdps"
        windows.mkdir(exist_ok=True)
        window_rows = []
        for index, center in enumerate(centers):
            name = f"window_{index:03d}_{center:.3f}nm"
            (windows / f"{name}_equil.mdp").write_text(
                umbrella_mdp(left, right, mapping, center, production=False), encoding="utf-8"
            )
            (windows / f"{name}_production.mdp").write_text(
                umbrella_mdp(left, right, mapping, center, production=True), encoding="utf-8"
            )
            window_rows.append({"window_index": index, "window_name": name, "center_nm": center})
        with (pathway / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("window_index", "window_name", "center_nm")
            )
            writer.writeheader()
            writer.writerows(window_rows)

        steering_ps = (
            abs(initial_nm - inward_target) + abs(outward_target - initial_nm)
        ) / PULL_RATE_NM_PS + 600.0
        pathway_metadata = {
            **config,
            "pair_column": column,
            "initial_distance_nm": initial_nm,
            "minimum_nm": MINIMUM_NM,
            "maximum_nm": maximum_nm,
            "window_spacing_nm": WINDOW_SPACING_NM,
            "window_count": len(centers),
            "force_constant_kJ_mol_nm2": FORCE_CONSTANT,
            "steering_rate_nm_ps": PULL_RATE_NM_PS,
            "estimated_total_steering_ps": steering_ps,
            "equilibration_ps_per_window": 200.0,
            "production_ps_per_window": 2000.0,
            "source_replica": f"replica_{replica:02d}",
            "source_endpoint_total_ns": 4.0,
        }
        (pathway / "pathway.json").write_text(
            json.dumps(pathway_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        manifest.append(pathway_metadata)

    total_windows = sum(int(row["window_count"]) for row in manifest)
    total_window_sampling_ns = total_windows * 2.2
    total_steering_ns = sum(float(row["estimated_total_steering_ps"]) for row in manifest) / 1000.0
    submission = {
        "analysis": "A4-m-C5 q0 pair-specific umbrella comparison",
        "gromacs_image": "registry.dp.tech/dptech/gromacs:2024.2",
        "temperature_K": 313.15,
        "pathways": manifest,
        "total_windows": total_windows,
        "estimated_window_sampling_ns": total_window_sampling_ns,
        "estimated_steering_ns": total_steering_ns,
        "recommended_dispatch": "one independent GPU job per pathway",
        "command": (
            "python3 analysis/run_a4_m_c8_umbrella_path.py "
            "--pathway PATHWAY_DIR --ntomp 4 --gpu --stage all"
        ),
        "analysis_plan": {
            "contact_region_nm": [0.325, 0.500],
            "primary_normalization": "integrate contact versus the full sampled noncontact support",
            "reason": (
                "A4-m-C5 same-arm distances have a physical upper support near 1.6 nm, "
                "so the A4-m-C8 1.5-2.1 nm reference interval is not common to both topologies."
            ),
            "comparability": (
                "Restraint, spacing, temperature, equilibration, production, WHAM, and bootstrap "
                "settings match A4-m-C8; pathway maxima follow each molecule's physical support."
            ),
        },
        "interpretation": (
            "These PMFs quantify catalyst-free conformational preorganization, not the Ru-catalyzed "
            "chemical activation barrier."
        ),
    }
    (output / "submission_manifest.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(submission, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
