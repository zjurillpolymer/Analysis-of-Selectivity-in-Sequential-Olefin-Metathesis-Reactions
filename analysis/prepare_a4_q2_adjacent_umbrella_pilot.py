#!/usr/bin/env python3
"""Prepare anchor-matched q2-adjacent same-arm/cross-arm umbrella pathways."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from analysis.prepare_a4_q2_umbrella_pilot import (
    EQUILIBRATION_PS,
    FORCE_CONSTANT,
    MINIMUM_NM,
    PRODUCTION_PS,
    PULL_RATE_NM_PS,
    STEERING_FORCE_CONSTANT,
    WHAM_BEGIN_PS,
    WINDOW_SPACING_NM,
    append_pair_groups,
    centers,
    pair_distance_nm,
    steering_mdp,
    umbrella_mdp,
)


PATHWAYS = (
    {
        "pathway": "c5_adj_same_v2_v3", "system": "A4_m_C5",
        "state": "A4_m_C5_2_adjacent_cis", "state_index": 1, "replica": 2,
        "topology": "same_arm", "route_family": "same_arm", "anchor": "V2",
        "left": 2, "right": 3, "maximum_nm": 2.65,
    },
    {
        "pathway": "c5_adj_cross_v2_v1", "system": "A4_m_C5",
        "state": "A4_m_C5_2_adjacent_cis", "state_index": 1, "replica": 2,
        "topology": "cross_arm", "route_family": "switch", "anchor": "V2",
        "left": 2, "right": 1, "maximum_nm": 2.65,
    },
    {
        "pathway": "c8_adj_same_v1_v2", "system": "A4_m_C8",
        "state": "A4_m_C8_2_adjacent_cis", "state_index": 3, "replica": 3,
        "topology": "same_arm", "route_family": "same_arm", "anchor": "V1",
        "left": 1, "right": 2, "maximum_nm": 2.65,
    },
    {
        "pathway": "c8_adj_cross_v1_v4", "system": "A4_m_C8",
        "state": "A4_m_C8_2_adjacent_cis", "state_index": 3, "replica": 3,
        "topology": "cross_arm", "route_family": "switch", "anchor": "V1",
        "left": 1, "right": 4, "maximum_nm": 2.65,
    },
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--export-root", type=Path,
        default=root / "results/a4_postreaction_nac_exports/b6-gromacs-parallel-kcl7v",
    )
    parser.add_argument(
        "--run-root", type=Path,
        default=root / "autodl-tmp/Molecular_dynamics_project/gromacs_dcm/runs",
    )
    parser.add_argument("--output", type=Path, default=root / "umbrella/a4_q2_adjacent_pilot")
    args = parser.parse_args()
    export_root = args.export_root.resolve()
    run_root = args.run_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for config in PATHWAYS:
        state = str(config["state"])
        replica = int(config["replica"])
        setup = run_root / state / "setup"
        endpoint = (
            export_root / f"gromacs-run-{int(config['state_index']):02d}-{replica:02d}"
            / state / f"replica_{replica:02d}"
        )
        setup_metadata = json.loads((setup / "metadata.json").read_text(encoding="utf-8"))
        mapping = {
            str(key): [int(value) for value in values]
            for key, values in setup_metadata["vinyl_mapping_gromacs_1based"].items()
        }
        left, right = int(config["left"]), int(config["right"])
        maximum_nm = float(config["maximum_nm"])
        pathway = output / str(config["pathway"])
        pathway.mkdir(parents=True, exist_ok=True)
        start = endpoint / "production_2ns.gro"
        initial_nm, half_box_nm = pair_distance_nm(start, mapping, left, right)
        if maximum_nm >= 0.98 * half_box_nm:
            raise ValueError(
                f"{pathway.name}: maximum {maximum_nm:.3f} nm violates 0.49L="
                f"{0.98 * half_box_nm:.3f} nm"
            )
        shutil.copy2(setup / "system.top", pathway / "system.top")
        shutil.copy2(start, pathway / "start.gro")
        (pathway / "index.ndx").write_text(
            append_pair_groups(
                (setup / "index.ndx").read_text(encoding="utf-8"), left, right, mapping,
            ),
            encoding="utf-8",
        )
        (pathway / "steer_in.mdp").write_text(
            steering_mdp(left, right, mapping, initial_nm, MINIMUM_NM - 0.10),
            encoding="utf-8",
        )
        (pathway / "steer_out.mdp").write_text(
            steering_mdp(left, right, mapping, initial_nm, maximum_nm),
            encoding="utf-8",
        )
        mdp_root = pathway / "window_mdps"
        mdp_root.mkdir(exist_ok=True)
        rows = []
        for index, center in enumerate(centers(maximum_nm)):
            name = f"window_{index:03d}_{center:.3f}nm"
            (mdp_root / f"{name}_equil.mdp").write_text(
                umbrella_mdp(left, right, mapping, center, False), encoding="utf-8",
            )
            (mdp_root / f"{name}_production.mdp").write_text(
                umbrella_mdp(left, right, mapping, center, True), encoding="utf-8",
            )
            rows.append({"window_index": index, "window_name": name, "center_nm": f"{center:.3f}"})
        with (pathway / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("window_index", "window_name", "center_nm"))
            writer.writeheader()
            writer.writerows(rows)
        pathway_metadata = {
            **config,
            "background_relationship": "adjacent",
            "comparison_design": "anchor-matched same-arm versus cross-arm",
            "initial_distance_nm": initial_nm,
            "half_box_limit_nm": half_box_nm,
            "gromacs_pull_guard_nm": 0.98 * half_box_nm,
            "minimum_nm": MINIMUM_NM,
            "maximum_nm": maximum_nm,
            "window_spacing_nm": WINDOW_SPACING_NM,
            "window_count": len(rows),
            "force_constant_kJ_mol_nm2": FORCE_CONSTANT,
            "steering_force_constant_kJ_mol_nm2": STEERING_FORCE_CONSTANT,
            "steering_rate_nm_ps": PULL_RATE_NM_PS,
            "equilibration_ps_per_window": EQUILIBRATION_PS,
            "production_ps_per_window": PRODUCTION_PS,
            "wham_begin_ps": WHAM_BEGIN_PS,
            "wham_end_ps": PRODUCTION_PS,
            "start_mode": "full-system endpoint GRO with newly generated velocities",
            "source_endpoint": str(start.relative_to(root)),
        }
        (pathway / "pathway.json").write_text(
            json.dumps(pathway_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        manifest.append(pathway_metadata)

    submission = {
        "analysis": "A4 q2-adjacent anchor-matched same/cross umbrella pilot",
        "gromacs_image": "registry.dp.tech/dptech/gromacs:2024.2",
        "temperature_K": 313.15,
        "pathways": manifest,
        "total_window_sampling_ns": sum(
            int(row["window_count"]) * (EQUILIBRATION_PS + PRODUCTION_PS) / 1000.0
            for row in manifest
        ),
        "interpretation": (
            "These PMFs test same-arm versus cross-arm contact preorganization on an adjacent-q2 "
            "background. They are distinct from opposite-q2 reuse/switch PMFs and from Ru chemistry."
        ),
    }
    (output / "submission_manifest.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(submission, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
