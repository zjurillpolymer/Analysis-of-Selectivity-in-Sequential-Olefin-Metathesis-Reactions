#!/usr/bin/env python3
"""Audit three paired B6 q1 umbrella starts and reweight contact/orientation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from analyze_a4_m_c6_c7_q1_umbrella_2d import prepare_topology
from analyze_a4_q2_umbrella_2d_reweight import RT_KCAL, analyze_pathway, write_csv
from analyze_b6_q1_umbrella import wham_audit


PROJECT = Path(__file__).resolve().parents[1]
DOWNLOAD = PROJECT.parent / "bohrium_download/b6_local_q1_paired_umbrella_complete_20261001"
OUTPUT = PROJECT / "results/b6_local_q1_arm1_competition_20260923/paired_terminal_umbrella_20261001"
JOBS = {
    1: {"A1_to_A6": 20806766, "A1_to_A2": 20806768},
    2: {"A1_to_A6": 20806769, "A1_to_A2": 20806770},
    3: {"A1_to_A6": 20806772, "A1_to_A2": 20806773},
}


def ratio(a6: float, a2: float) -> dict[str, float]:
    weight_a6, weight_a2 = a6, 2.0 * a2
    total = weight_a6 + weight_a2
    return {
        "A6_to_A2_weight_ratio": weight_a6 / weight_a2,
        "P_A6_if_two_channels_only": weight_a6 / total,
        "P_A2_if_two_channels_only": weight_a2 / total,
        "deltaG_A6_minus_A2_with_degeneracy_kcal_mol": -RT_KCAL * math.log(weight_a6 / weight_a2),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    topology_root = OUTPUT / "reactive_pair_topologies"
    rows: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for pair_id, channels in JOBS.items():
        current: dict[str, dict[str, object]] = {}
        for channel, job_id in channels.items():
            pathway = DOWNLOAD / str(job_id) / "extracted/pathway"
            metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
            if int(metadata["basin_id"]) != pair_id or metadata["local_branch"] != channel:
                raise ValueError(f"mismatched pathway metadata: {job_id}")
            audit = wham_audit(pathway)
            prepare_topology(pathway, topology_root)
            result = analyze_pathway(pathway, PROJECT, OUTPUT, "gmx", topology_root)
            row: dict[str, object] = {
                "bohr_id": job_id,
                "pair_id": pair_id,
                "source_replica": metadata["source_replica"],
                "terminal_cluster_id": metadata["cluster_id"],
                "channel": channel,
                "pull_pair_old_labels": metadata["pull_pair_old_labels"],
                "current_mapping": metadata["current_mapping"],
                "multiplicity": metadata["orbit_multiplicity"],
                "windows": len(result.centers_nm),
                "frames": len(result.distance_nm),
                "mbar_iterations": result.mbar_iterations,
                "mbar_residual": result.mbar_residual,
                "effective_sample_size": result.effective_sample_size,
                "geometry_pull_rms_A": result.geometry_pull_rms_A,
                "geometry_pull_max_A": result.geometry_pull_max_A,
                **audit,
                **result.metrics,
            }
            for half, metrics in result.half_metrics.items():
                row.update({f"{half}_{key}": value for key, value in metrics.items()})
            rows.append(row)
            current[channel] = row
            write_csv(OUTPUT / "pathway_audit.csv", rows)
            print(f"{job_id} {channel} NAC={row['soft_nac_probability']:.6g} "
                  f"ESS={row['effective_sample_size']:.1f}", flush=True)
        a6, a2 = current["A1_to_A6"], current["A1_to_A2"]
        pair: dict[str, object] = {
            "pair_id": pair_id,
            "source_replica": a6["source_replica"],
            "terminal_cluster_id": a6["terminal_cluster_id"],
            "A6_bohr_id": a6["bohr_id"],
            "A2_bohr_id": a2["bohr_id"],
        }
        for label, prefix in (("soft_contact_probability", "contact"),
                              ("conditional_orientation", "orientation"),
                              ("soft_nac_probability", "nac")):
            pair[f"A6_{prefix}"] = a6[label]
            pair[f"A2_{prefix}"] = a2[label]
        pair.update(ratio(float(a6["soft_nac_probability"]), float(a2["soft_nac_probability"])))
        for half in ("first_half", "second_half"):
            vals = ratio(float(a6[f"{half}_soft_nac_probability"]),
                         float(a2[f"{half}_soft_nac_probability"]))
            pair.update({f"{half}_{key}": value for key, value in vals.items()})
        pair["half_difference_deltaG_kcal_mol"] = (
            float(pair["second_half_deltaG_A6_minus_A2_with_degeneracy_kcal_mol"])
            - float(pair["first_half_deltaG_A6_minus_A2_with_degeneracy_kcal_mol"])
        )
        pairs.append(pair)
        write_csv(OUTPUT / "paired_comparison.csv", pairs)
        print(f"pair{pair_id} A6/(2*A2)={pair['A6_to_A2_weight_ratio']:.6g} "
              f"P_A2={pair['P_A2_if_two_channels_only']:.6g}", flush=True)

    summary = {
        "source": str(DOWNLOAD),
        "temperature_K": 313.15,
        "pair_count": 3,
        "source_note": "Three independent complete 10 ns terminal boxes. Pairs 1 and 2 share the same broad k-means label; these are not three proven thermodynamic basins.",
        "multiplicity": {"A1_to_A6": 1, "A1_to_A2": 2},
        "paths": rows,
        "pairs": pairs,
        "interpretation_limit": "Restrained conformational near-attack weights; not chemical rate constants or product yields. Finite sampling and start dependence must be checked before a pooled probability is claimed.",
    }
    (OUTPUT / "paired_comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
