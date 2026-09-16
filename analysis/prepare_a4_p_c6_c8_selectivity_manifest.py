#!/usr/bin/env python3
"""Build the q1 -> q2 neighbour-only selectivity manifest.

The Amber bundle contains q1 and q2 structures generated from an audited
reaction lineage.  This script does not run MD.  It makes that lineage
explicit and filters candidate vinyl pairs to adjacent molecular arms only,
which is the model assumption for the A4-p-C6/C8 extension.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def edge(arms: list[str] | tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted(arms))  # type: ignore[return-value]


def adjacent_arm_pair(left: str, right: str) -> bool:
    """Return whether two of A1..A4 are neighbours on the cyclic scaffold."""
    numbers = sorted((int(left.removeprefix("A")), int(right.removeprefix("A"))))
    return numbers in ([1, 2], [2, 3], [3, 4], [1, 4])


def read_audit(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "PASS":
        raise ValueError(f"audit is not PASS: {path}")
    return data


def read_states(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    states = data.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError(f"no states in {path}")
    if any("mixed" in str(state) for state in states):
        raise ValueError("q2 mixed states must remain excluded")
    return [str(state) for state in states]


def ranked_contact_lookup(q2: dict[str, Any]) -> dict[tuple[str, str], float]:
    result: dict[tuple[str, str], float] = {}
    for item in q2.get("candidate_pairs_ranked", []):
        pair = tuple(sorted(str(value).removeprefix("V") for value in item["pair"]))
        result[pair] = float(item.get("q0_mean_soft_contact", 0.0))
    return result


def build_rows(audit: dict[str, Any], selected_states: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for system_info in audit["systems"]:
        system = str(system_info["system"])
        arm_mapping = system_info["arm_mapping_1based"]
        q1 = system_info["q1"]
        q1_pair = tuple(str(value) for value in q1["reacted_pair"])
        q1_edge = edge([arm_mapping[q1_pair[0]], arm_mapping[q1_pair[1]]])

        if q1["state"] in selected_states:
            # q1 candidate pairs are reconstructed from the active vinyls.
            active = [str(value) for value in q1["active_vinyls"]]
            for index, left in enumerate(active):
                for right in active[index + 1 :]:
                    left_arm, right_arm = arm_mapping[left], arm_mapping[right]
                    topology = "same" if left_arm == right_arm else "cross"
                    if topology == "cross" and not adjacent_arm_pair(left_arm, right_arm):
                        continue
                    rows.append({
                        "system": system,
                        "state": q1["state"],
                        "stage": "q1",
                        "branch": "cross",
                        "lineage_q1_pair": "-".join(q1_pair),
                        "lineage_q1_arm_edge": "-".join(q1_edge),
                        "lineage_q2_pair": "",
                        "pair": f"{left}-{right}",
                        "left": int(left.removeprefix("V")),
                        "right": int(right.removeprefix("V")),
                        "topology_class": topology,
                        "route_class": "",
                        "allowed_by_neighbour_model": True,
                        "q0_mean_soft_contact": "",
                    })

        for branch in ("adjacent", "opposite"):
            q2 = system_info["q2"][branch]
            state = str(q2["state"])
            if state not in selected_states:
                continue
            q2_pair = tuple(str(value) for value in q2["reacted_pair"])
            q2_edge = edge([arm_mapping[q2_pair[0]], arm_mapping[q2_pair[1]]])
            previous_edges = {q1_edge, q2_edge}
            contact = ranked_contact_lookup(q2)
            active = [str(value) for value in q2["active_vinyls"]]
            for index, left in enumerate(active):
                for right in active[index + 1 :]:
                    left_arm, right_arm = arm_mapping[left], arm_mapping[right]
                    topology = "same" if left_arm == right_arm else "cross"
                    allowed = topology == "same" or adjacent_arm_pair(left_arm, right_arm)
                    if not allowed:
                        continue
                    pair_edge = edge([left_arm, right_arm])
                    route = (
                        "reuse" if topology == "cross" and pair_edge in previous_edges
                        else "switch" if topology == "cross" else ""
                    )
                    rows.append({
                        "system": system,
                        "state": state,
                        "stage": "q2",
                        "branch": branch,
                        "lineage_q1_pair": "-".join(q1_pair),
                        "lineage_q1_arm_edge": "-".join(q1_edge),
                        "lineage_q2_pair": "-".join(q2_pair),
                        "pair": f"{left}-{right}",
                        "left": int(left.removeprefix("V")),
                        "right": int(right.removeprefix("V")),
                        "topology_class": topology,
                        "route_class": route,
                        "allowed_by_neighbour_model": True,
                        "q0_mean_soft_contact": contact.get(
                            tuple(sorted((left.removeprefix("V"), right.removeprefix("V")))), ""
                        ),
                    })
    return rows


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--selected-states", type=Path)
    parser.add_argument("--output", type=Path, default=project / "results/A4_p_C6_C8_selectivity_workflow")
    args = parser.parse_args()
    audit_path = args.audit or (project / "review/audit.json")
    selected_states_path = args.selected_states or (project / "review/selected_states.json")
    if not audit_path.is_file():
        audit_path = project / "results/A4_p_C6_C8_amber_md_bundle/review/audit.json"
    if not selected_states_path.is_file():
        selected_states_path = project / "results/A4_p_C6_C8_amber_md_bundle/review/selected_states.json"
    audit = read_audit(audit_path.resolve())
    states = read_states(selected_states_path.resolve())
    rows = build_rows(audit, states)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with (output / "neighbour_pair_classes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "model": "A4 neighbour-arm only",
        "mixed_excluded": True,
        "q2_is_generated_from_q1": True,
        "states": states,
        "pilot": {"replicas": 3, "duration_ns": 2.0, "purpose": "basin/contact/PBC sanity check"},
        "extension": {"duration_ns_total": 10.0, "trigger": "pilot indicates sparse events, basin memory, or replica disagreement"},
        "umbrella": {"purpose": "selected-path PMF explanation and sparse-event rescue", "minimum_nm": 0.35, "maximum_nm": 2.75, "spacing_nm": 0.10, "force_constant_kJ_mol_nm2": 500.0, "equilibration_ps": 100.0, "production_ps": 500.0},
        "rows": len(rows),
    }
    (output / "workflow_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
