#!/usr/bin/env python3
"""Build the minimal B6 q1 candidate table under the neighbour-only model.

The model used here is deliberately smaller than the legacy labelled B6 DAG:

* six arms are vertices of a C6 arm-adjacency cycle;
* only cross-arm pairs on neighbouring arm edges are allowed;
* one q0 cross pair is fixed as the representative first event;
* q1 candidates are grouped by the overlap of their arm edge with q0 and by
  the residual reflection symmetry that leaves q0 fixed.

This script only creates an audit table.  It does not choose force fields or
submit MD jobs.  The table is the input for selecting one umbrella path per
orbit and retaining the orbit multiplicity in the conditional probability.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# Authoritative arm order for the present neighbour-only model. The six arms
# are (V1,V2), (V3,V4), ..., (V11,V12). Do not infer this from legacy labels.
ARM_CYCLE = tuple(range(6))


def parse_pair(text: str) -> tuple[int, int]:
    values = tuple(sorted(int(item.removeprefix("V")) for item in text.split(",")))
    if len(values) != 2 or values[0] == values[1] or not all(1 <= value <= 12 for value in values):
        raise ValueError(f"expected two distinct vinyl labels, got {text!r}")
    return values


def load_arm_map(root: Path) -> dict[int, int]:
    return {vinyl: (vinyl - 1) // 2 for vinyl in range(1, 13)}


def arm_edge(a: int, b: int) -> tuple[int, int]:
    return tuple(sorted((a, b)))


def cycle_edges() -> set[tuple[int, int]]:
    return {
        arm_edge(ARM_CYCLE[index], ARM_CYCLE[(index + 1) % len(ARM_CYCLE)])
        for index in range(len(ARM_CYCLE))
    }


def residual_orbit(q0_edge: tuple[int, int], candidate_edge: tuple[int, int]) -> str:
    """Return a stable geometric orbit name relative to the fixed q0 edge.

    For a C6 cycle the stabilizer of one edge contains one reflection.  Edges
    with the same distance from q0 under this reflection are equivalent.  We
    use the minimum graph distance between the two arm edges as a compact,
    label-independent descriptor; the full edge representative is retained in
    the output for auditability.
    """
    q0 = set(q0_edge)
    candidate = set(candidate_edge)
    overlap = len(q0 & candidate)
    if overlap == 2:
        return "full_reuse"
    if overlap == 1:
        return "one_arm_reuse"

    def edge_distance(e1: tuple[int, int], e2: tuple[int, int]) -> int:
        # Distance between two edges in C6: minimum vertex distance.
        n = len(ARM_CYCLE)
        distances = []
        for left in e1:
            for right in e2:
                delta = abs(left - right)
                distances.append(min(delta, n - delta))
        return min(distances)

    distance = edge_distance(q0_edge, candidate_edge)
    return f"full_switch_d{distance}"


def build_rows(arm_of: dict[int, int], q0_pair: tuple[int, int]) -> list[dict[str, object]]:
    q0_arms = arm_edge(arm_of[q0_pair[0]], arm_of[q0_pair[1]])
    if q0_arms not in cycle_edges():
        raise ValueError(
            f"q0 pair V{q0_pair[0]}-V{q0_pair[1]} uses arm edge {q0_arms}, "
            "which is not a neighbouring C6 edge"
        )
    consumed = set(q0_pair)
    rows: list[dict[str, object]] = []
    for left in range(1, 13):
        for right in range(left + 1, 13):
            if left in consumed or right in consumed:
                continue
            left_arm, right_arm = arm_of[left], arm_of[right]
            if left_arm == right_arm:
                continue  # same-arm is outside this cross-only q1 model
            edge = arm_edge(left_arm, right_arm)
            if edge not in cycle_edges():
                continue
            category = residual_orbit(q0_arms, edge)
            rows.append(
                {
                    "q0_pair": f"V{q0_pair[0]}-V{q0_pair[1]}",
                    "q0_arm_edge": f"A{q0_arms[0]}-A{q0_arms[1]}",
                    "q1_pair": f"V{left}-V{right}",
                    "q1_arm_edge": f"A{edge[0]}-A{edge[1]}",
                    "category": category,
                    "coarse_category": (
                        "full_reuse"
                        if category == "full_reuse"
                        else "one_arm_reuse"
                        if category == "one_arm_reuse"
                        else "full_switch"
                    ),
                    "candidate_count_in_orbit": 0,
                }
            )
    counts: defaultdict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["category"])] += 1
    for row in rows:
        row["candidate_count_in_orbit"] = counts[str(row["category"])]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--q0", default="V1,V12", help="representative q0 pair, e.g. V1,V12")
    parser.add_argument(
        "--output",
        default="results/b6_neighbor_orbit_table",
        help="output directory relative to project root",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    q0_pair = parse_pair(args.q0)
    arm_of = load_arm_map(root)
    rows = build_rows(arm_of, q0_pair)

    fieldnames = list(rows[0]) if rows else [
        "q0_pair", "q0_arm_edge", "q1_pair", "q1_arm_edge",
        "category", "candidate_count_in_orbit",
    ]
    with (output / "q1_neighbor_orbits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "model": "B6 neighbour-arm-only q1 orbit table",
        "q0_pair": f"V{q0_pair[0]}-V{q0_pair[1]}",
        "arm_cycle": list(ARM_CYCLE),
        "allowed_arm_edges": [list(edge) for edge in sorted(cycle_edges())],
        "q1_candidate_count": len(rows),
        "orbit_counts": {
            category: sum(1 for row in rows if row["category"] == category)
            for category in sorted({str(row["category"]) for row in rows})
        },
        "coarse_category_counts": {
            category: sum(1 for row in rows if row["coarse_category"] == category)
            for category in sorted({str(row["coarse_category"]) for row in rows})
        },
        "representative_q1_pairs": {
            category: next(
                row["q1_pair"]
                for row in rows
                if row["category"] == category
            )
            for category in sorted({str(row["category"]) for row in rows})
        },
        "warning": (
            "Counts are conditional on the fixed q0 endpoint pair. A new MD/umbrella "
            "path is needed only for each non-equivalent orbit; counts are retained "
            "as degeneracy and must not be silently discarded."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# B6 neighbour-only q1 orbit table\n\n"
        "This is a bookkeeping/audit table, not a submitted MD calculation. "
        "Only adjacent arm edges are allowed; same-arm and non-neighbour cross "
        "pairs are excluded. Use one representative umbrella path per orbit and "
        "retain `candidate_count_in_orbit` in the conditional probability.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
