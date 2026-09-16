#!/usr/bin/env python3
"""Analyze A4-p pilot XTC files with PBC-aware contact/orientation NAC gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

R_KCAL = 0.00198720425864083
TEMPERATURE_K = 313.15
RT = R_KCAL * TEMPERATURE_K


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def parse_pair(text: str) -> tuple[str, str]:
    fields = text.replace("_", "-").split("-")
    if len(fields) != 2:
        raise ValueError(f"invalid pair: {text}")
    return tuple(sorted((fields[0].replace("V", ""), fields[1].replace("V", ""))))  # type: ignore[return-value]


def load_workflow(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {(row["state"], f"{row['left']}-{row['right']}"): row for row in csv.DictReader(handle)}


def minimum_image(delta: np.ndarray, box: np.ndarray) -> np.ndarray:
    return delta - box * np.round(delta / box)


def soft_upper(values: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip((values - center) / width, -60.0, 60.0)))


def soft_lower(values: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(np.clip((center - values) / width, -60.0, 60.0)))


def analyze_replica(
    topology: Path,
    xtc: Path,
    mapping: dict[str, list[int]],
    annotations: list[dict[str, str]],
) -> list[dict[str, object]]:
    import MDAnalysis as mda
    from MDAnalysis.transformations import unwrap

    universe = mda.Universe(str(topology), str(xtc))
    universe.trajectory.add_transformations(unwrap(universe.atoms))
    labels = sorted(mapping, key=lambda value: int(value.removeprefix("V")))
    label_index = {label: index for index, label in enumerate(labels)}
    atom_pairs = [[int(value) - 1 for value in mapping[label]] for label in labels]
    frame_records: dict[str, list[float]] = defaultdict(list)
    distance_records: dict[str, list[float]] = defaultdict(list)
    axis_records: dict[str, list[float]] = defaultdict(list)
    side_records: dict[str, list[float]] = defaultdict(list)
    for timestep in universe.trajectory:
        positions = universe.atoms.positions.astype(float)
        box = np.asarray(timestep.dimensions[:3], dtype=float)
        first = np.asarray([positions[pair[0]] for pair in atom_pairs])
        second = np.asarray([positions[pair[1]] for pair in atom_pairs])
        vectors = minimum_image(second - first, box)
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any((lengths < 1.0) | (lengths > 2.0)):
            continue
        axes = vectors / lengths[:, None]
        centers = first + 0.5 * vectors
        for annotation in annotations:
            pair_name = str(annotation["pair"])
            left, right = parse_pair(pair_name)
            i, j = label_index[f"V{left}"], label_index[f"V{right}"]
            connector = minimum_image(centers[j] - centers[i], box)
            distance = float(np.linalg.norm(connector))
            if distance <= 1.0e-8:
                continue
            connector_axis = connector / distance
            axis_angle = math.degrees(math.acos(float(np.clip(abs(np.dot(axes[i], axes[j])), 0.0, 1.0))))
            side_angle = min(
                math.degrees(math.acos(float(np.clip(abs(np.dot(axes[i], connector_axis)), 0.0, 1.0)))),
                math.degrees(math.acos(float(np.clip(abs(np.dot(axes[j], connector_axis)), 0.0, 1.0)))),
            )
            contact = float(soft_upper(np.asarray(distance), 5.0, 0.5))
            orientation = float(soft_upper(np.asarray(axis_angle), 45.0, 7.5) * soft_lower(np.asarray(side_angle), 45.0, 7.5))
            key = pair_name
            distance_records[key].append(distance)
            axis_records[key].append(axis_angle)
            side_records[key].append(side_angle)
            frame_records[key].append(contact * orientation)
    output = []
    for annotation in annotations:
        key = str(annotation["pair"])
        nac = np.asarray(frame_records[key], dtype=float)
        distances = np.asarray(distance_records[key], dtype=float)
        axes = np.asarray(axis_records[key], dtype=float)
        sides = np.asarray(side_records[key], dtype=float)
        output.append({
            **annotation,
            "frames": int(len(nac)),
            "nac_probability": float(np.mean(nac)) if len(nac) else 0.0,
            "hard_contact_fraction": float(np.mean(distances <= 5.0)) if len(distances) else 0.0,
            "hard_nac_fraction": float(np.mean((distances <= 5.0) & (axes <= 45.0) & (sides >= 45.0))) if len(distances) else 0.0,
            "minimum_distance_A": float(np.min(distances)) if len(distances) else float("nan"),
            "median_distance_A": float(np.median(distances)) if len(distances) else float("nan"),
        })
    return output


def bootstrap_delta(values_a: list[float], values_b: list[float], samples: int, seed: int) -> tuple[float, float, float]:
    if not values_a or not values_b or min(values_a) <= 0.0 or min(values_b) <= 0.0:
        return float("nan"), float("nan"), float("nan")
    point = -RT * math.log(float(np.mean(values_a)) / float(np.mean(values_b)))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        a = rng.choice(values_a, size=len(values_a), replace=True)
        b = rng.choice(values_b, size=len(values_b), replace=True)
        if np.mean(a) > 0 and np.mean(b) > 0:
            draws.append(-RT * math.log(float(np.mean(a)) / float(np.mean(b))))
    return point, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-root", type=Path, default=project / "results/A4_p_C6_C8_unbiased_pilot_full_trajectories")
    parser.add_argument("--workflow", type=Path, default=project / "results/A4_p_C6_C8_selectivity_workflow/neighbour_pair_classes.csv")
    parser.add_argument("--output", type=Path, default=project / "results/A4_p_C6_C8_selectivity_xtc_nac")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    trajectory_root, output = args.trajectory_root.resolve(), args.output.resolve()
    workflow = load_workflow(args.workflow.resolve())
    manifest_path = trajectory_root / "gromacs_dcm/state_manifest.csv"
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    pair_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in manifest_rows:
        state = row["state"]
        state_root = trajectory_root / "gromacs_dcm" / "runs" / state
        metadata = json.loads((state_root / "setup/metadata.json").read_text(encoding="utf-8"))
        annotations = []
        for item in metadata["pair_schema"]:
            key = (state, f"{item['left']}-{item['right']}")
            if key not in workflow:
                continue
            annotations.append({
                "pair": f"V{item['left']}-V{item['right']}",
                "topology_class": workflow[key]["topology_class"],
                "route_class": workflow[key]["route_class"],
            })
        mapping = {str(key): [int(value) for value in values] for key, values in metadata["vinyl_mapping_gromacs_1based"].items()}
        topology = state_root / "setup/solute_parameters/slu.mol2"
        for replica in range(1, 4):
            xtc = state_root / f"replica_{replica:02d}/production_2ns.xtc"
            results = analyze_replica(topology, xtc, mapping, annotations)
            for result in results:
                record = {"state": state, "replica": replica, **result}
                pair_rows.append(record)
                # q1 and q2-adjacent contrasts are cross-vs-same; q2-opposite
                # is the path-memory contrast switch-vs-reuse.
                label = (
                    result["topology_class"]
                    if row["stage"] == "q1" or row["branch"] == "adjacent"
                    else result["route_class"]
                )
                if not label:
                    continue
                grouped[(state, row["branch"], str(label))].append(float(result["nac_probability"]))
    output.mkdir(parents=True, exist_ok=True)
    fields = ["state", "replica", "pair", "topology_class", "route_class", "frames", "nac_probability", "hard_contact_fraction", "hard_nac_fraction", "minimum_distance_A", "median_distance_A"]
    with (output / "pair_replica_nac.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(pair_rows)
    contrasts = []
    for state, branch in sorted({(key[0], key[1]) for key in grouped}):
        available = {key[2] for key in grouped if key[0] == state and key[1] == branch}
        for alternative, reference in (("cross", "same"), ("switch", "reuse")):
            if alternative not in available or reference not in available:
                continue
            point, low, high = bootstrap_delta(grouped[(state, branch, alternative)], grouped[(state, branch, reference)], args.bootstrap, stable_seed(state, branch, alternative, reference))
            contrasts.append({"state": state, "branch": branch, "alternative": alternative, "reference": reference, "delta_delta_g_kcal_mol": point, "ci_low": low, "ci_high": high, "probability_ratio": math.exp(-point / RT) if math.isfinite(point) else float("nan")})
    with (output / "class_free_energy_contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("state", "branch", "alternative", "reference", "delta_delta_g_kcal_mol", "ci_low", "ci_high", "probability_ratio")); writer.writeheader(); writer.writerows(contrasts)
    print(json.dumps({"states": len({row['state'] for row in pair_rows}), "pair_replica_rows": len(pair_rows), "contrasts": len(contrasts), "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
