#!/usr/bin/env python3
"""Extract orientation-screened q2 QM candidates from the closest umbrella windows."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from analyze_explicit_dcm_q0_geometry_gate import soft_lower, soft_upper


PATHWAY_LOCATIONS = (
    "results/a4_q2_adjacent_umbrella_bohrium/download/c5_adj_cross_v2_v1/pathway",
    "results/a4_q2_adjacent_umbrella_bohrium/download/c5_adj_same_v2_v3/pathway",
    "results/a4_q2_adjacent_umbrella_bohrium/download/c8_adj_cross_v1_v4/pathway",
    "results/a4_q2_adjacent_umbrella_bohrium/download/c8_adj_same_v1_v2/pathway",
    "results/a4_q2_umbrella_bohrium/download/c5_reuse_v1_v6/pathway",
    "results/a4_q2_umbrella_bohrium/download/c5_reuse_v2_v4/pathway",
    "results/a4_q2_umbrella_bohrium/download/c5_switch_v4_v6/pathway",
    "results/a4_q2_umbrella_bohrium/partial/c8_switch_v4_v5/pathway",
    "results/a4_q2_umbrella_bohrium/resumed_download/c8_reuse_v5_v7/pathway",
)


def parse_ndx(path: Path) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    current: str | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].strip()
            groups[current] = []
        elif current is not None:
            groups[current].extend(int(value) for value in stripped.split())
    return groups


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("no candidate rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def extract_solute_gro(full_gro: Path, solute_serials: Sequence[int], output: Path) -> None:
    lines = full_gro.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[1].strip())
    atom_lines = lines[2:2 + atom_count]
    if max(solute_serials) > atom_count:
        raise ValueError(f"solute index exceeds atom count in {full_gro}")
    selected = [atom_lines[serial - 1] for serial in solute_serials]
    if [int(line[15:20]) for line in selected] != list(range(1, len(selected) + 1)):
        raise ValueError(f"solute atoms are not contiguous/one-based in {full_gro}")
    output.write_text(
        f"Solute topology extracted from {full_gro.name}\n{len(selected):5d}\n"
        + "\n".join(selected) + "\n" + lines[2 + atom_count] + "\n",
        encoding="utf-8",
    )


def run(command: Sequence[str], *, stdin: str | None = None) -> str:
    result = subprocess.run(
        list(command), input=stdin, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-35:])
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}\n{tail}")
    return result.stdout


def read_xvg(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and stripped[0] not in "#@&":
            rows.append([float(value) for value in stripped.split()])
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != 13:
        raise ValueError(f"expected time + four XYZ coordinates in {path}, got {values.shape}")
    return values


def read_numeric_xvg(path: Path) -> np.ndarray:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped and stripped[0] not in "#@&":
            rows.append([float(value) for value in stripped.split()])
    return np.asarray(rows, dtype=float)


def minimum_image_orthorhombic(vectors: np.ndarray, lengths_A: np.ndarray) -> np.ndarray:
    return vectors - lengths_A * np.round(vectors / lengths_A)


def geometry(
    values: np.ndarray, box_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates_A = values[:, 1:].reshape(-1, 4, 3) * 10.0
    if box_values.shape != (len(values), 7) or not np.allclose(values[:, 0], box_values[:, 0]):
        raise ValueError(f"coordinate/box frame mismatch: {values.shape}, {box_values.shape}")
    if np.max(np.abs(box_values[:, 4:])) > 1.0e-6:
        raise ValueError("triclinic umbrella boxes are not supported")
    box_A = box_values[:, 1:4] * 10.0
    left_vector = minimum_image_orthorhombic(coordinates_A[:, 1] - coordinates_A[:, 0], box_A)
    right_vector = minimum_image_orthorhombic(coordinates_A[:, 3] - coordinates_A[:, 2], box_A)
    left_length = np.linalg.norm(left_vector, axis=1)
    right_length = np.linalg.norm(right_vector, axis=1)
    physical = (
        (left_length >= 1.15) & (left_length <= 1.55)
        & (right_length >= 1.15) & (right_length <= 1.55)
    )
    # A subset of compressed coordinates can straddle a periodic boundary.
    # Exclude those frames rather than attempting bond reconstruction without
    # the full trajectory topology; the selected frame must retain both vinyl
    # bonds in the physical 1.15--1.55 A range.
    if np.mean(physical) < 0.50:
        raise ValueError(f"only {np.mean(physical):.3f} frames retain physical vinyl bonds")
    left_axis = left_vector / left_length[:, None]
    right_axis = right_vector / right_length[:, None]
    left_center = 0.5 * (coordinates_A[:, 0] + coordinates_A[:, 1])
    right_center = 0.5 * (coordinates_A[:, 2] + coordinates_A[:, 3])
    connector = minimum_image_orthorhombic(right_center - left_center, box_A)
    distance = np.linalg.norm(connector, axis=1)
    connector_axis = connector / distance[:, None]
    axis_angle = np.degrees(np.arccos(np.clip(np.abs(np.sum(left_axis * right_axis, axis=1)), 0, 1)))
    side_left = np.degrees(np.arccos(np.clip(np.abs(np.sum(left_axis * connector_axis, axis=1)), 0, 1)))
    side_right = np.degrees(np.arccos(np.clip(np.abs(np.sum(right_axis * connector_axis, axis=1)), 0, 1)))
    side_angle = np.minimum(side_left, side_right)
    score = soft_upper(distance, 5.0, 0.5) * soft_upper(axis_angle, 45.0, 7.5) * soft_lower(
        side_angle, 45.0, 7.5
    )
    score[~physical] = -1.0
    return distance, axis_angle, side_angle, score, physical


def element_from_atom_name(name: str) -> str:
    letters = "".join(character for character in name if character.isalpha()).upper()
    if letters.startswith(("CL", "BR", "SI", "NA", "MG", "AL", "CA", "FE", "ZN")):
        return letters[:2].title()
    if not letters:
        return "X"
    return letters[0]


def gro_to_xyz(gro: Path, xyz: Path, comment: str) -> None:
    lines = gro.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[1].strip())
    atoms = []
    for line in lines[2:2 + atom_count]:
        atom_name = line[10:15].strip()
        x, y, z = (float(line[20:28]) * 10.0, float(line[28:36]) * 10.0, float(line[36:44]) * 10.0)
        atoms.append(f"{element_from_atom_name(atom_name):<2s} {x:14.8f} {y:14.8f} {z:14.8f}")
    xyz.write_text(f"{atom_count}\n{comment}\n" + "\n".join(atoms) + "\n", encoding="utf-8")


def parse_first_molecule_bonds(topology: Path) -> list[tuple[int, int]]:
    section = ""
    molecule_seen = False
    bonds: list[tuple[int, int]] = []
    for raw in topology.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip().lower()
            if section == "moleculetype":
                if molecule_seen and bonds:
                    break
                molecule_seen = True
            continue
        if molecule_seen and section == "bonds":
            fields = line.split()
            if len(fields) >= 2:
                bonds.append((int(fields[0]) - 1, int(fields[1]) - 1))
    if not bonds:
        raise ValueError(f"no first-molecule bonds found in {topology}")
    return bonds


def unwrap_solute_gro(gro: Path, topology: Path) -> None:
    lines = gro.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[1].strip())
    atom_lines = lines[2:2 + atom_count]
    box = np.asarray([float(value) for value in lines[2 + atom_count].split()[:3]], dtype=float)
    wrapped = np.asarray([
        [float(line[20:28]), float(line[28:36]), float(line[36:44])] for line in atom_lines
    ])
    adjacency = [[] for _ in range(atom_count)]
    for left, right in parse_first_molecule_bonds(topology):
        if left < atom_count and right < atom_count:
            adjacency[left].append(right)
            adjacency[right].append(left)
    unwrapped = np.full_like(wrapped, np.nan)
    unwrapped[0] = wrapped[0]
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbor in adjacency[current]:
            if np.all(np.isfinite(unwrapped[neighbor])):
                continue
            displacement = wrapped[neighbor] - wrapped[current]
            displacement -= box * np.round(displacement / box)
            unwrapped[neighbor] = unwrapped[current] + displacement
            stack.append(neighbor)
    if np.any(~np.isfinite(unwrapped)):
        missing = int(np.sum(~np.isfinite(unwrapped[:, 0])))
        raise ValueError(f"solute bond graph is disconnected: {missing} atoms not reconstructed")
    unwrapped += 0.5 * box - np.mean(unwrapped, axis=0)
    rebuilt = [
        line[:20] + f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}" + line[44:]
        for line, xyz in zip(atom_lines, unwrapped)
    ]
    gro.write_text("\n".join(lines[:2] + rebuilt + [lines[2 + atom_count]]) + "\n", encoding="utf-8")


def gro_to_pdb(gro: Path, pdb: Path, title: str) -> None:
    lines = gro.read_text(encoding="utf-8").splitlines()
    atom_count = int(lines[1].strip())
    records = [f"TITLE     {title}"]
    for index, line in enumerate(lines[2:2 + atom_count], start=1):
        residue_number = int(line[:5])
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        x, y, z = (float(line[20:28]) * 10.0, float(line[28:36]) * 10.0, float(line[36:44]) * 10.0)
        element = element_from_atom_name(atom_name)
        records.append(
            f"ATOM  {index:5d} {atom_name:<4s} {residue_name:>3s} A{residue_number:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2s}"
        )
    records.extend(("TER", "END"))
    pdb.write_text("\n".join(records) + "\n", encoding="utf-8")


def select_candidate(pathway: Path, output: Path, gmx: str) -> dict[str, object]:
    metadata = json.loads((pathway / "pathway.json").read_text(encoding="utf-8"))
    groups = parse_ndx(pathway / "index.ndx")
    left = int(metadata["left"])
    right = int(metadata["right"])
    left_group = f"V{left}_midpoint"
    right_group = f"V{right}_midpoint"
    pair_serials = groups[left_group] + groups[right_group]
    if len(pair_serials) != 4:
        raise ValueError(f"{pathway}: expected four reactive-pair atoms, got {pair_serials}")
    closest_windows = sorted((pathway / "windows").glob("window_000_*"))
    if len(closest_windows) != 1:
        raise ValueError(f"{pathway}: expected one window_000, got {closest_windows}")
    window = closest_windows[0]
    trajectory = window / "production.xtc"
    full_gro = window / "production.gro"
    if not trajectory.is_file() or not full_gro.is_file():
        raise FileNotFoundError(f"closest-window trajectory incomplete: {window}")

    candidate_dir = output / str(metadata["pathway"])
    candidate_dir.mkdir(parents=True, exist_ok=True)
    solute_gro = candidate_dir / "solute_topology.gro"
    extract_solute_gro(full_gro, groups["Solute"], solute_gro)
    pair_index = candidate_dir / "reactive_pair.ndx"
    pair_index.write_text("[ Reactive_pair ]\n" + " ".join(str(value) for value in pair_serials) + "\n", encoding="utf-8")
    coordinate_xvg = candidate_dir / "reactive_pair_coordinates_nm.xvg"
    box_xvg = candidate_dir / "box_vectors_nm.xvg"
    begin = float(metadata["wham_begin_ps"])
    end = float(metadata["wham_end_ps"])
    run([
        gmx, "traj", "-f", str(trajectory), "-s", str(solute_gro), "-n", str(pair_index),
        "-ox", str(coordinate_xvg), "-ob", str(box_xvg),
        "-b", str(begin), "-e", str(end), "-x", "-y", "-z",
    ], stdin="Reactive_pair\n")
    values = read_xvg(coordinate_xvg)
    box_values = read_numeric_xvg(box_xvg)
    distance, axis, side, score, physical = geometry(values, box_values)
    chosen = int(np.argmax(score))
    time_ps = float(values[chosen, 0])
    stem = str(metadata["pathway"])
    candidate_gro = candidate_dir / f"{stem}_qm_candidate.gro"
    candidate_pdb = candidate_dir / f"{stem}_qm_candidate.pdb"
    run([
        gmx, "trjconv", "-f", str(trajectory), "-s", str(solute_gro),
        "-o", str(candidate_gro), "-dump", f"{time_ps:.6f}",
    ], stdin="System\n")
    unwrap_solute_gro(candidate_gro, pathway / "system.top")
    gro_to_pdb(candidate_gro, candidate_pdb, f"{stem}; t={time_ps:.1f} ps")
    candidate_xyz = candidate_dir / f"{stem}_qm_candidate.xyz"
    gro_to_xyz(candidate_gro, candidate_xyz, f"{stem}; umbrella 0.35 nm window; t={time_ps:.1f} ps")
    return {
        "pathway": metadata["pathway"],
        "state": metadata["state"],
        "system": metadata["system"],
        "branch": metadata.get("background_relationship", "opposite"),
        "topology": metadata.get("topology", "cross_arm"),
        "route_family": metadata["route_family"],
        "left_vinyl": f"V{left}",
        "right_vinyl": f"V{right}",
        "source_replica": metadata["replica"],
        "source_window": str(window.relative_to(pathway.parents[4])),
        "analysis_begin_ps": begin,
        "analysis_end_ps": end,
        "selected_time_ps": time_ps,
        "midpoint_distance_A": float(distance[chosen]),
        "axis_angle_deg": float(axis[chosen]),
        "side_angle_deg": float(side[chosen]),
        "soft_nac": float(score[chosen]),
        "physical_coordinate_fraction": float(np.mean(physical)),
        "strict_gate_pass": int(distance[chosen] <= 5.0 and axis[chosen] <= 45.0 and side[chosen] >= 45.0),
        "solute_atoms": len(groups["Solute"]),
        "candidate_gro": str(candidate_gro),
        "candidate_pdb": str(candidate_pdb),
        "candidate_xyz": str(candidate_xyz),
        "qm_use_note": "starting geometry only; add catalyst and locate the chemical transition state",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("results/a4_q2_qm_candidates"))
    parser.add_argument("--gmx", default="gmx")
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    output = args.output if args.output.is_absolute() else workspace / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = [select_candidate(workspace / location, output, args.gmx) for location in PATHWAY_LOCATIONS]
    write_csv(output / "qm_candidate_manifest.csv", rows)
    summary = {
        "candidate_count": len(rows),
        "strict_gate_pass": sum(int(row["strict_gate_pass"]) for row in rows),
        "selection": "maximum soft NAC in the analysed portion of the 0.35 nm umbrella window",
        "scope": "solute-only starting structures; no catalyst; not transition states",
    }
    (output / "README.md").write_text(
        "# q2 QM candidate structures\n\n"
        "These solute-only structures were selected from the analysed portion of each 0.35 nm umbrella window by maximizing the same soft NAC orientation score used in the q2 analysis. Periodic-boundary reconstruction used the bonded solute graph from each pathway topology. They are starting geometries, not transition states. Add the metathesis catalyst consistently before comparing chemical barriers.\n\n"
        "Because the frames come from distance-biased windows, do not use their raw energies or occurrence frequencies to infer equilibrium conformational selectivity. Their purpose is to provide comparable near-reactive geometries for subsequent catalyst-aware chemical-barrier calculations.\n\n"
        "See `qm_candidate_manifest.csv` for the selected time, reactive vinyl pair, geometry, and source pathway.\n",
        encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"output={output}")


if __name__ == "__main__":
    main()
