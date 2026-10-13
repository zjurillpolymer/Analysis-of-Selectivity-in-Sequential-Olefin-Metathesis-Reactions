#!/usr/bin/env python3
"""Validate representative solute snapshots against the 43-distance XTC CVs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / "results/b6_q1_candidate_fingerprint_clusters_20261003/snapshots_full_system"
INPUT = ROOT / "b6_q1_candidate_equilibration_results_20261003/analysis_basin_persistence"
OUT = ROOT / "results/b6_q1_candidate_fingerprint_clusters_20261003/representative_snapshot_geometry.csv"
CANDIDATES = {
    "A6_00690": "A6_origin_00690ps_high_novelty_positive_pc1",
    "A6_01270": "A6_origin_01270ps_high_novelty_crossing_frame",
    "A2_05480": "A2_origin_05480ps_positive_pc1_return",
}


def read_gro(path: Path) -> tuple[np.ndarray, tuple[float, ...]]:
    lines = path.read_text().splitlines()
    natoms = int(lines[1].strip())
    if natoms != 438:
        raise ValueError(f"Expected solute-only 438-atom snapshot, got {natoms}: {path}")
    xyz = np.array([[float(line[20:28]), float(line[28:36]), float(line[36:44])]
                    for line in lines[2:2 + natoms]])
    box = tuple(float(v) for v in lines[2 + natoms].split())
    return xyz, box


def gro_heavy_coordinates(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    natoms = int(lines[1].strip())
    return np.array([[float(line[20:28]), float(line[28:36]), float(line[36:44])]
                     for line in lines[2:2 + natoms] if not line[10:15].strip().startswith("H")])


def aligned_rmsd(reference: np.ndarray, mobile: np.ndarray) -> float:
    p = reference - reference.mean(axis=0)
    q = mobile - mobile.mean(axis=0)
    u, _, vh = np.linalg.svd(q.T @ p)
    fix = np.eye(3)
    fix[2, 2] = np.linalg.det(u @ vh)
    rotation = u @ fix @ vh
    aligned = q @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - p) ** 2, axis=1))))


def vinyl_center_distance(xyz: np.ndarray, left: int, right: int) -> float:
    left_atoms = xyz[189 + 2 * (left - 3) - 1:189 + 2 * (left - 3) + 1]
    right_atoms = xyz[189 + 2 * (right - 3) - 1:189 + 2 * (right - 3) + 1]
    return float(np.linalg.norm(left_atoms.mean(axis=0) - right_atoms.mean(axis=0)))


def main() -> None:
    cvdef = json.loads((ROOT / "b6_q1_contact_pc1_wtmetad_server_20261002/cv_definition.json").read_text())
    features = cvdef["features"]
    rows = []
    structures = {}
    pbc_rows = []
    selections = []
    for feature in features:
        left, right = map(int, re.match(r"V(\d+)-V(\d+)_nm", feature).groups())
        la, ra = 189 + 2 * (left - 3), 189 + 2 * (right - 3)
        selections.append(f"com of (atomnr {la} or atomnr {la+1}) plus com of (atomnr {ra} or atomnr {ra+1})")
    selection_arg = "; ".join(selections)
    for short, candidate in CANDIDATES.items():
        traj = np.loadtxt(INPUT / f"{candidate}_43_pair_distances.xvg")
        full_dir = ROOT / "b6_q1_candidate_equilibration_results_20261003" / candidate
        prefix = "npt" if short == "A6_00690" else "npt_full"
        no_pbc_path = args_path = ROOT / "results/b6_q1_candidate_fingerprint_clusters_20261003" / f"{short}_43_pair_distances_no_pbc.xvg"
        command = ["gmx", "distance", "-f", str(full_dir / f"{prefix}.xtc"), "-s", str(full_dir / f"{prefix}.tpr"),
                   "-select", selection_arg, "-oall", str(no_pbc_path), "-xvg", "none", "-nopbc"]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        no_pbc = np.loadtxt(no_pbc_path)
        if no_pbc.shape != traj.shape:
            raise ValueError(f"PBC/no-PBC shape mismatch for {short}: {traj.shape} vs {no_pbc.shape}")
        delta_full = np.abs(no_pbc[:, 1:] - traj[:, 1:])
        for idx, feature in enumerate(features):
            pbc_rows.append({"candidate": short, "pair": feature,
                             "frames_over_0p005nm": int(np.sum(delta_full[:, idx] > .005)),
                             "fraction_over_0p005nm": float(np.mean(delta_full[:, idx] > .005)),
                             "maximum_pbc_shift_nm": float(delta_full[:, idx].max())})
        for gro in sorted(SNAP.glob(f"{short}_*ps_solute.gro")):
            match = re.search(r"_(\d+)ps_solute\.gro$", gro.name)
            time_ps = float(match.group(1))
            row = int(np.argmin(np.abs(traj[:, 0] - time_ps)))
            xyz, box = read_gro(gro)
            structures[gro.stem] = gro_heavy_coordinates(gro)
            got = []
            for feature in features:
                a, b = map(int, re.match(r"V(\d+)-V(\d+)_nm", feature).groups())
                got.append(vinyl_center_distance(xyz, a, b))
            got = np.asarray(got)
            expected = traj[row, 1:]
            delta = np.abs(got - expected)
            wrapped = [features[i] for i in np.flatnonzero(delta > 0.005)]
            rg = float(np.sqrt(np.mean(np.sum((xyz - xyz.mean(axis=0)) ** 2, axis=1))))
            rows.append({"candidate": short, "time_ps": time_ps, "atom_count": len(xyz),
                         "box_nm": " ".join(f"{x:.5f}" for x in box),
                         "solute_rg_nm": rg, "max_43_distance_abs_error_nm": float(delta.max()),
                         "max_error_pair": features[int(np.argmax(delta))],
                         "pairs_with_pbc_image_shift": len(wrapped),
                         "pbc_shifted_pairs": ";".join(wrapped),
                         "rms_43_distance_abs_error_nm": float(np.sqrt(np.mean(delta ** 2)))})
    pd.DataFrame(rows).to_csv(OUT, index=False)
    rmsd_rows = []
    for i, name_a in enumerate(structures):
        for name_b in list(structures)[i + 1:]:
            rmsd_rows.append({"structure_a": name_a, "structure_b": name_b,
                              "heavy_atom_aligned_rmsd_nm": aligned_rmsd(structures[name_a], structures[name_b])})
    rmsd_out = OUT.with_name("representative_heavy_atom_rmsd.csv")
    pd.DataFrame(rmsd_rows).to_csv(rmsd_out, index=False)
    pbc_out = OUT.with_name("trajectory_pbc_image_audit.csv")
    pd.DataFrame(pbc_rows).to_csv(pbc_out, index=False)
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nPBC image-shift frequency (>0.005 nm difference from whole-molecule non-PBC distance):")
    summary = pd.DataFrame(pbc_rows).groupby("candidate").agg(
        pair_features_ever_shifted=("frames_over_0p005nm", lambda x: int((x > 0).sum())),
        max_fraction_for_any_pair=("fraction_over_0p005nm", "max"),
        total_frame_pair_shifts=("frames_over_0p005nm", "sum"),
        max_shift_nm=("maximum_pbc_shift_nm", "max"))
    print(summary.to_string(float_format=lambda x: f"{x:.5f}"))
    print(f"Wrote PBC audit {pbc_out}")
    print(f"Wrote representative heavy-atom RMSD matrix pairs {rmsd_out}")
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
