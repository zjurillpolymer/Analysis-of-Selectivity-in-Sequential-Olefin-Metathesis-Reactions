#!/usr/bin/env python3
"""Diagnose candidate second CVs from completed B6 q1 paired 1D umbrellas.

This is a CV-screening diagnostic, not a new PMF estimator. It reconstructs
the whole solute, then measures the competing V2--V11 and V2--V3 distances and
solute Rg in every production window. It discards the first 100 ps of each
window and reports window-stratified descriptive statistics.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT / "bohrium_submit/b6_rest2_q1_umbrella_20261006/results"
SEEDS = (5000, 12500, 19500)
ROUTES = {
    "to_arm6_V11": {"name": "arm1_to_arm6", "target": "d6"},
    "to_arm2_V3": {"name": "arm1_to_arm2", "target": "d2"},
}
DISTANCE_SELECTION = (
    "com of (atomnr 207 or atomnr 208) plus com of (atomnr 205 or atomnr 206); "
    "com of (atomnr 207 or atomnr 208) plus com of (atomnr 189 or atomnr 190)"
)


def read_xvg(path: Path) -> list[list[float]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line or line[0] in "@#":
                continue
            rows.append([float(token) for token in line.split()])
    return rows


def gmx_run(gmx: str, args: list[str], cwd: Path, env: dict[str, str]) -> None:
    result = subprocess.run(
        [gmx, *args], cwd=cwd, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(f"GROMACS failed in {cwd}:\n{result.stdout[-5000:]}")


def analyze_window(args: tuple[str, Path, int, str, float, Path]) -> dict[str, object]:
    gmx, window, seed, route_key, center, output_tmp = args
    xtc, tpr = window / "production.xtc", window / "production.tpr"
    if not xtc.is_file() or not tpr.is_file():
        raise FileNotFoundError(f"missing production trajectory or TPR in {window}")
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    route = ROUTES[route_key]
    stem = f"s{seed}_{route_key}_{window.name}"
    distance_xvg = output_tmp / f"{stem}_dist.xvg"
    rg_xvg = output_tmp / f"{stem}_rg.xvg"
    gmx_run(gmx, [
        "distance", "-s", str(tpr), "-f", str(xtc), "-select", DISTANCE_SELECTION,
        "-oav", str(distance_xvg), "-xvg", "none", "-rmpbc", "yes", "-pbc", "no",
    ], window, env)
    gmx_run(gmx, [
        "gyrate", "-s", str(tpr), "-f", str(xtc), "-sel", "resname SLU",
        "-o", str(rg_xvg), "-xvg", "none",
    ], window, env)
    distances, gyration = read_xvg(distance_xvg), read_xvg(rg_xvg)
    rg_by_t = {round(row[0], 3): row[1] for row in gyration}
    frames = []
    for row in distances:
        time_ps = row[0]
        if time_ps < 100.0:
            continue
        rg = rg_by_t.get(round(time_ps, 3))
        if rg is None:
            continue
        frames.append((time_ps, row[1], row[2], rg))
    if not frames:
        raise RuntimeError(f"no aligned post-equilibration frames in {window}")
    col = 1 if route["target"] == "d6" else 2
    target = [frame[col] for frame in frames]
    competitor = [frame[2 if col == 1 else 1] for frame in frames]
    rgs = [frame[3] for frame in frames]

    def avg(values: list[float]) -> float:
        return sum(values) / len(values)

    def sd(values: list[float]) -> float:
        mean = avg(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    contacts = [frame for frame in frames if frame[col] < 0.65]
    far = [frame for frame in frames if 1.5 <= frame[col] <= 2.5]
    contact_rg = [frame[3] for frame in contacts]
    far_rg = [frame[3] for frame in far]
    joint_contact = sum(frame[2 if col == 1 else 1] < 0.65 for frame in contacts)
    return {
        "seed_ps": seed,
        "route": route["name"],
        "window_center_nm": center,
        "window": window.name,
        "frames_used": len(frames),
        "target_distance_mean_nm": avg(target),
        "target_distance_sd_nm": sd(target),
        "competitor_distance_mean_nm": avg(competitor),
        "competitor_distance_sd_nm": sd(competitor),
        "Rg_mean_nm": avg(rgs),
        "Rg_sd_nm": sd(rgs),
        "contact_frames_lt_0p65nm": len(contacts),
        "contact_Rg_mean_nm": avg(contact_rg) if contact_rg else "",
        "far_frames_1p5_to_2p5nm": len(far),
        "far_Rg_mean_nm": avg(far_rg) if far_rg else "",
        "fraction_competitor_contact_given_target_contact": (
            joint_contact / len(contacts) if contacts else ""
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/b6_q1_second_cv_diagnostics_20261006")
    parser.add_argument("--gmx", default="/opt/homebrew/bin/gmx")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)

    jobs = []
    for seed in SEEDS:
        seed_dir = root / f"seed_{seed:05d}"
        for route_key in ROUTES:
            route_dir = seed_dir / "pathways" / f"rest2_frame_{seed:05d}ps__{ROUTES[route_key]['name']}"
            # Directory names in this project identify the endpoints explicitly.
            if not route_dir.is_dir():
                endpoint = "arm6_V11" if route_key == "to_arm6_V11" else "arm2_V3"
                route_dir = seed_dir / "pathways" / f"rest2_frame_{seed:05d}ps__to_{endpoint}"
            for window in sorted((route_dir / "windows").glob("window_*")):
                match = re.search(r"window_\d+_([0-9.]+)nm", window.name)
                if not match:
                    raise ValueError(f"cannot read umbrella center from {window.name}")
                jobs.append((args.gmx, window, seed, route_key, float(match.group(1)), output))
    if len(jobs) != 150:
        raise RuntimeError(f"expected 150 completed windows, found {len(jobs)}")

    rows: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyze_window, job) for job in jobs]
        for number, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if number % 10 == 0 or number == len(futures):
                print(f"analyzed {number}/{len(futures)} windows", flush=True)
    rows.sort(key=lambda row: (int(row["seed_ps"]), str(row["route"]), float(row["window_center_nm"])))
    out_csv = output / "window_level_global_cv_diagnostics.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# B6 q1 second-CV screening diagnostics",
        "",
        "Computed from the completed 1D paired umbrella production trajectories. For each window, the first 100 ps were discarded; V2–V11 and V2–V3 distances were measured after reconstructing the solute as whole and disabling minimum-image distances; solute Rg was computed from the SLU residue.",
        "",
        "These are biased, window-stratified diagnostics, **not** equilibrium populations or 2D PMFs. The direct two-distance surface still requires its own joint sampling. Rg is tested only as a cheap candidate global CV.",
        "",
        "| Seed (ns) | Path | Rg mean across windows (nm) | Contact-window Rg mean (nm) | Far-window Rg mean (nm) | Target-contact frames where competitor also <0.65 nm |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        for route_name in ("arm1_to_arm6", "arm1_to_arm2"):
            group = [row for row in rows if row["seed_ps"] == seed and row["route"] == route_name]
            def weighted(key: str) -> float | None:
                valid = [(float(row[key]), int(row["contact_frames_lt_0p65nm"] if "contact" in key else row["far_frames_1p5_to_2p5nm"])) for row in group if row[key] != ""]
                denom = sum(weight for _, weight in valid)
                return sum(value * weight for value, weight in valid) / denom if denom else None
            joint_num = sum(
                float(row["fraction_competitor_contact_given_target_contact"]) * int(row["contact_frames_lt_0p65nm"])
                for row in group if row["fraction_competitor_contact_given_target_contact"] != ""
            )
            joint_den = sum(int(row["contact_frames_lt_0p65nm"]) for row in group)
            lines.append(
                f"| {seed/1000:.1f} | {route_name} | "
                f"{sum(float(row['Rg_mean_nm']) for row in group)/len(group):.3f} | "
                f"{weighted('contact_Rg_mean_nm') if weighted('contact_Rg_mean_nm') is not None else float('nan'):.3f} | "
                f"{weighted('far_Rg_mean_nm') if weighted('far_Rg_mean_nm') is not None else float('nan'):.3f} | "
                f"{joint_num/joint_den if joint_den else float('nan'):.3f} |"
            )
    lines.extend([
        "",
        "## Interpretation guardrails",
        "",
        "- The current sampling supports a first check of whether Rg changes across restrained-distance windows and whether both candidate sites approach simultaneously. Because the two channels were simulated in separate 1D umbrella sets, these data do not give a jointly sampled (d6,d2) free-energy surface.",
        "- Compare only like target-distance windows across paths and seeds. A difference in raw pooled Rg histograms can simply reflect different distance restraints.",
        "- If Rg distributions remain strongly overlapping despite opposite seed-specific route preferences, replace Rg with a slow descriptor built from solute internal distances/torsions (e.g. TICA/VAMP); do not assume Rg is sufficient.",
        "- The arm-2 multiplicity of two applies to route weighting, not to the geometric distance measurement in this diagnostic.",
        "",
        "Per-window measurements are in `window_level_global_cv_diagnostics.csv`.",
        "",
    ])
    report = output / "REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT={report}", flush=True)


if __name__ == "__main__":
    main()
