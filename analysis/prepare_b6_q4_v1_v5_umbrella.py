#!/usr/bin/env python3
"""Prepare B6 q4 V1-V5 one-dimensional distance umbrella inputs."""
from __future__ import annotations
import argparse, csv, json, math, shutil, subprocess
from pathlib import Path

MIN_NM, MAX_NM, SPACING_NM = 0.35, 2.75, 0.10
K_NM2, DT_PS, TEMP_K = 500.0, 0.002, 313.15
EQUIL_PS, PROD_PS, PULL_RATE = 100.0, 500.0, 0.0005
STATE = "B6_p_C6_q4_V3_V6_V11_V12_V8_V10_V7_V9_cis"

def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def centers():
    n = int(round((MAX_NM - MIN_NM) / SPACING_NM))
    return [MIN_NM + i * SPACING_NM for i in range(n + 1)]

def extract_start(gmx, tpr, xtc, index, time_ps, out):
    cmd = [gmx, "trjconv", "-s", str(tpr), "-f", str(xtc), "-n", str(index),
           "-o", str(out), "-dump", f"{time_ps:.6f}", "-pbc", "whole"]
    result = subprocess.run(cmd, input="Solute\n", text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode != 0 or not out.is_file() or out.stat().st_size == 0:
        raise RuntimeError(f"trjconv failed ({result.returncode}):\n{result.stdout[-4000:]}")

def merge_solute_into_full_gro(solute_path, full_path, output_path, solute_count):
    solute = solute_path.read_text(encoding="utf-8").splitlines()
    full = full_path.read_text(encoding="utf-8").splitlines()
    if int(solute[1].strip()) != solute_count:
        raise ValueError("extracted solute GRO atom count mismatch")
    if int(full[1].strip()) <= solute_count:
        raise ValueError("full-system GRO does not contain solvent atoms")
    merged = list(full)
    for index in range(solute_count):
        source = solute[2 + index]
        target = full[2 + index]
        merged[2 + index] = target[:20] + source[20:44] + target[44:]
    output_path.write_text("\n".join(merged) + "\n", encoding="utf-8")

def base_mdp(nsteps, gen_vel, xtc_stride, xtc_group="Solute"):
    return "\n".join([
        "integrator = md", f"dt = {DT_PS:g}", f"nsteps = {nsteps}",
        "continuation = no", "constraint-algorithm = lincs", "constraints = h-bonds",
        "lincs-iter = 1", "lincs-order = 4", "cutoff-scheme = Verlet",
        "nstlist = 20", "rlist = 1.2", "coulombtype = PME", "rcoulomb = 1.2",
        "pme-order = 4", "fourierspacing = 0.12", "vdwtype = Cut-off",
        "rvdw = 1.2", "DispCorr = EnerPres", "tcoupl = V-rescale",
        "tc-grps = System", "tau-t = 1.0", f"ref-t = {TEMP_K:g}",
        "pcoupl = C-rescale", "pcoupltype = isotropic", "tau-p = 5.0",
        "ref-p = 1.0", "compressibility = 1.0e-4",
        f"gen-vel = {'yes' if gen_vel else 'no'}", f"gen-temp = {TEMP_K:g}",
        "gen-seed = -1", "nstxout = 0", "nstvout = 0", "nstfout = 0",
        f"nstxout-compressed = {xtc_stride}", f"compressed-x-grps = {xtc_group}",
        "compressed-x-precision = 1000", "nstenergy = 500", "nstlog = 5000",
        "nstcalcenergy = 100", "pbc = xyz", "comm-mode = Linear", "nstcomm = 100", ""
    ])

def pull_block(center=None, rate=0.0):
    start = "yes" if center is None else "no"
    init = 0.0 if center is None else center
    return "\n".join([
        "pull = yes", "pull-ncoords = 1", "pull-ngroups = 2",
        "pull-group1-name = V1_midpoint", "pull-group2-name = V5_midpoint",
        "pull-group1-pbcatom = 187", "pull-group2-pbcatom = 194",
        "pull-pbc-ref-prev-step-com = yes", "pull-coord1-type = umbrella",
        "pull-coord1-geometry = distance", "pull-coord1-groups = 1 2",
        "pull-coord1-dim = Y Y Y", f"pull-coord1-start = {start}",
        f"pull-coord1-init = {init:.6f}", f"pull-coord1-rate = {rate:.7f}",
        f"pull-coord1-k = {K_NM2:.1f}", "pull-nstxout = 250", "pull-nstfout = 250", ""
    ])

def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=project / "bohrium_download/b6_q4_md_pilot_20260922_fix1/20764906/results/workspace")
    parser.add_argument("--output", type=Path, default=project / "umbrella/b6_q4_v1_v5_distance")
    parser.add_argument("--gmx", default="gmx")
    parser.add_argument("--replica", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_root = args.source_root.resolve() / "gromacs_dcm/runs" / STATE
    setup = run_root / "setup"
    replica_setup = setup / "initial_conformers" / f"replica_{args.replica:02d}"
    replica = run_root / f"replica_{args.replica:02d}"
    rows = read_rows(replica / "vinyl_distances.csv")
    best = min(rows, key=lambda row: float(row["V1_V5_跨臂"]))
    time_ps, distance_A = float(best["time_ps"]), float(best["V1_V5_跨臂"])
    if args.dry_run:
        print(json.dumps({"state": STATE, "replica": args.replica, "frame": int(best["frame"]),
                          "time_ps": time_ps, "distance_A": distance_A,
                          "window_count": len(centers()), "centers_nm": centers()}, indent=2))
        return
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    solute_start = output / "solute_start.gro"
    extract_start(args.gmx, replica / "production_2ns.tpr", replica / "production_2ns.xtc",
                  replica_setup / "index.ndx", time_ps, solute_start)
    merge_solute_into_full_gro(solute_start, replica / "production_2ns.gro", output / "start.gro", 420)
    shutil.copy2(replica_setup / "system.top", output / "system.top")
    metadata = json.loads((setup / "metadata.json").read_text(encoding="utf-8"))
    mapping = {k: [int(v) for v in vals] for k, vals in metadata["vinyl_mapping_gromacs_1based"].items()}
    index_text = (replica_setup / "index.ndx").read_text(encoding="utf-8").rstrip() + "\n\n"
    index_text += "[ V1_midpoint ]\n" + " ".join(map(str, mapping["V1"])) + "\n\n"
    index_text += "[ V5_midpoint ]\n" + " ".join(map(str, mapping["V5"])) + "\n"
    (output / "index.ndx").write_text(index_text, encoding="utf-8")
    endpoint = rows[-1]
    endpoint_time_ps = float(endpoint["time_ps"])
    endpoint_distance_A = float(endpoint["V1_V5_跨臂"])
    shutil.copy2(replica / "production_2ns.gro", output / "start.gro")
    shutil.copy2(replica / "production_2ns.cpt", output / "start.cpt")
    initial_nm = endpoint_distance_A / 10.0
    for direction, target, rate in (("in", MIN_NM - 0.10, -PULL_RATE), ("out", MAX_NM + 0.10, PULL_RATE)):
        duration_ps = abs(target - initial_nm) / PULL_RATE + 300.0
        steps = math.ceil(duration_ps / DT_PS)
        (output / f"steer_{direction}.mdp").write_text(
            base_mdp(steps, False, 2500, "System") + pull_block(None, rate), encoding="utf-8")
    window_dir = output / "window_mdps"
    window_dir.mkdir(exist_ok=True)
    window_rows = []
    for i, center in enumerate(centers()):
        name = f"window_{i:03d}_{center:.3f}nm"
        (window_dir / f"{name}_equil.mdp").write_text(
            base_mdp(round(EQUIL_PS / DT_PS), True, 0) + pull_block(center), encoding="utf-8")
        (window_dir / f"{name}_production.mdp").write_text(
            base_mdp(round(PROD_PS / DT_PS), False, 250) + pull_block(center), encoding="utf-8")
        window_rows.append({"window_index": i, "window_name": name, "center_nm": center})
    with (output / "windows.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("window_index", "window_name", "center_nm"))
        writer.writeheader(); writer.writerows(window_rows)
    pathway = {
        "analysis": "B6 q4 V1-V5 one-dimensional distance umbrella pilot",
        "state": STATE, "pair": "V1_V5_跨臂", "left": 1, "right": 5, "topology": "cross_arm", "source_replica": args.replica,
        "pilot_closest_frame": int(best["frame"]), "pilot_closest_time_ps": time_ps,
        "pilot_closest_distance_A": distance_A, "starting_point": "replica production_2ns endpoint with matched solvent",
        "source_frame": int(endpoint["frame"]), "source_time_ps": endpoint_time_ps,
        "source_distance_A": endpoint_distance_A, "minimum_nm": MIN_NM, "maximum_nm": MAX_NM,
        "window_spacing_nm": SPACING_NM, "window_count": len(window_rows),
        "force_constant_kJ_mol_nm2": K_NM2, "temperature_K": TEMP_K,
        "equilibration_ps_per_window": EQUIL_PS, "production_ps_per_window": PROD_PS,
        "wham_begin_ps": EQUIL_PS, "wham_end_ps": PROD_PS,
        "gromacs_image": "registry.dp.tech/dptech/gromacs:2024.2",
        "primary_method": "1D distance umbrella; orientation is a post-analysis diagnostic"
    }
    (output / "pathway.json").write_text(json.dumps(pathway, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"pathways": [str(output)], "runner": "analysis/run_a4_m_c8_umbrella_path.py",
                "recommended_command": f"python3 analysis/run_a4_m_c8_umbrella_path.py --pathway {output} --ntomp 4 --gpu --stage all"}
    (output / "submission_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "source_time_ps": time_ps,
                      "source_distance_A": distance_A, "window_count": len(window_rows)}, indent=2))

if __name__ == "__main__":
    main()
