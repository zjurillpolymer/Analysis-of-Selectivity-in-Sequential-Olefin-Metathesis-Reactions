#!/usr/bin/env python3
"""Prepare matched distance-umbrella windows for selected A4-p paths.

Run ``prepare_a4_p_c6_c8_selectivity_manifest.py`` after the Amber setup and
pilot.  This tool deliberately requires an explicit JSON path specification;
it never silently chooses a path from product labels or from a single frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any


MINIMUM_NM = 0.35
WINDOW_SPACING_NM = 0.10
FORCE_CONSTANT = 500.0
STEERING_FORCE_CONSTANT = 1000.0
PULL_RATE_NM_PS = 0.0005
DT_PS = 0.002
EQUILIBRATION_PS = 100.0
PRODUCTION_PS = 500.0


def centers(maximum_nm: float) -> list[float]:
    count = int(round((maximum_nm - MINIMUM_NM) / WINDOW_SPACING_NM))
    values = [MINIMUM_NM + i * WINDOW_SPACING_NM for i in range(count + 1)]
    if not math.isclose(values[-1], maximum_nm, abs_tol=1e-9):
        raise ValueError("maximum_nm must lie on the 0.10 nm window grid")
    return values


def gro_geometry(path: Path) -> tuple[dict[int, tuple[float, float, float]], tuple[float, float, float]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    n_atoms = int(lines[1].strip())
    atoms = {
        int(line[15:20]): (float(line[20:28]), float(line[28:36]), float(line[36:44]))
        for line in lines[2 : 2 + n_atoms]
    }
    box = tuple(float(value) for value in lines[2 + n_atoms].split())
    if len(box) != 3:
        raise ValueError(f"only orthorhombic GRO boxes are supported: {path}")
    return atoms, box  # type: ignore[return-value]


def minimum_image(delta: float, box: float) -> float:
    return delta - box * round(delta / box)


def midpoint(atoms: dict[int, tuple[float, float, float]], pair: list[int], box: tuple[float, float, float]) -> tuple[float, float, float]:
    a, b = atoms[pair[0]], atoms[pair[1]]
    return tuple((a[k] + 0.5 * minimum_image(b[k] - a[k], box[k])) % box[k] for k in range(3))


def pair_distance_nm(gro: Path, mapping: dict[str, list[int]], left: int, right: int) -> tuple[float, float]:
    atoms, box = gro_geometry(gro)
    a = midpoint(atoms, mapping[f"V{left}"], box)
    b = midpoint(atoms, mapping[f"V{right}"], box)
    distance = math.sqrt(sum(minimum_image(b[k] - a[k], box[k]) ** 2 for k in range(3)))
    return distance, min(box) / 2.0


def common_mdp(nsteps: int, continuation: bool, gen_vel: bool, xtc_group: str, stride: int) -> str:
    return f"""integrator = md
dt = {DT_PS:g}
nsteps = {nsteps}
continuation = {'yes' if continuation else 'no'}
constraint-algorithm = lincs
constraints = h-bonds
lincs-iter = 1
lincs-order = 4
cutoff-scheme = Verlet
nstlist = 20
rlist = 1.2
coulombtype = PME
rcoulomb = 1.2
pme-order = 4
fourierspacing = 0.12
vdwtype = Cut-off
rvdw = 1.2
DispCorr = EnerPres
tcoupl = V-rescale
tc-grps = System
tau-t = 1.0
ref-t = 313.15
pcoupl = C-rescale
pcoupltype = isotropic
tau-p = 5.0
ref-p = 1.0
compressibility = 1.0e-4
gen-vel = {'yes' if gen_vel else 'no'}
gen-temp = 313.15
gen-seed = -1
nstxout = 0
nstvout = 0
nstfout = 0
nstxout-compressed = {stride}
compressed-x-grps = {xtc_group}
compressed-x-precision = 1000
nstenergy = 500
nstlog = 5000
nstcalcenergy = 100
pbc = xyz
comm-mode = Linear
nstcomm = 100
"""


def pull_block(left: int, right: int, mapping: dict[str, list[int]], center: float | None, rate: float = 0.0, force: float = FORCE_CONSTANT) -> str:
    return f"""pull = yes
pull-ncoords = 1
pull-ngroups = 2
pull-group1-name = V{left}_midpoint
pull-group2-name = V{right}_midpoint
pull-group1-pbcatom = {mapping[f'V{left}'][0]}
pull-group2-pbcatom = {mapping[f'V{right}'][0]}
pull-pbc-ref-prev-step-com = yes
pull-coord1-type = umbrella
pull-coord1-geometry = distance
pull-coord1-groups = 1 2
pull-coord1-dim = Y Y Y
pull-coord1-start = {'yes' if center is None else 'no'}
pull-coord1-init = {0.0 if center is None else center:.6f}
pull-coord1-rate = {rate:.7f}
pull-coord1-k = {force:.1f}
pull-nstxout = 250
pull-nstfout = 250
"""


def append_pair_groups(text: str, left: int, right: int, mapping: dict[str, list[int]]) -> str:
    # A reused index may already contain these groups; replace rather than duplicate.
    targets = {f"V{left}_midpoint", f"V{right}_midpoint"}
    kept = []
    skip = False
    for line in text.splitlines():
        header = re.match(r"^\s*\[\s*([^]]+?)\s*\]", line)
        if header:
            skip = header.group(1).strip() in targets
        if not skip:
            kept.append(line)
    return "\n".join(kept).rstrip() + f"\n\n[ V{left}_midpoint ]\n" + " ".join(map(str, mapping[f"V{left}"])) + f"\n\n[ V{right}_midpoint ]\n" + " ".join(map(str, mapping[f"V{right}"])) + "\n"


def path_mdp(left: int, right: int, mapping: dict[str, list[int]], center: float, production: bool) -> str:
    duration = PRODUCTION_PS if production else EQUILIBRATION_PS
    return common_mdp(int(round(duration / DT_PS)), production, not production, "Solute", 250 if production else 0) + pull_block(left, right, mapping, center)


def steering_mdp(left: int, right: int, mapping: dict[str, list[int]], initial: float, target: float) -> str:
    rate = math.copysign(PULL_RATE_NM_PS, target - initial)
    nsteps = int(math.ceil(abs(target - initial) / PULL_RATE_NM_PS / DT_PS))
    return common_mdp(nsteps, False, True, "System", 2500) + pull_block(left, right, mapping, None, rate, STEERING_FORCE_CONSTANT)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=project / "results/A4_p_C6_C8_selectivity_workflow/neighbour_pair_classes.csv")
    parser.add_argument("--run-root", type=Path, default=project / "gromacs_dcm/runs")
    parser.add_argument("--path-spec", type=Path, help="JSON list: [{state,replica,left,right,route_class,topology_class,maximum_nm}]")
    parser.add_argument("--output", type=Path, default=project / "umbrella/a4_p_c6_c8_selectivity")
    parser.add_argument("--allow-unlisted-path", action="store_true", help="allow an explicitly audited path absent from the legacy neighbour CSV")
    args = parser.parse_args()
    workflow, run_root, output = args.workflow.resolve(), args.run_root.resolve(), args.output.resolve()
    rows = load_rows(workflow)
    if args.path_spec is None:
        candidates = [row for row in rows if row["topology_class"] in {"same", "cross"}]
        (output / "path_spec_template.json").parent.mkdir(parents=True, exist_ok=True)
        (output / "path_spec_template.json").write_text(json.dumps([
            {"state": row["state"], "replica": 1, "left": int(row["left"]), "right": int(row["right"]), "route_class": row["route_class"], "topology_class": row["topology_class"], "maximum_nm": 2.75}
            for row in candidates
        ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote candidate path template: {output / 'path_spec_template.json'}")
        return
    spec = json.loads(args.path_spec.resolve().read_text(encoding="utf-8"))
    if not isinstance(spec, list) or not spec:
        raise ValueError("path spec must be a non-empty JSON list")
    output.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for item in spec:
        state, replica = str(item["state"]), int(item["replica"])
        left, right = int(item["left"]), int(item["right"])
        maximum = float(item.get("maximum_nm", 2.75))
        match = next((row for row in rows if row["state"] == state and int(row["left"]) == left and int(row["right"]) == right), None)
        if match is None and not args.allow_unlisted_path:
            raise ValueError(f"path is not present in neighbour-only workflow manifest: {state} V{left}-V{right}")
        if match is None:
            match = {"state": state, "left": str(left), "right": str(right),
                     "topology_class": str(item.get("topology_class", "cross")),
                     "route_class": str(item.get("route_class", "reuse")),
                     "allowed_by_neighbour_model": "explicit_audit_override",
                     "selection_note": "Legacy numbered-cycle filter omitted this graph-valid candidate; explicitly restored from pilot audit."}
        setup = run_root / state / "setup"
        metadata = json.loads((setup / "metadata.json").read_text(encoding="utf-8"))
        mapping = {str(k): [int(v) for v in values] for k, values in metadata["vinyl_mapping_gromacs_1based"].items()}
        start = run_root / state / f"replica_{replica:02d}" / "production_2ns.gro"
        if not start.is_file():
            raise FileNotFoundError(f"pilot endpoint not found: {start}")
        initial, half_box = pair_distance_nm(start, mapping, left, right)
        if initial >= 0.98 * half_box:
            raise ValueError(f"{state} V{left}-V{right}: initial distance {initial:.6f} nm exceeds GROMACS 0.49-box limit {0.98 * half_box:.6f} nm; select another pilot seed")
        if maximum >= 0.98 * half_box:
            raise ValueError(f"{state} V{left}-V{right}: umbrella maximum exceeds half-box limit")
        name = f"{state}_V{left}_V{right}_{item.get('route_class') or item.get('topology_class') or 'path'}"
        path = output / name
        path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(setup / "system.top", path / "system.top")
        (path / "start.gro").write_bytes(start.read_bytes())
        (path / "index.ndx").write_text(append_pair_groups((setup / "index.ndx").read_text(encoding="utf-8"), left, right, mapping), encoding="utf-8")
        (path / "steer_in.mdp").write_text(steering_mdp(left, right, mapping, initial, MINIMUM_NM - 0.10), encoding="utf-8")
        (path / "steer_out.mdp").write_text(steering_mdp(left, right, mapping, initial, maximum), encoding="utf-8")
        window_rows = []
        window_root = path / "window_mdps"
        window_root.mkdir()
        for index, center in enumerate(centers(maximum)):
            window = f"window_{index:03d}_{center:.3f}nm"
            (window_root / f"{window}_equil.mdp").write_text(path_mdp(left, right, mapping, center, False), encoding="utf-8")
            (window_root / f"{window}_production.mdp").write_text(path_mdp(left, right, mapping, center, True), encoding="utf-8")
            window_rows.append({"window_index": index, "window_name": window, "center_nm": f"{center:.3f}"})
        with (path / "windows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("window_index", "window_name", "center_nm"))
            writer.writeheader(); writer.writerows(window_rows)
        metadata_out = {**item, "state": state, "replica": replica, "initial_distance_nm": initial, "half_box_limit_nm": half_box, "window_count": len(window_rows), "source_manifest_row": match}
        (path / "pathway.json").write_text(json.dumps(metadata_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest.append(metadata_out)
    (output / "submission_manifest.json").write_text(json.dumps({"temperature_K": 313.15, "paths": manifest, "same_window_protocol": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "paths": len(manifest)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
