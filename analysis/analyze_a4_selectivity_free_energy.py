#!/usr/bin/env python3
"""Convert completed umbrella PMFs into class probability ratios.

Each pathway is normalized against its own non-contact reference interval, so
the calculation is valid for comparing selected candidate pairs without
pretending that independently run PMFs share an arbitrary zero.  This is a
path-level PMF analysis; the unbiased ensemble remains the class-coverage
check.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

R_KCAL = 0.00198720425864083
TEMPERATURE_K = 313.15


def read_pmf(path: Path) -> list[tuple[float, float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "@")):
            continue
        fields = line.split()
        if len(fields) >= 2:
            rows.append((float(fields[0]), float(fields[1])))
    if len(rows) < 3:
        raise ValueError(f"too few PMF rows: {path}")
    return rows


def interval_weight(rows: list[tuple[float, float]], low: float, high: float, beta: float) -> float:
    selected = [(x, y) for x, y in rows if low <= x <= high and math.isfinite(y)]
    if len(selected) < 2:
        return 0.0
    total = 0.0
    for (x0, y0), (x1, y1) in zip(selected, selected[1:]):
        total += 0.5 * (math.exp(-beta * y0) + math.exp(-beta * y1)) * (x1 - x0)
    return max(total, 0.0)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--umbrella-root", type=Path, default=project / "umbrella/a4_p_c6_c8_selectivity")
    parser.add_argument("--pmf-name", default="pmf.xvg")
    parser.add_argument("--contact-cutoff-nm", type=float, default=0.50)
    parser.add_argument("--reference-low-nm", type=float, default=0.90)
    parser.add_argument("--reference-high-nm", type=float, default=1.50)
    parser.add_argument("--pmf-unit", choices=("kj", "kcal"), default="kj", help="unit emitted by WHAM; default is GROMACS kJ mol^-1")
    parser.add_argument("--temperature-K", type=float, default=TEMPERATURE_K)
    parser.add_argument("--output", type=Path, default=project / "results/A4_p_C6_C8_selectivity_umbrella_analysis")
    args = parser.parse_args()
    beta = 1.0 / (R_KCAL * args.temperature_K)
    path_root, output = args.umbrella_root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    path_rows = []
    for path in sorted(path_root.iterdir()):
        metadata_path = path / "pathway.json"
        pmf_path = path / args.pmf_name
        if not metadata_path.is_file() or not pmf_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        rows = read_pmf(pmf_path)
        if args.pmf_unit == "kj":
            rows = [(distance, energy / 4.184) for distance, energy in rows]
        contact = interval_weight(rows, min(x for x, _ in rows), args.contact_cutoff_nm, beta)
        reference = interval_weight(rows, args.reference_low_nm, args.reference_high_nm, beta)
        if contact <= 0 or reference <= 0:
            continue
        dg = -R_KCAL * args.temperature_K * math.log(contact / reference)
        path_rows.append({**metadata, "delta_g_contact_vs_reference_kcal_mol": dg, "contact_weight": contact, "reference_weight": reference})
    with (output / "path_free_energies.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["state", "branch", "left", "right", "route_class", "topology_class", "delta_g_contact_vs_reference_kcal_mol"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(path_rows)
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in path_rows:
        label = row.get("route_class") or row.get("topology_class")
        grouped[(str(row["state"]), str(row.get("branch", "")), str(label))]["values"].append(float(row["delta_g_contact_vs_reference_kcal_mol"]))
    contrasts = []
    for state, branch in sorted({(key[0], key[1]) for key in grouped}):
        classes = {key[2] for key in grouped if key[0] == state and key[1] == branch}
        for alternative, reference in (("cross", "same"), ("switch", "reuse")):
            if alternative not in classes or reference not in classes:
                continue
            # Per-pair mean of the contact/reference probability proxy.
            p_alt = sum(math.exp(-dg * beta) for dg in grouped[(state, branch, alternative)]["values"]) / len(grouped[(state, branch, alternative)]["values"])
            p_ref = sum(math.exp(-dg * beta) for dg in grouped[(state, branch, reference)]["values"]) / len(grouped[(state, branch, reference)]["values"])
            ddg = -R_KCAL * args.temperature_K * math.log(p_alt / p_ref)
            contrasts.append({"state": state, "branch": branch, "alternative": alternative, "reference": reference, "delta_delta_g_alternative_minus_reference_kcal_mol": ddg, "probability_ratio_alternative_over_reference": p_alt / p_ref})
    with (output / "class_free_energy_contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["state", "branch", "alternative", "reference", "delta_delta_g_alternative_minus_reference_kcal_mol", "probability_ratio_alternative_over_reference"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(contrasts)
    print(json.dumps({"pathways_analyzed": len(path_rows), "contrasts": len(contrasts), "output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
