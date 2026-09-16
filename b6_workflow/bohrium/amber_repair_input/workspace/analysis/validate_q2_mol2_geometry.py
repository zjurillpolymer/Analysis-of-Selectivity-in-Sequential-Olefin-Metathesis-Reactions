"""Reject B6 q2 Amber inputs with graph bonds spanning nonphysical distances."""

import argparse
import math
from pathlib import Path


def check_mol2(path: Path, expected_atoms: int, maximum_bond_A: float) -> tuple[int, float]:
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms_start = lines.index("@<TRIPOS>ATOM") + 1
    bonds_start = lines.index("@<TRIPOS>BOND")
    atoms = {}
    for line in lines[atoms_start:bonds_start]:
        parts = line.split()
        if parts:
            atoms[int(parts[0])] = tuple(float(value) for value in parts[2:5])
    if len(atoms) != expected_atoms:
        raise ValueError(f"{path.name}: {len(atoms)} atoms; expected {expected_atoms}")
    distances = []
    for line in lines[bonds_start + 1 :]:
        if line.startswith("@<TRIPOS>"):
            break
        parts = line.split()
        if len(parts) >= 4:
            left, right = (atoms[int(parts[1])], atoms[int(parts[2])])
            distances.append(math.dist(left, right))
    if not distances:
        raise ValueError(f"{path.name}: no bonds")
    longest = max(distances)
    if longest > maximum_bond_A:
        raise ValueError(f"{path.name}: longest bond {longest:.2f} A exceeds {maximum_bond_A:.2f} A")
    return len(atoms), longest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structures", type=Path, required=True)
    parser.add_argument("--expected-atoms", type=int, default=432)
    parser.add_argument("--maximum-bond-A", type=float, default=2.20)
    parser.add_argument("states", nargs="+")
    args = parser.parse_args()
    for state in args.states:
        atoms, longest = check_mol2(
            args.structures / f"{state}_antechamber.mol2",
            args.expected_atoms,
            args.maximum_bond_A,
        )
        print(f"[OK] {state}: {atoms} atoms, longest bond {longest:.3f} A")
