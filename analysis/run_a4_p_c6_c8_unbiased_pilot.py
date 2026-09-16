#!/usr/bin/env python3
"""Run the A4-p q1 pilot before the derived q2 pilot on one GPU."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

Q1 = ("A4_p_C6_1_cis", "A4_p_C8_1_cis")
Q2 = (
    "A4_p_C6_2_adjacent_cis", "A4_p_C6_2_opposite_cis",
    "A4_p_C8_2_adjacent_cis", "A4_p_C8_2_opposite_cis",
)


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--ntomp", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.replicas < 1 or args.ntomp < 1:
        raise SystemExit("replicas and ntomp must be positive")
    runner = project / "analysis/run_explicit_dcm_gromacs.py"
    for label, states in (("q1", Q1), ("q2-derived", Q2)):
        command = [sys.executable, str(runner), "--replicas", str(args.replicas), "--ntomp", str(args.ntomp), "--states", *states]
        print(f"[{label}] " + " ".join(command))
        if not args.dry_run:
            subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
