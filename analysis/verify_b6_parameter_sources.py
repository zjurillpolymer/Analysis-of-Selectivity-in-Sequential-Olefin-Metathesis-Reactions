#!/usr/bin/env python3
"""Fail closed if a B6 MD input contains unapproved Amber parameter files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from setup_explicit_dcm_gromacs import (
    cached_sqm_tolerance_satisfies,
    mol2_charge_and_count,
    read_pdb,
)


EXPECTED_JOB_Q1 = 20735563
EXPECTED_JOB_Q2 = 20744510


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    states = manifest["states"]
    if len(states) != 5 or sum("_q1_" in state for state in states) != 1:
        raise ValueError("expected one B6 q1 plus four q2 parameter sets")

    for state, record in states.items():
        expected_job = EXPECTED_JOB_Q1 if "_q1_" in state else EXPECTED_JOB_Q2
        if record["bohr_job_id"] != expected_job:
            raise ValueError(f"uncertified parameterization job for {state}")
        folder = root / manifest["root"] / state / "setup/solute_parameters"
        for filename, key in (
            ("slu.mol2", "slu_mol2_sha256"),
            ("slu.frcmod", "slu_frcmod_sha256"),
        ):
            actual = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
            if actual != record[key]:
                raise ValueError(f"SHA256 mismatch for {state}/{filename}")
        metadata = json.loads(
            (folder / "parameterization_metadata.json").read_text(encoding="utf-8")
        )
        if metadata.get("input_format") != "mol2_explicit_pdb_bonds":
            raise ValueError(f"wrong input format for {state}")
        if metadata.get("antechamber_judge_mode") != 5:
            raise ValueError(f"wrong Antechamber bond judgement for {state}")
        if not cached_sqm_tolerance_satisfies(metadata, 0.05):
            raise ValueError(f"SQM tolerance not satisfied for {state}")
        charge, atoms = mol2_charge_and_count(folder / "slu.mol2")
        expected_atoms = 438 if "_q1_" in state else 432
        if atoms != expected_atoms or len(read_pdb(folder / "slu.pdb")) != atoms:
            raise ValueError(f"atom-count mismatch for {state}: {atoms}")
        if abs(charge) > 1e-4:
            raise ValueError(f"non-neutral solute for {state}: {charge}")
        print(f"[APPROVED] {state}: job={expected_job} atoms={atoms} charge={charge:+.6f}")


if __name__ == "__main__":
    main()
