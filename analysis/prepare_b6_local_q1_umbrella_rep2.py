#!/usr/bin/env python3
"""Prepare the missing replica-2 B6 q1 umbrella paths."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import prepare_b6_neighbor_q1_umbrella as base

base.REPLICAS = (2,)
base.SOURCE_JOB = 20750428
base.main()
