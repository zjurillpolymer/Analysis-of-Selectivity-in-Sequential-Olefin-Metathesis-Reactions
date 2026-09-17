# B6 q2 repaired-geometry Amber input — completed

The first parameterization job (Bohr ID 20735563) exited successfully, but
all four q2 products contained a newly created double bond spanning 5–27 Å.
This independent input bundle contains only the four repaired q2 states;
the valid q1 parameterization from the first job does not need repeating.

Each q2 structure was regenerated with ETKDGv3 and MMFF94 (four conformers,
deterministic seed). The lowest-energy converged conformer retained both
internal alkenes as Z/cis. The preflight script rejects any MOL2 with a bond
longer than 2.20 Å before AmberTools is called. All four inputs passed with
maximum bond lengths below 1.58 Å. The atom order and bond graph are unchanged
from the first input, but the AM1-BCC charges should be recalculated from the
repaired geometry rather than reused from the distorted input.

This directory is a self-contained Bohrium `bohr job submit` input. Its
`job.json` uses project 4377225 and the same AmberTools image as the first
job. Submitted as Bohr job **20744510** (group **7422312**); see
`submission.json`. The job completed with exit code 0 on 2026-09-17. All four
downloaded parameter sets passed file, charge, geometry, topology and log
checks; see `results/b6_neighbor_amber_geometry_audit/REPORT.md`. Solvated
setup and MD have not yet been run.
