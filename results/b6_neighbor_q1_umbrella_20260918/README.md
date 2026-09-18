# B6-p-C6 q1 neighbour-orbit umbrella submission

Eight one-T4 Bohrium jobs were submitted from the completed 10 ns q1
checkpoint trajectory: four orbit representatives × independent replica 1
and replica 3 endpoints. Each uses the same 25 midpoint-distance windows,
0.35–2.75 nm in 0.10 nm increments, with 500 kJ mol⁻¹ nm⁻² restraint,
100 ps equilibration and 500 ps production per window. WHAM uses the last
400 ps of each production window and the runner performs first/second-half
checks. Local GROMACS 2026.3 `grompp -maxwarn 0` preflight passed for all
eight packages. The remote image is GROMACS 2024.2 and will run its own
`grompp` for each stage.

| q1 orbit | Pair | Multiplicity | 10 ns source replicas | Bohrium IDs |
|---|---|---:|---|---|
| Full reuse | V3–V12 | 1 | 1, 3 | 20752763, 20752764 |
| One-arm reuse | V3–V10 | 4 | 1, 3 | 20752770, 20752771 |
| Full switch d1 | V4–V6 | 8 | 1, 3 | 20752765, 20752766 |
| Full switch d2 | V6–V8 | 4 | 1, 3 | 20752767, 20752768 |

The simulation bias is one-dimensional **distance**. As in the established
A4 workflow, the planned analysis reconstructs unbiased weights and evaluates
distance-plus-orientation near-attack geometry in two dimensions, with an
explicit orientation-mixing audit. This is not a separately biased 2D umbrella
grid. If orientation has not mixed within the distance windows, the reweighted
2D inference must be withheld and targeted orientation enhancement added.

Orbit multiplicities are applied after obtaining per-path PMF/NAC weights,
not to the raw WHAM curves. Distinct conditional q2 states are not included
in these jobs. Submission does not establish successful completion.

The generated job input bundle and exact job-ID record are outside Git at
`../bohrium_submit/b6_neighbor_q1_umbrella_rep13_20260918/`.
