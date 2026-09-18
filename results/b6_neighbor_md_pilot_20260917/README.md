# B6-p-C6 neighbour-only q1/q2 explicit-DCM MD pilot

## Scope and data quality

Five independent conditional molecular states were run as one Bohrium T4 job
each: q1 (`20748984`), q2 full-reuse (`20749009`), one-arm-reuse (`20749047`),
full-switch-d1 (`20749048`), and full-switch-d2 (`20749051`). Each state has
three velocity-reseeded 2 ns production trajectories. All 15 GROMACS logs
report normal completion. The XTC-derived distance tables each contain 4001
frames on a 0.5 ps grid; all 15 passed the existing structural data-quality
gate. The downloaded `out.zip` archives and expanded trajectories are held in
`../bohrium_download/b6_neighbor_md_pilot_20260917/`, outside Git.
The bonded solute topology has one connected fragment in each state; the
existing extractor unwraps that fragment before measuring vinyl-midpoint
distances across periodic boundaries.

For the focused analysis, only vinyl pairs on adjacent B6 arms are retained.
The original q1 orbit table conditional on q0 = V1–V2 gives 17 eligible
labelled pairs: 1 full-reuse, 4 one-arm-reuse, 8 full-switch-d1, and 4
full-switch-d2. No same-arm or non-neighbour-arm pairs enter the scores. The
encounter proxy is the earlier fixed sigmoid
`1/[1+exp((d−5 Å)/0.5 Å)]` of the vinyl-bond midpoint distance; a hard
`d < 5 Å` threshold is reported separately. Orbit scores **sum** across
their eligible labelled pairs, retaining their multiplicities. Each q2 state
has 12 eligible neighbouring-arm pairs.

## q1 pilot observations

| q1 orbit | Eligible pairs | Mean summed soft score | Replicas: min–max | Frames with any pair <5 Å / 12003 | Descriptive score share |
|---|---:|---:|---:|---:|---:|
| Full reuse | 1 | 0.0318 | ~0–0.0611 | 350 | 13.9% |
| One-arm reuse | 4 | 0.00327 | ~0–0.00975 | 33 | 1.4% |
| Full switch d1 | 8 | 0.0870 | 0.0428–0.1449 | 925 | 38.0% |
| Full switch d2 | 4 | 0.1067 | 0.0521–0.1845 | 1152 | 46.7% |

This pilot suggests accessible switch contacts, particularly d2, and rare
one-arm-reuse contacts. However, d1 exceeds d2 in replicas 1 and 3 whereas
d2 exceeds d1 in replica 2. Full-reuse has 251, 99, and 0 hard-contact frames
in the three replicas. The first/second-half summed scores also shift strongly
(d1: 0.1499→0.0240; d2: 0.1482→0.0652). The 46.7% or 38.0% numbers are
**not reaction-path probabilities** and cannot be used as an experimental
product-ratio prediction.

## q2 conditional-state pilot observations

| q2 state | Mean summed soft score over 12 allowed pairs | Replicas: min–max | Frames with any pair <5 Å / 12003 |
|---|---:|---:|---:|
| Full reuse | 0.0112 | 0.00043–0.0315 | 95 |
| One-arm reuse | 0.1131 | 0.0470–0.2157 | 1152 |
| Full switch d1 | 0.0850 | 0.0456–0.1176 | 898 |
| Full switch d2 | 0.2137 | 0.1163–0.2732 | 2017 |

The q2 rows are **separately initialized conditional structures**. Their raw
scores must not be normalized against one another as if a single q1 trajectory
visited all four states. Full-reuse q2 is particularly sparse: the three
replicas have 0, 2, and 93 frames with any eligible pair below 5 Å. The
per-pair CSV identifies the leading geometric contacts for subsequent PMFs.

## Interpretation and next calculation

The entire pilot passes trajectory-integrity checks, but encounter statistics
remain basin- and replica-dependent. In line with the A4 analysis, identical
umbrella protocols for the four q1 orbit representatives are the appropriate
next *main* evidence for free-energy-based conditional weights. Retain orbit
multiplicities 1/4/8/4 in the final q1 calculation. The four q2 structures
require their own conditional PMFs for selected adjacent-arm pairs, not a
direct comparison of their raw unrestrained encounter fractions. Extra
unbiased sampling can check basin memory, but is not a replacement for PMFs.

Reproduce the focused tables with
`python analysis/analyze_b6_neighbor_pilot.py` in an environment with NumPy
and pandas. The five generic audits were generated with
`analysis/analyze_explicit_dcm_q0.py` (reused only for its state-agnostic
distance/trajectory gate). See `neighbor_analysis/summary.json` and CSV files
for the exact values. No umbrella result or experimental-yield estimate is
claimed here.
