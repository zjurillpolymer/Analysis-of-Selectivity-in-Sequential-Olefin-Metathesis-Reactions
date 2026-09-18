# B6-p-C6 neighbour-only 2 → 10 ns extension analysis

The five Bohrium jobs were marked failed by a post-run filename check, not by
GROMACS. Their archives contain `extension_metadata.json` with
`target_total_ns = 10`, `production_10ns.tpr`, and a merged full trajectory in
`production_2ns.xtc`. All 15 replicas were recovered and yielded 20,001 frames
on a 0.5 ps grid (0–10,000 ps).

## q1 orbit comparison

The scores below use the same neighbour-only model and the same sigmoid
encounter proxy as the 2 ns pilot. q1 is conditioned on fixed q0 = V1–V2 and
retains the 17 labelled candidates with orbit multiplicities 1/4/8/4. Scores
are summed across labelled candidates within each orbit, then averaged over
three replicas.

| q1 orbit | candidates | 2 ns mean summed score | 10 ns mean | 10 ns replica range | 10 ns descriptive score share |
|---|---:|---:|---:|---:|---:|
| Full reuse | 1 | 0.0318 | 0.0348 | 0.0295–0.0409 | 12.7% |
| One-arm reuse | 4 | 0.00327 | 0.00674 | 0.000088–0.0194 | 2.5% |
| Full switch d1 | 8 | 0.0870 | 0.1020 | 0.0385–0.1539 | 37.1% |
| Full switch d2 | 4 | 0.1067 | 0.1313 | 0.0862–0.1689 | 47.8% |

The extension materially improves the full-reuse result: its 2 ns range was
approximately 0–0.0611, whereas the 10 ns range is 0.0295–0.0409. The
full-switch ordering is now more suggestive of d2 > d1 in the replica mean,
but d1 remains higher than d2 in replica 1. One-arm reuse remains close to the
contact detection limit and is dominated by replica 3; it should not be
assigned a precise probability from this MD alone.

The first/second-half summed scores still show basin evolution. For example,
q1 d1 changes from 0.1499 to 0.0966 and d2 from 0.1482 to 0.1294 when averaged
over replicas. Thus 10 ns is a substantially better pilot, not proof of full
equilibrium.

## q2 conditional-state comparison

| q2 state | 2 ns mean summed score | 10 ns mean | 10 ns replica range | 10 ns any-pair <5 Å fraction |
|---|---:|---:|---:|---:|
| Full reuse | 0.0112 | 0.0307 | 0.0223–0.0426 | 2.72% |
| One-arm reuse | 0.1131 | 0.1730 | 0.1189–0.2602 | 14.8% |
| Full switch d1 | 0.0850 | 0.0632 | 0.0501–0.0719 | 5.49% |
| Full switch d2 | 0.2137 | 0.2337 | 0.0839–0.3102 | 19.4% |

These are four separately initialized conditional q2 states, not four routes
visited by one trajectory. Their raw score means therefore cannot be divided
directly into q2 product probabilities. The d2 and one-arm q2 states remain
more contact-rich, while q2 full-reuse becomes detectable but sparse.

## Decision

The 10 ns extension was necessary and useful. It resolves the most severe
full-reuse instability and supports moving to the main umbrella calculation.
The remaining replica spread, especially for one-arm reuse and switch d1/d2,
means that the scores above remain encounter diagnostics rather than reaction
probabilities. The next main evidence should be identical PMF/umbrella
calculations for the four q1 orbit representatives, with multiplicities 1, 4,
8, and 4 applied only after obtaining the free-energy weights. q2 umbrella
calculations should then be performed as conditional states; their 10 ns raw
contact fractions should not be used as final yield estimates.

Reproducible tables are in `neighbor_analysis/`; the extension archives and
expanded trajectories are in the sibling `bohrium_download` directory and are
not committed to Git.
