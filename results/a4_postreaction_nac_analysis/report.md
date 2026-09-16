# A4-m-C5/C8 q1/q2 near-attack geometry analysis

Primary gate: midpoint distance 5.0 Å (width 0.5 Å), vinyl-axis angle ≤45° and side-on angle ≥45° (angular width 7.5°).
All free energies are catalyst-free effective proxies at 313.15 K. Negative values favor the numerator group.

| state | contrast | replicas | ΔΔG contact | ΔΔG orientation | ΔΔG total NAC | 95% replica-bootstrap CI | odds (numerator/denominator) |
|---|---|---:|---:|---:|---:|---:|---:|
| A4_m_C5_2_adjacent_cis | cross_minus_same | 3 | -0.875 | -0.379 | -1.254 | [-3.119, -0.840] | 7.504 |
| A4_m_C5_2_opposite_cis | reuse_minus_switch | 3 | -0.897 | -0.068 | -0.965 | [-1.675, -0.580] | 4.715 |
| A4_m_C8_2_adjacent_cis | cross_minus_same | 3 | -4.731 | -0.013 | -4.744 | [-8.558, -4.453] | 2045 |
| A4_m_C8_2_opposite_cis | reuse_minus_switch | 3 | +8.724 | -0.065 | +8.658 | [+8.568, +14.949] | 9.064e-07 |

## Primary result

For A4-m-C8, reuse-minus-switch is strongly positive and its confidence interval stays above zero: the previously reacted arm pair is suppressed, while switching to another arm pair is favored. The decomposition shows that this effect is dominated by contact accessibility rather than the orientation gate.
For A4-m-C5, the pooled point estimate favors reuse, but the three replicas are heterogeneous and the confidence interval crosses zero. It is therefore a hypothesis requiring additional independent replicas, not yet a robust conclusion.

## Interpretation boundary

The per-pair comparison removes route multiplicity. `topology` compares cross-arm with same-arm candidates; `route_family` compares exact reuse of an already reacted arm pair with all alternative cross-arm switches.
Replica bootstrap treats independent GROMACS replicas as the uncertainty unit. With fewer than three replicas, confidence intervals are only pilot diagnostics.
These values quantify substrate preorganization and orientation, not a Ru-catalyzed activation barrier.

## Periodic-boundary quality control

Quality-controlled exports: 12. Retained frame fraction: 100.000%–100.000%.
0 export(s) did not contain per-frame box vectors, so minimum-image reconstruction used the final GRO box and discarded frames whose reconstructed vinyl bonds remained nonphysical. Future exports include per-frame box vectors and do not require this fallback.
