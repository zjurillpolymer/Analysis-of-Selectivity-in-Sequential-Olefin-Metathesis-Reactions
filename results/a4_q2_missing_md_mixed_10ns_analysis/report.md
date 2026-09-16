# A4-m-C5/C8 q1/q2 near-attack geometry analysis

Primary gate: midpoint distance 5.0 Å (width 0.5 Å), vinyl-axis angle ≤45° and side-on angle ≥45° (angular width 7.5°).
All free energies are catalyst-free effective proxies at 313.15 K. Negative values favor the numerator group.

| state | contrast | replicas | ΔΔG contact | ΔΔG orientation | ΔΔG total NAC | 95% replica-bootstrap CI | odds (numerator/denominator) |
|---|---|---:|---:|---:|---:|---:|---:|
| A4_m_C5_2_mixed_same_cross_cis | reuse_minus_switch | 3 | -2.505 | -0.245 | -2.750 | [-3.204, -2.397] | 83.01 |
| A4_m_C5_2_mixed_same_cross_cis | cross_minus_same | 3 | -0.982 | -0.446 | -1.428 | [-1.849, -1.168] | 9.919 |
| A4_m_C8_2_mixed_same_cross_cis | reuse_minus_switch | 3 | +1.486 | +0.259 | +1.745 | [+0.693, +5.697] | 0.06052 |
| A4_m_C8_2_mixed_same_cross_cis | cross_minus_same | 3 | -0.479 | -0.184 | -0.663 | [-1.741, +0.131] | 2.904 |

## Primary result

- `A4_m_C5_2_mixed_same_cross_cis` / `reuse_minus_switch`: ΔΔG=-2.750 kcal mol⁻¹ (95% CI [-3.204, -2.397]); robustly favors reuse. The larger decomposition term is contact accessibility.
- `A4_m_C5_2_mixed_same_cross_cis` / `cross_minus_same`: ΔΔG=-1.428 kcal mol⁻¹ (95% CI [-1.849, -1.168]); robustly favors cross_arm. The larger decomposition term is contact accessibility.
- `A4_m_C8_2_mixed_same_cross_cis` / `reuse_minus_switch`: ΔΔG=+1.745 kcal mol⁻¹ (95% CI [+0.693, +5.697]); robustly favors switch. The larger decomposition term is contact accessibility.
- `A4_m_C8_2_mixed_same_cross_cis` / `cross_minus_same`: ΔΔG=-0.663 kcal mol⁻¹ (95% CI [-1.741, +0.131]); is unresolved across independent replicas. The larger decomposition term is contact accessibility.

## Interpretation boundary

The per-pair comparison removes route multiplicity. `topology` compares cross-arm with same-arm candidates; `route_family` compares reuse of a previously reacted pair of molecular arms with alternative cross-arm switches.
Replica bootstrap treats independent GROMACS replicas as the uncertainty unit. With fewer than three replicas, confidence intervals are only pilot diagnostics.
These values quantify substrate preorganization and orientation, not a Ru-catalyzed activation barrier.

## Periodic-boundary quality control

Quality-controlled exports: 6. Retained frame fraction: 100.000%–100.000%.
All exports used per-frame box vectors; no final-GRO periodic-box fallback was needed.
