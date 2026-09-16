# A4-m-C5–C8 q1/q2 near-attack geometry analysis

Primary gate: midpoint distance 5.0 Å (width 0.5 Å), vinyl-axis angle ≤45° and side-on angle ≥45° (angular width 7.5°).
All free energies are catalyst-free effective proxies at 313.15 K. Negative values favor the numerator group.

| state | contrast | replicas | ΔΔG contact | ΔΔG orientation | ΔΔG total NAC | 95% replica-bootstrap CI | odds (numerator/denominator) |
|---|---|---:|---:|---:|---:|---:|---:|
| A4_m_C6_2_mixed_same_cross_cis | reuse_minus_switch | 3 | +1.073 | +0.184 | +1.256 | [+1.007, +14.961] | 0.1328 |
| A4_m_C6_2_mixed_same_cross_cis | cross_minus_same | 3 | -0.986 | -0.203 | -1.189 | [-2.410, -0.255] | 6.761 |
| A4_m_C7_2_mixed_same_cross_cis | reuse_minus_switch | 3 | +0.391 | -0.037 | +0.354 | [-1.515, +9.030] | 0.5663 |
| A4_m_C7_2_mixed_same_cross_cis | cross_minus_same | 3 | -0.879 | -0.086 | -0.965 | [-1.635, +0.760] | 4.717 |

## Primary result

- `A4_m_C6_2_mixed_same_cross_cis` / `reuse_minus_switch`: ΔΔG=+1.256 kcal mol⁻¹ (95% CI [+1.007, +14.961]); robustly favors switch. The larger decomposition term is contact accessibility.
- `A4_m_C6_2_mixed_same_cross_cis` / `cross_minus_same`: ΔΔG=-1.189 kcal mol⁻¹ (95% CI [-2.410, -0.255]); robustly favors cross_arm. The larger decomposition term is contact accessibility.
- `A4_m_C7_2_mixed_same_cross_cis` / `reuse_minus_switch`: ΔΔG=+0.354 kcal mol⁻¹ (95% CI [-1.515, +9.030]); is unresolved across independent replicas. The larger decomposition term is contact accessibility.
- `A4_m_C7_2_mixed_same_cross_cis` / `cross_minus_same`: ΔΔG=-0.965 kcal mol⁻¹ (95% CI [-1.635, +0.760]); is unresolved across independent replicas. The larger decomposition term is contact accessibility.

## Interpretation boundary

The per-pair comparison removes route multiplicity. `topology` compares cross-arm with same-arm candidates; `route_family` compares reuse of a previously reacted pair of molecular arms with alternative cross-arm switches.
Candidate scope: `same_or_cyclic_neighbour_arms`. In `same_or_cyclic_neighbour_arms` mode, cross-arm pairs on opposite A4 arms are excluded by the model assumption and cannot contribute to reuse/switch probabilities.
Replica bootstrap treats independent GROMACS replicas as the uncertainty unit. With fewer than three replicas, confidence intervals are only pilot diagnostics.
These values quantify substrate preorganization and orientation, not a Ru-catalyzed activation barrier.

## Periodic-boundary quality control

Quality-controlled exports: 6. Retained frame fraction: 100.000%–100.000%.
All exports used per-frame box vectors; no final-GRO periodic-box fallback was needed.
