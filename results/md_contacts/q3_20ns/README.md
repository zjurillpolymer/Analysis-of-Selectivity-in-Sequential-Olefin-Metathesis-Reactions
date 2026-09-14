# Replicated explicit-DCM trajectory audit

- Files: 6/6 passed structural QC.
- Frames: 240006/240006 read.
- Uncertainty unit: one independently reseeded replica; frame-level independence is not assumed.
- Primary score: sigmoid contact with center 5 Å and width 0.5 Å; hard 4–7 Å occupancies are sensitivity checks.
- Convergence: split-Rhat plus autocorrelation-corrected effective sample size (ESS).

## State decisions

| State | duration (ps) | material pairs | max Rhat | min ESS | max tau (ps) | decision |
|---|---:|---:|---:|---:|---:|---|
| B6_p_C6_q3_V3_V6_V11_V12_V8_V10_cis | 20000 | 6 | 1.053 | 1342.7 | 336.1 | usable_with_caution |
| B6_p_C6_q3_V3_V6_V11_V12_V7_V9_cis | 20000 | 8 | 1.055 | 442.0 | 296.6 | usable_with_caution |

A material pair is one carrying at least 2% of the state's total mean soft-contact score. This threshold is a simulation-extension triage rule, not part of the chemical model.

## Pairs that control an extension decision

| State | pair | topology | q mass | Rhat | ESS | status |
|---|---|---|---:|---:|---:|---|
| B6_p_C6_q3_V3_V6_V11_V12_V7_V9_cis | V2_V10_跨臂 | cross_arm | 17.98% | 1.054655859937657 | 46165.6 | caution |
| B6_p_C6_q3_V3_V6_V11_V12_V8_V10_cis | V1_V7_跨臂 | cross_arm | 13.35% | 1.0525677918266272 | 40967.2 | caution |

## Files

- `trajectory_quality.csv`: completeness, time grid, and numeric integrity.
- `pair_replica_summary.csv`: distance/contact statistics and ESS for each independent replica.
- `pair_state_summary.csv`: replica-level uncertainty, split-Rhat, and pair-resolved contact features.
- `topology_state_summary.csv`: same-arm/cross-arm/all-pair summaries.
- `state_convergence_summary.csv`: compact current-duration sufficiency decision.
- `figures/`: pair scores and convergence diagnostics.
