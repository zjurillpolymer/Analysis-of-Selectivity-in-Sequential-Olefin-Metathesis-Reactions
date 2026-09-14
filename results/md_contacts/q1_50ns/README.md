# Replicated explicit-DCM trajectory audit

- Files: 3/3 passed structural QC.
- Frames: 300003/300003 read.
- Uncertainty unit: one independently reseeded replica; frame-level independence is not assumed.
- Primary score: sigmoid contact with center 5 Å and width 0.5 Å; hard 4–7 Å occupancies are sensitivity checks.
- Convergence: split-Rhat plus autocorrelation-corrected effective sample size (ESS).

## State decisions

| State | duration (ps) | material pairs | max Rhat | min ESS | max tau (ps) | decision |
|---|---:|---:|---:|---:|---:|---|
| B6_p_C6_q1_V3_V6_cis | 50000 | 12 | 1.029 | 622.4 | 953.1 | sufficient_for_current_contact_observable |

A material pair is one carrying at least 2% of the state's total mean soft-contact score. This threshold is a simulation-extension triage rule, not part of the chemical model.

## Pairs that control an extension decision

| State | pair | topology | q mass | Rhat | ESS | status |
|---|---|---|---:|---:|---:|---|
| — | — | — | — | — | — | no material pair requires extension |

## Files

- `trajectory_quality.csv`: completeness, time grid, and numeric integrity.
- `pair_replica_summary.csv`: distance/contact statistics and ESS for each independent replica.
- `pair_state_summary.csv`: replica-level uncertainty, split-Rhat, and pair-resolved contact features.
- `topology_state_summary.csv`: same-arm/cross-arm/all-pair summaries.
- `state_convergence_summary.csv`: compact current-duration sufficiency decision.
- `figures/`: pair scores and convergence diagnostics.
