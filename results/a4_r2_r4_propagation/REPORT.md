# A4-m-C6/C7 target-only R2/R4 propagation

Primary setting: `d5_axis45_side45`. q0 is fixed to cross. q1 same-arm and q2 R1/R3 outcomes are reported as leakage but excluded from the conditional R2/R4 prediction.

## Product bookkeeping

q1 full reuse contributes 2 R2; q1 partial reuse enters q2 adjacent and contributes R4 on the target cross branch; q1 full switch enters q2 opposite, where R2x2 contributes 2 R2 and each R4 matching contributes 1 R4.

| system | q1 full reuse | q1 partial reuse | q1 full switch | q2 opposite R2x2 | q2 opposite R4 | predicted R2/R4 (molecule) | predicted R2/R4 (M4/M2=2 mass proxy) | experiment R2/R4 (wt%, target-normalized) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C6 | 0.152 | 0.572 | 0.276 | 0.863 | 0.137 | 0.590/0.410 | 0.419/0.581 | 0.415/0.585 |
| C7 | 0.267 | 0.408 | 0.325 | 0.000 | 1.000 | 0.430/0.570 | 0.274/0.726 | 0.220/0.780 |

The mass-proxy column is only a diagnostic with M4/M2=2; the final comparison to SEC wt% should use the exact post-cleavage molecular formulas. The source tables retain candidate multiplicity explicitly, so the four partial-reuse q1 candidates are not collapsed to a single per-pair mean.

Files: `q1_q2_branch_weights.csv`, `q2_matching_outcomes.csv`, and `r2_r4_predictions.csv`.

## Pending umbrella confirmation

The q1 partial-reuse/adjacent umbrella set was submitted as four independent
Bohrium T4 jobs (C6 V1–V5, replicas 1/2; C7 V5–V6, replicas 1/2).  The current
R2/R4 values above use the existing unbiased q1 all-pair weights; the new PMFs
will be added as a pathway-level robustness check rather than replacing the
candidate-summed q1 branching weights.
