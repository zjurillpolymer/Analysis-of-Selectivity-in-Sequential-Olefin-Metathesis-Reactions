# Bohrium submission record

`amber_input/` is the historical AmberTools submission bundle for five
B6 neighbour-only states (Bohr ID 20735563):

- q1: `B6_p_C6_q1_V1_V2_cis`;
- q2: `full_reuse`, `one_arm_reuse`, `full_switch_d1`, and `full_switch_d2`.

The job exited successfully, but the four q2 inputs had nonphysical new-bond
distances of 5–27 Å. Do not use those q2 parameter files for MD. The q1
parameterization passed the basic geometry check. `amber_repair_input/` is a
separate four-state q2 reparameterization bundle with optimized cis structures
and a preflight bond-length gate. It was submitted as Bohr job **20744510**;
completion and output QC are still pending.

These jobs run only GAFF2/AM1-BCC parameterization in the Amber image. Packing,
tleap/ParmEd conversion and GROMACS runs are separate stages. The checked-in
`submission.json` is provenance only, not a credential. AccessKey/SSO state
remains in the local Bohrium profile and must never be copied into this repository.

Do not resubmit either bundle while job 20744510 is active. Do not add
downloaded archives or trajectories to Git.
