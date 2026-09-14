# Bohrium submission record

`amber_input/` is the current direct AmberTools submission bundle for the five
new B6 neighbour-only states:

- q1: `B6_p_C6_q1_V1_V2_cis`;
- q2: `full_reuse`, `one_arm_reuse`, `full_switch_d1`, and `full_switch_d2`.

The job runs only GAFF2/AM1-BCC parameterisation in the Amber image. Packing,
tleap/ParmEd conversion and GROMACS runs are separate stages. The checked-in
`submission.json` records the completed submission IDs; it is provenance only,
not a credential. AccessKey/SSO state remains in the local Bohrium profile and
must never be copied into this repository.

To resubmit from this directory after logging in with `bohr auth login`, use
the platform's normal job-submit command with `job.json` and `run_amber.sh` as
the input. Do not add downloaded archives or trajectory products to Git.
