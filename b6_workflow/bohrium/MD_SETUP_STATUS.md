# B6 neighbour q1/q2 explicit-DCM MD setup status

The Amber parameter sets are approved in `validated_parameter_sources.json`: q1
comes from Bohr job 20735563 and all four repaired q2 sets from 20744510.

On 2026-09-17, two CPU setup attempts (20747198 and 20747200) exited before
topology generation because the Amber image lacked Packmol. Packmol 21.2.1 was
then run locally with its cached Fortran runtime. Six boxes were generated and
revalidated: one pure DCM plus five B6 q1/q2 solute boxes. Their hashes are in
the local, unversioned bundle
`/Users/arcadio/Molecular_dynamics_project/bohrium_submit/b6_neighbor_md_setup_prepacked_20260917/workspace/prepacked_boxes.json`.

The corrected prepacked setup task, Bohr ID **20748125**, also exited with
code 1. Its detailed log has not yet been retrieved, so the cause is unconfirmed.
Do not submit GPU MD until its failure is diagnosed and all five `system.top`,
`system.gro`, `metadata.json` files pass `verify_b6_md_setup.py`. In particular,
do not equate completed Amber parameterization with completed solvated setup.

No B6 q1/q2 GPU MD job has been submitted as of this note.
