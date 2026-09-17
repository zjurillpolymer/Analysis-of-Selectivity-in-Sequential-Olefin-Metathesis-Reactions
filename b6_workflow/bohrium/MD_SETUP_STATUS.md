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
code 1: the Amber image lacked ParmEd. The entire remaining setup was then
performed locally with the `chem_dl` environment, which contains AmberTools
and ParmEd. All five solute boxes and the pure-DCM box passed
`verify_b6_md_setup.py`: LEaP reported zero errors; each GROMACS topology used
Lorentz–Berthelot combination rule 2 and rigid CH2Cl2-DC constraints; GRO atom
counts and vinyl mapping matched. Local GROMACS 2026.3 `grompp -maxwarn 0`
also passed for all five solute states. The GPU package is generated from these
local outputs, not from the failed Bohrium setup jobs.

The GPU image is GROMACS 2024.2, so its own `grompp` remains the final
version-specific check when MD starts. A previously passed 5 ns pure-DCM
density gate is reused only because the DCM model files match byte-for-byte;
the gate and model hashes are included in each GPU input bundle.

Five one-T4-per-state, three-replica, 2 ns pilot input bundles have been
prepared locally under
`/Users/arcadio/Molecular_dynamics_project/bohrium_submit/b6_neighbor_md_pilot_localsetup_v2_20260917/`.
The first GPU submission request was rejected before execution by the local
automatic permission reviewer (capacity error). No GPU MD job ID has been
issued; check Bohrium before retrying if a previous submission becomes
uncertain.
