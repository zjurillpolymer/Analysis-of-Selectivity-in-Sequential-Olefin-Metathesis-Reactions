# A4 umbrella protocol records

These are lightweight copies of selected completed or audited A4 batches:
path identities, window centers, atom indices, submission metadata and MDP
settings. The directory names retain the original batch names, including
`repair_v3` where that is the audited repair rather than an earlier attempt.

The simulation starting coordinates, checkpoints, solute/solvent topologies,
window trajectories and PMF outputs are intentionally not copied here. As a
result, `run_a4_selectivity_umbrella.py` cannot be pointed at these folders
and expected to restart them directly. Use the related `prepare_*` scripts
and the original trajectory archive to reconstruct a runnable work directory.

Some JSON files retain source paths from the machine on which a batch was
prepared. These are provenance references, not portable paths in this clone.
