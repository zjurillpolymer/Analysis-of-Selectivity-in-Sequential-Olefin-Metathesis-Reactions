#!/usr/bin/env python3
"""Analyze the newly completed B6 q1 replica-2 umbrella paths."""
from pathlib import Path
import csv, json
from analyze_b6_q1_umbrella import wham_audit
from analyze_a4_m_c6_c7_q1_umbrella_2d import prepare_topology
from analyze_a4_q2_umbrella_2d_reweight import analyze_pathway, write_csv

PROJECT = Path(__file__).resolve().parents[1]
DOWNLOAD = PROJECT.parent / "bohrium_download/b6_neighbor_q1_umbrella_rep2_20260923"
OUTPUT = PROJECT / "results/b6_local_q1_arm1_competition_20260923"
JOBS = (20771983, 20771984)

rows=[]
topology_root=OUTPUT/"reactive_pair_topologies_rep2"
for job in JOBS:
    pathway=DOWNLOAD/str(job)/"pathway"
    meta=json.loads((pathway/"pathway.json").read_text())
    audit=wham_audit(pathway)
    prepare_topology(pathway,topology_root)
    result=analyze_pathway(pathway,PROJECT,OUTPUT,"gmx",topology_root)
    row={"bohr_id":job,"pathway":meta["pathway"],"orbit":meta["route_family"],"pair":f"V{meta['left']}-V{meta['right']}","replica":meta["replica"],"orbit_multiplicity":meta["orbit_multiplicity"],"windows":len(result.centers_nm),"frames":len(result.distance_nm),"mbar_iterations":result.mbar_iterations,"mbar_residual":result.mbar_residual,"effective_sample_size":result.effective_sample_size,**audit,"geometry_pull_rms_A":result.geometry_pull_rms_A,"geometry_pull_max_A":result.geometry_pull_max_A,**result.metrics}
    for half,metrics in result.half_metrics.items(): row.update({f"{half}_{k}":v for k,v in metrics.items()})
    rows.append(row)
    print(job,meta["route_family"],"NAC",result.metrics["soft_nac_probability"],"ESS",result.effective_sample_size)
write_csv(OUTPUT/"replica2_pathway_audit.csv",rows)
(OUTPUT/"replica2_pathway_audit.json").write_text(json.dumps(rows,indent=2)+"\n")
