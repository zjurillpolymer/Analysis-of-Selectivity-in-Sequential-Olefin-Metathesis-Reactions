#!/usr/bin/env python3
"""Read-only pilot contact/reference diagnostic, without the numbered-arm filter."""
import itertools
import json
from pathlib import Path
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/a4p_detailed_analysis'
RUNS=ROOT/'results/A4_p_C6_C8_unbiased_pilot_full_trajectories/gromacs_dcm/runs'

def main():
    rows=[]
    audit=json.loads((OUT/'audit.json').read_text())
    for topo in audit['topology']:
        state=topo['state'];state_root=RUNS/state
        metadata=json.loads((state_root/'setup/metadata.json').read_text())
        mapping=metadata['vinyl_mapping_gromacs_1based']
        labels=sorted(mapping,key=lambda v:int(v[1:]))
        atoms=np.array([mapping[l] for l in labels])-1
        for replica in range(1,4):
            values={p['pair']:[] for p in topo['all_remaining_pairs_before_neighbour_filter']}
            with XTCReader(str(state_root/f'replica_{replica:02d}/production_2ns.xtc')) as reader:
                for ts in reader:
                    if ts.time < 200 or ts.time > 2000: continue
                    xyz=ts.positions*.1;box=ts.dimensions[:3]*.1
                    if not np.allclose(ts.dimensions[3:],90): raise ValueError('triclinic box not supported')
                    vector=xyz[atoms[:,1]]-xyz[atoms[:,0]]
                    vector-=box*np.round(vector/box)
                    if not np.all((np.linalg.norm(vector,axis=1)>.10)&(np.linalg.norm(vector,axis=1)<.20)):
                        raise ValueError(f'vinyl mapping length mismatch {state}')
                    centers=xyz[atoms[:,0]]+vector/2
                    for pair in values:
                        left,right=pair.split('-')
                        delta=centers[labels.index(left)]-centers[labels.index(right)]
                        delta-=box*np.round(delta/box)
                        values[pair].append(float(np.linalg.norm(delta)))
            for p in topo['all_remaining_pairs_before_neighbour_filter']:
                d=np.array(values[p['pair']]);n=len(d)
                rows.append({'state':state,'replica':replica,**p,'n_frames':n,
                             'Pcontact_035_050':float(np.mean((d>=.35)&(d<=.5))),
                             'Pcontact_le050':float(np.mean(d<=.5)),
                             'Preference_090_150':float(np.mean((d>=.9)&(d<=1.5))),
                             'distance_min_nm':float(d.min()),'distance_median_nm':float(np.median(d))})
    out={'begin_ps':200,'end_ps':2000,'n_replicas':3,'warning':'Pilot diagnostics only; sparse events and initial-state memory may dominate. No new trajectory or reaction is generated.', 'pairs':rows}
    (OUT/'pilot_reference_audit.json').write_text(json.dumps(out,indent=2)+'\n')
    for r in audit['pathways']:
        m=r['metadata'];pair=f"V{m['left']}-V{m['right']}"
        selected=[v for v in rows if v['state']==m['state'] and v['pair']==pair]
        print(r['pathway'], 'pilot Pref',[round(v['Preference_090_150'],3) for v in selected], 'pilot Pcontact',[round(v['Pcontact_035_050'],4) for v in selected])

if __name__=='__main__':main()
