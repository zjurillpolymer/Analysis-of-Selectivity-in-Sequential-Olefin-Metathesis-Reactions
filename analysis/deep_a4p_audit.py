#!/usr/bin/env python3
"""Independent A4-p normalization, time-block and graph-label audit.

No remote jobs or scientific input files are modified. All WHAM runs use a
fresh output tree and record input SHA-256 digests for provenance.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np
from rdkit import Chem
from analyze_a4p_downloads import xvg, interval_logweight, RT
from analyze_react1_structures import vinyl_mapping

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/a4p_detailed_analysis'
CONTACT=(.35,.5)
REFS=[(.6,1.0),(.75,1.25),(.9,1.5),(1.2,1.8),(1.5,2.1)]
BLOCKS=[('full',100,500),('b1',100,200),('b2',200,300),('b3',300,400),('b4',400,500),
        ('late200',300,500),('after200',200,500)]

def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def tail_mass(data,lower):
    if not np.any(np.isfinite(data[:,1]) & (data[:,0]>=lower)):
        return 0.0
    return math.exp(interval_logweight(data,(lower,3.))-interval_logweight(data,(.15,3.)))

def profile(folder,label):
    data=xvg(folder/f'pmf_{label}.xvg')
    h=xvg(folder/f'hist_{label}.xvg')
    data[h[:,1:].sum(axis=1)==0,1]=np.inf
    return data

def metrics(data,contact=CONTACT,reference=(.9,1.5),upper=3.):
    lc=interval_logweight(data,contact)
    lr=interval_logweight(data,reference)
    lz=interval_logweight(data,(.15,upper))
    return {'logK':lc-lr,'Pcontact_truncated':math.exp(lc-lz),'Preference_truncated':math.exp(lr-lz),
            'logPcontact_truncated':lc-lz,'dg_contact_reference_kj':-RT*(lc-lr)}

def topology():
    audit=json.loads((ROOT/'results/A4_p_C6_C8_amber_md_bundle/review/audit.json').read_text())
    selected=json.loads((ROOT/'results/a4p_bohrium_analysis/summary.json').read_text())['pathways']
    output=[]
    for sys in audit['systems']:
        name=sys['system']; arms=sys['arm_mapping_1based']
        molecule=Chem.MolFromPDBFile(str(ROOT/f'structures/{name}.pdb'),removeHs=False,sanitize=False)
        serial={a.GetPDBResidueInfo().GetSerialNumber():a.GetIdx() for a in molecule.GetAtoms()}
        vinyls=vinyl_mapping(ROOT/f'trace/{name}_md/md_simulation.log')
        heavy={a.GetIdx() for a in molecule.GetAtoms() if a.GetAtomicNum()>1}
        adj={i:{a.GetIdx() for a in molecule.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() in heavy} for i in heavy}
        core_evidence=[]
        for core in heavy:
            if len(adj[core])!=4:continue
            seen={core};components=[]
            for start in sorted(heavy):
                if start in seen:continue
                stack=[start];seen.add(start);comp=set()
                while stack:
                    v=stack.pop();comp.add(v)
                    for nb in adj[v]-seen: seen.add(nb);stack.append(nb)
                components.append(comp)
            groups=[[v for v,pair in vinyls.items() if {serial[s] for s in pair}<=c] for c in components]
            if len(groups)==4 and all(len(g)==2 for g in groups):
                core_evidence.append({'serial':molecule.GetAtomWithIdx(core).GetPDBResidueInfo().GetSerialNumber(),
                                      'vinyl_groups_after_cut':groups,'heavy_atoms_per_branch':[len(c) for c in components]})
        if len(core_evidence)!=1:raise ValueError(f'central four-arm carbon not uniquely identified: {name}')
        def edge(pair):return tuple(sorted(arms[v] for v in pair))
        def numbered_cycle(e):
            nums=sorted(int(x[1:]) for x in e)
            return nums in [[1,2],[2,3],[3,4],[1,4]]
        for branch,state in [('q1',sys['q1']),('adjacent',sys['q2']['adjacent']),('opposite',sys['q2']['opposite'])]:
            history=[edge(sys['q1']['reacted_pair'])]
            if branch!='q1':history.append(edge(state['reacted_pair']))
            pairs=[]
            for pair in itertools.combinations(state['active_vinyls'],2):
                e=edge(pair)
                label='same' if e[0]==e[1] else 'reuse' if e in history else 'switch'
                pairs.append({'pair':'-'.join(pair),'edge':e,'history_class':label,
                              'kept_by_numbered_cycle_filter':label=='same' or numbered_cycle(e)})
            chosen=[r['pathway'] for r in selected if r['metadata']['state']==state['state']]
            output.append({'system':name,'state':state['state'],'core_graph':core_evidence[0],
                           'arm_labels':arms,'historical_edges':history,
                           'history_passes_current_numbered_filter':all(numbered_cycle(e) for e in history),
                           'all_remaining_pairs_before_neighbour_filter':pairs,'selected_pathways':chosen})
    return output

def run_one(row):
    name=row['pathway']; src=Path(row['source_path']);folder=OUT/name;folder.mkdir(parents=True,exist_ok=True)
    tprs=[];pulls=[];windows=[]
    for i in range(25):
        stem=f'window_{i:03d}_{.35+.1*i:.3f}nm_production'
        tpr=src/(stem+'.tpr');pull=src/(stem+'_pullx.xvg');gro=src/(stem+'.gro')
        d=xvg(pull)
        if not gro.is_file() or not len(d) or abs(d[-1,0]-500)>1e-4:raise ValueError(stem)
        tprs.append(tpr);pulls.append(pull)
        r=d[(d[:,0]>=100)&(d[:,0]<=500),1]
        halves=[r[:len(r)//2],r[len(r)//2:]]
        windows.append({'index':i,'center':.35+.1*i,'samples':len(r),'min':float(r.min()),'max':float(r.max()),
                        'mean':float(r.mean()),'sd':float(r.std()),'half_mean_shift_nm':float(halves[1].mean()-halves[0].mean()),
                        'tpr_sha256':digest(tpr),'pullx_sha256':digest(pull)})
    (folder/'tprs.dat').write_text('\n'.join(map(str,tprs))+'\n')
    (folder/'pulls.dat').write_text('\n'.join(map(str,pulls))+'\n')
    signature=hashlib.sha256(json.dumps(windows,sort_keys=True).encode()).hexdigest()
    for label,begin,end in BLOCKS:
        cmd=['gmx','wham','-it',str(folder/'tprs.dat'),'-ix',str(folder/'pulls.dat'),'-temp','313.15','-unit','kJ',
             '-min','.15','-max','3.0','-bins','285','-b',str(begin),'-e',str(end),'-ac',
             '-oiact',str(folder/f'iact_{label}.xvg'),'-o',str(folder/f'pmf_{label}.xvg'),'-hist',str(folder/f'hist_{label}.xvg')]
        if label=='full':
            cmd+=['-nBootstrap','100','-bs-method','traj','-bs-seed',str(20260909+int(name.split('_')[2][1:])),
                  '-bsres',str(folder/'bootstrap_result.xvg'),'-bsprof',str(folder/'bootstrap_profiles.xvg')]
        record={'input_hash':signature,'command':cmd}
        cache=folder/f'{label}_provenance.json'
        if cache.is_file() and json.loads(cache.read_text())==record:continue
        with (folder/f'wham_{label}.log').open('w') as f:
            r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','GMX_MAXBACKUP':'-1'},timeout=180)
        if r.returncode:raise ValueError(f'WHAM failed: {name} {label}')
        cache.write_text(json.dumps(record,indent=2))
    full=profile(folder,'full');h=xvg(folder/'hist_full.xvg');hs=h[:,1:]/h[:,1:].sum(axis=0)[None,:]
    overlap=[float(np.minimum(hs[:,i],hs[:,i+1]).sum()) for i in range(24)]
    result={'pathway':name,'metadata':row['metadata'],'windows':windows,'overlaps':overlap,
            'metrics':{label:metrics(profile(folder,label)) for label,_,_ in BLOCKS},
            'contact_sensitivity':[{'interval':c,'metrics':metrics(full,contact=c)} for c in [(.3,.45),(.3,.5),(.35,.45),(.35,.5),(.35,.55),(.35,.6)]],
            'reference_sensitivity':[{'interval':r,'metrics':metrics(full,reference=r)} for r in REFS],
            'normalization_range_sensitivity':[{'upper':u,'metrics':metrics(full,upper=u)} for u in [2.,2.4,2.65,2.85,3.]],
            'tail_mass_above_2_5':tail_mass(full,2.5),
            'tail_mass_above_2_8':tail_mass(full,2.8),
            'density_mode_nm':float(full[np.argmin(full[:,1]),0])}
    bs=xvg(folder/'bootstrap_profiles.xvg').reshape(100,len(full),2)
    bs[:,:,1][:,h[:,1:].sum(axis=1)==0]=np.inf
    # The interval limits and finite-support mask are fixed across replicates.
    # These are conditional histogram-bootstrap diagnostics, not independent MD replicates.
    result['bootstrap_metrics']=[metrics(d) for d in bs]
    iact=[]
    for line in (folder/'iact_full.xvg').read_text().splitlines():
        parts=line.split()
        if len(parts)==3 and parts[0]=='#' and parts[1].isdigit():iact.append(float(parts[2]))
    result['iact_ps']=iact
    result['approx_effective_samples']=[801/(1+2*t/.5) for t in iact]
    (folder/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(name, result['metrics']['full'], 'last overlap',overlap[-1],flush=True)
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original=json.loads((ROOT/'results/a4p_bohrium_analysis/summary.json').read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run_one,original['pathways']))
    comparisons=[]
    rng=np.random.default_rng(7412)
    for prior in original['contrasts']:
        state=prior['state'];alt=prior['alternative'];ref=prior['reference']
        lookup={(r['metadata'].get('route_class') or r['metadata']['topology_class']):r for r in results if r['metadata']['state']==state}
        a,b=lookup[alt],lookup[ref]
        def contrast(ma,mb):
            return {'Kratio':math.exp(ma['logK']-mb['logK']),
                    'Pratio_truncated':math.exp(ma['logPcontact_truncated']-mb['logPcontact_truncated']),
                    'ddg_K_kj':-RT*(ma['logK']-mb['logK']),
                    'ddg_P_kj':-RT*(ma['logPcontact_truncated']-mb['logPcontact_truncated'])}
        boot=[contrast(a['bootstrap_metrics'][i],b['bootstrap_metrics'][j]) for i,j in zip(rng.integers(100,size=10000),rng.integers(100,size=10000))]
        ci={k:np.quantile([v[k] for v in boot],[.025,.5,.975]).tolist() for k in boot[0]}
        comparisons.append({'state':state,'alternative':alt,'reference':ref,
                            'time':{label:contrast(a['metrics'][label],b['metrics'][label]) for label,_,_ in BLOCKS},
                            'bootstrap_conditional_95pct':ci,
                            'contact_sensitivity':[{'interval':x['interval'],**contrast(x['metrics'],y['metrics'])} for x,y in zip(a['contact_sensitivity'],b['contact_sensitivity'])],
                            'reference_sensitivity':[{'interval':x['interval'],**contrast(x['metrics'],y['metrics'])} for x,y in zip(a['reference_sensitivity'],b['reference_sensitivity'])],
                            'normalization_range_sensitivity':[{'upper':x['upper'],**contrast(x['metrics'],y['metrics'])} for x,y in zip(a['normalization_range_sensitivity'],b['normalization_range_sensitivity'])]})
    out={'temperature_K':313.15,'contact_nm':CONTACT,'reference_nm':[.9,1.5],
         'normalization_domain_nm':[.15,3.0],'n_independent_trajectories_per_window':1,
         'bootstrap_warning':'Synthetic trajectory bootstrap within sampled basins; does not include hidden-basin or between-start uncertainty.',
         'topology':topology(),'pathways':results,'comparisons':comparisons}
    (OUT/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
    print(OUT/'audit.json')

if __name__=='__main__':main()
