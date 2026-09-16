#!/usr/bin/env python3
"""Audit native Colvars Rg WT-MetaD; screen discoveries, NOT product probabilities.

Reference and biased occupancies measure coverage only. Neither the raw MetaD
contact frequency nor the Rg PMF is propagated to reaction-path yields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.XTC import XTCReader
from scipy.ndimage import binary_dilation, label
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
N = 247
DT_PS = 0.002
VINYLS = np.array([[56,106],[63,107],[70,108],[77,109],[84,110],[105,111]]) - 1
TRI = np.triu_indices(6, 1)
RG_EDGES = np.arange(0.30, 2.201, 0.025)
K2_EDGES = np.arange(0.0, 1.001, 0.02)
FEATURES = ('time_ps','rg_nm','kappa2','distance_nm','axis_deg','side_deg',
            'soft_contact','soft_nac','max_bond_nm')


def numeric(path: Path) -> np.ndarray:
    return np.loadtxt(path, comments='#', ndmin=2)


def unique_time(data: np.ndarray) -> np.ndarray:
    # Prefer the last record at a continuation boundary; retain chronological order.
    _, rev = np.unique(np.round(data[::-1,0], 5), return_index=True)
    return data[np.sort(len(data)-1-rev)]


def topology(path: Path) -> tuple[np.ndarray, list[tuple[int,int]], list[str]]:
    section = ''
    mol = 0
    masses, bonds, names = [], [], []
    for line in path.read_text().splitlines():
        line = line.split(';',1)[0].strip()
        if not line:
            continue
        if line.startswith('['):
            section = line.strip('[] ').strip()
            if section == 'moleculetype':
                mol += 1
            continue
        if mol != 1:
            continue
        fields = line.split()
        if section == 'atoms':
            names.append(fields[4])
            masses.append(float(fields[7]))
        elif section in ('bonds','constraints'):
            bonds.append((int(fields[0])-1, int(fields[1])-1))
    if len(names) != N:
        raise ValueError(f'Expected {N} solute atoms, got {len(names)}')
    return np.array(masses), sorted(set(bonds)), names


def soft_upper(x: np.ndarray, center: float, width: float) -> np.ndarray:
    return 1 / (1 + np.exp(np.clip((x-center)/width,-700,700)))


def extract(xtc: Path, gro: Path, masses: np.ndarray, bonds: list,
            names: list[str], cache: Path) -> tuple[np.ndarray,np.ndarray]:
    key = hashlib.sha256((str(xtc.resolve())+str(xtc.stat().st_size)+
                          str(xtc.stat().st_mtime_ns)).encode()).hexdigest()[:20]
    dest = cache / f'{key}.npz'
    if dest.exists():
        with np.load(dest) as z:
            return z['data'], z['fingerprint']
    u = mda.Universe(str(gro))
    if u.atoms[:N].names.tolist() != names:
        raise ValueError(f'Solute atom order mismatch: {gro}')
    with XTCReader(str(xtc)) as reader:
        count = reader.n_atoms
    if count == N:
        u = mda.Merge(u.atoms[:N])
    elif count != u.atoms.n_atoms:
        raise ValueError(f'Trajectory/topology atom mismatch: {xtc}, {count}')
    u.add_TopologyAttr('bonds', bonds)
    u.load_new(str(xtc))
    solute = u.atoms[:N]
    if len(solute.fragments) != 1:
        raise ValueError('Solute bond graph is not a single connected fragment')
    bi,bj = np.array(bonds).T
    records, fingerprints = [], []
    last = -np.inf
    for ts in u.trajectory:
        if ts.time < last + 5 - 1e-4:
            continue
        last = ts.time
        # GRO/XTC readers provide Angstrom coordinates; reconstructed positions -> nm.
        p = solute.unwrap(compound='fragments', reference=None, inplace=False).astype(float)/10
        centered = p-p.mean(axis=0)
        eig = np.linalg.eigvalsh(centered.T @ centered / N)
        rg = np.sqrt(eig.sum())
        k2 = 1 - 3*(eig[0]*eig[1]+eig[0]*eig[2]+eig[1]*eig[2])/eig.sum()**2
        ends = p[VINYLS]
        mids = ends.mean(axis=1)
        axes = ends[:,1]-ends[:,0]
        axes /= np.linalg.norm(axes,axis=1)[:,None]
        delta = mids[5]-mids[4]
        distance = np.linalg.norm(delta)
        axis = np.degrees(np.arccos(np.clip(abs(axes[4] @ axes[5]),0,1)))
        side = min(np.degrees(np.arccos(np.clip(abs(axes[k] @ delta)/distance,0,1)))
                   for k in (4,5))
        contact = soft_upper(distance, 0.5, 0.05)
        orient = soft_upper(axis,45,7.5)*(1-soft_upper(side,45,7.5))
        bondmax = np.max(np.linalg.norm(p[bi]-p[bj],axis=1))
        records.append([ts.time,rg,k2,distance,axis,side,contact,contact*orient,bondmax])
        fingerprints.append(np.linalg.norm(mids[TRI[0]]-mids[TRI[1]],axis=1))
    a,f = np.array(records),np.array(fingerprints)
    cache.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest,data=a,fingerprint=f)
    return a,f


def combine(parts: list[tuple[np.ndarray,np.ndarray]], begin: float) -> tuple[np.ndarray,np.ndarray]:
    data = np.concatenate([a for a,_ in parts])
    fingerprints = np.concatenate([f for _,f in parts])
    idx = np.argsort(data[:,0],kind='stable')
    data,fingerprints = data[idx],fingerprints[idx]
    _, rev = np.unique(np.round(data[::-1,0],5),return_index=True)
    idx = np.sort(len(data)-1-rev)
    data,fingerprints = data[idx],fingerprints[idx]
    keep = data[:,0] >= begin-1e-5
    return data[keep],fingerprints[keep]


def histogram(data: np.ndarray) -> np.ndarray:
    return np.histogram2d(data[:,1],data[:,2],bins=(RG_EDGES,K2_EDGES))[0]


def mask_frames(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    i = np.searchsorted(RG_EDGES,data[:,1],side='right')-1
    j = np.searchsorted(K2_EDGES,data[:,2],side='right')-1
    valid = (i>=0)&(j>=0)&(i<mask.shape[0])&(j<mask.shape[1])
    result = np.zeros(len(data),dtype=bool)
    result[valid] = mask[i[valid],j[valid]]
    return result


def visits(time: np.ndarray, mask: np.ndarray) -> list[dict]:
    blocks = []
    start = None
    for i,on in enumerate(np.r_[mask,False]):
        if on and start is None:
            start = i
        if not on and start is not None:
            blocks.append({'start_ps':float(time[start]),'end_ps':float(time[i-1]),
                           'duration_ps':float(time[i-1]-time[start]),
                           'frames':i-start,'start_index':start})
            start = None
    return sorted(blocks,key=lambda d:d['duration_ps'],reverse=True)


def synopsis(data: np.ndarray) -> dict:
    out = {'frames':len(data),'start_ps':float(data[0,0]),'end_ps':float(data[-1,0])}
    for col,name in ((1,'rg_nm'),(2,'kappa2'),(3,'distance_nm')):
        out[name] = dict(zip(('min','q01','median','q99','max'),
                            map(float,np.quantile(data[:,col],[0,.01,.5,.99,1]))))
    out.update({'mean_soft_contact_geometry':float(data[:,6].mean()),
                'mean_soft_nac_geometry':float(data[:,7].mean()),
                'hard_contact_frames':int((data[:,3]<=.5).sum()),
                'hard_nac_frames':int(((data[:,3]<=.5)&(data[:,4]<=45)&(data[:,5]>=45)).sum()),
                'max_bond_nm':float(data[:,8].max())})
    return out


def export_solute(rep: Path, time: float, destination: Path, bonds: list) -> None:
    """Export coordinates only: these 247-atom GROs need re-solvation before MD."""
    original=mda.Universe(str(rep/'start.gro'))
    u=mda.Merge(original.atoms[:N])
    u.add_TopologyAttr('bonds',bonds)
    u.load_new(str(rep/'wtmeta.xtc'))
    frame=int(round(time/5))
    ts=u.trajectory[frame]
    if abs(ts.time-time)>.001:
        raise ValueError('Representative frame time mismatch')
    whole=u.atoms.unwrap(compound='fragments',reference=None,inplace=False)
    whole+=ts.dimensions[:3]/2-whole.mean(axis=0)
    u.atoms.positions=whole
    with mda.Writer(str(destination),n_atoms=N) as writer:
        writer.write(u.atoms)


def reference_paths(root: Path) -> tuple[dict,dict]:
    old,extensions = {},{}
    for dirname in ('a4_m_c5_c8_q1_three_class_umbrella_download',
                    'a4_m_c5_c8_q1_three_class_umbrella_rep3_download'):
        for p in (root/'results'/dirname).glob('*/extracted/pathway/pathway.json'):
            meta = json.loads(p.read_text())
            if meta.get('system')=='A4_m_C8' and meta.get('stage')=='q1':
                old[meta['pathway']] = p.parent
    for p in (root/'results/a4_umbrella_extensions_fix1_extracted').glob('*/pathway/pathway.json'):
        meta = json.loads(p.read_text())
        if meta.get('system')=='A4_m_C8' and meta.get('stage')=='q1':
            extensions[meta['pathway']] = p.parent
    return old,extensions


def energy(gmx: str, rep: Path, output: Path) -> dict:
    xvg = output/f'{rep.name}_temperature_density.xvg'
    p = subprocess.run([gmx,'energy','-f',str(rep/'wtmeta.edr'),'-o',str(xvg),'-b','2000'],
                       input='Temperature\nDensity\n0\n',text=True,
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (output/f'{rep.name}_energy_audit.txt').write_text(p.stdout)
    if p.returncode:
        raise RuntimeError(p.stdout[-2000:])
    a = np.loadtxt(xvg,comments=('#','@'),ndmin=2)
    return {'temperature_K_mean':float(a[:,1].mean()),'temperature_K_std':float(a[:,1].std()),
            'density_kg_m3_mean':float(a[:,2].mean()),'density_kg_m3_std':float(a[:,2].std())}


def plot(output: Path, meta: dict, refs: dict, pmfs: dict, support: np.ndarray) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                         'font.size':7,'axes.spines.right':False,'axes.spines.top':False,
                         'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.8})
    colors = ['#4477AA','#CC6677','#228833']
    fig = plt.figure(figsize=(7.2,5.6),layout='constrained')
    grid=fig.add_gridspec(3,2,height_ratios=(1,1.2,.8))
    ax=fig.add_subplot(grid[0,:])
    for color,(name,data) in zip(colors,meta.items()):
        ax.plot(data[:,0]/1000,data[:,1],lw=.55,color=color,label=name,alpha=.85)
    ax.set(xlabel='Time (ns)',ylabel='Rg (nm)',title='a  Global-CV exploration (biased trajectories)')
    ax.legend(ncol=3,fontsize=7,loc='upper right')
    ax=fig.add_subplot(grid[1,0])
    allref=np.concatenate([d for name,d in refs.items() if name.startswith('umbrella_')])
    ax.scatter(allref[::3,1],allref[::3,2],s=1,c='.75',rasterized=True,label='q1 umbrella coverage')
    md=np.concatenate([d for name,d in refs.items() if name.startswith('md_')])
    ax.scatter(md[::3,1],md[::3,2],s=2,c='black',alpha=.4,rasterized=True,label='q1 unbiased MD')
    for color,(name,data) in zip(colors,meta.items()):
        ax.scatter(data[:,1],data[:,2],s=2,c=color,alpha=.4,rasterized=True,label=name)
    ax.set(xlabel='Geometric Rg (nm)',ylabel='Relative shape anisotropy',title='b  Coverage, not equilibrium population')
    ax.legend(fontsize=6,loc='best',markerscale=2)
    ax=fig.add_subplot(grid[1,1])
    for color,(name,data) in zip(colors,meta.items()):
        ax.scatter(data[:,1],data[:,3],s=2,c=color,alpha=.35,rasterized=True)
    ax.axhline(.5,color='.2',ls='--',lw=.8)
    ax.set(xlabel='Rg (nm)',ylabel='V5–V7 midpoint distance (nm)',title='c  The reaction distance is not biased')
    ax=fig.add_subplot(grid[2,:])
    for color,(name,series) in zip(colors,pmfs.items()):
        x,f=series[5000000].T
        offset=f[support].min()
        ax.plot(x[support],(f[support]-offset)/4.184,color=color,label=f'{name}, 10 ns')
        f5=series[2500000][:,1]
        offset5=f5[support].min()
        ax.plot(x[support],(f5[support]-offset5)/4.184,color=color,ls='--',alpha=.65)
    ax.set(xlabel='Rg (nm)',ylabel='Rg PMF estimate (kcal mol$^{-1}$)',
           title='d  Solid: 10 ns; dashed: 5 ns. Shown only on common visited support')
    for suffix in ('png','svg','pdf'):
        fig.savefig(output/f'colvars_sampling_diagnostics.{suffix}',dpi=300)
    plt.close(fig)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--metad-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=ROOT/'results/c8_q1_colvars_wtmetad_analysis_20260914')
    parser.add_argument('--gmx',default='gmx')
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    cache=output/'cache'
    masses,bonds,names=topology(args.metad_root/'rep1/system.top')
    meta,meta_f,pmfs,audits={},{},{},{}
    for name in ('rep1','rep2','rep3'):
        rep=args.metad_root/name
        all_data,f=extract(rep/'wtmeta.xtc',rep/'start.gro',masses,bonds,names,cache)
        cv=unique_time(numeric(rep/'wtmeta.colvars.traj'))
        keep=all_data[:,0]>=2000
        meta[name],meta_f[name]=all_data[keep],f[keep]
        hill=numeric(rep/'wtmeta.colvars.wtmeta.hills.traj')
        log=(rep/'wtmeta.log').read_text(errors='replace')
        audits[name]={'full':synopsis(all_data),'post_2ns':synopsis(meta[name]),
                      'cv_rg_rms_error_nm':float(np.sqrt(np.mean((all_data[:,1]-cv[:,1])**2))),
                      'cv_rg_max_error_nm':float(np.max(abs(all_data[:,1]-cv[:,1]))),
                      'cv_times_max_error_ps':float(np.max(abs(all_data[:,0]-cv[:,0]*DT_PS))),
                      'hill_count':len(hill),'hill_first_step':float(hill[0,0]),
                      'hill_last_step':float(hill[-1,0]),'hill_sigma_nm':float(np.median(hill[:,2])),
                      'last_1000_hills_height_mean_kJ_mol':float(hill[-1000:,3].mean()),
                      'first_1000_hills_height_mean_kJ_mol':float(hill[:1000,3].mean()),
                      'finished':bool(re.search('Finished mdrun on rank 0',log)),
                      'lincs_warning_count':log.count('LINCS WARNING'),
                      'input_sha256':hashlib.sha256((rep/'start.gro').read_bytes()).hexdigest(),
                      **energy(args.gmx,rep,output)}
        audits[name]['hard_contact_visits']=visits(meta[name][:,0],meta[name][:,3]<=.5)
        hard=(meta[name][:,3]<=.5)&(meta[name][:,4]<=45)&(meta[name][:,5]>=45)
        audits[name]['hard_nac_visits']=visits(meta[name][:,0],hard)
        pmfs[name]={step:numeric(rep/f'wtmeta.{step}.pmf')
                    for step in (2500000,4000000,5000000)}
        print(f'Audited {name}: {len(all_data)} frames',flush=True)
    refs,ref_f,manifest={},{},[]
    mdroot=args.root/'results/a4_m_c5_c8_q1_10ns_download'
    for job,rep in (('20730422',1),('20730424',2),('20730423',3)):
        base=mdroot/job/'extracted/workspace/gromacs_dcm/runs/A4_m_C8_1_cis'/f'replica_{rep:02d}'
        xtcs=[base/'production_2ns.xtc']+sorted((base/'extension_segments').glob('**/*.xtc'))
        parts=[extract(x,base/'production_2ns.gro',masses,bonds,names,cache) for x in xtcs]
        # Coverage is conservative: include the discarded early production segment.
        data,f=combine(parts,0)
        refs[f'md_rep{rep}'],ref_f[f'md_rep{rep}']=data,f
        manifest.append({'reference':f'md_rep{rep}','files':[str(x) for x in xtcs],
                         'frames':len(data),'begin_ps':float(data[0,0]),'end_ps':float(data[-1,0])})
        print(f'Reference unbiased rep{rep}: {len(data)} frames',flush=True)
    old,ext=reference_paths(args.root)
    for alias,path in sorted(old.items()):
        if alias not in ext:
            raise FileNotFoundError(f'Missing extension for {alias}')
        parts=[]
        for win in sorted((path/'windows').glob('window_*')):
            xtcs=[win/'production.xtc',ext[alias]/'windows'/win.name/'extension.xtc']
            pair=[extract(x,path/'start.gro',masses,bonds,names,cache) for x in xtcs]
            parts.append(combine(pair,0))
        data=np.concatenate([a for a,_ in parts]);f=np.concatenate([b for _,b in parts])
        refs['umbrella_'+alias],ref_f['umbrella_'+alias]=data,f
        manifest.append({'reference':'umbrella_'+alias,'frames':len(data),
                         'windows':len(parts),'original':str(path),'extension':str(ext[alias])})
        print(f'Reference {alias}: {len(parts)} windows, {len(data)} frames',flush=True)
    allref=np.concatenate(list(refs.values()))
    for data in [allref,*meta.values()]:
        if not ((data[:,1]>=RG_EDGES[0])&(data[:,1]<RG_EDGES[-1])&
                (data[:,2]>=K2_EDGES[0])&(data[:,2]<K2_EDGES[-1])).all():
            raise ValueError('Coverage grid excludes observed frames; expand grid, do not discard them')
    refmask=histogram(allref)>0
    expanded=binary_dilation(refmask,structure=np.ones((3,3)))
    allmeta=np.concatenate(list(meta.values()))
    novelmask=(histogram(allmeta)>0)&~expanded
    components,ncomp=label(novelmask,structure=np.ones((3,3)))
    candidate=[]
    for k in range(1,ncomp+1):
        mask=components==k;cells=int(mask.sum())
        records=[]
        for name,data in meta.items():
            flag=mask_frames(data,mask);one=visits(data[:,0],flag)
            record={'replica':name,'frames':int(flag.sum()),
                            'fraction':float(flag.mean()),'longest_visit_ps':one[0]['duration_ps'] if one else 0,
                            'longest_visits':one[:3],
                            'visits_ge_100ps':sum(d['duration_ps']>=100 for d in one)}
            if flag.any():
                record['geometry']=synopsis(data[flag])
            records.append(record)
        count=sum(d['frames'] for d in records)
        if cells>=3 and count/len(allmeta)>=.01:
            occupied=np.argwhere(mask)
            candidate.append({'component_label':k,'cells':cells,'frames':count,'biased_frame_fraction':count/len(allmeta),
                              'rg_bin_range_nm':[float(RG_EDGES[occupied[:,0].min()]),float(RG_EDGES[occupied[:,0].max()+1])],
                              'k2_bin_range':[float(K2_EDGES[occupied[:,1].min()]),float(K2_EDGES[occupied[:,1].max()+1])],
                              'records':records,'persistent_in_2rep':sum(d['visits_ge_100ps']>0 for d in records)>=2})
    # Rotation-invariant, labelled 6-vinyl distance fingerprint; independent of Rg bins.
    # Calibrate the coverage threshold with held-out reference repeats.
    reference_fp=np.concatenate(list(ref_f.values()))
    calibration=[]
    for rep in (1,2,3):
        held=np.concatenate([f for name,f in ref_f.items() if name.endswith(f'rep{rep}')])
        train=np.concatenate([f for name,f in ref_f.items() if not name.endswith(f'rep{rep}')])
        distances=cKDTree(train[::2]).query(held[::10],workers=2)[0]/np.sqrt(15)
        calibration.extend(distances.tolist())
    threshold=float(np.quantile(calibration,.99))
    tree=cKDTree(reference_fp)
    for name,data in meta.items():
        distances=tree.query(meta_f[name],workers=2)[0]/np.sqrt(15)
        flag=distances>threshold
        one=visits(data[:,0],flag)
        audits[name]['novelty']={'raw_new_rg_k2_frame_fraction':float(mask_frames(data,~refmask).mean()),
                                'buffered_new_rg_k2_frame_fraction':float(mask_frames(data,~expanded).mean()),
                                'fingerprint_coverage_threshold_nm':threshold,
                                'fingerprint_nn_rms_nm_median':float(np.median(distances)),
                                'fingerprint_nn_rms_nm_q99':float(np.quantile(distances,.99)),
                                'fingerprint_nn_rms_nm_max':float(distances.max()),
                                'fingerprint_novel_frame_fraction':float(flag.mean()),
                                'fingerprint_longest_novel_visit_ps':one[0]['duration_ps'] if one else 0,
                                'fingerprint_visits_ge_100ps':sum(d['duration_ps']>=100 for d in one)}
        audits[name]['compact_threshold_sensitivity']={}
        for cutoff in (.72,.75,.78):
            low=data[:,1]<cutoff
            one=visits(data[:,0],low)
            audits[name]['compact_threshold_sensitivity'][str(cutoff)]={
                'frames':int(low.sum()),'biased_fraction':float(low.mean()),
                'reference_frames_below_threshold':int((allref[:,1]<cutoff).sum()),
                'longest_visit_ps':one[0]['duration_ps'] if one else 0,
                'geometry':synopsis(data[low]) if low.any() else None}
        np.savez_compressed(output/f'{name}_source_data.npz',data=data,features=np.array(FEATURES),
                            fingerprint=meta_f[name],fingerprint_nn_rms_nm=distances)
    x=pmfs['rep1'][5000000][:,0]
    support=np.ones(len(x),bool)
    for data in meta.values():
        lower,upper=np.quantile(data[:,1],[.01,.99]);support&=(x>=lower)&(x<=upper)
    if support.sum()<3:
        raise ValueError('Insufficient common visited PMF support')
    final=[]
    for name,series in pmfs.items():
        f10=series[5000000][:,1]
        def diff(f):
            d=(f10[support]-f[support])/4.184;d-=d.mean()
            return {'rms_kcal_mol':float(np.sqrt(np.mean(d*d))),
                    'max_abs_kcal_mol':float(np.max(abs(d)))}
        audits[name]['pmf']={'common_support_nm':[float(x[support].min()),float(x[support].max())],
                             'common_support_grid_points':int(support.sum()),
                             '5_vs_10ns':diff(series[2500000][:,1]),
                             '8_vs_10ns':diff(series[4000000][:,1]),
                             'final_min_visited_rg_nm':float(x[support][np.argmin(f10[support])])}
        aligned=f10[support]/4.184;final.append(aligned-aligned.mean())
    matrix=[]
    for i in range(3):
        matrix.append([float(np.sqrt(np.mean((final[i]-final[j])**2))) for j in range(3)])
    reference_summaries={name:synopsis(data) for name,data in refs.items()}
    representatives=[]
    for component in candidate:
        if not component['persistent_in_2rep']:
            continue
        mask=components==component['component_label']
        for name,data in meta.items():
            flag=mask_frames(data,mask)
            blocks=visits(data[:,0],flag)
            if not blocks:
                continue
            longest=blocks[0]
            middle=longest['start_index']+(longest['frames']-1)//2
            best=np.flatnonzero(flag)[np.argmax(data[flag,7])]
            for kind,index in (('longest_visit_midpoint',middle),('best_soft_nac',best)):
                dest=output/f'candidate_{component["component_label"]}_{name}_{kind}_solute_only.gro'
                export_solute(args.metad_root/name,float(data[index,0]),dest,bonds)
                representatives.append({'file':dest.name,'replica':name,'kind':kind,
                                        **dict(zip(FEATURES,map(float,data[index]))),
                                        'solute_atoms_only':True,'must_resolvate_before_md':True})
    summary={'system':'A4_m_C8_1_cis','target_pair':'V5-V7','scope':'supplemental basin exploration only',
             'actual_bias_cvs':['geometric equal-atom-weight Rg'],
             'not_biased':['V5-V7 reaction distance','kappa2','orientation'],
             'duration_ns_per_replica':10,'post_start_ns':2,'meta_replicas':audits,
             'reference_frames':len(allref),'reference_umbrella_paths':len(old),
             'reference_coverage_begin_ps':0,
             'reference_rg_min_nm':float(allref[:,1].min()),
             'reference_rg_max_nm':float(allref[:,1].max()),
             'reference_umbrella_windows':sum(d.get('windows',0) for d in manifest),
             'reference_summary':reference_summaries,'candidate_new_components':candidate,
             'reproducible_persistent_rg_k2_candidate':any(d['persistent_in_2rep'] for d in candidate),
             'reference_fingerprint_heldout_q99_nm':threshold,
             'pmf_crossrep_rms_kcal_mol':matrix,
             'representative_structures':representatives,
             'primary_umbrella_result_not_modified':True}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'reference_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    np.savez_compressed(output/'reference_source_data.npz',**refs)
    plot(output,meta,refs,pmfs,support)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('reference_summary','meta_replicas')},indent=2))


if __name__=='__main__':
    main()
