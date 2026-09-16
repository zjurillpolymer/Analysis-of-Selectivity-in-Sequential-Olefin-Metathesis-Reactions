#!/usr/bin/env python3
"""Audit downloaded A4-p windows and run matched-range WHAM (no kinetics)."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RT = 0.008314462618 * 313.15
CONTACT = (0.35, 0.50)
REFERENCE = (0.90, 1.50)

def xvg(path):
    rows = [[float(x) for x in line.split()] for line in path.read_text().splitlines()
            if line.strip() and line.lstrip()[0] not in '#@&']
    return np.array(rows) if rows else np.empty((0, 2))

def interval_logweight(data, limits):
    x, y = data[:, 0], data[:, 1]
    finite = np.isfinite(y)
    if not finite.any():
        raise ValueError('no finite PMF')
    offset = np.min(y[finite])
    w = np.zeros(len(y))
    w[finite] = np.exp(-(y[finite] - offset) / RT)
    a, b = limits
    mask = (x > a) & (x < b)
    xx = np.r_[a, x[mask], b]
    yy = np.r_[np.interp(a, x, w), w[mask], np.interp(b, x, w)]
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    total = integrate(yy, xx)
    if total <= 0:
        raise ValueError(f'no interval weight: {limits}')
    return math.log(total) - offset / RT

def dg(path, histogram=None):
    data = xvg(path)
    # GROMACS writes 0, not +inf, in unsampled PMF bins. They must not
    # acquire fictitious Boltzmann weight during interval integration.
    if histogram is not None:
        hist = xvg(histogram)
        if not np.allclose(hist[:, 0], data[:, 0]):
            raise ValueError('PMF/histogram grid mismatch')
        data[hist[:, 1:].sum(axis=1) == 0, 1] = np.inf
    return -RT * (interval_logweight(data, CONTACT) - interval_logweight(data, REFERENCE))

def audit(path):
    rows = []
    for mdp in sorted((path / 'window_mdps').glob('*_production.mdp')):
        stem = mdp.stem
        pull = path / (stem + '_pullx.xvg')
        files_ok = all((path / (stem + suffix)).is_file() for suffix in ('.tpr', '.gro', '_pullx.xvg'))
        raw = xvg(pull) if pull.is_file() else None
        complete = bool(files_ok and raw is not None and len(raw) and np.isfinite(raw).all() and abs(raw[-1, 0] - 500) < .001)
        row = {'window': stem, 'complete': complete, 'last_ps': float(raw[-1, 0]) if raw is not None and len(raw) else None}
        if complete:
            r = raw[(raw[:, 0] >= 100) & (raw[:, 0] <= 500), 1]
            row.update({'min_nm': float(r.min()), 'max_nm': float(r.max()), 'mean_nm': float(r.mean()), 'samples': len(r)})
        rows.append(row)
    return rows

def wham(path, dest, nwindow, label, begin, end, bootstraps):
    dest.mkdir(parents=True, exist_ok=True)
    stems = [f'window_{i:03d}_{.35 + .1*i:.3f}nm_production' for i in range(nwindow)]
    tprs = [path / (s + '.tpr') for s in stems]
    pulls = [path / (s + '_pullx.xvg') for s in stems]
    for p in tprs + pulls:
        if not p.is_file(): raise FileNotFoundError(p)
    (dest / 'tpr-files.dat').write_text('\n'.join(map(str, tprs)) + '\n')
    (dest / 'pullx-files.dat').write_text('\n'.join(map(str, pulls)) + '\n')
    out = dest / f'pmf_{label}.xvg'
    command = ['gmx', 'wham', '-it', str(dest/'tpr-files.dat'), '-ix', str(dest/'pullx-files.dat'),
               '-temp', '313.15', '-unit', 'kJ', '-min', '0.15', '-max', '3.0', '-bins', '285',
               '-b', str(begin), '-e', str(end), '-ac', '-oiact', str(dest/f'iact_{label}.xvg'),
               '-o', str(out), '-hist', str(dest/f'histograms_{label}.xvg')]
    if bootstraps:
        command += ['-nBootstrap', str(bootstraps), '-bs-method', 'traj', '-bs-seed', '20260909',
                    '-bsres', str(dest/'bootstrap_result.xvg'), '-bsprof', str(dest/'bootstrap_profiles.xvg')]
    # Reuse only if this exact invocation completed previously.
    command_file = dest / f'command_{label}.json'
    if not (out.is_file() and command_file.is_file() and json.loads(command_file.read_text()) == command):
        env = {**os.environ, 'OMP_NUM_THREADS': '1', 'GMX_MAXBACKUP': '-1'}
        with (dest/f'wham_{label}.log').open('w') as f:
            result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT, env=env, timeout=240)
        if result.returncode:
            raise RuntimeError(f'WHAM failed: {dest}/wham_{label}.log')
        command_file.write_text(json.dumps(command, indent=2))
    return dg(out, dest/f'histograms_{label}.xvg')

def process(task):
    path, output, bootstraps = task
    metadata = json.loads((path/'pathway.json').read_text())
    rows = audit(path)
    if len(rows) != 25 or not all(r['complete'] for r in rows[:24]):
        raise ValueError(f'common 24 windows incomplete: {path}')
    dest = output/path.name/'common24'
    metrics = {label: wham(path, dest, 24, label, begin, end, bootstraps if label == 'full' else 0)
               for label, begin, end in [('full',100,500), ('first',100,300), ('second',300,500)]}
    h = xvg(dest/'histograms_full.xvg')[:,1:]
    h = h / h.sum(axis=0)[None,:]
    overlap = [float(np.minimum(h[:,i], h[:,i+1]).sum()) for i in range(h.shape[1]-1)]
    result = {'pathway':path.name, 'source_path':str(path), 'metadata':metadata, 'windows':rows,
              'completed_windows':sum(r['complete'] for r in rows),
              'common24_dg_kj_mol':metrics, 'half_difference_kj_mol':abs(metrics['first']-metrics['second']),
              'neighbour_histogram_overlaps':overlap, 'minimum_neighbour_overlap':min(overlap)}
    fullhist = xvg(dest/'histograms_full.xvg')
    metric = ((fullhist[:,0] >= .345) & (fullhist[:,0] <= .505)) | ((fullhist[:,0] >= .895) & (fullhist[:,0] <= 1.505))
    result['unsampled_metric_boundary_bins'] = fullhist[metric & (fullhist[:,1:].sum(axis=1)==0),0].tolist()
    if all(r['complete'] for r in rows):
        full25 = wham(path, output/path.name/'full25', 25, 'full',100,500,0)
        result['full25_dg_kj_mol'] = full25
        result['terminal_window_sensitivity_kj_mol'] = full25 - metrics['full']
    (output/path.name/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'pathway':path.name, 'complete':result['completed_windows'], 'dg':metrics,
                      'min_overlap':min(overlap)}), flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'results/a4p_bohrium_recovery')
    p.add_argument('--output',type=Path,default=ROOT/'results/a4p_bohrium_analysis')
    p.add_argument('--bootstraps',type=int,default=100)
    args=p.parse_args()
    paths=sorted(path.parent for path in args.input.resolve().glob('207131*/umbrella/a4_p_c6_c8_selectivity/*/pathway.json'))
    if len(paths)!=11: raise ValueError(f'expected 11 paths, got {len(paths)}')
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(process, [(path,output,args.bootstraps) for path in paths]))
    contrasts=[]
    for state in sorted({r['metadata']['state'] for r in results}):
        selected=[r for r in results if r['metadata']['state']==state]
        byclass={r['metadata'].get('route_class') or r['metadata']['topology_class']:r for r in selected}
        for alt,ref in [('cross','same'),('switch','reuse'),('switch','same')]:
            if alt not in byclass or ref not in byclass: continue
            delta={label:byclass[alt]['common24_dg_kj_mol'][label]-byclass[ref]['common24_dg_kj_mol'][label]
                   for label in ['full','first','second']}
            contrasts.append({'state':state,'alternative':alt,'reference':ref,'ddg_kj_mol':delta,
                              'contact_reference_odds_ratio':{label:math.exp(-v/RT) for label,v in delta.items()},
                              'half_difference_kj_mol':abs(delta['first']-delta['second'])})
    summary={'protocol':{'temperature_K':313.15,'contact_nm':CONTACT,'reference_nm':REFERENCE,
                        'common_window_centers_nm':[.35,2.65],'production_analysis_ps':[100,500],
                        'bootstraps':args.bootstraps,'metric':'ratio of independently normalized contact/reference odds; not a kinetic branching probability'},
             'pathways':results,'contrasts':contrasts,
             'limitations':['One trajectory per window; bootstrap does not resolve hidden-basin sampling bias.',
                            'C8 opposite reuse is absent from the original 11-path set; no C8 opposite switch/reuse ratio.',
                            'No common kinetic alpha is fitted from these preliminary PMFs.']}
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# A4-p-C6/C8 umbrella 初步分析','',
           '## 下载与故障恢复','',
           '原11任务中9个完成25/25窗口，2个完成24/25窗口，共273个完整生产窗口；归档均已下载，原文件保留。',
           '失败任务20713128（C6 opposite reuse）和20713129（C6 opposite switch）均在2.75 nm窗口触发pull距离超过0.49倍盒长的安全检查：分别为2.945475 > 2.945261 nm和2.941901 > 2.941012 nm。',
           '恢复任务20714398、20714399各使用一张T4，仅重算末窗口。起点取前一窗口完整构象，重新平衡100 ps、生产500 ps；谐和偏置k从500提高至2000 kJ mol⁻¹ nm⁻²。恢复结果已下载并并入对应25窗口WHAM；共同24窗口仍作为主比较。','',
           '## 自由能与区间权重','',
           '固定比较前24窗口（中心0.35–2.65 nm）；每窗口生产段100–500 ps。',
           '接触区间0.35–0.50 nm，参考区间0.90–1.50 nm；313.15 K。',
           '下列比值是各路径接触/参考权重之比，不是未经校准的化学反应概率。','',
           '| 状态 | 比较 | ΔΔG (kJ/mol) | 权重比 | 前/后半段权重比 |','|---|---|---:|---:|---|']
    for c in contrasts:
        ratio=c['contact_reference_odds_ratio']
        lines.append(f"| {c['state']} | {c['alternative']}/{c['reference']} | {c['ddg_kj_mol']['full']:.3f} | {ratio['full']:.3g} | {ratio['first']:.3g} / {ratio['second']:.3g} |")
    lines += ['', '## 质量检查', '', '| 路径 | 完成窗口 | 最小相邻直方图重叠 | 前后半段 ΔG 差 (kJ/mol) |','|---|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['pathway']} | {r['completed_windows']}/25 | {r['minimum_neighbour_overlap']:.3f} | {r['half_difference_kj_mol']:.3f} |")
    tail_sensitivity=max(abs(r.get('terminal_window_sensitivity_kj_mol',0)) for r in results)
    lines += ['', f'九个完整任务中，25窗口与共同24窗口所得区间ΔG最大差值为{tail_sensitivity:.2g} kJ/mol；该末窗口对所选近接触/参考指标影响可忽略，但这不能证明整体采样收敛。',
              '', '## 限制', '', '- 前后半段差异大或窗口重叠不足时，不将比值作为收敛结论。',
              '- 现有11条路径缺少C8 opposite reuse，不能构造它的switch/reuse比。',
              '- 末窗口恢复使用k=2000，其余窗口k=500；恢复后必须用对应TPR重新WHAM，并与共同24窗口比较。',
              '- GROMACS在未采样bin中输出的零自由能已按直方图掩蔽，防止积分产生虚假权重。',
              '- 目前未拟合单一alpha，也未推算产物收率。', '',
              'WHAM方法说明：https://manual.gromacs.org/documentation/2026.1/onlinehelp/gmx-wham.html','']
    (output/'REPORT.md').write_text('\n'.join(lines))
    print(output/'REPORT.md')

if __name__=='__main__': main()
