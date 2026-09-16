#!/usr/bin/env python3
"""Reanalyse completed C8 reuse and its matched switch; preserve prior reports."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import numpy as np
import deep_a4p_audit as deep
from analyze_a4p_downloads import audit, RT

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/a4p_c8_reuse_comparison'

def contrast(a, b):
    alpha = a['logPcontact_truncated'] - b['logPcontact_truncated']
    lk = a['logK'] - b['logK']
    return {'Kratio': math.exp(lk), 'Pratio_truncated': math.exp(alpha),
            'alpha_truncated': alpha, 'ddg_P_kj': -RT * alpha, 'ddg_K_kj': -RT * lk}

def main():
    deep.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {
        'reuse': ROOT / 'results/a4p_c8_reuse_restart/20714508/umbrella/a4_p_c6_c8_selectivity/A4_p_C8_2_opposite_cis_V3_V6_reuse',
        'switch': ROOT / 'results/a4p_bohrium_recovery/20713134/umbrella/a4_p_c6_c8_selectivity/A4_p_C8_2_opposite_cis_V1_V3_switch',
    }
    rows = []
    for label, src in sources.items():
        windows = audit(src)
        assert len(windows) == 25 and all(w['complete'] for w in windows), (label, windows)
        errors = []
        for log in src.glob('*.log'):
            for line in log.read_text(errors='replace').splitlines():
                if 'fatal error' in line.lower() or 'lincs warning' in line.lower():
                    errors.append({'file': str(log), 'line': line})
        assert not errors, errors
        rows.append({'pathway': src.name, 'source_path': str(src),
                     'metadata': json.loads((src/'pathway.json').read_text())})
    with ThreadPoolExecutor(max_workers=2) as pool:
        pathways = list(pool.map(deep.run_one, rows))
    b, a = pathways  # reuse, switch
    rng = np.random.default_rng(20714508)
    boots = [contrast(a['bootstrap_metrics'][i], b['bootstrap_metrics'][j])
             for i, j in zip(rng.integers(100, size=10000), rng.integers(100, size=10000))]
    comparison = {
        'time': {label: contrast(a['metrics'][label], b['metrics'][label]) for label, _, _ in deep.BLOCKS},
        'conditional_bootstrap_95pct': {k: np.quantile([r[k] for r in boots], [.025,.5,.975]).tolist() for k in boots[0]},
    }
    for key, field in [('contact_sensitivity','interval'), ('reference_sensitivity','interval'), ('normalization_range_sensitivity','upper')]:
        comparison[key] = [{field: x[field], **contrast(x['metrics'], y['metrics'])}
                           for x,y in zip(a[key],b[key])]
    result = {'source_jobs': {'reuse':20714508,'switch':20713134},
              'temperature_K':313.15, 'contact_nm':[.35,.5], 'reference_nm':[.9,1.5],
              'normalization_domain_nm':[.15,3.0], 'analysis_ps':[100,500],
              'independent_trajectories_per_window':1,
              'warning':'Different pilot seeds (reuse replica 1, switch replica 3); bootstrap is conditional on sampled basins, not between-start uncertainty. Single-pair comparison, not class-summed selectivity.',
              'pathways':pathways, 'switch_over_reuse':comparison}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(comparison,indent=2))

if __name__ == '__main__':
    main()
