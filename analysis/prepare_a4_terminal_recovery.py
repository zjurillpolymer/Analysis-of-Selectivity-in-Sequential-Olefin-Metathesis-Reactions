#!/usr/bin/env python3
"""Prepare two isolated, provenance-tracked repairs of the failed last windows."""
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAIRS = {20713128: 'A4_p_C6_2_opposite_cis_V2_V3_reuse',
         20713129: 'A4_p_C6_2_opposite_cis_V2_V8_switch'}
for job, name in PAIRS.items():
    source = ROOT / f'results/a4p_bohrium_recovery/{job}/umbrella/a4_p_c6_c8_selectivity/{name}'
    target = ROOT / f'bohrium_submit/terminal_recovery/{name}'
    workspace = target / 'input'
    path = workspace / 'pathway'
    path.mkdir(parents=True, exist_ok=True)
    for filename in ('pathway.json', 'system.top', 'index.ndx', 'window_023_2.650nm_production.gro'):
        shutil.copy2(source / filename, path / filename)
    for stage in ('equil', 'production'):
        filename = f'window_024_2.750nm_{stage}.mdp'
        text = (source / 'window_mdps' / filename).read_text()
        text, count = re.subn(r'(?m)^pull-coord1-k\s*=.*$', 'pull-coord1-k = 2000.0', text)
        assert count == 1
        if stage == 'equil':
            text = re.sub(r'(?m)^gen-seed\s*=.*$', f'gen-seed = {job}', text)
        (path / filename).write_text(text)
    provenance = {'source_job': job, 'pathway': name, 'repair_window': 24,
                  'cause': 'Pull distance crossed 0.49 times the box size',
                  'original_k_kj_mol_nm2': 500, 'repair_k_kj_mol_nm2': 2000,
                  'unchanged': ['distance coordinate', 'window center 2.75 nm', 'temperature', 'NPT', 'force field'],
                  'seed': 'window_023_2.650nm_production.gro',
                  'protocol': 'fresh 100 ps equilibration, then 500 ps production; do not mix old failed tail',
                  'analysis': 'WHAM reads actual bias from TPR; compare with common windows 0..23'}
    (path / 'recovery.json').write_text(json.dumps(provenance, indent=2) + '\n')
    shutil.copy2(ROOT / 'bohrium_submit/run_terminal_recovery.sh', workspace)
    config = {'job_name': f'A4-p repair-last {name}', 'command': 'bash run_terminal_recovery.sh',
              'log_file': 'umbrella_job.log', 'backward_files': ['terminal_results.tar.gz', 'umbrella_job.log'],
              'project_id': 4377225, 'machine_type': 'c4_m15_1 * NVIDIA T4',
              'image_address': 'registry.dp.tech/dptech/gromacs:2024.2',
              'input_directory': str(workspace), 'nnode': 1,
              'max_reschedule_times': 0, 'max_run_time': 120}
    (target / 'job.json').write_text(json.dumps(config, indent=2) + '\n')
    print(target / 'job.json')
