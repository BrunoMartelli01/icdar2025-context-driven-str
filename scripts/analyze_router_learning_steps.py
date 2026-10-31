"""OOF learning-curve controls with equal or nearly equal optimizer updates."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from scripts.analyze_router_cause_tests import count_array, paired_interval
from scripts.run_yolo_f1_quality_study import read, sha, seal

REPO = Path(__file__).resolve().parents[1]


def analyze(args):
    if args.mode != 'approximate' or args.config.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit isolated approximate analysis config required')
    cfg = read(args.config); root = args.run_root.resolve(); plan = read(root/'resolved_configuration.json')
    if cfg['reproduction_mode'] != 'approximate' or plan['reproduction_mode'] != 'approximate':
        raise ValueError('Approximate analysis required')
    pairs = cfg['smoke_pairs'] if plan['result_class'] == 'SMOKE_TEST' else cfg['constant_update_pairs']
    seed = plan['config']['seeds'][0]
    files = {name: root/'training_trials'/(name+'__seed'+str(seed)+'.json') for pair in pairs for name in (pair['left'][0], pair['right'][0])}
    while any(not p.exists() for p in files.values()):
        if not args.wait:
            raise FileNotFoundError('Required completed TRAIN trials not yet available')
        if list(root.glob('failure_*.json')):
            raise RuntimeError('Source experiment failed; fixed-update analysis not started')
        time.sleep(cfg['poll_seconds'])
    trials = {name: read(p) for name, p in files.items()}; samples = {s['sample_id']: s for s in plan['samples']['train']}; ids = sorted(samples)
    source = Path(plan['paths']['source_root']); branches = read(source/'train/branch_metrics.json')['branches']
    results = []
    for pair in pairs:
        records = []; counts = []
        for name, epoch in (pair['left'], pair['right']):
            trial = trials[name]; point = trial['oof_points'][str(epoch)]; updates = []; populations = []
            for ck in trial['checkpoint_receipts']:
                if '__fold' not in ck['model_path'] or not ck['model_path'].endswith('__epoch'+str(epoch)+'.pt'):
                    continue
                path = root/ck['model_path']
                if sha(path) != ck['model_sha256']:
                    raise ValueError('Matched checkpoint changed')
                saved = torch.load(path, map_location='cpu', weights_only=True)
                updates.append(saved['optimizer_steps']); populations.append(len(saved['preprocessing']['training_ids']))
            if len(updates) != plan['config']['cv_folds']:
                raise ValueError('Fixed-update checkpoint population incomplete')
            records.append({'variant': name, 'epoch': epoch, 'fitting_images_per_fold': populations,
                'optimizer_steps_per_fold': updates, 'mean_optimizer_steps': float(np.mean(updates)), 'OOF_F1': point['combined']['f1']})
            counts.append(count_array(branches, ids, point['accepted_ids']))
        interval = paired_interval(counts[0][None], counts[1][None], [samples[s]['rgb_sha256'] for s in ids], cfg)
        results.append({'left': records[0], 'right': records[1], 'paired_OOF_left_minus_right_pp': interval,
            'exact_update_match': records[0]['optimizer_steps_per_fold'] == records[1]['optimizer_steps_per_fold']})
    output = {'status': 'SUCCESS', 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'result_class': plan['result_class'], 'plan_sha256': sha(root/'resolved_configuration.json'),
        'analysis_config_sha256': sha(args.config), 'analysis_code_sha256': sha(__file__), 'seed': seed,
        'uses_TEST_outcomes': False, 'trials_sha256': {name: sha(path) for name, path in files.items()},
        'comparisons': results, 'limitation': 'Conditional single-seed OOF contrast; nearly matched update counts remain explicit. No TEST model or epoch selection.'}
    seal(args.output/'analysis.json', output)
    print(str(args.output/'analysis.json'))


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--mode', choices=['approximate'], required=True)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--wait', action='store_true'); analyze(p.parse_args())


if __name__ == '__main__':
    main()
