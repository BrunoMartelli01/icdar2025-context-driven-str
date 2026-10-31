"""Predeclared approximate tests of five context-router failure hypotheses."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch

from scripts.run_yolo_f1_quality_study import read, seal, sha, checked
from scripts.token_context_data import key, arrays, annotations
from recognition.router_cause_tests import row_supervision

REPO = Path(__file__).resolve().parents[1]


def prepare(args):
    if args.mode != 'approximate' or any(p.resolve().parent != REPO/'configs/approximate' for p in (args.config, args.paths)):
        raise ValueError('Explicit isolated approximate configs required')
    cfg, paths = read(args.config), read(args.paths)
    if cfg['reproduction_mode'] != 'approximate' or paths['reproduction_mode'] != 'approximate' or cfg['table_5_claim']:
        raise ValueError('Approximate non-Table5 tests required')
    source = Path(paths['source_root']); pin = cfg['source_pins']
    base = checked(source/'resolved_configuration.json', pin['plan'])
    results = checked(source/'results.json', pin['results']); audit = checked(source/'independent_audit.json', pin['audit'])
    if results['status'] != 'SUCCESS' or audit['status'] != 'PASS' or audit['results_sha256'] != pin['results']:
        raise ValueError('Completed audited source required')
    for name in cfg['unchanged_source_modules']:
        if sha(REPO/name) != base['code_sha256'][name]:
            raise ValueError('Qualified reused implementation changed: '+name)
    if not torch.cuda.is_available() or cfg['execution']['precision'] != 'float32':
        raise RuntimeError('CUDA FP32 required; no substitution')
    samples = copy.deepcopy(base['samples']); cfg = copy.deepcopy(cfg)
    if args.smoke:
        for split, n in cfg['smoke']['images'].items():
            samples[split] = samples[split][:n]
        cfg['seeds'] = cfg['seeds'][:1]; cfg['cv_folds'] = cfg['smoke']['cv_folds']
        cfg['checkpoints'] = cfg['smoke']['checkpoints']; cfg['bootstrap_replicates'] = cfg['smoke']['bootstrap_replicates']
        cfg['blip_probe_images'] = cfg['smoke']['blip_probe_images']
    files = {}
    for split, ss in samples.items():
        for s in ss:
            stem = key(s['sample_id'])
            for stage in ('features', 'light', 'blip'):
                p = source/split/stage/(stem+'.json'); files[str(p)] = sha(p)
            meta = read(source/split/'features'/(stem+'.json'))
            files[str(source/split/'features'/(stem+'.npz'))] = meta['arrays_sha256']
            if meta['ground_truth_used'] or meta['actual_fallback_output_used']:
                raise ValueError('Source feature leakage')
    for relative in ('frozen_choices.json', 'cv_predictions.json', 'resolved_configuration.json', 'results.json', 'independent_audit.json', 'train/branch_metrics.json', 'test/branch_metrics.json'):
        files[str(source/relative)] = sha(source/relative)
    code = {}
    for package in cfg['snapshot_packages']:
        for p in sorted((REPO/package).rglob('*.py')):
            code[str(p.relative_to(REPO))] = sha(p)
    software = {n: importlib.metadata.version(n) for n in ('torch', 'numpy', 'transformers', 'scikit-learn', 'Pillow', 'scipy')}
    plan = {'schema_version': 1, 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'table_5_claim': False, 'result_class': 'SMOKE_TEST' if args.smoke else 'APPROXIMATE_ROUTER_CAUSE_TESTS',
        'config': cfg, 'paths': paths, 'config_sha256': sha(args.config), 'paths_sha256': sha(args.paths),
        'base_config': base['config'], 'base_paths': base['paths'], 'samples': samples,
        'input_sha256': files, 'code_sha256': code, 'software': software, 'gpu': torch.cuda.get_device_name(0),
        'cuda_build': torch.version.cuda, 'base_word_outputs_changed': False, 'historically_explored_test': True,
        'all_tests_predeclared': True, 'new_expert_inference': False,
        'strict_defaults_changed': False, 'test_used_for_epoch_selection': False}
    output = Path(paths['output_root']).resolve()
    if output.is_relative_to(REPO) or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Separate external output required')
    tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    return plan, output/('smoke' if args.smoke else 'full')/tag


def snapshot(plan, root):
    seal(root/'resolved_configuration.json', plan)
    for name, digest in plan['code_sha256'].items():
        p = root/'code_snapshot'/name; p.parent.mkdir(parents=True, exist_ok=True)
        if sha(REPO/name) != digest:
            raise ValueError('Code changed during snapshot')
        if p.exists():
            if sha(p) != digest:
                raise ValueError('Snapshot changed')
        else:
            p.write_bytes((REPO/name).read_bytes())


def inputs(plan, root, split):
    source = Path(plan['paths']['source_root']); features, rows, labels = {}, {}, {}
    for s in plan['samples'][split]:
        sid = s['sample_id']; a, meta = arrays(source, split, sid)
        a['eligible'] = meta['eligible']; features[sid] = a
        rp = source/split/'light'/(key(sid)+'.json')
        if sha(rp) != plan['input_sha256'][str(rp)] or sha(s['validity_path']) != s['validity_sha256']:
            raise ValueError('Raw rows or validity changed')
        rows[sid] = read(rp)['rows']; gt = annotations(s)
        p = root/split/'labels'/(key(sid)+'.json')
        if p.exists():
            value = read(p)
            if value['source_rows_sha256'] != sha(rp):
                raise ValueError('Diagnostic label input changed')
            labels[sid] = {k: np.array(value[k], np.float32) for k in ('primary', 'auxiliary')}
        else:
            label = row_supervision(s, rows[sid], gt, plan['base_config']['metric']['native_ignore_size'], plan['base_config']['dense_training']['auxiliary_iou'])
            branch = read(source/split/'branch_metrics.json')['branches'][sid]['light']
            if label['metrics'] != branch:
                raise ValueError('Exact primary supervision replay failed: '+sid)
            seal(p, {'source_rows_sha256': sha(rp), 'primary': label['primary'].tolist(),
                'auxiliary': label['auxiliary'].tolist(), 'metrics': label['metrics'],
                'ground_truth_use': 'fitting labels or privileged offline diagnostic only'})
            labels[sid] = {k: label[k] for k in ('primary', 'auxiliary')}
    return features, rows, labels


def execute(root):
    from scripts.crop_clip_runtime import _cuda_peak
    from scripts.router_cause_observations import observations, finish, audit_outputs
    from scripts.router_cause_gpu import extract_crops, omission_probes
    from scripts.router_cause_training import train, evaluate_models
    plan = read(root/'resolved_configuration.json'); cfg = plan['config']; stage = 'qualification'
    torch.cuda.init(); torch.cuda.set_device(torch.device(cfg['execution']['device']))
    torch.set_num_threads(cfg['execution']['cpu_threads']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    try:
        for path, digest in plan['input_sha256'].items():
            if sha(path) != digest:
                raise ValueError('Frozen input changed: '+path)
        stage = 'observational_and_budget_tests'; observations(plan, root)
        stage = 'train_crop_interventions'; extract_crops(plan, root, 'train')
        stage = 'matched_training_interventions'; train(plan, root)
        if not (root/'frozen_choices.json').exists():
            raise ValueError('Fitting must freeze before new TEST crop arrays')
        stage = 'test_crop_interventions'; extract_crops(plan, root, 'test')
        stage = 'held_out_models_and_deletions'; evaluate_models(plan, root)
        stage = 'blip_omission_and_language_controls'; omission_probes(plan, root)
        stage = 'report'; finish(plan, root)
        stage = 'audit'; audit_outputs(plan, root)
        seal(root/'completion.json', {'status': 'SUCCESS', 'results_sha256': sha(root/'results.json'), 'audit_sha256': sha(root/'independent_audit.json')})
        print(json.dumps({'status': 'SUCCESS', 'root': str(root)}), flush=True)
    except BaseException as error:
        value = {'status': 'FAILED', 'stage': stage, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'out_of_memory': isinstance(error, torch.cuda.OutOfMemoryError),
            'device': cfg['execution']['device'], 'precision': 'float32', 'cuda_peak': _cuda_peak(cfg['execution']['device'])}
        seal(root/('failure_'+hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:12]+'.json'), value)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--mode', choices=['approximate'])
    parser.add_argument('--config', type=Path); parser.add_argument('--paths', type=Path)
    parser.add_argument('--smoke', action='store_true'); parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--run-root', type=Path); args = parser.parse_args()
    if args.run_root:
        execute(args.run_root.resolve()); return
    if args.config is None or args.paths is None:
        parser.error('Explicit approximate config and paths required')
    plan, root = prepare(args)
    print(json.dumps({'status': 'PREFLIGHT_PASS', 'root': str(root), 'samples': {k: len(v) for k, v in plan['samples'].items()}}), flush=True)
    if args.preflight:
        return
    snapshot(plan, root)
    subprocess.run([sys.executable, '-u', '-m', 'scripts.run_router_cause_tests', '--run-root', str(root)],
        cwd=root/'code_snapshot', env={**os.environ, 'PYTHONPATH': str(root/'code_snapshot')}, check=True)


if __name__ == '__main__':
    main()
