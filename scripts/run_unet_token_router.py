"""Train matched U-Net/YOLO token routers with the qualified improved fallback."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from scripts.run_yolo_f1_quality_study import read, sha, seal

REPO = Path(__file__).resolve().parents[1]


def prepare(args):
    if args.mode != 'approximate' or any(p.resolve().parent != REPO/'configs/approximate' for p in (args.config, args.paths)):
        raise ValueError('Explicit isolated approximate profiles required')
    cfg, paths = read(args.config), read(args.paths)
    if cfg['reproduction_mode'] != args.mode or paths['reproduction_mode'] != args.mode or cfg['table_5_claim']:
        raise ValueError('Approximate non-Table5 experiment required')
    if not torch.cuda.is_available() or cfg['execution']['precision'] != 'float32':
        raise RuntimeError('Qualified FP32 CUDA execution required')
    source = Path(paths['token_root']); cause = Path(paths['cause_root'])
    for role, entries in cfg['source_sha256'].items():
        for name, digest in entries.items():
            if sha(Path(paths[role])/name) != digest:
                raise ValueError('Pinned experiment source changed: '+role+'/'+name)
    for role in ('token_root', 'cause_root', 'improved_root'):
        if read(Path(paths[role])/'independent_audit.json')['status'] != 'PASS':
            raise ValueError('Source audit did not pass: '+role)
    original = read(source/'resolved_configuration.json'); qualified = read(cause/'resolved_configuration.json')['input_sha256']
    samples = json.loads(json.dumps(original['samples']))
    improved = read(Path(paths['improved_root'])/'processed_predictions.json')['splits']
    known = {s['sample_id'] for s in improved['train']}
    if args.smoke:
        common = [s for s in samples['train'] if s['sample_id'] in known][:cfg['smoke']['cached_train']]
        new = [s for s in samples['train'] if s['sample_id'] not in known][:cfg['smoke']['fresh_train']]
        samples = dict(train=common+new, test=samples['test'][:cfg['smoke']['test']])
        cfg = json.loads(json.dumps(cfg)); cfg['cv_folds'] = cfg['smoke']['cv_folds']
        cfg['training']['epochs'] = cfg['smoke']['epochs']; cfg['training']['checkpoints'] = cfg['smoke']['checkpoints']
    inputs = {}
    for split, ss in samples.items():
        for s in ss:
            for field in ('image', 'annotation', 'validity'):
                p = Path(s[field+'_path'])
                if sha(p) != s[field+'_sha256']:
                    raise ValueError('Frozen dataset evidence changed')
                inputs[str(p)] = s[field+'_sha256']
            for suffix in ('.json', '.npz'):
                p = source/split/'features'/(s['sample_id'].replace(':', '__')+suffix)
                digest = qualified[str(p)]
                if sha(p) != digest:
                    raise ValueError('Qualified scene/word feature bytes changed')
                inputs[str(p)] = digest
            p = source/split/'light'/(s['sample_id'].replace(':', '__')+'.json')
            if sha(p) != qualified[str(p)]:
                raise ValueError('Frozen YOLO transcript changed')
            inputs[str(p)] = qualified[str(p)]
    assets = {}
    for field, digest in cfg['assets'].items():
        if sha(paths[field]) != digest:
            raise ValueError('Qualified asset changed: '+field)
        assets[paths[field]] = digest
    for name, digest in original['config']['assets']['clip_files'].items():
        p = Path(paths['clip_snapshot'])/name
        if sha(p) != digest:
            raise ValueError('CLIP asset changed')
        assets[str(p)] = digest
    code = {str(p.relative_to(REPO)): sha(p) for folder in cfg['snapshot_packages'] for p in sorted((REPO/folder).rglob('*.py'))}
    plan = dict(schema_version=1, reproduction_mode='approximate', claim_label='approximate reproduction',
        result_class='SMOKE_TEST' if args.smoke else 'APPROXIMATE_MATCHED_UNET_YOLO_TOKEN_ROUTERS', table_5_claim=False,
        config=cfg, paths=paths, config_sha256=sha(args.config), paths_sha256=sha(args.paths), samples=samples,
        excluded_train_test_pixel_duplicates=original['excluded_train_test_pixel_duplicates'],
        inputs_sha256=inputs, assets_sha256=assets, code_sha256=code,
        source_plan_sha256=sha(source/'resolved_configuration.json'),
        software={n: importlib.metadata.version(n) for n in ['torch','torchvision','transformers','numpy','scipy','scikit-learn','Pillow']},
        gpu=torch.cuda.get_device_name(0), cuda_build=torch.version.cuda,
        historically_explored_test=True, test_used_for_fitting=False, strict_defaults_changed=False,
        existing_global_context='Qualified image-only CLIP patches and BLIP2 queries reused by exact sample/RGB/NPZ identity',
        matched_training='Identical architecture, seed, grouped folds, optimizer, labels and updated fallback for U-Net and YOLO')
    tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    output = Path(args.output_root or paths['output_root']).resolve()
    if output.is_relative_to(REPO):
        raise ValueError('External scientific artifact directory required')
    root = output/('smoke' if args.smoke else 'full')/tag
    return plan, root


def freeze(plan, root):
    seal(root/'resolved_configuration.json', plan)
    for name, digest in plan['code_sha256'].items():
        p = root/'code_snapshot'/name; p.parent.mkdir(parents=True, exist_ok=True)
        if sha(REPO/name) != digest:
            raise ValueError('Source changed during snapshot')
        if p.exists() and sha(p) != digest:
            raise ValueError('Immutable snapshot changed')
        if not p.exists():
            p.write_bytes((REPO/name).read_bytes())


def wait_memory(plan, stage):
    minimum = plan['config']['execution']['minimum_free_gpu_bytes'][stage]
    started = time.monotonic()
    while torch.cuda.mem_get_info()[0] < minimum:
        if time.monotonic()-started > plan['config']['execution']['maximum_gpu_wait_seconds']:
            raise RuntimeError('Insufficient GPU memory for '+stage)
        print('[GPU resource wait] '+stage, flush=True)
        time.sleep(plan['config']['execution']['gpu_poll_seconds'])


def execute(root):
    from scripts.unet_token_router_stages import light, fallback, branches, features, train, evaluate, audit
    plan = read(root/'resolved_configuration.json'); cfg = plan['config']
    torch.cuda.init(); torch.set_num_threads(cfg['execution']['cpu_threads']); torch.manual_seed(cfg['execution']['seed'])
    torch.cuda.manual_seed_all(cfg['execution']['seed']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    stage = 'validate_sources'
    try:
        for name, digest in {**plan['inputs_sha256'], **plan['assets_sha256']}.items():
            if sha(name) != digest:
                raise ValueError('Frozen input changed after launch: '+name)
        for split in ('train', 'test'):
            stage = split+'_unet'; wait_memory(plan, 'unet'); light(plan, root, split)
            stage = split+'_fallback'; wait_memory(plan, 'fallback'); fallback(plan, root, split)
            stage = split+'_counts'; branches(plan, root, split)
            stage = split+'_word_features'; wait_memory(plan, 'clip'); features(plan, root, split)
            if split == 'train':
                stage = 'matched_router_training'; wait_memory(plan, 'router'); train(plan, root)
        stage = 'final_threshold_evaluation'; evaluate(plan, root)
        stage = 'independent_audit'; audit(plan, root)
        seal(root/'completion.json', dict(status='SUCCESS', results_sha256=sha(root/'results.json'), audit_sha256=sha(root/'independent_audit.json')))
    except BaseException as error:
        record = dict(status='FAILED', stage=stage, error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
            out_of_memory=isinstance(error, torch.cuda.OutOfMemoryError), allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())
        seal(root/('failure_'+str(time.time_ns())+'.json'), record)
        raise


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--mode', choices=['approximate']); p.add_argument('--config', type=Path)
    p.add_argument('--paths', type=Path); p.add_argument('--output-root', type=Path); p.add_argument('--smoke', action='store_true')
    p.add_argument('--preflight', action='store_true'); p.add_argument('--run-root', type=Path); args = p.parse_args()
    if args.run_root:
        execute(args.run_root.resolve()); return
    if args.mode is None or args.config is None or args.paths is None:
        p.error('Explicit approximate mode/config/paths required')
    plan, root = prepare(args); print(json.dumps(dict(status='PREFLIGHT_PASS', root=str(root), images={k:len(v) for k,v in plan['samples'].items()})), flush=True)
    if args.preflight:
        return
    freeze(plan, root)
    subprocess.run([sys.executable, '-u', '-m', 'scripts.run_unet_token_router', '--run-root', str(root)],
        cwd=root/'code_snapshot', env={**os.environ, 'PYTHONPATH':str(root/'code_snapshot')}, check=True)


if __name__ == '__main__':
    main()
