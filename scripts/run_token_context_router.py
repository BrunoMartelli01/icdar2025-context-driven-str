"""Full-TRAIN approximate token-context, transcript and dense-support routers."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
from PIL import Image
import torch
import yaml

from scripts.run_yolo_f1_quality_study import read, sha, seal, checked
from scripts.crop_clip_runtime import array_sha
from scripts.token_context_data import (key, rgb, contract, shard, stage_complete, cached_stage,
    detect_and_read, fallback, evaluate_branches, write_arrays)
from recognition.token_context_router import base_features, word_geometry

REPO = Path(__file__).resolve().parents[1]


def prepare(args):
    if args.mode != 'approximate' or any(p.resolve().parent != REPO/'configs/approximate' for p in (args.config, args.paths)):
        raise ValueError('Explicit isolated approximate configuration files required')
    cfg, paths = read(args.config), read(args.paths)
    if cfg['reproduction_mode'] != 'approximate' or paths['reproduction_mode'] != 'approximate' or cfg['table_5_claim']:
        raise ValueError('Approximate non-Table5 experiment required')
    if cfg['execution']['precision'] != 'float32' or not torch.cuda.is_available():
        raise RuntimeError('FP32 CUDA required; no automatic precision/device substitution')
    if cfg['primary_arm'] != 'token_context' or cfg['arms'] != ['scalar', 'pooled_context', 'token_context', 'likelihood', 'dense', 'combined']:
        raise ValueError('Predeclared three-proposal ablation contract differs')
    if not all(0 < f < 1 for f in cfg['bypass_fractions']) or cfg['cv_folds'] < 2:
        raise ValueError('Explicit valid CV/bypass configuration required')
    source = Path(paths['source_root']); source_plan = checked(source/'resolved_configuration.json', cfg['source_pins']['plan'])
    source_result = checked(source/'results.json', cfg['source_pins']['results']); audit = checked(source/'independent_audit.json', cfg['source_pins']['audit'])
    if audit['status'] != 'PASS' or source_result['status'] != 'SUCCESS' or audit['results_sha256'] != cfg['source_pins']['results']:
        raise ValueError('Complete audited frozen source required')
    coverage = Path(paths['coverage_root']); covp = read(coverage/'resolved_configuration.json')
    if source_plan['source_coverage_root'] != str(coverage):
        raise ValueError('Wrong fixed branch source')
    ds = Path(paths['deepsolo_source_run']); dsp = checked(ds/'resolved_configuration.json', cfg['deepsolo_pins']['plan'])
    dsr = checked(ds/'results.json', cfg['deepsolo_pins']['results'])
    if dsr['status'] != 'SUCCESS':
        raise ValueError('Complete qualified fallback source required')
    inputs = {}; asset_files = {}
    for field, digest in cfg['assets']['files'].items():
        path = Path(paths[field])
        if sha(path) != digest:
            raise ValueError('Pinned asset bytes changed: '+field)
        asset_files[str(path)] = digest
    for prefix in ('clip', 'blip'):
        snapshot = Path(paths[prefix+'_snapshot'])
        for file, digest in cfg['assets'][prefix+'_files'].items():
            if sha(snapshot/file) != digest:
                raise ValueError('Pinned visual model asset changed: '+str(snapshot/file))
            asset_files[str(snapshot/file)] = digest
    from recognition.parseq import verify_parseq_runtime_source, PARSEQ_ARCHIVE_SOURCE
    identity = verify_parseq_runtime_source(Path(paths['parseq_source']), source_type=PARSEQ_ARCHIVE_SOURCE,
        source_archive=Path(paths['parseq_archive']))
    samples = {}; duplicate_groups = {}; manifests = {}
    for split in ('train', 'test'):
        manifest = Path(paths[split+'_manifest']); m = yaml.safe_load(manifest.read_text()); manifests[split] = sha(manifest)
        ss = []; root = Path(paths['data_root'])
        for record in m['samples']:
            image = root/record['image']; validity = root/record['validity_annotation']
            if sha(image) != record['image_sha256'] or sha(validity) != record['validity_annotation_sha256']:
                raise ValueError('Full dataset image/validity bytes changed: '+record['id'])
            with Image.open(image) as im:
                pixels = np.asarray(im.convert('RGB')); orientation = int(im.getexif().get(274, 1)); wh = list(im.size)
            annotation = root/paths['annotation_subdir_template'].format(split=split)/(image.stem+'.txt')
            sample = {'sample_id': record['id'], 'image_path': str(image), 'image_sha256': sha(image),
                'rgb_sha256': array_sha(pixels), 'image_wh': wh, 'orientation': orientation,
                'validity_path': str(validity), 'validity_sha256': sha(validity),
                'annotation_path': str(annotation), 'annotation_sha256': sha(annotation)}
            lp = coverage/'640'/split/'features'/(key(record['id'])+'.json'); dp = ds/split/(key(record['id'])+'.json')
            if lp.exists():
                old = read(lp)
                if old['image_sha256'] != sample['image_sha256'] or old['ground_truth_used']:
                    raise ValueError('Historical source has different pixels')
                sample.update(cached_light=str(lp), cached_light_sha256=sha(lp))
            if dp.exists():
                old = read(dp)
                if old['contract']['image_sha256'] != sample['image_sha256']:
                    raise ValueError('Historical fallback pixels differ')
                sample.update(cached_deepsolo=str(dp), cached_deepsolo_sha256=sha(dp))
            ss.append(sample)
        samples[split] = ss
    test_hash = {s['rgb_sha256']: s['sample_id'] for s in samples['test']}
    excluded = [{**s, 'duplicate_test_id': test_hash[s['rgb_sha256']]} for s in samples['train'] if s['rgb_sha256'] in test_hash]
    samples['train'] = [s for s in samples['train'] if s['rgb_sha256'] not in test_hash]
    legacy = source_plan['fit_ids']; val = source_plan['validation_ids']
    if args.smoke:
        limits = cfg['smoke']; chosen = set(legacy[:limits['legacy_fit']]+val[:limits['legacy_validation']])
        fresh = [s for s in samples['train'] if not s.get('cached_light')][:limits['fresh_train']]
        chosen.update(s['sample_id'] for s in fresh)
        samples = {'train': [s for s in samples['train'] if s['sample_id'] in chosen], 'test': samples['test'][:limits['test']]}
        legacy = [s for s in legacy if s in chosen]
        cfg = json.loads(json.dumps(cfg)); cfg['training']['epochs'] = limits['epochs']; cfg['training']['checkpoints'] = limits['checkpoints']
        cfg['cv_folds'] = limits['cv_folds']; cfg['dense_training']['epochs'] = limits['dense_epochs']
    train_ids = {s['sample_id'] for s in samples['train']}
    if not set(legacy) <= train_ids or train_ids & {s['sample_id'] for s in samples['test']}:
        raise ValueError('Invalid TRAIN/TEST population or legacy fitting subset')
    packages = {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'transformers', 'scipy', 'scikit-learn', 'Pillow', 'PyYAML')}
    if packages['torch'] != covp['software']['torch'] or packages['transformers'] != covp['software']['transformers']:
        raise ValueError('Frozen source/main model software versions differ')
    code = {}
    for directory in cfg['snapshot_packages']:
        for file in sorted((REPO/directory).rglob('*.py')):
            code[str(file.relative_to(REPO))] = sha(file)
    plan = {'schema_version': 1, 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'table_5_claim': False, 'result_class': 'SMOKE_TEST' if args.smoke else 'APPROXIMATE_FULL_TRAIN_THREE_CONTEXT_ROUTERS',
        'config': cfg, 'paths': paths, 'config_sha256': sha(args.config), 'paths_sha256': sha(args.paths),
        'samples': samples, 'excluded_train_test_pixel_duplicates': excluded, 'legacy_fit_ids': legacy,
        'manifest_sha256': manifests, 'code_sha256': code, 'asset_sha256': asset_files, 'software': packages,
        'parseq_source_identity': identity, 'gpu': torch.cuda.get_device_name(0), 'cuda_build': torch.version.cuda,
        'deepsolo_runtime': dsp['runtime'], 'deepsolo_config_sha256': dsp['candidate_resolution']['config_sha256'],
        'deepsolo_runtime_sha256': sha(ds/'runtime.json'), 'source_files_sha256': {str(source/'resolved_configuration.json'): cfg['source_pins']['plan'],
            str(source/'results.json'): cfg['source_pins']['results'], str(source/'independent_audit.json'): cfg['source_pins']['audit'],
            str(coverage/'results.json'): sha(coverage/'results.json'), str(ds/'resolved_configuration.json'): cfg['deepsolo_pins']['plan'],
            str(ds/'results.json'): cfg['deepsolo_pins']['results']},
        'test_used_for_fitting': False, 'historically_explored_test': True, 'expert_saw_train_images': True,
        'base_word_outputs_changed': False, 'precision': 'float32', 'deterministic_algorithms': True, 'tf32': False}
    output = Path(args.output_root or paths['output_root']).resolve()
    if output.is_relative_to(REPO) or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Separate external output root required')
    tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    root = output/('smoke' if args.smoke else 'full')/tag
    return plan, root


def freeze_plan(plan, root):
    seal(root/'resolved_configuration.json', plan)
    for name, digest in plan['code_sha256'].items():
        target = root/'code_snapshot'/name; target.parent.mkdir(parents=True, exist_ok=True)
        if sha(REPO/name) != digest:
            raise ValueError('Source changed during snapshot')
        if target.exists() and sha(target) != digest:
            raise ValueError('Immutable code snapshot changed')
        if not target.exists():
            target.write_bytes((REPO/name).read_bytes())


def extract(plan, root, split, kind):
    from recognition.token_context_runtime import FrozenDenseCLIP, FrozenTranscriptBLIP
    from scripts.crop_clip_runtime import _cuda_peak
    ss = plan['samples'][split]; stage = kind
    if cached_stage(root, split, stage, ss):
        return
    if split == 'test' and not (root/'frozen_choices.json').exists():
        raise ValueError('Freeze all choices before fresh TEST context inference')
    cfg = plan['config']; device = cfg['execution']['device']; start = time.monotonic()
    model = (FrozenDenseCLIP(plan['paths']['clip_snapshot'], cfg['clip'], device) if kind == 'clip' else
             FrozenTranscriptBLIP(plan['paths']['blip_snapshot'], cfg['blip'], device))
    try:
        for i, sample in enumerate(ss, 1):
            path, old = shard(root, split, stage, sample); sid = sample['sample_id']
            binding = sha(root/'frozen_choices.json') if split == 'test' else None
            if old is not None:
                if old['frozen_choices_sha256'] != binding or sha(root/split/stage/(key(sid)+'.npz')) != old['arrays_sha256']:
                    raise ValueError('Raw context cache changed')
                check = model.qualification is None if kind == 'clip' else model.provenance['native_forward_check'] is None
                if check:
                    rows = read(root/split/'light'/(key(sid)+'.json'))['rows']
                    if kind == 'clip' or any(r['raw_text'].strip() for r in rows):
                        if kind == 'clip':
                            tensors, _ = model.infer(rgb(sample), rows)
                        else:
                            query, values, _ = model.infer(rgb(sample), rows)
                            tensors = {'queries': query, 'likelihood': np.array(list(values.values()), np.float32)}
                        with np.load(root/split/stage/(key(sid)+'.npz'), allow_pickle=False) as saved:
                            if set(saved.files) != set(tensors) or any(not np.array_equal(saved[k], v) for k, v in tensors.items()):
                                raise ValueError('Resumed native-forward qualification changed cached tensors')
                continue
            rows_path = root/split/'light'/(key(sid)+'.json'); rows = read(rows_path)['rows']; image = rgb(sample)
            if kind == 'clip':
                tensors, raw = model.infer(image, rows)
            else:
                query, features, raw = model.infer(image, rows)
                tensors = {'queries': query, 'likelihood': np.array(list(features.values()), np.float32)}
                raw['feature_names'] = list(features)
            output = root/split/stage/(key(sid)+'.npz'); write_arrays(output, tensors)
            seal(path, {'contract': contract(root, sample), 'rows_sha256': sha(rows_path), 'arrays_sha256': sha(output),
                'frozen_choices_sha256': binding, 'raw': raw, 'ground_truth_used': False})
            if i % cfg['execution']['log_every'] == 0 or i == len(ss):
                print(f'[{kind} {split}] {i}/{len(ss)}', flush=True)
        metadata = model.finish()
        stage_complete(root, split, stage, ss, {'seconds': time.monotonic()-start, 'model': metadata,
            'cuda_peak': _cuda_peak(device), 'new_neural_weight_updates': False})
    except BaseException as error:
        record = {'status': 'FAILED', 'extractor': kind, 'sample_id': sample['sample_id'] if 'sample' in locals() else None,
            'input_shapes': getattr(model, 'last_input_shapes', None), 'precision': 'float32',
            'vision_batch_size': cfg[kind].get('image_batch_size', 1),
            'language_batch_size': cfg[kind].get('language_batch_size'), 'error': str(error),
            'out_of_memory': isinstance(error, torch.cuda.OutOfMemoryError), 'cuda_peak': _cuda_peak(device)}
        seal(root/split/(kind+'_failure_'+hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]+'.json'), record)
        raise
    finally:
        model.close()


def combine_features(plan, root, split):
    for sample in plan['samples'][split]:
        sid = sample['sample_id']; data = {}; raw_hashes = {}
        for stage in ('clip', 'blip'):
            path = root/split/stage/(key(sid)+'.npz'); meta = read(path.with_suffix('.json'))
            if sha(path) != meta['arrays_sha256']:
                raise ValueError('Frozen extractor arrays changed')
            with np.load(path, allow_pickle=False) as z:
                data.update({k: z[k] for k in z.files})
            raw_hashes[stage] = sha(path.with_suffix('.json'))
        rows_path = root/split/'light'/(key(sid)+'.json'); rows = read(rows_path)['rows']
        data['scalars'] = base_features(rows, sample['image_wh']); data['word_geometry'] = word_geometry(rows, sample['image_wh'])
        if data['scalars'].shape != (plan['config']['architecture']['scalar_dim'],) or data['likelihood'].shape != (plan['config']['architecture']['likelihood_dim'],):
            raise ValueError('Declared scalar feature schema differs')
        output = root/split/'features'/(key(sid)+'.npz'); write_arrays(output, data)
        seal(output.with_suffix('.json'), {'contract': contract(root, sample), 'arrays_sha256': sha(output), 'raw_sha256': raw_hashes,
            'rows_sha256': sha(rows_path), 'eligible': bool(rows) and all(r['raw_text'].strip() for r in rows),
            'image_rgb_sha256': sample['rgb_sha256'], 'ground_truth_used': False, 'actual_fallback_output_used': False,
            'frozen_choices_sha256': sha(root/'frozen_choices.json') if split == 'test' else None,
            'array_shapes': {k: list(v.shape) for k, v in data.items()}})


def execute(root, stage=None, split=None):
    plan = read(root/'resolved_configuration.json'); cfg = plan['config']
    if plan['reproduction_mode'] != 'approximate' or plan['table_5_claim']:
        raise ValueError('Approximate isolated plan required')
    torch.cuda.init(); torch.cuda.set_device(torch.device(cfg['execution']['device']))
    seed = cfg['execution']['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg['execution']['cpu_threads']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    if cfg['execution']['attention_backend'] != 'math':
        raise ValueError('Explicit deterministic math attention required')
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if stage == 'light':
        detect_and_read(plan, root, split); return
    from scripts.token_context_training import train_and_freeze, evaluate
    current = 'asset_validation'
    try:
        for path, digest in {**plan['asset_sha256'], **plan['source_files_sha256']}.items():
            if sha(path) != digest:
                raise ValueError('Frozen external input changed: '+path)
        for subset in ('train', 'test'):
            current = subset+'_light'
            command = [plan['paths']['yolo_python'], '-u', '-m', 'scripts.run_token_context_router', '--run-root', str(root), '--stage', 'light', '--split', subset]
            subprocess.run(command, cwd=root/'code_snapshot', env={**os.environ, 'PYTHONPATH': str(root/'code_snapshot')}, check=True)
            current = subset+'_deepsolo'; fallback(plan, root, subset)
            current = subset+'_clip'; extract(plan, root, subset, 'clip')
            current = subset+'_blip'; extract(plan, root, subset, 'blip')
            combine_features(plan, root, subset)
            if subset == 'train':
                current = 'train_evaluation'; branches = evaluate_branches(plan, root, 'train')
                current = 'router_training'; train_and_freeze(plan, root, branches)
            else:
                current = 'test_evaluation'; evaluate_branches(plan, root, 'test')
        current = 'final_evaluation'; evaluate(plan, root)
        current = 'independent_audit'
        from scripts.audit_token_context_router import audit
        audit(root)
        seal(root/'completion.json', {'status': 'SUCCESS', 'results_sha256': sha(root/'results.json')})
    except BaseException as error:
        from scripts.crop_clip_runtime import _cuda_peak
        record = {'status': 'FAILED', 'stage': current, 'error_type': type(error).__name__, 'error': str(error),
            'traceback': traceback.format_exc(), 'out_of_memory': isinstance(error, torch.cuda.OutOfMemoryError),
            'device': cfg['execution']['device'], 'precision': cfg['execution']['precision'], 'cuda_peak': _cuda_peak(cfg['execution']['device'])}
        file = 'failure_'+hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]+'.json'; seal(root/file, record)
        raise


def main():
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--mode', choices=['approximate']); parser.add_argument('--config', type=Path)
    parser.add_argument('--paths', type=Path); parser.add_argument('--output-root', type=Path); parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--preflight', action='store_true'); parser.add_argument('--run-root', type=Path)
    parser.add_argument('--stage', choices=['light']); parser.add_argument('--split', choices=['train', 'test'])
    args = parser.parse_args()
    if args.run_root:
        execute(args.run_root.resolve(), args.stage, args.split); return
    if args.config is None or args.paths is None:
        parser.error('Explicit mode/config/paths required')
    plan, root = prepare(args)
    print(json.dumps({'status': 'PREFLIGHT_PASS', 'root': str(root), 'samples': {k: len(v) for k, v in plan['samples'].items()},
        'excluded_pixel_duplicates': len(plan['excluded_train_test_pixel_duplicates'])}), flush=True)
    if args.preflight:
        return
    freeze_plan(plan, root)
    subprocess.run([sys.executable, '-u', '-m', 'scripts.run_token_context_router', '--run-root', str(root)],
        cwd=root/'code_snapshot', env={**os.environ, 'PYTHONPATH': str(root/'code_snapshot')}, check=True)


if __name__ == '__main__':
    main()
