"""Frozen CRAFT inventory and active-region audits for approximate image routing."""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import GroupKFold

from localize.external_text_detectors import DetectorRuntime, crop_region
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer, verify_parseq_runtime_source)
from recognition.shadow_inventory_router import inventory_features, probe_indices
from recognition.f1_quality_study import aggregate, evaluate
from recognition.trainable_image_router import branch_f1, oracle
from scripts import run_trainable_image_router as old
from scripts.run_yolo_f1_quality_study import checked, read, seal, sha
from scripts.crop_clip_runtime import array_sha, _cuda_peak
from scripts.contextual_lm_verifier_runtime import state_digest

REPO = Path(__file__).resolve().parents[1]
CODE = tuple(dict.fromkeys(('scripts/run_shadow_inventory_router.py',
    'recognition/shadow_inventory_router.py', 'localize/external_text_detectors.py',
    'localize/independent_word_slots.py', 'recognition/parseq.py',
    'scripts/contextual_lm_verifier_runtime.py', 'scripts/crop_clip_runtime.py', *old.CODE)))


def image(sample):
    if sha(sample['image_path']) != sample['image_sha256']:
        raise ValueError('Original stored image checksum differs')
    with Image.open(sample['image_path']) as im:
        rgb = np.asarray(im.convert('RGB'))
    if list(rgb.shape[:2][::-1]) != sample['image_wh']:
        raise ValueError('Stored raster orientation or dimensions differ')
    return rgb


def prepare(args):
    if args.mode != 'approximate' or any(p.resolve().parent != REPO/'configs/approximate' for p in (args.config, args.paths)):
        raise ValueError('Explicit isolated approximate configurations required')
    cfg, paths = read(args.config), read(args.paths)
    if cfg['reproduction_mode'] != 'approximate' or paths['reproduction_mode'] != 'approximate' or cfg['table_5_claim']:
        raise ValueError('Approximate non-Table5 experiment required')
    if cfg['rules'] != ['geometry', 'probe', 'recapture', 'agreement'] or cfg['feature_sets'] != ['core', 'shadow']:
        raise ValueError('Predeclared rules/ablation contract differs')
    if cfg['execution']['precision'] != 'float32' or not torch.cuda.is_available():
        raise RuntimeError('Explicit FP32 CUDA execution required; no automatic CPU substitute')
    if type(cfg['inventory']['maximum_probes']) is not int or cfg['inventory']['maximum_probes'] < 1:
        raise ValueError('Positive integer active probe budget required')
    source = Path(paths['source_root']).resolve(); pins = cfg['source_pins']
    p = checked(source/'resolved_configuration.json', pins['plan'])
    r = checked(source/'results.json', pins['results']); a = checked(source/'independent_audit.json', pins['audit'])
    if r['status'] != 'SUCCESS' or a['status'] != 'PASS' or a['results_sha256'] != pins['results']:
        raise ValueError('Complete audited source required')
    for file, digest in p['code_sha256'].items():
        if sha(source/'code_snapshot'/file) != digest:
            raise ValueError('Source snapshot differs')
    inputs = {str(source/file): pins[k] for file, k in
              [('resolved_configuration.json', 'plan'), ('results.json', 'results'), ('independent_audit.json', 'audit')]}
    fit, val = p['fit_ids'], p['validation_ids']; samples = p['samples']
    if args.smoke:
        fit, val = fit[:cfg['smoke_sizes']['fit']], val[:cfg['smoke_sizes']['validation']]
        samples = {'train': [s for s in samples['train'] if s['sample_id'] in fit+val],
                   'test': samples['test'][:cfg['smoke_sizes']['test']]}
    if set(fit)&set(val) or set(fit+val) != {s['sample_id'] for s in samples['train']}:
        raise ValueError('Fitting and calibration partition differs')
    if set(fit+val)&{s['sample_id'] for s in samples['test']}:
        raise ValueError('TRAIN/TEST ID overlap')
    observed = {}; coverage = Path(p['source_coverage_root'])
    for split, ss in samples.items():
        observed[split] = {}
        for sample in ss:
            sid = sample['sample_id']; key = sid.replace(':', '__')+'.json'
            fpath, wpath = source/split/'features'/key, coverage/'640'/split/'features'/key
            f, words = read(fpath), read(wpath)
            if (f['plan_sha256'] != pins['plan'] or f['image_sha256'] != sample['image_sha256'] or
                words['image_sha256'] != sample['image_sha256'] or f['per_image_ground_truth_used'] or
                f['actual_deepsolo_outputs_used'] or words['ground_truth_used']):
                raise ValueError('Label-free image input contract differs')
            if sha(sample['image_path']) != sample['image_sha256']:
                raise ValueError('Input image changed')
            inputs[str(fpath)], inputs[str(wpath)] = sha(fpath), sha(wpath)
            observed[split][sid] = {'sample': sample, 'source': f, 'rows': words['rows']}
    bp = coverage/'results.json'; inputs[str(bp)] = sha(bp)
    if inputs[str(bp)] != p['input_sha256'][str(bp)]:
        raise ValueError('Frozen branch evaluation changed')
    for name, digest in cfg['checkpoint_sha256'].items():
        if sha(paths[name]) != digest:
            raise ValueError('Pinned checkpoint/archive differs: '+name)
        inputs[paths[name]] = digest
    if cfg['checkpoint_sha256']['parseq_checkpoint'] != PARSEQ_CHECKPOINT_SHA256 or cfg['checkpoint_sha256']['parseq_archive'] != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError('Qualified PARSeq identity differs')
    runtime = verify_parseq_runtime_source(Path(paths['parseq_source']), source_type=PARSEQ_ARCHIVE_SOURCE,
        source_archive=Path(paths['parseq_archive']))
    if importlib.metadata.version('easyocr') != cfg['detector']['version']:
        raise ValueError('Pinned EasyOCR version required')
    package = Path(importlib.util.find_spec('easyocr').origin).parent
    package_hashes = {str(q.relative_to(package)): sha(q) for q in sorted(package.rglob('*.py'))}
    dummy, _, _ = inventory_features([], [], [], cfg['inventory'])
    schemas = {'core': p['feature_schemas']['core'], 'shadow': p['feature_schemas']['core']+list(dummy)}
    families = ['rule_'+s for s in cfg['rules']]+[kind+'_'+f for kind in cfg['model_kinds'] for f in cfg['feature_sets']]
    families += ['control_'+s for s in cfg['controls']]
    output = Path(args.output_root or paths['output_root']).resolve()
    if output.is_relative_to(REPO) or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Use separate external output directory')
    # Record only actual source dependencies; reject a stale provenance list.
    for name in CODE:
        if not (REPO/name).is_file():
            raise ValueError('Missing declared runtime source: '+name)
    plan = {'schema_version': 1, 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'table_5_claim': False, 'result_class': 'SMOKE_TEST' if args.smoke else 'APPROXIMATE_SHADOW_INVENTORY_ROUTER',
        'config': cfg, 'paths': paths, 'config_sha256': sha(args.config), 'paths_sha256': sha(args.paths),
        'samples': samples, 'fit_ids': fit, 'validation_ids': val, 'families': families, 'feature_schemas': schemas,
        'source_coverage_root': str(coverage), 'metric': p['metric'], 'input_sha256': inputs,
        'code_sha256': {f: sha(REPO/f) for f in CODE}, 'easyocr_source_sha256': package_hashes,
        'parseq_source_identity': runtime, 'gpu': torch.cuda.get_device_name(0), 'cuda_build': torch.version.cuda,
        'software': {n: importlib.metadata.version(n) for n in ('torch', 'torchvision', 'numpy', 'scipy', 'scikit-learn', 'Pillow', 'easyocr')},
        'base_models_frozen': True, 'new_neural_weight_updates': False, 'historically_explored_test': True,
        'deep_solo_trained_on_totaltext_train': True, 'deterministic_algorithms': True, 'tf32': False}
    tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    return cfg, plan, output/('smoke' if args.smoke else 'full')/tag, observed


def infer_stage(cfg, plan, root, split, observed, stage):
    ph = sha(root/'resolved_configuration.json'); receipt = root/split/(stage+'_receipt.json')
    if receipt.exists():
        if read(receipt)['plan_sha256'] != ph:
            raise ValueError('Receipt identity differs')
        return
    if split == 'test' and not (root/'frozen_choices.json').exists():
        raise ValueError('Freeze fitting and calibration before TEST inference')
    device = cfg['execution']['device']; torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic(); model = None; shape = None
    binding = sha(root/'frozen_choices.json') if split == 'test' else None
    try:
        if stage == 'detector':
            model = DetectorRuntime('craft', cfg['detector'], Path(plan['paths']['craft_checkpoint']), device)
            net = model.model
        else:
            model = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(plan['paths']['parseq_source']),
                runtime_source_archive=Path(plan['paths']['parseq_archive']), runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                checkpoint=Path(plan['paths']['parseq_checkpoint']), device=device))
            net = model.model
        net.requires_grad_(False).eval(); before = state_digest(net); counts = 0
        for index, (sid, v) in enumerate(observed.items(), 1):
            path = root/split/stage/(sid.replace(':', '__')+'.json')
            if path.exists():
                raw = read(path)
                if raw['plan_sha256'] != ph or raw['image_sha256'] != v['sample']['image_sha256'] or raw['frozen_choices_sha256'] != binding:
                    raise ValueError('Raw inference cache contract differs')
            else:
                rgb = image(v['sample']); started = time.monotonic()
                if stage == 'detector':
                    regions = model.infer(rgb); shape = model.last_input_shape
                    payload = {'regions': regions, 'input_shape': shape,
                        'probe_region_ids': [r['source_index'] for i, r in enumerate([r for r in regions if r['valid']])
                                             if i in probe_indices(v['rows'], [r for r in regions if r['valid']], cfg['inventory'])]}
                else:
                    dpath = root/split/'detector'/(sid.replace(':', '__')+'.json'); det = read(dpath); rows = []
                    requests = [(r, view) for r in det['regions'] if r['valid'] for view in ('axis_aligned', 'rectified')]
                    batch = cfg['execution']['parseq_batch_size']
                    for offset in range(0, len(requests), batch):
                        group = requests[offset:offset+batch]; crops = [crop_region(rgb, r, view) for r, view in group]
                        values = model.recognize_candidate_batch(crops); shape = [len(group), 3, 32, 128]
                        if len(values) != len(group):
                            raise ValueError('Incomplete native PARSeq batch')
                        for (region, view), crop, value in zip(group, crops, values):
                            rows.append({'source_index': region['source_index'], 'view': view,
                                'candidate_id': f'{sid}:craft:{region["source_index"]}:{view}', 'bbox': region['bbox'],
                                'crop_shape': list(crop.shape), 'crop_rgb_sha256': array_sha(crop), **value.to_dict()})
                    payload = {'rows': rows, 'detector_sha256': sha(dpath)}
                torch.cuda.synchronize(device)
                raw = {'plan_sha256': ph, 'sample_id': sid, 'image_sha256': v['sample']['image_sha256'],
                    'frozen_choices_sha256': binding, 'seconds': time.monotonic()-started,
                    'ground_truth_used': False, **payload}; seal(path, raw)
            counts += len(raw['regions'] if stage == 'detector' else raw['rows'])
            if index % 25 == 0 or index == len(observed):
                print(f'[{stage} {split}] {index}/{len(observed)} records={counts}', flush=True)
        after = state_digest(net)
        if after != before:
            raise ValueError('Frozen neural model state changed')
        seal(receipt, {'plan_sha256': ph, 'model': 'CRAFT' if stage == 'detector' else 'qualified PARSeq',
            'device': device, 'precision': 'float32', 'batch_size': 1 if stage == 'detector' else cfg['execution']['parseq_batch_size'],
            'last_input_shape': shape, 'model_state_sha256_before': before, 'model_state_sha256_after': after,
            'optimizer_steps': 0, 'parameters': sum(p.numel() for p in net.parameters()),
            'images': len(observed), 'records': counts, 'seconds_including_load_and_state_hashes': time.monotonic()-start,
            'cuda_peak': _cuda_peak(device)})
    except BaseException as error:
        seal(root/split/(stage+'_failure.json'), {'status': 'FAILED', 'plan_sha256': ph, 'stage': stage,
            'error_type': type(error).__name__, 'error': str(error), 'traceback': traceback.format_exc(),
            'device': device, 'precision': 'float32', 'batch_size': 1 if stage == 'detector' else cfg['execution']['parseq_batch_size'],
            'last_input_shape': shape, 'out_of_memory': isinstance(error, torch.cuda.OutOfMemoryError), 'cuda_peak': _cuda_peak(device)})
        raise
    finally:
        net = None
        if model is not None:
            model.release()
        model = None; gc.collect(); torch.cuda.empty_cache()


def feature_stage(cfg, plan, root, split, observed):
    values = {}; ph = sha(root/'resolved_configuration.json')
    for sid, v in observed.items():
        key = sid.replace(':', '__')+'.json'; dp, op = root/split/'detector'/key, root/split/'ocr'/key
        det, ocr = read(dp), read(op)
        new, scores, evidence = inventory_features(v['rows'], det['regions'], ocr['rows'], cfg['inventory'])
        if set(det['probe_region_ids']) != set(evidence['probe_region_ids']):
            raise ValueError('Active probe selection changed after OCR')
        value = {'plan_sha256': ph, 'sample_id': sid, 'image_sha256': v['sample']['image_sha256'],
            'frozen_choices_sha256': sha(root/'frozen_choices.json') if split == 'test' else None,
            'values': {**{n: v['source']['values'][n] for n in plan['feature_schemas']['core']}, **new},
            'rules': scores, 'controls': v['source']['controls'], 'eligible': v['source']['eligible'],
            'evidence': evidence, 'per_image_ground_truth_used': False, 'actual_deepsolo_outputs_used': False,
            'detector_sha256': sha(dp), 'ocr_sha256': sha(op)}
        seal(root/split/'features'/key, value); values[sid] = value
    return values


def predict_ids(model, features, ids):
    if model['recipe']['kind'] in ('rule', 'control'):
        kind = model['recipe']['kind']; field = model['recipe']['field']
        return {s: features[s]['rules' if kind == 'rule' else 'controls'][field] if features[s]['eligible'] else None for s in ids}
    return old.predict_ids(model, features, ids)


def fit_and_freeze(cfg, plan, root, features, branches):
    fit, val = plan['fit_ids'], plan['validation_ids']
    folds = list(GroupKFold(n_splits=cfg['cv_folds']).split(fit, groups=[features[s]['image_sha256'] for s in fit]))
    frozen = {'plan_sha256': sha(root/'resolved_configuration.json'), 'fit_ids': fit, 'validation_ids': val,
        'budgets': {}, 'models_sha256': {}, 'test_labels_used': False, 'test_predictions_present_at_freeze': 0,
        'test_features_present_at_freeze': 0}; cv = {}; models = {}
    if list((root/'test').rglob('*.json')) and not (root/'frozen_choices.json').exists():
        raise ValueError('Cannot select a router after TEST inference')
    for fraction in cfg['bypass_fractions']:
        key = str(fraction); cv[key] = {}; models[key] = {}; trials = []; cuts = {}
        for index, family in enumerate(plan['families']):
            kind, field = family.split('_', 1)
            if kind in ('rule', 'control'):
                model = {'recipe': {'kind': kind, 'field': field}, 'family': family, 'training_ids': []}
            else:
                candidates = []
                for recipe in old.recipes(kind, cfg):
                    oof = {}; trace = []
                    for a, b in folds:
                        training, held = [fit[i] for i in a], [fit[i] for i in b]
                        m = old.train_one(cfg, plan, features, branches, training, recipe, family, fraction)
                        oof.update(predict_ids(m, features, held)); trace.append({'training_ids': training, 'held_out_ids': held})
                    accepted = old.top(oof, fit, fraction)
                    candidates.append({'recipe': recipe, 'oof_scores': oof, 'accepted_ids': accepted,
                        'oof_f1': float(branch_f1(branches, fit, accepted)), 'fold_trace': trace})
                winner = max(range(len(candidates)), key=lambda i: (branch_f1(branches, fit, candidates[i]['accepted_ids']), -i))
                cv[key][family] = {'selected_trial': winner, 'trials': candidates}
                model = old.train_one(cfg, plan, features, branches, fit, candidates[winner]['recipe'], family, fraction)
            path = root/'models'/(key+'__'+family+'.json'); seal(path, model)
            frozen['models_sha256'][str(path.relative_to(root))] = sha(path); models[key][family] = model
            scores = predict_ids(model, features, val); accepted = old.top(scores, val, fraction)
            cuts[family] = min((scores[s] for s in accepted), default=None)
            trials.append((branch_f1(branches, val, accepted), -index, family))
        frozen['budgets'][key] = {'selected_family': max(trials)[2], 'scalar_cuts': cuts,
            'validation_f1': {name: float(score) for score, _, name in trials}}
        print(f'[fit {fraction}] VAL selected {max(trials)[2]}', flush=True)
    seal(root/'cv_results.json', cv); frozen['cv_results_sha256'] = sha(root/'cv_results.json')
    seal(root/'frozen_choices.json', frozen)
    return frozen, models


def evaluate_split(cfg, plan, features, branches, ids, models, frozen):
    out = {'images': len(ids), 'light_only': aggregate(branches[s]['light'] for s in ids),
           'deepsolo_only': aggregate(branches[s]['deepsolo'] for s in ids), 'budgets': {}}; predictions = {}
    eligible = [s for s in ids if features[s]['eligible']]; oracles = {}
    def point(accepted):
        n = len(accepted)
        if n not in oracles:
            oracles[n] = oracle({s: branches[s] for s in ids}, ids, eligible, n)
        return {**evaluate(branches, ids, accepted, cfg['quality_f1_threshold']), 'oracle_reference_f1': oracles[n]['f1']}
    for fraction in cfg['bypass_fractions']:
        key = str(fraction); families = {}; scalar = {}; predictions[key] = {}
        for family, model in models[key].items():
            scores = predict_ids(model, features, ids); predictions[key][family] = scores
            families[family] = point(old.top(scores, ids, fraction)); cut = frozen['budgets'][key]['scalar_cuts'][family]
            scalar[family] = {'threshold': cut, **point([s for s in ids if cut is not None and scores[s] is not None and scores[s] >= cut])}
        selected = frozen['budgets'][key]['selected_family']
        out['budgets'][key] = {'families': families, 'scalar_thresholds': scalar, 'selected_family': selected,
            'selected': families[selected], 'primary_family': cfg['primary_family'], 'primary': families[cfg['primary_family']],
            'oracle': oracles[round(len(ids)*fraction)]}
    return out, predictions


def run(args):
    cfg, plan, root, observed = prepare(args)
    print(json.dumps({'status': 'PREFLIGHT_PASS', 'root': str(root), 'feature_counts': {k: len(v) for k, v in plan['feature_schemas'].items()}}), flush=True)
    if args.preflight:
        return
    seal(root/'resolved_configuration.json', plan)
    for file in plan['code_sha256']:
        target = root/'code_snapshot'/file; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != plan['code_sha256'][file]:
            raise ValueError('Immutable runtime snapshot changed')
        if not target.exists():
            target.write_bytes((REPO/file).read_bytes())
    seed = cfg['model_settings']['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(cfg['execution']['cpu_threads']); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for split in ('train', 'test'):
        for stage in ('detector', 'ocr'):
            infer_stage(cfg, plan, root, split, observed[split], stage)
        values = feature_stage(cfg, plan, root, split, observed[split])
        if split == 'train':
            train = values
            # Only fitting/calibration branch outcomes enter model selection.
            b = read(Path(plan['source_coverage_root'])/'results.json')['resolutions']['640']['branch_metrics']['train']
            frozen, models = fit_and_freeze(cfg, plan, root, train, b)
        else:
            test = values
    branches = read(Path(plan['source_coverage_root'])/'results.json')['resolutions']['640']['branch_metrics']
    result = {'status': 'SUCCESS', 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'result_class': plan['result_class'], 'table_5_claim': False, 'plan_sha256': sha(root/'resolved_configuration.json'),
        'frozen_choices_sha256': sha(root/'frozen_choices.json'), 'base_models_frozen': True,
        'original_word_outputs_changed': False, 'router_training': True, 'splits': {}}
    for split, data, b, ids in [('fit', train, branches['train'], plan['fit_ids']),
        ('validation', train, branches['train'], plan['validation_ids']), ('train', train, branches['train'], sorted(train)),
        ('test', test, branches['test'], sorted(test))]:
        result['splits'][split], predictions = evaluate_split(cfg, plan, data, b, ids, models, frozen)
        seal(root/'predictions'/(split+'.json'), predictions)
    seal(root/'results.json', result)
    print(json.dumps({'status': 'SUCCESS', 'root': str(root), 'results_sha256': sha(root/'results.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--mode', required=True, choices=['approximate'])
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output-root', type=Path); parser.add_argument('--preflight', action='store_true'); parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
