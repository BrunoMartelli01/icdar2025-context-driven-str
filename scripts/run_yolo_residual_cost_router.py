"""Offline approximate residual-region/error-cost router with frozen DeepSolo."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

from recognition.f1_quality_study import aggregate, evaluate, good, sorted_ids
from recognition.residual_cost_router import (
    NEW_FAMILIES, predicted_probabilities, residual_regions, routing_scores, word_evidence,
)
from scripts.run_yolo_f1_quality_study import checked, checked_feature, read, seal, sha

REPO = Path(__file__).resolve().parents[1]
CODE = ('scripts/run_yolo_residual_cost_router.py', 'recognition/residual_cost_router.py',
        'recognition/text_coverage_router.py', 'recognition/f1_quality_study.py',
        'scripts/run_yolo_f1_quality_study.py')


def prepare(args):
    if args.mode != 'approximate':
        raise ValueError('This extension supports approximate reproduction only')
    if any(p.resolve().parent != REPO/'configs/approximate' for p in (args.config, args.paths)):
        raise ValueError('Explicit isolated approximate configurations required')
    cfg, paths = read(args.config), read(args.paths)
    if any(v['reproduction_mode'] != 'approximate' for v in (cfg, paths)) or cfg['table_5_claim']:
        raise ValueError('Approximate claim metadata required')
    if tuple(cfg['new_families']) != NEW_FAMILIES or cfg['primary_family'] not in NEW_FAMILIES:
        raise ValueError('Declared family definitions differ')
    if not cfg['bypass_fractions'] or any(not 0 < f <= 1 for f in cfg['bypass_fractions']):
        raise ValueError('Invalid explicit bypass budgets')
    source = Path(paths['source_study_root']).resolve(); pins = cfg['source_study']
    old = checked(source/'resolved_configuration.json', pins['plan'])
    result = checked(source/'results.json', pins['results'])
    audit = checked(source/'independent_audit.json', pins['audit'])
    if result['status'] != 'SUCCESS' or audit['status'] != 'PASS' or audit['results_sha256'] != pins['results']:
        raise ValueError('A complete audited source is required')
    for file, digest in old['code_sha256'].items():
        if sha(source/'code_snapshot'/file) != digest:
            raise ValueError('Source snapshot changed: '+file)
    coverage = Path(old['paths']['sources']['coverage']); consistency = Path(old['paths']['sources']['consistency'])
    cpins = old['config']['sources']['coverage']
    cp = checked(coverage/'resolved_configuration.json', cpins['plan'])
    cr = checked(coverage/'results.json', cpins['results'])
    ca = checked(coverage/cpins['audit_file'], cpins['audit'])
    if ca['status'] != 'PASS' or cr['status'] != 'SUCCESS':
        raise ValueError('Frozen coverage source is incomplete')
    selected, calibration = old['selection_ids'], old['threshold_calibration_ids']
    samples = old['samples']
    if args.smoke:
        n = cfg['smoke_size_per_partition']; selected = selected[:n]; calibration = calibration[:n]
        samples = {'train': [s for s in samples['train'] if s['sample_id'] in selected+calibration],
                   'test': samples['test'][:n]}
    if set(selected)&set(calibration) or set(selected+calibration) != {s['sample_id'] for s in samples['train']}:
        raise ValueError('TRAIN selection/calibration partition differs')
    if {s['sample_id'] for s in samples['train']}&{s['sample_id'] for s in samples['test']}:
        raise ValueError('TRAIN/TEST overlap')
    inputs = {}; records = {}; branches = {}
    coverage_samples = {s['sample_id']: s for group in cp['samples'].values() for s in group}
    for split, group in samples.items():
        records[split] = {}; branches[split] = {}
        for sample in group:
            sid = sample['sample_id']; key = sid.replace(':', '__')+'.json'
            if sample != coverage_samples[sid]:
                raise ValueError('Frozen sample, image frame or map changed')
            fp = source/split/'features'/key; cf = coverage/'640'/split/'features'/key
            sf = consistency/split/'features'/key; mp = Path(sample['map_path'])
            oldf = checked_feature(fp, pins['plan'])
            covf = checked_feature(cf, cpins['plan'])
            conf = checked_feature(sf, old['config']['sources']['consistency']['plan'])
            expected = cp['cached_input_sha256'][str(mp)]
            if sha(mp) != expected or covf['map_sha256'] != expected:
                raise ValueError('Predicted map changed: '+str(mp))
            if any(v['ground_truth_used'] is not False or v['image_sha256'] != sample['image_sha256']
                   or v['sample_id'] != sid for v in (oldf, covf, conf)):
                raise ValueError('Invalid inference-source provenance')
            for p in (fp, cf, sf, mp):
                inputs[str(p)] = sha(p)
            records[split][sid] = {'old': oldf, 'coverage': covf, 'consistency': conf}
            branches[split][sid] = cr['resolutions']['640']['branch_metrics'][split][sid]
    output = (args.output_root or Path(paths['output_root'])).resolve()
    if output.is_relative_to(REPO) or any(output.is_relative_to(p) or p.is_relative_to(output)
                                        for p in (source, coverage, consistency)):
        raise ValueError('Outputs must be external and separate from all frozen sources')
    plan = {'schema_version': 1, 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'table_5_claim': False, 'result_class': 'SMOKE_TEST' if args.smoke else 'APPROXIMATE_EXPLORATORY_COMPARISON',
        'config': cfg, 'paths': paths, 'config_sha256': sha(args.config), 'paths_sha256': sha(args.paths),
        'samples': samples, 'selection_ids': selected, 'threshold_calibration_ids': calibration,
        'families': cfg['control_families']+cfg['new_families'], 'clip_reference': old['clip_reference'],
        'metric': old['metric'], 'source_coverage_root': str(coverage),
        'input_sha256': inputs, 'code_sha256': {f: sha(REPO/f) for f in CODE},
        'source_history': old['test_history'], 'new_model_inference': False, 'network_training': False,
        'optimizer_steps': 0, 'software': {p: importlib.metadata.version(p) for p in ('numpy', 'opencv-python', 'Pillow', 'scipy')},
        'python': sys.version}
    tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    return cfg, plan, output/('smoke' if args.smoke else 'full')/tag, records, branches


def feature_split(cfg, plan, root, split, records):
    features = {}; ph = sha(root/'resolved_configuration.json')
    for sample in plan['samples'][split]:
        sid = sample['sample_id']; path = root/split/'features'/(sid.replace(':', '__')+'.json')
        if path.exists():
            features[sid] = checked_feature(path, ph); continue
        record = records[sid]; rows = record['coverage']['rows']
        maps = predicted_probabilities(sample['map_path'])
        regions = residual_regions(maps, sample['image_wh'], [r['bbox'] for r in rows], cfg['region_settings'])
        for name in ('text', 'kernel'):
            if abs(regions['heads'][name]['coverage']-record['coverage']['coverage'][name+'_coverage']) > 1e-12:
                raise ValueError('Coverage geometry differs from the frozen control')
        evidence = word_evidence(rows, record['consistency'], plan['clip_reference'])
        controls = {name: record['old']['scores'][name] for name in cfg['control_families']}
        scores = {**controls, **routing_scores(regions, evidence, controls)}
        value = {'sample_id': sid, 'image_sha256': sample['image_sha256'],
            'map_sha256': plan['input_sha256'][sample['map_path']], 'regions': regions,
            'word_evidence': evidence, 'scores': scores, 'ground_truth_used': False,
            'network_training': False, 'new_model_inference': False, 'original_word_outputs_changed': False,
            'plan_sha256': ph}
        value['payload_sha256'] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        seal(path, value); features[sid] = value
    print(f'[features {split}] {len(features)}', flush=True)
    return features


def ranked(features, ids, family, fraction):
    n = round(len(ids)*fraction)
    return sorted_ids({s: features[s]['scores'][family] for s in ids}, ids)[:n]


def freeze(cfg, plan, root, features, branches):
    selection = plan['selection_ids']; calibration = plan['threshold_calibration_ids']
    frozen = {'plan_sha256': sha(root/'resolved_configuration.json'), 'test_labels_used': False,
              'test_features_present_at_freeze': 0, 'budgets': {}}
    for fraction in cfg['bypass_fractions']:
        trials = []; cuts = {}
        for index, family in enumerate(plan['families']):
            accepted = ranked(features, selection, family, fraction)
            p = evaluate(branches, selection, accepted, cfg['quality_f1_threshold'])
            trials.append((p['combined']['f1'], -index, family))
            chosen = ranked(features, calibration, family, fraction)
            cuts[family] = min((features[s]['scores'][family] for s in chosen), default=None)
        frozen['budgets'][str(fraction)] = {'selected_family': max(trials)[2], 'scalar_cuts': cuts,
            'selection_trials': {f: value for value, _, f in trials}}
    path = root/'frozen_train_choices.json'
    if not path.exists() and list((root/'test/features').glob('*.json')):
        raise ValueError('Cannot calibrate after new TEST features')
    seal(path, frozen); print('[frozen] TRAIN choices before TEST derived features', flush=True)
    return frozen


def point(cfg, branches, ids, accepted):
    p = evaluate(branches, ids, accepted, cfg['quality_f1_threshold'])
    p['secondary_good_images'] = sum(good(branches[s]['light'], cfg['secondary_quality_f1_threshold']) for s in accepted)
    return p


def evaluate_split(cfg, plan, features, branches, ids, frozen):
    out = {'images': len(ids), 'light_only': aggregate(branches[s]['light'] for s in ids),
           'deepsolo_only': aggregate(branches[s]['deepsolo'] for s in ids), 'budgets': {}}
    for fraction in cfg['bypass_fractions']:
        fits = frozen['budgets'][str(fraction)]; families = {}; scalar = {}
        for family in plan['families']:
            accepted = ranked(features, ids, family, fraction)
            families[family] = {'requested_images': round(len(ids)*fraction),
                'unfilled_budget': round(len(ids)*fraction)-len(accepted), **point(cfg, branches, ids, accepted)}
            cut = fits['scalar_cuts'][family]
            accepted = [s for s in ids if cut is not None and features[s]['scores'][family] is not None
                        and features[s]['scores'][family] >= cut]
            scalar[family] = {'threshold': cut, **point(cfg, branches, ids, accepted)}
        out['budgets'][str(fraction)] = {'families': families, 'scalar_thresholds': scalar,
            'selected_family': fits['selected_family'], 'selected': families[fits['selected_family']],
            'primary_family': cfg['primary_family'], 'primary': families[cfg['primary_family']]}
    return out


def intervals(cfg, result, branches):
    ids = sorted(branches); rng = np.random.default_rng(cfg['seed'])
    weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=cfg['bootstrap_iterations']).astype(float)
    def distribution(accepted):
        accepted = set(accepted)
        rows = np.array([[branches[s]['light' if s in accepted else 'deepsolo'][k]
                          for k in ('tp', 'fp', 'fn')] for s in ids])
        c = weights@rows
        return 2*c[:, 0]/(2*c[:, 0]+c[:, 1]+c[:, 2])
    ds = distribution([]); out = {}
    for budget, data in result['splits']['test']['budgets'].items():
        baseline = distribution(data['families']['base_proxy_f1']['accepted_ids']); out[budget] = {}
        for name in ('primary', 'selected'):
            values = distribution(data[name]['accepted_ids'])
            out[budget][name] = {'delta_vs_base_ci95': np.quantile(values-baseline, [.025, .975]).tolist(),
                'delta_vs_deepsolo_ci95': np.quantile(values-ds, [.025, .975]).tolist(),
                'iterations': cfg['bootstrap_iterations'], 'gates_fixed': True, 'multiplicity_correction': False}
    return out


def run(args):
    start = time.monotonic(); cfg, plan, root, records, branches = prepare(args)
    if args.preflight:
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'root': str(root), 'families': plan['families']})); return
    seal(root/'resolved_configuration.json', plan)
    for file, digest in plan['code_sha256'].items():
        path = root/'code_snapshot'/file; path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and sha(path) != digest:
            raise ValueError('Code snapshot changed')
        if not path.exists(): path.write_bytes((REPO/file).read_bytes())
    train = feature_split(cfg, plan, root, 'train', records['train'])
    frozen = freeze(cfg, plan, root, train, branches['train'])
    test = feature_split(cfg, plan, root, 'test', records['test'])
    populations = [('train', train, branches['train'], sorted(train)),
        ('selection', train, branches['train'], plan['selection_ids']),
        ('calibration', train, branches['train'], plan['threshold_calibration_ids']),
        ('test', test, branches['test'], sorted(test))]
    result = {'status': 'SUCCESS', 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'result_class': plan['result_class'], 'table_5_claim': False, 'plan_sha256': sha(root/'resolved_configuration.json'),
        'frozen_choices_sha256': sha(root/'frozen_train_choices.json'), 'network_training': False,
        'optimizer_steps': 0, 'new_model_inference': False, 'metric': plan['metric'],
        'splits': {name: evaluate_split(cfg, plan, f, b, ids, frozen) for name, f, b, ids in populations}}
    result['paired_intervals'] = intervals(cfg, result, branches['test'])
    seal(root/'results.json', result)
    if not (root/'runtime.json').exists():
        seal(root/'runtime.json', {'seconds': time.monotonic()-start, 'execution': 'CPU offline cached-output analysis',
            'new_gpu_inference': False, 'peak_allocated_vram_bytes': None, 'peak_reserved_vram_bytes': None})
    print(json.dumps({'status': 'SUCCESS', 'root': str(root), 'results_sha256': sha(root/'results.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', required=True, choices=['approximate'])
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output-root', type=Path); parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
