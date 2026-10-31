"""Train and evaluate approximate image routers against a fixed-budget GT oracle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold

from recognition.f1_quality_study import aggregate, evaluate
from recognition.trainable_image_router import (
    branch_f1, feature_dict, fit_model, oracle, predict, regression_targets, score,
)
from scripts.run_yolo_f1_quality_study import checked, checked_feature, read, seal, sha

REPO = Path(__file__).resolve().parents[1]
CODE = ('scripts/run_trainable_image_router.py', 'recognition/trainable_image_router.py',
        'recognition/f1_quality_study.py', 'scripts/run_yolo_f1_quality_study.py')


def prepare(args):
    if args.mode != 'approximate' or any(p.resolve().parent != REPO/'configs/approximate' for p in (args.config, args.paths)):
        raise ValueError('Isolated explicit approximate configurations required')
    cfg, paths = read(args.config), read(args.paths)
    if cfg['reproduction_mode'] != 'approximate' or paths['reproduction_mode'] != 'approximate' or cfg['table_5_claim']:
        raise ValueError('Approximate claim metadata required')
    if set(cfg['model_kinds']) != {'ridge', 'trees', 'oracle_logistic', 'oracle_trees'}:
        raise ValueError('Unknown supervised model family')
    if cfg['cv_folds'] < 2 or any(not 0 < f < 1 for f in cfg['bypass_fractions']):
        raise ValueError('Invalid explicit CV or budget settings')
    roots = {name: Path(paths[name+'_root']).resolve() for name in ('residual', 'quality')}
    plans = {}; results = {}; inputs = {}
    for name, root in roots.items():
        pins = cfg['sources'][name]
        plans[name] = checked(root/'resolved_configuration.json', pins['plan'])
        results[name] = checked(root/'results.json', pins['results'])
        audit = checked(root/'independent_audit.json', pins['audit'])
        if results[name]['status'] != 'SUCCESS' or audit['status'] != 'PASS' or audit['results_sha256'] != pins['results']:
            raise ValueError('A complete audited source is required')
        for file, digest in plans[name]['code_sha256'].items():
            if sha(root/'code_snapshot'/file) != digest: raise ValueError('Source code snapshot changed')
    old = plans['residual']; coverage = Path(old['source_coverage_root'])
    branch_path = coverage/'results.json'; expected = plans['quality']['config']['sources']['coverage']['results']
    branches = checked(branch_path, expected)['resolutions']['640']['branch_metrics']; inputs[str(branch_path)] = expected
    fit_ids, validation_ids = old['selection_ids'], old['threshold_calibration_ids']; samples = old['samples']
    if args.smoke:
        fit_ids = fit_ids[:cfg['smoke_fit_images']]; validation_ids = validation_ids[:cfg['smoke_validation_images']]
        samples = {'train': [s for s in samples['train'] if s['sample_id'] in fit_ids+validation_ids],
                   'test': samples['test'][:cfg['smoke_test_images']]}
    if set(fit_ids)&set(validation_ids) or set(fit_ids+validation_ids) != {s['sample_id'] for s in samples['train']}:
        raise ValueError('Fitting and validation must partition TRAIN')
    if {s['sample_id'] for s in samples['test']}&set(fit_ids+validation_ids): raise ValueError('TRAIN/TEST overlap')
    records = {}
    for split, group in samples.items():
        records[split] = {}
        for sample in group:
            sid = sample['sample_id']; key = sid.replace(':', '__')+'.json'; parts = {}
            for name, root in roots.items():
                path = root/split/'features'/key; value = checked_feature(path, cfg['sources'][name]['plan'])
                if value['ground_truth_used'] is not False or value['sample_id'] != sid or value['image_sha256'] != sample['image_sha256']:
                    raise ValueError('Inference-source sample identity mismatch')
                inputs[str(path)] = sha(path); parts[name] = value
            records[split][sid] = parts
    output = (args.output_root or Path(paths['output_root'])).resolve()
    if output.is_relative_to(REPO) or any(output.is_relative_to(p) or p.is_relative_to(output) for p in roots.values()):
        raise ValueError('Use external outputs separate from frozen sources')
    first = samples['train'][0]; parts = records['train'][first['sample_id']]
    _, schemas, _ = feature_dict(parts['residual'], parts['quality'], first['image_wh'], cfg['feature_settings'])
    families = [kind+'_'+feature_set for kind in cfg['model_kinds'] for feature_set in cfg['feature_sets']]
    if cfg['primary_family'] not in families: raise ValueError('Primary family is undeclared')
    plan = {'schema_version': 1, 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'table_5_claim': False, 'result_class': 'SMOKE_TEST' if args.smoke else 'APPROXIMATE_EXPLORATORY_SUPERVISED_ROUTER',
        'config': cfg, 'paths': paths, 'config_sha256': sha(args.config), 'paths_sha256': sha(args.paths),
        'samples': samples, 'fit_ids': fit_ids, 'validation_ids': validation_ids, 'families': families,
        'feature_schemas': schemas, 'source_coverage_root': str(coverage), 'metric': plans['quality']['metric'],
        'input_sha256': inputs, 'code_sha256': {f: sha(REPO/f) for f in CODE},
        'software': {p: importlib.metadata.version(p) for p in ('numpy', 'scipy', 'scikit-learn')},
        'python': sys.version, 'router_training': True, 'base_models_frozen': True,
        'new_visual_model_inference': False, 'test_labels_used_for_training_or_selection': False,
        'limitations': 'Historically explored TEST; small repeatedly reused TRAIN; DeepSolo trained on Total-Text TRAIN; nonofficial lexical metric.'}
    tag = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    return cfg, plan, output/('smoke' if args.smoke else 'full')/tag, records, branches


def features(cfg, plan, root, split, records):
    result = {}; ph = sha(root/'resolved_configuration.json')
    # Bind TEST vectors to choices that already exist; filesystem timestamps
    # may share the same clock tick for consecutive small atomic writes.
    binding = sha(root/'frozen_choices.json') if split == 'test' else None
    for sample in plan['samples'][split]:
        sid = sample['sample_id']; parts = records[sid]
        vector, schemas, eligible = feature_dict(parts['residual'], parts['quality'], sample['image_wh'], cfg['feature_settings'])
        if schemas != plan['feature_schemas']: raise ValueError('Feature schema changed')
        value = {'sample_id': sid, 'image_sha256': sample['image_sha256'], 'values': vector, 'eligible': eligible,
            'controls': {f: parts['residual']['scores'][f] for f in cfg['controls']},
            'per_image_ground_truth_used': False, 'actual_deepsolo_outputs_used': False,
            'frozen_choices_sha256': binding}
        seal(root/split/'features'/(sid.replace(':', '__')+'.json'), {'plan_sha256': ph, **value})
        result[sid] = value
    return result


def matrix(features, ids, names):
    return np.array([[features[s]['values'][name] for name in names] for s in ids], dtype=float)


def recipes(kind, cfg):
    ms = cfg['model_settings']
    if kind == 'ridge': return [{'kind': kind, 'alpha': a} for a in ms['ridge_alphas']]
    if kind == 'oracle_logistic': return [{'kind': kind, 'C': c} for c in ms['logistic_C']]
    return [{'kind': kind, 'max_depth': d, 'min_samples_leaf': n}
            for d, n in itertools.product(ms['tree_depths'], ms['tree_leaf_sizes'])]


def family_parts(family, cfg):
    kind, feature_set = family.rsplit('_', 1)
    if kind not in cfg['model_kinds'] or feature_set not in cfg['feature_sets']: raise ValueError('Unknown model family')
    return kind, feature_set


def train_one(cfg, plan, features, branches, ids, recipe, family, fraction):
    if not set(ids) <= set(plan['fit_ids']):
        raise ValueError('Fit ids outside the declared TRAIN fitting population')
    kind, feature_set = family_parts(family, cfg); labels = {s: branches[s] for s in ids}
    teacher = None
    if kind.startswith('oracle'):
        teacher = oracle(labels, ids, [s for s in ids if features[s]['eligible']], round(len(ids)*fraction))
        accepted = set(teacher['accepted_ids']); y = np.array([int(s in accepted) for s in ids])
    else:
        y = regression_targets(labels, ids)
    names = plan['feature_schemas'][feature_set]
    model = fit_model(matrix(features, ids, names), y, ids, recipe, cfg['model_settings'])
    model.update(family=family, feature_names=names, fraction=fraction,
        reference_f1=float(branch_f1(labels, ids, [])), teacher_oracle=teacher)
    return model


def predict_ids(model, features, ids):
    values = predict(model, matrix(features, ids, model['feature_names']))
    scores = score(values, model['recipe']['kind'], model['reference_f1'])
    return {s: float(v) if features[s]['eligible'] else None for s, v in zip(ids, scores)}


def top(scores, ids, fraction):
    if set(scores) != set(ids): raise ValueError('Ranking population differs')
    return sorted((s for s in ids if scores[s] is not None), key=lambda s: (-scores[s], s))[:round(len(ids)*fraction)]


def fit_and_freeze(cfg, plan, root, train, branches):
    fit_ids, val_ids = plan['fit_ids'], plan['validation_ids']
    folds = list(KFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=cfg['model_settings']['seed']).split(fit_ids))
    chosen = {'plan_sha256': sha(root/'resolved_configuration.json'), 'test_labels_used': False,
        'test_predictions_present_at_freeze': 0, 'test_features_present_at_freeze': 0,
        'fit_ids': fit_ids, 'validation_ids': val_ids, 'budgets': {}, 'models_sha256': {}}
    all_models = {}; cv = {}; fit_calls = 0
    for fraction in cfg['bypass_fractions']:
        key = str(fraction); all_models[key] = {}; cv[key] = {}; validation_scores = {}; trial_f1 = []
        for index, family in enumerate(plan['families']):
            kind, _ = family_parts(family, cfg); trials = []
            for recipe in recipes(kind, cfg):
                oof = {}; trace = []
                for a, b in folds:
                    training = [fit_ids[i] for i in a]; held = [fit_ids[i] for i in b]
                    model = train_one(cfg, plan, train, branches, training, recipe, family, fraction); fit_calls += 1
                    oof.update(predict_ids(model, train, held))
                    trace.append({'training_ids': training, 'held_out_ids': held,
                        'reference_f1': model['reference_f1'], 'teacher_oracle': model['teacher_oracle']})
                accepted = top(oof, fit_ids, fraction); counts = evaluate(branches, fit_ids, accepted, cfg['quality_f1_threshold'])
                trials.append({'recipe': recipe, 'oof_scores': oof, 'oof_point': counts, 'fold_trace': trace})
            winner_index = max(range(len(trials)), key=lambda i: (branch_f1(branches, fit_ids,
                trials[i]['oof_point']['accepted_ids']), -i))
            cv[key][family] = {'selected_trial': winner_index, 'trials': trials}
            winner = trials[winner_index]
            model = train_one(cfg, plan, train, branches, fit_ids, winner['recipe'], family, fraction); fit_calls += 1
            path = root/'models'/(key+'__'+family+'.json'); seal(path, model)
            chosen['models_sha256'][str(path.relative_to(root))] = sha(path); all_models[key][family] = model
            validation_scores[family] = predict_ids(model, train, val_ids)
            accepted = top(validation_scores[family], val_ids, fraction)
            trial_f1.append((branch_f1(branches, val_ids, accepted), -index, family))
        selected = max(trial_f1)[2]; cuts = {}
        for family, scores in validation_scores.items():
            accepted = top(scores, val_ids, fraction); cuts[family] = min((scores[s] for s in accepted), default=None)
        chosen['budgets'][key] = {'selected_family': selected, 'scalar_cuts': cuts,
            'validation_f1': {family: float(f) for f, _, family in trial_f1}}
        print(f'[fit {fraction}] {len(plan["families"])} families; VAL-selected {selected}', flush=True)
    seal(root/'cv_results.json', cv)
    chosen['cv_results_sha256'] = sha(root/'cv_results.json'); chosen['fit_calls'] = fit_calls
    path = root/'frozen_choices.json'
    if not path.exists() and (list((root/'test/features').glob('*.json')) or (root/'predictions/test.json').exists()):
        raise ValueError('Cannot retrain/calibrate after TEST predictions or derived features')
    seal(path, chosen); print('[frozen] all trained models/choices before new TEST predictions', flush=True)
    return chosen, all_models, cv


def evaluate_split(cfg, plan, features, branches, ids, models, frozen):
    out = {'images': len(ids), 'light_only': aggregate(branches[s]['light'] for s in ids),
        'deepsolo_only': aggregate(branches[s]['deepsolo'] for s in ids), 'budgets': {}}
    predictions = {}
    for fraction in cfg['bypass_fractions']:
        key = str(fraction); fits = frozen['budgets'][key]; families = {}; scalar = {}; predictions[key] = {}
        reference = oracle({s: branches[s] for s in ids}, ids,
            [s for s in ids if features[s]['eligible']], round(len(ids)*fraction))
        oracle_cache = {round(len(ids)*fraction): reference}
        def point(accepted):
            p = evaluate(branches, ids, accepted, cfg['quality_f1_threshold']); accepted = set(accepted)
            if len(accepted) not in oracle_cache:
                oracle_cache[len(accepted)] = oracle({s: branches[s] for s in ids}, ids,
                    [s for s in ids if features[s]['eligible']], len(accepted))
            matched = oracle_cache[len(accepted)]; reference_set = set(matched['accepted_ids'])
            p.update(oracle_reference_accepted_images=len(accepted), oracle_reference_f1=matched['f1'],
                oracle_f1_gap=matched['f1']-p['combined']['f1'],
                canonical_oracle_overlap=len(accepted&reference_set),
                canonical_decision_agreement=1-len(accepted^reference_set)/len(ids),
                canonical_symmetric_difference=len(accepted^reference_set))
            return p
        for family, model in models[key].items():
            values = predict_ids(model, features, ids); predictions[key][family] = values
            accepted = top(values, ids, fraction)
            families[family] = {'unfilled_budget': round(len(ids)*fraction)-len(accepted), **point(accepted)}
            cut = fits['scalar_cuts'][family]
            scalar[family] = {'threshold': cut, **point([s for s in ids if cut is not None and values[s] is not None and values[s] >= cut])}
        controls = {family: point(top({s: features[s]['controls'][family] for s in ids}, ids, fraction))
                    for family in cfg['controls']}
        out['budgets'][key] = {'families': families, 'scalar_thresholds': scalar, 'controls': controls,
            'oracle': {**reference, 'point': point(reference['accepted_ids'])},
            'selected_family': fits['selected_family'], 'selected': families[fits['selected_family']],
            'primary_family': cfg['primary_family'], 'primary': families[cfg['primary_family']]}
    return out, predictions


def intervals(cfg, result, branches):
    ids = sorted(branches); rng = np.random.default_rng(cfg['model_settings']['seed'])
    weights = rng.multinomial(len(ids), np.full(len(ids), 1/len(ids)), size=cfg['bootstrap_iterations']).astype(float)
    def distribution(accepted):
        a = set(accepted); rows = np.array([[branches[s]['light' if s in a else 'deepsolo'][k] for k in ('tp', 'fp', 'fn')] for s in ids])
        c = weights@rows; return 2*c[:, 0]/(2*c[:, 0]+c[:, 1]+c[:, 2])
    out = {}; ds = distribution([])
    for key, data in result['splits']['test']['budgets'].items():
        out[key] = {}
        for policy in ('primary', 'selected'):
            p = distribution(data[policy]['accepted_ids']); out[key][policy] = {'all_deepsolo': np.quantile(p-ds, [.025, .975]).tolist()}
            for name, control in data['controls'].items():
                out[key][policy][name] = np.quantile(p-distribution(control['accepted_ids']), [.025, .975]).tolist()
    return {'intervals': out, 'iterations': cfg['bootstrap_iterations'], 'gates_fixed': True, 'multiplicity_correction': False}


def run(args):
    warnings.simplefilter('error', ConvergenceWarning)
    start = time.monotonic(); cfg, plan, root, records, branches = prepare(args)
    if args.preflight:
        print(json.dumps({'status': 'PREFLIGHT_PASS', 'root': str(root), 'families': plan['families'],
            'features': {k: len(v) for k, v in plan['feature_schemas'].items()}})); return
    seal(root/'resolved_configuration.json', plan)
    for file, digest in plan['code_sha256'].items():
        path = root/'code_snapshot'/file; path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and sha(path) != digest: raise ValueError('Frozen code changed')
        if not path.exists(): path.write_bytes((REPO/file).read_bytes())
    train = features(cfg, plan, root, 'train', records['train'])
    frozen, models, cv = fit_and_freeze(cfg, plan, root, train, branches['train'])
    test = features(cfg, plan, root, 'test', records['test'])
    result = {'status': 'SUCCESS', 'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction',
        'result_class': plan['result_class'], 'table_5_claim': False, 'plan_sha256': sha(root/'resolved_configuration.json'),
        'frozen_choices_sha256': sha(root/'frozen_choices.json'), 'router_training': True,
        'base_models_frozen': True, 'new_visual_model_inference': False, 'splits': {}}
    for name, feature, b, ids in [('train', train, branches['train'], sorted(train)),
        ('fit', train, branches['train'], plan['fit_ids']), ('validation', train, branches['train'], plan['validation_ids']),
        ('test', test, branches['test'], sorted(test))]:
        result['splits'][name], predictions = evaluate_split(cfg, plan, feature, b, ids, models, frozen)
        seal(root/'predictions'/(name+'.json'), predictions)
    result['paired_intervals'] = intervals(cfg, result, branches['test'])
    seal(root/'results.json', result)
    if not (root/'runtime.json').exists():
        seal(root/'runtime.json', {'seconds': time.monotonic()-start, 'device': 'CPU', 'fit_calls': frozen['fit_calls'],
            'base_model_updates': 0, 'new_visual_model_inference': False, 'peak_allocated_vram_bytes': None, 'peak_reserved_vram_bytes': None})
    print(json.dumps({'status': 'SUCCESS', 'root': str(root), 'results_sha256': sha(root/'results.json')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--mode', required=True, choices=['approximate'])
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--paths', type=Path, required=True)
    parser.add_argument('--output-root', type=Path); parser.add_argument('--preflight', action='store_true'); parser.add_argument('--smoke', action='store_true')
    run(parser.parse_args())
