"""Select a fixed-emission image gate using FIT cross-fitting and validation."""

import argparse
from pathlib import Path
import shutil

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import grouped_folds, fit_count_head, predict_counts, choose_light, route, micro
from recognition.fixed_expert_budget import fixed_heuristic
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def prepare(config_path):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate configuration required')
    c = read(config_path)
    if c['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for path, digest in c['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed cross-fitted source: '+path)
    source = Path(c['source_root']); q = read(source/'deployment_qualification.json')
    if q['status'] != 'PASS' or sha(q['audit_path']) != q['audit_sha256']:
        raise ValueError('Independently qualified source gate required')
    for path, digest in q['artifact_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Changed source gate qualification')
    original = read(source/'resolved_configuration.json'); view = c['source_view']; case = read(source/'data'/view/'case.json')
    if c['require_expert_excluded'] and not case['expert_image_excluded']:
        raise ValueError('Declared expert-image-excluded training view is required')
    profiles = [r for r in read(source/'models'/view/'completed.json')['results'] if r['primary_eligible']]
    if not profiles:
        raise ValueError('No eligible original fallback profile')
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    settings = dict(original['config']); settings['output_root'] = str(root)
    settings['cpu_threads'] = c.get('cpu_threads', settings['cpu_threads'])
    settings['required_views'] = [view]; settings['scientific_choices'] = c['scientific_choices']
    files = {}
    for package in settings['snapshot_packages']:
        for path in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
            files[str(path.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    seal(root/'resolved_configuration.json', dict(original, config=settings, source_sha256=files,
        crossfit_config=c, crossfit_config_sha256=sha(config_path), source_root=str(source),
        source_configuration_sha256=sha(source/'resolved_configuration.json'),
        source_qualification_sha256=sha(source/'deployment_qualification.json'),
        test_labels_opened=False, test_used_for_selection=False))
    destination = root/'data'/view; destination.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source/'data'/view/'case.json', destination/'case.json')
    folder = root/'models'/view; folder.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source/'models'/view/'word_full.joblib', folder/'word_full.joblib')
    pins = {}
    for rel in ['resolved_configuration.json', 'deployment_qualification.json', 'data/'+view+'/case.json',
                'data/'+view+'/observations.json', 'data/'+view+'/labels.json',
                'models/'+view+'/features.npz', 'models/'+view+'/word_probabilities.json',
                'models/'+view+'/word_full.joblib']:
        path = source/rel; pins[str(path)] = sha(path)
    if settings.get('emission_override') is not None:
        path = Path(settings['emission_override']['policy']); pins[str(path)] = sha(path)
    for r in profiles:
        path = source/'models'/view/r['target_name']/'train_branches.json'; pins[str(path)] = sha(path)
        for gi in range(len(settings['feature_groups'])):
            for ri in range(len(settings['head_recipes'])):
                path = path.parent/f'head_{gi}_{ri}.joblib'; pins[str(path)] = sha(path)
                path = path.parent/'train_branches.json'
    seal(root/'source_input_sha256.json', pins); seal(root/'targets.json', profiles)
    print('PREPARED CROSSFIT GATE', view, len(case['fit_ids']), len(case['validation_ids']), len(profiles), flush=True)


def fit_target(root, target_name):
    plan = read(root/'resolved_configuration.json'); c = plan['config']; cv = plan['crossfit_config']; source = Path(plan['source_root'])
    view = cv['source_view']; case = read(root/'data'/view/'case.json'); fit = case['fit_ids']; val = case['validation_ids']; ids = fit+val
    details = {r['target_name']: r for r in read(root/'targets.json')}; detail = details[target_name]
    branches = read(source/'models'/view/target_name/'train_branches.json')
    obs = read(source/'data'/view/'observations.json'); probabilities = read(source/'models'/view/'word_probabilities.json')
    with np.load(source/'models'/view/'features.npz') as z:
        if z['ids'].tolist() != ids:
            raise ValueError('Crossfit image feature ordering differs')
        groups = {g: z[g].copy() for g in z.files if g != 'ids'}
    positions = {s: i for i, s in enumerate(ids)}; scale = {s: len(obs[s]['raw'])+1 for s in ids}
    def matrix(samples, names):
        return np.concatenate([groups[g][[positions[s] for s in samples]] for g in names], axis=1)
    weights = np.array([scale[s] for s in fit]); y = np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
        branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']] for s in fit])/weights[:, None]
    folds = grouped_folds(fit, case['keys'], c['folds'], c['seed'])
    target = root/'models'/view/target_name; target.mkdir(parents=True, exist_ok=False)
    full_predictions = {}; cv_predictions = {}; scalars = {}; groups_by_model = {}
    for gi, names in enumerate(c['feature_groups']):
        x = matrix(fit, names); xv = matrix(val, names)
        for ri, recipe in enumerate(c['head_recipes']):
            name = f'head_{gi}_{ri}.joblib'; path = source/'models'/view/target_name/name
            bundle = joblib.load(path)
            if bundle['groups'] != names or bundle['recipe'] != recipe:
                raise ValueError('Changed full-FIT count head recipe')
            full_predictions[name] = predict_counts(bundle, xv); values = np.zeros((len(fit), 2))
            for i, held in enumerate(folds):
                test_idx = [fit.index(s) for s in held]; train_idx = [j for j, s in enumerate(fit) if s not in set(held)]
                local = fit_count_head(x[train_idx], y[train_idx], weights[train_idx], recipe, c['seed'])
                values[test_idx] = predict_counts(local, x[test_idx])
            cv_predictions[name] = values; groups_by_model[name] = names
            scalars[name] = (sum(m.coef_.size+m.intercept_.size for m in bundle['models'])+2*x.shape[1]+2
                if recipe['kind'] == 'ridge' else path.stat().st_size//8)
    np.savez_compressed(target/'head_predictions.npz', fit_ids=np.array(fit), validation_ids=np.array(val),
        **{'fit__'+k: v for k, v in cv_predictions.items()}, **{'val__'+k: v for k, v in full_predictions.items()})
    quota_fit = round(len(fit)*case['fallback_images']/case['total_images'])
    quota_val = round(len(val)*case['fallback_images']/case['total_images'])
    trials = []; model_best = {}
    def candidate(fit_scores, val_scores, names, features, size, *, reference=None, heuristic=None):
        best = None
        for guard in c['guards']:
            fi = [s for s in fit if obs[s]['rows']] if guard else fit
            va = [s for s in val if obs[s]['rows']] if guard else val
            if len(fi) < len(fit)-quota_fit or len(va) < len(val)-quota_val:
                continue
            fit_keep = choose_light(dict(zip(fit, fit_scores)), {s: case['keys'][s] for s in fit}, quota_fit, fi)
            val_keep = choose_light(dict(zip(val, val_scores)), {s: case['keys'][s] for s in val}, quota_val, va)
            fit_result = route({s: branches[s] for s in fit}, fit_keep)
            val_result = route({s: branches[s] for s in val}, val_keep)
            selected = set(fit_keep+val_keep)
            pooled = micro(branches[s]['light' if s in selected else 'deepsolo'] for s in ids)
            value = dict(models=names, groups=features, scalars=size, reference=reference, heuristic=heuristic,
                guard=guard, result=val_result, crossfit=fit_result, pooled_counts=pooled,
                selection_score=pooled['f1'], total_selection_images=len(ids),
                deepsolo_selection_images=quota_fit+quota_val,
                deepsolo_selection_percentage=100*(quota_fit+quota_val)/len(ids))
            trials.append(value)
            if best is None or value['selection_score'] > best['selection_score']:
                best = value
        return best
    for name in cv_predictions:
        best = None
        for ref in c['reference_grid']:
            f = (2*cv_predictions[name][:, 0]-ref*cv_predictions[name][:, 1])*weights
            v = (2*full_predictions[name][:, 0]-ref*full_predictions[name][:, 1])*np.array([scale[s] for s in val])
            value = candidate(f, v, [name], [groups_by_model[name]], scalars[name], reference=ref)
            if best is None or value['selection_score'] > best['selection_score']:
                best = value
        model_best[name] = best
    order = sorted(model_best, key=lambda name: (-model_best[name]['selection_score'], scalars[name], name))
    for n in c['ensemble_sizes']:
        names = order[:n]; fp = np.mean([cv_predictions[x] for x in names], axis=0)
        vp = np.mean([full_predictions[x] for x in names], axis=0)
        for ref in c['reference_grid']:
            candidate((2*fp[:, 0]-ref*fp[:, 1])*weights,
                (2*vp[:, 0]-ref*vp[:, 1])*np.array([scale[s] for s in val]),
                names, [groups_by_model[x] for x in names], sum(scalars[x] for x in names), reference=ref)
    for name in c['heuristics']:
        scores = {s: fixed_heuristic(obs[s]['raw'], obs[s]['rows'], probabilities[s], name) for s in ids}
        candidate([scores[s] for s in fit], [scores[s] for s in val], [], [], 0, heuristic=name)
    eligible = [t for t in trials if t['result']['combined']['f1'] >= cv['minimum_validation_f1']]
    if not eligible:
        raise ValueError('No cross-fitted candidate meets declared validation floor')
    best = max(t['selection_score'] for t in eligible)
    winner = min((t for t in eligible if t['selection_score'] >= best-cv['selection_tolerance']),
        key=lambda t: (t['scalars'], -t['selection_score'], len(t['models']), t['models'],
                       t['reference'] or 0, t['heuristic'] or '', t['guard']))
    seal(target/'trials.json', trials); seal(target/'winner.json', winner)
    seal(target/'completed.json', dict(status='COMPLETE', folds=folds, winner=winner,
        target_name=target_name, expert_image_excluded=case['expert_image_excluded'],
        test_used_for_selection=False, fit_images=len(fit), validation_images=len(val)))
    print(__import__('json').dumps(dict(status='CROSSFIT_COMPLETE', target=target_name,
        pooled_f1=winner['selection_score'], validation_f1=winner['result']['combined']['f1'],
        deepsolo_images=winner['deepsolo_selection_images'], total_images=len(ids),
        deepsolo_percentage=winner['deepsolo_selection_percentage'])), flush=True)


def freeze(root):
    plan = read(root/'resolved_configuration.json'); c = plan['config']; cv = plan['crossfit_config']; view = cv['source_view']
    case = read(root/'data'/view/'case.json')
    choices = []
    for detail in read(root/'targets.json'):
        target = root/'models'/view/detail['target_name']; status = read(target/'completed.json')
        if status['status'] != 'COMPLETE' or status['test_used_for_selection']:
            raise ValueError('Every eligible crossfit target must complete before selection')
        selected = status['winner']
        extra_maps = int(case['map_available'] and any('maps' in g for g in selected['groups']))*case['map_extra_models']
        choices.append(dict(detail, winner=selected, additional_map_models=extra_maps))
    best = max(r['winner']['selection_score'] for r in choices)
    winner = min((r for r in choices if r['winner']['selection_score'] >= best-cv['selection_tolerance']),
        key=lambda r: (r['additional_map_models'], r['extra_reader_crops'], r['winner']['scalars'],
                       -r['winner']['selection_score'], r['target_name']))
    target = root/'models'/view/winner['target_name']; source = Path(plan['source_root'])/'models'/view/winner['target_name']
    for model in winner['winner']['models']:
        shutil.copyfile(source/model, target/model)
    paths = [root/'models'/view/'word_full.joblib', root/'data'/view/'case.json', target/'winner.json', root/'resolved_configuration.json']+[target/m for m in winner['winner']['models']]
    for path, digest in read(root/'source_input_sha256.json').items():
        if sha(path) != digest:
            raise ValueError('Crossfit source changed during training')
    seal(root/'selection_frozen.json', dict(status='FROZEN', winner=winner, choices=choices,
        artifact_sha256={str(p.relative_to(root)): sha(p) for p in paths}, test_used_for_selection=False,
        validation_target_met=winner['winner']['result']['combined']['f1'] >= cv['minimum_validation_f1']))
    print('FROZEN CROSSFIT GATE', winner['target_name'], winner['winner']['selection_score'],
        winner['winner']['result']['combined']['f1'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path); p.add_argument('--run-root', type=Path)
    p.add_argument('--stage', choices=['prepare', 'fit', 'freeze'], required=True); p.add_argument('--target')
    a = p.parse_args()
    if a.stage == 'prepare':
        prepare(a.config)
    else:
        plan = read(a.run_root/'resolved_configuration.json')
        with threadpool_limits(limits=plan['config']['cpu_threads']):
            if a.stage == 'fit':
                fit_target(a.run_root, a.target)
            else:
                freeze(a.run_root)
