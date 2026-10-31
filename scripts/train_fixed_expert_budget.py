"""Train common word-risk/count gates without modifying fixed expert emissions."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import (
    grouped_folds, fit_word_risk, fit_count_head, counts, micro, choose_light, route,
)
from recognition.fixed_expert_budget import (
    training_word_labels, observed_features, with_word_risk, fixed_words_with_risk,
    conditional_complement, fixed_heuristic, count_scores,
)
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal


def complexity(bundle, path, width):
    if bundle['recipe']['kind'] == 'ridge':
        return sum(m.coef_.size+m.intercept_.size for m in bundle['models'])+2*width+2
    return Path(path).stat().st_size//8


def train_view(plan, root, view):
    c = plan['config']; data = root/'data'/view; case = read(data/'case.json')
    obs = read(data/'observations.json'); labels = read(data/'labels.json')
    fallbacks = read(root/'data/fallbacks.json'); fit = case['fit_ids']; val = case['validation_ids']
    ids = fit+val; folder = root/'models'/view; folder.mkdir(parents=True, exist_ok=False)
    xword = {}; groups = {}; targets = {}; care = {}; probabilities = {}
    for sid in ids:
        o = obs[sid]
        xword[sid], groups[sid] = observed_features(o['raw'], o['rows'], o['trace'],
            o['image_wh'], o['map_summary'])
        if sid in fit:
            targets[sid] = training_word_labels(o['raw'], o['rows'], labels[sid]['ground_truth'])
            care[sid] = np.logical_not(labels[sid]['raw_ignored'])
    folds = grouped_folds(fit, case['keys'], c['folds'], c['seed']); word_audit = []
    def fit_word(samples, seed):
        x = np.concatenate([xword[s][care[s]] for s in samples])
        y = np.concatenate([targets[s][care[s]] for s in samples])
        return fit_word_risk(x, y, seed, c['word_risk'])
    for i, heldout in enumerate(folds):
        remaining = sorted(set(fit)-set(heldout)); model = fit_word(remaining, c['seed']+i)
        path = folder/f'word_fold_{i}.joblib'; joblib.dump(model, path)
        for sid in heldout:
            probabilities[sid] = model.predict_proba(xword[sid])[:, 1] if len(xword[sid]) else np.empty(0)
        word_audit.append(dict(fit_ids=remaining, heldout_ids=heldout, sha256=sha(path)))
    model = fit_word(fit, c['seed']); joblib.dump(model, folder/'word_full.joblib')
    for sid in val:
        probabilities[sid] = model.predict_proba(xword[sid])[:, 1] if len(xword[sid]) else np.empty(0)
    for sid in ids:
        groups[sid] = with_word_risk(groups[sid], probabilities[sid], obs[sid]['trace'])
    seal(folder/'word_folds.json', dict(folds=word_audit,
        expert_image_excluded=case['expert_image_excluded'],
        scope='Word-fold predictions condition on a fixed bank of source experts; not a nested unbiased estimate'))
    seal(folder/'word_probabilities.json', {s: v.tolist() for s, v in probabilities.items()})
    np.savez_compressed(folder/'features.npz', ids=np.array(ids),
        **{g: np.stack([groups[s][g] for s in ids]) for g in groups[ids[0]]})
    scales = {s: len(obs[s]['raw'])+1 for s in ids}
    def matrix(samples, names):
        return np.stack([np.concatenate([groups[s][g] for g in names]) for s in samples])
    matrices_val = {tuple(names): matrix(val, names) for names in c['feature_groups']}
    quota = round(len(val)*case['fallback_images']/case['total_images'])
    keys = {s: case['keys'][s] for s in val}; results = []
    complement_policies = [dict(confidence=None, maximum_overlap=c['complement_iou'])]+[
        dict(confidence=p, maximum_overlap=c['complement_iou']) for p in c['complement_confidences']]
    for fallback_name in case['fallback_profiles']:
        profile = c['fallback_profiles'][fallback_name]
        policies = complement_policies if profile['primary_eligible'] else complement_policies[:1]
        for ci, complement in enumerate(policies):
            name = fallback_name+'__'+str(ci); target_folder = folder/name
            target_folder.mkdir(exist_ok=False); branches = {}
            for sid in ids:
                light = fixed_words_with_risk(obs[sid]['rows'], probabilities[sid])
                fallback = fallbacks[fallback_name][sid]
                combined = conditional_complement(light, fallback['rows'], complement)
                ignored = [labels[sid]['raw_ignored'][int(r['source_row'])]
                    if r.get('fixed_light_complement') else r['ignored'] for r in combined]
                measured = counts(combined, labels[sid]['ground_truth'], ignored)
                branches[sid] = dict(light=labels[sid]['light'], deepsolo=measured)
            seal(target_folder/'train_branches.json', branches)
            y = np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
                branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']]
                for s in fit])/np.array([scales[s] for s in fit])[:, None]
            weights = np.array([scales[s] for s in fit])
            refs = sorted(set(c['reference_grid']+[micro(branches[s]['deepsolo'] for s in fit)['f1']]))
            vb = {s: branches[s] for s in val}; all_trials = []; predictions = {}; model_meta = {}
            def trial_scores(scores, names, used_groups, scalars, reference=None, heuristic=None):
                best = None
                for guard in c['guards']:
                    eligible = [s for s in val if obs[s]['rows']] if guard else val
                    if len(eligible) < len(val)-quota:
                        continue
                    accepted = choose_light(scores, keys, quota, eligible)
                    result = route(vb, accepted)
                    t = dict(models=names, groups=used_groups, scalars=scalars,
                        reference=reference, heuristic=heuristic, guard=guard, result=result)
                    all_trials.append(t)
                    if best is None or result['combined']['f1'] > best['result']['combined']['f1']:
                        best = t
                return best
            for gi, names in enumerate(c['feature_groups']):
                x = matrix(fit, names)
                for ri, recipe in enumerate(c['head_recipes']):
                    bundle = fit_count_head(x, y, weights, recipe, c['seed']); bundle['groups'] = names
                    model_name = f'head_{gi}_{ri}.joblib'; path = target_folder/model_name
                    joblib.dump(bundle, path)
                    from recognition.common_budget_pipeline import predict_counts
                    predicted = predict_counts(bundle, matrices_val[tuple(names)])
                    predictions[model_name] = predicted
                    scalars = complexity(bundle, path, x.shape[1]); best = None
                    for ref in refs:
                        values = (2*predicted[:, 0]-ref*predicted[:, 1])*np.array([scales[s] for s in val])
                        t = trial_scores(dict(zip(val, values)), [model_name], [names], scalars, reference=ref)
                        if best is None or t['result']['combined']['f1'] > best['result']['combined']['f1']:
                            best = t
                    model_meta[model_name] = best
            ordered = sorted(model_meta, key=lambda k: (-model_meta[k]['result']['combined']['f1'],
                                                       model_meta[k]['scalars'], k))
            for n in c['ensemble_sizes']:
                names = ordered[:n]; predicted = np.mean([predictions[s] for s in names], axis=0)
                used = [g for name in names for g in model_meta[name]['groups']]
                scalars = sum(model_meta[s]['scalars'] for s in names)
                for ref in refs:
                    values = (2*predicted[:, 0]-ref*predicted[:, 1])*np.array([scales[s] for s in val])
                    trial_scores(dict(zip(val, values)), names, used, scalars, reference=ref)
            for heuristic in c['heuristics']:
                values = {s: fixed_heuristic(obs[s]['raw'], obs[s]['rows'], probabilities[s], heuristic) for s in val}
                trial_scores(values, [], [], 0, heuristic=heuristic)
            best = max(t['result']['combined']['f1'] for t in all_trials)
            winner = min((t for t in all_trials if t['result']['combined']['f1'] >= best-c['selection_tolerance']),
                key=lambda t: (t['scalars'], -t['result']['combined']['f1'], len(t['models']),
                               t['models'], t['reference'] or 0, t['heuristic'] or '', t['guard']))
            seal(target_folder/'trials.json', all_trials)
            seal(target_folder/'winner.json', winner)
            detail = dict(view=view, target_name=name, fallback_profile=fallback_name,
                complement=complement, winner=winner, primary_eligible=profile['primary_eligible'],
                expert_image_excluded=case['expert_image_excluded'],
                additional_map_models=int(case['map_available'] and any('maps' in gs for gs in winner['groups']))*case['map_extra_models'],
                extra_reader_crops=profile['extra_reader_crops'],
                word_model_bytes=(folder/'word_full.joblib').stat().st_size)
            results.append(detail)
            print(__import__('json').dumps(dict(stage='fixed_gate_trained', view=view,
                fallback=fallback_name, complement=complement['confidence'],
                f1=winner['result']['combined']['f1'], deepsolo_images=quota,
                total_images=len(val), deepsolo_percentage=100*quota/len(val))), flush=True)
    seal(folder/'completed.json', dict(status='COMPLETE', results=results,
        data_input_sha256=sha(data/'input_sha256.json'), test_used_for_selection=False))


def freeze(plan, root):
    c = plan['config']; choices = []
    for view in c['required_views']:
        result = read(root/'models'/view/'completed.json')
        if result['status'] != 'COMPLETE' or result['test_used_for_selection']:
            raise ValueError('Complete TRAIN-only view fitting required')
        choices.extend(r for r in result['results'] if r['primary_eligible'])
    best = max(r['winner']['result']['combined']['f1'] for r in choices)
    winner = min((r for r in choices if r['winner']['result']['combined']['f1'] >= best-c['selection_tolerance']),
        key=lambda r: (r['additional_map_models'], r['extra_reader_crops'],
            not r['expert_image_excluded'], r['winner']['scalars'],
            -r['winner']['result']['combined']['f1'], r['view'], r['target_name']))
    folder = root/'models'/winner['view']; target = folder/winner['target_name']
    paths = [folder/'word_full.joblib', root/'data'/winner['view']/'case.json',
        target/'winner.json', root/'resolved_configuration.json']+[target/p for p in winner['winner']['models']]
    seal(root/'selection_frozen.json', dict(status='FROZEN', winner=winner,
        choices=choices, artifact_sha256={str(p.relative_to(root)): sha(p) for p in paths},
        test_used_for_selection=False, validation_target_met=winner['winner']['result']['combined']['f1'] >= c['minimum_f1']))
    print('FROZEN FIXED EXPERT PIPELINE', winner['view'], winner['target_name'],
          winner['winner']['result']['combined']['f1'], flush=True)


def main():
    p = argparse.ArgumentParser(__doc__); p.add_argument('--run-root', type=Path, required=True)
    p.add_argument('--views', nargs='+'); p.add_argument('--stage', choices=['train', 'freeze', 'all'], default='all')
    args = p.parse_args(); root = args.run_root; plan = read(root/'resolved_configuration.json')
    if plan['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    with threadpool_limits(limits=plan['config']['cpu_threads']):
        if args.stage in ('train', 'all'):
            for view in args.views or plan['config']['required_views']:
                train_view(plan, root, view)
        if args.stage in ('freeze', 'all'):
            freeze(plan, root)


if __name__ == '__main__':
    main()
