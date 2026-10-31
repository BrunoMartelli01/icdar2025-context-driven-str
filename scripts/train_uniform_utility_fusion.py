"""Fit an explicit common count-utility family and conservative fallback fusion."""

import argparse
import itertools
import math
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import predictions, key
from segmentation.uniform_unet_fusion import annotation, word_counts, aggregate_counts, file_hash
from recognition.uniform_lexical_risk import lexicon, prepare_words
from recognition.uniform_utility_fusion import word_observation, image_vector, paired_f1_lower
from recognition.common_budget_pipeline import grouped_folds, fit_word_risk, word_features, fit_count_head, predict_counts, choose_light
from recognition.fixed_expert_budget import training_word_labels
from recognition.uniform_risk_fusion import combine
from recognition.uniform_observation_bank import observation_sources, observation_map_path
from recognition.uniform_encoder_context import training_context
from recognition.hard_fallback_budget import validate_policy, batch_quota


def train(config_path, dataset):
    started = time.monotonic()
    c = read(config_path)
    if c['mode'] != 'approximate' or config_path.resolve().parent.name != 'approximate':
        raise ValueError('Explicit approximate common-utility profile required')
    for path, digest in c['pinned_inputs'].items():
        if file_hash(path) != digest:
            raise ValueError('Changed qualified common-utility input: '+path)
    if 'count_recipes' in c:
        recipes = c['count_recipes']
        if not recipes or len({r['kind'] for r in recipes}) != 1:
            raise ValueError('Explicit nonempty single shared count-model family required')
    else:
        recipes = [dict(kind='ridge', alpha=a) for a in c['ridge_alphas']]
    source = Path(c['training_root'])/dataset
    plan = read(source/'resolved_configuration.json')
    lexical_frozen = read(Path(c['lexical_root'])/dataset/'lexical_selection_frozen.json')
    if lexical_frozen['test_used_for_selection'] or lexical_frozen['status'] != 'FROZEN':
        raise ValueError('Validation-only lexical policy required')
    policy = lexical_frozen['winner']['policy']
    dictionary = lexicon(c['lexicons'][dataset], 'train')
    out = Path(c['output_root'])/dataset
    out.mkdir(parents=True, exist_ok=False)
    fit, val = plan['partitions']['fit'], plan['partitions']['selection']
    ids = fit+val
    lookup = {r['sample_id']: r for r in plan['train_records']}
    context = training_context(c, {s:lookup[s] for s in ids})
    index, expert_condition = observation_sources(c, dataset, source, plan)
    fallback = read(c['fallback_train_sources'][dataset])['native_benchmark']
    scenes, truth, ignored, xword, yword, prepared, keys, maps = {}, {}, {}, {}, {}, {}, {}, {}
    for sid in ids:
        entry = index[sid]
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('Changed qualified native U-Net observation')
        scenes[sid] = read(entry['path'])
        if scenes[sid]['ground_truth_used']:
            raise ValueError('Expert generation used annotations')
        if scenes[sid]['rgb_sha256'] != lookup[sid]['rgb_sha256'] or scenes[sid]['image_wh'] != lookup[sid]['image_wh']:
            raise ValueError('Expert observation refers to a different native image')
        raw = scenes[sid]['raw_rows']
        truth[sid], valid = annotation(lookup[sid])
        ignored[sid] = [p['ignored'] for p in predictions(raw, lookup[sid], valid)]
        prepared[sid], trace = prepare_words(raw, dictionary, policy)
        xword[sid] = word_features(raw, trace, scenes[sid]['image_wh'])
        yword[sid] = training_word_labels(raw, prepared[sid], truth[sid])
        keys[sid] = lookup[sid]['rgb_sha256']
        mp = observation_map_path(source, sid, scenes[sid])
        if file_hash(mp) != scenes[sid]['maps_sha256']:
            raise ValueError('Changed actual predicted U-Net map')
        with np.load(mp, allow_pickle=False) as z:
            maps[sid] = z['probabilities']
        measured = word_counts(fallback[sid]['rows'], truth[sid], [r['ignored'] for r in fallback[sid]['rows']])
        if measured != fallback[sid]['counts']:
            raise ValueError('Native fallback/care protocol differs')
    folds = grouped_folds(fit, keys, c['folds'], c['seed'])
    probabilities, fold_receipts = {}, []
    def fit_word(samples, seed):
        xx = np.concatenate([xword[s][np.logical_not(ignored[s])] for s in samples])
        yy = np.concatenate([yword[s][np.logical_not(ignored[s])] for s in samples])
        return fit_word_risk(xx, yy, seed, c['word_risk'])
    for i, held in enumerate(folds):
        remaining = sorted(set(fit)-set(held))
        model = fit_word(remaining, c['seed']+i)
        path = out/f'word_fold_{i}.joblib'; joblib.dump(model, path)
        for sid in held:
            probabilities[sid] = model.predict_proba(xword[sid])[:, 1].tolist() if len(xword[sid]) else []
        fold_receipts.append(dict(fit_ids=remaining, heldout_ids=held, sha256=file_hash(path)))
    model = fit_word(fit, c['seed']); joblib.dump(model, out/'word_model.joblib')
    for sid in val:
        probabilities[sid] = model.predict_proba(xword[sid])[:, 1].tolist() if len(xword[sid]) else []
    write(out/'word_folds.json', dict(folds=fold_receipts, upstream_expert_image_excluded=False,
        fitting_observations=expert_condition))
    write(out/'word_probabilities.json', probabilities)
    teacher = {s: fallback[s]['counts'] for s in ids}
    reference = aggregate_counts(teacher[s] for s in fit)['f1']
    scales = {s: len(scenes[s]['raw_rows'])+1 for s in ids}
    total = len(plan['partitions']['test'])
    quotas = sorted(set([plan['budget']]+[math.ceil(f*total) for f in c['alternate_fallback_fractions'] if math.ceil(f*total) >= plan['budget']]))
    if 'hard_budgets' in c:
        for quota in quotas:validate_policy(c['hard_budgets'],dataset,quota)
    winners, gate_trials, observations = {}, [], {}
    for ai, admission in enumerate(c['admission_thresholds']):
        obs = {}
        for sid in ids:
            # OOF probabilities are used for FIT count targets/features; validation
            # uses the full-FIT model. Filtering never feeds back into word features.
            p = probabilities[sid]
            rows = [dict(r, word_probability=float(p[int(r['source_row'])])) for r in prepared[sid] if p[int(r['source_row'])] >= admission]
            _, trace = prepare_words(scenes[sid]['raw_rows'], dictionary, policy)
            kept = {int(r['source_row']) for r in rows}
            trace = [dict(t, accepted=i in kept, changed=i in kept and t['changed']) for i, t in enumerate(trace)]
            obs[sid] = dict(rows=rows, probabilities=p, trace=trace)
        observations[str(admission)] = obs
        matrix = {s: image_vector(scenes[s]['raw_rows'], scenes[s]['image_wh'], maps[s], obs[s], c['map_settings'],
            context[s] if context is not None else None) for s in ids}
        light = {s: word_counts(obs[s]['rows'], truth[s], [ignored[s][int(r['source_row'])] for r in obs[s]['rows']]) for s in ids}
        x = np.stack([matrix[s] for s in fit])
        y = np.array([[light[s]['tp']-teacher[s]['tp'], light[s]['detected_words']-teacher[s]['detected_words']] for s in fit])/np.array([scales[s] for s in fit])[:, None]
        for ri, recipe in enumerate(recipes):
            bundle = fit_count_head(x, y, np.array([scales[s] for s in fit]), recipe, c['seed'])
            name = f'count_{ai}_{ri}.joblib'; joblib.dump(bundle, out/name)
            pairs = predict_counts(bundle, np.stack([matrix[s] for s in val]))
            scores = {s: float((2*p[0]-reference*p[1])*scales[s]) for s, p in zip(val, pairs)}
            for quota, guard in itertools.product(quotas, c['guards']):
                k = (batch_quota(validate_policy(c['hard_budgets'],dataset,quota),quota,total,len(val))
                     if 'hard_budgets' in c else round(len(val)*quota/total))
                eligible = [s for s in val if obs[s]['rows']] if guard else val
                if len(eligible) < len(val)-k:
                    continue
                accepted = choose_light(scores, {s: keys[s] for s in val}, k, eligible)
                counts = aggregate_counts(light[s] if s in accepted else teacher[s] for s in val)
                trial = dict(model=name, admission_threshold=admission, alpha=recipe.get('alpha'), count_recipe=recipe, reference=reference,
                    guard=guard, test_fallback_images=quota, deepsolo_images=k, total_images=len(val),
                    deepsolo_percentage=100*k/len(val), counts=counts, accepted_validation_ids=accepted)
                gate_trials.append(trial)
                if quota not in winners or counts['f1'] > winners[quota]['counts']['f1']:
                    winners[quota] = trial
    write(out/'gate_trials.json', gate_trials)
    rng = np.random.default_rng(c['seed'])
    indices = rng.integers(len(val), size=(c['bootstrap_draws'], len(val)))
    fusion_grid = [dict(enabled=False, minimum_iou=None, minimum_word_probability=None, maximum_edit_distance=None)]
    fusion_grid += [dict(enabled=True, minimum_iou=iou, minimum_word_probability=p, maximum_edit_distance=d)
        for iou, p, d in itertools.product(c['fusion_minimum_ious'], c['fusion_minimum_probabilities'], c['fusion_maximum_edit_distances'])]
    complements = [dict(confidence=p, maximum_overlap=c['complement_maximum_overlap']) for p in c['complement_confidences']]
    policies, fusion_trials = {}, {}
    for quota, gate in winners.items():
        obs = observations[str(gate['admission_threshold'])]; accepted = set(gate['accepted_validation_ids'])
        baseline = {s: word_counts(obs[s]['rows'], truth[s], [ignored[s][int(r['source_row'])] for r in obs[s]['rows']]) if s in accepted else teacher[s] for s in val}
        base_matrix = np.array([[baseline[s][k] for k in ('tp','fp','fn')] for s in val])
        trial_rows = []
        for fusion, complement in itertools.product(fusion_grid, complements):
            counts, changed = [], 0
            for sid in val:
                if sid in accepted:
                    counts.append(baseline[sid]); continue
                rows = combine(obs[sid], fallback[sid]['rows'], fusion, complement)
                ff = [ignored[sid][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
                counts.append(word_counts(rows, truth[sid], ff))
                changed += sum(bool(r.get('replaced_by_light')) or bool(r.get('fixed_light_complement')) for r in rows)
            aggregate = aggregate_counts(counts)
            identity = not fusion['enabled'] and complement['confidence'] is None
            lower = 0. if identity else paired_f1_lower(base_matrix, np.array([[r[k] for k in ('tp','fp','fn')] for r in counts]), indices, c['bootstrap_alpha'])
            trial_rows.append(dict(fusion=fusion, complement=complement, counts=aggregate,
                changed_words=changed, paired_lower_gain=lower, eligible=identity or lower > 0.))
        eligible = [t for t in trial_rows if t['eligible']]
        best = max(t['counts']['f1'] for t in eligible)
        chosen = min((t for t in eligible if t['counts']['f1'] >= best-c['selection_tolerance']), key=lambda t:(t['changed_words'], -t['counts']['f1']))
        policies[str(quota)] = dict(gate, **{k:v for k,v in chosen.items() if k != 'counts'}, counts=chosen['counts'],
            gate_counts_before_fusion=gate['counts'])
        fusion_trials[str(quota)] = trial_rows
    write(out/'fusion_trials.json', fusion_trials)
    recommended = next((q for q in quotas if policies[str(q)]['counts']['f1'] > plan['target']), None)
    write(out/'selection_frozen.json', dict(status='FROZEN', mode='approximate', dataset=dataset,
        image_score='two_count_'+(c['count_recipes'][0]['kind'] if c.get('count_recipes') else 'ridge')+'_utility', policies=policies, lexical_policy=policy,
        primary_fallback_images=plan['budget'], validation_recommended_fallback_images=recommended,
        config_sha256=file_hash(config_path), lexical_frozen_sha256=file_hash(Path(c['lexical_root'])/dataset/'lexical_selection_frozen.json'),
        source_training_plan_sha256=file_hash(source/'resolved_configuration.json'),
        model_sha256={p.name:file_hash(p) for p in out.glob('*.joblib')}, test_used_for_selection=False,
        source_sha256=file_hash(__file__), map_encoder_additional_models=0, extra_fallback_reader_passes=0,
        historical_test_exposure=True, elapsed_cpu_seconds=time.monotonic()-started,
        fitting_observations=expert_condition,
        caution='Conditional paired bootstrap after candidate search; no formal generalization guarantee; see fitting_observations for decoder exclusion and upstream exposure'))
    print('FROZEN UNIFORM UTILITY', dataset, 'primary', policies[str(plan['budget'])]['counts'], 'recommended quota', recommended, flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', choices=['icdar13','icdar2015','totaltext'], required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2): train(a.config,a.dataset)
