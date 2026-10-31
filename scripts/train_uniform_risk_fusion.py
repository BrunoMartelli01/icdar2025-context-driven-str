"""Fit one identical word-risk/expected-error/fusion procedure per dataset."""

import argparse
import itertools
import math
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import predictions
from segmentation.uniform_unet_fusion import file_hash, annotation, word_counts, aggregate_counts
from recognition.common_budget_pipeline import grouped_folds, fit_word_risk
from recognition.fixed_expert_budget import training_word_labels
from recognition.uniform_risk_fusion import features, observe, route, combine


def train(config_path, dataset):
    started = time.monotonic()
    config = read(config_path)
    if config['mode'] != 'approximate' or config_path.resolve().parent.name != 'approximate':
        raise ValueError('Explicit approximate router profile required')
    for path, digest in config['pinned_inputs'].items():
        if file_hash(path) != digest:
            raise ValueError('Changed pinned fallback source: '+path)
    root = Path(config['training_root'])/dataset
    plan = read(root/'resolved_configuration.json')
    audit = read(root/'independent_training_audit.json')
    export = root/'export_train'
    contract = read(export/'contract.json')
    if (audit['status'] != 'PASS' or read(export/'complete.json')['status'] != 'COMPLETE' or
            contract['plan_sha256'] != file_hash(root/'resolved_configuration.json') or
            contract['policy_sha256'] != file_hash(root/'emission_selection/frozen.json') or
            contract['checkpoint_sha256'] != audit['best_checkpoint_sha256']):
        raise ValueError('Audited training and qualified complete TRAIN export required')
    out = root/'router'
    out.mkdir(exist_ok=False)
    shared = read(root.parent/'resolved_configuration.json')
    settings = shared['router']
    fit, val = plan['partitions']['fit'], plan['partitions']['selection']
    lookup = {r['sample_id']: r for r in plan['train_records']}
    index = read(export/'index.json')
    if set(index) != set(fit+val):
        raise ValueError('Incomplete TRAIN export population')
    observations, labels, flags, x, y, keys = {}, {}, {}, {}, {}, {}
    pins = {}
    for sid in fit+val:
        item = index[sid]
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Changed exported sample')
        o = read(item['path'])
        if o['ground_truth_used'] or o['image_sha256'] != lookup[sid]['image_sha256']:
            raise ValueError('Changed RGB/export boundary')
        observations[sid] = o
        labels[sid], validity = annotation(lookup[sid])
        flags[sid] = [p['ignored'] for p in predictions(o['raw_rows'], lookup[sid], validity)]
        x[sid] = features(o['raw_rows'], o['rows'], o['image_wh'])
        y[sid] = training_word_labels(o['raw_rows'], o['rows'], labels[sid])
        keys[sid] = lookup[sid]['rgb_sha256']
        pins[item['path']] = item['sha256']
    folds = grouped_folds(fit, keys, settings['folds'], shared['shared']['seed'])
    word_settings = {k: settings[k] for k in ('iterations', 'leaves', 'minimum_leaf', 'l2')}
    def fit_model(ids, seed):
        care = {s: np.logical_not(flags[s]) for s in ids}
        xx = np.concatenate([x[s][care[s]] for s in ids])
        yy = np.concatenate([y[s][care[s]] for s in ids])
        return fit_word_risk(xx, yy, seed, word_settings)
    probabilities, fold_records = {}, []
    for i, held in enumerate(folds):
        remaining = sorted(set(fit)-set(held))
        model = fit_model(remaining, shared['shared']['seed']+i)
        path = out/f'word_fold_{i}.joblib'
        joblib.dump(model, path)
        for s in held:
            probabilities[s] = model.predict_proba(x[s])[:, 1].tolist() if len(x[s]) else []
        fold_records.append(dict(fit_ids=remaining, heldout_ids=held, model_sha256=file_hash(path)))
    model = fit_model(fit, shared['shared']['seed'])
    joblib.dump(model, out/'word_model.joblib')
    for s in val:
        probabilities[s] = model.predict_proba(x[s])[:, 1].tolist() if len(x[s]) else []
    write(out/'word_folds.json', dict(folds=fold_records, upstream_expert_image_excluded=False,
        scope='Word models exclude heldout RGB groups; source U-Net and pretraining exposure remains'))
    write(out/'word_probabilities.json', probabilities)
    fallback = read(config['fallback_train_sources'][dataset])['native_benchmark']
    for s in fit+val:
        source = fallback[s]
        recount = word_counts(source['rows'], labels[s], [r['ignored'] for r in source['rows']])
        if recount != source['counts']:
            raise ValueError('Qualified fallback/care boundary differs: '+s)
    write(out/'input_sha256.json', pins)
    fusion_grid = [dict(enabled=False, minimum_iou=None, minimum_word_probability=None, maximum_edit_distance=None)]
    fusion_grid += [dict(enabled=True, minimum_iou=iou, minimum_word_probability=p, maximum_edit_distance=d)
        for iou, p, d in itertools.product(settings['fusion_minimum_ious'],
             settings['fusion_minimum_probabilities'], settings['fusion_maximum_edit_distances'])]
    complements = [dict(confidence=p, maximum_overlap=settings['complement_maximum_overlap'])
                   for p in settings['complement_confidences']]
    total = len(plan['partitions']['test'])
    quotas = sorted(set([plan['budget']]+[math.ceil(f*total) for f in shared['alternate_fallback_fractions']
                                         if math.ceil(f*total) >= plan['budget']]))
    # Export/classifier inputs do not read fallbacks. All fallback accesses above
    # and below are TRAIN supervision/validation of the post-routing policy.
    winners, trials = {}, []
    for admission in settings['admission_thresholds']:
        obs = {s: observe(observations[s]['raw_rows'], observations[s]['rows'],
               observations[s]['image_wh'], model, admission) for s in val}
        light = {s: word_counts(obs[s]['rows'], labels[s],
                  [flags[s][int(r['source_row'])] for r in obs[s]['rows']]) for s in val}
        for fusion, complement in itertools.product(fusion_grid, complements):
            heavy = {}
            for s in val:
                rows = combine(obs[s], fallback[s]['rows'], fusion, complement)
                ignored = [flags[s][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
                heavy[s] = word_counts(rows, labels[s], ignored)
            for quota, guard in itertools.product(quotas, config['guards']):
                scaled = round(len(val)*quota/total)
                if guard and sum(bool(o['rows']) for o in obs.values()) < len(val)-scaled:
                    continue
                decision = route(obs, {s: keys[s] for s in val}, scaled, guard)
                accepted = set(decision['accepted_ids'])
                counts = aggregate_counts(light[s] if s in accepted else heavy[s] for s in val)
                trial = dict(admission_threshold=admission, fusion=fusion, complement=complement,
                    guard=guard, test_fallback_images=quota, counts=counts,
                    deepsolo_images=scaled, total_images=len(val), deepsolo_percentage=100*scaled/len(val))
                trials.append(trial)
                if quota not in winners or counts['f1'] > winners[quota]['counts']['f1']:
                    winners[quota] = dict(trial, accepted_validation_ids=decision['accepted_ids'])
        write(out/'trials.json', trials)
    recommended = next((q for q in quotas if winners[q]['counts']['f1'] > plan['target']), None)
    write(out/'selection_frozen.json', dict(status='FROZEN', mode='approximate', dataset=dataset,
        image_score='negative_expected_errors', word_feature_count=18, word_model_kind='HistGradientBoostingClassifier',
        primary_fallback_images=plan['budget'], validation_recommended_fallback_images=recommended,
        policies={str(k): v for k, v in winners.items()}, test_used_for_selection=False,
        model_sha256=file_hash(out/'word_model.joblib'), model_bytes=(out/'word_model.joblib').stat().st_size,
        training_plan_sha256=file_hash(root/'resolved_configuration.json'),
        emission_policy_sha256=file_hash(root/'emission_selection/frozen.json'),
        export_index_sha256=file_hash(export/'index.json'), config_sha256=file_hash(config_path),
        source_sha256=file_hash(__file__), inference_source_sha256=file_hash(Path(__file__).resolve().parents[1]/'recognition/uniform_risk_fusion.py'),
        historical_test_exposure=True, additional_neural_models=0, extra_fallback_reader_passes=0,
        cpu_fit_and_selection_seconds=time.monotonic()-started,
        validation_selection_rule='Highest validation F1, earliest declared grid tie, separately per predeclared quota'))
    print('FROZEN UNIFORM ROUTER', dataset, 'primary validation', winners[plan['budget']]['counts'],
          'recommended quota', recommended, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', choices=['icdar13', 'icdar2015', 'totaltext'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        train(args.config, args.dataset)
