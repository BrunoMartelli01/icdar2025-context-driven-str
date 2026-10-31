"""Refit word-risk models and recount every saved validation routing trial."""

import argparse
from collections import Counter
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import predictions
from segmentation.uniform_unet_fusion import file_hash, annotation, aggregate_counts
from recognition.common_budget_pipeline import fit_word_risk
from recognition.uniform_risk_fusion import features, observe, combine


def normalize(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def count(rows, truth, flags):
    assert len(rows) == len(flags)
    care = [r for r, skip in zip(rows, flags) if not skip]
    a = Counter(normalize(r['raw_text']) for r in care if normalize(r['raw_text']))
    b = Counter(normalize(r['text']) for r in truth if normalize(r['text']))
    tp = sum(min(n, b[word]) for word, n in a.items())
    return dict(tp=tp, fp=len(care)-tp, fn=len(truth)-tp, detected_words=len(care),
                gt_words=len(truth), ignored_outputs=sum(flags), all_outputs=len(rows))


def audit(config_path, dataset):
    config = read(config_path)
    root = Path(config['training_root'])/dataset
    plan = read(root/'resolved_configuration.json')
    out = root/'router'
    frozen = read(out/'selection_frozen.json')
    if (frozen['status'] != 'FROZEN' or frozen['test_used_for_selection'] or
            frozen['image_score'] != 'negative_expected_errors' or frozen['word_feature_count'] != 18 or
            frozen['model_sha256'] != file_hash(out/'word_model.joblib') or
            frozen['config_sha256'] != file_hash(config_path)):
        raise ValueError('Changed explicit uniform router contract')
    inference_source = Path(__file__).resolve().parents[1]/'recognition/uniform_risk_fusion.py'
    assert file_hash(inference_source) == frozen['inference_source_sha256']
    index = read(root/'export_train/index.json')
    assert file_hash(root/'export_train/index.json') == frozen['export_index_sha256']
    observed = {}
    for sid, entry in index.items():
        assert file_hash(entry['path']) == entry['sha256']
        observed[sid] = read(entry['path'])
        assert not observed[sid]['ground_truth_used']
    model = joblib.load(out/'word_model.joblib')
    full = read(root.parent/'resolved_configuration.json')
    settings = full['router']
    params = {k: settings[k] for k in ('iterations', 'leaves', 'minimum_leaf', 'l2')}
    lookup = {r['sample_id']: r for r in plan['train_records']}
    fit, val = plan['partitions']['fit'], plan['partitions']['selection']
    truths, flags, x, y = {}, {}, {}, {}
    for sid in fit+val:
        o = observed[sid]
        truths[sid], valid = annotation(lookup[sid])
        flags[sid] = [p['ignored'] for p in predictions(o['raw_rows'], lookup[sid], valid)]
        x[sid] = features(o['raw_rows'], o['rows'], o['image_wh'])
        assert x[sid].shape == (len(o['raw_rows']), 18)
        choices = {int(r['source_row']): r for r in o['rows']}
        care_words = {normalize(g['text']) for g in truths[sid] if normalize(g['text'])}
        y[sid] = np.array([int(bool(normalize(choices.get(i, r)['raw_text'])) and
            normalize(choices.get(i, r)['raw_text']) in care_words) for i, r in enumerate(o['raw_rows'])])
    folds = read(out/'word_folds.json')['folds']
    saved_probabilities = read(out/'word_probabilities.json')
    heldout = [s for f in folds for s in f['heldout_ids']]
    assert sorted(heldout) == sorted(fit)
    fitted_models = 0
    for i, (ids, held) in enumerate([(f['fit_ids'], f['heldout_ids']) for f in folds]+[(fit, val)]):
        if i < len(folds):
            assert file_hash(out/f'word_fold_{i}.joblib') == folds[i]['model_sha256']
            assert not set(ids) & set(held) and set(ids) | set(held) == set(fit)
            assert not ({lookup[s]['rgb_sha256'] for s in ids} & {lookup[s]['rgb_sha256'] for s in held})
            seed = full['shared']['seed']+i
        else:
            seed = full['shared']['seed']
        xx = np.concatenate([x[s][np.logical_not(flags[s])] for s in ids])
        yy = np.concatenate([y[s][np.logical_not(flags[s])] for s in ids])
        rebuilt = fit_word_risk(xx, yy, seed, params)
        for sid in held:
            result = rebuilt.predict_proba(x[sid])[:, 1] if len(x[sid]) else []
            np.testing.assert_allclose(result, saved_probabilities[sid], rtol=0, atol=1e-12)
        fitted_models += 1
    fallback = read(config['fallback_train_sources'][dataset])['native_benchmark']
    trials = read(out/'trials.json')
    cache, replayed, winners = {}, 0, {}
    for trial in trials:
        key = json.dumps([trial[k] for k in ['admission_threshold', 'fusion', 'complement']], sort_keys=True)
        if key not in cache:
            obs, light, heavy = {}, {}, {}
            for sid in val:
                o = observed[sid]
                obs[sid] = observe(o['raw_rows'], o['rows'], o['image_wh'], model, trial['admission_threshold'])
                kept = {int(r['source_row']) for r in obs[sid]['rows']}
                expected_score = -sum((1-p) if i in kept else p for i, p in enumerate(obs[sid]['probabilities']))
                assert abs(obs[sid]['score']-expected_score) < 1e-12
                light[sid] = count(obs[sid]['rows'], truths[sid], [flags[sid][int(r['source_row'])] for r in obs[sid]['rows']])
                rows = combine(obs[sid], fallback[sid]['rows'], trial['fusion'], trial['complement'])
                ff = [flags[sid][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
                heavy[sid] = count(rows, truths[sid], ff)
            cache[key] = obs, light, heavy
        obs, light, heavy = cache[key]
        quota = trial['test_fallback_images']
        k = round(len(val)*quota/len(plan['partitions']['test']))
        eligible = [s for s in val if obs[s]['rows']] if trial['guard'] else val
        accepted = set(sorted(eligible, key=lambda s: (-obs[s]['score'], lookup[s]['rgb_sha256'], s))[:len(val)-k])
        assert len(accepted) == len(val)-k
        aggregate = aggregate_counts(light[s] if s in accepted else heavy[s] for s in val)
        assert aggregate == trial['counts']
        assert trial['deepsolo_images'] == k and trial['total_images'] == len(val)
        replayed += len(val)
        if quota not in winners or aggregate['f1'] > winners[quota]['counts']['f1']:
            winners[quota] = dict(trial, accepted_validation_ids=sorted(accepted))
    for quota, selected in frozen['policies'].items():
        rebuilt = winners[int(quota)]
        assert set(rebuilt['accepted_validation_ids']) == set(selected['accepted_validation_ids'])
        assert {k: v for k, v in rebuilt.items() if k != 'accepted_validation_ids'} == {
               k: v for k, v in selected.items() if k != 'accepted_validation_ids'}
    write(out/'independent_router_audit.json', dict(status='PASS', word_model_refits=fitted_models,
        validation_trial_image_replays=replayed, selected_quotas=len(winners),
        model_sha256=frozen['model_sha256'], frozen_selection_sha256=file_hash(out/'selection_frozen.json'),
        source_sha256=file_hash(__file__), test_labels_used=False,
        scope='Independent Counter recount and score/quota ordering; declared geometry/feature primitives shared'))
    print('ROUTER AUDIT PASS', dataset, fitted_models, 'refits', replayed, 'trial-image replays', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', choices=['icdar13', 'icdar2015', 'totaltext'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        audit(args.config, args.dataset)
