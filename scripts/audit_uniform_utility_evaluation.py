"""Recount full TEST utility/fusion results and independently rank all images."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import key, predictions
from scripts.audit_uniform_risk_fusion import count
from scripts.evaluate_uniform_unet_fusion import fallback_rows
from segmentation.uniform_unet_fusion import annotation, file_hash, aggregate_counts
from recognition.uniform_lexical_risk import lexicon
from recognition.uniform_utility_fusion import score
from recognition.uniform_risk_fusion import combine
from recognition.uniform_encoder_context import training_context
from recognition.uniform_word_union import add_union_rows, combine as combine_union


def audit(config_path, evaluation_config, dataset):
    c, e = read(config_path), read(evaluation_config)
    source = Path(c['training_root'])/dataset
    root = Path(c['output_root'])/dataset
    plan, frozen = read(source/'resolved_configuration.json'), read(root/'selection_frozen.json')
    qualified = read(root/'independent_audit.json')
    assert qualified['status'] == 'PASS' and qualified['frozen_sha256'] == file_hash(root/'selection_frozen.json')
    assert frozen['config_sha256'] == file_hash(config_path) and not frozen['test_used_for_selection']
    for path, digest in {**c['pinned_inputs'], **e['pinned_inputs']}.items():
        assert file_hash(path) == digest, path
    for name, digest in frozen['model_sha256'].items():
        assert file_hash(root/name) == digest, name
    out = root/'evaluation'
    result, decision = read(out/'results.json'), read(out/'decisions_before_fallback.json')
    observations, outputs = read(out/'observations.json'), read(out/'per_image.json')
    assert result['status'] == 'COMPLETE' and result['mode'] == 'approximate'
    assert result['config_sha256'] == file_hash(config_path)
    assert result['evaluation_config_sha256'] == file_hash(evaluation_config)
    assert decision['status'] == 'FROZEN' and not decision['ground_truth_used'] and not decision['fallback_outputs_loaded']
    assert decision['observations_sha256'] == file_hash(out/'observations.json')
    assert decision['frozen_sha256'] == file_hash(root/'selection_frozen.json')
    assert decision['export_index_sha256'] == file_hash(source/'export_test/index.json')
    index, ids = read(source/'export_test/index.json'), plan['partitions']['test']
    assert len(ids) == len(set(ids)) == e['expected_images'][dataset] and set(index) == set(ids)
    assert set(frozen['policies']) == set(observations) == set(outputs) == set(result['results']) == set(decision['decisions'])
    samples, maps = {}, {}
    for sid, entry in index.items():
        assert file_hash(entry['path']) == entry['sha256']
        samples[sid] = read(entry['path'])
        assert not samples[sid]['ground_truth_used']
        path = source/'export_test/maps'/(key(sid)+'.npz')
        assert file_hash(path) == samples[sid]['maps_sha256']
        with np.load(path, allow_pickle=False) as z:
            maps[sid] = z['probabilities']
    dictionary = lexicon(c['lexicons'][dataset], 'test')
    context = training_context(c, {s:dict(rgb_sha256=samples[s]['rgb_sha256']) for s in ids})
    model = joblib.load(root/'word_model.joblib')
    union_model=joblib.load(root/'union_model.joblib') if 'learned_word_union' in c else None
    lookup = {r['sample_id']: r for r in plan['test_records']}
    truth, flags, heavy, controls = {}, {}, {}, {}
    spec = e['fallback_test_sources'][dataset]
    archive = read(spec['path']) if spec['kind'] != 'icdar13_files' else None
    for sid in ids:
        assert samples[sid]['rgb_sha256'] == lookup[sid]['rgb_sha256']
        truth[sid], valid = annotation(lookup[sid])
        flags[sid] = [r['ignored'] for r in predictions(samples[sid]['raw_rows'], lookup[sid], valid)]
        heavy[sid], expected, _ = fallback_rows(e, dataset, sid, archive)
        controls[sid] = count(heavy[sid], truth[sid], [r['ignored'] for r in heavy[sid]])
        assert expected is None or controls[sid] == expected
    assert aggregate_counts(controls.values()) == result['all_deepsolo']['counts']
    replayed = 0
    for quota, policy in frozen['policies'].items():
        head = joblib.load(root/policy['model'])
        recalculated = {}
        assert set(observations[quota]) == set(outputs[quota]) == set(ids)
        for sid in ids:
            s = samples[sid]
            o = score(s['raw_rows'], s['image_wh'], maps[sid], dictionary,
                frozen['lexical_policy'], model, head, policy['admission_threshold'],
                policy['reference'], c['map_settings'], context[sid] if context is not None else None)
            if union_model is not None:
                o=add_union_rows(o,s['raw_rows'],dictionary,frozen['lexical_policy']);o['image_wh']=s['image_wh']
            recalculated[sid] = o
            saved = observations[quota][sid]
            assert saved['rows'] == o['rows'] and saved['trace'] == o['trace']
            np.testing.assert_allclose(saved['probabilities'], o['probabilities'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(saved['score'], o['score'], rtol=0, atol=1e-12)
        eligible = [s for s in ids if recalculated[s]['rows']] if policy['guard'] else ids
        accepted = set(sorted(eligible, key=lambda s: (-recalculated[s]['score'], samples[s]['rgb_sha256'], s))[:len(ids)-int(quota)])
        d = decision['decisions'][quota]
        assert len(accepted) == len(ids)-int(quota)
        assert set(d['accepted_ids']) == accepted and set(d['fallback_ids']) == set(ids)-accepted
        assert len(d['accepted_ids'])+len(d['fallback_ids']) == len(ids)
        assert d['deepsolo_images'] == int(quota) and d['total_images'] == len(ids)
        assert abs(d['deepsolo_percentage']-100*int(quota)/len(ids)) < 1e-12
        counts = []
        for sid in ids:
            o = recalculated[sid]
            if sid in accepted:
                rows = o['rows']; ignored = [flags[sid][int(r['source_row'])] for r in rows]
            else:
                rows = combine_union(o,heavy[sid],samples[sid]['image_wh'],union_model,policy,c['learned_word_union']) if union_model is not None and policy['learned_union_threshold'] is not None else combine(o, heavy[sid], policy['fusion'], policy['complement'])
                ignored = [flags[sid][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
            saved = outputs[quota][sid]
            assert rows == saved['rows'] and ignored == saved['ignored'] and truth[sid] == saved['ground_truth']
            assert saved['action'] == ('light' if sid in accepted else 'deepsolo')
            measured = count(rows, truth[sid], ignored)
            assert measured == saved['counts']
            counts.append(measured); replayed += 1
        measured = aggregate_counts(counts); reported = result['results'][quota]
        assert measured == reported['counts'] and measured['gt_words'] == e['expected_care_words'][dataset]
        assert reported['target_met'] == (measured['f1'] > plan['target'])
        assert reported['deepsolo_images'] == int(quota) and reported['total_images'] == len(ids)
        assert abs(reported['deepsolo_percentage']-100*int(quota)/len(ids)) < 1e-12
    assert result['primary'] == result['results'][str(plan['budget'])]
    write(out/'independent_audit.json', dict(status='PASS', full_test_images=len(ids),
        care_words=e['expected_care_words'][dataset], policy_image_replays=replayed,
        quotas=len(frozen['policies']), result_sha256=file_hash(out/'results.json'),
        router_sha256=file_hash(root/'selection_frozen.json'), source_sha256=file_hash(__file__),
        scope='Independent Counter metric, independent score ranking and strict source hashes; shared declared feature and geometry primitives'))
    print('UTILITY TEST AUDIT PASS', dataset, replayed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--evaluation-config', type=Path, required=True)
    p.add_argument('--dataset', choices=['icdar13', 'icdar2015', 'totaltext'], required=True)
    a = p.parse_args()
    with threadpool_limits(limits=2):
        audit(a.config, a.evaluation_config, a.dataset)
