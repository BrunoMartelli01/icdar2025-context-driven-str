"""Independently replay every full-TEST uniform gate and word-count result."""

import argparse
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import predictions
from scripts.audit_uniform_risk_fusion import count
from scripts.evaluate_uniform_unet_fusion import fallback_rows
from segmentation.uniform_unet_fusion import annotation, file_hash, aggregate_counts
from recognition.uniform_risk_fusion import observe, combine


def audit(config_path, dataset):
    config = read(config_path)
    root = Path(config['training_root'])/dataset
    plan = read(root/'resolved_configuration.json')
    gate = read(root/'router/selection_frozen.json')
    out = root/'evaluation'
    result = read(out/'results.json')
    decision = read(out/'decisions_before_fallback.json')
    observations = read(out/'observations.json')
    outputs = read(out/'per_image.json')
    index = read(root/'export_test/index.json')
    assert result['config_sha256'] == file_hash(config_path)
    assert decision['status'] == 'FROZEN' and not decision['ground_truth_used'] and not decision['fallback_outputs_loaded']
    assert decision['observations_sha256'] == file_hash(out/'observations.json')
    assert decision['router_sha256'] == file_hash(root/'router/selection_frozen.json')
    assert decision['export_index_sha256'] == file_hash(root/'export_test/index.json')
    model = joblib.load(root/'router/word_model.joblib')
    assert file_hash(root/'router/word_model.joblib') == gate['model_sha256']
    samples = {}
    for sid, entry in index.items():
        assert file_hash(entry['path']) == entry['sha256']
        samples[sid] = read(entry['path'])
        assert not samples[sid]['ground_truth_used']
    ids = plan['partitions']['test']
    assert len(ids) == config['expected_images'][dataset] and set(ids) == set(samples)
    lookup = {r['sample_id']: r for r in plan['test_records']}
    truths, flags, heavy, controls = {}, {}, {}, {}
    source = config['fallback_test_sources'][dataset]
    archive = read(source['path']) if source['kind'] != 'icdar13_files' else None
    for sid in ids:
        truths[sid], validity = annotation(lookup[sid])
        flags[sid] = [p['ignored'] for p in predictions(samples[sid]['raw_rows'], lookup[sid], validity)]
        heavy[sid], expected, _ = fallback_rows(config, dataset, sid, archive)
        controls[sid] = count(heavy[sid], truths[sid], [r['ignored'] for r in heavy[sid]])
        assert expected is None or controls[sid] == expected
    assert aggregate_counts(controls.values()) == result['all_deepsolo']['counts']
    replayed = 0
    for quota, policy in gate['policies'].items():
        recalculated = {}
        for sid in ids:
            s = samples[sid]
            recalculated[sid] = observe(s['raw_rows'], s['rows'], s['image_wh'], model, policy['admission_threshold'])
            saved = observations[quota][sid]
            assert saved['rows'] == recalculated[sid]['rows']
            np.testing.assert_allclose(saved['score'], recalculated[sid]['score'], atol=1e-12, rtol=0)
        eligible = [s for s in ids if recalculated[s]['rows']] if policy['guard'] else ids
        accepted = set(sorted(eligible, key=lambda s: (-recalculated[s]['score'], samples[s]['rgb_sha256'], s))[:len(ids)-int(quota)])
        d = decision['decisions'][quota]
        assert set(d['accepted_ids']) == accepted and set(d['fallback_ids']) == set(ids)-accepted
        assert d['deepsolo_images'] == int(quota) and d['total_images'] == len(ids)
        counts = []
        for sid in ids:
            o = recalculated[sid]
            if sid in accepted:
                rows = o['rows']; ignored = [flags[sid][int(r['source_row'])] for r in rows]
            else:
                rows = combine(o, heavy[sid], policy['fusion'], policy['complement'])
                ignored = [flags[sid][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
            saved = outputs[quota][sid]
            assert rows == saved['rows'] and ignored == saved['ignored'] and truths[sid] == saved['ground_truth']
            measured = count(rows, truths[sid], ignored)
            assert measured == saved['counts']
            counts.append(measured); replayed += 1
        aggregate = aggregate_counts(counts)
        assert aggregate == result['results'][quota]['counts']
        assert aggregate['gt_words'] == config['expected_care_words'][dataset]
        assert result['results'][quota]['target_met'] == (aggregate['f1'] > plan['target'])
    write(out/'independent_audit.json', dict(status='PASS', full_test_images=len(ids),
        care_words=config['expected_care_words'][dataset], policy_image_replays=replayed,
        quotas=len(gate['policies']), result_sha256=file_hash(out/'results.json'),
        router_sha256=file_hash(root/'router/selection_frozen.json'), source_sha256=file_hash(__file__),
        scope='Independent Counter metric and independent ranking; shared declared feature and geometry primitives'))
    print('FULL TEST AUDIT PASS', dataset, replayed, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', choices=['icdar13', 'icdar2015', 'totaltext'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        audit(args.config, args.dataset)
