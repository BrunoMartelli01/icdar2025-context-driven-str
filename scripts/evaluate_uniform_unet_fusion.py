"""Freeze annotation-free TEST routes, then evaluate all predeclared budgets."""

import argparse
from pathlib import Path
import time

import joblib
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import key, predictions
from segmentation.uniform_unet_fusion import file_hash, annotation, word_counts, aggregate_counts
from recognition.uniform_risk_fusion import observe, route, combine


def fallback_rows(config, dataset, sid, archive=None):
    source = config['fallback_test_sources'][dataset]
    if source['kind'] == 'totaltext_branches':
        # A single authenticated archive contains this dataset's complete cache.
        value = archive['records'][sid]
        rows = value['deepsolo']
        expected = None
        digest = config['pinned_inputs'][source['path']]
    elif source['kind'] == 'icdar2015_index':
        entry = archive['samples'][sid]
        path = Path(entry['evidence_path'])
        if file_hash(path) != entry['evidence_sha256']:
            raise ValueError('Changed qualified ICDAR2015 fallback envelope')
        value = read(path)
        if file_hash(value['raw_path']) != value['raw_sha256']:
            raise ValueError('Changed raw DeepSolo neural response')
        rows, expected, digest = value['predictions'], value['counts'], entry['evidence_sha256']
    elif source['kind'] == 'icdar13_files':
        path = Path(source['directory'])/(key(sid)+'.json')
        digest = config['pinned_inputs'][str(path)]
        if file_hash(path) != digest:
            raise ValueError('Changed qualified ICDAR2013 fallback envelope')
        value = read(path)
        if file_hash(value['raw_path']) != value['raw_sha256']:
            raise ValueError('Changed raw DeepSolo neural response')
        rows, expected = value['predictions'], value['counts']
    else:
        raise ValueError('Unknown explicitly configured fallback source')
    rows = [dict(r, raw_text='' if r.get('invalid_envelope') is not None else r['text'], word_source='deepsolo') for r in rows]
    return rows, expected, digest


def evaluate(config_path, dataset):
    started = time.monotonic()
    config = read(config_path)
    if config['mode'] != 'approximate' or config_path.resolve().parent.name != 'approximate':
        raise ValueError('Explicit approximate evaluation profile required')
    root = Path(config['training_root'])/dataset
    plan = read(root/'resolved_configuration.json')
    gate = read(root/'router/selection_frozen.json')
    audit = read(root/'router/independent_router_audit.json')
    index = read(root/'export_test/index.json')
    contract = read(root/'export_test/contract.json')
    if (gate['status'] != 'FROZEN' or gate['test_used_for_selection'] or audit['status'] != 'PASS' or
            audit['frozen_selection_sha256'] != file_hash(root/'router/selection_frozen.json') or
            contract['frozen_router_sha256'] != file_hash(root/'router/selection_frozen.json') or
            read(root/'export_test/complete.json')['status'] != 'COMPLETE'):
        raise ValueError('Qualified frozen gate and complete bound TEST export required')
    ids = plan['partitions']['test']
    if set(ids) != set(index) or len(ids) != config['expected_images'][dataset]:
        raise ValueError('Complete official TEST population required')
    out = root/'evaluation'
    out.mkdir(exist_ok=False)
    model = joblib.load(root/'router/word_model.joblib')
    if file_hash(root/'router/word_model.joblib') != gate['model_sha256']:
        raise ValueError('Changed deployed word model')
    samples = {}
    for sid in ids:
        entry = index[sid]
        if file_hash(entry['path']) != entry['sha256']:
            raise ValueError('Changed frontend observation')
        samples[sid] = read(entry['path'])
        if samples[sid]['ground_truth_used']:
            raise ValueError('TEST frontend consumed annotations')
    observations, decisions = {}, {}
    for quota, policy in gate['policies'].items():
        observed = {s: observe(samples[s]['raw_rows'], samples[s]['rows'], samples[s]['image_wh'],
                    model, policy['admission_threshold']) for s in ids}
        decisions[quota] = route(observed, {s: samples[s]['rgb_sha256'] for s in ids}, int(quota), policy['guard'])
        observations[quota] = observed
    write(out/'observations.json', observations)
    write(out/'decisions_before_fallback.json', dict(status='FROZEN', decisions=decisions,
        observations_sha256=file_hash(out/'observations.json'), router_sha256=file_hash(root/'router/selection_frozen.json'),
        export_index_sha256=file_hash(root/'export_test/index.json'), ground_truth_used=False, fallback_outputs_loaded=False))
    # Only after all quota plans are persisted may annotation/fallback data enter.
    for path, digest in config['pinned_inputs'].items():
        if file_hash(path) != digest:
            raise ValueError('Changed qualified evaluation input: '+path)
    lookup = {r['sample_id']: r for r in plan['test_records']}
    truths, flags, fallbacks, fallback_counts, evidence = {}, {}, {}, {}, {}
    source = config['fallback_test_sources'][dataset]
    archive = read(source['path']) if source['kind'] != 'icdar13_files' else None
    for sid in ids:
        truths[sid], validity = annotation(lookup[sid])
        flags[sid] = [p['ignored'] for p in predictions(samples[sid]['raw_rows'], lookup[sid], validity)]
        rows, expected, digest = fallback_rows(config, dataset, sid, archive)
        count = word_counts(rows, truths[sid], [r['ignored'] for r in rows])
        if expected is not None and count != expected:
            raise ValueError('Fallback and declared metric/care boundary disagree: '+sid)
        fallbacks[sid], fallback_counts[sid], evidence[sid] = rows, count, digest
    control = aggregate_counts(fallback_counts.values())
    light_control = aggregate_counts(word_counts(samples[s]['rows'], truths[s],
        [flags[s][int(r['source_row'])] for r in samples[s]['rows']]) for s in ids)
    if control['gt_words'] != config['expected_care_words'][dataset]:
        raise ValueError('Full care-word denominator differs')
    per_image, results = {}, {}
    for quota, policy in gate['policies'].items():
        accepted = set(decisions[quota]['accepted_ids'])
        per_image[quota] = {}
        for sid in ids:
            o = observations[quota][sid]
            if sid in accepted:
                rows = o['rows']; ignored = [flags[sid][int(r['source_row'])] for r in rows]
            else:
                rows = combine(o, fallbacks[sid], policy['fusion'], policy['complement'])
                ignored = [flags[sid][int(r['source_row'])] if r.get('fixed_light_complement') else r['ignored'] for r in rows]
            measured = word_counts(rows, truths[sid], ignored)
            per_image[quota][sid] = dict(rows=rows, ignored=ignored, counts=measured,
                ground_truth=truths[sid], action='light' if sid in accepted else 'deepsolo', score=o['score'])
        counts = aggregate_counts(v['counts'] for v in per_image[quota].values())
        if counts['images'] != len(ids) or counts['gt_words'] != config['expected_care_words'][dataset]:
            raise ValueError('A TEST sample/care word was omitted')
        results[quota] = dict(counts=counts, deepsolo_images=int(quota), total_images=len(ids),
            deepsolo_percentage=100*int(quota)/len(ids), target=plan['target'], target_met=counts['f1'] > plan['target'],
            policy_selected_on_validation=True, actual_new_deepsolo_neural_calls=0,
            scope='Logical replay of authenticated cached fallbacks; no online latency measurement')
    write(out/'per_image.json', per_image)
    write(out/'fallback_evidence_sha256.json', evidence)
    write(out/'results.json', dict(status='COMPLETE', mode='approximate', claim_label='approximate reproduction',
        dataset=dataset, primary_quota=plan['budget'], validation_recommended_quota=gate['validation_recommended_fallback_images'],
        primary=results[str(plan['budget'])], results=results,
        all_deepsolo=dict(counts=control, deepsolo_images=len(ids), total_images=len(ids), deepsolo_percentage=100.),
        all_light=dict(counts=light_control, deepsolo_images=0, total_images=len(ids), deepsolo_percentage=0.),
        historical_test_exposure=True, test_used_for_selection=False, additional_neural_models=0,
        actual_new_deepsolo_neural_calls=0, seconds_cpu_evaluation=time.monotonic()-started,
        config_sha256=file_hash(config_path), source_sha256=file_hash(__file__),
        caveat='Other predeclared quotas are descriptive TEST evaluations, not TEST-selected deployment changes'))
    print('UNIFORM FULL TEST', dataset, 'primary', results[str(plan['budget'])], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--dataset', choices=['icdar13', 'icdar2015', 'totaltext'], required=True)
    args = p.parse_args()
    with threadpool_limits(limits=2):
        evaluate(args.config, args.dataset)
