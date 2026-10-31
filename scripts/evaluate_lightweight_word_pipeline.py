"""Evaluate one validation-frozen lightweight pipeline with exactly61 fallbacks."""

import argparse
from collections import Counter
from pathlib import Path
import time

from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import counts, micro
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lightweight_word_deployment import LightweightWordPipeline
from scripts.icdar_budget_data import read, sha, annotation, predictions
from scripts.run_lightweight_word_selection import load_source
from scripts.train_lightweight_word_router import teacher
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def independent(rows, truth, ignored):
    normalize = lambda s: ''.join(c for c in s.casefold() if c.isalnum())
    remaining = [normalize(g['text']) for g in truth]; tp = n = 0
    for row, skip in zip(rows, ignored):
        if skip:
            continue
        n += 1; word = normalize(row['raw_text'])
        if word and word in remaining:
            remaining.remove(word); tp += 1
    return dict(tp=tp, fp=n-tp, fn=len(truth)-tp, detected_words=n,
        gt_words=len(truth), all_outputs=len(rows), ignored_outputs=sum(ignored))


def evaluate(config_path):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate evaluation configuration required')
    config = read(config_path)
    if config['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for path, digest in config['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed frozen evaluation input: '+path)
    root = Path(config['output_root']); root.mkdir(parents=True, exist_ok=False)
    router = Path(config['router_root']); plan = read(router/'resolved_configuration.json')
    frozen = read(router/'pipeline_frozen.json'); a = plan['admission_plan']
    if bool(a['smoke']) != bool(config['smoke']):
        raise ValueError('Smoke/full result classes must agree')
    if frozen['choice']['winner']['result']['combined']['f1'] < config['minimum_validation_f1']:
        raise ValueError('Declared validation qualification required before TEST')
    pipeline = LightweightWordPipeline(router)
    profile = read(config['fallback_profile'])
    dictionary = DeepSoloLexicon.from_files(profile['lexicon_paths']['vocabulary'],
        profile['lexicon_paths']['pairs'], distance_exclusive=profile['distance_exclusive'])
    ids = a['partitions']['test']; expected = config['total_images']
    if len(ids) != expected:
        raise ValueError('Complete TEST population required')
    seal(root/'resolved_configuration.json', dict(mode='approximate',
        claim_label='approximate reproduction', config=config,
        config_sha256=sha(config_path), producer_sha256=sha(__file__),
        selected_arm=pipeline.arm, frozen_choice_sha256=sha(router/'pipeline_frozen.json'),
        result_class='smoke-test result' if a['smoke'] else 'approximate reproduction',
        annotation_free_deployment_sha256=sha(REPO/'recognition/lightweight_word_deployment.py'),
        test_used_for_selection=False, historical_test_exposure=True,
        actual_new_neural_inferences=0))
    observations = {}; input_pins = {}; started = time.monotonic()
    for sid in ids:
        record = a['records'][sid]; sources = {}
        for name in pipeline.sources:
            rows, path, digest = load_source(a['config'], record, name, 'test')
            input_pins[path] = digest; sources[name] = rows
        # No annotation, validity or DeepSolo output has been loaded here.
        observations[sid] = pipeline.observe(sources, record['image_wh'], dictionary)
    decision = pipeline.route(observations, {s: a['records'][s]['rgb_sha256'] for s in ids},
                              fallback_images=config['fallback_images'])
    seal(root/'decisions_before_fallback.json', dict(decision,
        scores={s: v['score'] for s, v in observations.items()},
        ground_truth_used=False, fallback_outputs_loaded=False,
        seconds_cpu_cached_scoring_only=time.monotonic()-started))
    accepted = set(decision['accepted_ids']); results = {}; all_light = []
    fallback_config = dict(fallback_profile=config['fallback_profile'],
        train_protocol=config['test_protocol'], raw_train_root=config['raw_test_root'])
    for sid in ids:
        record = a['records'][sid]; obs = observations[sid]
        gt, valid = annotation(record)
        flags = [p['ignored'] for p in predictions(obs['rows'], record, valid)]
        all_light.append(counts(obs['rows'], gt, flags))
        if sid in accepted:
            rows = obs['rows']; source_digest = None
        else:
            fallback = teacher(fallback_config, a, record, dictionary)
            input_pins[fallback['source_path']] = fallback['source_sha256']
            rows = pipeline.combine_after_fallback(obs, fallback['rows'])
            flags = [r['ignored'] if r['word_source'] == 'deepsolo' else
                predictions([r], record, valid)[0]['ignored'] for r in rows]
            source_digest = fallback['source_sha256']
        measured = counts(rows, gt, flags)
        if measured != independent(rows, gt, flags):
            raise ValueError('Independent TEST word recount differs')
        results[sid] = dict(action='light' if sid in accepted else 'deepsolo',
            rows=rows, ignored=flags, counts=measured, ground_truth=gt,
            score=obs['score'], fallback_sha256=source_digest,
            light_rows=obs['rows'], admission_probabilities=obs['probabilities'])
    aggregate = micro(v['counts'] for v in results.values())
    if aggregate['images'] != expected or aggregate['gt_words'] != config['care_words']:
        raise ValueError('Complete care/image denominator changed')
    for path, digest in input_pins.items():
        if sha(path) != digest:
            raise ValueError('Scientific input changed during TEST')
    if len([r for r in results.values() if r['action'] == 'deepsolo']) != config['fallback_images']:
        raise ValueError('Fallback count changed after pre-fallback decisions')
    seal(root/'per_image.json', results); seal(root/'input_sha256.json', input_pins)
    controls = []
    for path in config['protected_qualifications']:
        q = read(path)
        run = Path(path).parent
        if q['status'] != 'PASS' or sha(run/q['audit_path']) != q['audit_sha256']:
            raise ValueError('Changed protected dataset qualification')
        for rel, digest in q['artifact_sha256'].items():
            if sha(run/rel) != digest:
                raise ValueError('Changed protected dataset model')
        controls.append(dict(qualification=path, sha256=sha(path),
            scope='Previously qualified TT/IC15 profiles preserved; no new method applied to them in this IC13 research test'))
    result = dict(status='COMPLETE', mode='approximate', claim_label='approximate reproduction',
        result_class='smoke-test result' if a['smoke'] else 'approximate reproduction',
        selected_arm=pipeline.arm, counts=aggregate, target_met=aggregate['f1'] >= config['minimum_f1'],
        minimum_f1=config['minimum_f1'], deepsolo_images=config['fallback_images'],
        total_images=expected, deepsolo_percentage=100*config['fallback_images']/expected,
        all_light=dict(counts=micro(all_light), deepsolo_images=0, total_images=expected,
                       deepsolo_percentage=0.),
        actual_new_deepsolo_neural_inferences=0, actual_new_neural_inferences=0,
        fallback_cache_envelopes_loaded=config['fallback_images'],
        test_used_for_selection=False, protected_dataset_profiles=controls,
        selected_cpu_model_bytes=pipeline.word_policy['model_bytes']+pipeline.router['model_bytes'],
        scope='Logical routing over authenticated cached real predictions; no online latency measurement')
    seal(root/'results.json', result)
    print(__import__('json').dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=read(args.config)['cpu_threads']):
        evaluate(args.config)
