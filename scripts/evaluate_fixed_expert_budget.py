"""Freeze annotation-free TEST routing, then evaluate only selected fallbacks."""

import argparse
from pathlib import Path
import shutil
import time

from threadpoolctl import threadpool_limits

from recognition.common_budget_pipeline import counts, micro
from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.fusion_deployment_factory import load_fusion_budget_pipeline
from recognition.adapted_word_policy import emit
from recognition.selected_deepsolo_evidence import inspect_fresh_export
from scripts.audit_lightweight_word_pipeline import recount
from scripts.icdar_budget_data import read, sha, annotation, predictions
from scripts.train_lightweight_word_router import teacher
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def check_inputs(config):
    if config['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for path, digest in config['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed frozen TEST input: '+path)


def expected_frontend_checkpoint(config, gate_plan, gate_root):
    original = read(gate_plan['config']['expert_frozen'])['winner']['checkpoint_sha256']
    contract_path = config.get('frontend_refit_contract')
    if contract_path is None:
        return original
    if contract_path not in config['pinned_inputs']:
        raise ValueError('The final-refit contract must be an explicit pinned input')
    contract = read(contract_path)
    if (contract['status'] != 'PASS' or contract['mode'] != 'approximate' or
            contract['test_used_for_selection'] or contract['gate_root'] != str(gate_root) or
            contract['gate_selection_sha256'] != sha(gate_root/'selection_frozen.json') or
            contract['gate_qualification_sha256'] != sha(gate_root/'deployment_qualification.json') or
            contract['emission_policy_sha256'] != sha(gate_root/'emission_frozen.json') or
            config['pinned_inputs'].get(contract['checkpoint_path']) != contract['checkpoint_sha256']):
        raise ValueError('The final detector refit is not qualified for the frozen gate')
    return contract['checkpoint_sha256']


def route(config_path):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate evaluation namespace required')
    c = read(config_path); check_inputs(c)
    gate = Path(c['router_root']); plan = read(gate/'resolved_configuration.json')
    choice = read(gate/'selection_frozen.json')['winner']
    if choice['winner']['result']['combined']['f1'] < c['minimum_validation_f1']:
        raise ValueError('Declared validation qualification required before TEST')
    if c['fallback_profile'] != plan['config']['fallback_profiles'][choice['fallback_profile']]['fallback_profile']:
        raise ValueError('Frozen fallback checkpoint/profile changed')
    pipeline = load_fusion_budget_pipeline(gate)
    original = read(plan['config']['original_plan'])
    records = {r['sample_id']: r for r in original['records']['icdar13']['test']}
    ids = sorted(records)
    if len(ids) != c['total_images'] or sorted(plan['partitions']['test']) != ids:
        raise ValueError('Complete frozen TEST population required')
    export = Path(c['expert_export']); completion = read(export/'completion.json')
    index = read(export/'index.json')
    if (completion['status'] != 'COMPLETE' or completion['split'] != 'test' or
            completion['ground_truth_used'] or sorted(index) != ids):
        raise ValueError('Complete GT-free TEST frontend export required')
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    source = {}
    for package in plan['config']['snapshot_packages']:
        for path in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/path.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
            source[str(path.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    seal(root/'resolved_configuration.json', dict(config=c, config_sha256=sha(config_path),
        mode='approximate', claim_label='approximate reproduction',
        source_sha256=source, test_used_for_selection=False, historical_test_exposure=True,
        result_class='approximate reproduction', router_choice=choice))
    observations = {}; pins = {}; started = time.monotonic()
    checkpoint_sha256 = expected_frontend_checkpoint(c, plan, gate)
    override = plan['config'].get('emission_override'); lexicon = None
    if override is not None:
        profile = read(c['fallback_profile'])
        lexicon = DeepSoloLexicon.from_files(profile['lexicon_paths']['vocabulary'],
            profile['lexicon_paths']['pairs'], distance_exclusive=profile['distance_exclusive'])
    for sid in ids:
        entry = index[sid]
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Changed real TEST frontend output')
        sample = read(entry['path'])
        if (sample['ground_truth_used'] or sample['image_sha256'] != records[sid]['image_sha256'] or
                sample['checkpoint_sha256'] != checkpoint_sha256 or
                sample['frozen_expert_sha256'] != sha(plan['config']['expert_frozen'])):
            raise ValueError('TEST frontend/source identity differs')
        emitted = sample['rows']
        if override is not None:
            emitted, _ = emit(sample['raw_rows'], lexicon, read(override['policy'])['winner']['policy'])
        observations[sid] = pipeline.observe(sample['raw_rows'], emitted, records[sid]['image_wh'])
        pins[entry['path']] = entry['sha256']
    decision = pipeline.route(observations, {s: records[s]['rgb_sha256'] for s in ids},
                              fallback_images=c['fallback_images'])
    if c.get('fresh_deepsolo'):
        prior_path = c['fresh_deepsolo']['prior_decisions']
        if c['pinned_inputs'].get(prior_path) != sha(prior_path):
            raise ValueError('The pre-inference route must be an explicit pinned input')
        prior = read(prior_path)
        if (prior['ground_truth_used'] or prior['fallback_outputs_loaded'] or
                decision['fallback_ids'] != prior['fallback_ids'] or
                decision['accepted_ids'] != prior['accepted_ids']):
            raise ValueError('Fresh inference requires the original pre-fallback decisions')
    seal(root/'observations.json', observations); seal(root/'frontend_input_sha256.json', pins)
    seal(root/'decisions_before_fallback.json', dict(decision,
        scores={s: v['score'] for s, v in observations.items()}, ground_truth_used=False,
        fallback_outputs_loaded=False, observations_sha256=sha(root/'observations.json'),
        router_frozen_sha256=sha(gate/'selection_frozen.json'),
        seconds_cpu_cached_scoring_only=time.monotonic()-started))
    print('FROZEN TEST ROUTE', decision['deepsolo_images'], decision['total_images'],
          decision['deepsolo_percentage'], flush=True)


def evaluate(root):
    plan = read(root/'resolved_configuration.json'); c = plan['config']; check_inputs(c)
    pipeline = load_fusion_budget_pipeline(c['router_root'])
    decision = read(root/'decisions_before_fallback.json')
    observations = read(root/'observations.json')
    if (decision['ground_truth_used'] or decision['fallback_outputs_loaded'] or
            decision['observations_sha256'] != sha(root/'observations.json') or
            decision['router_frozen_sha256'] != sha(Path(c['router_root'])/'selection_frozen.json')):
        raise ValueError('Changed pre-fallback decisions/observations')
    fresh = inspect_fresh_export(c, decision)
    gate_plan = pipeline.plan; original = read(gate_plan['config']['original_plan'])
    records = {r['sample_id']: r for r in original['records']['icdar13']['test']}
    ids = sorted(records); fallback_ids = decision['fallback_ids']; accepted = set(decision['accepted_ids'])
    if (set(fallback_ids) & accepted or set(fallback_ids) | accepted != set(ids) or
            len(fallback_ids) != c['fallback_images'] or len(ids) != c['total_images']):
        raise ValueError('Complete exact TEST budget changed')
    profile = read(c['fallback_profile'])
    dictionary = DeepSoloLexicon.from_files(profile['lexicon_paths']['vocabulary'],
        profile['lexicon_paths']['pairs'], distance_exclusive=profile['distance_exclusive'])
    fallback_config = dict(fallback_profile=c['fallback_profile'],
        train_protocol=c['test_protocol'], raw_train_root=c['raw_test_root'])
    pins = read(root/'frontend_input_sha256.json'); reread_crops = 0
    if pipeline.choice['fallback_profile'].startswith('reread_'):
        reread = Path(c['reread_export']); done = read(reread/'completion.json')
        rplan = read(reread/'resolved_configuration.json'); index = read(reread/'index.json')
        if (done['status'] != 'COMPLETE' or done['smoke'] or done['split'] != 'test' or
                done['ground_truth_used'] or sorted(index) != sorted(fallback_ids) or
                rplan['config']['decision_path'] != str(root/'decisions_before_fallback.json') or
                rplan['config']['pinned_inputs'][rplan['config']['decision_path']] != sha(root/'decisions_before_fallback.json')):
            raise ValueError('Only frozen selected TEST fallbacks may have reread outputs')
        frozen = gate_plan['config']['fallback_profiles'][pipeline.choice['fallback_profile']]['reread_profile']['policy']
        policy = read(frozen)['winner']['policy']
        if rplan['config']['paddings'] != [policy['padding']]:
            raise ValueError('TEST rereading must use only the frozen padding')
        fallback_config['reread_profile'] = dict(index=str(reread/'index.json'), policy=frozen)
        reread_crops = done['actual_parseq_crop_inferences']
        for file in ('index.json', 'completion.json', 'resolved_configuration.json'):
            pins[str(reread/file)] = sha(reread/file)
        for entry in index.values():
            pins[entry['path']] = entry['sha256']
    results = {}; all_light = []
    for sid in ids:
        record = records[sid]; obs = observations[sid]
        gt, valid = annotation(record)
        light_flags = [p['ignored'] for p in predictions(obs['rows'], record, valid)]
        light_counts = counts(obs['rows'], gt, light_flags); all_light.append(light_counts)
        if sid in accepted:
            rows = obs['rows']; flags = light_flags; fallback_digest = None
        else:
            fallback = teacher(fallback_config, None, record, dictionary)
            for row in fallback['rows']:
                if row.get('invalid_envelope') is not None:
                    row.update(raw_text='', text='')
            rows = pipeline.combine_after_fallback(obs, fallback['rows'])
            flags = [predictions([r], record, valid)[0]['ignored']
                if r.get('fixed_light_complement') else r['ignored'] for r in rows]
            pins[fallback['source_path']] = fallback['source_sha256']; fallback_digest = fallback['source_sha256']
        measured = counts(rows, gt, flags)
        if measured != recount(rows, gt, flags):
            raise ValueError('Independent TEST lexical multiset recount differs')
        results[sid] = dict(action='light' if sid in accepted else 'deepsolo', rows=rows,
            ignored=flags, counts=measured, ground_truth=gt, score=obs['score'],
            fallback_sha256=fallback_digest, light_counts=light_counts)
    aggregate = micro(v['counts'] for v in results.values())
    if aggregate['images'] != c['total_images'] or aggregate['gt_words'] != c['care_words']:
        raise ValueError('Complete TEST image/care denominator changed')
    for path, digest in pins.items():
        if sha(path) != digest:
            raise ValueError('Input changed during frozen TEST evaluation')
    frontend = read(Path(c['expert_export'])/'completion.json')
    seal(root/'per_image.json', results); seal(root/'input_sha256.json', pins)
    result = dict(status='COMPLETE', mode='approximate', claim_label='approximate reproduction',
        result_class='approximate reproduction', counts=aggregate,
        target_met=aggregate['f1'] >= c['minimum_f1'], minimum_f1=c['minimum_f1'],
        deepsolo_images=len(fallback_ids), total_images=len(ids),
        deepsolo_percentage=100*len(fallback_ids)/len(ids),
        all_light=dict(counts=micro(all_light), deepsolo_images=0, total_images=len(ids), deepsolo_percentage=0.),
        actual_new_deepsolo_neural_inferences=fresh['actual_new_deepsolo_neural_inferences'],
        fresh_deepsolo_evidence=fresh if c.get('fresh_deepsolo') else None,
        fallback_cache_envelopes_loaded=len(fallback_ids),
        actual_new_detector_image_inferences=0 if c.get('frontend_inference_reused') else frontend['images'],
        actual_new_frontend_parseq_crop_inferences=0 if c.get('frontend_inference_reused') else frontend['actual_parseq_crop_inferences'],
        frontend_cached_generator_image_inferences=frontend['images'],
        frontend_cached_generator_parseq_crop_inferences=frontend['actual_parseq_crop_inferences'],
        actual_new_fallback_parseq_crop_inferences=reread_crops,
        test_used_for_selection=False, historical_test_exposure=True,
        scope=(('Frozen exact routing with fresh selected DeepSolo; ' if c.get('fresh_deepsolo') else
                'Logical exact routing over cached genuine DeepSolo; ')
               + ('reused cached' if c.get('frontend_inference_reused') else 'new real')
               + ' YOLO/PARSeq frontend' + (' and new selected-box PARSeq rereads' if reread_crops else '')
               + '. No online latency measurement.'))
    seal(root/'results.json', result)
    print(__import__('json').dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path)
    p.add_argument('--run-root', type=Path); p.add_argument('--stage', choices=['route', 'evaluate'], required=True)
    a = p.parse_args(); c = read(a.config) if a.config else read(a.run_root/'resolved_configuration.json')['config']
    with threadpool_limits(limits=c['cpu_threads']):
        if a.stage == 'route':
            route(a.config)
        else:
            evaluate(a.run_root)
