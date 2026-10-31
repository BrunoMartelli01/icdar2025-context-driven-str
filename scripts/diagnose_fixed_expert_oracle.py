"""Privileged exact-budget diagnostic over a frozen TEST run and cached native fallback."""

import argparse
from fractions import Fraction
from pathlib import Path

from recognition.budget_target_oracle import analyze_budget
from recognition.common_budget_pipeline import counts, micro
from recognition.deepsolo_lexicon import DeepSoloLexicon
from scripts.icdar_budget_data import annotation, read, sha
from scripts.train_lightweight_word_router import teacher
from scripts.run_yolo_f1_quality_study import seal


def run(test_root, output):
    plan = read(test_root/'resolved_configuration.json'); c = plan['config']
    decisions = read(test_root/'decisions_before_fallback.json'); result = read(test_root/'results.json')
    if (plan['mode'] != 'approximate' or decisions['ground_truth_used'] or decisions['fallback_outputs_loaded'] or
            result['test_used_for_selection'] or result['status'] != 'COMPLETE'):
        raise ValueError('A previously frozen complete approximate TEST result is required')
    gate = read(Path(c['router_root'])/'resolved_configuration.json')
    original = read(gate['config']['original_plan'])
    records = {r['sample_id']: r for r in original['records']['icdar13']['test']}
    per_image = read(test_root/'per_image.json')
    if len(records) != c['total_images'] or set(records) != set(per_image):
        raise ValueError('Complete TEST population is required for a privileged bound')
    profile = read(c['fallback_profile']); lex = DeepSoloLexicon.from_files(
        profile['lexicon_paths']['vocabulary'], profile['lexicon_paths']['pairs'],
        distance_exclusive=profile['distance_exclusive'])
    cfg = dict(fallback_profile=c['fallback_profile'], train_protocol=c['test_protocol'], raw_train_root=c['raw_test_root'])
    branches = {}; pins = {}
    for sid, r in records.items():
        value = teacher(cfg, None, r, lex)
        for row in value['rows']:
            if row.get('invalid_envelope') is not None:
                row.update(raw_text='', text='')
        gt, _ = annotation(r)
        if gt != per_image[sid]['ground_truth']:
            raise ValueError('Changed TEST care annotation')
        ds = counts(value['rows'], gt, [p['ignored'] for p in value['rows']])
        branches[sid] = dict(light=per_image[sid]['light_counts'], deepsolo=ds)
        pins[value['source_path']] = value['source_sha256']
    oracle = analyze_budget(branches, sorted(branches), c['fallback_images'], Fraction(9, 10))
    accepted = set(decisions['accepted_ids'])
    native_same_route = micro(v['light' if s in accepted else 'deepsolo'] for s, v in branches.items())
    seal(output, dict(status='COMPLETE', mode='approximate', claim_label='privileged oracle diagnostic',
        test_used_for_selection=False, privileged_ground_truth_used=True,
        source_test_result_sha256=sha(test_root/'results.json'), native_same_route=native_same_route,
        learned_route_deepsolo_images=c['fallback_images'], learned_route_total_images=len(records),
        learned_route_deepsolo_percentage=100*c['fallback_images']/len(records), oracle=oracle,
        fallback_cache_envelopes_inspected=len(records), fallback_cache_total_images=len(records),
        fallback_cache_percentage=100., actual_new_deepsolo_neural_inferences=0,
        fallback_variant='native qualified DeepSolo only, without additional PARSeq rereads',
        input_sha256=pins, code_sha256=sha(__file__),
        scope='Nondeployable bound. The oracle uses TEST ground truth and all233 cached fallback outputs to choose61; no model, threshold or policy is selected from this diagnostic.'))
    point = oracle['exact_budget']
    print(__import__('json').dumps(dict(status='COMPLETE', oracle_f1=point['f1'],
        oracle_deepsolo_images=point['deepsolo_images'], total_images=point['total_images'],
        deepsolo_percentage=100*point['deepsolo_images']/point['total_images'],
        native_same_route_f1=native_same_route['f1'], fallback_cache_images=len(records))), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--test-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); run(a.test_root, a.output)
