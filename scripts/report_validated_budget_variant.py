"""Bind an audited frozen variant selector to independently audited TEST results."""

import argparse
from pathlib import Path

from recognition.validated_budget_variant import load_validated_budget_variant
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


def verify_decisions(decision, images, fallback):
    selected = decision['fallback_ids']; accepted = decision['accepted_ids']
    if (decision['ground_truth_used'] or decision['fallback_outputs_loaded'] or
            len(selected) != fallback or len(accepted) != images - fallback or
            len(set(selected)) != len(selected) or len(set(accepted)) != len(accepted) or
            set(selected) & set(accepted)):
        raise ValueError('Invalid frozen pre-fallback decision or exact quota')


def run(path):
    if path.resolve().parent != REPO / 'configs/approximate':
        raise ValueError('Explicit approximate TEST reporting configuration required')
    config = read(path)
    if config['mode'] != 'approximate':
        raise ValueError('APPROXIMATE only')
    for file, digest in config['pinned_inputs'].items():
        if sha(file) != digest:
            raise ValueError('Changed TEST evidence: ' + file)
    meta = Path(config['selector_root']); frozen = read(meta / 'selection_frozen.json')
    summary = {}
    for name, spec in config['cases'].items():
        pipeline = load_validated_budget_variant(meta, name)
        selected = Path(frozen['selections'][name]['selected_root'])
        test = Path(spec['test_root']); audit = read(spec['independent_audit'])
        result_path = test / 'results.json'; result = read(result_path)
        decision_path = test / 'decisions_before_fallback.json'; decision = read(decision_path)
        previous_path = Path(spec['previous_common_result']); previous = read(previous_path)
        previous_audit = read(spec['previous_common_audit'])
        if (audit['status'] != 'PASS' or result['status'] != 'COMPLETE' or
                result['mode'] != 'approximate' or result['test_used_for_selection']):
            raise ValueError('Full independently audited approximate TEST result required')
        if spec['kind'] == 'ic13':
            if (read(test / 'resolved_configuration.json')['config']['router_root'] != str(selected) or
                    audit['result_sha256'] != sha(result_path) or
                    audit['decision_sha256'] != sha(decision_path) or previous_audit['status'] != 'PASS' or
                    previous_audit['result_sha256'] != sha(previous_path)):
                raise ValueError('ICDAR2013 TEST does not match the selected candidate')
        elif spec['kind'] == 'retention':
            if (test.parent != selected or audit['results'][name]['results_sha256'] != sha(result_path) or
                    result['case'] != name or previous_audit['status'] != 'PASS' or
                    previous_audit['results'][name]['results_sha256'] != sha(previous_path)):
                raise ValueError('Retention TEST does not match the selected candidate')
        else:
            raise ValueError('Unknown TEST evidence kind')
        verify_decisions(decision, spec['total_images'], spec['deepsolo_images'])
        counts = result['counts']; den = 2 * counts['tp'] + counts['fp'] + counts['fn']
        if (counts['images'] != spec['total_images'] or counts['gt_words'] != spec['care_words'] or
                counts['tp'] + counts['fn'] != counts['gt_words'] or
                counts['tp'] + counts['fp'] != counts['detected_words'] or not den or
                abs(counts['f1'] - 2 * counts['tp'] / den) > 1e-15 or
                result['deepsolo_images'] != spec['deepsolo_images'] or
                result['total_images'] != spec['total_images']):
            raise ValueError('TEST lexical counts or complete dataset differ')
        if (pipeline.choice['additional_map_models'] != 0 or pipeline.choice['extra_reader_crops'] != 0 or
                result['actual_new_deepsolo_neural_inferences'] != spec.get('expected_new_deepsolo_inferences', 0)):
            raise ValueError('Frozen no-extra-neural or DeepSolo inference accounting differs')
        target = counts['f1'] > spec['minimum_f1'] if spec['strictly_greater'] else counts['f1'] >= spec['minimum_f1']
        summary[name] = dict(counts=counts, deepsolo_images=spec['deepsolo_images'],
            total_images=spec['total_images'], deepsolo_percentage=100 * spec['deepsolo_images'] / spec['total_images'],
            minimum_f1=spec['minimum_f1'], target_met=target,
            previous_common_f1=previous['counts']['f1'],
            change_vs_previous_common_points=100 * (counts['f1'] - previous['counts']['f1']),
            selected_label=frozen['selections'][name]['selected_label'], selected_root=str(selected),
            additional_map_models=0, extra_reader_crops=0,
            actual_new_deepsolo_neural_inferences=result['actual_new_deepsolo_neural_inferences'],
            result_path=str(result_path), result_sha256=sha(result_path),
            independent_audit=spec['independent_audit'], independent_audit_sha256=sha(spec['independent_audit']))
    out = Path(config['output_path'])
    if out.exists():
        raise ValueError('TEST summary output already exists')
    new_inferences = sum(v['actual_new_deepsolo_neural_inferences'] for v in summary.values())
    seal(out, dict(status='COMPLETE', mode='approximate', claim_label='approximate reproduction',
        goal_met=all(v['target_met'] for v in summary.values()), cases=summary,
        selector_root=str(meta), selector_sha256=sha(meta / 'selection_frozen.json'),
        config_sha256=sha(path), historical_test_exposure=True, adaptive_research=True,
        actual_new_deepsolo_neural_inferences=new_inferences,
        scope=('Complete-batch routing, cached frontend and '
               + ('fresh selected DeepSolo outputs for explicitly recorded cases' if new_inferences else 'cached DeepSolo outputs')
               + '; no online latency measurement')))
    print(__import__('json').dumps(dict(goal_met=all(v['target_met'] for v in summary.values()), cases=summary)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    run(p.parse_args().config)
