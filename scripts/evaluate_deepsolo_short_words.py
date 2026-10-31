"""Reassess frozen DeepSolo profiles after an explicit symmetric length filter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from recognition.deepsolo_word_f1 import aggregate, normalize_word
from scripts.audit_deepsolo_word_f1 import independent_counts
from scripts.evaluate_deepsolo_word_f1_fusion import counts, predictions
from scripts.run_deepsolo_word_f1_study import read, save, sha


def filter_population(preds, gt, minimum):
    if type(minimum) is not int or minimum < 1:
        raise ValueError('Minimum normalized word length must be a positive integer')
    kept_p = [p for p in preds if len(normalize_word(p['text'])) >= minimum]
    kept_g = [g for g in gt if len(normalize_word(g['text'])) >= minimum]
    return kept_p, kept_g, dict(original_gt_words=len(gt), excluded_gt_words=len(gt)-len(kept_g),
        original_care_predictions=sum(not p['ignored'] for p in preds),
        excluded_care_predictions=sum(not p['ignored'] for p in preds)-sum(not p['ignored'] for p in kept_p),
        excluded_all_predictions=len(preds)-len(kept_p))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['approximate'])
    for key in ('config', 'run', 'output-root'):
        parser.add_argument('--'+key, required=True, type=Path)
    args = parser.parse_args(); cfg = read(args.config)
    if (cfg['schema_version'] != 1 or cfg['reproduction_mode'] != args.mode
            or cfg['claim_label'] != 'approximate reproduction' or cfg['official_evaluation'] is not False
            or cfg['table_5_claim'] is not False or 'approximate' not in args.config.parts
            or args.output_root.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError('Explicit APPROXIMATE configuration and external output required')
    filter_population([], [], cfg['minimum_normalized_characters'])
    inputs = {}
    for name, expected in cfg['input_sha256'].items():
        path = args.run/name
        if sha(path) != expected:
            raise ValueError('Changed frozen source: '+str(path))
        inputs[str(path)] = expected
    processed = read(args.run/'processed_predictions.json')
    previous = read(args.run/'results.json'); profiles = read(args.run/'deployment_profiles.json')
    rows = processed['splits']['test']
    if (previous['status'] != 'SUCCESS' or len(rows) != cfg['expected_test_images']
            or len({r['sample_id'] for r in rows}) != len(rows)
            or sum(len(r['ground_truth']) for r in rows) != cfg['expected_unfiltered_test_care_words']):
        raise ValueError('Incomplete or changed full TEST population')
    policies = dict(baseline=cfg['baseline_policy'], **{k:v['policy'] for k,v in profiles['profiles'].items()})
    plan = dict(config=cfg, claim_label=cfg['claim_label'], reproduction_mode=args.mode,
        result_class='approximate reproduction', official_evaluation=False,
        input_sha256=inputs, policies=policies, test_fitting=False,
        code_sha256={str(p):sha(p) for p in (Path(__file__), Path(__file__).parents[1]/'recognition/deepsolo_word_f1.py',
            Path(__file__).parents[1]/'scripts/evaluate_deepsolo_word_f1_fusion.py',
            Path(__file__).parents[1]/'scripts/audit_deepsolo_word_f1.py')})
    save(args.output_root/'resolved_configuration.json', plan)
    result = dict(status='SUCCESS', reproduction_mode=args.mode, claim_label=cfg['claim_label'],
        result_class='approximate reproduction', protocol='symmetric normalized minimum word length',
        official_evaluation=False, minimum_normalized_characters=cfg['minimum_normalized_characters'],
        test_fitting=False, profiles={},
        routing=dict(type='DeepSolo-only cached replay', deepsolo_images=len(rows), total_images=len(rows),
                     deepsolo_percentage=100.0, new_online_inference_calls=0))
    checks = 0
    for name, policy in policies.items():
        per_image = []
        for row in rows:
            original = predictions(row, policy)
            kept_p, kept_g, excluded = filter_population(original, row['ground_truth'], cfg['minimum_normalized_characters'])
            record = dict(sample_id=row['sample_id'], ground_truth=kept_g, predictions=kept_p,
                excluded=excluded, metrics={}, original_metrics={})
            for metric in ('word', 'localized'):
                record['original_metrics'][metric] = counts(original, row['ground_truth'], metric, cfg['matching_iou'])
                record['metrics'][metric] = counts(kept_p, kept_g, metric, cfg['matching_iou'])
                independent = independent_counts(kept_p, kept_g, metric == 'localized', cfg['matching_iou'])
                if any(record['metrics'][metric][k] != independent[k] for k in independent):
                    raise ValueError('Independent filtered recount differs')
                checks += 1
            per_image.append(record)
        metrics = {}
        for metric in ('word', 'localized'):
            old = aggregate(r['original_metrics'][metric] for r in per_image)
            expected = (previous['baseline']['test'][metric] if name == 'baseline'
                        else profiles['profiles'][name]['splits']['test'][metric]['metrics'])
            if old != expected:
                raise ValueError('Unfiltered historical baseline replay differs')
            new = aggregate(r['metrics'][metric] for r in per_image)
            metrics[metric] = dict(unfiltered=old, filtered=new, delta_f1=new['f1']-old['f1'])
        result['profiles'][name] = dict(policy=policy, metrics=metrics, per_image=per_image,
            routing=dict(result['routing']),
            removed={k:sum(r['excluded'][k] for r in per_image) for k in per_image[0]['excluded']},
            images_with_no_filtered_gt=sum(not r['ground_truth'] for r in per_image))
    result['independent_image_metric_checks'] = checks
    save(args.output_root/'results.json', result)
    print(json.dumps({k:dict(metrics=v['metrics'], removed=v['removed'], images_with_no_filtered_gt=v['images_with_no_filtered_gt'])
                      for k,v in result['profiles'].items()}))


if __name__ == '__main__':
    main()
