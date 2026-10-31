"""Independently recount deployable profiles and compare real GT-free GPU smokes."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.audit_deepsolo_word_f1 import independent_counts
from scripts.evaluate_deepsolo_word_f1_fusion import predictions
from scripts.run_deepsolo_word_f1_study import read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--smoke-root', required=True, type=Path)
    args = parser.parse_args()
    report = read(args.run/'deployment_profiles.json')
    processed = read(args.run/'processed_predictions.json')
    selection = read(args.run/'selection.json')
    checks, smokes, max_confidence_difference = 0, 0, 0.
    for name, profile in report['profiles'].items():
        if sha(Path(profile['config_path'])) != profile['config_sha256']:
            raise ValueError('Changed deployable configuration')
        policy = profile['policy']
        for split, rows in processed['splits'].items():
            for metric in ('word', 'localized'):
                total = Counter()
                for row in rows:
                    total.update(independent_counts(predictions(row, policy), row['ground_truth'],
                        metric == 'localized', selection['config']['matching_iou']))
                    checks += 1
                declared = profile['splits'][split][metric]['metrics']
                if any(total[k] != declared[k] for k in ('tp', 'fp', 'fn')):
                    raise ValueError('Independent deployment profile matching differs')
        smoke_path = args.smoke_root/('smoke-native-final' if name == 'native_word' else 'smoke-global-final')/'results.json'
        smoke = read(smoke_path)
        load = smoke['load_validation']
        if (smoke['status'] != 'SUCCESS' or smoke['selected_policy'] != policy
                or smoke['config_sha256'] != profile['config_sha256'] or not load['strict_compatibility_check']
                or any(load[k] for k in ('missing_keys', 'unexpected_keys', 'incorrect_shapes'))):
            raise ValueError('Real GPU smoke did not preserve checkpoint/policy')
        rows = {r['image_sha256']:r for split in processed['splits'].values() for r in split}
        for image in smoke['images']:
            expected = predictions(rows[image['image_sha256']], policy)
            actual = image['selected_predictions']
            identity = lambda p:(p['instance_id'], p.get('view', 'native'), p['text'], p['bbox'])
            if [identity(p) for p in actual] != [identity(p) for p in expected]:
                raise ValueError('GT-free GPU output differs in text, boxes, order or selection')
            for p, q in zip(actual, expected):
                for key in ('score', 'recognition_confidence'):
                    difference = abs(p[key]-q[key])
                    max_confidence_difference = max(max_confidence_difference, difference)
                    if difference > 1e-6:
                        raise ValueError('GT-free GPU confidence differs beyond float32 tolerance')
            smokes += 1
    result = dict(status='PASS', independent_image_metric_checks=checks,
        real_gpu_image_profile_checks=smokes, exact_text_box_order_selection=True,
        maximum_confidence_absolute_difference=max_confidence_difference,
        confidence_tolerance=1e-6, deployment_profiles_sha256=sha(args.run/'deployment_profiles.json'),
        audit_code_sha256=sha(Path(__file__)))
    save(args.run/'deployment_independent_audit.json', result)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
