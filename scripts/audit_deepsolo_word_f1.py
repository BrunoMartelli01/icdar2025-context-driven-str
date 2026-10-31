"""Independent Hungarian recount and decision replay for the DeepSolo F1 study."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from recognition.deepsolo_word_f1 import apply_policy, fuse_views
from recognition.word_slot_text import ALPHANUMERIC, normalize_word_text
from scripts.run_deepsolo_word_f1_study import read, save, sha


def independent_counts(predictions, ground_truth, localized, threshold):
    care = [p for p in predictions if not p['ignored']]
    weights = np.zeros((len(care), len(ground_truth)), dtype=np.int8)
    for i, p in enumerate(care):
        text = normalize_word_text(p['text'], ALPHANUMERIC)
        if not text or p.get('invalid_envelope') is not None:
            continue
        for j, g in enumerate(ground_truth):
            if text != normalize_word_text(g['text'], ALPHANUMERIC):
                continue
            if localized:
                if p['bbox'] is None:
                    continue
                a, b = np.array(p['bbox'], dtype=np.float64), np.array(g['bbox'], dtype=np.float64)
                common = np.maximum(np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2]), 0).prod()
                union = np.maximum(a[2:]-a[:2], 0).prod()+np.maximum(b[2:]-b[:2], 0).prod()-common
                if union <= 0 or common/union < threshold:
                    continue
            weights[i, j] = 1
    idx, jdx = linear_sum_assignment(-weights)
    tp = int(weights[idx, jdx].sum())
    return dict(tp=tp, fp=len(care)-tp, fn=len(ground_truth)-tp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    args = parser.parse_args()
    selection = read(args.run/'selection.json'); result = read(args.run/'results.json')
    processed = read(args.run/'processed_predictions.json')
    if (result['status'] != 'SUCCESS' or result['reproduction_mode'] != 'approximate'
            or result['baseline_replay'] != 'PASS' or selection['test_fitting'] is not False
            or processed['input_sha256'] != selection['input_sha256']):
        raise ValueError('Incomplete or inconsistent experiment')
    for path, digest in selection['input_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Changed source evidence: '+path)
    for path, digest in selection['source_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Changed evaluation implementation: '+path)
    checked = 0
    for metric, policy in selection['selected'].items():
        # Confirm the declared finite-grid TRAIN optimum and stable tie rule.
        best = max(selection['train_grid'], key=lambda r:r[metric]['f1'])['policy']
        if policy != best or result['selected'][metric] != policy:
            raise ValueError('Selection is not the declared TRAIN optimum')
        for split, rows in processed['splits'].items():
            original = {r['sample_id']:r for r in rows}
            scored = result['protocols'][metric][split]
            if len(scored['per_image']) != len(original):
                raise ValueError('Missing or duplicated result images')
            total = Counter()
            seen = set()
            for row in scored['per_image']:
                sid = row['sample_id']
                if sid in seen:
                    raise ValueError('Duplicate image')
                seen.add(sid); raw = original[sid]
                replay = (fuse_views(raw['native'], raw['large'], policy) if policy['resolution'] == 'fused'
                          else apply_policy(raw[policy['resolution']], policy))
                if replay != row['predictions'] or raw['ground_truth'] != row['ground_truth']:
                    raise ValueError('Frozen policy/GT replay failed')
                counts = independent_counts(replay, row['ground_truth'], metric == 'localized', selection['config']['matching_iou'])
                if any(counts[k] != row['selected'][k] for k in counts):
                    raise ValueError('Independent matching differs: '+sid+'/'+metric)
                total.update(counts); checked += 1
            observed = scored['selected']
            if any(total[k] != observed[k] for k in ('tp', 'fp', 'fn')):
                raise ValueError('Aggregate differs from per-image counts')
            tp, fp, fn = total['tp'], total['fp'], total['fn']
            expected = dict(f1=2*tp/(2*tp+fp+fn), precision=tp/(tp+fp), recall=tp/(tp+fn))
            if any(abs(expected[k]-observed[k]) > 1e-14 for k in expected):
                raise ValueError('Metric formula differs')
    audit = dict(status='PASS', independent_image_metric_checks=checked,
        matching='Independent SciPy Hungarian assignment; separate repository text policy',
        source_hashes_checked=len(selection['input_sha256']), all_samples_retained=True,
        selection_split='train', test_fitting=False,
        result_sha256=sha(args.run/'results.json'), selection_sha256=sha(args.run/'selection.json'),
        processed_predictions_sha256=sha(args.run/'processed_predictions.json'), audit_code_sha256=sha(Path(__file__)))
    save(args.run/'independent_audit.json', audit)
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
