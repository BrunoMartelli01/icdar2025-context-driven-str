"""Replay immutable DeepSolo reports; select symbol-free F1 policies on TRAIN only."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from recognition.deepsolo_word_f1 import (
    aggregate, apply_policy, beam_decode, error_inventory, recognition_confidence, word_counts,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def load_report(root, beam_width):
    report = json.loads((root/'results.json').read_text())
    plan = json.loads((root/'resolved_configuration.json').read_text())
    if report['status'] != 'SUCCESS' or plan['reproduction_mode'] != 'approximate' or plan['limit']:
        raise ValueError('Require complete successful APPROXIMATE source')
    splits = {}
    inputs = {str(root/name): sha(root/name) for name in ('results.json', 'resolved_configuration.json')}
    for split, result in report['splits'].items():
        gt = {s['sample_id']: s for s in plan['images'][split]}
        if len(gt) != len(plan['images'][split]) or len(result['per_image']) != len(gt):
            raise ValueError('Incomplete source population')
        rows = []
        for r in result['per_image']:
            path = Path(r['raw_response_path'])
            if sha(path) != r['raw_response_sha256']:
                raise ValueError('Raw source changed: '+str(path))
            raw = json.loads(path.read_text())
            response = raw['response']
            if not response['success']:
                raise ValueError('Failed source sample')
            inputs[str(path)] = r['raw_response_sha256']
            detections = {d['source_index']: d for d in response['detections']}
            preds = []
            for p in r['predictions']:
                d = detections[p['instance_id']]
                if p['text'] != d['text'] or p['score'] != d['score']:
                    raise ValueError('Raw prediction mismatch')
                probs = d['raw']['recognition_class_probabilities']
                preds.append(dict(p, recognition_confidence=recognition_confidence(probs),
                                  beam_text=beam_decode(probs, beam_width)))
            rows.append(dict(sample_id=r['sample_id'], ground_truth=gt.pop(r['sample_id'])['ground_truth'],
                             predictions=preds))
        if gt:
            raise ValueError('Missing sample')
        splits[split] = rows
    return splits, inputs


def evaluate(rows, policy):
    return aggregate(word_counts(apply_policy(r['predictions'], policy), r['ground_truth']) for r in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['approximate'])
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--r50-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if (cfg['reproduction_mode'] != args.mode or cfg['claim_label'] != 'approximate reproduction'
            or cfg['table_5_claim'] is not False or cfg['selection_split'] != 'train'
            or cfg['evaluation_split'] != 'test' or 'approximate' not in args.config.parts):
        raise ValueError('Explicit APPROXIMATE TRAIN/TEST configuration required')
    rows, inputs = load_report(args.source_root, cfg['beam_width'])
    r50, r50_inputs = load_report(args.r50_root, cfg['beam_width'])
    inputs.update(r50_inputs)
    # Persist the selector before evaluating TEST. Tie rule prefers the earlier
    # declared policy (greedy/no filtering first), with no TEST access in fitting.
    policies = [dict(decoder=d, score_threshold=s, recognition_threshold=q, same_text_nms_iou=n)
                for d, s, q, n in itertools.product(cfg['decoders'], cfg['score_thresholds'],
                    cfg['recognition_thresholds'], cfg['same_text_nms_ious'])]
    training = [dict(policy=p, metrics=evaluate(rows['train'], p)) for p in policies]
    winner = max(range(len(training)), key=lambda i: training[i]['metrics']['f1'])
    selected = training[winner]['policy']
    source_files = [Path(__file__), Path(__file__).parents[1]/'recognition/deepsolo_word_f1.py', args.config]
    plan = dict(config=cfg, input_sha256=inputs, source_sha256={str(p):sha(p) for p in source_files},
                claim_label=cfg['claim_label'], reproduction_mode=args.mode, selected_policy=selected,
                selection_split='train', test_fitting=False, official_evaluation=False)
    write_new(args.output_root/'resolved_configuration.json', plan)
    write_new(args.output_root/'selection.json', dict(selected=selected, train_grid=training))
    baseline = dict(decoder='greedy', score_threshold=cfg['baseline_score_threshold'],
                    recognition_threshold=0., same_text_nms_iou=None)
    results = dict(claim_label=cfg['claim_label'], reproduction_mode=args.mode, official_evaluation=False,
                   metric=cfg['metric'], selected_policy=selected, splits={})
    for split, samples in rows.items():
        original = evaluate(samples, baseline)
        improved = evaluate(samples, selected)
        if split in r50 and [r['sample_id'] for r in r50[split]] != [r['sample_id'] for r in samples]:
            raise ValueError('R50 comparison population differs')
        per_image = []
        for row in samples:
            p = apply_policy(row['predictions'], baseline)
            q = apply_policy(row['predictions'], selected)
            per_image.append(dict(sample_id=row['sample_id'], ground_truth=row['ground_truth'],
                baseline=word_counts(p, row['ground_truth']), selected=word_counts(q, row['ground_truth']),
                baseline_predictions=p, selected_predictions=q,
                errors=error_inventory(p, row['ground_truth'], cfg['diagnostic_bbox_iou'])))
        results['splits'][split] = dict(baseline=original, selected=improved,
            r50_baseline=evaluate(r50[split], baseline) if split in r50 else None,
            delta_f1=improved['f1']-original['f1'], per_image=per_image)
    results['status'] = 'SUCCESS'
    write_new(args.output_root/'results.json', results)
    print(json.dumps({s:{k:v for k,v in r.items() if k!='per_image'} for s,r in results['splits'].items()}))


if __name__ == '__main__':
    main()
