"""Privileged error diagnosis; never changes the frozen expert or router."""

import argparse
from collections import Counter
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment

from segmentation.unet_dense_instances import box_iou
from scripts.icdar_budget_data import read, sha, key, annotation
from scripts.run_yolo_f1_quality_study import seal


def normal(text):
    return ''.join(character for character in text.casefold() if character.isalnum())


def assign(truth, available, candidates):
    """Lexical multiplicity first, maximum bbox overlap only for identity diagnosis."""
    matched = []
    for word in sorted({normal(truth[i]['text']) for i in available}-{''}):
        gt_ids = [i for i in available if normal(truth[i]['text']) == word]
        raw_ids = [j for j, row in enumerate(candidates) if normal(row['raw_text']) == word]
        if not raw_ids:
            continue
        overlap = np.array([[box_iou(truth[i]['bbox'], candidates[j]['bbox'])
                             for j in raw_ids] for i in gt_ids])
        rows, columns = linear_sum_assignment(-overlap)
        matched.extend((gt_ids[i], raw_ids[j]) for i, j in zip(rows, columns))
    return matched


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--mode', required=True, choices=['approximate'])
    parser.add_argument('--expert-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--diagnostic-iou', type=float, required=True)
    parser.add_argument('--expected-images', type=int, required=True)
    parser.add_argument('--expected-care-words', type=int, required=True)
    args = parser.parse_args()
    assert 0 < args.diagnostic_iou <= 1
    root = args.expert_root
    assert read(root/'independent_audit.json')['status'] == 'PASS'
    config = read(root/'resolved_configuration.json')['config']
    training = read(Path(config['training_root'])/'resolved_configuration.json')
    frozen = read(root/'policy/frozen.json')
    assert frozen['choice']['policy']['lexical'] == 'none', 'Diagnosis needs raw/emitted identity correspondence'
    records = {record['sample_id']: record for record in training['test_records']}
    ids = training['partitions']['test']
    assert len(ids) == args.expected_images
    args.output_root.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output_root/'executed_source.py')
    sources = {str(root/'policy/frozen.json'): sha(root/'policy/frozen.json'),
               str(root/'independent_audit.json'): sha(root/'independent_audit.json')}
    totals = Counter()
    for sid in ids:
        record = records[sid]
        path = root/'selected_expert/test/samples'/(key(sid)+'.json')
        data = read(path)
        sources[str(path)] = sha(path)
        truth, _ = annotation(record)
        available = set(range(len(truth)))
        emitted = assign(truth, available, data['rows'])
        assert len(emitted) == data['counts']['tp']
        for index, _ in emitted:
            available.remove(index)
        kept_ids = {row['candidate_id'] for row in data['rows']}
        rejected = [row for row in data['raw_rows'] if row['candidate_id'] not in kept_ids]
        rescued = assign(truth, available, rejected)
        for index, _ in rescued:
            available.remove(index)
        rows = [dict(gt_index=index, text=truth[index]['text'], category='correct_emitted',
                     candidate=data['rows'][candidate]['candidate_id']) for index, candidate in emitted]
        rows.extend(dict(gt_index=index, text=truth[index]['text'], category='correct_raw_rejected',
                         candidate=rejected[candidate]['candidate_id']) for index, candidate in rescued)
        for index in sorted(available):
            candidates = data['raw_rows']
            best = max(range(len(candidates)), key=lambda j: box_iou(
                truth[index]['bbox'], candidates[j]['bbox']), default=None)
            overlap = 0. if best is None else box_iou(truth[index]['bbox'], candidates[best]['bbox'])
            category = ('symbol_only_gt' if not normal(truth[index]['text']) else
                        'viable_box_wrong_reading' if overlap >= args.diagnostic_iou else
                        'no_viable_raw_box')
            rows.append(dict(gt_index=index, text=truth[index]['text'], category=category,
                best_iou=overlap, best_candidate=None if best is None else candidates[best]['candidate_id'],
                best_raw_text=None if best is None else candidates[best]['raw_text']))
        assert sorted(row['gt_index'] for row in rows) == list(range(len(truth)))
        totals.update(row['category'] for row in rows)
        seal(args.output_root/'samples'/(key(sid)+'.json'),
             dict(sample_id=sid, gt_words=len(truth), rows=rows, source_sha256=sha(path)))
    assert sum(totals.values()) == args.expected_care_words
    assert totals['correct_emitted'] == read(root/'selected_expert/results.json')['roles']['test']['counts']['tp']
    seal(args.output_root/'results.json', dict(status='PASS', mode=args.mode,
        claim_label='approximate reproduction', privileged_gt=True,
        primary_metric_unchanged=True, categories=dict(totals), total_images=len(ids),
        total_care_words=sum(totals.values()), deepsolo_images=0, deepsolo_percentage=0.,
        diagnostic_iou=args.diagnostic_iou, source_sha256=sources,
        code_sha256=sha(args.output_root/'executed_source.py'), actual_new_neural_inferences=0,
        scope='GT diagnostic only. Lexical identity assignment preserves the primary multiset TP count; '
              'bboxIoU only separates raw crop coverage from wrong readings after that. '
              'Any-proposal coverage is not an official one-to-one detector recall. '
              'No threshold, expert output or routing policy is changed.'))
    print('DIAGNOSIS', dict(totals), flush=True)


if __name__ == '__main__':
    main()
