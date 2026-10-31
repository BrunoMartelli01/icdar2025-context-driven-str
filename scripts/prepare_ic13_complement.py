"""Freeze a small U-Net complement to YOLO on selection images only."""

import argparse
from pathlib import Path

from localize.independent_word_slots import overlap_iou
from recognition.lexical_budget_router import correct_rows
from scripts.icdar_budget_data import annotation, key, predictions, read, sha
from scripts.retrain_icdar_lexicon_router import independent
from scripts.run_lexical_budget_router import lexicon
from scripts.run_yolo_f1_quality_study import seal


POLICY = dict(enabled=True, confidence_cap=1.01, protect_numeric=False,
              unmatched_policy='source_reject')


def complement(yolo, unet, dictionary, confidence, maximum_overlap):
    y_corrected, _ = correct_rows(yolo, dictionary, POLICY)
    u_corrected, _ = correct_rows(unet, dictionary, POLICY)
    if confidence is None:
        return list(yolo), 0
    extra = [r for r in u_corrected if r['confidence'] >= confidence and
             not any(overlap_iou(r['bbox'], y['bbox']) >= maximum_overlap for y in y_corrected)]
    return list(yolo) + [unet[r['source_row']] for r in extra], len(extra)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    assert config['mode'] == 'approximate'
    root = Path(config['study_root'])
    old = Path(config['original_router_root'])
    plan = read(old / 'resolved_configuration.json')
    records = {r['sample_id']: r for group in plan['records']['icdar13'].values() for r in group}
    dictionaries = {role: lexicon(config, 'icdar13', role) for role in ('train', 'test')}

    def sources(sid):
        y_path = old / 'experts/yolo' / (key(sid) + '.json')
        u_path = Path(config['unet_source']) / (key(sid) + '.json')
        y, u = read(y_path), read(u_path)
        assert y['image_sha256'] == u['image_sha256'] == records[sid]['image_sha256']
        assert not y['ground_truth_used'] and not u['ground_truth_used']
        return y_path, u_path, y['rows'], u['rows']

    grid = []
    options = [(None, 0)] + [(confidence, overlap) for confidence in config['confidence_grid']
                             for overlap in config['maximum_overlap_grid']]
    for confidence, overlap in options:
        totals = {k: 0 for k in ('tp', 'fp', 'fn', 'detected_words', 'gt_words',
                                'all_outputs', 'ignored_outputs')}
        added = 0
        for sid in plan['partitions']['icdar13']['selection']:
            _, _, y, u = sources(sid)
            combined, n = complement(y, u, dictionaries['train'], confidence, overlap)
            added += n
            rows, _ = correct_rows(combined, dictionaries['train'], POLICY)
            record = records[sid]
            gt, validity = annotation(record)
            counts = independent(predictions(rows, record, validity), gt)
            for k in totals:
                totals[k] += counts[k]
        f1 = 2 * totals['tp'] / (totals['detected_words'] + totals['gt_words'])
        grid.append(dict(confidence=confidence, maximum_overlap=overlap,
                         added_selection_words=added, counts=totals, f1=f1))
    chosen = max(grid, key=lambda p: (p['f1'], -p['added_selection_words'],
                                      p['confidence'] or 0., -p['maximum_overlap']))
    seal(root / 'complement_frozen.json', dict(status='FROZEN', grid=grid, selected=chosen,
        selection_ids=plan['partitions']['icdar13']['selection'],
        test_used_for_selection=False, config_sha256=sha(args.config)))
    staging = root / 'staging'
    seal(staging / 'resolved_configuration.json', dict(plan,
        light_expert_extension='YOLO640 plus validation-selected disjoint U-Net words',
        complement_frozen_sha256=sha(root / 'complement_frozen.json')))
    seal(staging / 'frozen_choices.json', read(old / 'frozen_choices.json'))
    manifest = {}
    for sid, record in records.items():
        y_path, u_path, y, u = sources(sid)
        role = 'train' if ':train:' in sid else 'test'
        combined, _ = complement(y, u, dictionaries[role], chosen['confidence'],
                                  chosen['maximum_overlap'])
        path = staging / 'experts/yolo' / (key(sid) + '.json')
        seal(path, dict(model='yolo_plus_unet_complement', image_sha256=record['image_sha256'],
            rows=combined, ground_truth_used=False,
            yolo_source_sha256=sha(y_path), unet_source_sha256=sha(u_path)))
        manifest[str(path)] = sha(path)
    seal(root / 'complement_receipt.json', dict(status='PASS', selected=chosen,
        files_sha256=manifest, new_neural_inferences=0,
        logical_deepsolo_count='determined by downstream router'))
    print('FROZEN complement', chosen, flush=True)


if __name__ == '__main__':
    main()
