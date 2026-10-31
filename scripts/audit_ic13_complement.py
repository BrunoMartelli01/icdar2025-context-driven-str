"""Replay the selection-only complement and every staged light image."""

import argparse
from collections import Counter
from pathlib import Path

from recognition.lexical_budget_router import correct_rows
from scripts.icdar_budget_data import annotation, key, predictions, read, sha
from scripts.prepare_ic13_complement import POLICY, complement
from scripts.run_lexical_budget_router import lexicon
from scripts.run_yolo_f1_quality_study import seal


def count(rows, record, truth, validity):
    predictions_ = predictions(rows, record, validity)
    norm = lambda text: ''.join(c for c in text.casefold() if c.isalnum())
    care = [p for p in predictions_ if not p['ignored']]
    emitted = Counter(norm(p['text']) for p in care if norm(p['text']))
    correct = Counter(norm(g['text']) for g in truth if norm(g['text']))
    tp = sum((emitted & correct).values())
    return dict(tp=tp, fp=len(care)-tp, fn=len(truth)-tp,
        detected_words=len(care), gt_words=len(truth),
        all_outputs=len(predictions_), ignored_outputs=len(predictions_)-len(care))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = read(args.config)
    root = Path(config['study_root'])
    old = Path(config['original_router_root'])
    plan = read(old / 'resolved_configuration.json')
    records = {r['sample_id']: r for group in plan['records']['icdar13'].values() for r in group}
    freeze = read(root / 'complement_frozen.json')
    receipt = read(root / 'complement_receipt.json')
    assert freeze['config_sha256'] == sha(args.config)
    assert not freeze['test_used_for_selection']
    assert freeze['selection_ids'] == plan['partitions']['icdar13']['selection']
    dictionaries = {role: lexicon(config, 'icdar13', role) for role in ('train', 'test')}
    source = {}
    for sid, record in records.items():
        y_path = old / 'experts/yolo' / (key(sid) + '.json')
        u_path = Path(config['unet_source']) / (key(sid) + '.json')
        yolo, unet = read(y_path), read(u_path)
        assert yolo['image_sha256'] == unet['image_sha256'] == record['image_sha256']
        assert not yolo['ground_truth_used'] and not unet['ground_truth_used']
        source[sid] = (y_path, u_path, yolo['rows'], unet['rows'])
    options = [(None, 0)] + [(confidence, overlap) for confidence in config['confidence_grid']
        for overlap in config['maximum_overlap_grid']]
    assert len(options) == len(freeze['grid'])
    for (confidence, overlap), saved in zip(options, freeze['grid']):
        total = Counter()
        added = 0
        for sid in freeze['selection_ids']:
            _, _, yolo, unet = source[sid]
            combined, n = complement(yolo, unet, dictionaries['train'], confidence, overlap)
            added += n
            rows, _ = correct_rows(combined, dictionaries['train'], POLICY)
            gt, valid = annotation(records[sid])
            total.update(count(rows, records[sid], gt, valid))
        assert saved['confidence'] == confidence and saved['maximum_overlap'] == overlap
        assert saved['added_selection_words'] == added
        for name, value in total.items():
            assert saved['counts'][name] == value
        assert saved['f1'] == 2*total['tp']/(total['detected_words']+total['gt_words'])
    chosen = max(freeze['grid'], key=lambda p: (p['f1'], -p['added_selection_words'],
        p['confidence'] or 0., -p['maximum_overlap']))
    assert chosen == freeze['selected'] == receipt['selected']
    assert len(receipt['files_sha256']) == len(records)
    for sid, record in records.items():
        y_path, u_path, yolo, unet = source[sid]
        role = 'train' if ':train:' in sid else 'test'
        expected, _ = complement(yolo, unet, dictionaries[role],
            chosen['confidence'], chosen['maximum_overlap'])
        path = root / 'staging/experts/yolo' / (key(sid) + '.json')
        actual = read(path)
        assert receipt['files_sha256'][str(path)] == sha(path)
        assert actual['rows'] == expected
        assert not actual['ground_truth_used']
        assert actual['image_sha256'] == record['image_sha256']
        assert actual['yolo_source_sha256'] == sha(y_path)
        assert actual['unet_source_sha256'] == sha(u_path)
    seal(root / 'complement_independent_audit.json', dict(status='PASS',
        selection_grid=len(options), staged_images=len(records),
        selected=chosen, test_used_for_selection=False,
        frozen_sha256=sha(root / 'complement_frozen.json'), code_sha256=sha(__file__)))
    print('PASS', len(options), 'validation policies,', len(records), 'staged images', flush=True)


if __name__ == '__main__':
    main()
