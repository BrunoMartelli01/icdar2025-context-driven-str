"""Freeze a validation-only lexical policy, then replay the complete U-Net expert."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lexical_budget_router import correct_rows
from recognition.icdar_budget_router import lexical_counts
from recognition.f1_quality_study import aggregate
from scripts.icdar_budget_data import read, sha, key, annotation, predictions
from scripts.run_icdar13_dense_instances import (
    runtime, new_head, read_sample, emitted, cache, recognizer,
)
from scripts.run_yolo_f1_quality_study import seal


def dictionary(config, plan, root):
    canonical = {word.strip().upper(): word.strip() for word in
                 Path(config['common_vocabulary']).read_text().splitlines()}
    for sid in plan['partitions']['fit']:
        gt, _ = annotation(plan['records'][sid])
        for word in gt:
            canonical[word['text'].strip().upper()] = word['text'].strip()
    words = sorted(canonical)
    if not all(words):
        raise ValueError('Empty training vocabulary entry')
    vocab = root/'fit_generic_vocabulary.txt'
    pairs = root/'fit_generic_pairs.json'
    if vocab.exists():
        assert vocab.read_text().splitlines() == words and read(pairs) == canonical
    else:
        vocab.write_text('\n'.join(words)+'\n')
        seal(pairs, canonical)
    return DeepSoloLexicon(words, canonical, distance_exclusive=1.5)


def apply_policy(raw, policy, lexicon):
    selected = emitted(raw, dict(policy, maximum_candidates=policy['maximum_candidates']))
    lexical = dict(enabled=policy['lexical'] != 'none', confidence_cap=1.01,
                   protect_numeric=False, unmatched_policy=(
                       'source_reject' if policy['lexical'] == 'reject' else 'keep_unmatched'))
    return correct_rows(selected, lexicon, lexical)


def load_head(plan, root):
    state_path = root/'head/best.pt'
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    if state['plan_sha256'] != sha(root/'resolved_configuration.json'):
        raise ValueError('Head checkpoint/plan mismatch')
    head = new_head(plan['config'])
    head.load_state_dict(state['model_state_dict'], strict=True)
    return head.eval(), state


def select(config, plan, root):
    out = root/'policy'
    if (out/'frozen.json').exists():
        raise ValueError('Policy already frozen; preserve it and use infer')
    head, state = load_head(plan, root)
    lexicon = dictionary(config, plan, root)
    raw_policy = dict(plan['config']['inference'], peak_threshold=min(config['peak_thresholds']))
    raw = {}
    ocr = recognizer(plan)
    try:
        for sid in plan['partitions']['selection']:
            record = plan['records'][sid]
            rows = read_sample(plan, root, head, ocr, record, raw_policy)
            raw[sid] = rows
            seal(out/'selection_raw'/(key(sid)+'.json'),
                 dict(rows=rows, image_sha256=record['image_sha256'],
                      ground_truth_used_for_prediction=False))
    finally:
        ocr.release()
    grid = []
    for peak in config['peak_thresholds']:
        for sequence in config['sequence_thresholds']:
            for lexical in config['lexical_policies']:
                policy = dict(raw_policy, peak_threshold=peak, sequence_threshold=sequence,
                              lexical=lexical)
                counts, changes = [], 0
                for sid in plan['partitions']['selection']:
                    rows, trace = apply_policy(raw[sid], policy, lexicon)
                    record = plan['records'][sid]
                    gt, valid = annotation(record)
                    counts.append(lexical_counts(predictions(rows, record, valid), gt))
                    normalize = lambda s: ''.join(c for c in s.casefold() if c.isalnum())
                    changes += sum(not t['accepted'] or (t['match'] is not None and
                        normalize(t['original_text']) != normalize(t['match']['text'])) for t in trace)
                grid.append(dict(policy=policy, counts=aggregate(counts), normalized_changes=changes))
    best = max(enumerate(grid), key=lambda item: (item[1]['counts']['f1'],
                                                 -item[1]['normalized_changes'], -item[0]))[1]
    seal(out/'grid.json', dict(points=grid, deepsolo_images=0,
        total_images=len(plan['partitions']['selection']), deepsolo_percent=0.))
    seal(out/'frozen.json', dict(status='FROZEN', choice=best,
        test_used_for_selection=False, selection_ids=plan['partitions']['selection'],
        head_sha256=sha(root/'head/best.pt'), config_sha256=sha(config['_config_path']),
        fit_vocabulary_sha256=sha(root/'fit_generic_vocabulary.txt'),
        fit_pairs_sha256=sha(root/'fit_generic_pairs.json'), selected_epoch=state['epoch']))
    print('POLICY FROZEN', best, flush=True)


def infer(config, plan, root):
    freeze = read(root/'policy/frozen.json')
    if freeze['head_sha256'] != sha(root/'head/best.pt') or freeze['test_used_for_selection']:
        raise ValueError('Frozen selection contract changed')
    if freeze['config_sha256'] != sha(config['_config_path']):
        raise ValueError('Frozen policy config changed')
    if not (root/'test_cache_receipt.json').exists():
        cache(plan, root, test=True)
    head, state = load_head(plan, root)
    train_lexicon = dictionary(config, plan, root)
    test_lexicon = DeepSoloLexicon.from_files(config['test_vocabulary'], config['test_pairs'],
                                             distance_exclusive=1.5)
    policy = freeze['choice']['policy']
    lookup = dict(plan['records'], **{r['sample_id']: r for r in plan['test_records']})
    ocr = recognizer(plan)
    totals = {}
    try:
        for role in ('fit', 'selection', 'calibration', 'test'):
            directory = root/'selected_expert'/role
            counts = []
            for sid in plan['partitions'][role]:
                path = directory/'samples'/(key(sid)+'.json')
                if path.exists():
                    data = read(path)
                    if data['image_sha256'] != lookup[sid]['image_sha256'] or data['policy_sha256'] != sha(root/'policy/frozen.json'):
                        raise ValueError('Selected inference resume contract changed')
                    counts.append(data['counts'])
                    continue
                record = lookup[sid]
                raw = read_sample(plan, root, head, ocr, record, policy)
                rows, trace = apply_policy(raw, policy, test_lexicon if role == 'test' else train_lexicon)
                gt, valid = annotation(record)
                counted = lexical_counts(predictions(rows, record, valid), gt)
                seal(path, dict(sample_id=sid, rows=rows, raw_rows=raw, trace=trace,
                    counts=counted, image_sha256=record['image_sha256'],
                    policy_sha256=sha(root/'policy/frozen.json'),
                    ground_truth_used_for_prediction=False))
                counts.append(counted)
            total = aggregate(counts)
            totals[role] = dict(counts=total, deepsolo_images=0,
                total_images=len(plan['partitions'][role]), deepsolo_percent=0.)
            seal(directory/'metrics.json', totals[role])
            print(role, total, 'DeepSolo0/'+str(len(plan['partitions'][role])), flush=True)
    finally:
        ocr.release()
    seal(root/'selected_expert/results.json', dict(status='PASS', mode='approximate',
        claim_label='approximate reproduction', roles=totals,
        frozen_sha256=sha(root/'policy/frozen.json'), actual_new_deepsolo_inferences=0))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['select', 'infer'], required=True)
    args = parser.parse_args()
    config = read(args.config)
    if config['mode'] != 'approximate':
        raise ValueError('Explicit approximate mode required')
    for path, digest in config['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Pinned evaluation input changed: '+path)
    config['_config_path'] = str(args.config.resolve())
    root = Path(config['run_root'])
    plan = read(root/'resolved_configuration.json')
    runtime(plan['config'])
    {'select': select, 'infer': infer}[args.stage](config, plan, root)


if __name__ == '__main__':
    main()
