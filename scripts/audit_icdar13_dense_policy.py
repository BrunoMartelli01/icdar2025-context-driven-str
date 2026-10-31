"""Reconstruct FIT-only vocabulary and every frozen dense-expert policy count."""

import argparse
from collections import Counter
from pathlib import Path

import torch

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.lexical_budget_router import correct_rows
from scripts.icdar_budget_data import read, sha, key, annotation
from scripts.audit_icdar13_dense_instances import count
from scripts.evaluate_icdar13_dense_instances import apply_policy
from scripts.run_yolo_f1_quality_study import seal


FIELDS = ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'all_outputs', 'ignored_outputs')


def normal(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-only', action='store_true')
    args = parser.parse_args()
    c = read(args.config)
    root = Path(c['run_root'])
    plan = read(root/'resolved_configuration.json')
    for path, digest in c['pinned_inputs'].items():
        assert sha(path) == digest
    frozen = read(root/'policy/frozen.json')
    assert not frozen['test_used_for_selection']
    assert frozen['config_sha256'] == sha(args.config)
    assert frozen['head_sha256'] == sha(root/'head/best.pt')
    assert frozen['selection_ids'] == plan['partitions']['selection']
    assert frozen['fit_vocabulary_sha256'] == sha(root/'fit_generic_vocabulary.txt')
    assert frozen['fit_pairs_sha256'] == sha(root/'fit_generic_pairs.json')
    canonical = {w.strip().upper(): w.strip() for w in Path(c['common_vocabulary']).read_text().splitlines()}
    for sid in plan['partitions']['fit']:
        gt, _ = annotation(plan['records'][sid])
        for word in gt:
            canonical[word['text'].strip().upper()] = word['text'].strip()
    assert read(root/'fit_generic_pairs.json') == canonical
    assert (root/'fit_generic_vocabulary.txt').read_text().splitlines() == sorted(canonical)
    dictionary = DeepSoloLexicon(sorted(canonical), canonical, distance_exclusive=1.5)
    points = read(root/'policy/grid.json')['points']
    options = [(p, s, l) for p in c['peak_thresholds'] for s in c['sequence_thresholds']
               for l in c['lexical_policies']]
    assert len(options) == len(points)
    checks = Counter()
    raw = {sid: read(root/'policy/selection_raw'/(key(sid)+'.json'))
           for sid in plan['partitions']['selection']}
    for (peak, sequence, lexical), point in zip(options, points):
        policy = point['policy']
        assert (policy['peak_threshold'], policy['sequence_threshold'], policy['lexical']) == (
            peak, sequence, lexical)
        total = Counter()
        changes = 0
        for sid, data in raw.items():
            record = plan['records'][sid]
            assert data['image_sha256'] == record['image_sha256']
            assert not data['ground_truth_used_for_prediction']
            rows, trace = apply_policy(data['rows'], policy, dictionary)
            total.update(count(rows, record))
            changes += sum(not t['accepted'] or (t['match'] is not None and
                normal(t['original_text']) != normal(t['match']['text'])) for t in trace)
        for field in FIELDS:
            assert total[field] == point['counts'][field]
        assert point['counts']['f1'] == 2*total['tp']/(total['detected_words']+total['gt_words'])
        assert changes == point['normalized_changes']
        checks['selection_policies'] += 1
    chosen = max(enumerate(points), key=lambda item: (
        item[1]['counts']['f1'], -item[1]['normalized_changes'], -item[0]))[1]
    assert chosen == frozen['choice']
    head = torch.load(root/'head/best.pt', map_location='cpu', weights_only=False)
    assert head['epoch'] == frozen['selected_epoch']
    if args.selection_only:
        rejection = read(root/'validation_rejection.json')
        assert rejection['new_dense_head'] == chosen
        assert not rejection['test_neural_inference_run']
        assert not (root/'test_cache_receipt.json').exists()
        sources = rejection['input_sha256']
        for path, digest in sources.items():
            assert sha(path) == digest
        for point in rejection['qualified_pipeline_baseline']:
            total = Counter()
            for sid in plan['partitions']['selection']:
                paths = [path for path in sources if Path(path).name == key(sid)+'.json']
                assert len(paths) == 1
                data = read(paths[0])
                assert data['image_sha256'] == plan['records'][sid]['image_sha256']
                lexical = dict(enabled=point['lexical'] != 'none', confidence_cap=1.01,
                    protect_numeric=False, unmatched_policy=(
                        'source_reject' if point['lexical'] == 'reject' else 'keep_unmatched'))
                rows, _ = correct_rows(data['rows'], dictionary, lexical)
                total.update(count(rows, plan['records'][sid]))
            assert dict(total) == point['counts']
            assert point['f1'] == 2*total['tp']/(total['detected_words']+total['gt_words'])
            checks['qualified_baseline_policies'] += 1
        assert chosen['counts']['f1'] < max(row['f1'] for row in rejection['qualified_pipeline_baseline'])
        seal(args.output, dict(status='PASS', checks=dict(checks),
            scope='TRAIN selection policy and negative pipeline comparison only; no full TEST expert result',
            fit_dictionary_label_ids=plan['partitions']['fit'], heldout_labels_in_dictionary=0,
            full_control_test_inference_images=0, code_sha256=sha(__file__)))
        print('PASS', dict(checks), 'selection-only negative control', flush=True)
        return
    totals = read(root/'selected_expert/results.json')
    assert totals['frozen_sha256'] == sha(root/'policy/frozen.json')
    test_dictionary = DeepSoloLexicon.from_files(c['test_vocabulary'], c['test_pairs'],
                                                 distance_exclusive=1.5)
    lookup = dict(plan['records'], **{r['sample_id']: r for r in plan['test_records']})
    for role in ('fit', 'selection', 'calibration', 'test'):
        total = Counter()
        for sid in plan['partitions'][role]:
            data = read(root/'selected_expert'/role/'samples'/(key(sid)+'.json'))
            assert not data['ground_truth_used_for_prediction']
            assert data['image_sha256'] == lookup[sid]['image_sha256']
            assert data['policy_sha256'] == sha(root/'policy/frozen.json')
            rows, trace = apply_policy(data['raw_rows'], chosen['policy'],
                                      test_dictionary if role == 'test' else dictionary)
            assert rows == data['rows'] and trace == data['trace']
            value = count(rows, lookup[sid])
            assert value == data['counts']
            total.update(value)
            checks['expert_image_counts'] += 1
        saved = totals['roles'][role]
        for field in FIELDS:
            assert total[field] == saved['counts'][field]
        assert saved['deepsolo_images'] == 0 and saved['deepsolo_percent'] == 0
        assert saved['total_images'] == len(plan['partitions'][role])
    if not plan['smoke']:
        assert totals['roles']['test']['total_images'] == 233
        assert totals['roles']['test']['counts']['gt_words'] == 1095
    seal(args.output, dict(status='PASS', checks=dict(checks),
        fit_dictionary_label_ids=plan['partitions']['fit'], heldout_labels_in_dictionary=0,
        results_sha256=sha(root/'selected_expert/results.json'), code_sha256=sha(__file__)))
    print('PASS', dict(checks), flush=True)


if __name__ == '__main__':
    main()
