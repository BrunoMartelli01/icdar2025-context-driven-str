"""Independently recount finite selection, RGB crops and the full frozen expert."""

import argparse
from collections import Counter
import itertools
from pathlib import Path

import numpy as np
from PIL import Image

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.unet_token_routing import unet_proposals
from recognition.word_unet_policy import apply_word_policy, normalized_changes, select_word_policy
from segmentation.icdar13_word_refit import padded_rows
from scripts.audit_icdar13_dense_instances import count
from scripts.analyze_budget_target_router import certify
from scripts.crop_clip_runtime import array_sha
from scripts.icdar_budget_data import read, sha, key, annotation
from scripts.run_yolo_f1_quality_study import seal

FIELDS = ('tp', 'fp', 'fn', 'detected_words', 'gt_words', 'all_outputs', 'ignored_outputs')


def verify_total(total, saved, images):
    for field in FIELDS:
        assert total[field] == saved[field], (field, total[field], saved[field])
    assert saved['f1'] == 2*total['tp']/(total['detected_words']+total['gt_words'])
    assert saved['images'] == images


def check_raw(data, record, policy, counters):
    assert not data['ground_truth_used_for_prediction']
    assert data['image_sha256'] == sha(record['image_path']) == record['image_sha256']
    assert sha(data['maps_path']) == data['maps_sha256']
    with np.load(data['maps_path'], allow_pickle=False) as z:
        p = z['probabilities']
    proposals, _, _ = unet_proposals(p, record['image_wh'], policy)
    proposals = padded_rows(proposals, record['image_wh'], policy['padding'])
    assert len(proposals) == len(data['rows'])
    with Image.open(record['image_path']) as image:
        rgb = np.ascontiguousarray(image.convert('RGB'))
    for proposal, row in zip(proposals, data['rows']):
        for name in ('bbox', 'bbox_native', 'instance_id', 'foreground_area', 'detector_score'):
            assert proposal[name] == row[name], name
        a, b, c, d = row['bbox']
        assert array_sha(np.ascontiguousarray(rgb[b:d, a:c])) == row['crop_rgb_sha256']
        counters['rgb_crop_hashes'] += 1


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selection-only', action='store_true')
    args = parser.parse_args()
    root = args.run_root
    resolved = read(root/'resolved_configuration.json')
    config = resolved['config']
    for name, digest in config['pinned_inputs'].items():
        assert sha(name) == digest
    for name, digest in resolved['source_sha256'].items():
        assert sha(root/'code_snapshot'/name) == digest
    training = Path(config['training_root'])
    plan = read(training/'resolved_configuration.json')
    assert sha(training/'resolved_configuration.json') == resolved['training_plan_sha256']
    frozen = read(root/'policy/frozen.json')
    assert not frozen['test_used_for_selection']
    assert frozen['config_sha256'] == sha(root/'resolved_configuration.json')
    assert frozen['selection_ids'] == plan['partitions']['selection']
    assert frozen['fit_dictionary_label_ids'] == plan['partitions']['fit']
    assert frozen['training_audit_sha256'] == sha(training/'independent_audit.json')
    records = {r['sample_id']: r for r in plan['train_records']+plan['test_records']}
    canonical = {w.strip().upper(): w.strip()
                 for w in Path(config['common_vocabulary']).read_text().splitlines()}
    for sid in plan['partitions']['fit']:
        truth, _ = annotation(records[sid])
        for word in truth:
            canonical[word['text'].strip().upper()] = word['text'].strip()
    assert read(root/'fit_generic_pairs.json') == canonical
    assert (root/'fit_generic_vocabulary.txt').read_text().splitlines() == sorted(canonical)
    assert frozen['fit_vocabulary_sha256'] == sha(root/'fit_generic_vocabulary.txt')
    assert frozen['fit_pairs_sha256'] == sha(root/'fit_generic_pairs.json')
    lexicon = DeepSoloLexicon(sorted(canonical), canonical, distance_exclusive=1.5)
    points = read(root/'policy/grid.json')['points']
    options = list(itertools.product(config['arms'], config['text_thresholds'],
        config['paddings'], config['sequence_thresholds'], config['lexical_policies']))
    assert len(points) == len(options)
    checks, raw_seen = Counter(), set()
    for option, point in zip(options, points):
        arm, threshold, padding, sequence, lexical = option
        p = point['policy']
        assert (point['arm'], p['text_threshold'], p['padding'],
                p['minimum_transcript_probability'], p['lexical']) == option
        assert point['split'] == 'selection'
        complete = read(training/arm/'complete.json')
        assert point['checkpoint_sha256'] == complete['checkpoint_sha256'] == sha(point['checkpoint'])
        assert point['selected_epoch'] == complete['selected_epoch']
        total, changes = Counter(), 0
        for sid in plan['partitions']['selection']:
            path = Path(point['raw_directory'])/(key(sid)+'.json')
            data = read(path)
            assert data['checkpoint_sha256'] == point['checkpoint_sha256']
            if str(path) not in raw_seen:
                check_raw(data, records[sid], p, checks)
                raw_seen.add(str(path))
                checks['selection_rgb_map_sources'] += 1
            rows, trace = apply_word_policy(data['rows'], p, lexicon)
            total.update(count(rows, records[sid]))
            changes += normalized_changes(trace)
        verify_total(total, point['counts'], len(plan['partitions']['selection']))
        assert changes == point['normalized_changes']
        checks['selection_policies'] += 1
    assert select_word_policy(points) == frozen['choice']
    assert frozen['promoted'] == (frozen['choice']['counts']['f1'] >= config['minimum_selection_f1'])
    if not args.selection_only:
        results = read(root/'selected_expert/results.json')
        assert results['frozen_policy_sha256'] == sha(root/'policy/frozen.json')
        test_dictionary = DeepSoloLexicon.from_files(config['test_vocabulary'],
            config['test_pairs'], distance_exclusive=1.5)
        chosen = frozen['choice']
        selection_maps = training/chosen['arm']/'selection'/(
            'epoch_'+str(chosen['selected_epoch']).zfill(3))/'maps'
        for role in config['inference_roles']:
            total = Counter()
            for sid in plan['partitions'][role]:
                data = read(root/'selected_expert'/role/'samples'/(key(sid)+'.json'))
                mp = root/'selected_expert'/role/'maps'/(key(sid)+'.npz')
                assert data['sample_id'] == sid
                assert data['policy_sha256'] == sha(root/'policy/frozen.json')
                check_raw(dict(data, rows=data['raw_rows'], maps_path=str(mp)), records[sid],
                          chosen['policy'], checks)
                with np.load(mp, allow_pickle=False) as z:
                    assert z['shared_context'].shape == (192,)
                    assert np.isfinite(z['shared_context']).all()
                    if role == 'selection':
                        with np.load(selection_maps/(key(sid)+'.npz'), allow_pickle=False) as previous:
                            assert np.array_equal(z['probabilities'], previous['probabilities'])
                        checks['selected_neural_map_exact_replays'] += 1
                rows, trace = apply_word_policy(data['raw_rows'], chosen['policy'],
                    test_dictionary if role == 'test' else lexicon)
                assert rows == data['rows'] and trace == data['trace']
                value = count(rows, records[sid])
                assert value == data['counts']
                total.update(value)
                checks['expert_image_counts'] += 1
            saved = results['roles'][role]
            verify_total(total, saved['counts'], len(plan['partitions'][role]))
            assert saved['deepsolo_images'] == 0 and saved['deepsolo_percentage'] == 0.
            assert saved['total_images'] == len(plan['partitions'][role])
        if not plan['smoke']:
            assert results['roles']['test']['total_images'] == 233
            assert results['roles']['test']['counts']['gt_words'] == 1095
        oracle = read(root/'oracle/results.json')
        for name, digest in oracle['source_sha256'].items():
            assert sha(name) == digest
        assert oracle['expert_results_sha256'] == sha(root/'selected_expert/results.json')
        assert oracle['fixed_deepsolo_counts_sha256'] == sha(config['fixed_deepsolo_counts'])
        teacher = read(config['fixed_deepsolo_counts'])
        normalize = lambda text: ''.join(c for c in text.casefold() if c.isalnum())
        for sid, branches in oracle['branches'].items():
            reference = teacher['samples'][sid]
            data = read(reference['evidence_path'])
            assert data['counts'] == branches['deepsolo'] == reference['counts']
            assert data['checkpoint_sha256'] == config['fixed_deepsolo_checkpoint_sha256']
            truth, _ = annotation(records[sid])
            assert data['ground_truth'] == truth
            care = [p for p in data['predictions'] if not p['ignored']]
            predicted = Counter(normalize(p['text']) for p in care
                                if normalize(p['text']) and not p['invalid_envelope'])
            target = Counter(normalize(g['text']) for g in truth if normalize(g['text']))
            tp = sum((predicted & target).values())
            expected = dict(tp=tp, fp=len(care)-tp, fn=len(truth)-tp,
                detected_words=len(care), gt_words=len(truth), all_outputs=len(data['predictions']),
                ignored_outputs=len(data['predictions'])-len(care))
            assert expected == branches['deepsolo']
            checks['fixed_deepsolo_evidence_counts'] += 1
        assert certify(oracle['branches'], list(oracle['branches']), oracle['oracle'],
                       min(config['deepsolo_budget'], len(oracle['branches'])), config['target_f1'])
        checks['rational_oracle_optimality_certificates'] += 1
    seal(args.output, dict(status='PASS', mode='approximate', checks=dict(checks),
        evaluation_plan_sha256=sha(root/'resolved_configuration.json'),
        selection_only=args.selection_only, actual_new_deepsolo_recognizer_calls=0,
        code_sha256=sha(__file__), scope='Independent word Counter totals, RGB/map/crop hashes, '
        'declared policy scope and lexical replay; full expert neural map replay and '
        'fixed fallback/oracle certificates when requested. Not an independent refit.'))
    print('PASS', dict(checks), flush=True)


if __name__ == '__main__':
    main()
