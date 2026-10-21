import copy
import json
from pathlib import Path

import pytest

from diagnostics.contextual_ocr_fewshot_train import (
    assign_folds, select_demonstrations, assert_disjoint, make_prompt, validate_config,
)


def cfg():
    return json.loads(Path('configs/approximate/contextual_ocr_fewshot_train_v1.json').read_text())


def population():
    samples, rows = [], []
    for i in range(30):
        sid = str(100+i)
        text = 'OPEN' if i % 2 == 0 else 'OPFN'
        samples.append({'sample_id': sid, 'image_sha256': str(i), 'caption': 'A shop.',
                        'blocks': [{'id': 0, 'candidate_id': sid+':0', 'text': text,
                                    'bbox': [0, 0, 10, 10]}]})
        rows.append({'sample_id': sid, 'gt_index': 0, 'gt_text': 'OPEN',
                     'available': True, 'best_iou': .8, 'significant_gt_words_in_crop': 1,
                     'crop_bbox': [0, 0, 10, 10], 'parseq_text': text})
    return samples, rows


def test_config_is_train_and_weight_training_forbidden():
    validate_config(cfg())
    for key, value in [('split', 'TEST'), ('weight_training', True), ('sample_count', 20)]:
        c = cfg()
        c[key] = value
        with pytest.raises(ValueError):
            validate_config(c)


def test_folds_deterministic_and_identical_images_together():
    samples, _ = population()
    samples[1]['image_sha256'] = samples[0]['image_sha256']
    folds = assign_folds(samples, 0)
    assert folds == assign_folds(list(reversed(samples)), 0)
    assert folds['100'] == folds['101']
    assert set(folds.values()) == set(range(5))


def test_demo_selection_balanced_disjoint_and_deterministic():
    samples, rows = population()
    folds = assign_folds(samples, 0)
    for fold in range(5):
        demos = select_demonstrations(samples, rows, folds, fold, cfg())
        assert [d['kind'] for d in demos] == ['keep', 'correct', 'keep', 'correct']
        assert len({d['image_sha256'] for d in demos}) == 4
        assert demos == select_demonstrations(samples[::-1], rows[::-1], folds, fold, cfg())
        for sample in samples:
            if folds[sample['sample_id']] == fold:
                assert_disjoint(sample, demos, folds)


def test_heldout_gt_not_read_when_selecting_examples():
    samples, rows = population()
    folds = assign_folds(samples, 0)
    expected = select_demonstrations(samples, rows, folds, 0, cfg())
    poisoned = [{'sample_id': r['sample_id']} if folds[r['sample_id']] == 0 else r for r in rows]
    assert select_demonstrations(samples, poisoned, folds, 0, cfg()) == expected


def test_low_iou_multiword_and_ambiguous_crops_excluded():
    samples, rows = population()
    folds = assign_folds(samples, 0)
    selected = select_demonstrations(samples, rows, folds, 0, cfg())
    ids = [d['sample_id'] for d in selected[:3]]
    for row in rows:
        if row['sample_id'] == ids[0]:
            row['best_iou'] = .1
        if row['sample_id'] == ids[1]:
            row['significant_gt_words_in_crop'] = 2
    duplicate = copy.deepcopy(next(r for r in rows if r['sample_id'] == ids[2]))
    duplicate['gt_index'] = 1
    rows.append(duplicate)
    actual = select_demonstrations(samples, rows, folds, 0, cfg())
    assert not set(ids) & {d['sample_id'] for d in actual}


def test_prompt_only_contains_other_fold_answers_not_target_gt():
    samples, rows = population()
    folds = assign_folds(samples, 0)
    sample = next(s for s in samples if folds[s['sample_id']] == 0)
    sample = {**sample, 'fold': 0, 'gt_text': 'FORBIDDEN_LABEL',
              'demonstrations': select_demonstrations(samples, rows, folds, 0, cfg())}
    for variant in cfg()['variants']:
        prompt = make_prompt(sample, sample['blocks'][0], variant, cfg())
        assert 'FORBIDDEN_LABEL' not in prompt
        assert prompt.count('Example ') == 4
        assert prompt.endswith('Corrected target:')
        assert ('Scene description (fallible):' in prompt) == (variant == 'ocr_caption')
        assert sample['blocks'][0]['candidate_id'] not in prompt
    sample['demonstrations'][0]['source_fold'] = 0
    with pytest.raises(RuntimeError):
        make_prompt(sample, sample['blocks'][0], 'ocr_only', cfg())


def test_same_image_hash_demo_rejected_even_with_wrong_fold_label():
    samples, rows = population()
    folds = assign_folds(samples, 0)
    sample = samples[0]
    demos = select_demonstrations(samples, rows, folds, folds[sample['sample_id']], cfg())
    demos[0]['image_sha256'] = sample['image_sha256']
    with pytest.raises(RuntimeError):
        assert_disjoint(sample, demos, folds)


def test_missing_examples_fail_instead_of_using_heldout_gt():
    samples, rows = population()
    folds = assign_folds(samples, 0)
    with pytest.raises(RuntimeError):
        select_demonstrations(samples, [r for r in rows if folds[r['sample_id']] == 0], folds, 0, cfg())
