"""Invariants for controlled approximate router cause tests."""
from collections import Counter

import numpy as np
from PIL import Image
import pytest
from transformers import CLIPImageProcessor

from recognition.router_cause_tests import (word_views, pool_views, literal_foils, row_supervision,
    choose_omissions, omit_features, fixed_budget, weighted_auc, decisions)
from scripts.router_cause_training import fitting_subset, intervention_features
from scripts.router_cause_gpu import processor_boundary_check


SETTINGS = dict(image_size=224, padding_rgb=[0, 0, 0], window_stride_fraction=.75,
                variants=['pad', 'stretch', 'windows'])


@pytest.mark.parametrize('shape', [(3, 19), (19, 3), (1, 9), (9, 1), (7, 7)])
def test_windows_cover_every_original_pixel(shape):
    h, w = shape; values = np.arange(h*w, dtype=np.uint8).reshape(h, w)
    rgb = np.repeat(values[..., None], 3, 2); views = word_views(rgb, 'windows', SETTINGS)
    assert all(v.shape[0] == v.shape[1] == min(h, w) for v in views)
    assert set(np.concatenate([v[..., 0].flatten() for v in views])) == set(values.flatten())


def test_square_padding_preserves_all_original_pixels():
    rgb = np.arange(3*7*3, dtype=np.uint8).reshape(3, 7, 3)
    padded = word_views(rgb, 'pad', SETTINGS)[0]
    np.testing.assert_array_equal(padded[2:5], rgb)
    assert padded.shape == (7, 7, 3)


def test_actual_native_processor_removes_boundaries_and_complete_views_retain_them():
    processor = CLIPImageProcessor(size={'shortest_edge': 224}, crop_size={'height': 224, 'width': 224})
    result = processor_boundary_check(processor, SETTINGS)
    assert result['native']['left_marker_pixels'] == result['native']['right_marker_pixels'] == 0
    for kind in SETTINGS['variants']:
        assert result[kind]['left_marker_pixels'] > 0 and result[kind]['right_marker_pixels'] > 0


def test_view_pooling_normalizes_and_rejects_degeneracy():
    np.testing.assert_allclose(pool_views(np.array([[1., 0.], [0., 1.]], 'f')), [2**-.5]*2)
    with pytest.raises(ValueError):
        pool_views(np.zeros((2, 2), 'f'))


def test_character_foils_change_only_the_predeclared_position():
    text = 'COFFEE'; value = literal_foils(text, 'ABCDEF')
    for name, pos in [('prefix', 0), ('middle', 3), ('suffix', 5)]:
        assert len(value[name]) == len(text)
        assert [i for i, (a, b) in enumerate(zip(text, value[name])) if a != b] == [pos]


def row(text, confidence=.9, box=(1, 1, 8, 6), index=0):
    return dict(raw_text=text, confidence=confidence, detector_score=.8, bbox=list(box), candidate_id=str(index))


def test_primary_and_spatial_auxiliary_labels_have_declared_different_matching(tmp_path):
    path = tmp_path/'valid.png'; Image.new('L', (32, 16), 255).save(path)
    sample = dict(image_wh=[32, 16], validity_path=str(path))
    value = row_supervision(sample, [row('Cafe')], [dict(text='CAFE', bbox=[20, 9, 30, 15])], 256, .5)
    assert value['primary'].tolist() == [1.] and value['auxiliary'].tolist() == [0.]


def test_dontcare_predictions_do_not_become_primary_negative_labels(tmp_path):
    path = tmp_path/'valid.png'; Image.new('L', (32, 16), 0).save(path)
    sample = dict(image_wh=[32, 16], validity_path=str(path))
    value = row_supervision(sample, [row('CAFE')], [], 256, .5)
    assert value['primary'].tolist() == [-1.]
    assert value['metrics']['ignored_outputs'] == 1 and value['metrics']['fp'] == 0


def test_omission_excludes_correct_assignment_with_a_surplus_duplicate():
    rows = [row('A', index=0), row('A', index=1), row('B', index=2), row('C', index=3)]
    result = choose_omissions({'primary': [1., 0., 1., 1.]}, rows)
    for deleted in result.values():
        assert not set(deleted)&{0, 1}
        before = Counter(r['raw_text'] for r in rows); after = Counter(r['raw_text'] for i, r in enumerate(rows) if i not in deleted)
        gt = Counter('ABC')
        assert sum((before & gt).values())-sum((after & gt).values()) == len(deleted)


def test_feature_omission_changes_words_and_counts_but_keeps_original_image_tokens():
    rows = [row('A', index=0), row('B', index=1)]
    f = dict(word_text=np.ones((2, 8)), word_image=np.ones((2, 8)), word_geometry=np.ones((2, 12)),
             patches=np.ones((7, 8)), queries=np.ones((3, 6)), scalars=np.zeros(18))
    value, remaining = omit_features(f, rows, [0], [32, 16])
    assert value['patches'] is f['patches'] and value['queries'] is f['queries']
    assert len(value['word_text']) == len(remaining) == 1
    assert len(f['word_text']) == len(rows) == 2
    assert value['scalars'][0] == pytest.approx(np.log(2))


def test_fixed_budget_is_invariant_under_positive_affine_calibration():
    scores = {'a': -4., 'b': 2., 'c': 2., 'd': None}; ids = sorted(scores)
    expected = fixed_budget(scores, ids, .5)
    for slope in [.1, 1., 10.]:
        for offset in [-10., 0., 10.]:
            actual = fixed_budget({s: None if v is None else slope*v+offset for s, v in scores.items()}, ids, .5)
            assert actual == expected


def test_learning_curve_subsets_preserve_rgb_groups():
    samples = {'a': dict(rgb_sha256='one'), 'b': dict(rgb_sha256='one'), 'c': dict(rgb_sha256='two')}
    selected = fitting_subset(list(samples), samples, .5, 7)
    assert ('a' in selected) == ('b' in selected)


def test_context_donor_is_fit_only_and_target_words_stay_unchanged():
    fields = ('patches', 'patch_geometry', 'patch_valid', 'patch_mass', 'queries')
    values = {s: {**{f: np.array([i]) for f in fields}, 'word_text': np.array([100+i])} for i, s in enumerate(['fit1', 'fit2', 'held'])}
    samples = {s: dict(rgb_sha256=s) for s in values}
    new, donor = intervention_features(values, dict(crop='native', shuffle_context=True), {}, ['fit1', 'fit2'], samples, 1)
    assert donor['held'] in ['fit1', 'fit2']
    assert new['held']['word_text'] is values['held']['word_text']
    assert new['held']['patches'] is values[donor['held']]['patches']


def test_auc_does_not_invent_a_number_when_resampling_leaves_one_class():
    assert weighted_auc([0, 1], [0., 1.]) == 1.
    assert weighted_auc([0, 1], [0., 1.], [1., 0.]) is None


def test_interval_decisions_do_not_turn_uncertainty_into_a_causal_claim():
    assert decisions(.1, -.2, .3) == 'inconclusive'
    assert decisions(.1, .01, .3) == 'positive_effect_supported'
    assert decisions(-.1, -.3, -.01) == 'negative_effect_supported'
