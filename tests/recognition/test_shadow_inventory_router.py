"""Scientific boundaries for independent-inventory whole-image routing."""
from copy import deepcopy

import pytest

from recognition.shadow_inventory_router import (
    area, geometric_pairs, inventory_features, probe_indices, quality, union_coverage,
)

SETTINGS = {'matching_iou': .25, 'maximum_probes': 1, 'covered_uncertainty_weight': .25}


def light(box=(0, 0, 10, 10), text='WORD', confidence=.9):
    return {'bbox': list(box), 'raw_text': text, 'confidence': confidence}


def region(index=0, box=(0, 0, 10, 10), score=.9):
    return {'source_index': index, 'bbox': list(box), 'score': score, 'valid': True}


def ocr(index=0, text='WORD', p=.9):
    return [{'source_index': index, 'candidate_id': f'{index}:{view}', 'view': view,
             'raw_text': text, 'token_probabilities': [p]*max(1, len(text))}
            for view in ('axis_aligned', 'rectified')]


def test_union_does_not_double_count_overlapping_rectangles():
    assert union_coverage([0, 0, 10, 10], [[0, 0, 8, 10], [2, 0, 10, 10]]) == 1
    assert union_coverage([0, 0, 10, 10], [[-2, -2, 5, 5], [4, 4, 8, 8]]) == .4
    assert union_coverage([0, 0, 10, 10], []) == 0


@pytest.mark.parametrize('box', [[0, 0, 0, 1], [0, 0, float('nan'), 1], [2, 2, 1, 1]])
def test_invalid_boxes_fail(box):
    with pytest.raises(ValueError):
        area(box)


def test_spatial_matching_is_one_to_one_and_inclusive():
    p = geometric_pairs([light(), light()], [region()], 1.)
    assert len(p) == 1 and p[0]['iou'] == 1
    assert geometric_pairs([light()], [region(box=(20, 20, 30, 30))], .25) == []


def test_character_confidence_keeps_symbols_and_zeros():
    assert quality(ocr(text='!', p=.8)[0]) == pytest.approx(.8)
    assert quality(ocr(text=' ', p=.8)[0]) == 0
    assert quality(ocr(text='X', p=0)[0]) == 0
    with pytest.raises(ValueError):
        quality(ocr(p=float('nan'))[0])


def test_perfect_inventory_has_no_omission_and_preserves_casefold():
    f, scores, evidence = inventory_features([light(text='Word')], [region()], ocr(text='word'), SETTINGS)
    assert f['shadow_bag_dice'] == 1 and f['shadow_spatial_lexical_agreement'] == 1
    assert f['shadow_log_confirmed_missing'] == f['shadow_log_missing_geometry'] == 0
    assert scores['geometry'] == pytest.approx(-.2)
    assert evidence['probe_region_ids'] == [0]
    assert not evidence['ground_truth_used'] and not evidence['chapman_is_calibrated_count']


def test_probe_is_selected_before_ocr_and_bounded():
    a = [light()]; regions = [region(), region(1, (20, 20, 30, 30))]
    assert probe_indices(a, regions, SETTINGS) == [1]
    f, scores, evidence = inventory_features(a, regions, ocr()+ocr(1, 'NEW'), SETTINGS)
    assert evidence['probe_region_ids'] == [1] and f['shadow_log_confirmed_missing'] > 0
    _, changed, _ = inventory_features(a, regions, ocr(text='WRONG')+ocr(1, 'NEW'), SETTINGS)
    assert scores['probe'] == changed['probe']
    assert scores['geometry'] == changed['geometry'] and scores['recapture'] == changed['recapture']


def test_punctuation_is_not_silently_normalized_away():
    f, _, _ = inventory_features([light(text='WORD!')], [region()], ocr(text='WORD'), SETTINGS)
    assert f['shadow_bag_dice'] == 0 and f['shadow_log_disagreement'] > 0


def test_duplicate_words_are_multiset_matches():
    f, _, _ = inventory_features([light(), light((20, 0, 30, 10))], [region()], ocr(), SETTINGS)
    assert f['shadow_bag_dice'] == pytest.approx(2/3)


def test_empty_reading_is_ineligible_but_missing_diagnostic_survives():
    f, scores, _ = inventory_features([light(text='')], [region()], ocr(), SETTINGS)
    assert all(v is None for v in scores.values()) and f['shadow_log_disagreement'] > 0
    f, scores, _ = inventory_features([], [], [], SETTINGS)
    assert all(v is None for v in scores.values()) and f['shadow_log_count'] == 0


def test_raw_view_population_and_order_fail_closed():
    with pytest.raises(ValueError):
        inventory_features([light()], [region()], ocr()[:1], SETTINGS)
    with pytest.raises(ValueError):
        inventory_features([light()], [region()], list(reversed(ocr())), SETTINGS)
    with pytest.raises(ValueError):
        inventory_features([light()], [region()], [], SETTINGS)


def test_recapture_depends_on_spatial_overlap_and_never_claims_calibration():
    f, _, evidence = inventory_features([light()], [region(box=(20, 20, 30, 30))], ocr(), SETTINGS)
    assert f['shadow_log_recapture_unseen'] > 0
    assert evidence['chapman_is_calibrated_count'] is False
