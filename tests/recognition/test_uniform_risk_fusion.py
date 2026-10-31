import pytest

from recognition.uniform_risk_fusion import route, combine
from recognition.fixed_expert_budget import fixed_heuristic


def test_expected_errors_charges_both_wrong_kept_and_correct_rejected_words():
    raw = [{'source_row': i} for i in range(3)]
    emitted = [{'source_row': 0}, {'source_row': 2}]
    assert fixed_heuristic(raw, emitted, [.9, .8, .6], 'expected_errors') == pytest.approx(-1.3)


def test_batch_route_obeys_budget_and_rgb_ties_without_fallback_predictions():
    observations = {s: dict(rows=[{}], score=1.) for s in ['a', 'b', 'c']}
    result = route(observations, dict(a='z', b='x', c='y'), 1, False)
    assert result['accepted_ids'] == ['b', 'c'] and result['fallback_ids'] == ['a']
    assert result['deepsolo_images'] == 1 and result['total_images'] == 3
    assert result['deepsolo_percentage'] == pytest.approx(100/3)
    assert not result['ground_truth_used'] and not result['fallback_outputs_loaded']


def test_guard_does_not_silently_change_requested_quota():
    observations = dict(a=dict(rows=[], score=0.), b=dict(rows=[], score=0.))
    with pytest.raises(ValueError, match='Guard cannot satisfy'):
        route(observations, dict(a='a', b='b'), 1, True)


def test_post_fallback_replacement_preserves_heavy_geometry_and_complements_distinct_word():
    heavy = [dict(raw_text='sh0p', text='sh0p', bbox=[0, 0, 10, 10], ignored=False)]
    light = [dict(raw_text='shop', bbox=[0, 0, 10, 10], word_probability=.99, source_row=0),
             dict(raw_text='open', bbox=[20, 0, 30, 10], word_probability=.99, source_row=1)]
    out = combine(dict(rows=light), heavy,
        dict(enabled=True, minimum_word_probability=.9, minimum_iou=.5, maximum_edit_distance=1),
        dict(confidence=.95, maximum_overlap=.1))
    assert len(out) == 2 and out[0]['raw_text'] == 'shop' and out[1]['raw_text'] == 'open'
    assert out[0]['bbox'] == heavy[0]['bbox'] and out[0]['original_deepsolo_text'] == 'sh0p'
    assert heavy[0]['raw_text'] == 'sh0p'
