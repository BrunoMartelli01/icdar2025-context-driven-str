import numpy as np
import pytest

from recognition.fixed_expert_budget import (
    fixed_trace, training_word_labels, observed_features, fixed_words_with_risk,
    conditional_complement, fixed_heuristic,
)


def raw_word(text='cat', bbox=(0, 0, 10, 5)):
    return dict(raw_text=text, bbox=list(bbox), confidence=.9, detector_score=.8,
        token_probabilities=[.9]*len(text), sequence_confidence_native=.8)


def test_fixed_emission_alignment_rejects_duplicate_or_unknown_sources():
    raw = [raw_word()]
    with pytest.raises(ValueError, match='distinct'):
        fixed_trace(raw, [dict(raw[0], source_row=0)]*2)
    with pytest.raises(ValueError, match='valid'):
        fixed_trace(raw, [dict(raw[0], source_row=1)])
    with pytest.raises(ValueError, match='geometry'):
        fixed_trace(raw, [dict(raw[0], bbox=[1, 0, 10, 5], source_row=0)])


def test_word_labels_describe_corrected_emission_without_changing_words():
    raw = [raw_word('cqt'), raw_word('dog')]
    selected = [dict(raw[0], raw_text='cat', source_row=0)]
    np.testing.assert_array_equal(training_word_labels(raw, selected,
        [dict(text='cat'), dict(text='dog')]), [1, 1])
    assert raw[0]['raw_text'] == 'cqt'


def test_observation_requires_no_gt_and_marks_missing_maps():
    raw = [raw_word()]; selected = [dict(raw[0], source_row=0)]
    trace = fixed_trace(raw, selected)
    x, groups = observed_features(raw, selected, trace, [20, 20], None)
    assert x.shape == (1, 18)
    np.testing.assert_array_equal(groups['maps'], np.zeros(14))
    empty_but_observed = np.zeros(14); empty_but_observed[-2] = 1
    _, actual = observed_features(raw, selected, trace, [20, 20], empty_but_observed)
    assert actual['maps'][-2] == 1
    with pytest.raises(ValueError, match='real predicted'):
        observed_features(raw, selected, trace, [20, 20], np.zeros(14))


def test_complement_retains_all_fallback_words_and_fixed_expert_is_unchanged():
    selected = [dict(raw_word('dog', (20, 0, 30, 5)), source_row=0)]
    light = fixed_words_with_risk(selected, [.99])
    fallback = [raw_word('cat')]
    out = conditional_complement(light, fallback, dict(confidence=.95, maximum_overlap=.1))
    assert [r['raw_text'] for r in out] == ['cat', 'dog']
    assert len(selected) == len(fallback) == 1
    assert 'word_probability' not in selected[0]


def test_risk_routes_include_rejected_candidate_risk_and_empty_images():
    raw = [raw_word(), raw_word('dog')]
    selected = [dict(raw[0], source_row=0)]
    assert fixed_heuristic(raw, selected, [.9, .8], 'expected_errors') == pytest.approx(-.9)
    assert fixed_heuristic([], [], [], 'weakest_word') == 0.
    assert fixed_heuristic(raw, selected, [.9, .8], 'weakest_sequence') == .8
