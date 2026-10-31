import copy

import numpy as np
import pytest

from recognition.residual_cost_router import (
    NEW_FAMILIES, predicted_probabilities, residual_regions, routing_scores, word_evidence,
)


SETTINGS = {'probability_threshold': .5, 'minimum_component_area': 4,
            'minimum_uncovered_fraction': .5, 'minimum_other_head_support': .25}


def maps():
    p = np.zeros((2, 10, 10), np.float32)
    p[:, 1:3, 1:3] = 1
    p[:, 1:3, 5:7] = 1
    p[:, 6:8, 5:7] = 1
    return p


def test_missing_regions_count_words_and_mass_without_double_counting_heads():
    p = maps(); p.setflags(write=False)
    r = residual_regions(p, [10, 10], [[1, 1, 3, 3]], SETTINGS)
    assert r['missing_count'] == r['missing_mass'] == r['missing_consensus'] == 2
    assert r['heads']['text']['coverage'] == pytest.approx(1/3)
    duplicated = residual_regions(p, [10, 10], [[1, 1, 3, 3]]*2, SETTINGS)
    assert duplicated['missing_count'] == duplicated['missing_mass'] == 2


def test_small_exposed_box_edge_is_not_a_missing_word():
    p = np.zeros((2, 10, 10)); p[:, 2:5, 2:5] = 1
    r = residual_regions(p, [10, 10], [[2, 2, 4, 5]], SETTINGS)
    assert r['heads']['kernel']['uncovered_pixels'] == 3
    assert r['missing_count'] == r['missing_mass'] == r['missing_consensus'] == 0


def test_padding_canvas_is_not_text_and_geometry_is_exclusive():
    p = np.zeros((2, 10, 10)); p[:, :, :3] = 1
    r = residual_regions(p, [4, 10], [[0, 0, 4, 10]], SETTINGS)
    assert r['canvas_pixels'] == 40
    assert r['heads']['text']['foreground_pixels'] == 0
    assert r['missing_consensus'] == 0
    with pytest.raises(ValueError, match='clipped'):
        residual_regions(p, [4, 10], [[0, 0, 5, 10]], SETTINGS)


def test_single_head_noise_does_not_count_as_consensus():
    p = maps(); p[1] = 0
    r = residual_regions(p, [10, 10], [[1, 1, 3, 3]], SETTINGS)
    assert r['missing_count'] == 2 and r['missing_consensus'] == 0


def test_component_exposure_and_cross_head_support_boundaries_are_inclusive():
    p = np.zeros((2, 10, 10)); p[0, 2:4, 2:6] = 1; p[1, 2, 4] = 1
    r = residual_regions(p, [10, 10], [[2, 2, 4, 4]], SETTINGS)
    c = r['heads']['text']['components'][0]
    assert c['uncovered_fraction'] == .5 and c['other_head_support'] == .25
    assert c['residual'] and c['corroborated']
    strict = residual_regions(p, [10, 10], [[2, 2, 4, 4]],
        {**SETTINGS, 'minimum_other_head_support': .25001})
    assert not strict['heads']['text']['components'][0]['corroborated']


def test_empty_maps_are_neutral_and_empty_word_outputs_abstain():
    r = residual_regions(np.zeros((2, 10, 10)), [10, 10], [], SETTINGS)
    assert r['missing_count'] == r['missing_mass'] == 0
    assert r['heads']['kernel']['coverage'] == 1
    e = {'predicted_word_count': 0, 'has_empty_reading': False}
    assert set(routing_scores(r, e, {}).values()) == {None}


def test_npz_loader_never_accesses_annotations(monkeypatch):
    class Archive:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def __getitem__(self, name):
            assert name == 'probabilities', 'Annotation array was accessed'
            return maps()
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: Archive())
    assert predicted_probabilities('unused').shape == (2, 10, 10)


def test_absolute_error_risk_distinguishes_word_counts_at_identical_mean_quality():
    region = {'missing_count': 0, 'missing_mass': 0, 'missing_consensus': 0}
    base = {'all_checks_softcrop': .9, 'base_proxy_f1': .9}
    def scores(n):
        return routing_scores(region, {'predicted_word_count': n, 'has_empty_reading': False,
            'native_error_proxy': n*.1, 'visual_error_proxy': n*.1}, base)
    assert scores(1)['risk_visual_consensus'] > scores(10)['risk_visual_consensus']
    assert scores(1)['f1_consensus_proxy'] == pytest.approx(scores(10)['f1_consensus_proxy'])
    assert set(scores(1)) == set(NEW_FAMILIES)


def test_word_evidence_uses_only_frozen_inference_signals_and_preserves_rows():
    rows = [{'candidate_id': 'a', 'confidence': .8, 'clip_word_advantage': .5, 'raw_text': 'OPEN'}]
    con = {'crop_stability': {'words': [{'candidate_id': 'a', 'minimum_edit_agreement': .5}]}}
    before = copy.deepcopy(rows)
    e = word_evidence(rows, con, [0., 1.])
    assert e['words'][0]['visual'] == pytest.approx((.8*.5*.5)**(1/3))
    assert e['ground_truth_used'] is False and not e['probabilities_calibrated']
    assert rows == before
    with pytest.raises(ValueError, match='population'):
        word_evidence(rows, {'crop_stability': {'words': []}}, [0., 1.])


def test_invalid_prediction_maps_fail_explicitly():
    p = maps(); p[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='Invalid prediction'):
        residual_regions(p, [10, 10], [], SETTINGS)
