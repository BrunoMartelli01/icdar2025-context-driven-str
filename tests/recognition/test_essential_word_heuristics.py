import numpy as np
import pytest

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.essential_word_heuristics import risk_scores, select_deepsolo


def test_budget_boundaries_and_label_independent_hash_ties():
    scores = dict(first=1., second=1., third=0.)
    keys = dict(first='z', second='a', third='b')
    assert select_deepsolo(scores, keys, 1) == ['second']
    assert select_deepsolo(scores, keys, 0) == []
    assert set(select_deepsolo(scores, keys, 3)) == set(scores)
    with pytest.raises(ValueError):
        select_deepsolo(scores, keys, 4)
    with pytest.raises(ValueError, match='Nonfinite'):
        select_deepsolo(dict(first=float('nan')), dict(first='a'), 1)


def test_empty_inference_has_finite_risks_without_gt_or_extra_encoder():
    lexicon = DeepSoloLexicon(['WORD'], {'WORD': 'Word'}, distance_exclusive=1.5)
    settings = dict(kernel_cut=.5, observations=dict(grid_side=1,
        map_cuts=[.5, .7], sequence_cuts=[.7, .9]))
    scores = risk_scores([], [], np.zeros((2, 8, 8), np.float32), [100, 50], lexicon, settings)
    assert len(scores) == 9 and all(np.isfinite(value) for value in scores.values())
    assert scores['weakest_sequence'] == 1. and scores['word_count'] == 0.
