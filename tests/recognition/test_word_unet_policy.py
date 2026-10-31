import copy
import math

import pytest

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.word_unet_policy import apply_word_policy, select_word_policy


def test_boundary_uses_native_eos_sequence_probability_and_preserves_uncertainty():
    lexicon = DeepSoloLexicon(['HELLO'], {'HELLO': 'Hello'}, distance_exclusive=1.5)
    raw = [dict(raw_text='HEL1O', sequence_confidence_native=.7,
                token_probabilities=[.95]*5, detector_score=.8, candidate_id='one'),
           dict(raw_text='HELLO', sequence_confidence_native=.699999,
                token_probabilities=[.999]*5, detector_score=.9, candidate_id='two'),
           dict(raw_text='...', sequence_confidence_native=.99,
                token_probabilities=[.999]*3, detector_score=.9, candidate_id='three')]
    for row in raw:
        row['token_log_probabilities'] = [math.log(p) for p in row['token_probabilities']]
    original = copy.deepcopy(raw)
    policy = dict(minimum_transcript_probability=.7, lexical='keep')
    rows, trace = apply_word_policy(raw, policy, lexicon)
    assert len(rows) == 1 and rows[0]['raw_text'] == 'Hello'
    assert rows[0]['sequence_confidence_native'] == .7
    assert rows[0]['token_probabilities'] == [.95]*5
    assert rows[0]['candidate_id'] == 'one' and rows[0]['confidence'] < 1
    assert trace[0]['original_text'] == 'HEL1O' and raw == original


def test_unmatched_rejection_is_an_explicit_policy_not_metric_denominator_change():
    lexicon = DeepSoloLexicon(['HELLO'], {'HELLO': 'Hello'}, distance_exclusive=1.5)
    raw = [dict(raw_text='unlistedword', sequence_confidence_native=.9,
                token_probabilities=[.99]*12, token_log_probabilities=[math.log(.99)]*12,
                detector_score=.9)]
    kept, _ = apply_word_policy(raw, dict(minimum_transcript_probability=.7, lexical='keep'), lexicon)
    rejected, trace = apply_word_policy(raw, dict(minimum_transcript_probability=.7, lexical='reject'), lexicon)
    assert len(kept) == 1 and rejected == [] and not trace[0]['accepted']


def test_selection_refuses_test_results_and_ties_favor_fewer_edits_then_order():
    points = [dict(split='selection', counts=dict(f1=.8), normalized_changes=2, label='first'),
              dict(split='selection', counts=dict(f1=.8), normalized_changes=1, label='second'),
              dict(split='selection', counts=dict(f1=.8), normalized_changes=1, label='third')]
    assert select_word_policy(points)['label'] == 'second'
    with pytest.raises(ValueError, match='selection'):
        select_word_policy(points+[dict(split='test', counts=dict(f1=1.), normalized_changes=0)])
