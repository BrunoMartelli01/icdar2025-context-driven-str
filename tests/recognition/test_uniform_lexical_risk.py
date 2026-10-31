import pytest

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.uniform_lexical_risk import prepare_words


def row(text, probability, confidence=.99):
    return dict(raw_text=text, confidence=confidence, sequence_confidence_native=probability,
                bbox=[0, 0, 10, 10])


def policy(**changes):
    return dict(dict(enabled=True, confidence_cap=1.01, protect_numeric=True,
                     unmatched_policy='source_reject', minimum_sequence_probability=.4), **changes)


def test_recover_lower_confidence_dictionary_word_without_changing_source_indices():
    lex = DeepSoloLexicon(['SHOP'], {'SHOP': 'Shop'}, distance_exclusive=1.5)
    raw = [row('unknown', .99), row('sh0p', .5)]
    words, trace = prepare_words(raw, lex, policy())
    assert len(words) == 1 and words[0]['raw_text'] == 'Shop' and words[0]['source_row'] == 1
    assert trace[1]['changed'] and not trace[0]['accepted']
    assert raw[1]['raw_text'] == 'sh0p'


def test_sequence_rejection_does_not_leave_trace_accepted_after_lexical_match():
    lex = DeepSoloLexicon(['SHOP'], {'SHOP': 'Shop'}, distance_exclusive=1.5)
    words, trace = prepare_words([row('SHOP', .2)], lex, policy())
    assert words == [] and not trace[0]['accepted']


def test_missing_dictionary_is_explicit_and_numeric_protection_is_preserved():
    with pytest.raises(ValueError, match='explicit vocabulary'):
        prepare_words([], None, policy())
    lex = DeepSoloLexicon(['SHOP'], {'SHOP': 'Shop'}, distance_exclusive=1.5)
    words, _ = prepare_words([row('123', .9)], lex, policy())
    assert words[0]['raw_text'] == '123'
