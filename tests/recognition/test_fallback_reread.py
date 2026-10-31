import math
import pytest

from recognition.fallback_reread import padded_box, choose_transcript
from recognition.fallback_reread import reread_rows


def test_padding_is_exclusive_clamped_and_does_not_drop_invalid_boxes():
    assert padded_box([0, 1, 9, 10], [10, 12], 4) == [0, 0, 10, 12]
    with pytest.raises(ValueError, match='exclusive'):
        padded_box([4, 1, 4, 10], [10, 12], 0)


def test_reread_keeps_original_on_uncertainty_or_empty_text():
    policy = dict(enabled=True, padding=2, confidence_kind='sequence',
                  threshold=.9, require_lexical_match=False)
    detection = dict(text='cat')
    assert choose_transcript(detection, {'2': dict(raw_text='dog',
        sequence_confidence_native=.8)}, None, policy)[0] == 'cat'
    assert choose_transcript(detection, {'2': dict(raw_text='',
        sequence_confidence_native=1.)}, None, policy)[0] == 'cat'


def test_reread_can_replace_without_mutating_instances_or_readings():
    policy = dict(enabled=True, padding=0, confidence_kind='balanced',
                  threshold=.9, require_lexical_match=False)
    detection = dict(text='cqt', source_index=3)
    reading = dict(raw_text='cat', token_probabilities=[.99]*3,
                   token_log_probabilities=[math.log(.99)]*3)
    text, trace = choose_transcript(detection, {'0': reading}, None, policy)
    assert text == 'cat' and trace['changed']
    assert detection == dict(text='cqt', source_index=3)


def test_missing_invalid_envelope_retains_fallback_transcript():
    policy = dict(enabled=True, padding=0, confidence_kind='sequence',
                  threshold=.9, require_lexical_match=False)
    assert choose_transcript(dict(text='cat'), {}, None, policy)[0] == 'cat'


def test_row_replacement_preserves_geometry_and_requires_actual_reading():
    row=dict(instance_id=2,text='cqt',raw_text='cqt',bbox=[0,0,10,10],ignored=False)
    policy=dict(enabled=True,padding=0,confidence_kind='sequence',threshold=.9,require_lexical_match=False)
    result=reread_rows([row],{'2':{'0':dict(raw_text='cat',sequence_confidence_native=.99)}},None,policy)
    assert result[0]['raw_text']=='cat' and result[0]['bbox']==row['bbox'] and result[0]['ignored'] is False
    assert row['raw_text']=='cqt'
    with pytest.raises(ValueError,match='Missing actual'):
        reread_rows([row],{},None,policy)
