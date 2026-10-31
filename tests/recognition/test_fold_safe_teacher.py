import copy
import hashlib
import json

import pytest

from recognition.deepsolo_lexicon import DeepSoloLexicon
from recognition.fold_safe_teacher import canonical_vocabulary, checked_response, correct_detections


def test_dictionary_rejects_out_of_scope_labels_and_preserves_literal_spaces():
    words = canonical_vocabulary(['hello'], ['fit'], {'fit': ['Two Words']})
    assert words == {'HELLO': 'hello', 'TWO WORDS': 'Two Words'}
    with pytest.raises(ValueError, match='declared fitting'):
        canonical_vocabulary(['hello'], ['fit'], {'fit': ['seen'], 'heldout': ['secret']})


def test_raw_payload_binding_image_and_failure_are_enforced():
    binding = dict(plan_sha256='plan', profile_sha256='profile', code_sha256='code')
    data = dict(contract=dict(binding, sample_id='image1', image_sha256='rgb'),
                response=dict(success=True, detections=[]))
    data['payload_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    assert checked_response(data, binding, 'image1', 'rgb')['success']
    broken = copy.deepcopy(data)
    broken['response']['success'] = False
    with pytest.raises(ValueError, match='payload'):
        checked_response(broken, binding, 'image1', 'rgb')
    broken['payload_sha256'] = hashlib.sha256(json.dumps(
        {k: v for k, v in broken.items() if k != 'payload_sha256'}, sort_keys=True).encode()).hexdigest()
    with pytest.raises(RuntimeError, match='inference failed'):
        checked_response(broken, binding, 'image1', 'rgb')
    with pytest.raises(ValueError, match='identity'):
        checked_response(data, binding, 'image2', 'rgb')
    with pytest.raises(ValueError, match='binding'):
        checked_response(data, dict(binding, plan_sha256='other'), 'image1', 'rgb')


def test_teacher_keeps_unmatched_words_and_crop_geometry_at_inclusive_score_cut():
    lexicon = DeepSoloLexicon(['HELLO'], {'HELLO': 'Hello'}, distance_exclusive=1.5)
    response = dict(success=True, detections=[
        dict(source_index=2, text='unknown', score=.4, boundary=[[1., 2.]],
             raw=dict(recognition_class_probabilities=[[.9, .1]])),
        dict(source_index=1, text='HEL1O', score=.9, boundary=[[3., 4.]],
             raw=dict(recognition_class_probabilities=[[.9, .1]]))])
    before = copy.deepcopy(response)
    rows, trace = correct_detections(response, lexicon,
        dict(score_threshold=.4, recognition_threshold=0.), 'keep_unmatched')
    assert [row['text'] for row in rows] == ['Hello', 'unknown']
    assert rows[0]['boundary'] == [[3., 4.]] and response == before
    assert all(t['accepted'] for t in trace)
    with pytest.raises(ValueError, match='Duplicate'):
        correct_detections(dict(detections=response['detections']*2), lexicon,
            dict(score_threshold=.4, recognition_threshold=0.), 'keep_unmatched')
