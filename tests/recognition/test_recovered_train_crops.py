import json
import pytest
from diagnostics.recognize_recovered_train_crops import selected_words, scores, read_cached, digest


def test_selection_excludes_correct_incomplete_or_contaminated_oracle():
    row = {'exact': False, 'actual': {'all_own_glyph_pixels_inside': False},
           'oracle': {'all_own_glyph_pixels_inside': True, 'other_care_pixels': 0, 'dont_care_pixels': 0}}
    assert selected_words([row]) == [row]
    assert selected_words([{**row, 'exact': True}]) == []
    assert selected_words([{**row, 'oracle': {**row['oracle'], 'other_care_pixels': 1}}]) == []
    assert selected_words([{**row, 'oracle': {**row['oracle'], 'all_own_glyph_pixels_inside': False}}]) == []


def test_metrics_casefold_and_missing():
    assert scores('Word', 'word')['exact_casefold']
    assert not scores(None, 'word')['fuzzy_ge_0_8']


def test_resume_fails_on_corruption_or_wrong_contract(tmp_path):
    path = tmp_path/'cache.json'
    request = {'bbox': [0, 0, 2, 2]}
    assert read_cached(path, 'abc', request) is None
    payload = {'contract': 'abc', 'request': request, 'result': {'raw_text': 'OK'}}
    path.write_text(json.dumps({'sha256': digest(payload), 'payload': payload}))
    assert read_cached(path, 'abc', request)['raw_text'] == 'OK'
    with pytest.raises(RuntimeError): read_cached(path, 'different', request)
    payload['result']['raw_text'] = 'changed'
    path.write_text(json.dumps({'sha256': 'wrong', 'payload': payload}))
    with pytest.raises(RuntimeError): read_cached(path, 'abc', request)
