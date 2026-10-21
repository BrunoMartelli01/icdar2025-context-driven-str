import pytest
from diagnostics.conservative_ocr_edit_gates_train import apply_gate, decision
from diagnostics.contextual_ocr_correction import lexical_counts, word_scores


@pytest.mark.parametrize('proposal', ['hello world', 'hello\tworld', 'hello\nworld', 'hello\u00a0world', 'hello '])
def test_any_whitespace_rejects_without_salvage(proposal):
    r = apply_gate('hello', proposal, 2)
    assert r['text'] == 'hello' and r['reason'] == 'rejected_whitespace'


def test_unit_cost_insert_delete_substitute_not_transposition():
    for original, proposed in [('DIGI', 'DIGIT'), ('BOOKS', 'BOOK'), ('OPFN', 'OPEN')]:
        r = apply_gate(original, proposed, 1)
        assert r['changed'] and r['edit_distance_casefold'] == 1
    assert not apply_gate('ab', 'ba', 1)['changed']
    assert apply_gate('ab', 'ba', 2)['changed']


def test_casefold_only_punctuation_is_not_removed():
    r = apply_gate('OPEN', 'open', 1)
    assert r['edit_distance_casefold'] == 0 and r['edit_distance_raw'] == 4
    assert r['text'] == 'open'
    assert apply_gate('(412', '(412)', 1)['changed']
    assert not apply_gate('x', '', 1)['changed']
    assert apply_gate('x', 'x', 1)['reason'] == 'unchanged'


def test_go_requires_benefit_and_token_budget():
    s = {'recovered': 2, 'damaged': 1, 'exact': 503, 'image_lexical_exact_matches': 504, 'predicted_tokens': 1203}
    assert decision(s) == 'GO_TRAIN_DIAGNOSTIC_ONLY'
    assert decision({**s, 'damaged': 2}) == 'NO-GO'
    assert decision({**s, 'predicted_tokens': 1204}) == 'NO-GO'


def test_lexical_duplicates_capped_and_not_geometric_assignment():
    s = {'sample_id': '100', 'blocks': [
        {'id': 0, 'candidate_id': 'a', 'bbox': [0, 0, 2, 2], 'text': 'dog'},
        {'id': 1, 'candidate_id': 'b', 'bbox': [3, 0, 5, 2], 'text': 'cat'}]}
    gt = [{'sample_id': '100', 'gt_index': 0, 'crop_bbox': [0, 0, 2, 2], 'available': True,
           'parseq_text': 'dog', 'gt_text': 'cat', 'exact': False},
          {'sample_id': '100', 'gt_index': 1, 'crop_bbox': [3, 0, 5, 2], 'available': True,
           'parseq_text': 'cat', 'gt_text': 'dog', 'exact': False}]
    assert sum(r['exact'] for r in word_scores(s, gt, {0: 'dog', 1: 'cat'})) == 0
    assert lexical_counts(['dog', 'cat'], ['cat', 'dog'])['matches'] == 2
    assert lexical_counts(['cat', 'cat'], ['cat'])['matches'] == 1
