import json

import pytest

from recognition.fixed_expert_deployment import FixedExpertPipeline
from tests.recognition.test_fixed_expert_budget import raw_word


def pipeline(tmp_path, guard=False):
    (tmp_path/'resolved_configuration.json').write_text(json.dumps(dict(mode='approximate')))
    winner = dict(models=[], groups=[], heuristic='weakest_sequence', guard=guard)
    choice = dict(view='seen', target_name='native', winner=winner,
                  complement=dict(confidence=None, maximum_overlap=.1))
    (tmp_path/'selection_frozen.json').write_text(json.dumps(dict(status='FROZEN',
        test_used_for_selection=False, artifact_sha256={}, winner=choice)))
    return FixedExpertPipeline(tmp_path, verify_qualification=False)


def test_deployment_scores_and_routes_without_labels_maps_or_word_model(tmp_path):
    p = pipeline(tmp_path)
    raw = [raw_word()]; words = [dict(raw[0], source_row=0)]
    a = p.observe(raw, words, [20, 20]); b = p.observe([], [], [20, 20])
    assert a['rows'] == words and a['probabilities'] is None
    decision = p.route(dict(a=a, b=b), dict(a='1', b='2'), fallback_images=1)
    assert decision['accepted_ids'] == ['a'] and decision['fallback_ids'] == ['b']
    assert decision['deepsolo_images'] == 1 and decision['total_images'] == 2
    assert decision['deepsolo_percentage'] == 50.


def test_guard_fails_if_nonempty_experts_cannot_fill_exact_quota(tmp_path):
    p = pipeline(tmp_path, guard=True); empty = p.observe([], [], [20, 20])
    with pytest.raises(ValueError, match='Guard cannot satisfy'):
        p.route(dict(a=empty, b=empty), dict(a='1', b='2'), fallback_images=1)


def test_unqualified_runtime_refuses_default_deployment(tmp_path):
    pipeline(tmp_path)
    with pytest.raises(FileNotFoundError):
        FixedExpertPipeline(tmp_path)
