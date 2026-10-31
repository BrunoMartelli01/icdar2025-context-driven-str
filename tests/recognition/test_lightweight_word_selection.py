import numpy as np
import pytest

from recognition.lightweight_word_selection import (
    proposals, feature_matrix, correctness_proxy, admit, reconcile, risk_summary,
)
from recognition.deepsolo_lexicon import DeepSoloLexicon


def word(text, box=(0, 0, 10, 5)):
    return dict(raw_text=text, bbox=list(box), confidence=.9, detector_score=.8,
                sequence_confidence_native=.7, token_probabilities=[.8]*len(text))


def test_proposals_keep_original_and_lexical_alternative_without_mutation():
    raw = word('cqt')
    lex = DeepSoloLexicon(['CAT'], {'CAT': 'cat'}, distance_exclusive=1.5)
    rows = proposals({'a': [raw]}, lex)
    assert [r['raw_text'] for r in rows] == ['cqt', 'cat']
    assert raw['raw_text'] == 'cqt'
    assert [r['lexical_alternative'] for r in rows] == [0, 1]


def test_features_require_no_labels_and_mark_agreement():
    rows = proposals({'a': [word('cat')], 'b': [word('cat')]}, None)
    x = feature_matrix(rows, [40, 30], ['a', 'b'], [.9, .99])
    assert x.shape == (2, 31) and np.isfinite(x).all()
    assert x[0, 25] == pytest.approx(.9)
    assert feature_matrix([], [40, 30], ['a', 'b'], [.9, .99]).shape == (0, 31)


def test_admission_suppresses_alternatives_but_preserves_spatial_repetitions():
    rows = proposals({'a': [word('cat'), word('cat', (20, 0, 30, 5))]}, None)
    rows.append(dict(rows[0], proposal_id='b:0:0', raw_text='cot'))
    selected = admit(rows, [.9, .8, .7], threshold=.5, duplicate_iou=.5)
    assert len(selected) == 2 and all(r['raw_text'] == 'cat' for r in selected)
    with pytest.raises(ValueError, match='probabilities'):
        admit(rows, [.9, float('nan'), .7], threshold=.5, duplicate_iou=.5)


def test_empty_and_wrong_words_cannot_be_correctness_labels():
    rows = [word('cat'), word('!!!'), word('cot')]
    np.testing.assert_array_equal(correctness_proxy(rows, [dict(text='CAT')]), [1, 0, 0])


def test_reconciliation_adds_only_distinct_confident_words_and_preserves_fallback():
    light = [dict(word('cat'), admission_probability=.99),
             dict(word('dog', (20, 0, 30, 5)), admission_probability=.98)]
    fallback = [word('cat')]
    assert len(reconcile(light, fallback, confidence=None, maximum_overlap=.5)) == 1
    combined = reconcile(light, fallback, confidence=.95, maximum_overlap=.5)
    assert [r['raw_text'] for r in combined] == ['cat', 'dog']
    assert combined[1]['fallback_complement'] is True
    assert len(fallback) == 1 and light[1].get('word_source') is None


def test_risk_summary_handles_empty_outputs():
    assert risk_summary([], [], []).shape == (12,)
    assert risk_summary([], [], [])[-2] == 1.


def test_word_unet_raw_adapter_computes_existing_balanced_confidence(tmp_path):
    import json
    import math
    from scripts.run_lightweight_word_selection import load_source
    raw = word('cat')
    del raw['confidence']
    raw['token_log_probabilities'] = [math.log(.8)]*3
    target = tmp_path/'fit/samples/icdar13__train__1.json'
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps(dict(raw_rows=[raw],
        image_sha256='rgb', ground_truth_used_for_prediction=False)))
    config = dict(sources=dict(u=dict(kind='word_unet', root=str(tmp_path))),
                  original_fit_ids=['icdar13:train:1'], original_selection_ids=[])
    rows, _, _ = load_source(config, dict(sample_id='icdar13:train:1',
                                        image_sha256='rgb'), 'u', 'fit')
    assert rows[0]['confidence'] == pytest.approx(.8)
    assert 'confidence' not in raw


def test_pre_fallback_exact_plan_requires_no_fallback_predictions():
    from recognition.lightweight_word_router import exact_plan
    result = exact_plan({'a': -.2, 'b': .5, 'c': .1},
                        {'a': 'rgb1', 'b': 'rgb2', 'c': 'rgb3'}, 1)
    assert result['fallback_ids'] == ['a']
    assert (result['deepsolo_images'], result['total_images']) == (1, 3)


def test_deployment_rejects_changed_qualification_before_loading_models(tmp_path):
    import json
    from recognition.lightweight_word_deployment import LightweightWordPipeline
    audit = tmp_path/'audit.json'; audit.write_text('{}')
    (tmp_path/'deployment_qualification.json').write_text(json.dumps(dict(
        status='PASS', audit_path=str(audit), audit_sha256='0'*64,
        artifact_sha256={})))
    with pytest.raises(ValueError, match='qualification'):
        LightweightWordPipeline(tmp_path)
