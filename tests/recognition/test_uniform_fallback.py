import pytest

from recognition.uniform_fallback import UniformFallbackPostprocessor
from segmentation.uniform_unet_fusion import file_hash


def profile():
    return dict(mode='approximate', claim_label='approximate reproduction', checkpoint_sha256='qualified', lexicon=None,
        policy=dict(decoder='greedy', same_text_nms_iou=None, resolution='native', score_threshold=.4, recognition_threshold=.8))


def detection(index=1, text='cot', score=.4, probability=.8):
    row = [0.]*37; row[0] = probability; row[36] = 1-probability
    return dict(source_index=index, text=text, score=score,
        geometry=dict(boundary_points=[[1., 1.], [8., 1.], [8., 5.], [1., 5.]]),
        raw=dict(recognition_class_probabilities=[row]))


def test_thresholds_are_inclusive_and_duplicate_words_remain_separate():
    model = UniformFallbackPostprocessor(profile())
    response = dict(success=True, detections=[detection(2), detection(1), detection(3, probability=.79)])
    result = model.apply(response, [20, 10], 1, 'qualified')
    assert [r['instance_id'] for r in result['rows']] == [1, 2]
    assert [r['raw_text'] for r in result['rows']] == ['cot', 'cot']
    assert not result['ground_truth_used'] and not any(r['ignored'] for r in result['rows'])


def test_checkpoint_failures_and_duplicate_detections_propagate():
    model = UniformFallbackPostprocessor(profile())
    with pytest.raises(RuntimeError, match='failure'):
        model.apply(dict(success=False), [20, 10], 1, 'qualified')
    with pytest.raises(ValueError, match='checkpoint'):
        model.apply(dict(success=True, detections=[]), [20, 10], 1, 'other')
    with pytest.raises(ValueError, match='duplicated'):
        model.apply(dict(success=True, detections=[detection(), detection()]), [20, 10], 1, 'qualified')


def test_lexicon_rule_preserves_source_order_and_rejects_only_when_declared(tmp_path):
    vocabulary, pairs = tmp_path/'words.txt', tmp_path/'pairs.txt'
    vocabulary.write_text('CAT\nCUT\n'); pairs.write_text('CAT Cat\nCUT Cut\n')
    c = profile(); c['lexicon'] = dict(vocabulary=str(vocabulary), pairs=str(pairs),
        vocabulary_sha256=file_hash(vocabulary), pairs_sha256=file_hash(pairs),
        distance_exclusive=1.5, unmatched_policy='source_reject')
    response = dict(success=True, detections=[detection(), detection(2, text='1996')])
    result = UniformFallbackPostprocessor(c).apply(response, [20, 10], 1, 'qualified')
    assert [r['text'] for r in result['rows']] == ['Cat']
    c['lexicon']['unmatched_policy'] = 'keep_unmatched'
    result = UniformFallbackPostprocessor(c).apply(response, [20, 10], 1, 'qualified')
    assert [r['text'] for r in result['rows']] == ['Cat', '1996']


def test_empty_clipped_envelope_is_retained_as_unmatchable_output():
    d = detection(); d['geometry']['boundary_points'] = [[30., 1.], [38., 1.], [38., 5.], [30., 5.]]
    result = UniformFallbackPostprocessor(profile()).apply(dict(success=True, detections=[d]), [20, 10], 1, 'qualified')
    assert len(result['rows']) == 1 and result['rows'][0]['bbox'] is None
    assert result['rows'][0]['raw_text'] == '' and result['rows'][0]['invalid_envelope'] is not None
