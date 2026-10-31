from recognition.fallback_word_fusion import replace_fallback_transcripts


def word(text, box, probability, source):
    return dict(raw_text=text, bbox=box, word_probability=probability, source_row=source)


def fallback(text, box, instance):
    return dict(raw_text=text, text=text, bbox=box, instance_id=instance, ignored=False)


def test_fusion_replaces_only_matched_transcript_and_preserves_original_inputs():
    light = [word('cat', [0, 0, 10, 10], .98, 7)]
    heavy = [fallback('car', [0, 0, 10, 10], 1), fallback('other', [20, 0, 30, 10], 2)]
    policy = dict(enabled=True, minimum_word_probability=.9, minimum_iou=.5, maximum_edit_distance=1)
    out = replace_fallback_transcripts(light, heavy, policy)
    assert [r['raw_text'] for r in out] == ['cat', 'other']
    assert out[0]['bbox'] == heavy[0]['bbox'] and out[0]['instance_id'] == 1
    assert out[0]['original_deepsolo_text'] == 'car' and heavy[0]['raw_text'] == 'car'


def test_fusion_uses_one_to_one_matches_and_respects_confidence_and_distance():
    light = [word('same', [0, 0, 10, 10], .99, 0),
             word('far', [20, 0, 30, 10], .7, 1)]
    heavy = [fallback('same', [0, 0, 10, 10], 0), fallback('wrong', [0, 0, 9, 9], 1),
             fallback('bar', [20, 0, 30, 10], 2)]
    p = dict(enabled=True, minimum_word_probability=.9, minimum_iou=.3, maximum_edit_distance=None)
    assert [r['raw_text'] for r in replace_fallback_transcripts(light, heavy, p)] == ['same', 'wrong', 'bar']
    p.update(minimum_word_probability=.5, maximum_edit_distance=0)
    assert [r['raw_text'] for r in replace_fallback_transcripts(light, heavy, p)] == ['same', 'wrong', 'bar']
