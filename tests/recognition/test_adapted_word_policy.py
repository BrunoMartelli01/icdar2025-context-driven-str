from recognition.adapted_word_policy import emit
from recognition.deepsolo_lexicon import DeepSoloLexicon


def word(text, index, box, confidence=.99):
    return dict(raw_text=text, candidate_id=str(index), bbox=box,
        confidence=confidence, detector_score=.95, sequence_confidence_native=.99)


def test_filtered_and_edited_words_keep_original_ignore_indices():
    lex = DeepSoloLexicon(['CAT'], {'CAT': 'cat'}, distance_exclusive=1.5)
    rows = [word('', 0, [0, 0, 10, 5]), word('cqt', 1, [0, 0, 10, 5])]
    policy = dict(sequence_threshold=.9, joint_threshold=.6, lexical='keep',
                  same_text_nms_iou=.5, require_alphanumeric=True)
    output, changes = emit(rows, lex, policy)
    assert output[0]['raw_text'] == 'cat' and output[0]['source_row'] == 1
    assert changes == 1 and rows[1]['raw_text'] == 'cqt'


def test_text_nms_preserves_spatial_repeated_words():
    lex = DeepSoloLexicon(['CAT'], {'CAT': 'cat'}, distance_exclusive=1.5)
    rows = [word('cat', 0, [0, 0, 10, 5]), word('CAT', 1, [0, 0, 10, 5], .9),
            word('cat', 2, [20, 0, 30, 5])]
    policy = dict(sequence_threshold=.9, joint_threshold=.6, lexical='none',
                  same_text_nms_iou=.5, require_alphanumeric=True)
    output, _ = emit(rows, None, policy)
    assert {r['candidate_id'] for r in output} == {'0', '2'}
