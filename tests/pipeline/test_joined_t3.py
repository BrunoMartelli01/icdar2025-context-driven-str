import numpy as np
import pytest

from diagnostics.joined_t3 import compare_joined, join_crop_transcriptions


def crop(name, text, box):
    return {"candidate_id": name, "text": text, "bbox": box}


def test_joins_every_block_in_geometric_reading_order_not_area_order():
    rows = [crop("bottom", "NEXT", [0, 30, 20, 40]),
            crop("right", "WORLD", [30, 1, 60, 11]),
            crop("left", "HELLO", [0, 0, 20, 10])]
    joined = join_crop_transcriptions(rows)
    assert joined["text"] == "HELLO WORLD NEXT"
    assert [r["candidate_id"] for r in joined["ordered_blocks"]] == ["left", "right", "bottom"]
    assert rows[0]["candidate_id"] == "bottom"


def test_preserves_repeated_words_punctuation_and_empty_metadata():
    rows = [crop("1", " NO ", [0, 0, 10, 10]), crop("2", "NO!", [20, 0, 30, 10]),
            crop("3", "", [40, 0, 50, 10])]
    result = join_crop_transcriptions(rows)
    assert result["text"] == "NO NO!"
    assert result["input_crop_count"] == 3 and result["nonempty_crop_count"] == 2
    assert join_crop_transcriptions([])["text"] == ""


def test_join_is_deterministic_under_input_shuffle():
    rows = [crop("a", "one", [0, 0, 10, 10]), crop("b", "two", [20, 1, 30, 11])]
    assert join_crop_transcriptions(rows) == join_crop_transcriptions(rows[::-1])


def test_rejects_invalid_boxes_and_duplicate_ids():
    with pytest.raises(ValueError, match="bbox"):
        join_crop_transcriptions([crop("a", "text", [0, 0, 0, 1])])
    with pytest.raises(ValueError, match="duplicate"):
        join_crop_transcriptions([crop("a", "x", [0, 0, 1, 1])]*2)


def test_compares_complete_joined_string_and_keeps_t1_in_competition():
    joined = {"text": "two words"}
    decision = compare_joined("other", joined, "two words", [[0, 1], [1, 0], [1, 0]], "selected_candidate_paired")
    assert decision["T3"] == "two words" and decision["selected_output"] == "two words"
    assert decision["C"] == 1 and not decision["needs_fallback"]
    decision = compare_joined("two words", {"text": "other"}, "two words", [[1, 0], [0, 1], [1, 0]], "selected_candidate_paired")
    assert decision["semantic_winner"] == "T1" and not decision["needs_fallback"]


def test_separate_strategies_expose_confidence_output_disagreement():
    embeddings = np.array([[1, 0], [.9, np.sqrt(.19)], [1, 0]])
    paired = compare_joined("xxxx", {"text": "word"}, "word", embeddings, "selected_candidate_paired")
    repository = compare_joined("xxxx", {"text": "word"}, "word", embeddings, "repository_paired_threshold")
    assert paired["selected_output"] == repository["selected_output"] == "xxxx"
    assert paired["needs_fallback"] and not repository["needs_fallback"]
