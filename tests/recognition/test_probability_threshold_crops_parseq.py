from scripts.evaluate_probability_threshold_crops_parseq_train import _wordness


def test_threshold_crop_wordness_categories_are_geometry_only():
    words = ((0, 0, 10, 10), (20, 0, 30, 10))
    assert _wordness((0, 0, 10, 10), words) == "WORD_GOOD"
    assert _wordness((0, 0, 4, 10), words) == "WORD_PARTIAL"
    assert _wordness((0, 0, 30, 10), words) == "MULTIWORD_MERGE"
    assert _wordness((40, 0, 50, 10), words) == "UNMATCHED"
