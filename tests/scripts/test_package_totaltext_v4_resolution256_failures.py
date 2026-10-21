from scripts.package_totaltext_v4_resolution256_failures import (
    matched_gt_indices, without_punctuation_symbols,
)


def test_symmetric_punctuation_symbol_normalization():
    assert without_punctuation_symbols("DON'T!") == "dont"
    assert without_punctuation_symbols("don't") == "dont"
    assert without_punctuation_symbols("€25") == "25"
    assert without_punctuation_symbols("café") == "café"


def test_spatial_one_to_one_matching_keeps_case_and_symbol_modes_separate():
    words = [{"bbox": [0, 0, 30, 10], "text": "Don't!"},
             {"bbox": [40, 0, 70, 10], "text": "SHOP"}]
    predictions = [{"bbox": [0, 0, 30, 10], "text": "dont"},
                   {"bbox": [40, 0, 70, 10], "text": "shop"}]
    assert matched_gt_indices(predictions, words, "geometry") == {0, 1}
    assert matched_gt_indices(predictions, words, "strict") == set()
    assert matched_gt_indices(predictions, words, "casefold") == {1}
    assert matched_gt_indices(predictions, words, "no_punctuation_symbols") == {0, 1}


def test_punctuation_only_truth_cannot_match_empty_prediction():
    words = [{"bbox": [0, 0, 10, 10], "text": "!"}]
    predictions = [{"bbox": [0, 0, 10, 10], "text": ""}]
    assert matched_gt_indices(predictions, words, "no_punctuation_symbols") == set()
    assert matched_gt_indices(predictions, words, "geometry") == {0}
