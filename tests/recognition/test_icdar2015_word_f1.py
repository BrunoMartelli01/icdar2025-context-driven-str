"""Word F1 boundaries: symbols, duplicates, spatial matching and empty outputs."""

import itertools

import pytest

from recognition.icdar2015_word_f1 import aggregate_counts, normalize_word, sample_counts


def p(text, bbox=(0, 0, 10, 10)):
    return {"text": text, "bbox": list(bbox)}


def test_symbols_casefold_and_unicode_letters_are_handled_symmetrically():
    assert normalize_word("L'été-2026!") == "lété2026"
    assert normalize_word("Straße®") == "strasse"
    counts = sample_counts([p("L'ÉTÉ-2026!")], [p("lété2026")], iou_threshold=0.5)
    assert counts["alphanumeric_casefold_tp"] == 1 and counts["casefold_exact_tp"] == 0
    assert aggregate_counts([counts])["alphanumeric_casefold"]["f1"] == 1


def test_repeated_predictions_and_symbol_only_gt_are_not_removed():
    counts = sample_counts([p("word!"), p("WORD"), p("")],
        [p("word"), p("###", (30, 0, 40, 10))], iou_threshold=0.5)
    result = aggregate_counts([counts])
    assert result["alphanumeric_casefold"] == {"tp": 1, "fp": 2, "fn": 1,
        "precision": 1 / 3, "recall": 0.5, "f1": 0.4}
    assert result["symbol_only_gt"] == 1 and result["empty_normalized_predictions"] == 1


def test_wrong_location_and_one_combined_output_cannot_count_as_two_words():
    counts = sample_counts([p("right", (20, 0, 30, 10)), p("two words")],
        [p("right"), p("two"), p("words")], iou_threshold=0.5)
    assert counts["alphanumeric_casefold_tp"] == 0
    assert counts["geometry_tp"] == 1


def test_iou_boundary_and_empty_normalized_strings_never_match():
    assert sample_counts([p("a")], [p("A", (0, 0, 20, 10))], iou_threshold=0.5)["alphanumeric_casefold_tp"] == 1
    assert sample_counts([p("a")], [p("A", (0, 0, 20, 10))], iou_threshold=0.50001)["alphanumeric_casefold_tp"] == 0
    assert sample_counts([p("!$")], [p("#")], iou_threshold=0.5)["alphanumeric_casefold_tp"] == 0
    with pytest.raises(ValueError, match="threshold"):
        sample_counts([], [], iou_threshold=0)


def test_empty_images_stay_in_micro_aggregation():
    empty = sample_counts([], [], iou_threshold=0.5)
    missing = sample_counts([], [p("a")], iou_threshold=0.5)
    false_positive = sample_counts([p("b")], [], iou_threshold=0.5)
    assert aggregate_counts([empty])["alphanumeric_casefold"]["f1"] is None
    result = aggregate_counts([empty, missing, false_positive])
    assert result["images"] == 3 and result["alphanumeric_casefold"]["f1"] == 0
    assert result["alphanumeric_casefold"]["fp"] == 1 and result["alphanumeric_casefold"]["fn"] == 1


def test_matching_is_maximum_cardinality_and_order_independent():
    predictions = [p("a", (0, 0, 20, 10)), p("a", (0, 0, 10, 10))]
    targets = [p("a", (0, 0, 10, 10)), p("a", (10, 0, 20, 10))]
    for order in itertools.permutations(predictions):
        counts = sample_counts(order, targets, iou_threshold=0.5)
        assert counts["alphanumeric_casefold_tp"] == 2


def test_independent_scipy_matcher_agrees_across_spatial_and_symbol_cases():
    from scripts.audit_icdar2015_word_f1 import independent_tp
    candidates = [p("a!"), p("A", (10, 0, 20, 10)), p(""), p("b", (0, 0, 20, 10))]
    targets = [p("a"), p("A", (10, 0, 20, 10)), p("b", (0, 0, 20, 10)), p("$")]
    for predictions in itertools.product(candidates, repeat=2):
        for words in itertools.product(targets, repeat=2):
            counts = sample_counts(predictions, words, iou_threshold=0.5)
            for mode in ("geometry", "casefold_exact", "alphanumeric_casefold"):
                assert independent_tp(predictions, words, 0.5, mode) == counts[mode + "_tp"]
