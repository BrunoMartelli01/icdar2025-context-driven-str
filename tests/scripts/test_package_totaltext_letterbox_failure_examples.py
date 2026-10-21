import numpy as np
import pytest

from scripts.package_totaltext_letterbox_failure_examples import (
    box_coverage, classify, native_word_box, select_examples, render, filter_and_reselect, fit_crop, lexical_audit,
)


def evidence(**changes):
    row = dict(normalized_correct=False, valid_exact_candidate_exists=False,
               predicted_pixels_in_word_box_native=10, significantly_intersected_gt_words=1,
               best_iou=.65, word_box_coverage=.9, component_count_in_word_box=1,
               component_union_iou=.65)
    row.update(changes)
    return row


@pytest.mark.parametrize("updates,expected", [
    ({"normalized_correct": True}, "PUNCTUATION_OR_SYMBOL_ONLY"),
    ({"valid_exact_candidate_exists": True}, "ONE_TO_ONE_MATCHING_CONFLICT"),
    ({"predicted_pixels_in_word_box_native": 0}, "NO_PRED_FOREGROUND_IN_WORD_BOX"),
    ({"best_iou": .3, "significantly_intersected_gt_words": 2}, "MULTIWORD_FUSION"),
    ({"best_iou": .2, "component_count_in_word_box": 3}, "FRAGMENTATION_OR_TRUNCATION"),
    ({"word_box_coverage": .6}, "FRAGMENTATION_OR_TRUNCATION"),
    ({"best_iou": .2, "component_union_iou": .2}, "BOX_SHAPE_OR_EXTENT_MISMATCH"),
    ({}, "OCR_MISMATCH_WITH_ACCEPTABLE_BOX"),
])
def test_exclusive_classification(updates, expected):
    assert classify(evidence(**updates)) == expected


def test_coverage_and_letterbox_coordinate_frames():
    assert box_coverage([0, 0, 100, 100], [10, 10, 20, 20]) == 1
    assert box_coverage([10, 10, 15, 20], [10, 10, 20, 20]) == .5
    assert native_word_box([0, 0, 100, 200], (100, 200)) == [64, 0, 192, 256]
    assert native_word_box([0, 0, 200, 100], (200, 100)) == [0, 64, 256, 192]
    assert native_word_box([0, 0, 256, 256], (256, 256)) == [0, 0, 256, 256]


def test_selection_is_deterministic_and_prefers_distinct_images():
    rows = [dict(category="MULTIWORD_FUSION", gt_text="word", significantly_intersected_gt_words=3,
                 component_count_in_word_box=2, best_iou=.2, sample_id=f"s{i//2}", word_index=i)
            for i in range(8)]
    first = select_examples(rows, 3)
    assert first == select_examples(list(reversed(rows)), 3)
    assert len({r["sample_id"] for r in first}) == 3
    assert len(first) == 3


def test_composite_has_all_panels_and_no_inference():
    from io import BytesIO
    from PIL import Image
    row = evidence()
    row.update(sample_id="fixture", word_index=0, category="OCR_MISMATCH_WITH_ACCEPTABLE_BOX",
               reason="Diagnostic evidence only", gt_bbox=[2, 2, 20, 12], crop_bbox=[2, 2, 20, 12],
               gt_text="WORD", parseq_text="W0RD")
    payload = render(row, np.zeros((24, 32, 3), dtype=np.uint8), Image.new("L", (32, 24)),
                     np.zeros((256, 256), dtype=np.uint8), np.zeros((256, 256), dtype=np.uint8))
    assert Image.open(BytesIO(payload)).size == (1600, 1100)


def test_partial_box_does_not_imply_recognizer_only_failure():
    assert classify(evidence(best_iou=.55, word_box_coverage=.6)) == "FRAGMENTATION_OR_TRUNCATION"
    assert classify(evidence(best_iou=.4, component_union_iou=.4)) == "BOX_SHAPE_OR_EXTENT_MISMATCH"


def test_normalized_reselection_excludes_punctuation_without_mutating_source():
    from collections import Counter
    rows = [dict(category="BOX_SHAPE_OR_EXTENT_MISMATCH", normalized_correct=False,
                 gt_text="WORD", significantly_intersected_gt_words=1, component_count_in_word_box=1,
                 best_iou=.1, sample_id=f"s{i}", word_index=0) for i in range(5)]
    punctuation = dict(rows[0], category="PUNCTUATION_OR_SYMBOL_ONLY", normalized_correct=True)
    source = dict(per_word_failures=rows+[punctuation],
                  category_counts=dict(Counter(r["category"] for r in rows+[punctuation])),
                  categories={"PUNCTUATION_OR_SYMBOL_ONLY":"symbols", "BOX_SHAPE_OR_EXTENT_MISMATCH":"geometry"})
    result = filter_and_reselect(source, 5)
    assert result["failures"] == 5
    assert result["excluded_punctuation_symbol_only"] == 1
    assert len(result["selected_examples"]) == 5
    assert len(source["per_word_failures"]) == 6
    assert "PUNCTUATION_OR_SYMBOL_ONLY" not in result["categories"]


def test_small_crop_enlargement_is_display_only():
    from PIL import Image
    source = Image.new("RGB", (10, 5), "red")
    canvas = Image.new("RGB", (100, 100))
    assert fit_crop(canvas, source, (0, 0, 100, 100)) == (0, 25, 100, 50)
    assert source.size == (10, 5)


def test_lexical_audit_consumes_duplicates_once_and_stays_within_image():
    report = dict(gt_words=4, images=2,
                  aggregate=dict(exact_casefold_iou_ge_50=0, exact_no_symbols_iou_ge_50=0),
                  per_image=[
                      dict(sample_id="a", ground_truth=[dict(text="WORD"), dict(text="WORD"), dict(text="OTHER")],
                           predictions=[dict(text="word"), dict(text="else")]),
                      dict(sample_id="b", ground_truth=[dict(text="else")], predictions=[dict(text="other")]),
                  ])
    audit = lexical_audit(report, [])
    assert audit["lexical_counts"] == dict(case_sensitive=0, casefold=1, no_symbols=1)
    assert audit["gain_without_position"]["casefold"] == 1


def test_lexical_audit_normalizes_symbols_but_does_not_match_empty_strings():
    report = dict(gt_words=2, images=1,
                  aggregate=dict(exact_casefold_iou_ge_50=0, exact_no_symbols_iou_ge_50=0),
                  per_image=[dict(sample_id="a", ground_truth=[dict(text="can't"), dict(text="!")],
                                  predictions=[dict(text="cant"), dict(text="?")])])
    audit = lexical_audit(report, [])
    assert audit["lexical_counts"]["no_symbols"] == 1
    assert audit["lexical_counts"]["casefold"] == 0


def test_render_uses_english_explanations_not_legacy_italian_reasons(monkeypatch):
    from PIL import Image
    import scripts.package_totaltext_letterbox_failure_examples as module
    labels = []
    original = Image.Image.paste
    # Capture wrapped explanations without changing the frozen OCR/GT strings.
    monkeypatch.setattr(module, "_wrapped", lambda draw, text, *a, **kw: labels.append(text))
    row = evidence()
    row.update(sample_id="fixture", word_index=0, category="OCR_MISMATCH_WITH_ACCEPTABLE_BOX",
               reason="Testo legacy italiano", gt_bbox=[2,2,20,12], crop_bbox=[2,2,20,12],
               gt_text="WORD", parseq_text="W0RD")
    render(row, np.zeros((24,32,3), dtype=np.uint8), Image.new("L", (32,24)),
           np.zeros((256,256), dtype=np.uint8), np.zeros((256,256), dtype=np.uint8))
    assert labels[0].startswith("IoU>=.50 and coverage>=.80")
    assert labels[-2:] == ["GT WORD: WORD", "PARSeq: W0RD"]
