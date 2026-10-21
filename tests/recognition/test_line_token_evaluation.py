import copy
import json
import math

import pytest

from recognition.line_token_evaluation import evaluate_line_tokens, tokens
from scripts.evaluate_ctw1500_word_recovery import (
    DEFAULT_CONFIG, independent_word_counts, prefilter_controls, safe_prefilter_eligible,
    select_predictions, validate_config, verify_detector_rows, verify_physical_gt,
)
from recognition.line_token_evaluation import word_metrics
from recognition.yolo_word_refinement import joint_score


def prediction(cid, text, box=(0, 0, 100, 20)):
    return {"candidate_id": cid, "raw_text": text, "bbox": list(box)}


def line(text, box=(0, 0, 100, 20)):
    return {"text": text, "bbox": list(box)}


def test_separate_words_recover_a_multiword_line_without_full_line_output():
    gt = [line("HARBOR POINT")]
    predictions = [prediction("a", "Harbor!", (0, 0, 45, 20)),
                   prediction("b", "POINT", (50, 0, 100, 20))]
    result = evaluate_line_tokens(predictions, gt, 0.5)
    assert result["gt_lines"] == 1
    assert result["gt_words"] == result["correct_words"] == 2
    assert result["precision"] == result["recall"] == result["hmean"] == 1
    assert independent_word_counts(predictions, gt, 0.5) == (2, 2, 2)


@pytest.mark.parametrize("predictions,correct", [
    ([prediction("a", "GO")], 1),
    ([prediction("a", "GO"), prediction("b", "go")], 2),
    ([prediction("a", "GO"), prediction("b", "go"), prediction("c", "GO")], 2),
])
def test_repeated_tokens_are_occurrences_and_duplicates_cannot_exceed_gt(predictions, correct):
    result = evaluate_line_tokens(predictions, [line("GO GO")], 0.5)
    assert result["gt_words"] == 2 and result["correct_words"] == correct
    assert result["fp"] == len(predictions) - correct


def test_one_prediction_cannot_match_two_overlapping_gt_lines():
    result = evaluate_line_tokens([prediction("a", "same")], [line("same"), line("same")], 0.5)
    assert result["correct_words"] == 1 and result["fn"] == 1


def test_matching_text_elsewhere_in_image_is_not_spatial_word_recovery():
    gt = [line("alpha", (0, 0, 50, 20)), line("beta", (0, 30, 50, 50))]
    pred = [prediction("a", "beta", (0, 0, 50, 20)),
            prediction("b", "alpha", (0, 30, 50, 50))]
    result = evaluate_line_tokens(pred, gt, 0.5)
    assert result["correct_words"] == 0
    assert result["image_lexical_multiset_correct_without_geometry"] == 2


def test_spatial_support_boundary_and_wrong_substrings():
    pred = [prediction("a", "word", (0, 0, 20, 10))]
    gt = [line("word", (10, 0, 30, 10))]
    assert evaluate_line_tokens(pred, gt, 0.5)["correct_words"] == 1
    assert evaluate_line_tokens(pred, gt, math.nextafter(0.5, 1))["correct_words"] == 0
    for text in ("HARB", "HARBORPOINT"):
        assert evaluate_line_tokens([prediction("a", text)], [line("HARBOR POINT")], 0.5)["correct_words"] == 0


def test_unicode_normalization_whitespace_and_empty_population_accounting():
    assert tokens("Straße!\tA-B / €") == ["strasse", "ab"]
    result = evaluate_line_tokens([prediction("a", "!!!")], [line("!!!")], 0.5)
    assert result["gt_lines"] == result["normalized_empty_gt_lines"] == 1
    assert result["gt_words"] == 0
    assert result["predicted_words"] == result["fp"] == 1
    assert result["normalized_empty_predictions"] == 1
    assert evaluate_line_tokens([], [], 0.5)["hmean"] == 0


def test_multitoken_prediction_and_missing_words_produce_expected_micro_f1():
    result = evaluate_line_tokens([prediction("a", "one TWO"), prediction("b", "wrong")],
                                  [line("one two three four")], 0.5)
    assert (result["tp"], result["fp"], result["fn"]) == (2, 1, 2)
    assert result["word_recovery_f1"] == pytest.approx(4 / 7)
    assert result["hmean"] == result["word_recovery_f1"]


def test_metric_does_not_change_predictions_or_gt_and_json_roundtrips():
    predictions, gt = [prediction("a", "Word!")], [line("word other")]
    originals = copy.deepcopy((predictions, gt))
    result = evaluate_line_tokens(predictions, gt, 0.5)
    assert (predictions, gt) == originals
    assert json.loads(json.dumps(result)) == result


@pytest.mark.parametrize("box", [[0, 0, 0, 1], [0, 2, 1, 1], [0, 0, float("nan"), 1]])
def test_invalid_geometry_fails_instead_of_omitting_predictions(box):
    with pytest.raises(ValueError):
        evaluate_line_tokens([prediction("a", "word", box)], [line("word")], 0.5)


def test_approximate_configuration_and_threshold_boundaries():
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    validate_config(cfg)
    for value in (0, float("nan"), True):
        changed = copy.deepcopy(cfg)
        changed["evaluation"]["minimum_prediction_coverage"] = value
        with pytest.raises(ValueError):
            validate_config(changed)
    cfg["reproduction_mode"] = "strict"
    with pytest.raises(ValueError):
        validate_config(cfg)


@pytest.mark.parametrize("threshold", [0.1, 0.6, 0.64])
def test_safe_bound_preserves_a_perfect_ocr_float_tie_below_squared_cutoff(threshold):
    detector_score = math.nextafter(threshold * threshold, 0.)
    row = {**prediction("a", "word"), "detector_score": detector_score,
        "token_probabilities": [1.] * 4, "token_log_probabilities": [0.] * 4}
    # The old squared comparison rejects this candidate, although its computed
    # final score meets the threshold. The direct upper bound preserves it.
    assert detector_score < threshold * threshold
    assert joint_score(row) >= threshold
    assert safe_prefilter_eligible(detector_score, threshold)
    selection = {"joint_threshold": threshold, "same_text_nms_iou": 0.5}
    eligible, selected = select_predictions([row], selection, None)
    assert eligible == selected == [row]


def test_safe_bound_changes_with_final_threshold_and_full_grid_is_explicit():
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    controls = prefilter_controls(cfg["selection"])
    assert len(controls) == 96 and controls["safe_upper_bound"] is None
    assert controls["0.05"] == 0.05 and controls["0.99"] == 0.99
    assert safe_prefilter_eligible(0.36, 0.6)
    assert not safe_prefilter_eligible(0.36, 0.7)
    assert safe_prefilter_eligible(0.49, 0.7)


def test_cached_gt_must_equal_the_physical_ctw_transcription_and_geometry(tmp_path):
    annotation = tmp_path / "gt.txt"
    polygon = [[i, 0] for i in range(7)] + [[i, 4] for i in range(6, -1, -1)]
    coordinates = ",".join(str(x) for point in polygon for x in point)
    annotation.write_text('1\n' + coordinates + ',"HARBOR POINT"\n')
    record = {"sample_id": "ctw1500:example", "annotation_path": str(annotation),
              "image_size_wh": [100, 20]}
    cached = {"care": [{"text": "HARBOR POINT", "bbox": [0, 0, 7, 5],
                        "polygon_xy": polygon}], "ignored": [], "clipped_polygons": 0}
    verify_physical_gt(record, cached)
    changed = copy.deepcopy(cached)
    changed["care"][0]["text"] = "OTHER WORD"
    with pytest.raises(RuntimeError, match="physical annotation"):
        verify_physical_gt(record, changed)
    changed = copy.deepcopy(cached)
    changed["care"][0]["bbox"][2] += 1
    with pytest.raises(RuntimeError, match="physical annotation"):
        verify_physical_gt(record, changed)


def test_frozen_ocr_must_have_the_original_detector_geometry_scores_and_all_rows():
    row = {**prediction("ctw1500:example:yolo:0", "word", (0, 0, 10, 10)),
        "detector_score": 0.8, "crop_shape": [10, 10, 3],
        "token_probabilities": [1.] * 4, "token_log_probabilities": [0.] * 4}
    detector = {"slots": [{"source_index": 0, "bbox": [0, 0, 10, 10], "score": 0.8}]}
    record = {"sample_id": "ctw1500:example", "image_size_wh": [100, 20]}
    verify_detector_rows([row], detector, record)
    changed = copy.deepcopy(row)
    changed["bbox"][2] += 1
    with pytest.raises(RuntimeError, match="detector slot"):
        verify_detector_rows([changed], detector, record)
    with pytest.raises(RuntimeError, match="incomplete"):
        verify_detector_rows([], detector, record)
    changed = copy.deepcopy(row)
    changed["detector_score"] = float("nan")
    with pytest.raises(RuntimeError):
        verify_detector_rows([changed], detector, record)


def test_physical_gt_preserves_source_parser_clipping_and_reports_raw_border_changes(tmp_path):
    annotation = tmp_path / "border.txt"
    polygon = [[x, 0] for x in range(94, 101)] + [[x, 4] for x in range(100, 93, -1)]
    clipped = [[min(x, 99), y] for x, y in polygon]
    coordinates = ",".join(str(x) for point in polygon for x in point)
    annotation.write_text('1\n' + coordinates + ',"WORD"\n')
    record = {"sample_id": "ctw1500:border", "annotation_path": str(annotation),
              "image_size_wh": [100, 20]}
    cached = {"care": [{"text": "WORD", "bbox": [94, 0, 100, 5],
                        "polygon_xy": clipped}], "ignored": [], "clipped_polygons": 0}
    assert verify_physical_gt(record, cached) == 1


def test_protocol_metadata_cannot_claim_a_different_ignore_or_matching_rule():
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    for key in ("matching", "ignore", "empty", "limitation"):
        changed = copy.deepcopy(cfg)
        changed["evaluation"][key] = "different semantics"
        with pytest.raises(ValueError, match="semantics"):
            validate_config(changed)
    for grid in ({"minimum": 0.05, "maximum": 0.99, "step": 0},
                 {"minimum": 0.05, "maximum": 0.99, "step": 0.03}):
        changed = copy.deepcopy(cfg)
        changed["selection"]["prefilter_grid"] = grid
        with pytest.raises(ValueError):
            validate_config(changed)


@pytest.mark.parametrize("value", [1.5, True, -1])
def test_word_counts_must_be_nonnegative_integer_occurrences(value):
    with pytest.raises(ValueError):
        word_metrics(value, 2, 2)


def test_evaluator_rejects_duplicate_identity_and_boolean_threshold():
    row = prediction("duplicate", "word")
    with pytest.raises(ValueError, match="candidate IDs"):
        evaluate_line_tokens([row, row], [line("word word")], 0.5)
    with pytest.raises(ValueError):
        evaluate_line_tokens([row], [line("word")], True)
