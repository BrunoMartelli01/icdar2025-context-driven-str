"""Approximate word recovery from line transcriptions, without invented word boxes.

GT tokens retain their line region. Predictions match complete normalized tokens
one-to-one, with intersection / prediction-box area as spatial support. This is
a line-supported recovery diagnostic, not official word localization evaluation.
"""
from __future__ import annotations

from collections import Counter
import math

from recognition.group_word_selection import normalize
from recognition.scene_text_v1_evaluation import maximum_matching


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def _valid_box(box) -> tuple[float, float, float, float]:
    if len(box) != 4 or any(not math.isfinite(float(value)) for value in box):
        raise ValueError("expected a finite xyxy box")
    result = tuple(float(value) for value in box)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("box has no positive area")
    return result


def prediction_coverage(box, line_box) -> float:
    a, b = _valid_box(box), _valid_box(line_box)
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    return intersection / ((a[2] - a[0]) * (a[3] - a[1]))


def word_metrics(correct: int, predictions: int, gt_words: int) -> dict:
    if (any(type(value) is not int for value in (correct, predictions, gt_words))
        or not 0 <= correct <= min(predictions, gt_words)):
        raise ValueError("invalid one-to-one word counts")
    precision = correct / predictions if predictions else 0.
    recall = correct / gt_words if gt_words else 0.
    f1 = 2 * correct / (predictions + gt_words) if predictions + gt_words else 0.
    return {"correct_words": correct, "predicted_words": predictions,
        "gt_words": gt_words, "tp": correct, "fp": predictions - correct,
        "fn": gt_words - correct, "precision": precision, "recall": recall,
        "word_recovery_f1": f1, "hmean": f1}


def evaluate_line_tokens(predictions: list[dict], ground_truth: list[dict],
                         minimum_prediction_coverage: float) -> dict:
    if (type(minimum_prediction_coverage) not in (int, float)
        or not math.isfinite(minimum_prediction_coverage)
        or not 0 < minimum_prediction_coverage <= 1):
        raise ValueError("minimum prediction coverage must be in (0,1]")
    gt_tokens, pred_tokens = [], []
    for line_index, line in enumerate(ground_truth):
        _valid_box(line["bbox"])
        for token_index, token in enumerate(tokens(line["text"])):
            gt_tokens.append({"line_index": line_index, "token_index": token_index,
                              "text": token, "line_bbox": line["bbox"]})
    if len({prediction["candidate_id"] for prediction in predictions}) != len(predictions):
        raise ValueError("duplicate prediction candidate IDs")
    for prediction_index, prediction in enumerate(predictions):
        _valid_box(prediction["bbox"])
        # An empty normalized reading is one unmatched prediction, not discarded.
        for token_index, token in enumerate(tokens(prediction["raw_text"]) or [""]):
            pred_tokens.append({"prediction_index": prediction_index,
                "candidate_id": prediction["candidate_id"], "token_index": token_index,
                "text": token, "bbox": prediction["bbox"]})
    edges = [[pi, gi] for pi, pred in enumerate(pred_tokens)
             for gi, word in enumerate(gt_tokens)
             if pred["text"] and pred["text"] == word["text"]
             and prediction_coverage(pred["bbox"], word["line_bbox"]) >= minimum_prediction_coverage]
    matched = maximum_matching(len(pred_tokens), len(gt_tokens), edges)
    # A separate image-wide multiset count exposes how much geometry matters.
    lexical = sum((Counter(word["text"] for word in gt_tokens)
        & Counter(word["text"] for word in pred_tokens if word["text"])).values())
    return {**word_metrics(len(matched), len(pred_tokens), len(gt_tokens)),
        "gt_lines": len(ground_truth),
        "normalized_empty_gt_lines": sum(not tokens(line["text"]) for line in ground_truth),
        "normalized_empty_predictions": sum(not word["text"] for word in pred_tokens),
        "image_lexical_multiset_correct_without_geometry": lexical,
        "matched_gt_word_indices": sorted(matched), "gt_word_tokens": gt_tokens,
        "prediction_word_tokens": pred_tokens, "eligible_match_edges": edges}
