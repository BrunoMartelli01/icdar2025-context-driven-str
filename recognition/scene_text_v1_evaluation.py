"""Evaluation primitives for frozen practical-remap-v1 scene-text proposals.

This module is deliberately metric-only.  Candidate generation never consumes
ground truth; annotations enter only after every proposal and PARSeq output has
been frozen.  Boxes use ``xyxy`` coordinates with exclusive upper bounds.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Sequence

from rapidfuzz.fuzz import ratio


def box_iou(left: Sequence[int], right: Sequence[int]) -> float:
    lx1, ly1, lx2, ly2 = (int(value) for value in left)
    rx1, ry1, rx2, ry2 = (int(value) for value in right)
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = max(0, lx2 - lx1) * max(0, ly2 - ly1)
    right_area = max(0, rx2 - rx1) * max(0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def maximum_matching(
    prediction_count: int,
    gt_count: int,
    edges: Sequence[tuple[int, int]],
) -> set[int]:
    """Return GT indices in a deterministic maximum bipartite matching."""

    adjacency = {index: [] for index in range(prediction_count)}
    for prediction_index, gt_index in edges:
        if not (0 <= prediction_index < prediction_count and 0 <= gt_index < gt_count):
            raise ValueError("matching edge is outside the declared populations")
        adjacency[prediction_index].append(gt_index)
    for values in adjacency.values():
        values.sort()
    assigned: dict[int, int] = {}

    def visit(prediction_index: int, seen: set[int]) -> bool:
        for gt_index in adjacency[prediction_index]:
            if gt_index in seen:
                continue
            seen.add(gt_index)
            previous = assigned.get(gt_index)
            if previous is None or visit(previous, seen):
                assigned[gt_index] = prediction_index
                return True
        return False

    for prediction_index in range(prediction_count):
        visit(prediction_index, set())
    return set(assigned)


def _edges(
    predictions: Sequence[dict[str, Any]],
    ground_truth: Sequence[dict[str, Any]],
    predicate: Callable[[str, str], bool] | None,
    *,
    iou_threshold: float,
) -> list[tuple[int, int]]:
    output = []
    for prediction_index, prediction in enumerate(predictions):
        for gt_index, word in enumerate(ground_truth):
            if box_iou(prediction["bbox"], word["bbox"]) < iou_threshold:
                continue
            if predicate is None or predicate(str(prediction["text"]), str(word["text"])):
                output.append((prediction_index, gt_index))
    return output


def evaluate_images(rows: Sequence[dict[str, Any]], *, text_available: bool) -> dict[str, Any]:
    """Aggregate geometry and OCR with one-to-one spatial matching."""

    gt_count = proposal_count = nonempty_count = 0
    geometry_matches = {0.25: 0, 0.5: 0, 0.75: 0}
    strict_matches = exact_matches = fuzzy_matches = lexical_matches = 0
    best_spatial_strict = best_spatial_exact = best_spatial_fuzzy = 0
    for row in rows:
        ground_truth = list(row["ground_truth"])
        predictions = list(row["predictions"])
        gt_count += len(ground_truth)
        proposal_count += len(predictions)
        nonempty_count += sum(bool(str(value["text"]).strip()) for value in predictions)
        for threshold in geometry_matches:
            geometry_matches[threshold] += len(
                maximum_matching(
                    len(predictions),
                    len(ground_truth),
                    _edges(predictions, ground_truth, None, iou_threshold=threshold),
                )
            )
        if not text_available:
            continue
        strict = lambda left, right: left == right
        exact = lambda left, right: left.casefold() == right.casefold()
        fuzzy = lambda left, right: ratio(left.casefold(), right.casefold()) >= 80.0
        strict_matches += len(
            maximum_matching(
                len(predictions), len(ground_truth),
                _edges(predictions, ground_truth, strict, iou_threshold=0.5),
            )
        )
        exact_matches += len(
            maximum_matching(
                len(predictions), len(ground_truth),
                _edges(predictions, ground_truth, exact, iou_threshold=0.5),
            )
        )
        fuzzy_matches += len(
            maximum_matching(
                len(predictions), len(ground_truth),
                _edges(predictions, ground_truth, fuzzy, iou_threshold=0.5),
            )
        )
        for word in ground_truth:
            if not predictions:
                continue
            best = max(
                predictions,
                key=lambda value: (box_iou(value["bbox"], word["bbox"]), -int(value["rank"])),
            )
            if box_iou(best["bbox"], word["bbox"]) <= 0:
                continue
            best_spatial_strict += int(strict(str(best["text"]), str(word["text"])))
            best_spatial_exact += int(exact(str(best["text"]), str(word["text"])))
            best_spatial_fuzzy += int(fuzzy(str(best["text"]), str(word["text"])))
        predicted_words = Counter(
            str(value["text"]).casefold() for value in predictions if str(value["text"]).strip()
        )
        gt_words = Counter(str(value["text"]).casefold() for value in ground_truth)
        lexical_matches += sum((predicted_words & gt_words).values())

    geometry_at_50 = geometry_matches[0.5]
    precision = geometry_at_50 / proposal_count if proposal_count else 0.0
    recall = geometry_at_50 / gt_count if gt_count else 0.0
    geometry = {
        "gt_instances": gt_count,
        "retained_proposals": proposal_count,
        "nonempty_parseq_outputs": nonempty_count,
        "iou_ge_0_25_one_to_one": geometry_matches[0.25],
        "iou_ge_0_50_one_to_one": geometry_at_50,
        "iou_ge_0_75_one_to_one": geometry_matches[0.75],
        "proposal_precision_iou_ge_0_50": precision,
        "proposal_recall_iou_ge_0_50": recall,
        "proposal_f1_iou_ge_0_50": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
    }
    if not text_available:
        return {
            "geometry": geometry,
            "recognition": None,
            "recognition_unavailable_reason": (
                "Local CTW1500 artifact contains detection-only text-line annotations; "
                "word transcriptions are unavailable."
            ),
        }
    return {
        "geometry": geometry,
        "recognition": {
            "normalization": "case-sensitive exact and Unicode casefold exact; punctuation retained",
            "matching": "maximum one-to-one matching with bbox IoU>=0.50",
            "exact_case_sensitive_spatial_one_to_one": strict_matches,
            "exact_case_sensitive_rate_all_gt": strict_matches / gt_count if gt_count else 0.0,
            "exact_casefold_spatial_one_to_one": exact_matches,
            "exact_casefold_rate_all_gt": exact_matches / gt_count if gt_count else 0.0,
            "exact_casefold_rate_geometry_available": (
                exact_matches / geometry_at_50 if geometry_at_50 else 0.0
            ),
            "rapidfuzz_casefold_ge_0_8_spatial_one_to_one": fuzzy_matches,
            "rapidfuzz_casefold_ge_0_8_rate_all_gt": (
                fuzzy_matches / gt_count if gt_count else 0.0
            ),
            "best_spatial_crop_exact_casefold": best_spatial_exact,
            "best_spatial_crop_exact_case_sensitive": best_spatial_strict,
            "best_spatial_crop_rapidfuzz_ge_0_8": best_spatial_fuzzy,
            "image_level_lexical_exact_multiset_matches": lexical_matches,
        },
    }


__all__ = ["box_iou", "evaluate_images", "maximum_matching"]
