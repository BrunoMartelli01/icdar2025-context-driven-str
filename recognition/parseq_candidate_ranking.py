"""Post-hoc geometry/OCR ranking metrics for frozen TRAIN proposals."""

from __future__ import annotations

from statistics import median
from typing import Any, Final, Sequence

from localize.multihypothesis_word_proposals import box_iou


SCORE_FIELDS: Final = (
    "sequence_confidence_native",
    "mean_token_log_probability",
    "minimum_token_probability",
)


def _intersection(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def rank_candidates(
    candidates: Sequence[dict[str, Any]], gt_bbox: Sequence[int], score_field: str,
) -> tuple[dict[str, Any], ...]:
    """Rank geometrically intersecting proposals using exactly one PARSeq score."""
    if score_field not in SCORE_FIELDS:
        raise ValueError("unknown PARSeq candidate-ranking score")
    associated = [value for value in candidates if _intersection(value["bbox"], gt_bbox) > 0]

    def key(value: dict[str, Any]) -> tuple[float, str]:
        score = value["ocr"][score_field]
        return (-(float(score) if score is not None else float("-inf")), value["candidate_id"])

    return tuple(sorted(associated, key=key))


def evaluate_gt_word(
    candidates: Sequence[dict[str, Any]], *, gt_bbox: Sequence[int], gt_text: str,
) -> dict[str, Any] | None:
    ious = {value["candidate_id"]: box_iou(value["bbox"], gt_bbox) for value in candidates}
    correct_050 = tuple(value["candidate_id"] for value in candidates if ious[value["candidate_id"]] >= 0.50)
    if not correct_050:
        return None
    correct_075 = tuple(value["candidate_id"] for value in candidates if ious[value["candidate_id"]] >= 0.75)
    simultaneous_exact = tuple(
        value["candidate_id"] for value in candidates
        if ious[value["candidate_id"]] >= 0.50
        and value["ocr"]["raw_text"].casefold() == gt_text.casefold()
    )
    rankings = {}
    for field in SCORE_FIELDS:
        ranked = rank_candidates(candidates, gt_bbox, field)
        if not ranked:
            raise RuntimeError("eligible GT word has no geometrically associated candidates")
        ranks = {
            value["candidate_id"]: rank for rank, value in enumerate(ranked, start=1)
        }
        first = ranked[0]
        first_iou = ious[first["candidate_id"]]
        first_correct_rank = min(ranks[value] for value in correct_050)
        rankings[field] = {
            "associated_candidate_count": len(ranked),
            "top1_candidate_id": first["candidate_id"],
            "top1_bbox_iou": first_iou,
            "top1_ge_0_50": first_iou >= 0.50,
            "top3_ge_0_50": any(ious[value["candidate_id"]] >= 0.50 for value in ranked[:3]),
            "top5_ge_0_50": any(ious[value["candidate_id"]] >= 0.50 for value in ranked[:5]),
            "top1_ge_0_75": first_iou >= 0.75,
            "top1_parseq_exact_casefold": first["ocr"]["raw_text"].casefold() == gt_text.casefold(),
            "first_iou_ge_0_50_rank": first_correct_rank,
        }
    return {
        "gt_bbox": tuple(int(value) for value in gt_bbox),
        "gt_text": gt_text,
        "candidate_ids_iou_ge_0_50": correct_050,
        "candidate_ids_iou_ge_0_75": correct_075,
        "candidate_ids_iou_ge_0_50_and_parseq_exact_casefold": simultaneous_exact,
        "exists_iou_ge_0_50_and_parseq_exact_casefold": bool(simultaneous_exact),
        "rankings": rankings,
    }


def aggregate_rankings(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("ranking aggregate requires eligible GT rows")
    output = {}
    for field in SCORE_FIELDS:
        values = [row["rankings"][field] for row in rows]
        output[field] = {
            "eligible_gt_words": len(values),
            "top1_ge_0_50": sum(value["top1_ge_0_50"] for value in values),
            "top1_ge_0_50_rate": sum(value["top1_ge_0_50"] for value in values) / len(values),
            "top3_ge_0_50": sum(value["top3_ge_0_50"] for value in values),
            "top3_ge_0_50_rate": sum(value["top3_ge_0_50"] for value in values) / len(values),
            "top5_ge_0_50": sum(value["top5_ge_0_50"] for value in values),
            "top5_ge_0_50_rate": sum(value["top5_ge_0_50"] for value in values) / len(values),
            "top1_ge_0_75": sum(value["top1_ge_0_75"] for value in values),
            "top1_ge_0_75_rate": sum(value["top1_ge_0_75"] for value in values) / len(values),
            "top1_parseq_exact_casefold": sum(
                value["top1_parseq_exact_casefold"] for value in values
            ),
            "top1_parseq_exact_casefold_rate": sum(
                value["top1_parseq_exact_casefold"] for value in values
            ) / len(values),
            "exists_iou_ge_0_50_and_parseq_exact_casefold": sum(
                row["exists_iou_ge_0_50_and_parseq_exact_casefold"] for row in rows
            ),
            "exists_iou_ge_0_50_and_parseq_exact_casefold_rate": sum(
                row["exists_iou_ge_0_50_and_parseq_exact_casefold"] for row in rows
            ) / len(rows),
            "median_rank_first_iou_ge_0_50": float(median(
                value["first_iou_ge_0_50_rank"] for value in values
            )),
        }
    return output


__all__ = ["SCORE_FIELDS", "aggregate_rankings", "evaluate_gt_word", "rank_candidates"]
