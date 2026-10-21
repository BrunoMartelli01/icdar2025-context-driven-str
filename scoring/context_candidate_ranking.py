"""GT-blind context scores and post-hoc ranking diagnostics for word proposals."""

from __future__ import annotations

from statistics import median
from typing import Any, Final, Sequence

import numpy as np

from localize.multihypothesis_word_proposals import box_iou


CONTEXT_SCORE_FIELDS: Final = (
    "semantic_similarity",
    "lexical_similarity",
    "fixed_context_score",
)
BASELINE_SCORE_FIELD: Final = "mean_token_log_probability"
ALL_SCORE_FIELDS: Final = CONTEXT_SCORE_FIELDS + (BASELINE_SCORE_FIELD,)


def intersection_area(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def candidate_score(candidate: dict[str, Any], field: str) -> float | None:
    if field in CONTEXT_SCORE_FIELDS:
        value = candidate["context_scores"][field]
    elif field == BASELINE_SCORE_FIELD:
        value = candidate["ocr"][field]
    else:
        raise ValueError(f"unknown ranking score: {field}")
    return None if value is None else float(value)


def rank_associated(
    candidates: Sequence[dict[str, Any]], gt_bbox: Sequence[int], field: str,
) -> tuple[dict[str, Any], ...]:
    """Rank only positive-intersection candidates; GT never enters score values."""
    associated = [row for row in candidates if intersection_area(row["bbox"], gt_bbox) > 0]

    def key(row: dict[str, Any]) -> tuple[float, str]:
        value = candidate_score(row, field)
        sortable = value if value is not None and np.isfinite(value) else float("-inf")
        return -sortable, str(row["candidate_id"])

    return tuple(sorted(associated, key=key))


def evaluate_word(
    candidates: Sequence[dict[str, Any]], *, gt_bbox: Sequence[int], gt_text: str,
) -> dict[str, Any] | None:
    ious = {row["candidate_id"]: box_iou(row["bbox"], gt_bbox) for row in candidates}
    correct_050 = {candidate_id for candidate_id, value in ious.items() if value >= 0.50}
    if not correct_050:
        return None
    correct_exact = {
        row["candidate_id"]
        for row in candidates
        if ious[row["candidate_id"]] >= 0.50
        and row["ocr"]["raw_text"].casefold() == gt_text.casefold()
    }
    rankings: dict[str, Any] = {}
    for field in ALL_SCORE_FIELDS:
        ranked = rank_associated(candidates, gt_bbox, field)
        if not ranked:
            raise RuntimeError("eligible GT word has no associated proposal")
        rank_by_id = {
            row["candidate_id"]: index for index, row in enumerate(ranked, start=1)
        }
        top = ranked[0]
        top_iou = ious[top["candidate_id"]]
        rankings[field] = {
            "associated_candidate_count": len(ranked),
            "top1_candidate_id": top["candidate_id"],
            "top1_bbox_iou": top_iou,
            "top1_ge_0_50": top_iou >= 0.50,
            "top3_ge_0_50": any(ious[row["candidate_id"]] >= 0.50 for row in ranked[:3]),
            "top5_ge_0_50": any(ious[row["candidate_id"]] >= 0.50 for row in ranked[:5]),
            "top1_ge_0_75": top_iou >= 0.75,
            "top1_parseq_exact_casefold": (
                top["ocr"]["raw_text"].casefold() == gt_text.casefold()
            ),
            "top1_exact_among_iou_ge_0_50_exact_eligible": (
                top["candidate_id"] in correct_exact
            ),
            "first_iou_ge_0_50_rank": min(rank_by_id[value] for value in correct_050),
        }
    return {
        "gt_bbox": [int(value) for value in gt_bbox],
        "gt_text": gt_text,
        "exists_iou_ge_0_50_and_parseq_exact_casefold": bool(correct_exact),
        "rankings": rankings,
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("ranking aggregate requires eligible GT words")
    exact_eligible = [
        row for row in rows if row["exists_iou_ge_0_50_and_parseq_exact_casefold"]
    ]
    output: dict[str, Any] = {}
    for field in ALL_SCORE_FIELDS:
        values = [row["rankings"][field] for row in rows]
        counts = {
            name: sum(bool(value[name]) for value in values)
            for name in (
                "top1_ge_0_50", "top3_ge_0_50", "top5_ge_0_50",
                "top1_ge_0_75", "top1_parseq_exact_casefold",
            )
        }
        exact_selected = sum(
            bool(row["rankings"][field]["top1_exact_among_iou_ge_0_50_exact_eligible"])
            for row in exact_eligible
        )
        exact_text_selected = sum(
            bool(row["rankings"][field]["top1_parseq_exact_casefold"])
            for row in exact_eligible
        )
        output[field] = {
            "eligible_gt_words": len(values),
            **{
                name: count
                for name, count in counts.items()
            },
            **{
                f"{name}_rate": count / len(values)
                for name, count in counts.items()
            },
            "exact_selectable_gt_words": len(exact_eligible),
            "exact_selectable_top1_parseq_exact": exact_text_selected,
            "exact_selectable_top1_parseq_exact_rate": exact_text_selected / len(exact_eligible),
            "exact_selectable_selected_top1": exact_selected,
            "exact_selectable_selected_top1_rate": exact_selected / len(exact_eligible),
            "median_rank_first_iou_ge_0_50": float(median(
                value["first_iou_ge_0_50_rank"] for value in values
            )),
        }
    return output


def flip_diagnostics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare context-only rankings against the fixed mean-log-prob baseline."""
    baseline = BASELINE_SCORE_FIELD
    output: dict[str, Any] = {}
    for field in CONTEXT_SCORE_FIELDS:
        metrics: dict[str, Any] = {}
        for outcome in ("top1_ge_0_50", "top1_parseq_exact_casefold"):
            improved = sum(
                not row["rankings"][baseline][outcome]
                and row["rankings"][field][outcome]
                for row in rows
            )
            worsened = sum(
                row["rankings"][baseline][outcome]
                and not row["rankings"][field][outcome]
                for row in rows
            )
            metrics[outcome] = {
                "improved": improved,
                "worsened": worsened,
                "net": improved - worsened,
                "unchanged": len(rows) - improved - worsened,
            }
        output[field] = metrics
    return output


def distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(array.max()),
    }


__all__ = [
    "ALL_SCORE_FIELDS", "BASELINE_SCORE_FIELD", "CONTEXT_SCORE_FIELDS",
    "aggregate", "candidate_score", "distribution", "evaluate_word",
    "flip_diagnostics", "intersection_area", "rank_associated",
]
