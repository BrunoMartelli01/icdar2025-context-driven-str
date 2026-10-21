"""Pure helpers for the APPROXIMATE faithful-vs-practical E2E diagnostic."""

from __future__ import annotations

from collections import Counter
from typing import Any, Final, Sequence

from rapidfuzz.fuzz import ratio

from localize.practical_candidate_selection import box_iou, intersection_area
from localize.practical_remap_protocol import proposal_geometry


CLASSIFICATION: Final = "PRACTICAL_APPROXIMATION_DIAGNOSTIC"
PAPER_ICDAR13_R50_ACCURACY_PERCENT: Final = 88.8


def repository_route(
    *, t1: str, caption: str, s1: float,
    candidates: Sequence[dict[str, Any]], alpha: float = 0.6,
    beta: float = 0.4, tau: float = 0.8,
) -> dict[str, Any]:
    """Apply the separately frozen author-repository crop/scoring interpretation."""
    if (alpha, beta, tau) != (0.6, 0.4, 0.8):
        raise ValueError("repository route is frozen at alpha=.6, beta=.4, tau=.8")
    l1 = ratio(t1, caption) / 100.0
    c1 = alpha * s1 + beta * l1
    ranked = []
    for row in candidates:
        s3 = float(row["semantic_similarity"])
        l3 = ratio(str(row["text"]), caption) / 100.0
        ranked.append({**row, "lexical_similarity": l3, "paired_confidence": alpha * s3 + beta * l3})
    selected = max(
        enumerate(ranked), key=lambda value: (value[1]["paired_confidence"], -value[0]),
        default=(None, None),
    )[1]
    if selected is None:
        s3, l3, c3, t3 = -1.0, 0.0, -0.6, ""
    else:
        s3 = float(selected["semantic_similarity"])
        l3 = float(selected["lexical_similarity"])
        c3 = float(selected["paired_confidence"])
        t3 = str(selected["text"])
    confidence = max(c1, c3)
    semantic_winner = "T1" if s1 >= s3 else "T3"
    return {
        "T1": t1, "T2": caption, "T3": t3,
        "S1": float(s1), "S3": s3, "L1": l1, "L3": l3,
        "C1": c1, "C3": c3, "C": confidence,
        "selected_crop": selected,
        "semantic_winner": semantic_winner,
        "pre_fallback_output": t1 if semantic_winner == "T1" else t3,
        "needs_fallback": confidence < tau,
        "strategy": "REPOSITORY_SPECIFIED: first maximum C3; threshold max(C1,C3); semantic winner output",
    }


def multiset_exact(predictions: Sequence[str], references: Sequence[str]) -> dict[str, Any]:
    """Case-folded one-to-one multiset overlap; an explicit diagnostic, not paper accuracy."""
    predicted = Counter(value.casefold() for value in predictions if value)
    expected = Counter(value.casefold() for value in references if value)
    matches = sum((predicted & expected).values())
    predicted_count, reference_count = sum(predicted.values()), sum(expected.values())
    precision = matches / predicted_count if predicted_count else 0.0
    recall = matches / reference_count if reference_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "matches": matches, "predictions": predicted_count, "references": reference_count,
        "precision": precision, "recall": recall, "f1": f1,
    }


def word_crop_diagnostic(
    proposals_by_image: Sequence[Sequence[dict[str, Any]]],
    gt_by_image: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    """Best-geometric-crop OCR and proposal geometry, with GT used post hoc only."""
    if len(proposals_by_image) != len(gt_by_image):
        raise ValueError("proposal/GT image populations differ")
    counts = {"iou_ge_0_25": 0, "iou_ge_0_50": 0, "iou_ge_0_75": 0,
              "exact_casefold": 0, "fuzzy_ge_0_8": 0, "available": 0}
    proposal_total = sum(len(rows) for rows in proposals_by_image)
    for proposals, words in zip(proposals_by_image, gt_by_image):
        for word in words:
            best = max(proposals, key=lambda row: box_iou(row["bbox"], word["bbox"]), default=None)
            iou = box_iou(best["bbox"], word["bbox"]) if best else 0.0
            for threshold, field in (
                (0.25, "iou_ge_0_25"), (0.50, "iou_ge_0_50"), (0.75, "iou_ge_0_75"),
            ):
                counts[field] += int(iou >= threshold)
            if best is not None and intersection_area(best["bbox"], word["bbox"]) > 0:
                text = str(best["text"])
                counts["available"] += 1
                counts["exact_casefold"] += int(text.casefold() == str(word["text"]).casefold())
                counts["fuzzy_ge_0_8"] += int(
                    ratio(text.casefold(), str(word["text"]).casefold()) / 100.0 >= 0.8
                )
    gt_total = sum(len(rows) for rows in gt_by_image)
    geometry = proposal_geometry(
        (
            ([row["bbox"] for row in proposals], [word["bbox"] for word in words],
             len(proposals), 0, 0)
            for proposals, words in zip(proposals_by_image, gt_by_image)
        )
    ).as_dict()
    for field in ("gt_with_proposal_iou_ge_0_25", "gt_with_proposal_iou_ge_0_50", "gt_with_proposal_iou_ge_0_75"):
        expected = counts[field.replace("gt_with_proposal_", "")]
        if geometry[field] != expected:
            raise RuntimeError("best-IoU geometry aggregation mismatch")
    return {
        **geometry, **counts,
        "exact_rate_gt_denominator": counts["exact_casefold"] / gt_total,
        "fuzzy_rate_gt_denominator": counts["fuzzy_ge_0_8"] / gt_total,
    }


__all__: Final = [
    "CLASSIFICATION", "PAPER_ICDAR13_R50_ACCURACY_PERCENT", "multiset_exact",
    "repository_route", "word_crop_diagnostic",
]
