from __future__ import annotations

import inspect

from localize.practical_candidate_selection import (
    VARIANTS, attach_family_consensus, percentile_ranks, select_candidates,
    span_boundary_score, weighted_interval_selection,
)


def _candidate(index, box, score, *, threshold=0.5):
    return {
        "candidate_id": f"0001:{index:05d}", "candidate_index": index,
        "bbox": list(box), "source_bbox_native": list(box),
        "source_threshold": threshold, "bbox_variant": "standard_pad6",
        "foreground_area_native": 10, "foreground_density": 0.5,
        "boundary_score": 0.5, "v1_support": 0.5, "context_score": 0.0,
        "ocr": {"mean_token_log_probability": score, "raw_text": "x"},
    }


def test_percentile_ranks_are_stable_mid_ranks():
    assert percentile_ranks([2.0, 1.0, 2.0]) == (0.75, 0.0, 0.75)


def test_boundary_score_rewards_large_external_and_small_internal_gaps():
    boxes = ((0, 0, 2, 4), (3, 0, 5, 4), (12, 0, 14, 4))
    assert span_boundary_score(component_boxes=boxes, start=0, end=2) > span_boundary_score(
        component_boxes=boxes, start=1, end=3
    )


def test_family_consensus_rewards_all_threshold_support():
    rows = [_candidate(i, (0, 0, 4, 4), -0.2, threshold=t) for i, t in enumerate((0.2, 0.3, 0.4, 0.5), 1)]
    rows.append(_candidate(5, (20, 0, 24, 4), -0.2, threshold=0.5))
    attach_family_consensus(rows)
    assert rows[0]["consensus_score"] == 1.0
    assert rows[-1]["consensus_score"] == 0.25


def test_weighted_interval_selection_rejects_overlapping_lower_utility():
    rows = [_candidate(1, (0, 0, 6, 4), -0.1), _candidate(2, (4, 0, 10, 4), -0.1), _candidate(3, (10, 0, 14, 4), -0.1)]
    for row, score in zip(rows, (0.9, 0.7, 0.8)):
        row["variant_scores"] = {variant: score for variant in VARIANTS}
    assert weighted_interval_selection(rows, (0, 1, 2), variant="OCR_FIRST") == (0, 2)


def test_selector_is_anchor_local_deterministic_and_capped():
    rows = []
    for index in range(12):
        row = _candidate(index + 1, (index * 10, 0, index * 10 + 6, 5), -0.01 * index)
        rows.append(row)
    anchors = [(index * 10, 0, index * 10 + 6, 5) for index in range(12)]
    first = select_candidates([dict(row) for row in rows], anchors=anchors, variant="OCR_FIRST")
    second = select_candidates([dict(row) for row in rows], anchors=anchors, variant="OCR_FIRST")
    assert len(first) <= 10
    assert [row["candidate_id"] for row in first] == [row["candidate_id"] for row in second]


def test_selector_api_has_no_ground_truth_argument():
    assert not ({"gt", "ground_truth", "transcription"} & set(inspect.signature(select_candidates).parameters))

