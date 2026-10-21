import pytest

from recognition.parseq_candidate_ranking import (
    SCORE_FIELDS,
    aggregate_rankings,
    evaluate_gt_word,
    rank_candidates,
)


def _candidate(candidate_id, bbox, text, sequence, mean_log, minimum):
    return {
        "candidate_id": candidate_id,
        "bbox": bbox,
        "ocr": {
            "raw_text": text,
            "sequence_confidence_native": sequence,
            "mean_token_log_probability": mean_log,
            "minimum_token_probability": minimum,
        },
    }


def test_ranking_uses_one_score_and_only_positive_geometry_association():
    candidates = (
        _candidate("a", (0, 0, 10, 10), "wrong", .2, -.1, .4),
        _candidate("b", (2, 2, 8, 8), "word", .9, -.3, .2),
        _candidate("outside", (20, 20, 30, 30), "word", 1., 0., 1.),
    )
    assert [value["candidate_id"] for value in rank_candidates(
        candidates, (0, 0, 10, 10), "sequence_confidence_native"
    )] == ["b", "a"]
    assert [value["candidate_id"] for value in rank_candidates(
        candidates, (0, 0, 10, 10), "mean_token_log_probability"
    )] == ["a", "b"]


def test_gt_evaluation_and_aggregate_have_frozen_denominator():
    candidates = (
        _candidate("high-wrong", (0, 0, 6, 10), "wo", .9, -.1, .8),
        _candidate("correct", (0, 0, 10, 10), "WORD", .8, -.2, .7),
    )
    row = evaluate_gt_word(candidates, gt_bbox=(0, 0, 10, 10), gt_text="word")
    assert row is not None
    assert row["candidate_ids_iou_ge_0_50"] == ("high-wrong", "correct")
    assert row["exists_iou_ge_0_50_and_parseq_exact_casefold"] is True
    assert row["rankings"]["sequence_confidence_native"]["top1_ge_0_50"] is True
    assert row["rankings"]["sequence_confidence_native"]["top1_parseq_exact_casefold"] is False
    assert row["rankings"]["sequence_confidence_native"]["first_iou_ge_0_50_rank"] == 1
    aggregate = aggregate_rankings([row])
    assert set(aggregate) == set(SCORE_FIELDS)
    assert aggregate["sequence_confidence_native"]["eligible_gt_words"] == 1
    assert aggregate["sequence_confidence_native"]["median_rank_first_iou_ge_0_50"] == 1


def test_gt_without_iou_half_candidate_is_excluded():
    candidates = (_candidate("fragment", (0, 0, 2, 2), "x", .9, -.1, .8),)
    assert evaluate_gt_word(candidates, gt_bbox=(0, 0, 10, 10), gt_text="word") is None


def test_unknown_score_and_empty_aggregate_fail_closed():
    with pytest.raises(ValueError, match="unknown"):
        rank_candidates([], (0, 0, 1, 1), "combined")
    with pytest.raises(ValueError, match="eligible"):
        aggregate_rankings([])
