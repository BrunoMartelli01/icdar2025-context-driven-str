from __future__ import annotations

import inspect

from localize.practical_candidate_selection_v2 import select_anchor_complete


def _row(candidate_id, box, score, source):
    return {
        "candidate_id": candidate_id, "bbox": list(box), "tight_bbox": list(box),
        "source": source, "foreground_area_native": 10,
        "foreground_density": 0.5, "consensus_score": 0.5,
        "boundary_score": 0.5, "context_score": 0.0,
        "ocr": {"mean_token_log_probability": score, "raw_text": candidate_id},
    }


def test_anchor_mean_prefers_one_strong_whole_word_over_two_weaker_fragments():
    anchors = [
        _row("left", (0, 0, 9, 10), 0.6, "v1_exact"),
        _row("right", (11, 0, 20, 10), 0.6, "v1_exact"),
    ]
    whole = _row("whole", (0, 0, 20, 10), 0.9, "multihypothesis_alternative")
    selected = select_anchor_complete(anchors, [whole], variant="OCR_ONLY_ANCHOR_MEAN")
    assert [row["candidate_id"] for row in selected] == ["whole"]


def test_anchor_mean_keeps_two_strong_fragments_over_weaker_merge():
    anchors = [
        _row("left", (0, 0, 9, 10), 0.9, "v1_exact"),
        _row("right", (11, 0, 20, 10), 0.9, "v1_exact"),
    ]
    whole = _row("whole", (0, 0, 20, 10), 0.6, "multihypothesis_alternative")
    selected = select_anchor_complete(anchors, [whole], variant="OCR_ONLY_ANCHOR_MEAN")
    assert {row["candidate_id"] for row in selected} == {"left", "right"}


def test_exact_v1_fallback_makes_partition_complete():
    anchor = _row("base", (0, 0, 10, 10), 0.1, "v1_exact")
    selected = select_anchor_complete([anchor], [], variant="BALANCED_ANCHOR_MEAN")
    assert [row["candidate_id"] for row in selected] == ["base"]


def test_alternative_must_contain_complete_unpadded_anchor():
    anchor = _row("base", (0, 0, 10, 10), 0.1, "v1_exact")
    fragment = _row("fragment", (4, 0, 10, 10), 0.9, "multihypothesis_alternative")
    selected = select_anchor_complete([anchor], [fragment], variant="OCR_ONLY_ANCHOR_MEAN")
    assert [row["candidate_id"] for row in selected] == ["base"]


def test_selector_has_no_ground_truth_argument():
    assert not ({"gt", "ground_truth", "transcription"} & set(inspect.signature(select_anchor_complete).parameters))
