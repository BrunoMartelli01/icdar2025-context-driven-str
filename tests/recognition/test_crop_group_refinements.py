import numpy as np
import pytest

from recognition.crop_group_refinements import (confidence_guarded_selection,
                                                consolidate_near_identical_outputs,
                                                robust_group_evidence,
                                                robust_reference_bbox)
from recognition.group_word_selection import ProbabilityMass, group_evidence, select_groups


def crop(cid, box, text, probability=.9):
    return {"candidate_id": cid, "bbox": box, "methods": ["generator"],
            "origins": [{"method": "generator", "threshold": .5}],
            "ocr": {"raw_text": text,
                    "token_probabilities": [probability] * len(text),
                    "token_log_probabilities": [float(np.log(probability))] * len(text)}}


def test_robust_reference_can_recover_a_full_word_from_partial_medoid():
    rows = [crop("full", [0, 0, 10, 4], "word"),
            crop("partial", [2, 0, 10, 4], "ord")]
    group = {"group_id": "g", "candidate_ids": ["full", "partial"],
             "core_candidate_ids": ["full", "partial"],
             "reference_candidate_id": "partial", "reference_bbox": [2, 0, 10, 4]}
    p = ProbabilityMass(np.ones((10, 10), np.float32), (10, 10))
    original = group_evidence(rows, [group], allowed_methods={"generator"}, mass=p)
    assert select_groups(original, "mask_completeness_confidence")[0]["raw_text"] == "ord"
    robust, references = robust_group_evidence(original, [group], rows, p, quantile=.25)
    assert references["g"] == [0.5, 0., 10., 4.]
    assert select_groups(robust, "robust_mask_confidence")[0]["raw_text"] == "word"
    assert "robust_mask_confidence" not in original[0]["candidates"]["full"]["scores"]


def test_reference_uses_core_boxes_without_shared_multiword_parent_or_ocr():
    rows = [crop("left", [0, 0, 10, 5], "wrong"),
            crop("close", [1, 0, 11, 5], "other"),
            crop("parent", [0, 0, 25, 5], "two words")]
    group = {"group_id": "g", "candidate_ids": [r["candidate_id"] for r in rows],
             "core_candidate_ids": ["left", "close"],
             "reference_candidate_id": "left", "reference_bbox": [0, 0, 10, 5]}
    p = ProbabilityMass(np.ones((25, 25), np.float32), (25, 5))
    original = group_evidence(rows, [group], allowed_methods={"generator"}, mass=p)
    _, references = robust_group_evidence(original, [group], rows, p, quantile=.25)
    assert references["g"] == [0.25, 0., 10.75, 5.]
    for row in rows:
        row["ground_truth"] = "secret"
        row["ocr"]["raw_text"] = "arbitrary"
    _, again = robust_group_evidence(original, [group], list(reversed(rows)), p, quantile=.25)
    assert again == references


def output(gid, word, box, score):
    return {"group_id": gid, "candidate_id": gid, "word": word,
            "raw_text": word, "bbox": box, "score": score}


def test_consolidation_requires_nearly_identical_location_and_reading():
    rows = [output("a", "glass", [0, 0, 100, 20], .9),
            output("b", "glas", [1, 0, 101, 20], .8),
            output("c", "glass", [30, 0, 130, 20], .7),
            output("d", "metal", [1, 0, 101, 20], .6)]
    kept, rejected = consolidate_near_identical_outputs(rows, box_iou_min=.9,
                                                        min_edit_length=4)
    assert [row["group_id"] for row in kept] == ["a", "c", "d"]
    assert rejected == [{"group_id": "b", "candidate_id": "b",
                         "winner_group_id": "a", "winner_candidate_id": "a",
                         "box_iou": pytest.approx(99 / 101),
                         "reason": "one_character_variant"}]


def test_consolidation_preserves_short_distinct_words_and_validates_inputs():
    rows = [output("a", "in", [0, 0, 100, 20], .9),
            output("b", "on", [0, 0, 100, 20], .8)]
    kept, rejected = consolidate_near_identical_outputs(rows, box_iou_min=.9,
                                                        min_edit_length=4)
    assert kept == rows and rejected == []
    with pytest.raises(ValueError):
        robust_reference_bbox([], quantile=.25)
    with pytest.raises(ValueError):
        robust_reference_bbox([[0, 0, 10, 10]], quantile=.5)
    with pytest.raises(ValueError):
        consolidate_near_identical_outputs(rows + rows[:1], box_iou_min=.9,
                                           min_edit_length=4)


def test_confidence_guard_switches_only_if_ocr_confidence_does_not_fall():
    base = [output("g", "partial", [0, 0, 10, 5], .8)]
    new = [dict(output("g", "complete", [0, 0, 12, 5], .9), candidate_id="new")]
    evidence = [{"group_id": "g", "candidates": {
        "g": {"scores": {"confidence_balanced": .8, "robust_mask_confidence": .75}},
        "new": {"scores": {"confidence_balanced": .9, "robust_mask_confidence": .85}}}}]
    chosen, audit = confidence_guarded_selection(base, new, evidence)
    assert chosen[0]["candidate_id"] == "new"
    assert audit["switched_from_baseline"] == 1
    evidence[0]["candidates"]["new"]["scores"]["confidence_balanced"] = .7
    chosen, audit = confidence_guarded_selection(base, new, evidence)
    assert chosen[0]["candidate_id"] == "g"
    assert chosen[0]["score"] == .75
    assert audit["switched_from_baseline"] == 0


def test_confidence_guard_reverts_colliding_crop_assignments():
    base = [output("a", "alpha", [0, 0, 10, 5], .8),
            dict(output("b", "beta", [8, 0, 18, 5], .8), candidate_id="shared")]
    new = [dict(output("a", "beta", [8, 0, 18, 5], .9), candidate_id="shared"),
           output("b", "beta", [8, 0, 18, 5], .8)]
    evidence = [
        {"group_id": "a", "candidates": {
            "a": {"scores": {"confidence_balanced": .8, "robust_mask_confidence": .8}},
            "shared": {"scores": {"confidence_balanced": .9, "robust_mask_confidence": .9}}}},
        {"group_id": "b", "candidates": {
            "shared": {"scores": {"confidence_balanced": .8, "robust_mask_confidence": .8}},
            "b": {"scores": {"confidence_balanced": .7, "robust_mask_confidence": .8}}}},
    ]
    chosen, audit = confidence_guarded_selection(base, new, evidence)
    assert [item["candidate_id"] for item in chosen] == ["a", "shared"]
    assert audit["conflict_reverted_group_ids"] == ["a", "b"]
    with pytest.raises(ValueError):
        confidence_guarded_selection(base, new[:1], evidence)
