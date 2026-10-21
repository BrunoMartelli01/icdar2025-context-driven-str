"""Inference-only postfilters for frozen crop-group predictions.

These approximate experiments are separate from the paper's Algorithm 1.
All boxes use exclusive xyxy coordinates in the original image frame.
"""
from __future__ import annotations

import math

from recognition.group_word_selection import normalize


def _area(box: list[int]) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def _intersection(left: list[int], right: list[int]) -> list[int]:
    return [max(left[0], right[0]), max(left[1], right[1]),
            min(left[2], right[2]), min(left[3], right[3])]


def _iou(left: list[int], right: list[int]) -> float:
    common = _area(_intersection(left, right))
    union = _area(left) + _area(right) - common
    return common / union if union else 0.0


def _soft_iou(left: list[int], right: list[int], mass) -> float:
    common = mass(_intersection(left, right))
    union = mass(left) + mass(right) - common
    if union <= 0:
        # The source mask has no support at either box: geometry remains auditable.
        return _iou(left, right)
    return max(0.0, min(1.0, common / union))


def _index(predictions: list[dict], groups: list[dict], rows: list[dict]):
    by_group = {group["group_id"]: group for group in groups}
    by_row = {row["candidate_id"]: row for row in rows}
    if len(by_group) != len(groups) or len(by_row) != len(rows):
        raise ValueError("duplicate group or candidate ID")
    seen = set()
    for prediction in predictions:
        gid, cid = prediction["group_id"], prediction["candidate_id"]
        if gid in seen or gid not in by_group or cid not in by_row:
            raise ValueError("invalid selected prediction identity")
        if cid not in by_group[gid]["candidate_ids"]:
            raise ValueError("selected crop is absent from its group")
        score = prediction["score"]
        if not isinstance(score, (float, int)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid selected score")
        if normalize(prediction["raw_text"]) != prediction["word"]:
            raise ValueError("selected reading differs from its normalized word")
        seen.add(gid)
    return by_group, by_row


def abstain_uncorroborated(predictions: list[dict], groups: list[dict], rows: list[dict]):
    """Keep a baseline-anchored group or one with two distinct corroborating crops.

    Shared parent crops do not count as a group's independent core evidence.
    The rule has no word-length, dictionary, confidence or dataset cutoff.
    """
    by_group, by_row = _index(predictions, groups, rows)
    kept, rejected = [], []
    for prediction in predictions:
        group = by_group[prediction["group_id"]]
        core = [by_row[cid] for cid in group["core_candidate_ids"]]
        baseline = any("baseline" in row["methods"] for row in core)
        corroborating_crops = {row["crop_rgb_sha256"] for row in core
                               if normalize(row["ocr"]["raw_text"]) == prediction["word"]}
        if baseline or len(corroborating_crops) >= 2:
            kept.append(prediction)
        else:
            rejected.append({"group_id": prediction["group_id"],
                             "candidate_id": prediction["candidate_id"],
                             "reason": "no_baseline_or_distinct_crop_reading_corroboration",
                             "corroborating_crop_count": len(corroborating_crops)})
    return kept, rejected


def suppress_duplicate_regions(predictions: list[dict], groups: list[dict], rows: list[dict],
                               mass, *, box_iou_min: float, soft_iou_min: float):
    """Keep the best reading when two outputs occupy the same mask-supported region.

    Different readings linked by an explicitly shared multiword crop survive:
    those groups may be distinct words within one parent proposal.
    """
    if not 0 < box_iou_min <= 1 or not 0 < soft_iou_min <= 1:
        raise ValueError("duplicate overlap cutoffs must lie in (0,1]")
    by_group, _ = _index(predictions, groups, rows)
    ordered = sorted(predictions, key=lambda value: (-value["score"], value["group_id"],
                                                     value["candidate_id"]))
    kept, rejected = [], []
    for prediction in ordered:
        group = by_group[prediction["group_id"]]
        for winner in kept:
            winning_group = by_group[winner["group_id"]]
            shared = set(group["candidate_ids"]) & set(winning_group["candidate_ids"])
            if shared and prediction["word"] != winner["word"]:
                continue
            if _iou(prediction["bbox"], winner["bbox"]) < box_iou_min:
                continue
            if _soft_iou(prediction["bbox"], winner["bbox"], mass) < soft_iou_min:
                continue
            rejected.append({"group_id": prediction["group_id"],
                             "candidate_id": prediction["candidate_id"],
                             "reason": "duplicate_mask_region",
                             "winner_group_id": winner["group_id"]})
            break
        else:
            kept.append(prediction)
    # Preserve input/group order in result shards; greedy order is only for choice.
    ids = {prediction["group_id"] for prediction in kept}
    return [prediction for prediction in predictions if prediction["group_id"] in ids], rejected
