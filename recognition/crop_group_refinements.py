"""GT-blind refinements for approximate crop-group diagnostics.

These rules are separate from the paper's localization and recognition path.
They consume frozen candidate geometry, PARSeq scores, and probability mass.
"""
from __future__ import annotations

import math

import numpy as np

from recognition.group_word_selection import completeness, geometric_mean
from recognition.scene_text_v1_evaluation import box_iou


def robust_reference_bbox(core_boxes: list[list[int]], *, quantile: float) -> list[float]:
    """Inner quartile envelope of core crop boxes, independent of their readings."""
    if not core_boxes or not math.isfinite(quantile) or not 0 <= quantile < .5:
        raise ValueError("expected core boxes and quantile in [0, 0.5)")
    boxes = np.asarray(core_boxes, dtype=np.float64)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all()
        or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError("expected finite nonempty xyxy core boxes")
    reference = [float(np.quantile(boxes[:, 0], quantile)),
                 float(np.quantile(boxes[:, 1], quantile)),
                 float(np.quantile(boxes[:, 2], 1 - quantile)),
                 float(np.quantile(boxes[:, 3], 1 - quantile))]
    if reference[2] <= reference[0] or reference[3] <= reference[1]:
        raise ValueError("robust reference is empty")
    return reference


def robust_group_evidence(evidence: list[dict], groups: list[dict], rows: list[dict],
                          mass, *, quantile: float) -> tuple[list[dict], dict[str, list[float]]]:
    """Add a score against a robust group envelope without changing frozen evidence.

    The envelope is built from core proposals, before multiword parent crops are
    shared with multiple groups. OCR text, confidence, and GT do not choose it.
    """
    by_group = {group["group_id"]: group for group in groups}
    by_row = {row["candidate_id"]: row for row in rows}
    if len(by_group) != len(groups) or len(by_row) != len(rows):
        raise ValueError("duplicate group or candidate ID")
    updated, references = [], {}
    for group in evidence:
        gid = group["group_id"]
        if gid not in by_group:
            raise ValueError("evidence references unknown group")
        core = by_group[gid]["core_candidate_ids"]
        if not core or any(cid not in by_row for cid in core):
            raise ValueError("missing core crop")
        reference = robust_reference_bbox([by_row[cid]["bbox"] for cid in core],
                                          quantile=quantile)
        references[gid] = reference
        candidates = {}
        for cid, candidate in group["candidates"].items():
            if cid not in by_row or cid not in by_group[gid]["candidate_ids"]:
                raise ValueError("evidence references crop outside group")
            scores = dict(candidate["scores"])
            support = completeness(candidate["bbox"], reference, mass)
            scores["robust_mask_support"] = support
            scores["robust_mask_confidence"] = geometric_mean([
                scores["confidence_balanced"], support])
            candidates[cid] = {**candidate, "scores": scores}
        updated.append({**group, "candidates": candidates})
    return updated, references


def confidence_guarded_selection(baseline: list[dict], robust: list[dict],
                                 robust_evidence: list[dict]) -> tuple[list[dict], dict]:
    """Accept a robust crop switch only if its PARSeq confidence does not fall.

    Both inputs are valid one-crop-per-group assignments. A mixed assignment can
    reuse one original crop, so colliding groups revert to their baseline picks
    until every crop ID is unique. Scores are put on the same robust-reference
    scale for subsequent duplicate consolidation.
    """
    base = {item["group_id"]: item for item in baseline}
    proposed = {item["group_id"]: item for item in robust}
    evidence = {group["group_id"]: group for group in robust_evidence}
    if (len(base) != len(baseline) or len(proposed) != len(robust)
        or len(evidence) != len(robust_evidence) or base.keys() != proposed.keys()
        or any(len({item["candidate_id"] for item in rows}) != len(rows)
               for rows in (baseline, robust))):
        raise ValueError("invalid original group/crop assignments")
    selected = {}
    for gid in sorted(base):
        if gid not in evidence:
            raise ValueError("selected group lacks robust evidence")
        candidates = evidence[gid]["candidates"]
        old_id, new_id = base[gid]["candidate_id"], proposed[gid]["candidate_id"]
        if old_id not in candidates or new_id not in candidates:
            raise ValueError("selected crop lacks robust evidence")
        old_confidence = candidates[old_id]["scores"]["confidence_balanced"]
        new_confidence = candidates[new_id]["scores"]["confidence_balanced"]
        if any(not math.isfinite(value) or not 0 <= value <= 1
               for value in (old_confidence, new_confidence)):
            raise ValueError("invalid PARSeq confidence")
        selected[gid] = proposed[gid] if new_confidence >= old_confidence else base[gid]
    reverted = set()
    for _ in range(len(selected) + 1):
        owners = {}
        for gid, item in selected.items():
            owners.setdefault(item["candidate_id"], []).append(gid)
        collisions = [gids for gids in owners.values() if len(gids) > 1]
        if not collisions:
            break
        for gids in collisions:
            for gid in gids:
                selected[gid] = base[gid]
                reverted.add(gid)
    else:
        raise RuntimeError("crop assignment conflict did not resolve")
    result = []
    for gid, item in selected.items():
        score = evidence[gid]["candidates"][item["candidate_id"]]["scores"][
            "robust_mask_confidence"]
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid robust score")
        result.append({**item, "score": score})
    return result, {"switched_from_baseline": sum(
        selected[gid]["candidate_id"] != base[gid]["candidate_id"] for gid in selected),
        "conflict_reverted_group_ids": sorted(reverted)}


def _one_edit_apart(left: str, right: str, *, min_length: int) -> bool:
    if left == right or min(len(left), len(right)) < min_length:
        return False
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) == 1
    if len(left) > len(right):
        left, right = right, left
    i = j = edits = 0
    while i < len(left):
        if left[i] == right[j]:
            i += 1
            j += 1
        else:
            edits += 1
            j += 1
            if edits > 1:
                return False
    return True  # The single extra character may also be at the end.


def consolidate_near_identical_outputs(predictions: list[dict], *,
                                       box_iou_min: float, min_edit_length: int
                                       ) -> tuple[list[dict], list[dict]]:
    """Group nearly coincident output hypotheses, retaining the higher score.

    This is a conservative second grouping stage. It never merges boxes solely
    because they have similar text; a strong spatial agreement is also required.
    """
    if not math.isfinite(box_iou_min) or not 0 < box_iou_min <= 1 or min_edit_length < 1:
        raise ValueError("invalid consolidation parameters")
    ids = set()
    for item in predictions:
        if item["group_id"] in ids or not item["word"] or not math.isfinite(item["score"]):
            raise ValueError("invalid or duplicate selected group")
        ids.add(item["group_id"])
    ordered = sorted(predictions, key=lambda item: (-item["score"], item["group_id"],
                                                 item["candidate_id"]))
    retained, rejected = [], []
    for item in ordered:
        for winner in retained:
            overlap = box_iou(item["bbox"], winner["bbox"])
            if overlap < box_iou_min:
                continue
            same = item["word"] == winner["word"]
            if not same and not _one_edit_apart(item["word"], winner["word"],
                                                 min_length=min_edit_length):
                continue
            rejected.append({"group_id": item["group_id"],
                             "candidate_id": item["candidate_id"],
                             "winner_group_id": winner["group_id"],
                             "winner_candidate_id": winner["candidate_id"],
                             "box_iou": overlap,
                             "reason": "same_reading" if same else "one_character_variant"})
            break
        else:
            retained.append(item)
    kept_groups = {item["group_id"] for item in retained}
    return [item for item in predictions if item["group_id"] in kept_groups], rejected
