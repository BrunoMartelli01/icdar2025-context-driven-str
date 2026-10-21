"""GT-blind crop variants and score selection for frozen Total-Text TRAIN masks.

This is an approximate diagnostic around the existing letterbox contour baseline.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from localize.multihypothesis_word_proposals import box_iou
from scripts.evaluate_totaltext_highres_contour_padding import contour_proposals
from segmentation.totaltext_image_frequency_augment import remap_letterbox_native_box


def match_anchors(baseline: Sequence[Any], alternative: Sequence[Any],
                  min_iou: float) -> dict[int, tuple[int, float]]:
    """One-to-one maximum-IoU greedy match, with stable rank tie breaks."""
    if not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0,1]")
    pairs = sorted(
        ((box_iou(a.bbox, b.bbox), a.rank, b.rank) for a in baseline for b in alternative),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = {}
    for iou, a_rank, b_rank in pairs:
        if iou < min_iou:
            break
        if a_rank not in used_a and b_rank not in used_b:
            matched[a_rank] = (b_rank, float(iou))
            used_a.add(a_rank)
            used_b.add(b_rank)
    return matched


def candidate_scope(row: dict[str, Any], scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "padding":
        return any(v["threshold"] == 0.5 for v in row["origins"])
    if scope == "threshold":
        return any(v["padding_native_pixels"] == 0 for v in row["origins"])
    raise ValueError("unknown candidate scope")


def select_per_anchor(candidates: Sequence[dict[str, Any]], *, score: str,
                      scope: str = "all", margin: float = 0.0) -> dict[int, str]:
    """Make one GT-blind decision per Total-Text anchor; ties keep baseline."""
    if margin < 0:
        raise ValueError("margin must be nonnegative")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[int(row["anchor_rank"])].append(row)
    choices = {}
    for anchor, rows in groups.items():
        base = rows[0]
        if not base["baseline"]:
            raise ValueError("first anchor candidate must be pad-0 threshold-0.5")
        eligible = [row for row in rows if candidate_scope(row, scope) and row["ocr"]["raw_text"]]
        if not eligible:
            choices[anchor] = base["candidate_id"]
            continue
        winner = max(eligible, key=lambda row: (float(row["scores"][score]), -rows.index(row)))
        base_score = float(base["scores"][score]) if base["ocr"]["raw_text"] else float("-inf")
        choices[anchor] = winner["candidate_id"] if (
            winner is not base and float(winner["scores"][score]) > base_score + margin
        ) else base["candidate_id"]
    return choices


def generate_totaltext_candidates(probability: np.ndarray, image: np.ndarray, *,
                                  thresholds: Sequence[float], pads: Sequence[int],
                                  min_anchor_iou: float = 0.3
                                  ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Variants of frozen native-256 letterbox contours on original RGB.

    Padding is measured in native mask pixels. Matching across thresholds uses
    tight native boxes and one-to-one association; unmatched contours are
    counted, not silently added as new anchors.
    """
    if probability.shape != (256, 256) or probability.dtype != np.float32:
        raise ValueError("native probability must be float32 256x256")
    if not np.isfinite(probability).all() or np.any(probability < 0) or np.any(probability > 1):
        raise ValueError("native probability outside [0,1]")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be RGB uint8")
    if len(set(thresholds)) != len(thresholds) or 0.5 not in thresholds or any(
            not 0 < t < 1 for t in thresholds):
        raise ValueError("threshold grid must contain unique probabilities and 0.5")
    if sorted(set(pads)) != list(pads) or 0 not in pads or any(p < 0 for p in pads):
        raise ValueError("padding grid must be sorted, unique, nonnegative and include 0")
    if not 0 < min_anchor_iou <= 1:
        raise ValueError("min_anchor_iou must be in (0,1]")
    all_proposals = {}
    for threshold in thresholds:
        native = np.ascontiguousarray((probability >= threshold).astype(np.uint8))
        by_pad = contour_proposals(native, image.shape[:2], list(pads))
        for part in by_pad.values():
            for row in part["proposals"]:
                # contour_proposals defaults to a direct resize; this frozen
                # baseline requires inverse-letterbox remap instead.
                row["bbox"] = remap_letterbox_native_box(
                    row["padded_bbox_native"], (image.shape[1], image.shape[0]))
        all_proposals[threshold] = by_pad
    base = all_proposals[0.5]["0"]["proposals"]
    base_boxes = [SimpleNamespace(rank=row["rank"], bbox=row["bbox_native"]) for row in base]
    match_by_threshold = {}
    unmatched = {}
    for threshold in thresholds:
        current = all_proposals[threshold]["0"]["proposals"]
        if threshold == 0.5:
            match = {row["rank"]: (row["rank"], 1.0) for row in base}
        else:
            boxes = [SimpleNamespace(rank=row["rank"], bbox=row["bbox_native"])
                     for row in current]
            match = match_anchors(base_boxes, boxes, min_anchor_iou)
        match_by_threshold[threshold] = match
        unmatched[f"{threshold:.2f}"] = len(current) - len(match)
    candidates = []
    for anchor in base:
        by_box: dict[tuple[int, ...], dict[str, Any]] = {}
        for threshold in sorted(thresholds, key=lambda t: (t != 0.5, t)):
            matched = match_by_threshold[threshold].get(anchor["rank"])
            if matched is None:
                continue
            rank, iou = matched
            selected = next(row for row in all_proposals[threshold]["0"]["proposals"]
                            if row["rank"] == rank)
            contour_index = selected["contour_index"]
            for pad in sorted(pads, key=lambda p: (p != 0, p)):
                row = next(item for item in all_proposals[threshold][str(pad)]["proposals"]
                           if item["contour_index"] == contour_index)
                box = tuple(row["bbox"])
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise RuntimeError("empty inverse-letterbox crop")
                origin = {"threshold": threshold, "padding_native_pixels": pad,
                          "matched_tight_native_iou": iou}
                if box in by_box:
                    by_box[box]["origins"].append(origin)
                    continue
                by_box[box] = {"anchor_rank": anchor["rank"], "bbox": list(box),
                               "tight_bbox_native": list(selected["bbox_native"]),
                               "origins": [origin],
                               "baseline": threshold == 0.5 and pad == 0}
        rows = list(by_box.values())
        if not rows or not rows[0]["baseline"]:
            raise RuntimeError("Total-Text pad-0 anchor missing or reordered")
        for index, row in enumerate(rows):
            row["candidate_id"] = f"a{anchor['rank']:04d}:c{index:03d}"
            candidates.append(row)
    stats = {"baseline_anchors": len(base), "candidates": len(candidates),
             "unmatched_threshold_contours": unmatched,
             "raw_contours_by_threshold": {f"{t:.2f}": all_proposals[t]["0"]["raw_contours"]
                                            for t in thresholds}}
    return candidates, stats


def totaltext_oracle_one_per_anchor(candidates: Sequence[dict[str, Any]],
                                    words: Sequence[dict[str, Any]]) -> dict[int, str]:
    """GT-assisted one-to-one ceiling with one candidate per baseline anchor."""
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        groups[row["anchor_rank"]].append(row)
    options = {}
    for anchor, rows in groups.items():
        options[anchor] = [(gi, row["candidate_id"])
                           for gi, word in enumerate(words) for row in rows
                           if row["ocr"]["raw_text"]
                           and row["ocr"]["raw_text"].casefold() == word["text"].casefold()
                           and box_iou(row["bbox"], word["bbox"]) >= 0.5]
    owner: dict[int, tuple[int, str]] = {}

    def augment(anchor: int, visited: set[int]) -> bool:
        for gt_index, candidate_id in options[anchor]:
            if gt_index in visited:
                continue
            visited.add(gt_index)
            previous = owner.get(gt_index)
            if previous is None or augment(previous[0], visited):
                owner[gt_index] = (anchor, candidate_id)
                return True
        return False

    for anchor in sorted(groups):
        augment(anchor, set())
    choices = {anchor: rows[0]["candidate_id"] for anchor, rows in groups.items()}
    choices.update({anchor: candidate_id for anchor, candidate_id in owner.values()})
    return choices
