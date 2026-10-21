"""Group original crop hypotheses without consuming OCR or annotations.

Approximate diagnostic, separate from paper Algorithm 1. Exclusive xyxy boxes
must already be mapped into the same original image coordinate system.
"""
from __future__ import annotations

from hashlib import sha256
import json

import numpy as np


ALLOWED_METHODS = frozenset(("baseline", "threshold_tree", "tiled_segmentation"))


def geometry(boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    size = boxes[:, 2:] - boxes[:, :2]
    if not np.isfinite(boxes).all() or np.any(size <= 0):
        raise ValueError("expected finite nonempty exclusive xyxy boxes")
    area = size.prod(1)
    intersection = np.maximum(
        0, np.minimum(boxes[:, None, 2:], boxes[None, :, 2:])
        - np.maximum(boxes[:, None, :2], boxes[None, :, :2])
    ).prod(2)
    iou = intersection / (area[:, None] + area[None, :] - intersection)
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    displacement = (np.abs(center[:, None] - center[None, :])
                    / np.minimum(size[:, None], size[None, :])).max(2)
    return iou, displacement, intersection / area[None, :]


def _priority(row: dict) -> tuple:
    # Each view counts once, independent of how many reconstruction origins it has.
    views = set()
    thresholds = set()
    for origin in row.get("origins", []):
        if origin["method"] not in ALLOWED_METHODS:
            continue
        view = origin.get("tile_id", "whole")
        views.add(view)
        if "threshold" in origin:
            thresholds.add((view, origin["threshold"]))
    box = row["bbox"]
    return (-len(views), -len(thresholds),
            (box[2] - box[0]) * (box[3] - box[1]), row["candidate_id"])


def group_crops(rows: list[dict], *, strategy: str, iou_min: float,
                center_limit: float, ambiguity: dict) -> dict:
    """Build core groups, then share ambiguous crops across compatible groups.

    Complete link requires compatibility with EVERY member of a group. The
    connected strategy is a negative control exposing transitive bridges.
    Containment links flag competing split/merged hypotheses. A shared crop
    retains its original ID and reading: membership does not replicate inference.
    """
    if strategy not in ("complete_link", "connected"):
        raise ValueError("unknown grouping strategy")
    if not 0 < iou_min <= 1 or not np.isfinite(center_limit) or center_limit < 0:
        raise ValueError("invalid geometric compatibility parameters")
    if (not 0 < ambiguity["coverage_min"] <= 1
        or not 0 < ambiguity["child_area_min"] < ambiguity["child_area_max"] < 1
        or not 0 <= ambiguity["child_iou_max"] < 1
        or not 0 < ambiguity["combined_area_min"] <= 1):
        raise ValueError("invalid ambiguity parameters")
    ids = [row["candidate_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate IDs")
    for row in rows:
        if row.get("quad") is not None or row.get("neutral_fraction", 0) != 0:
            raise ValueError("final crop variations are not allowed")
        if not set(row["methods"]) & ALLOWED_METHODS:
            raise ValueError("expected only threshold/window/baseline candidates")
    boxes = np.array([row["bbox"] for row in rows], dtype=np.float64).reshape(-1, 4)
    iou, displacement, coverage = geometry(boxes)
    compatible = (iou >= iou_min) & (displacement <= center_limit)
    groups = []
    if strategy == "connected":
        unseen = set(range(len(rows)))
        while unseen:
            first = min(unseen, key=lambda index: ids[index])
            stack, members = [first], []
            unseen.remove(first)
            while stack:
                current = stack.pop()
                members.append(current)
                neighbors = sorted(unseen.intersection(np.flatnonzero(compatible[current])))
                unseen.difference_update(neighbors)
                stack.extend(neighbors)
            groups.append(members)
    else:
        for index in sorted(range(len(rows)), key=lambda index: _priority(rows[index])):
            eligible = [(float(iou[index, members].min()), group_index)
                        for group_index, members in enumerate(groups)
                        if compatible[index, members].all()]
            if eligible:
                # Earliest group wins exact ties; input order cannot change it.
                group_index = max(eligible, key=lambda pair: (pair[0], -pair[1]))[1]
                groups[group_index].append(index)
            else:
                groups.append([index])
    groups = sorted((sorted(members, key=lambda index: ids[index]) for members in groups),
                    key=lambda members: ids[members[0]])
    references, records = [], []
    for members in groups:
        median = np.median(boxes[members], axis=0)
        reference = min(members, key=lambda index: (
            float(np.abs(boxes[index] - median).sum()), ids[index]))
        references.append(reference)
        member_ids = [ids[index] for index in members]
        group_id = sha256(json.dumps(member_ids).encode()).hexdigest()
        records.append({"group_id": group_id, "candidate_ids": member_ids.copy(),
                        "core_candidate_ids": member_ids,
                        "reference_candidate_id": ids[reference],
                        "reference_bbox": rows[reference]["bbox"],
                        "minimum_member_iou": float(iou[np.ix_(members, members)].min())})
    # A parent overlapping two disjoint cores is a competing multiword hypothesis,
    # not evidence that the cores belong to the same word.
    area = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    ambiguous = []
    for index in range(len(rows)):
        child_groups = [g for g, reference in enumerate(references)
                        if coverage[index, reference] >= ambiguity["coverage_min"]
                        and ambiguity["child_area_min"] <= area[reference] / area[index]
                        <= ambiguity["child_area_max"]]
        split_pairs = [(left, right) for k, left in enumerate(child_groups)
                       for right in child_groups[k + 1:]
                       if iou[references[left], references[right]] <= ambiguity["child_iou_max"]
                       and (area[references[left]] + area[references[right]]) / area[index]
                       >= ambiguity["combined_area_min"]]
        if split_pairs:
            linked = sorted({group for pair in split_pairs for group in pair})
            ambiguous.append({"candidate_id": ids[index],
                              "competing_group_ids": [records[g]["group_id"] for g in linked]})
            for group in linked:
                if ids[index] not in records[group]["candidate_ids"]:
                    records[group]["candidate_ids"].append(ids[index])
    ambiguous.sort(key=lambda row: row["candidate_id"])
    parent_ids = {row["candidate_id"] for row in ambiguous}
    for record in records:
        record["candidate_ids"].sort()
        record["shared_candidate_ids"] = sorted(set(record["candidate_ids"])
                                               - set(record["core_candidate_ids"]))
        record["parent_only"] = set(record["core_candidate_ids"]) <= parent_ids
    return {"groups": records, "ambiguous_parents": ambiguous,
            "candidates": len(rows), "strategy": strategy,
            "uses_ocr": False, "uses_ground_truth": False}
