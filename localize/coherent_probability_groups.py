"""Two independent mask-only grouping diagnostics; no GT/OCR/model inputs."""
from dataclasses import dataclass

import numpy as np

from .grouped_word_blocks import _extract, compatible_components


@dataclass(frozen=True)
class CoherenceParams:
    max_gap_to_other_median: float
    minimum_positive_gaps: int

    def __post_init__(self):
        if not np.isfinite(self.max_gap_to_other_median) or self.max_gap_to_other_median < 1:
            raise ValueError("gap ratio must be finite >=1")
        if self.minimum_positive_gaps < 3:
            raise ValueError("spacing evidence needs >=3 gaps")


@dataclass(frozen=True)
class CorridorParams:
    min_mean_probability: float
    min_column_max_probability: float

    def __post_init__(self):
        if any(not np.isfinite(v) or not 0 <= v <= 1 for v in
               (self.min_mean_probability, self.min_column_max_probability)):
            raise ValueError("probability gates must be in [0,1]")


def coherent_group(members, v1, params):
    heights = [c.bbox[3] - c.bbox[1] for c in members]
    median_h = float(np.median(heights))
    core = [c for c, h in zip(members, heights) if h >= median_h / v1.max_height_ratio]
    core_h = [c.bbox[3] - c.bbox[1] for c in core]
    common = max(0, min(c.bbox[3] for c in core) - max(c.bbox[1] for c in core)) / min(core_h)
    spread = (max(c.centroid[1] for c in core) - min(c.centroid[1] for c in core)) / np.median(core_h)
    ratio = max(core_h) / min(core_h)
    boxes = sorted((c.bbox for c in members), key=lambda b: (b[0], b[1], b[2], b[3]))
    rightmost = boxes[0]
    gaps = []
    for box in boxes[1:]:
        if box[0] > rightmost[2]:
            gaps.append((box[0] - rightmost[2]) / median_h)
        if box[2] > rightmost[2]:
            rightmost = box
    spacing = None
    if len(gaps) >= params.minimum_positive_gaps:
        ordered = sorted(gaps)
        spacing = ordered[-1] / float(np.median(ordered[:-1]))
    reasons = []
    if common < v1.vertical_overlap_min: reasons.append("global_vertical_overlap")
    if spread > v1.max_center_y_ratio: reasons.append("global_center_y_spread")
    if ratio > v1.max_height_ratio: reasons.append("global_height_ratio")
    if spacing is not None and spacing > params.max_gap_to_other_median:
        reasons.append("global_spacing_outlier")
    return {"accepted": not reasons, "reasons": reasons,
            "common_vertical_overlap": float(common), "center_y_spread": float(spread),
            "height_ratio": float(ratio), "gap_to_other_median": spacing,
            "core_component_ids": [c.component_id for c in core]}


def corridor_metrics(left, right, support, params):
    a, b = sorted((left.bbox, right.bbox), key=lambda x: (x[0], x[1], x[2], x[3]))
    if b[0] <= a[2]:
        return {"accepted": True, "reason": "no_positive_horizontal_gap",
                "mean_probability": None, "min_column_max_probability": None}
    y1, y2 = max(a[1], b[1]), min(a[3], b[3])
    if y2 <= y1:
        return {"accepted": False, "reason": "no_shared_vertical_band",
                "mean_probability": None, "min_column_max_probability": None}
    patch = support[y1:y2, a[2]:b[0]]
    mean = float(patch.mean(dtype=np.float64))
    bottleneck = float(patch.max(axis=0).min())
    accepted = mean >= params.min_mean_probability and bottleneck >= params.min_column_max_probability
    return {"accepted": accepted, "reason": "supported_corridor" if accepted else "weak_corridor",
            "mean_probability": mean, "min_column_max_probability": bottleneck}


def group_components(mask, v1, *, mode, coherence, corridor, support=None):
    if mask.ndim != 2 or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError("binary uint8 mask required")
    if mode not in ("GROUP_COHERENCE", "PROBABILITY_CORRIDOR"):
        raise ValueError("unknown grouping mode")
    if mode == "PROBABILITY_CORRIDOR":
        if support is None or support.shape != mask.shape or not np.isfinite(support).all() or np.any(support < 0) or np.any(support > 1):
            raise ValueError("finite support map in [0,1] at mask resolution required")
        if not np.array_equal(support >= .5, mask > 0):
            raise ValueError("support must reproduce the frozen binary mask at .5")
    raw = _extract(mask, v1.connectivity)
    by_id = {c.component_id: c for c in raw}
    edges = []
    for i, left in enumerate(raw):
        for right in raw[i + 1:]:
            edge = compatible_components(left, right, v1)
            if edge is not None: edges.append(edge)
    edges.sort(key=lambda e: (e["horizontal_gap_over_reference_height"],
        e["center_y_distance_over_reference_height"], e["left_component_id"], e["right_component_id"]))
    owner = {i: i for i in by_id}
    groups = {i: {i} for i in by_id}
    decisions = []
    for edge in edges:
        left, right = edge["left_component_id"], edge["right_component_id"]
        a, b = owner[left], owner[right]
        if a == b:
            continue
        ids = sorted(groups[a] | groups[b])
        decision = (coherent_group([by_id[i] for i in ids], v1, coherence)
                    if mode == "GROUP_COHERENCE" else
                    corridor_metrics(by_id[left], by_id[right], support, corridor))
        decisions.append({"edge": [left, right], "proposed_group": ids, **decision})
        if decision["accepted"]:
            root, other = min(a, b), max(a, b)
            groups[root] = set(ids)
            del groups[other]
            for i in ids: owner[i] = root
    h, w = mask.shape
    proposals = []
    for ids in groups.values():
        members = [by_id[i] for i in sorted(ids)]
        box = [min(c.bbox[0] for c in members), min(c.bbox[1] for c in members),
               max(c.bbox[2] for c in members), max(c.bbox[3] for c in members)]
        padded = [max(0, box[0]-v1.pad), max(0, box[1]-v1.pad),
                  min(w, box[2]+v1.pad), min(h, box[3]+v1.pad)]
        proposals.append({"raw_component_ids": sorted(ids), "bbox": box, "padded_bbox": padded,
                          "foreground_area": sum(c.foreground_area for c in members)})
    proposals.sort(key=lambda g: (-g["foreground_area"], g["bbox"][1], g["bbox"][0], g["raw_component_ids"]))
    return {"boxes": [g["padded_bbox"] for g in proposals[:v1.nmax]],
            "before_cap_boxes": [g["padded_bbox"] for g in proposals], "raw_count": len(raw),
            "merges": sum(len(g["raw_component_ids"])-1 for g in proposals),
            "multi": sum(len(g["raw_component_ids"]) > 1 for g in proposals),
            "groups": proposals, "decisions": decisions}
