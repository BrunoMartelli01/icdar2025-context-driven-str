"""CPU-only, one-hop appearance rescue of immutable practical-remap-v1 groups.

ASSUMPTION / PRACTICAL_APPROXIMATION, not paper Algorithm 1. No GT, model,
training, OCR, or morphology-based mask grouping is used. Thickness is a
distance-transform proxy on predicted foreground, NOT a full SWT detector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Any

import cv2
import numpy as np

from .grouped_word_blocks import (
    GroupedWordBlock, GroupedWordBlockParams, GroupedWordBlocksResult,
    group_word_blocks,
)


MODE = "appearance-guided-local-remap-v1"


@dataclass(frozen=True)
class AppearanceRemapParams:
    enabled: bool
    max_normalized_gap: float
    max_gap_to_internal_ratio: float
    min_vertical_overlap: float
    max_center_y_ratio: float
    max_height_ratio: float
    max_color_delta_e76: float
    min_abs_contrast_l: float
    max_contrast_ratio: float
    max_stroke_ratio: float
    background_ring_height_ratio: float
    min_native_foreground_pixels: int
    min_background_pixels: int
    min_area_original: int

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        integer_names = (
            "min_native_foreground_pixels", "min_background_pixels", "min_area_original",
        )
        for name in integer_names:
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "min_area_original" else 1):
                raise ValueError(f"invalid {name}")
        for name, value in asdict(self).items():
            if name == "enabled" or name in integer_names:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_vertical_overlap > 1:
            raise ValueError("min_vertical_overlap must be <= 1")
        for name in ("max_height_ratio", "max_contrast_ratio", "max_stroke_ratio"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(frozen=True)
class AppearanceRemapResult:
    params: AppearanceRemapParams
    base: GroupedWordBlocksResult
    raw_descriptors: tuple[dict[str, Any], ...]
    pair_decisions: tuple[dict[str, Any], ...]
    merge_decisions: tuple[dict[str, Any], ...]
    groups_before_filter: tuple[dict[str, Any], ...]
    groups_before_cap: tuple[dict[str, Any], ...]
    retained: tuple[GroupedWordBlock, ...]
    rejected: tuple[dict[str, Any], ...]
    control_groups_before_cap: tuple[dict[str, Any], ...]
    control_retained: tuple[GroupedWordBlock, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": MODE, "reproduction_mode": "approximate",
            "claim_label": "approximate reproduction",
            "result_class": "PRACTICAL_APPROXIMATION",
            "status": "prototype_not_calibrated_or_promoted",
            "table_5_accuracy": None,
            "parameters": asdict(self.params),
            "v1_parameters": asdict(self.base.params),
            "coordinate_convention": "xyxy_exclusive_original_pixels",
            "native_to_original": "floor-index nearest sampling, no inference",
            "thickness_estimator": "2 * median distance-transform ridge radius; predicted-mask proxy",
            "raw_descriptors": self.raw_descriptors,
            "pair_decisions": self.pair_decisions,
            "merge_decisions": self.merge_decisions,
            "groups_before_filter": self.groups_before_filter,
            "groups_before_cap": self.groups_before_cap,
            "retained": [
                {"rank": block.rank, "raw_component_ids": block.raw_component_ids,
                 "bbox": block.bbox, "padded_bbox": block.padded_bbox,
                 "foreground_area": block.foreground_area, "crop_shape": block.crop.shape}
                for block in self.retained
            ],
            "rejected": self.rejected,
        }


def project_native_labels(labels: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Exact integer nearest mapping; never stretch geometric distance ratios."""
    if labels.shape != (128, 128) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("native labels must be an integer 128x128 array")
    height, width = image_shape
    if min(height, width) < 1:
        raise ValueError("source image dimensions must be positive")
    ys = np.arange(height, dtype=np.int64) * 128 // height
    xs = np.arange(width, dtype=np.int64) * 128 // width
    return labels[np.ix_(ys, xs)]


def _vertical(a: tuple[int, ...], b: tuple[int, ...]) -> dict[str, float]:
    ah, bh = a[3] - a[1], b[3] - b[1]
    reference = (ah + bh) / 2
    return {
        "vertical_overlap": max(0, min(a[3], b[3]) - max(a[1], b[1])) / min(ah, bh),
        "center_y_ratio": abs((a[1] + a[3] - b[1] - b[3]) / 2) / reference,
        "height_ratio": max(ah, bh) / min(ah, bh),
    }


def _same_line(metrics: dict[str, float], p: AppearanceRemapParams) -> bool:
    return (metrics["vertical_overlap"] >= p.min_vertical_overlap
            and metrics["center_y_ratio"] <= p.max_center_y_ratio
            and metrics["height_ratio"] <= p.max_height_ratio)


def _gap(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    return max(0, max(a[0], b[0]) - min(a[2], b[2])) / ((a[3] - a[1] + b[3] - b[1]) / 2)


def _internal_gap(ids: tuple[int, ...], boxes: dict[int, tuple[int, ...]]) -> float | None:
    """Positive gaps between disjoint X intervals; nested boxes cannot add gaps."""
    ordered = sorted((boxes[k] for k in ids), key=lambda b: (b[0], b[1], b[2], b[3]))
    rightmost = ordered[0]
    gaps = []
    for box in ordered[1:]:
        if box[0] > rightmost[2]:
            gaps.append(_gap(rightmost, box))
        if box[2] > rightmost[2]:
            rightmost = box
    return float(np.median(gaps)) if gaps else None


def _describe(raw: Any, labels: np.ndarray, native_ids_full: np.ndarray,
              native_areas: np.ndarray, rgb: np.ndarray, p: AppearanceRemapParams) -> dict[str, Any]:
    x1, y1, x2, y2 = raw.bbox
    foreground = labels[y1:y2, x1:x2] == raw.component_id
    native_ids = np.unique(native_ids_full[y1:y2, x1:x2][foreground])
    if len(native_ids) != 1 or native_ids[0] == 0 or int(foreground.sum()) != raw.foreground_area:
        raise RuntimeError("native/original component identity or true area changed")
    native_id = int(native_ids[0])
    ring = max(1, math.ceil((y2 - y1) * p.background_ring_height_ratio))
    height, width = rgb.shape[:2]
    rx1, ry1, rx2, ry2 = max(0, x1-ring), max(0, y1-ring), min(width, x2+ring), min(height, y2+ring)
    patch_labels = labels[ry1:ry2, rx1:rx2]
    # Float RGB conversion gives L in [0,100] and signed a/b, not uint8 Lab units.
    lab = cv2.cvtColor(rgb[ry1:ry2, rx1:rx2].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    fg_lab = np.median(lab[patch_labels == raw.component_id], axis=0)
    background = patch_labels == 0  # excludes every predicted foreground component
    bg_count = int(background.sum())
    bg_l = float(np.median(lab[..., 0][background])) if bg_count else None
    contrast = float(fg_lab[0] - bg_l) if bg_l is not None else None
    distance = cv2.distanceTransform(np.pad(foreground.astype(np.uint8), 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # A local-maximum operator measures ridges; it does not modify the binary mask.
    ridge = (distance > 0) & (distance == cv2.dilate(distance, np.ones((3, 3), np.uint8)))
    stroke = float(2 * np.median(distance[ridge]))
    reasons = []
    if int(native_areas[native_id]) < p.min_native_foreground_pixels:
        reasons.append("insufficient_native_foreground")
    if bg_count < p.min_background_pixels:
        reasons.append("insufficient_background")
    if contrast is None or abs(contrast) < p.min_abs_contrast_l:
        reasons.append("insufficient_contrast")
    return {
        "raw_component_id": raw.component_id, "native_component_id": native_id,
        "native_foreground_area": int(native_areas[native_id]),
        "foreground_area_original": raw.foreground_area, "bbox_original": raw.bbox,
        "foreground_lab_median": tuple(float(v) for v in fg_lab),
        "background_l_median": bg_l, "background_pixels": bg_count,
        "signed_contrast_l": contrast, "stroke_width_proxy_original": stroke,
        "eligible_for_new_merge": not reasons, "ineligibility_reasons": reasons,
    }


def _appearance(ids: tuple[int, ...], descriptors: dict[int, dict[str, Any]],
                p: AppearanceRemapParams) -> tuple[list[str], dict[str, Any]]:
    rows = [descriptors[k] for k in ids]
    invalid = [k for k in ids if not descriptors[k]["eligible_for_new_merge"]]
    if invalid:
        return ["insufficient_appearance_evidence"], {"invalid_raw_component_ids": invalid}
    colors = [np.asarray(row["foreground_lab_median"]) for row in rows]
    color_delta = max((float(np.linalg.norm(a-b)) for a, b in combinations(colors, 2)), default=0.0)
    strokes = [row["stroke_width_proxy_original"] for row in rows]
    contrasts = [row["signed_contrast_l"] for row in rows]
    stroke_ratio = max(strokes) / min(strokes)
    contrast_ratio = max(map(abs, contrasts)) / min(map(abs, contrasts))
    polarity_consistent = all(v > 0 for v in contrasts) or all(v < 0 for v in contrasts)
    reasons = []
    if color_delta > p.max_color_delta_e76:
        reasons.append("color_incompatible")
    if stroke_ratio > p.max_stroke_ratio:
        reasons.append("stroke_incompatible")
    if not polarity_consistent:
        reasons.append("contrast_polarity_incompatible")
    if contrast_ratio > p.max_contrast_ratio:
        reasons.append("contrast_magnitude_incompatible")
    return reasons, {"max_color_delta_e76": color_delta, "stroke_ratio": stroke_ratio,
                     "contrast_ratio": contrast_ratio, "polarity_consistent": polarity_consistent}


def _finalize(groups: list[dict[str, Any]], image: np.ndarray,
              v1: GroupedWordBlockParams, min_area: int) -> tuple:
    ordered = sorted(groups, key=lambda g: (-g["foreground_area"], g["bbox"][1], g["bbox"][0], g["group_id"]))
    before_cap, retained, rejected = [], [], []
    height, width = image.shape[:2]
    for source in ordered:
        group = dict(source)
        x1, y1, x2, y2 = group["bbox"]
        group["padded_bbox"] = (max(0, x1-v1.pad), max(0, y1-v1.pad), min(width, x2+v1.pad), min(height, y2+v1.pad))
        if group["foreground_area"] < min_area:
            rejected.append({**group, "reason": "below_min_area"})
            continue
        before_cap.append(group)
        if len(retained) == v1.nmax:
            rejected.append({**group, "reason": "top10"})
            continue
        px1, py1, px2, py2 = group["padded_bbox"]
        retained.append(GroupedWordBlock(
            rank=len(retained)+1, raw_component_ids=group["raw_component_ids"],
            foreground_area=group["foreground_area"], bbox=group["bbox"],
            padded_bbox=group["padded_bbox"], crop=image[py1:py2, px1:px2].copy(),
        ))
    return tuple(before_cap), tuple(retained), tuple(rejected)


def appearance_guided_local_remap(
    native_mask: np.ndarray, image_rgb: np.ndarray, *,
    v1_params: GroupedWordBlockParams, params: AppearanceRemapParams,
) -> AppearanceRemapResult:
    """Return disjoint, at-most-three-base-group rescues plus frozen controls.

    Neighbours are chosen BEFORE appearance checks. A failed immediate neighbour
    cannot be skipped for a more distant compatible one. Merged groups are never
    fed back to the search; all hypotheses use immutable v1 base groups.
    """
    if native_mask.dtype != np.uint8 or native_mask.shape != (128, 128) or not np.isin(native_mask, [0, 1]).all():
        raise ValueError("native_mask must be uint8 binary 0/1 at 128x128")
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3 RGB uint8")
    if v1_params.connectivity != 8:
        raise ValueError("this qualified native-cache mode requires connectivity=8")
    _, native_labels, native_stats, _ = cv2.connectedComponentsWithStats(native_mask, connectivity=8)
    native_full = project_native_labels(native_labels, image_rgb.shape[:2])
    mask = (native_full > 0).astype(np.uint8)
    base = group_word_blocks(mask, image_rgb, v1_params)
    _, labels, _, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if len(base.raw_components) != len(native_stats)-1:
        raise RuntimeError("native raw components were lost or merged by coordinate projection")
    rows = tuple(_describe(raw, labels, native_full, native_stats[:, cv2.CC_STAT_AREA], image_rgb, params)
                 for raw in base.raw_components)
    if sorted(row["native_component_id"] for row in rows) != list(range(1, len(native_stats))):
        raise RuntimeError("native/original component mapping must remain one-to-one")
    descriptors = {row["raw_component_id"]: row for row in rows}
    boxes = {raw.component_id: raw.bbox for raw in base.raw_components}
    bases = {g["group_id"]: {**g, "base_group_ids": (g["group_id"],)} for g in base.groups_before_cap}
    internals = {gid: _internal_gap(g["raw_component_ids"], boxes) for gid, g in bases.items()}
    neighbours = {}
    for gid, group in bases.items():
        a = group["bbox"]
        for side in (-1, 1):
            pool = []
            for other_id, other in bases.items():
                b = other["bbox"]
                if gid == other_id or side * (b[0]+b[2]-a[0]-a[2]) <= 0:
                    continue
                vertical = _vertical(a, b)
                if _same_line(vertical, params):
                    # Original-pixel horizontal distance, then vertical distance.
                    gap_px = max(0, max(a[0], b[0])-min(a[2], b[2]))
                    pool.append((gap_px, abs(b[0]+b[2]-a[0]-a[2]), vertical["center_y_ratio"], other_id))
            neighbours[gid, side] = min(pool)[-1] if pool else None

    pair_rows, accepted_edges = [], set()
    for left_id in sorted(bases):
        right_id = neighbours[left_id, 1]
        if right_id is None:
            continue
        left, right = bases[left_id], bases[right_id]
        gap = _gap(left["bbox"], right["bbox"])
        ratios = {gid: (gap / internals[gid] if internals[gid] is not None else None) for gid in (left_id, right_id)}
        ids = tuple(sorted((*left["raw_component_ids"], *right["raw_component_ids"])))
        reasons, appearance = _appearance(ids, descriptors, params)
        if not params.enabled:
            reasons.append("new_merges_disabled")
        if neighbours[right_id, -1] != left_id:
            reasons.append("not_mutual_immediate_neighbours")
        if gap > params.max_normalized_gap:
            reasons.append("horizontal_gap")
        if any(value is not None and value > params.max_gap_to_internal_ratio for value in ratios.values()):
            reasons.append("gap_to_internal_spacing")
        pair_rows.append({"left_base_group_id": left_id, "right_base_group_id": right_id,
                          "normalized_gap_original": gap, **_vertical(left["bbox"], right["bbox"]),
                          "gap_to_internal_ratios": ratios, "appearance": appearance,
                          "accepted": not reasons, "reasons": reasons})
        if not reasons:
            accepted_edges.add((left_id, right_id))

    hypotheses = {tuple(sorted(edge)) for edge in accepted_edges}
    for gid in bases:
        left_id, right_id = neighbours[gid, -1], neighbours[gid, 1]
        if (left_id, gid) in accepted_edges and (gid, right_id) in accepted_edges:
            hypotheses.add(tuple(sorted((left_id, gid, right_id))))
    decisions, eligible = [], []
    for members in sorted(hypotheses):
        ids = tuple(sorted(raw_id for gid in members for raw_id in bases[gid]["raw_component_ids"]))
        reasons, appearance = _appearance(ids, descriptors, params)
        if any(not _same_line(_vertical(bases[a]["bbox"], bases[b]["bbox"]), params) for a, b in combinations(members, 2)):
            reasons.append("whole_group_vertical_incompatibility")
        row = {"base_group_ids": members, "raw_component_ids": ids, "appearance": appearance,
               "accepted": False, "reasons": reasons}
        decisions.append(row)
        if not reasons:
            max_gap = max(_gap(bases[a]["bbox"], bases[b]["bbox"]) for a,b in accepted_edges if a in members and b in members)
            eligible.append(((appearance["max_color_delta_e76"], appearance["stroke_ratio"], max_gap, -len(members), members), row))
    used, groups = set(), []
    for _, decision in sorted(eligible, key=lambda pair: pair[0]):
        members = decision["base_group_ids"]
        if used.intersection(members):
            decision["reasons"].append("base_group_already_consumed")
            continue
        used.update(members)
        decision["accepted"] = True
        member_boxes = [bases[gid]["bbox"] for gid in members]
        groups.append({
            "group_id": min(members), "base_group_ids": members,
            "raw_component_ids": decision["raw_component_ids"],
            "foreground_area": sum(bases[gid]["foreground_area"] for gid in members),
            "bbox": (min(b[0] for b in member_boxes), min(b[1] for b in member_boxes),
                     max(b[2] for b in member_boxes), max(b[3] for b in member_boxes)),
        })
    groups.extend(group for gid, group in bases.items() if gid not in used)
    flattened = [raw_id for group in groups for raw_id in group["raw_component_ids"]]
    if sorted(flattened) != sorted(boxes) or len(flattened) != len(set(flattened)):
        raise RuntimeError("rescue must partition, never duplicate/drop, original raw components")
    before_cap, retained, rejected = _finalize(groups, image_rgb, v1_params, params.min_area_original)
    control, control_retained, _ = _finalize(list(bases.values()), image_rgb, v1_params, params.min_area_original)
    return AppearanceRemapResult(params, base, rows, tuple(pair_rows), tuple(decisions),
                                 tuple(groups), before_cap, retained, rejected, control, control_retained)
