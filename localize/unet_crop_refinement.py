"""Opt-in APPROXIMATE RGB crop -> same segmenter -> contours experiment.

All masks/boxes here use original RGB pixels. Foreground component area, not
rectangle area, controls filtering and ranking. No OCR or annotation is read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class ContourConfig:
    min_component_area: int
    padding_pixels: int
    max_blocks: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.min_component_area < 0 or self.padding_pixels < 0 or self.max_blocks < 1:
            raise ValueError("Require nonnegative area/padding and positive max_blocks")


@dataclass(frozen=True)
class CropRefinementConfig:
    initial: ContourConfig
    second_pass: ContourConfig
    context_margin_fraction: float
    context_margin_pixels: int
    max_final_blocks: int
    reproduction_mode: str

    def __post_init__(self) -> None:
        if self.reproduction_mode != "approximate":
            raise ValueError("U-Net crop resegmentation is APPROXIMATE only; forbidden in STRICT")
        if (isinstance(self.context_margin_fraction, bool)
            or not isinstance(self.context_margin_fraction, (int, float))
            or not math.isfinite(self.context_margin_fraction)
            or self.context_margin_fraction < 0):
            raise ValueError("context_margin_fraction must be finite and nonnegative")
        for name in ("context_margin_pixels", "max_final_blocks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.context_margin_pixels < 0 or self.max_final_blocks < 1:
            raise ValueError("Require nonnegative context pixels and positive final cap")


def binary_mask(mask: np.ndarray, expected_hw: tuple[int, int] | None = None) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2 or min(value.shape) < 1:
        raise ValueError("Expected nonempty HxW binary mask")
    if expected_hw is not None and value.shape != expected_hw:
        raise ValueError(f"Mask shape {value.shape} does not match RGB {expected_hw}")
    if not np.isfinite(value).all() or not np.isin(value, (0, 1, 255)).all():
        raise ValueError("Expected finite binary values 0/1/255, not probabilities/logits")
    return np.ascontiguousarray(value > 0, dtype=np.uint8)


def component_contours(mask: np.ndarray, config: ContourConfig) -> dict:
    """External contours with exact 8-connected foreground-pixel areas.

    External contour retrieval omits islands nested inside another contour.
    Area includes foreground pixels only; holes are excluded. Ties use xyxy.
    """
    mask = binary_mask(mask)
    height, width = mask.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates, rejected = [], []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        point_x, point_y = contour[0, 0]
        component_id = int(labels[point_y, point_x])
        if component_id == 0:
            raise RuntimeError("External contour is not associated with foreground")
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        row = {"component_id": component_id, "component_area": area,
               "bbox": [x, y, x + w, y + h],
               "padded_bbox": [max(0, x - config.padding_pixels),
                               max(0, y - config.padding_pixels),
                               min(width, x + w + config.padding_pixels),
                               min(height, y + h + config.padding_pixels)]}
        if area < config.min_component_area:
            rejected.append({**row, "reason": "below_min_component_area"})
        else:
            candidates.append(row)
    candidates.sort(key=lambda row: (-row["component_area"], *row["bbox"]))
    rejected.extend({**row, "reason": "max_blocks"} for row in candidates[config.max_blocks:])
    return {"raw_external_contours": len(contours), "parameters": asdict(config),
            "area_unit": "original_rgb_foreground_pixels",
            "retained": [dict(row, rank=i + 1) for i, row in
                         enumerate(candidates[:config.max_blocks])], "rejected": rejected}


def expand_box(box: list[int], image_hw: tuple[int, int], config: CropRefinementConfig) -> list[int]:
    x1, y1, x2, y2 = box
    height, width = image_hw
    dx = config.context_margin_pixels + math.ceil((x2 - x1) * config.context_margin_fraction)
    dy = config.context_margin_pixels + math.ceil((y2 - y1) * config.context_margin_fraction)
    return [max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)]


def mask_diagnostics(before: np.ndarray, after: np.ndarray) -> dict:
    before, after = binary_mask(before), binary_mask(after, before.shape)
    union = int(np.count_nonzero(before | after))
    intersection = int(np.count_nonzero(before & after))
    before_count = cv2.connectedComponents(before, connectivity=8)[0] - 1
    after_count = cv2.connectedComponents(after, connectivity=8)[0] - 1
    borders = {"top": bool(after[0].any()), "bottom": bool(after[-1].any()),
               "left": bool(after[:, 0].any()), "right": bool(after[:, -1].any())}
    return {"before_components": before_count, "after_components": after_count,
            "component_count_change": after_count - before_count,
            "foreground_iou": 1.0 if union == 0 else intersection / union,
            "before_foreground_pixels": int(before.sum()),
            "after_foreground_pixels": int(after.sum()),
            "added_foreground_pixels": int(np.count_nonzero(after & (1 - before))),
            "removed_foreground_pixels": int(np.count_nonzero(before & (1 - after))),
            "foreground_at_crop_border": borders,
            "empty_second_pass": not bool(after.any()),
            "interpretation": "scale disagreement / border contact only; not word correctness"}


def refine_contour_crops(
    image_rgb: np.ndarray, initial_mask: np.ndarray,
    segmenter: Callable[[np.ndarray], dict], config: CropRefinementConfig,
) -> dict:
    """Exactly one new forward per retained seed; reuse the same callable/model.

    The callable must return a binary ``mask`` at its RGB input resolution.
    Empty outputs stay empty; all inference/validation failures propagate.
    Returned arrays are explicit artifacts, kept separate from the JSON audit.
    """
    if (image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8
        or min(image_rgb.shape[:2]) < 1):
        raise ValueError("Expected nonempty uint8 HxWx3 RGB image")
    initial_mask = binary_mask(initial_mask, image_rgb.shape[:2])
    initial = component_contours(initial_mask, config.initial)
    _, initial_labels = cv2.connectedComponents(initial_mask, connectivity=8)
    refined_mask = np.zeros_like(initial_mask)
    records, artifacts, candidates = [], [], []
    for seed in initial["retained"]:
        crop_box = expand_box(seed["padded_bbox"], initial_mask.shape, config)
        x1, y1, x2, y2 = crop_box
        crop = np.ascontiguousarray(image_rgb[y1:y2, x1:x2]).copy()
        before = initial_mask[y1:y2, x1:x2].copy()
        seed_support = (initial_labels[y1:y2, x1:x2] == seed["component_id"]).astype(np.uint8)
        output = segmenter(crop)
        after = binary_mask(output["mask"], crop.shape[:2])
        local = component_contours(after, config.second_pass)
        refined_mask[y1:y2, x1:x2] |= after
        rows = []
        for child in local["retained"]:
            bx1, by1, bx2, by2 = child["padded_bbox"]
            cx1, cy1, cx2, cy2 = child["bbox"]
            row = {"seed_rank": seed["rank"], "local_rank": child["rank"],
                   "component_area": child["component_area"],
                   "bbox": [cx1 + x1, cy1 + y1, cx2 + x1, cy2 + y1],
                   "padded_bbox": [bx1 + x1, by1 + y1, bx2 + x1, by2 + y1]}
            rows.append(row)
            candidates.append(row)
        diagnostics = mask_diagnostics(before, after)
        _, after_labels = cv2.connectedComponents(after, connectivity=8)
        associated = np.unique(after_labels[seed_support > 0])
        diagnostics.update(
            seed_associated_after_components=int(np.count_nonzero(associated)),
            seed_support_pixels=int(seed_support.sum()),
            seed_support_retained_fraction=float(np.count_nonzero(after & seed_support) / seed_support.sum()),
            seed_association_rule="8-connected predicted components sharing at least one original seed foreground pixel",
        )
        records.append({"seed": seed, "crop_bbox": crop_box,
                        "diagnostics": diagnostics,
                        "localization": local, "proposals_original": rows,
                        "segmentation_metadata": output.get("metadata", {})})
        artifacts.append({"seed_rank": seed["rank"], "crop_rgb": crop,
                          "before_mask": before, "seed_support_mask": seed_support, "after_mask": after,
                          **{key: output[key] for key in ("probability_native", "mask_native", "prob")
                             if key in output}})
    # Exact duplicate boxes only. Do not suppress overlapping, distinct words
    # using a guessed IoU threshold or relocalize the union of overlapping ROIs.
    candidates.sort(key=lambda row: (-row["component_area"], *row["padded_bbox"],
                                     row["seed_rank"], row["local_rank"]))
    unique, seen, rejected = [], {}, []
    for row in candidates:
        key = tuple(row["padded_bbox"])
        if key in seen:
            rejected.append({**row, "reason": "exact_duplicate_bbox",
                             "duplicate_of": seen[key]})
        else:
            seen[key] = {"seed_rank": row["seed_rank"], "local_rank": row["local_rank"]}
            unique.append(row)
    rejected.extend({**row, "reason": "max_final_blocks"}
                    for row in unique[config.max_final_blocks:])
    final = [dict(row, rank=i + 1) for i, row in enumerate(unique[:config.max_final_blocks])]
    report = {"schema_version": 1, "variant": "unet-crop-resegmentation-v1",
              "reproduction_mode": config.reproduction_mode,
              "claim_label": "approximate reproduction", "classification": "ASSUMPTION",
              "coordinate_convention": "original_rgb_xyxy_exclusive",
              "resolved_config": asdict(config), "initial_localization": initial,
              "inference_calls_second_pass": len(records), "crops": records,
              "refined_proposals": final, "rejected_proposals": rejected,
              "refined_mask_semantics": "diagnostic union of raw ROI masks; proposals are per ROI",
              "empty_output_policy": "record empty; no baseline substitution",
              "outside_refined_rois": "zero / unobserved; not a full-image replacement mask",
              "deduplication": "exact padded bbox only; other overlaps preserved",
              "word_correctness": "UNRESOLVED; no GT or recognizer used"}
    return {"report": report, "initial_mask": initial_mask,
            "refined_mask": refined_mask, "crop_artifacts": artifacts}
