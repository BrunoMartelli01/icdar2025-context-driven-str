"""Approximate, GT-blind assignment of frozen OCR crops to independent text slots."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def overlap_iou(a, b) -> float:
    left = max(float(a[0]), float(b[0]))
    top = max(float(a[1]), float(b[1]))
    right = min(float(a[2]), float(b[2]))
    bottom = min(float(a[3]), float(b[3]))
    common = max(0., right - left) * max(0., bottom - top)
    area_a = (float(a[2]) - float(a[0])) * (float(a[3]) - float(a[1]))
    area_b = (float(b[2]) - float(b[0])) * (float(b[3]) - float(b[1]))
    if min(area_a, area_b) <= 0:
        raise ValueError("slot and crop boxes must have positive area")
    return common / (area_a + area_b - common)


class EmptyDetectorBoundaryError(ValueError):
    """A finite boundary has an empty envelope after clipping to the image."""

    def __init__(self, unclipped_bbox: list[float], clipped_bbox: list[int]):
        super().__init__("detector polygon has no in-image area")
        self.unclipped_bbox = unclipped_bbox
        self.clipped_bbox = clipped_bbox


def _boundary_array(points, image_wh: tuple[int, int]) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if (array.ndim != 2 or array.shape[1] != 2 or len(array) < 3
        or not np.isfinite(array).all() or len(image_wh) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in image_wh)):
        raise ValueError("invalid detector boundary polygon")
    return array


def detector_boundary_in_original(points, image_wh: tuple[int, int],
                                  exif_orientation: int = 1) -> list[list[float]]:
    """Invert OpenCV's EXIF orientation in polygon pixel coordinates.

    PIL crop/annotation coordinates refer to the stored, untransposed raster.
    Orientations 0 (unspecified) and 1 both leave OpenCV's raster unchanged.
    """
    array = _boundary_array(points, image_wh)
    if (not isinstance(exif_orientation, int) or isinstance(exif_orientation, bool)
        or exif_orientation not in range(9)):
        raise ValueError("unsupported detector EXIF orientation")
    width, height = image_wh
    x, y = array[:, 0], array[:, 1]
    inverse = {
        0: (x, y), 1: (x, y), 2: (width - 1 - x, y),
        3: (width - 1 - x, height - 1 - y), 4: (x, height - 1 - y),
        5: (y, x), 6: (y, height - 1 - x),
        7: (width - 1 - y, height - 1 - x), 8: (width - 1 - y, x),
    }
    return np.column_stack(inverse[exif_orientation]).tolist()


def boundary_bbox(points, image_wh: tuple[int, int]) -> list[int]:
    """Envelope a detector polygon in original-image, exclusive xyxy coordinates."""
    array = _boundary_array(points, image_wh)
    width, height = image_wh
    box = [max(0, min(width, math.floor(array[:, 0].min()))),
           max(0, min(height, math.floor(array[:, 1].min()))),
           max(0, min(width, math.ceil(array[:, 0].max()))),
           max(0, min(height, math.ceil(array[:, 1].max())))]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise EmptyDetectorBoundaryError(
            [float(array[:, 0].min()), float(array[:, 1].min()),
             float(array[:, 0].max()), float(array[:, 1].max())], box)
    return box


def assign_to_slots(predictions: list[dict], slots: list[dict], *, min_iou: float
                    ) -> tuple[list[dict], list[dict], list[dict]]:
    """Keep all compatible crop memberships; emit at most one crop per slot.

    A crop can be eligible for multiple slots, including multiword parents.
    The final assignment uses each original crop once. A geometric mean of
    crop confidence and spatial IoU is an explicit, GT-blind assumption.
    """
    if not math.isfinite(min_iou) or not 0 < min_iou <= 1:
        raise ValueError("min_iou must be in (0,1]")
    ids = [p["candidate_id"] for p in predictions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate original crop ID")
    if len({s["source_index"] for s in slots}) != len(slots):
        raise ValueError("duplicate detector slot ID")
    weights = np.full((len(slots), len(predictions) + len(slots)), -np.inf)
    members = []
    for i, slot in enumerate(slots):
        weights[i, len(predictions) + i] = 0.
        eligible = []
        for j, prediction in enumerate(predictions):
            score = prediction["score"]
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid crop selection score")
            iou = overlap_iou(slot["bbox"], prediction["bbox"])
            if iou >= min_iou:
                weights[i, j] = math.sqrt(score * iou)
                eligible.append({"candidate_id": prediction["candidate_id"],
                                 "group_id": prediction["group_id"], "box_iou": iou})
        members.append({"slot_source_index": slot["source_index"],
                        "eligible_candidates": eligible})
    if not slots:
        return [], [], members
    slot_indices, columns = linear_sum_assignment(weights, maximize=True)
    selected, assignments = [], []
    for i, column in zip(slot_indices, columns):
        if column >= len(predictions):
            continue
        prediction = predictions[column]
        selected.append(prediction)
        assignments.append({"slot_source_index": slots[i]["source_index"],
                            "candidate_id": prediction["candidate_id"],
                            "group_id": prediction["group_id"],
                            "box_iou": overlap_iou(slots[i]["bbox"], prediction["bbox"]),
                            "assignment_score": float(weights[i, column])})
    return selected, assignments, members
