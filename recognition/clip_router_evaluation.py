"""Auditable word metrics for the APPROXIMATE whole-image CLIP/DeepSolo router."""
from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from localize.independent_word_slots import (
    EmptyDetectorBoundaryError, boundary_bbox, detector_boundary_in_original,
)
from recognition.clip_crop_admission import percentile
from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from segmentation.totaltext_image_frequency_augment import letterbox_geometry


def fixed_image_score(rows, reference):
    """Apply TRAIN-fitted mid-CDFs without fitting or inspecting any GT labels."""
    if not rows or any(not r["raw_text"].strip() or r["clip_word_advantage"] is None for r in rows):
        return None
    for r in rows:
        if not math.isfinite(r["confidence"]) or not 0 <= r["confidence"] <= 1:
            raise ValueError("Invalid native confidence")
        if not math.isfinite(r["clip_word_advantage"]):
            raise ValueError("Invalid CLIP word advantage")
    return min((percentile(reference["native"], r["confidence"])
                * percentile(reference["word_advantage"], r["clip_word_advantage"])) ** .5
               for r in rows)


def native_mask(mask, size):
    return TF.pil_to_tensor(TF.resize(letterbox_geometry(mask, fill=0), [size, size],
        interpolation=InterpolationMode.NEAREST))[0].numpy() > 0


def deepsolo_predictions(response, image_wh, orientation, validity, score_threshold):
    """All retained detections; inverse EXIF and evaluation-only polygon ignore.

    Empty clipped envelopes remain false-positive candidates (bbox=None). A
    tiny polygon with no native pixels remains a care candidate. These rules
    prevent dropping inconvenient detector outputs from the denominator.
    """
    if (validity.ndim != 2 or validity.shape[0] != validity.shape[1]
            or validity.dtype != np.bool_ or not 0 <= score_threshold <= 1):
        raise ValueError("Invalid DeepSolo evaluation geometry/threshold")
    seen, result = set(), []
    for detection in response["detections"]:
        index, score = detection["source_index"], detection["score"]
        if index in seen or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid or duplicate DeepSolo detection")
        seen.add(index)
        if score < score_threshold:
            continue
        points = detector_boundary_in_original(detection["geometry"]["boundary_points"],
                                                tuple(image_wh), orientation)
        row = {"instance_id": index, "text": detection["text"], "score": score,
               "boundary_original": points, "ignored": False, "invalid_envelope": None}
        try:
            row["bbox"] = boundary_bbox(points, tuple(image_wh))
        except EmptyDetectorBoundaryError as error:
            row.update(bbox=None, invalid_envelope={"unclipped_bbox": error.unclipped_bbox,
                "clipped_bbox": error.clipped_bbox, "treatment": "retain_as_unmatchable_FP"})
        if row["bbox"] is not None:
            # Rasterize the polygon, not its bounding-box area. All geometry is
            # expressed in the stored original RGB frame before letterboxing.
            support = np.zeros((image_wh[1], image_wh[0]), dtype=np.uint8)
            vertices = np.rint(np.clip(np.asarray(points), -2**29, 2**29)).astype(np.int32)
            cv2.fillPoly(support, [vertices], 255)
            footprint = native_mask(Image.fromarray(support), validity.shape[0])
            row["native_polygon_pixels"] = int(footprint.sum())
            row["ignored"] = bool(footprint.any() and not np.any(footprint & validity))
        result.append(row)
    return result


def evaluate_predictions(predictions, ground_truth, *, iou_threshold):
    """Micro word counts: exact Unicode casefold + one-to-one bbox IoU."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("Invalid evaluation IoU threshold")
    care = [p for p in predictions if not p["ignored"]]
    edges = [(i, j) for i, p in enumerate(care) for j, g in enumerate(ground_truth)
             if p["bbox"] is not None and p["text"] and p["text"].casefold() == g["text"].casefold()
             and box_iou(p["bbox"], g["bbox"]) >= iou_threshold]
    matched = maximum_matching(len(care), len(ground_truth), edges)
    tp = len(matched)
    return {"tp": tp, "fp": len(care) - tp, "fn": len(ground_truth) - tp, "tn": 0,
            "detected_words": len(care), "gt_words": len(ground_truth),
            "all_outputs": len(predictions), "ignored_outputs": len(predictions) - len(care),
            "invalid_envelopes": sum(p["bbox"] is None for p in predictions),
            "matched_gt_indices": sorted(matched)}


def aggregate_split(rows, expected_ids, expected_words):
    if (len(rows) != len(expected_ids) or len(set(expected_ids)) != len(expected_ids)
            or len({r["sample_id"] for r in rows}) != len(rows)
            or {r["sample_id"] for r in rows} != set(expected_ids)):
        raise ValueError("Missing, unexpected or duplicate split samples")
    if any(r["status"] != "SUCCESS" for r in rows):
        raise ValueError("Failed samples prevent complete split metrics")
    keys = ("tp", "fp", "fn", "tn", "detected_words", "gt_words", "all_outputs",
            "ignored_outputs", "invalid_envelopes")
    counts = {k: sum(r["metrics"][k] for r in rows) for k in keys}
    if counts["gt_words"] != expected_words or counts["tn"] != 0:
        raise ValueError("Frozen care-word denominator or TN changed")
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    accepted = sum(r["action"] == "accept_parseq" for r in rows)
    return {**counts, "images": len(rows), "failed_images": 0,
            "parseq_images": accepted, "deepsolo_images": len(rows) - accepted,
            "coverage_images": accepted / len(rows) if rows else None,
            "accuracy_on_detected_words": tp / (tp + fp) if tp + fp else None,
            "accuracy_tn0": tp / (tp + fp + fn) if tp + fp + fn else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "word_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None}
