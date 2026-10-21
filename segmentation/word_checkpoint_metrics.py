"""APPROXIMATE native-mask and operational-box metrics for checkpoint selection.

Native mask IoU follows the qualified v1/v2 TRAIN diagnostic. Original owner
IDs remain in denominators when resizing removes a word. No OCR or TEST access.
"""
from __future__ import annotations

import cv2
import numpy as np

from localize.native_repository_contours import remap_native_box
from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from recognition.scene_text_v1_evaluation import box_iou, maximum_matching


def counts(tp, predicted, gt):
    return {"tp": int(tp), "predicted": int(predicted), "gt": int(gt),
        "precision": tp / predicted if predicted else 0., "recall": tp / gt if gt else 0.,
        "f1": 2 * tp / (predicted + gt) if predicted + gt else 0.}


def native_metrics(prediction, target, valid, owners, expected_ids, *, connectivity, iou_threshold):
    if not (prediction.shape == target.shape == valid.shape == owners.shape):
        raise ValueError("Paired native geometry mismatch")
    if any(v.dtype != np.bool_ for v in (prediction, target, valid)) or connectivity not in (4, 8):
        raise ValueError("Boolean masks and explicit valid connectivity required")
    if not 0 < iou_threshold <= 1:
        raise ValueError("Invalid native mask IoU threshold")
    ids = sorted(set(int(v) for v in expected_ids))
    if not set(np.unique(owners[owners > 0])).issubset(ids):
        raise ValueError("Unrecorded native word owner")
    pred, truth = prediction & valid, target & valid
    _, labels = cv2.connectedComponents(pred.astype(np.uint8), connectivity=connectivity)
    component_count = int(labels.max())
    gt_labels = np.zeros_like(owners, dtype=np.int32)
    for index, owner in enumerate(ids, 1):
        gt_labels[owners == owner] = index
    bins = len(ids) + 1
    joint = np.bincount((labels.astype(np.int64) * bins + gt_labels).ravel(),
        minlength=(component_count + 1) * bins).reshape(component_count + 1, bins)
    overlap = joint[1:, 1:]
    sizes_p, sizes_g = joint[1:, :].sum(1)[:, None], joint[:, 1:].sum(0)[None, :]
    union = sizes_p + sizes_g - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)
    edges = [tuple(map(int, pair)) for pair in np.argwhere(iou >= iou_threshold)]
    tp = len(maximum_matching(component_count, len(ids), edges))
    touching = overlap > 0
    fused = touching.sum(1) >= 2
    intersection = int((pred & truth).sum())
    pixel_union = int((pred | truth).sum())
    return {"native_instance": counts(tp, component_count, len(ids)),
        "pixel_iou": intersection / pixel_union if pixel_union else 1.,
        "missing_gt_at_native": int((sizes_g == 0).sum()),
        "fused_components_touching_multiple_words": int(fused.sum()),
        "words_touched_by_fused_components": int(touching[fused].any(0).sum()) if fused.any() else 0,
        "words_split_across_components": int((touching.sum(0) >= 2).sum())}


def bbox_metrics(prediction, image_hw, ground_truth, *, min_area, padding, nmax, iou_threshold):
    audit = audit_repository_contours(prediction.astype(np.uint8), RepositoryContourParams(min_area, padding, nmax))
    boxes = [remap_native_box(p["bbox"], image_hw) for p in audit["retained"]]
    edges = [(pi, gi) for pi, box in enumerate(boxes) for gi, word in enumerate(ground_truth)
             if box_iou(box, word["bbox"]) >= iou_threshold]
    tp = len(maximum_matching(len(boxes), len(ground_truth), edges))
    return {"bbox_detection": counts(tp, len(boxes), len(ground_truth)),
        "raw_contours": audit["raw_external_contours"], "candidates_before_cap": audit["candidates_before_cap"]}


def selection_records(records):
    """Keep winners for three signals and last; exact ties retain earliest epoch."""
    if not records or [r["epoch"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("Checkpoint selection needs contiguous complete epoch records")
    best = {}
    for name, field in (("best_native_word_f1", "native_word_f1"),
                        ("best_bbox_word_f1", "bbox_word_f1"),
                        ("best_foreground_iou", "foreground_iou")):
        values = [r["validation"][field] for r in records]
        if not np.isfinite(values).all() or not all(0 <= v <= 1 for v in values):
            raise ValueError("Invalid validation checkpoint-selection signal")
        best[name] = max(records, key=lambda r: (r["validation"][field], -r["epoch"]))
    best["last"] = records[-1]
    return best
