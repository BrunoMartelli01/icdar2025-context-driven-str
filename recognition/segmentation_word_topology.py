"""GT-bridged native-mask topology and OCR diagnostics for first-ten traces.

This is an APPROXIMATE diagnostic only.  It reuses the frozen projected masks and
recovers their exact native 128x128 binary masks by inverse sampling the recorded
nearest-neighbour projection, then compares target/prediction topology and (when
fragment evidence exists) runs PARSeq on GT-box and diagnostic component-union
crops.  Ground truth never changes inference or Algorithm 1 outputs.
"""

from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio

from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam, localize_mask
from pipeline.frozen_segmentation import FROZEN_LOGITS_SHAPE
from recognition.icdar13_annotations import ICDAR13RecognitionSample
from recognition.icdar13_trace_diagnostics import diagnose_trace_index
from segmentation.data import (
    icdar13_task22_rgb_to_binary_mask,
    load_segmentation_manifest,
)
from segmentation.evaluate import resize_binary_target_to_native_logits


EXPECTED_SEGMENTATION_CHECKPOINT_SHA256 = (
    "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"
)
BASELINE_MIN_AREA = 150
BASELINE_PADDING = 6
BASELINE_CONNECTIVITY = 8
AREA_PURITY_THRESHOLD = 0.8


class SegmentationWordTopologyError(ValueError):
    """Raised when frozen inputs or diagnostic geometry violate the contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exclusive_gt_box(
    value: tuple[int, int, int, int], image_size_wh: tuple[int, int]
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = value
    width, height = image_size_wh
    if not (0 <= x1 <= x2 < width and 0 <= y1 <= y2 < height):
        raise SegmentationWordTopologyError("Ground-truth box is outside source image")
    return (x1, y1, x2 + 1, y2 + 1)


def _native_word_box(
    box_exclusive: tuple[int, int, int, int], image_size_wh: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Map an original-image half-open box to a clamped native 128 box."""

    x1, y1, x2, y2 = box_exclusive
    width, height = image_size_wh
    native_h, native_w = FROZEN_LOGITS_SHAPE
    nx1 = max(0, min(native_w - 1, math.floor(x1 * native_w / width)))
    ny1 = max(0, min(native_h - 1, math.floor(y1 * native_h / height)))
    nx2 = max(nx1 + 1, min(native_w, math.ceil(x2 * native_w / width)))
    ny2 = max(ny1 + 1, min(native_h, math.ceil(y2 * native_h / height)))
    return (nx1, ny1, nx2, ny2)


def _native_box_to_original(
    box: tuple[int, int, int, int], image_size_wh: tuple[int, int]
) -> tuple[int, int, int, int]:
    nx1, ny1, nx2, ny2 = box
    width, height = image_size_wh
    native_h, native_w = FROZEN_LOGITS_SHAPE
    return (
        max(0, min(width - 1, math.floor(nx1 * width / native_w))),
        max(0, min(height - 1, math.floor(ny1 * height / native_h))),
        max(1, min(width, math.ceil(nx2 * width / native_w))),
        max(1, min(height, math.ceil(ny2 * height / native_h))),
    )


def _box_metrics(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> dict[str, float]:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    left_area = (lx2 - lx1) * (ly2 - ly1)
    right_area = (rx2 - rx1) * (ry2 - ry1)
    union = left_area + right_area - intersection
    return {
        "intersection_area": intersection,
        "bbox_iou": intersection / union if union else 0.0,
        "left_coverage": intersection / left_area if left_area else 0.0,
        "right_coverage": intersection / right_area if right_area else 0.0,
    }


def _text_metrics(prediction: str | None, ground_truth: str) -> dict[str, Any]:
    prediction = "" if prediction is None else prediction
    both_nonempty = bool(prediction) and bool(ground_truth)
    return {
        "case_sensitive_exact": both_nonempty and prediction == ground_truth,
        "casefold_exact": both_nonempty and prediction.casefold() == ground_truth.casefold(),
        "rapidfuzz_ratio_casefold": ratio(
            prediction.casefold(), ground_truth.casefold()
        ) / 100.0,
    }


def recover_native_mask_from_projected(
    projected: np.ndarray, *, native_shape: tuple[int, int] = FROZEN_LOGITS_SHAPE
) -> np.ndarray:
    """Recover a native mask from a recorded PyTorch nearest-neighbour projection.

    For each native pixel, the last output pixel in its nearest-neighbour interval is
    sampled.  This handles non-integral source/native scale factors and is verified by
    the frozen adapter's native-mask SHA256 metadata.
    """

    if projected.ndim != 2 or projected.dtype != np.uint8:
        raise SegmentationWordTopologyError("Projected mask must be uint8 HxW")
    native_h, native_w = native_shape
    height, width = projected.shape
    ys = np.ceil((np.arange(native_h) + 1) * height / native_h).astype(int) - 1
    xs = np.ceil((np.arange(native_w) + 1) * width / native_w).astype(int) - 1
    ys = np.clip(ys, 0, height - 1)
    xs = np.clip(xs, 0, width - 1)
    recovered = projected[np.ix_(ys, xs)]
    if not set(np.unique(recovered).tolist()).issubset({0, 1}):
        raise SegmentationWordTopologyError("Recovered native mask must be binary 0/1")
    return recovered.astype(np.uint8, copy=False)


def _components(mask: np.ndarray, connectivity: int = 8) -> dict[str, Any]:
    if mask.shape != FROZEN_LOGITS_SHAPE or mask.dtype != np.uint8:
        raise SegmentationWordTopologyError("Native component mask must be uint8 128x128")
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=connectivity
    )
    records: list[dict[str, Any]] = []
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        records.append(
            {
                "component_id": component_id,
                "area": int(stats[component_id, cv2.CC_STAT_AREA]),
                "bbox_xyxy_exclusive_native": [x, y, x + width, y + height],
                "centroid_native": [float(centroids[component_id][0]), float(centroids[component_id][1])],
            }
        )
    return {"labels": labels, "records": records}


def _word_topology(
    *,
    word_index: int,
    transcription: str,
    gt_box_original: tuple[int, int, int, int],
    native_box: tuple[int, int, int, int],
    target: np.ndarray,
    prediction: np.ndarray,
    target_components: dict[str, Any],
    prediction_components: dict[str, Any],
    image_size_wh: tuple[int, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    nx1, ny1, nx2, ny2 = native_box
    region = np.zeros_like(prediction, dtype=bool)
    region[ny1:ny2, nx1:nx2] = True
    target_pixels = int((target & region).sum())
    predicted_pixels = int((prediction & region).sum())
    overlap_pixels = int((target & prediction & region).sum())

    def labels_in_region(components: dict[str, Any]) -> tuple[list[int], list[int]]:
        labels = components["labels"]
        ids = sorted(int(value) for value in np.unique(labels[region]) if value)
        centers: list[int] = []
        for component_id in ids:
            row = components["records"][component_id - 1]
            cx, cy = row["centroid_native"]
            if nx1 <= cx < nx2 and ny1 <= cy < ny2:
                centers.append(component_id)
        return ids, centers

    target_ids, target_center_ids = labels_in_region(target_components)
    prediction_ids, prediction_center_ids = labels_in_region(prediction_components)

    def largest_ratio(components: dict[str, Any], ids: list[int], total: int) -> float | None:
        if total == 0:
            return None
        labels = components["labels"]
        largest = max(
            (int(((labels == component_id) & region).sum()) for component_id in ids),
            default=0,
        )
        return largest / total

    topology = {
        "gt_index": word_index,
        "gt_text": transcription,
        "gt_bbox_original_xyxy_exclusive": list(gt_box_original),
        "gt_bbox_native_xyxy_exclusive": list(native_box),
        "target_foreground_pixel_count_in_word_region": target_pixels,
        "prediction_foreground_pixel_count_in_word_region": predicted_pixels,
        "target_foreground_coverage_of_word_region": target_pixels / max(1, int(region.sum())),
        "prediction_foreground_coverage_of_word_region": predicted_pixels / max(1, int(region.sum())),
        "target_component_count_intersecting_word": len(target_ids),
        "prediction_component_count_intersecting_word": len(prediction_ids),
        "target_component_count_with_center_inside_word": len(target_center_ids),
        "prediction_component_count_with_center_inside_word": len(prediction_center_ids),
        "target_largest_component_to_word_foreground_ratio": largest_ratio(
            target_components, target_ids, target_pixels
        ),
        "prediction_largest_component_to_word_foreground_ratio": largest_ratio(
            prediction_components, prediction_ids, predicted_pixels
        ),
        "word_recall_from_prediction_pixels": (
            overlap_pixels / target_pixels if target_pixels else None
        ),
    }

    component_rows: list[dict[str, Any]] = []
    labels = prediction_components["labels"]
    for component in prediction_components["records"]:
        component_id = component["component_id"]
        component_mask = labels == component_id
        inside_box = int((component_mask & region).sum())
        if inside_box == 0:
            continue
        component_rows.append(
            {
                "component_id": component_id,
                "area_native": component["area"],
                "bbox_native_xyxy_exclusive": component["bbox_xyxy_exclusive_native"],
                "bbox_original_xyxy_exclusive": list(
                    _native_box_to_original(
                        tuple(component["bbox_xyxy_exclusive_native"]), image_size_wh
                    )
                ),
                "foreground_pixels_inside_this_gt_box": inside_box,
                "box_purity": inside_box / component["area"],
                "target_foreground_pixels_inside_this_gt_box": int(
                    (component_mask & target & region).sum()
                ),
                "target_foreground_purity": int(
                    (component_mask & target & region).sum()
                )
                / component["area"],
            }
        )
    topology["prediction_components_intersecting_this_word"] = component_rows
    return topology, component_rows


def _add_component_associations(
    component_rows: list[dict[str, Any]],
    *,
    prediction_components: dict[str, Any],
    target: np.ndarray,
    word_regions: Sequence[tuple[int, int, int, int]],
    image_size_wh: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    """Classify every raw predicted component against all GT word rectangles."""

    labels = prediction_components["labels"]
    by_component: dict[str, dict[str, Any]] = {}
    for component in prediction_components["records"]:
        component_id = component["component_id"]
        component_mask = labels == component_id
        associations: list[dict[str, Any]] = []
        for gt_index, (x1, y1, x2, y2) in enumerate(word_regions, start=1):
            region = np.zeros_like(target, dtype=bool)
            region[y1:y2, x1:x2] = True
            pixels_in_box = int((component_mask & region).sum())
            if pixels_in_box == 0:
                continue
            target_pixels = int((component_mask & target & region).sum())
            associations.append(
                {
                    "gt_index": gt_index,
                    "pixels_inside_gt_box": pixels_in_box,
                    "box_purity": pixels_in_box / component["area"],
                    "target_foreground_pixels_inside_gt_box": target_pixels,
                    "target_foreground_purity": target_pixels / component["area"],
                }
            )
        if not associations:
            classification = "NO_GT_ASSOCIATION"
        elif len(associations) >= 2:
            classification = "CROSSES_MULTIPLE_GT_BOXES"
        elif max(item["box_purity"] for item in associations) >= AREA_PURITY_THRESHOLD:
            classification = "ALMOST_ENTIRELY_INSIDE_SINGLE_GT_BOX"
        else:
            classification = "SINGLE_GT_PARTIAL_OR_BACKGROUND"
        by_component[str(component_id)] = {
            "component_id": component_id,
            "area_native": component["area"],
            "bbox_native_xyxy_exclusive": component["bbox_xyxy_exclusive_native"],
            "bbox_original_xyxy_exclusive": list(
                _native_box_to_original(
                    tuple(component["bbox_xyxy_exclusive_native"]),
                    image_size_wh,
                )
            ),
            "associations": associations,
            "classification": classification,
        }
    return by_component


def _best_box_pair(
    crop_box: tuple[int, int, int, int], gt_box: tuple[int, int, int, int]
) -> dict[str, float]:
    return _box_metrics(tuple(float(v) for v in crop_box), tuple(float(v) for v in gt_box))


def _ocr_record(
    *,
    kind: str,
    box: tuple[int, int, int, int] | None,
    gt_index: int,
    gt_text: str,
    prediction: dict[str, Any] | None,
    geometry: dict[str, float] | None,
) -> dict[str, Any]:
    text = None if prediction is None else prediction.get("raw_text")
    return {
        "kind": kind,
        "gt_index": gt_index,
        "gt_text": gt_text,
        "box_original_xyxy_exclusive": list(box) if box is not None else None,
        "geometry": geometry,
        "parseq": prediction,
        "text_metrics": _text_metrics(text, gt_text) if prediction is not None else None,
    }


def _load_projected_mask(path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("L"))
    if value.shape != expected_shape or not set(np.unique(value).tolist()).issubset({0, 255}):
        raise SegmentationWordTopologyError("Projected mask is not a verified binary artifact")
    return (value > 0).astype(np.uint8)


def _load_target_native(mask_path: Path) -> np.ndarray:
    with Image.open(mask_path) as opened:
        binary = icdar13_task22_rgb_to_binary_mask(opened)
    tensor = resize_binary_target_to_native_logits(binary, size=FROZEN_LOGITS_SHAPE, threshold=0.5)
    return tensor.numpy().astype(np.uint8)


def _load_rgb(path: Path, image_size_wh: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"))
    if value.shape[:2] != (image_size_wh[1], image_size_wh[0]) or value.dtype != np.uint8:
        raise SegmentationWordTopologyError("Source image geometry changed")
    return value


def diagnose_segmentation_word_topology(
    *,
    index_path: Path,
    samples: Iterable[ICDAR13RecognitionSample],
    segmentation_manifest_path: Path,
    data_root: Path,
    segmentation_checkpoint: Path,
    parseq_provenance: dict[str, Any],
    recognize_crop: Callable[[np.ndarray], Any],
) -> dict[str, Any]:
    """Run native topology, purity, and conditional GT/union OCR diagnostics."""

    samples = tuple(samples)
    source_check = diagnose_trace_index(index_path, samples)
    if source_check["population"]["failed_sample_count"] != 0:
        raise SegmentationWordTopologyError("All source traces must succeed")
    segmentation_checkpoint = segmentation_checkpoint.resolve()
    checkpoint_sha = _sha256_file(segmentation_checkpoint)
    if checkpoint_sha != EXPECTED_SEGMENTATION_CHECKPOINT_SHA256:
        raise SegmentationWordTopologyError("Segmentation checkpoint SHA256 mismatch")
    segmentation_manifest = load_segmentation_manifest(segmentation_manifest_path)
    seg_by_id = {sample.sample_id: sample for sample in segmentation_manifest.samples}
    sample_by_id = {sample.sample_id: sample for sample in samples}
    index = json.loads(index_path.resolve().read_text(encoding="utf-8"))
    prediction_cache: dict[tuple[str, tuple[int, int, int, int]], dict[str, Any]] = {}
    per_sample: list[dict[str, Any]] = []
    aggregate_words: list[dict[str, Any]] = []
    aggregate_component_rows: list[dict[str, Any]] = []
    ocr_fragment_evidence = False
    native_hash_verifications = 0
    baseline_positive = 0
    baseline_iou_half = 0

    requested_ids = {sample.sample_id for sample in samples}
    records = [record for record in index["samples"] if str(record["sample_id"]) in requested_ids]
    if len(records) != len(samples):
        raise SegmentationWordTopologyError("Trace index does not contain exactly the requested samples")
    for record in records:
        sample_id = str(record["sample_id"])
        sample = sample_by_id[sample_id]
        seg_sample = seg_by_id.get(sample_id)
        if seg_sample is None:
            raise SegmentationWordTopologyError("Segmentation manifest lacks sample {}".format(sample_id))
        result = json.loads(Path(record["result"]).read_text(encoding="utf-8"))
        inference = result["segmentation"]["inference"]
        if inference["checkpoint"]["sha256"] != checkpoint_sha:
            raise SegmentationWordTopologyError("Trace checkpoint drift for sample {}".format(sample_id))
        image_rgb = _load_rgb(sample.image_path, sample.image_size_wh)
        projected_path = Path(result["segmentation"]["mask_artifact"]["path"])
        projected = _load_projected_mask(projected_path, image_rgb.shape[:2])
        native = recover_native_mask_from_projected(projected)
        native_hash = _sha256_bytes(native.tobytes())
        if native_hash != inference["native_mask_sha256"]:
            raise SegmentationWordTopologyError(
                "Recovered native mask hash mismatch for sample {}".format(sample_id)
            )
        native_hash_verifications += 1
        target_path = (data_root.resolve() / Path(seg_sample.mask_path)).resolve()
        expected_target_sha = seg_sample.mask_sha256
        if expected_target_sha and _sha256_file(target_path) != expected_target_sha:
            raise SegmentationWordTopologyError("GT segmentation mask hash mismatch for sample {}".format(sample_id))
        target = _load_target_native(target_path)
        prediction = native
        target_components = _components(target, 8)
        prediction_components = _components(prediction, 8)
        gt_words: list[dict[str, Any]] = []
        word_regions: list[tuple[int, int, int, int]] = []
        component_rows_for_sample: list[dict[str, Any]] = []
        for word_index, word in enumerate(sample.words, start=1):
            gt_box = _exclusive_gt_box(word.bbox_xyxy, sample.image_size_wh)
            native_box = _native_word_box(gt_box, sample.image_size_wh)
            word_regions.append(native_box)
            topology, _ = _word_topology(
                word_index=word_index,
                transcription=word.transcription,
                gt_box_original=gt_box,
                native_box=native_box,
                target=target,
                prediction=prediction,
                target_components=target_components,
                prediction_components=prediction_components,
                image_size_wh=sample.image_size_wh,
            )
            gt_words.append(topology)
        component_associations = _add_component_associations(
            [], prediction_components=prediction_components, target=target,
            word_regions=word_regions, image_size_wh=sample.image_size_wh
        )
        for component in component_associations.values():
            component["sample_id"] = sample_id
            component_rows_for_sample.append(component)
        fragmentation_count = sum(
            word["prediction_component_count_intersecting_word"] >= 2 for word in gt_words
        )
        if fragmentation_count:
            ocr_fragment_evidence = True

        # Recompute the frozen Algorithm-1 baseline geometry and verify the prior gate.
        localization = localize_mask(
            projected,
            image_rgb,
            BlockParam(
                min_area=BASELINE_MIN_AREA,
                pad=BASELINE_PADDING,
                connectivity=BASELINE_CONNECTIVITY,
                tie_break=APPROXIMATE_TIE_BREAK,
                nmax=10,
            ),
        )
        for word in gt_words:
            gt_box = tuple(word["gt_bbox_original_xyxy_exclusive"])
            overlaps = [
                _best_box_pair(tuple(block.padded_bbox), gt_box)
                for block in localization.retained
            ]
            best = max(overlaps, key=lambda item: item["bbox_iou"], default=None)
            if best and best["intersection_area"] > 0:
                baseline_positive += 1
            if best and best["bbox_iou"] >= 0.5:
                baseline_iou_half += 1

        # Existing ALG1 PARSeq outputs are reused; only GT_BOX and component-union are new.
        source_blocks = result["segmentation"]["localized_blocks"]
        source_predictions = result["T3"]["all"]
        for word in gt_words:
            gt_index = int(word["gt_index"])
            gt_text = str(word["gt_text"])
            gt_box = tuple(int(v) for v in word["gt_bbox_original_xyxy_exclusive"])
            word_region = tuple(int(v) for v in word["gt_bbox_native_xyxy_exclusive"])
            associated = [
                component for component in component_rows_for_sample
                if any(item["gt_index"] == gt_index for item in component["associations"])
            ]
            union_box = None
            if associated:
                original_boxes = [
                    tuple(int(v) for v in component["bbox_original_xyxy_exclusive"])
                    for component in associated
                ]
                union_box = (
                    min(box[0] for box in original_boxes),
                    min(box[1] for box in original_boxes),
                    max(box[2] for box in original_boxes),
                    max(box[3] for box in original_boxes),
                )
            ocr_records: list[dict[str, Any]] = []
            for kind, box in (("GT_BOX", gt_box), ("PRED_COMPONENT_UNION", union_box)):
                if box is None:
                    ocr_records.append(_ocr_record(kind=kind, box=None, gt_index=gt_index, gt_text=gt_text, prediction=None, geometry=None))
                    continue
                key = (sample_id, box)
                if key not in prediction_cache:
                    x1, y1, x2, y2 = box
                    value = recognize_crop(image_rgb[y1:y2, x1:x2].copy())
                    if hasattr(value, "to_dict"):
                        value = value.to_dict()
                    if not isinstance(value, dict):
                        raise SegmentationWordTopologyError(
                            "PARSeq recognizer must return a dict-like result"
                        )
                    prediction_cache[key] = {
                        key_name: value.get(key_name)
                        for key_name in ("raw_text", "token_ids", "confidence")
                    }
                ocr_records.append(
                    _ocr_record(
                        kind=kind,
                        box=box,
                        gt_index=gt_index,
                        gt_text=gt_text,
                        prediction=prediction_cache[key],
                        geometry=_best_box_pair(box, gt_box),
                    )
                )
            alg1_pairs: list[dict[str, Any]] = []
            for block, parseq in zip(source_blocks, source_predictions):
                box = tuple(int(v) for v in block["padded_bbox"])
                metrics = _best_box_pair(box, gt_box)
                if metrics["intersection_area"] > 0:
                    alg1_pairs.append({
                        "rank": block["rank"],
                        "box_original_xyxy_exclusive": list(box),
                        "parseq": {key: parseq.get(key) for key in ("raw_text", "token_ids", "confidence")},
                        "geometry": metrics,
                        "text_metrics": _text_metrics(parseq.get("raw_text"), gt_text),
                    })
            if alg1_pairs:
                alg1_best = max(
                    alg1_pairs,
                    key=lambda item: (item["geometry"]["bbox_iou"], item["geometry"]["right_coverage"], -item["rank"]),
                )
            else:
                alg1_best = None
            word["ocr_diagnostic"] = {
                "GT_BOX": ocr_records[0],
                "PRED_COMPONENT_UNION": ocr_records[1],
                "ALG1": {
                    "positive_overlap_candidate_count": len(alg1_pairs),
                    "best_spatial_candidate": alg1_best,
                },
            }
            aggregate_words.append({"sample_id": sample_id, **word})
        aggregate_component_rows.extend(component_rows_for_sample)
        per_sample.append({
            "sample_id": sample_id,
            "image_sha256": sample.image_sha256,
            "segmentation_mask_sha256": expected_target_sha,
            "projected_prediction_mask_sha256": _sha256_bytes(projected.tobytes()),
            "native_prediction_mask_sha256": native_hash,
            "target_native_foreground_pixel_count": int(target.sum()),
            "prediction_native_foreground_pixel_count": int(prediction.sum()),
            "target_native_component_count": len(target_components["records"]),
            "prediction_native_component_count": len(prediction_components["records"]),
            "prediction_component_class_counts": {
                name: sum(row["classification"] == name for row in component_rows_for_sample)
                for name in (
                    "ALMOST_ENTIRELY_INSIDE_SINGLE_GT_BOX",
                    "SINGLE_GT_PARTIAL_OR_BACKGROUND",
                    "CROSSES_MULTIPLE_GT_BOXES",
                    "NO_GT_ASSOCIATION",
                )
            },
            "words": gt_words,
        })

    if baseline_positive != 41 or baseline_iou_half != 7:
        raise SegmentationWordTopologyError(
            "ALG1 baseline gate mismatch: expected 41 positive and 7 IoU>=0.5, got {} and {}".format(
                baseline_positive, baseline_iou_half
            )
        )
    total_components = len(aggregate_component_rows)
    classification_counts = {
        name: sum(row["classification"] == name for row in aggregate_component_rows)
        for name in (
            "ALMOST_ENTIRELY_INSIDE_SINGLE_GT_BOX",
            "SINGLE_GT_PARTIAL_OR_BACKGROUND",
            "CROSSES_MULTIPLE_GT_BOXES",
            "NO_GT_ASSOCIATION",
        )
    }
    return {
        "schema_version": 1,
        "reproduction_mode": "APPROXIMATE",
        "result_class": "segmentation word-topology and conditional OCR diagnostics; not accuracy",
        "accuracy": None,
        "claims": {"table_3": False, "table_5": False, "table_6": False, "table_7": False},
        "evaluation_status": "UNRESOLVED",
        "source": {
            "trace_index": str(index_path.resolve()),
            "trace_index_sha256": _sha256_file(index_path.resolve()),
            "segmentation_manifest": str(Path(segmentation_manifest_path).resolve()),
            "segmentation_manifest_sha256": _sha256_file(Path(segmentation_manifest_path).resolve()),
            "segmentation_checkpoint": str(segmentation_checkpoint),
            "segmentation_checkpoint_sha256": checkpoint_sha,
            "recorded_sample_ids": [sample.sample_id for sample in samples],
        },
        "protocol": {
            "native_shape": list(FROZEN_LOGITS_SHAPE),
            "prediction_recovery": "inverse sampling of recorded nearest projection; last output pixel per native interval",
            "target_conversion": "icdar13_task2.2_exact_white_background_v1",
            "dont_care": "foreground",
            "target_resize": "bilinear to 128x128 then >=0.5",
            "word_box_mapping": "ASSUMPTION: original inclusive maxima converted to exclusive +1; floor/ceil scale mapping",
            "component_connectivity": 8,
            "purity_threshold": AREA_PURITY_THRESHOLD,
            "paper_fixed_algorithm1": ["true component area", "area descending", "padding/clipping", "nmax=10"],
            "matching": "DIAGNOSTIC_ONLY: geometry-first IoU; text never selects a GT pair",
        },
        "qualification": {
            "sample_count": len(samples),
            "word_count": len(aggregate_words),
            "native_mask_hash_verifications": native_hash_verifications,
            "alg1_baseline_positive_gt_overlap": baseline_positive,
            "alg1_baseline_best_iou_ge_0_5": baseline_iou_half,
            "alg1_baseline_gate": "passed",
        },
        "phase_1_native_topology": {
            "aggregate": {
                "target_foreground_pixel_count": sum(item["target_native_foreground_pixel_count"] for item in per_sample),
                "prediction_foreground_pixel_count": sum(item["prediction_native_foreground_pixel_count"] for item in per_sample),
                "target_component_count": sum(item["target_native_component_count"] for item in per_sample),
                "prediction_component_count": sum(item["prediction_native_component_count"] for item in per_sample),
                "words_with_prediction_component_count_ge_2": sum(
                    item["prediction_component_count_intersecting_word"] >= 2 for item in aggregate_words
                ),
            },
            "per_sample": [
                {
                    "sample_id": item["sample_id"],
                    "target_foreground_pixel_count": item["target_native_foreground_pixel_count"],
                    "prediction_foreground_pixel_count": item["prediction_native_foreground_pixel_count"],
                    "target_component_count": item["target_native_component_count"],
                    "prediction_component_count": item["prediction_native_component_count"],
                }
                for item in per_sample
            ],
        },
        "phase_2_component_correctness": {
            "raw_prediction_component_count": total_components,
            "classification_counts": classification_counts,
            "word_recall_from_components_range": {
                "minimum": min((item["word_recall_from_prediction_pixels"] for item in aggregate_words if item["word_recall_from_prediction_pixels"] is not None), default=None),
                "median": float(np.median([item["word_recall_from_prediction_pixels"] for item in aggregate_words if item["word_recall_from_prediction_pixels"] is not None])) if any(item["word_recall_from_prediction_pixels"] is not None for item in aggregate_words) else None,
                "maximum": max((item["word_recall_from_prediction_pixels"] for item in aggregate_words if item["word_recall_from_prediction_pixels"] is not None), default=None),
            },
            "component_rows": aggregate_component_rows,
        },
        "phase_3_ocr": {
            "executed": ocr_fragment_evidence,
            "conditional_rule": "run GT_BOX and PRED_COMPONENT_UNION only after raw prediction shows >=2 components intersecting at least one GT word",
            "new_parseq_crop_count": len(prediction_cache),
            "reused_alg1_outputs": True,
            "per_word": aggregate_words,
        },
        "per_sample": per_sample,
        "scientific_boundary": {
            "classification": "DIAGNOSTIC_ONLY",
            "hardware_deviation": parseq_provenance.get("device") == "cpu",
            "blocking_ambiguity_ids": ["A-008", "DS-003", "E-001"],
            "prohibition": "No metric here is recognition accuracy, segmentation Table 3, or a Table 5 result.",
        },
    }


def write_per_word_csv(report: dict[str, Any], path: Path) -> None:
    rows = []
    for word in report["phase_3_ocr"]["per_word"]:
        def ocr(kind: str) -> dict[str, Any]:
            return word["ocr_diagnostic"][kind]
        rows.append({
            "sample_id": word["sample_id"],
            "gt_index": word["gt_index"],
            "gt_text": word["gt_text"],
            "target_fg_pixels": word["target_foreground_pixel_count_in_word_region"],
            "prediction_fg_pixels": word["prediction_foreground_pixel_count_in_word_region"],
            "target_component_count": word["target_component_count_intersecting_word"],
            "prediction_component_count": word["prediction_component_count_intersecting_word"],
            "prediction_center_component_count": word["prediction_component_count_with_center_inside_word"],
            "prediction_largest_ratio": word["prediction_largest_component_to_word_foreground_ratio"],
            "word_recall_from_components": word["word_recall_from_prediction_pixels"],
            "gt_box_text": None if ocr("GT_BOX")["parseq"] is None else ocr("GT_BOX")["parseq"]["raw_text"],
            "gt_box_casefold_exact": None if ocr("GT_BOX")["text_metrics"] is None else ocr("GT_BOX")["text_metrics"]["casefold_exact"],
            "gt_box_rapidfuzz_casefold": None if ocr("GT_BOX")["text_metrics"] is None else ocr("GT_BOX")["text_metrics"]["rapidfuzz_ratio_casefold"],
            "union_text": None if ocr("PRED_COMPONENT_UNION")["parseq"] is None else ocr("PRED_COMPONENT_UNION")["parseq"]["raw_text"],
            "union_casefold_exact": None if ocr("PRED_COMPONENT_UNION")["text_metrics"] is None else ocr("PRED_COMPONENT_UNION")["text_metrics"]["casefold_exact"],
            "union_rapidfuzz_casefold": None if ocr("PRED_COMPONENT_UNION")["text_metrics"] is None else ocr("PRED_COMPONENT_UNION")["text_metrics"]["rapidfuzz_ratio_casefold"],
            "alg1_text": None if ocr("ALG1")["best_spatial_candidate"] is None else ocr("ALG1")["best_spatial_candidate"]["parseq"]["raw_text"],
            "alg1_casefold_exact": None if ocr("ALG1")["best_spatial_candidate"] is None else ocr("ALG1")["best_spatial_candidate"]["text_metrics"]["casefold_exact"],
            "alg1_rapidfuzz_casefold": None if ocr("ALG1")["best_spatial_candidate"] is None else ocr("ALG1")["best_spatial_candidate"]["text_metrics"]["rapidfuzz_ratio_casefold"],
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)


__all__ = [
    "EXPECTED_SEGMENTATION_CHECKPOINT_SHA256",
    "SegmentationWordTopologyError",
    "diagnose_segmentation_word_topology",
    "recover_native_mask_from_projected",
    "write_per_word_csv",
]
