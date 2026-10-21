"""Train-only calibration and frozen-test evaluation for practical-remap-v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from PIL import Image

from localize.grouped_word_blocks import (
    PRACTICAL_REMAP_V1,
    GroupedWordBlockParams,
    group_word_blocks,
)
from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam, localize_mask
from recognition.icdar13_annotations import ICDAR13RecognitionSample
from segmentation.data import SegmentationManifest, load_segmentation_manifest


TRAIN_GRID = {
    "horizontal_gap_ratio": (0.35, 0.5, 0.7, 1.0),
    "vertical_overlap_min": (0.3, 0.5, 0.7),
    "max_center_y_ratio": (0.35, 0.5),
    "max_height_ratio": (1.75, 2.5),
}
PADDING = 6
CONNECTIVITY = 8
NMAX = 10
SELECTION_F1_ATOL = 0.005
_TRAIN_LINE = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+).*"(.*)"\s*$')


@dataclass(frozen=True)
class TrainWordBox:
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class ProposalGeometry:
    gt_words: int
    gt_with_proposal_iou_ge_0_25: int
    gt_with_proposal_iou_ge_0_50: int
    gt_with_proposal_iou_ge_0_75: int
    retained_proposals: int
    true_positive_proposals_iou_ge_0_50: int
    proposal_precision_iou_ge_0_50: float
    recall_iou_ge_0_50: float
    f1_proposal_level_iou_ge_0_50: float
    average_proposals_per_image: float
    images_with_more_than_10_candidates_before_cap: int
    average_groups_before_cap_per_image: float
    total_raw_component_merges: int
    multi_component_group_count: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _box_iou(left: Sequence[int], right: Sequence[int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0, min(lx2, rx2) - max(lx1, rx1)) * max(0, min(ly2, ry2) - max(ly1, ry1))
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return intersection / union if union else 0.0


def _exclusive_test_box(value: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return value[0], value[1], value[2] + 1, value[3] + 1


def train_task22_word_boxes(mask_path: Path, annotation_path: Path) -> tuple[TrainWordBox, ...]:
    """Derive train word proposals from blank-line grouped Task-2 glyph records.

    The color-coded source BMP remains the authoritative geometry: each listed
    glyph's exact RGB pixels are unioned.  Character text is parsed only to
    recognize a normal glyph record; it is not passed to localizer calibration.
    """

    with Image.open(mask_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    groups: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    for raw in annotation_path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            if current:
                groups.append(current)
                current = []
            continue
        if raw.lstrip().startswith("#"):
            continue
        matched = _TRAIN_LINE.match(raw)
        if matched is None:
            raise ValueError("Malformed Task 2.2 glyph record: {}".format(annotation_path))
        current.append(tuple(int(matched.group(index)) for index in (1, 2, 3)))
    if current:
        groups.append(current)
    words = []
    for colors in groups:
        pixels = np.zeros(rgb.shape[:2], dtype=bool)
        for color in colors:
            pixels |= np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)
        ys, xs = np.where(pixels)
        if not len(xs):
            raise ValueError("Task 2.2 glyph colors absent from source target: {}".format(annotation_path))
        words.append(TrainWordBox((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)))
    if not words:
        raise ValueError("No non-Don\'t-Care word groups in {}".format(annotation_path))
    return tuple(words)


def declared_grid() -> tuple[GroupedWordBlockParams, ...]:
    return tuple(
        GroupedWordBlockParams(
            horizontal_gap_ratio=gap,
            vertical_overlap_min=overlap,
            max_center_y_ratio=center,
            max_height_ratio=height,
            pad=PADDING,
            connectivity=CONNECTIVITY,
            tie_break=APPROXIMATE_TIE_BREAK,
            nmax=NMAX,
        )
        for gap in TRAIN_GRID["horizontal_gap_ratio"]
        for overlap in TRAIN_GRID["vertical_overlap_min"]
        for center in TRAIN_GRID["max_center_y_ratio"]
        for height in TRAIN_GRID["max_height_ratio"]
    )


def _match_count_at_threshold(
    proposals: Sequence[Sequence[int]], gt_boxes: Sequence[Sequence[int]], threshold: float
) -> int:
    candidates = sorted(
        (
            (_box_iou(proposal, gt), proposal_index, gt_index)
            for proposal_index, proposal in enumerate(proposals)
            for gt_index, gt in enumerate(gt_boxes)
            if _box_iou(proposal, gt) >= threshold
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_proposals: set[int] = set()
    used_gt: set[int] = set()
    for _, proposal_index, gt_index in candidates:
        if proposal_index not in used_proposals and gt_index not in used_gt:
            used_proposals.add(proposal_index)
            used_gt.add(gt_index)
    return len(used_gt)


def proposal_geometry(
    per_image: Iterable[tuple[Sequence[Sequence[int]], Sequence[Sequence[int]], int, int, int]]
) -> ProposalGeometry:
    """Aggregate geometry without transcription/OCR-dependent matching."""

    gt_words = positive25 = positive50 = positive75 = proposals_total = true_positive = 0
    image_count = images_over_cap = groups_before_total = raw_merges = multi_groups = 0
    for proposals, gt_boxes, pre_cap_count, merges, multi_group_count in per_image:
        image_count += 1
        gt_words += len(gt_boxes)
        proposals_total += len(proposals)
        groups_before_total += pre_cap_count
        images_over_cap += int(pre_cap_count > NMAX)
        raw_merges += merges
        multi_groups += multi_group_count
        best_ious = [max((_box_iou(gt, proposal) for proposal in proposals), default=0.0) for gt in gt_boxes]
        positive25 += sum(value >= 0.25 for value in best_ious)
        positive50 += sum(value >= 0.5 for value in best_ious)
        positive75 += sum(value >= 0.75 for value in best_ious)
        true_positive += _match_count_at_threshold(proposals, gt_boxes, 0.5)
    precision = true_positive / proposals_total if proposals_total else 0.0
    recall = true_positive / gt_words if gt_words else 0.0
    return ProposalGeometry(
        gt_words=gt_words,
        gt_with_proposal_iou_ge_0_25=positive25,
        gt_with_proposal_iou_ge_0_50=positive50,
        gt_with_proposal_iou_ge_0_75=positive75,
        retained_proposals=proposals_total,
        true_positive_proposals_iou_ge_0_50=true_positive,
        proposal_precision_iou_ge_0_50=precision,
        recall_iou_ge_0_50=recall,
        f1_proposal_level_iou_ge_0_50=(2 * precision * recall / (precision + recall) if precision + recall else 0.0),
        average_proposals_per_image=proposals_total / image_count if image_count else 0.0,
        images_with_more_than_10_candidates_before_cap=images_over_cap,
        average_groups_before_cap_per_image=groups_before_total / image_count if image_count else 0.0,
        total_raw_component_merges=raw_merges,
        multi_component_group_count=multi_groups,
    )


def _practical_per_image(mask: np.ndarray, image: np.ndarray, params: GroupedWordBlockParams) -> tuple[Any, tuple[tuple[int, int, int, int], ...], int, int, int]:
    result = group_word_blocks(mask, image, params)
    boxes = tuple(block.padded_bbox for block in result.retained)
    groups = result.groups_before_cap
    multi = sum(len(group["raw_component_ids"]) > 1 for group in groups)
    merges = sum(len(group["raw_component_ids"]) - 1 for group in groups)
    return result, boxes, len(groups), merges, multi


def calibrate_on_train(
    train_records: Sequence[tuple[str, np.ndarray, np.ndarray, tuple[TrainWordBox, ...]]]
) -> dict[str, Any]:
    """Evaluate the predeclared geometry grid only on frozen train predictions."""

    variants = []
    for params in declared_grid():
        per_image = []
        for _, image, prediction, words in train_records:
            _, boxes, pre_cap, merges, multi = _practical_per_image(prediction, image, params)
            per_image.append((boxes, [word.bbox for word in words], pre_cap, merges, multi))
        geometry = proposal_geometry(per_image)
        variants.append({"parameters": asdict(params), "geometry": geometry.as_dict()})
    best_f1 = max(item["geometry"]["f1_proposal_level_iou_ge_0_50"] for item in variants)
    near_best = [
        item for item in variants
        if best_f1 - item["geometry"]["f1_proposal_level_iou_ge_0_50"] <= SELECTION_F1_ATOL
    ]
    # This order was fixed before seeing test: conservative geometric gates first,
    # then fewer raw-component merges, then fewer retained proposals.
    def selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
        parameters = item["parameters"]
        geometry = item["geometry"]
        return (
            parameters["horizontal_gap_ratio"],
            -parameters["vertical_overlap_min"],
            parameters["max_center_y_ratio"],
            parameters["max_height_ratio"],
            geometry["total_raw_component_merges"],
            geometry["average_proposals_per_image"],
        )
    chosen = min(near_best, key=selection_key)
    return {
        "mode": PRACTICAL_REMAP_V1,
        "classification": "PRACTICAL_APPROXIMATION",
        "split": "ICDAR13 official training all 229; test data not loaded or consulted",
        "declared_grid": {key: list(value) for key, value in TRAIN_GRID.items()},
        "fixed_non_grid_parameters": {"pad": PADDING, "connectivity": CONNECTIVITY, "nmax": NMAX},
        "selection_rule_declared_before_execution": {
            "primary": "maximize proposal-level F1 at IoU>=0.5",
            "near_tie": "within absolute F1 0.005 of the maximum",
            "ordering": [
                "more conservative geometry: lower horizontal gap, higher vertical overlap, lower center-y distance, lower height ratio",
                "fewer raw component merges",
                "fewer retained proposals",
            ],
        },
        "best_f1": best_f1,
        "near_best_variant_count": len(near_best),
        "selected": chosen,
        "grid_results": variants,
    }


def write_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["sample_id"])
        writer.writeheader()
        writer.writerows(rows)


def train_records_from_manifest(
    manifest: SegmentationManifest,
    data_root: Path,
    predict: Callable[[np.ndarray], np.ndarray],
) -> tuple[list[tuple[str, np.ndarray, np.ndarray, tuple[TrainWordBox, ...]]], dict[str, Any]]:
    """Create in-memory train predictions; GT is used only as geometry reference."""

    if len(manifest.samples) != 229 or manifest.split_role != "train":
        raise ValueError("Calibration requires exactly the 229-image train manifest")
    records = []
    source_hashes = []
    for sample in manifest.samples:
        image_path = (data_root / sample.image_path).resolve()
        mask_path = (data_root / sample.mask_path).resolve()
        annotation_path = (data_root / sample.validity_annotation_path).resolve()
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert("RGB"))
        prediction = predict(image)
        if prediction.shape != image.shape[:2] or prediction.dtype != np.uint8:
            raise ValueError("Frozen segmentation prediction violates source image geometry")
        words = train_task22_word_boxes(mask_path, annotation_path)
        records.append((sample.sample_id, image, prediction, words))
        source_hashes.append({
            "sample_id": sample.sample_id,
            "image_sha256": _sha256_file(image_path),
            "mask_sha256": _sha256_file(mask_path),
            "annotation_sha256": _sha256_file(annotation_path),
            "word_box_count": len(words),
        })
    return records, {"sample_count": len(records), "samples": source_hashes}


def evaluate_test_masks(
    samples: Sequence[ICDAR13RecognitionSample],
    masks: Sequence[np.ndarray],
    chosen: GroupedWordBlockParams,
) -> dict[str, Any]:
    """Frozen test geometry for Algorithm 1 and selected practical remap."""

    if len(samples) != 233 or len(masks) != 233:
        raise ValueError("Frozen test evaluation requires exactly 233 samples/masks")
    practical_per_image = []
    algorithm1_per_image = []
    per_sample = []
    algorithm1_params = BlockParam(150, PADDING, CONNECTIVITY, APPROXIMATE_TIE_BREAK, NMAX)
    for sample, mask in zip(samples, masks):
        image = np.zeros((*mask.shape, 3), dtype=np.uint8)
        gt_boxes = [_exclusive_test_box(word.bbox_xyxy) for word in sample.words]
        practical, practical_boxes, pre_cap, merges, multi = _practical_per_image(mask, image, chosen)
        alg1 = localize_mask(mask, image, algorithm1_params)
        alg1_boxes = [block.padded_bbox for block in alg1.retained]
        practical_per_image.append((practical_boxes, gt_boxes, pre_cap, merges, multi))
        algorithm1_per_image.append((alg1_boxes, gt_boxes, len(alg1.retained) + sum(item.reason == "top_10_limit" for item in alg1.rejected), 0, 0))
        per_sample.append({
            "sample_id": sample.sample_id,
            "gt_word_count": len(gt_boxes),
            "algorithm1_retained_proposal_count": len(alg1_boxes),
            "practical_remap_groups_before_cap": pre_cap,
            "practical_remap_retained_proposal_count": len(practical_boxes),
            "practical_remap_raw_component_merges": merges,
            "practical_remap_multi_component_group_count": multi,
        })
    return {
        "algorithm1_paper_faithful": proposal_geometry(algorithm1_per_image).as_dict(),
        "practical_remap_v1": proposal_geometry(practical_per_image).as_dict(),
        "per_sample": per_sample,
    }


__all__ = [
    "CONNECTIVITY", "NMAX", "PADDING", "SELECTION_F1_ATOL", "TRAIN_GRID",
    "ProposalGeometry", "TrainWordBox", "calibrate_on_train", "declared_grid",
    "evaluate_test_masks", "proposal_geometry", "train_records_from_manifest",
    "train_task22_word_boxes", "write_csv",
]
