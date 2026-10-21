#!/usr/bin/env python3
"""TRAIN-only OOF diagnostic for ``supervised-word-boundary-v1``.

The script consumes verified, frozen 128x128 segmentation masks. GT word
annotations create TRAIN labels and evaluate geometry; neither GT nor OCR is
an inference feature.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import tempfile
from typing import Any, Sequence

import cv2
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.line_word_blocks import Atom, V2PreparedMask, V2WordBlockParams, prepare_word_blocks_v2
from localize.practical_remap_failure_analysis import _box_union, _region, classify_failure
from localize.practical_remap_protocol import _box_iou, proposal_geometry, train_task22_word_boxes
from pipeline.frozen_segmentation import backproject_binary_mask
from segmentation.data import icdar13_task22_rgb_to_binary_mask, load_segmentation_manifest


EXPECTED_CACHE_SHA256 = "e9a7ce5badfbc3d99885fef5e655b20f7f5f167f29defa2d3f3c7e91c6d5851a"
FEATURE_NAMES = (
    "normalized_horizontal_gap", "vertical_overlap", "normalized_center_y",
    "height_ratio", "width_ratio", "area_ratio", "gap_over_local_median",
    "gap_over_local_mad", "x_overlap", "foreground_density",
)
BOUNDARY_THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass(frozen=True)
class WordEvidence:
    foreground_recall: float
    raw_component_ids: tuple[int, ...]
    raw_boxes: tuple[tuple[int, int, int, int], ...]


@dataclass(frozen=True)
class SupervisedTrainRecord:
    sample_id: str
    width: int
    height: int
    word_boxes: tuple[tuple[int, int, int, int], ...]
    prepared: V2PreparedMask
    pairs: tuple[dict[str, Any], ...]
    supervised_word_evidence: tuple[WordEvidence, ...]
    v1_groups_before_cap: tuple[dict[str, Any], ...]
    v1_retained_raw_ids: frozenset[tuple[int, ...]]
    v1_word_evidence: tuple[WordEvidence, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _area(box: Sequence[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _union_box(boxes: Sequence[Sequence[int]]) -> tuple[int, int, int, int]:
    if not boxes:
        raise ValueError("cannot union an empty set of boxes")
    return (
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    )


def _map_native_box(box: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        math.floor(box[0] * width / 128), math.floor(box[1] * height / 128),
        math.ceil(box[2] * width / 128), math.ceil(box[3] * height / 128),
    )


def _native_word_box(box: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    return (
        max(0, math.floor(box[0] * 128 / width)),
        max(0, math.floor(box[1] * 128 / height)),
        min(128, math.ceil(box[2] * 128 / width)),
        min(128, math.ceil(box[3] * 128 / height)),
    )


def _preparation_params() -> V2WordBlockParams:
    """Frozen geometric pre-stages; the classifier replaces the boundary rule."""
    return V2WordBlockParams(
        family="absolute", absolute_gap_floor=0.35, local_window=2,
        local_gap_ratio=2.0, local_fallback_absolute_floor=0.35,
        line_vertical_overlap_min=0.7, line_max_center_y_ratio=0.5,
        line_max_height_ratio=2.5, fragment_x_overlap_min=0.5,
        fragment_max_vertical_gap_ratio=0.5, fragment_small_height_ratio=0.45,
        pad=6,
    )


def _raw_boxes_from_labels(labels: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    boxes = {}
    for component_id in (int(value) for value in np.unique(labels) if value):
        ys, xs = np.nonzero(labels == component_id)
        boxes[component_id] = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return boxes


def _word_evidence(
    labels: np.ndarray,
    raw_boxes: dict[int, tuple[int, int, int, int]],
    prediction: np.ndarray,
    target: np.ndarray,
    word_boxes: Sequence[Sequence[int]],
) -> tuple[WordEvidence, ...]:
    values = []
    for word_box in word_boxes:
        region = _region(prediction.shape, word_box)
        raw_ids = tuple(sorted(int(value) for value in np.unique(labels[region]) if value))
        target_pixels = int((target & region).sum())
        predicted_target_pixels = int((target & prediction & region).sum())
        values.append(WordEvidence(
            foreground_recall=predicted_target_pixels / target_pixels if target_pixels else 0.0,
            raw_component_ids=raw_ids,
            raw_boxes=tuple(raw_boxes[value] for value in raw_ids),
        ))
    return tuple(values)


def _atom_gt_assignments(
    prepared: V2PreparedMask,
    native_labels: np.ndarray,
    word_boxes: Sequence[Sequence[int]],
    width: int,
    height: int,
) -> dict[int, int | None]:
    """Assign unambiguous atoms to one TRAIN word; ambiguous atoms are ignored."""
    native_words = [_native_word_box(box, width, height) for box in word_boxes]
    assignments = {}
    for atom in prepared.atoms:
        atom_pixels = np.isin(native_labels, atom.raw_component_ids)
        total = int(atom_pixels.sum())
        fractions = []
        for word_index, box in enumerate(native_words):
            inside = int(atom_pixels[box[1]:box[3], box[0]:box[2]].sum())
            fractions.append((inside / max(1, total), word_index))
        fractions.sort(reverse=True)
        best_fraction, best_word = fractions[0] if fractions else (0.0, -1)
        second_fraction = fractions[1][0] if len(fractions) > 1 else 0.0
        assignments[atom.atom_id] = best_word if best_fraction >= 0.5 and second_fraction < 0.2 else None
    return assignments


def _pair_features(members: Sequence[Atom], gaps: Sequence[float], position: int) -> tuple[float, ...]:
    left, right = members[position], members[position + 1]
    left_height, right_height = left.bbox[3] - left.bbox[1], right.bbox[3] - right.bbox[1]
    left_width, right_width = left.bbox[2] - left.bbox[0], right.bbox[2] - right.bbox[0]
    line_height = median(atom.bbox[3] - atom.bbox[1] for atom in members)
    gap = max(0, right.bbox[0] - left.bbox[2])
    vertical_overlap = max(0, min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1]))
    x_overlap = max(0, min(left.bbox[2], right.bbox[2]) - max(left.bbox[0], right.bbox[0]))
    local = [value for index, value in enumerate(gaps) if index != position and abs(index - position) <= 2]
    local_median = median(local) if local else 0.0
    local_mad = median(abs(value - local_median) for value in local) if local else 0.0
    pair_union = _union_box((left.bbox, right.bbox))
    return (
        gap / line_height,
        vertical_overlap / min(left_height, right_height),
        abs((left.bbox[1] + left.bbox[3]) / 2 - (right.bbox[1] + right.bbox[3]) / 2) / line_height,
        max(left_height, right_height) / min(left_height, right_height),
        max(left_width, right_width) / min(left_width, right_width),
        max(left.foreground_area, right.foreground_area) / min(left.foreground_area, right.foreground_area),
        gaps[position] / local_median if local_median > 0 else 0.0,
        (gaps[position] - local_median) / (1.4826 * local_mad) if local_mad > 0 else 0.0,
        x_overlap / min(left_width, right_width),
        (left.foreground_area + right.foreground_area) / _area(pair_union),
    )


def _adjacent_pairs(prepared: V2PreparedMask, assignments: dict[int, int | None]) -> tuple[dict[str, Any], ...]:
    atoms = {atom.atom_id: atom for atom in prepared.atoms}
    pairs = []
    for line in prepared.line_candidates:
        members = [atoms[int(value)] for value in line["atom_ids"]]
        gaps = [
            max(0, right.bbox[0] - left.bbox[2])
            / ((right.bbox[3] - right.bbox[1] + left.bbox[3] - left.bbox[1]) / 2)
            for left, right in zip(members, members[1:])
        ]
        for position, (left, right) in enumerate(zip(members, members[1:])):
            left_word, right_word = assignments[left.atom_id], assignments[right.atom_id]
            label = None if left_word is None or right_word is None else int(left_word != right_word)
            pairs.append({
                "line_id": int(line["line_id"]), "position": position,
                "left_atom_id": left.atom_id, "right_atom_id": right.atom_id,
                "features": _pair_features(members, gaps, position), "label": label,
            })
    return tuple(pairs)


def _prepare_record(
    sample_id: str,
    native_mask: np.ndarray,
    image_path: Path,
    source_target_path: Path,
    validity_annotation_path: Path,
    v1_params: GroupedWordBlockParams,
) -> SupervisedTrainRecord:
    with Image.open(image_path) as opened:
        width, height = opened.size
    word_boxes = tuple(word.bbox for word in train_task22_word_boxes(source_target_path, validity_annotation_path))
    with Image.open(source_target_path) as opened:
        target = (np.asarray(icdar13_task22_rgb_to_binary_mask(opened), dtype=np.uint8) > 0).astype(np.uint8)
    full_prediction = backproject_binary_mask(native_mask, (height, width))

    native_count, native_labels, _, _ = cv2.connectedComponentsWithStats(native_mask, 8)
    full_native_labels = cv2.resize(native_labels.astype(np.int32), (width, height), interpolation=cv2.INTER_NEAREST)
    native_raw_boxes = _raw_boxes_from_labels(full_native_labels)
    if len(native_raw_boxes) != native_count - 1:
        raise RuntimeError(f"native component loss during coordinate remap for sample {sample_id}")

    prepared = prepare_word_blocks_v2(native_mask, _preparation_params())
    assignments = _atom_gt_assignments(prepared, native_labels, word_boxes, width, height)
    pairs = _adjacent_pairs(prepared, assignments)
    supervised_evidence = _word_evidence(full_native_labels, native_raw_boxes, full_prediction, target, word_boxes)

    # v1 remains its frozen full-resolution implementation and consumes the
    # same cached native mask after qualified binary backprojection.
    v1_result = group_word_blocks(full_prediction, full_prediction, v1_params)
    _, v1_labels, _, _ = cv2.connectedComponentsWithStats(full_prediction, 8)
    v1_raw_boxes = {value.component_id: value.bbox for value in v1_result.raw_components}
    v1_evidence = _word_evidence(v1_labels, v1_raw_boxes, full_prediction, target, word_boxes)
    return SupervisedTrainRecord(
        sample_id=sample_id, width=width, height=height, word_boxes=word_boxes,
        prepared=prepared, pairs=pairs, supervised_word_evidence=supervised_evidence,
        v1_groups_before_cap=tuple(dict(value) for value in v1_result.groups_before_cap),
        v1_retained_raw_ids=frozenset(tuple(value.raw_component_ids) for value in v1_result.retained),
        v1_word_evidence=v1_evidence,
    )


def _supervised_groups(
    record: SupervisedTrainRecord,
    boundary_probabilities: Sequence[float],
    threshold: float,
) -> tuple[tuple[dict[str, Any], ...], frozenset[tuple[int, ...]]]:
    atoms = {atom.atom_id: atom for atom in record.prepared.atoms}
    probability_by_pair = {
        (pair["line_id"], pair["position"]): probability
        for pair, probability in zip(record.pairs, boundary_probabilities)
    }
    atom_groups = []
    for line in record.prepared.line_candidates:
        line_id = int(line["line_id"])
        atom_ids = tuple(int(value) for value in line["atom_ids"])
        start = 0
        for position in range(len(atom_ids) - 1):
            if probability_by_pair[(line_id, position)] >= threshold:
                atom_groups.append((line_id, atom_ids[start:position + 1]))
                start = position + 1
        atom_groups.append((line_id, atom_ids[start:]))

    groups = []
    for group_id, (line_id, atom_ids) in enumerate(atom_groups, start=1):
        members = [atoms[value] for value in atom_ids]
        native_box = _union_box(tuple(value.bbox for value in members))
        box = _map_native_box(native_box, record.width, record.height)
        padded_box = (
            max(0, box[0] - 6), max(0, box[1] - 6),
            min(record.width, box[2] + 6), min(record.height, box[3] + 6),
        )
        groups.append({
            "group_id": group_id, "line_id": line_id, "atom_ids": atom_ids,
            "raw_component_ids": tuple(sorted(
                component_id for member in members for component_id in member.raw_component_ids
            )),
            "foreground_area": sum(member.foreground_area for member in members),
            "bbox": box, "padded_bbox": padded_box,
        })
    groups.sort(key=lambda value: (
        -value["foreground_area"], value["padded_bbox"][1],
        value["padded_bbox"][0], value["raw_component_ids"],
    ))
    retained = frozenset(tuple(value["raw_component_ids"]) for value in groups[:10])
    return tuple(groups), retained


def _evaluate_groups(
    records: Sequence[SupervisedTrainRecord],
    groups_and_retained: Sequence[tuple[Sequence[dict[str, Any]], frozenset[tuple[int, ...]]]],
    *,
    evidence_attribute: str,
) -> dict[str, Any]:
    categories = (
        "NO_FOREGROUND", "GROUPING_UNDERMERGE", "GROUPING_OVERMERGE",
        "TOP10_TRUNCATION", "GEOMETRY_LIMIT", "OTHER",
    )
    category_counts, wordness, per_image, per_word = Counter(), Counter(), [], []
    for record, (groups, retained_ids) in zip(records, groups_and_retained):
        evidence: tuple[WordEvidence, ...] = getattr(record, evidence_attribute)
        component_to_words: dict[int, set[int]] = {}
        for word_index, word_evidence in enumerate(evidence, start=1):
            for component_id in word_evidence.raw_component_ids:
                component_to_words.setdefault(component_id, set()).add(word_index)
        group_to_words = {
            tuple(group["raw_component_ids"]): sorted(set().union(*(
                component_to_words.get(component_id, set()) for component_id in group["raw_component_ids"]
            )))
            for group in groups
        }
        retained = [group for group in groups if tuple(group["raw_component_ids"]) in retained_ids]
        for group in retained:
            significant = sum(
                _intersection(group["padded_bbox"], word_box) / max(1, _area(word_box)) >= 0.25
                for word_box in record.word_boxes
            )
            best_iou = max((_box_iou(group["padded_bbox"], word_box) for word_box in record.word_boxes), default=0.0)
            wordness[
                "MULTIWORD_MERGE" if significant >= 2 else
                "WORD_GOOD" if best_iou >= 0.5 else
                "WORD_PARTIAL" if significant == 1 else "UNMATCHED"
            ] += 1

        for word_index, (word_box, word_evidence) in enumerate(
            zip(record.word_boxes, evidence), start=1
        ):
            associated = [
                group for group in groups
                if set(group["raw_component_ids"]).intersection(word_evidence.raw_component_ids)
            ]
            best = max(retained, key=lambda group: _box_iou(group["padded_bbox"], word_box), default=None)
            best_iou = _box_iou(best["padded_bbox"], word_box) if best else 0.0
            if best_iou >= 0.5:
                category_counts["SUCCESS"] += 1
                per_word.append({
                    "sample_id": record.sample_id,
                    "gt_word_index": word_index,
                    "failure_category": "SUCCESS",
                    "best_group_iou": best_iou,
                })
                continue
            pre_cap_valid = any(_box_iou(group["padded_bbox"], word_box) >= 0.5 for group in groups)
            oracle_box = _box_union(word_evidence.raw_boxes)
            oracle_iou = _box_iou(oracle_box, word_box) if oracle_box is not None else 0.0
            best_group_word_count = len(group_to_words[tuple(best["raw_component_ids"])]) if best else 0
            category = classify_failure(
                foreground_recall=word_evidence.foreground_recall,
                best_retained_group_iou=best_iou,
                pre_cap_has_valid_group=pre_cap_valid,
                oracle_union_iou=oracle_iou,
                associated_group_count=len(associated),
                best_group_gt_word_count=best_group_word_count,
            )
            category_counts[category] += 1
            per_word.append({
                "sample_id": record.sample_id,
                "gt_word_index": word_index,
                "failure_category": category,
                "best_group_iou": best_iou,
            })

        proposals = [tuple(group["padded_bbox"]) for group in retained]
        positive_id_counts = [
            len([value for value in group["raw_component_ids"] if value > 0])
            for group in groups
        ]
        multi_groups = sum(count > 1 for count in positive_id_counts)
        merges = sum(max(0, count - 1) for count in positive_id_counts)
        per_image.append((proposals, record.word_boxes, len(groups), merges, multi_groups))

    geometry = proposal_geometry(per_image).as_dict()
    failures = sum(category_counts[category] for category in categories)
    return {
        "geometry": geometry,
        "failure_breakdown": {
            "success_count": category_counts["SUCCESS"], "failure_count": failures,
            "categories": {
                category: {
                    "count": category_counts[category],
                    "fraction_of_failures": category_counts[category] / failures if failures else 0.0,
                }
                for category in categories
            },
        },
        "wordness": {name: wordness[name] for name in (
            "WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED"
        )},
        "gt_words_reconstructed_by_single_group": category_counts["SUCCESS"],
        "per_word": per_word,
    }


def _model_factories() -> dict[str, tuple[int, Any]]:
    return {
        "logistic_regression": (0, lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=202503)
        )),
        "shallow_gradient_boosting": (1, lambda: GradientBoostingClassifier(
            n_estimators=50, max_depth=2, learning_rate=0.05, random_state=202503
        )),
    }


def _summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    geometry, categories = evaluation["geometry"], evaluation["failure_breakdown"]["categories"]
    return {
        "gt_words": geometry["gt_words"],
        "iou_ge_0_25": geometry["gt_with_proposal_iou_ge_0_25"],
        "iou_ge_0_50": geometry["gt_with_proposal_iou_ge_0_50"],
        "iou_ge_0_75": geometry["gt_with_proposal_iou_ge_0_75"],
        "precision_iou_ge_0_50": geometry["proposal_precision_iou_ge_0_50"],
        "recall_iou_ge_0_50": geometry["recall_iou_ge_0_50"],
        "f1_iou_ge_0_50": geometry["f1_proposal_level_iou_ge_0_50"],
        "undermerge": categories["GROUPING_UNDERMERGE"]["count"],
        "overmerge": categories["GROUPING_OVERMERGE"]["count"],
        "proposals_per_image": geometry["average_proposals_per_image"],
        "gt_words_reconstructed_by_single_group": evaluation["gt_words_reconstructed_by_single_group"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--v1-failure-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    cache_manifest = args.cache_manifest.resolve()
    if _sha256_file(cache_manifest) != EXPECTED_CACHE_SHA256:
        raise RuntimeError("cache manifest SHA256 mismatch")

    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    manifest = load_segmentation_manifest(args.train_manifest.resolve())
    if (
        manifest.split != "official-training-all-229-author-protocol"
        or manifest.split_role != "train" or len(manifest.samples) != 229
        or cache.get("sample_count") != 229 or cache.get("test_data_loaded") is not False
    ):
        raise ValueError("supervised boundary evaluation permits only official TRAIN-229")
    cache_by_id = {str(value["sample_id"]): value for value in cache["samples"]}
    if set(cache_by_id) != {sample.sample_id for sample in manifest.samples}:
        raise RuntimeError("cache/manifest sample identities differ")

    v1_calibration = json.loads(args.v1_calibration.read_text(encoding="utf-8"))
    v1_guard = v1_calibration["train_only_guard"]
    if (
        v1_guard["split_role"] != "train" or v1_guard["sample_count"] != 229
        or v1_guard["test_data_loaded"] is not False
    ):
        raise ValueError("v1 calibration provenance is not TRAIN-229 only")
    v1_params = GroupedWordBlockParams(**v1_calibration["calibration"]["selected"]["parameters"])

    records = []
    file_hashes_verified = array_hashes_verified = image_hashes_verified = 0
    for sample_index, sample in enumerate(manifest.samples, start=1):
        cached = cache_by_id[sample.sample_id]
        mask_path = (cache_manifest.parent / cached["native_mask_path"]).resolve()
        if _sha256_file(mask_path) != cached["native_mask_file_sha256"]:
            raise RuntimeError(f"cached mask file hash mismatch: sample {sample.sample_id}")
        file_hashes_verified += 1
        with Image.open(mask_path) as opened:
            native_mask = (np.asarray(opened.convert("L")) > 0).astype(np.uint8)
        if native_mask.shape != (128, 128):
            raise RuntimeError(f"cached mask shape mismatch: sample {sample.sample_id}")
        if hashlib.sha256(native_mask.tobytes()).hexdigest() != cached["native_mask_array_sha256"]:
            raise RuntimeError(f"cached mask array hash mismatch: sample {sample.sample_id}")
        array_hashes_verified += 1
        image_path = (args.data_root / sample.image_path).resolve()
        if _sha256_file(image_path) != cached["image_sha256"]:
            raise RuntimeError(f"source image hash mismatch: sample {sample.sample_id}")
        image_hashes_verified += 1
        records.append(_prepare_record(
            sample.sample_id, native_mask, image_path,
            (args.data_root / sample.mask_path).resolve(),
            (args.data_root / sample.validity_annotation_path).resolve(), v1_params,
        ))
        if sample_index % 10 == 0 or sample_index == 229:
            print(f"[verified cache + prepared] {sample_index}/229", flush=True)

    features, labels, image_groups = [], [], []
    for image_index, record in enumerate(records):
        for pair in record.pairs:
            if pair["label"] is not None:
                features.append(pair["features"])
                labels.append(pair["label"])
                image_groups.append(image_index)
    x, y = np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)
    image_groups_array = np.asarray(image_groups, dtype=np.int64)

    v1_evaluation = _evaluate_groups(
        records,
        [(record.v1_groups_before_cap, record.v1_retained_raw_ids) for record in records],
        evidence_attribute="v1_word_evidence",
    )
    frozen_v1_geometry = v1_calibration["calibration"]["selected"]["geometry"]
    for key in (
        "gt_words", "gt_with_proposal_iou_ge_0_25", "gt_with_proposal_iou_ge_0_50",
        "gt_with_proposal_iou_ge_0_75", "retained_proposals", "true_positive_proposals_iou_ge_0_50",
    ):
        if v1_evaluation["geometry"][key] != frozen_v1_geometry[key]:
            raise RuntimeError(f"v1 geometry drift for {key}")
    frozen_failure_report = json.loads(args.v1_failure_analysis.read_text(encoding="utf-8"))
    for category in v1_evaluation["failure_breakdown"]["categories"]:
        observed = v1_evaluation["failure_breakdown"]["categories"][category]["count"]
        expected = frozen_failure_report["failure_breakdown"]["categories"][category]["count"]
        if observed != expected:
            raise RuntimeError(f"v1 failure taxonomy drift for {category}: {observed} != {expected}")

    variants = []
    folds = GroupKFold(n_splits=5)
    for model_name, (complexity_rank, factory) in _model_factories().items():
        oof_probabilities = [[0.5] * len(record.pairs) for record in records]
        for fold_index, (train_indices, validation_indices) in enumerate(
            folds.split(np.arange(len(records)), groups=np.arange(len(records))), start=1
        ):
            train_images = set(train_indices.tolist())
            train_pair_indices = np.asarray([
                index for index, image_index in enumerate(image_groups_array) if image_index in train_images
            ])
            estimator = factory()
            estimator.fit(x[train_pair_indices], y[train_pair_indices])
            for image_index in validation_indices.tolist():
                matrix = np.asarray([pair["features"] for pair in records[image_index].pairs], dtype=np.float64)
                oof_probabilities[image_index] = estimator.predict_proba(matrix)[:, 1].tolist() if len(matrix) else []
            print(f"[OOF] model={model_name} fold={fold_index}/5", flush=True)

        for threshold in BOUNDARY_THRESHOLDS:
            evaluation = _evaluate_groups(
                records,
                [_supervised_groups(record, probabilities, threshold) for record, probabilities in zip(records, oof_probabilities)],
                evidence_attribute="supervised_word_evidence",
            )
            variants.append({
                "model": model_name, "complexity_rank": complexity_rank,
                "threshold": threshold, "evaluation": evaluation,
            })

    best_f1 = max(value["evaluation"]["geometry"]["f1_proposal_level_iou_ge_0_50"] for value in variants)
    exact_best = [
        value for value in variants
        if abs(value["evaluation"]["geometry"]["f1_proposal_level_iou_ge_0_50"] - best_f1) <= 1e-12
    ]
    selected = min(exact_best, key=lambda value: (
        value["evaluation"]["failure_breakdown"]["categories"]["GROUPING_OVERMERGE"]["count"],
        -value["evaluation"]["geometry"]["recall_iou_ge_0_50"], value["complexity_rank"],
    ))

    final_estimator = _model_factories()[selected["model"]][1]()
    final_estimator.fit(x, y)
    model_path = output.with_suffix(".model.joblib")
    joblib.dump({
        "mode": "supervised-word-boundary-v1", "classification": "PRACTICAL_APPROXIMATION",
        "feature_names": FEATURE_NAMES, "boundary_threshold": selected["threshold"],
        "estimator": final_estimator,
    }, model_path)

    selected_summary, v1_summary = _summary(selected["evaluation"]), _summary(v1_evaluation)
    go = (
        selected_summary["f1_iou_ge_0_50"] > v1_summary["f1_iou_ge_0_50"]
        and selected_summary["overmerge"] <= v1_summary["overmerge"]
    )
    report = {
        "schema_version": 1, "mode": "supervised-word-boundary-v1",
        "classification": "PRACTICAL_APPROXIMATION",
        "cache": {
            "path": str(cache_manifest), "manifest_sha256": EXPECTED_CACHE_SHA256,
            "verified_samples": 229, "mask_file_hashes_verified": file_hashes_verified,
            "mask_array_hashes_verified": array_hashes_verified,
            "source_image_hashes_verified": image_hashes_verified,
            "checkpoint_sha256": cache["checkpoint"]["sha256"],
            "native_shape": cache["inference"]["native_shape"],
            "backprojection_applied_in_cache": cache["inference"]["backprojection_applied"],
        },
        "train_only": {
            "manifest": str(args.train_manifest.resolve()), "images": 229,
            "gt_words": sum(len(record.word_boxes) for record in records),
            "test_loaded": False, "ocr_called": False, "gt_used_as_inference_feature": False,
        },
        "pipeline": [
            "frozen_native_mask_128", "raw_connected_components",
            "fixed_geometric_fragment_consolidation", "fixed_vertical_text_lines",
            "left_to_right_adjacent_pairs", "OOF_boundary_classifier",
            "cuts_to_word_groups", "bbox_remap_to_original", "padding_6_original_pixels",
            "foreground_area_sort", "top10",
        ],
        "features": FEATURE_NAMES,
        "labels": {
            "labeled_pairs": len(y), "same_word": int((y == 0).sum()),
            "boundary": int((y == 1).sum()),
            "ambiguous_pairs_ignored": sum(len(record.pairs) for record in records) - len(y),
        },
        "validation": {
            "method": "5-fold GroupKFold by image",
            "model_candidates": tuple(_model_factories()),
            "boundary_thresholds": BOUNDARY_THRESHOLDS,
            "selection": "maximum OOF proposal F1@IoU.5; exact-tie: fewer overmerge, higher recall, simpler model",
        },
        "v1_provenance": {
            "calibration_path": str(args.v1_calibration.resolve()),
            "calibration_sha256": _sha256_file(args.v1_calibration.resolve()),
            "failure_analysis_path": str(args.v1_failure_analysis.resolve()),
            "failure_analysis_sha256": _sha256_file(args.v1_failure_analysis.resolve()),
        },
        "oof_v1": v1_evaluation, "oof_supervised": selected["evaluation"],
        "selected": {
            "model": selected["model"], "threshold": selected["threshold"],
            "features": FEATURE_NAMES, "final_fit_on_all_labeled_train_pairs": True,
            "model_artifact": str(model_path), "model_sha256": _sha256_file(model_path),
        },
        "decision": {
            "label": "GO" if go else "NO-GO",
            "criterion": "strict OOF F1 improvement over v1 without increased GROUPING_OVERMERGE",
        },
        "grid": [
            {"model": value["model"], "threshold": value["threshold"], "summary": _summary(value["evaluation"])}
            for value in variants
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, output)
    print(json.dumps({
        "v1": v1_summary, "supervised": selected_summary,
        "selected": report["selected"], "decision": report["decision"],
    }, indent=2), flush=True)
    print(output, flush=True)
    print(_sha256_file(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
