from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluate_cached_totaltext_padding import (
    _artifact, _checked_artifact, _plan, _seal, assert_baseline_proposals,
    correct_word_ids, crop_hash, evaluation_rows, proposals_for_padding,
)
from recognition.scene_text_v1_evaluation import evaluate_images
from segmentation.data import file_sha256


PLAN_PATH = Path("configs/approximate/totaltext_separated_v1_cached_padding_test.json")


def test_padding_preserves_count_filter_and_cap_and_clips() -> None:
    plan = _plan(PLAN_PATH)
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[0:10, 0:15] = 1  # exactly150: retained, clipped at upper/left border
    mask[110:120, 110:124] = 1  #140: rejected
    values = proposals_for_padding(mask, (333, 511), plan)
    assert {len(v["proposals"]) for v in values.values()} == {1}
    for pad, variant in values.items():
        row = variant["proposals"][0]
        assert variant["raw_contours"] == 2
        assert variant["candidates_before_cap"] == 1
        assert row["bbox_native"] == [0, 0, 15, 10]
        assert row["padded_bbox_native"] == [0, 0, 15 + int(pad), 10 + int(pad)]
        assert row["padded_bbox_original"][:2] == [0, 0]
    dense = np.zeros_like(mask)
    for y in (1, 30, 60):
        for x in (1, 30, 60, 90):
            dense[y:y+10, x:x+15] = 1
    values = proposals_for_padding(dense, (256, 256), plan)
    assert all(v["candidates_before_cap"] == 12 and len(v["proposals"]) == 10
               for v in values.values())
    with pytest.raises(ValueError, match="native128"):
        proposals_for_padding(np.zeros((256, 256), dtype=np.uint8), (256, 256), plan)


def test_separate_crop_iou_from_unchanged_word_localization() -> None:
    item = {"rank": 1, "contour_index": 0, "bbox": [10, 10, 30, 20],
            "padded_bbox_original": [0, 0, 50, 40], "crop_array_sha256": "crop"}
    shards = [{"sample_id": "a", "variants": {"6": {"proposals": [item]}}},
              {"sample_id": "empty", "variants": {"6": {"proposals": []}}}]
    gt = {"a": [{"bbox": [10, 10, 30, 20], "text": "BOOK"}],
          "empty": [{"bbox": [0, 0, 20, 20], "text": "missed"}]}
    ocr = {"crop": {"raw_text": "book", "sequence_confidence_native": 0.9}}
    crop_rows = evaluation_rows(shards, gt, ocr, 6, padded=True)
    detection_rows = evaluation_rows(shards, gt, ocr, 6, padded=False)
    crop = evaluate_images(crop_rows, text_available=True)
    detect = evaluate_images(detection_rows, text_available=True)
    assert crop["recognition"]["exact_casefold_spatial_one_to_one"] == 0
    assert detect["recognition"]["exact_casefold_spatial_one_to_one"] == 1
    assert detect["recognition"]["exact_casefold_rate_all_gt"] == 0.5
    assert detect["geometry"]["gt_instances"] == 2
    assert correct_word_ids(detection_rows) == {"a:0"}
    assert correct_word_ids(crop_rows) == set()
    assert crop_rows[0]["predictions"][0]["text"] == detection_rows[0]["predictions"][0]["text"]


def test_padding_can_swap_top_cap_identity_without_changing_count() -> None:
    plan = _plan(PLAN_PATH)
    plan["max_proposals_per_image"] = 1
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[:10, :20] = 1  # area200, padding clipped against the border
    mask[60:70, 60:75] = 1  # area150, padding expands on all four sides
    variants = proposals_for_padding(mask, (128, 128), plan)
    zero, six = variants["0"]["proposals"], variants["6"]["proposals"]
    assert len(zero) == len(six) == 1
    assert variants["0"]["candidates_before_cap"] == variants["6"]["candidates_before_cap"] == 2
    assert zero[0]["bbox_native"] == [0, 0, 20, 10]
    assert six[0]["bbox_native"] == [60, 60, 75, 70]
    assert zero[0]["contour_index"] != six[0]["contour_index"]


def test_exact_matching_does_not_double_count_or_strip_punctuation() -> None:
    rows = [{"sample_id": "a", "ground_truth": [
        {"bbox": [0, 0, 20, 20], "text": "Book"},
        {"bbox": [30, 0, 50, 20], "text": "Book!"}], "predictions": [
        {"rank": 1, "bbox": [0, 0, 20, 20], "text": "book"},
        {"rank": 2, "bbox": [0, 0, 20, 20], "text": "BOOK"},
        {"rank": 3, "bbox": [30, 0, 50, 20], "text": "book"}]}]
    assert correct_word_ids(rows) == {"a:0"}


def test_cache_rejects_corruption_and_foreign_protocol(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    _seal(path, {"padding": 2})
    _seal(path, {"padding": 2})
    with pytest.raises(ValueError, match="another protocol"):
        _seal(path, {"padding": 6})
    record = {"file": path.name, "sha256": file_sha256(path)}
    assert _checked_artifact(tmp_path, record) == path
    path.write_text("changed")
    with pytest.raises(ValueError, match="checksum"):
        _checked_artifact(tmp_path, record)
    with pytest.raises(ValueError, match="unsafe"):
        _artifact(tmp_path, "../outside.json")


def test_baseline_replay_checks_identity_and_coordinates() -> None:
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:25, 10:30] = 1
    proposals = proposals_for_padding(mask, (512, 768), _plan(PLAN_PATH))["6"]["proposals"]
    assert_baseline_proposals(proposals, proposals)
    altered = json.loads(json.dumps(proposals))
    altered[0]["contour_index"] += 1
    with pytest.raises(ValueError, match="source cache"):
        assert_baseline_proposals(proposals, altered)


def test_pixel_cache_key_contains_shape_and_pixels() -> None:
    a, b = np.zeros((2, 6, 3), dtype=np.uint8), np.zeros((3, 4, 3), dtype=np.uint8)
    assert crop_hash(a) != crop_hash(b)
    b[0, 0, 0] = 1
    assert crop_hash(b) != crop_hash(np.zeros_like(b))
    with pytest.raises(ValueError, match="nonempty"):
        crop_hash(np.empty((0, 4, 3), dtype=np.uint8))


@pytest.mark.parametrize("field,value", [("reproduction_mode", "strict"),
    ("padding_native_pixels", [0, 6, 6]), ("padding_native_pixels", [0, 2]),
    ("padding_native_pixels", [True, 6])])
def test_plan_rejects_unsupported_modes_and_grids(tmp_path: Path, field: str, value) -> None:
    plan = json.loads(PLAN_PATH.read_text())
    plan[field] = value
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        _plan(path)


def test_cpu_runtime_is_explicit_and_locked(tmp_path: Path) -> None:
    plan = json.loads(PLAN_PATH.read_text())
    plan["runtime"]["device"] = "cuda:0"
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="CPU FP32"):
        _plan(path)
