import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from recognition.scene_text_manifest import build_scene_text_manifest
from recognition.yolo_word_refinement import arm_predictions
from scripts.evaluate_scene_text_yolo_parseq import (
    DEFAULT_CONFIG, exclusive_bbox, qualify_dataset, read_pixels, unit_metrics, validate_config,
)
from scripts.crop_clip_runtime import sha
from scripts.run_crop_group_word_selection import evaluate_predictions


def test_transfer_rejects_test_specific_threshold_changes_and_strict_mode():
    config = json.loads(DEFAULT_CONFIG.read_text())
    validate_config(config)
    for key, value in (("reproduction_mode", "strict"), ("joint_controls", [0.55]),
                       ("padding_pixels", 5), ("matching_iou", 0.3)):
        changed = copy.deepcopy(config)
        changed[key] = value
        with pytest.raises(ValueError):
            validate_config(changed)


def test_inclusive_annotation_maxima_and_border_have_one_coordinate_convention():
    assert exclusive_bbox([0, 0, 0, 0], (10, 10)) == [0, 0, 1, 1]
    assert exclusive_bbox([2, 3, 10, 10], (10, 10)) == [2, 3, 10, 10]
    with pytest.raises(ValueError):
        exclusive_bbox([10, 0, 11, 5], (10, 10))


def test_ctw_line_ground_truth_is_preserved_and_not_split_into_word_boxes(tmp_path):
    images, annotations = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.fromarray(np.zeros((10, 20, 3), dtype=np.uint8)).save(images / "1001.png")
    polygon = [(0, 0), (3, 0), (6, 0), (9, 0), (12, 0), (15, 0), (19, 0),
               (19, 9), (15, 9), (12, 9), (9, 9), (6, 9), (3, 9), (0, 9)]
    coordinates = ",".join(str(x) for p in polygon for x in p)
    (annotations / "1001.txt").write_text(f"2\n{coordinates},Hello World\n{coordinates},###\n")
    manifest = tmp_path / "recognition.json"
    build_scene_text_manifest(dataset="ctw1500", split="test", data_root=tmp_path,
        image_directory=images, annotation_directory=annotations, output=manifest,
        enforce_official_count=False)
    records, truth, qualified = qualify_dataset("ctw1500",
        {"data_root": str(tmp_path), "manifest": str(manifest)},
        {"images": 1, "granularity": "text_line"}, tmp_path / "qualified")
    assert len(records) == 1
    assert "care" not in records[0] and "text" not in records[0]
    assert truth["ctw1500:1001"]["care"][0]["text"] == "Hello World"
    assert len(truth["ctw1500:1001"]["care"]) == 1
    assert qualified["care_gt_units"] == qualified["ignored_gt_units"] == 1
    assert qualified["granularity"] == "text_line"
    with pytest.raises(ValueError, match="granularity"):
        qualify_dataset("ctw1500", {"data_root": str(tmp_path), "manifest": str(manifest)},
            {"images": 1, "granularity": "word"}, tmp_path / "qualified")


def test_image_checksum_prevents_stale_pixels(tmp_path):
    image_path = tmp_path / "image.png"
    pixels = np.zeros((3, 3, 3), dtype=np.uint8)
    pixels[:, :, 0] = 255
    Image.fromarray(pixels).save(image_path)
    record = {"image_path": str(image_path), "image_sha256": sha(image_path),
              "sample_id": "fixture:1", "image_size_wh": [3, 3]}
    assert np.array_equal(read_pixels(record), pixels)
    Image.fromarray(np.zeros_like(pixels)).save(image_path)
    with pytest.raises(RuntimeError, match="bytes changed"):
        read_pixels(record)


def test_transfer_uses_existing_joint_selection_and_one_to_one_normalized_metric():
    config = json.loads(DEFAULT_CONFIG.read_text())
    a = {"candidate_id": "a", "bbox": [0, 0, 20, 10], "raw_text": "HELLO!",
         "detector_score": 0.9, "token_probabilities": [0.9] * 6,
         "token_log_probabilities": [float(np.log(0.9))] * 6}
    b = {**a, "candidate_id": "b", "detector_score": 0.8}
    rows = arm_predictions([a, b], config)[config["primary_method"]]
    assert [r["candidate_id"] for r in rows] == ["a"]
    gt = [{"text": "Hello", "bbox": [0, 0, 20, 10]}]
    assert evaluate_predictions(rows, gt, config["matching_iou"])["correct"] == 1
    assert evaluate_predictions([a, b], gt, config["matching_iou"])["correct"] == 1
    assert evaluate_predictions([a], [{**gt[0], "text": "Hello World"}], 0.5)["correct"] == 0


def test_line_metric_has_explicit_units_and_micro_denominator():
    result = unit_metrics(3, 5, 10, "text_line")
    assert result["gt_units"] == 10 and result["gt_granularity"] == "text_line"
    assert result["hmean"] == pytest.approx(0.4)
    assert unit_metrics(0, 0, 0, "word")["hmean"] == 0.0
