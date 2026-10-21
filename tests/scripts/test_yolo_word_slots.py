"""Identity and geometry boundaries for the lightweight YOLO word-slot run."""
from __future__ import annotations

from hashlib import sha1
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.evaluate_totaltext_yolo_word_slots import (
    boxes_to_slots, git_blob_sha1, pil_to_yolo_bgr, validate_checkpoint, validate_config,
)


REPO = Path(__file__).resolve().parents[2]


def test_exact_git_blob_identity_and_corruption(tmp_path) -> None:
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"abc")
    expected = sha1(b"blob 3\0abc").hexdigest()
    assert git_blob_sha1(path) == expected
    source = {"size_bytes": 3, "git_blob_sha1": expected,
              "repository": "example/example", "commit": "0" * 40,
              "path": "weights.pt"}
    report = validate_checkpoint(path, source)
    assert report["sha256"]
    with pytest.raises(ValueError, match="SHA1 differs"):
        validate_checkpoint(path, {**source, "git_blob_sha1": "0" * 40})
    with pytest.raises(FileNotFoundError, match="missing"):
        validate_checkpoint(tmp_path / "absent.pt", source)


def test_word_object_boxes_use_original_xyxy_and_reject_empty_clipping() -> None:
    slots, rejected = boxes_to_slots(
        np.array([[1.2, 2.1, 8.6, 6.9], [11, 11, 15, 15]], dtype=float),
        np.array([0.8, 0.9]), np.array([0, 0]), (10, 10), 0.05)
    assert slots == [{"source_index": 0, "score": 0.8, "bbox": [1, 2, 9, 7]}]
    assert rejected[0]["source_index"] == 1
    assert rejected[0]["reason"] == "no_positive_in_image_bbox"
    with pytest.raises(ValueError, match="one-class"):
        boxes_to_slots(np.array([[1, 2, 3, 4]]), np.array([0.8]),
                       np.array([1]), (10, 10), 0.05)
    with pytest.raises(ValueError, match="one-class"):
        boxes_to_slots(np.array([[3, 2, 1, 4]]), np.array([0.8]),
                       np.array([0]), (10, 10), 0.05)


def test_approximate_config_keeps_inference_thresholds_explicit() -> None:
    config = json.loads((REPO / "configs/approximate/totaltext_yolo_word_slots_v1.json").read_text())
    validate_config(config)
    config["score_controls"] = [0.01]
    with pytest.raises(ValueError, match="invalid isolated"):
        validate_config(config)


def test_pil_input_has_ultralytics_bgr_channel_order() -> None:
    image = Image.fromarray(np.array([[[255, 12, 3]]], dtype=np.uint8), mode="RGB")
    bgr = pil_to_yolo_bgr(image)
    assert bgr.tolist() == [[[3, 12, 255]]]
    assert bgr.flags.c_contiguous
