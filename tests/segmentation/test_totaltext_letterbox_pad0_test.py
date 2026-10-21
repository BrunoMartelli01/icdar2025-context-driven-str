"""CPU-only checks for frozen letterbox TEST evaluation and restart provenance."""

import json
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts.evaluate_totaltext_letterbox_pad0_test import (
    _config, _refresh_failed_threshold_protocol, _sample_metrics, _validate_cached_sample,
)
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _base_model_source_matches
from scripts.prune_segmentation_checkpoints import sha256_file
from segmentation.totaltext_image_frequency_augment import remap_letterbox_native_box


def test_letterbox_test_protocol_is_frozen_and_not_table5():
    config = _config()
    assert config["selected_variant"] == "letterbox"
    assert config["test_images"] == 300
    assert config["test_care_words"] == 2204
    assert config["padding_native_pixels"] == 0
    assert config["area_filter"] == config["maximum_proposals"] == "disabled"
    assert "not Table 5" in config["result_class"]


def test_historical_single_head_source_is_verified_exactly_after_additive_class():
    historical = "e46901c6c4b85796f9db594c276110f06bdf39c48823fccb720994dbee394ee6"
    assert _base_model_source_matches(historical)
    assert not _base_model_source_matches("0" * 64)


def test_failed_threshold_protocol_refresh_is_narrow_and_atomic(tmp_path):
    path = tmp_path / "protocol.json"
    old = {"train_threshold_selection_report_sha256": "train-sha",
           "evaluator_source_sha256": "old-source", "threshold": 0.65}
    path.write_text(json.dumps(old), encoding="utf-8")
    revised = {**old, "evaluator_source_sha256": "new-source"}
    _refresh_failed_threshold_protocol(path, revised)
    assert json.loads(path.read_text(encoding="utf-8")) == revised
    with pytest.raises(ValueError, match="different protocol"):
        _refresh_failed_threshold_protocol(path, {**revised, "threshold": 0.5})
    (tmp_path / "masks").mkdir()
    with pytest.raises(ValueError, match="different protocol"):
        _refresh_failed_threshold_protocol(path, {**revised, "evaluator_source_sha256": "later"})
    assert json.loads(path.read_text(encoding="utf-8")) == revised


def test_letterbox_crop_mapping_and_one_to_one_word_counts():
    assert remap_letterbox_native_box((64, 64, 192, 192), (64, 32)) == [16, 0, 48, 32]
    gt = [{"bbox": [0, 0, 10, 10], "text": "Hello!"},
          {"bbox": [20, 0, 30, 10], "text": "World"}]
    predictions = [{"bbox": [0, 0, 10, 10], "text": "hello"},
                   {"bbox": [20, 0, 30, 10], "text": "WORLD"},
                   {"bbox": [20, 0, 30, 10], "text": "WORLD"}]
    result = _sample_metrics(predictions, gt)
    assert result["gt_words"] == 2 and result["proposals"] == 3
    assert result["geometry_iou_ge_50"] == 2
    assert result["exact_casefold_iou_ge_50"] == 1
    assert result["exact_no_symbols_iou_ge_50"] == 2
    assert result["matched_gt_indices"]["exact_no_symbols_iou_ge_50"] == [0, 1]


def test_cached_letterbox_mask_requires_identical_protocol_and_checkpoint(tmp_path):
    annotation = tmp_path / "annotation.txt"
    annotation.write_text("frozen annotation", encoding="utf-8")
    mask = tmp_path / "mask.png"
    Image.new("L", (256, 256), 0).save(mask)
    shard_path = tmp_path / "shard.json"
    sample = SimpleNamespace(sample_id="total_text:1", image_sha256="image-sha")
    shard = {"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
             "protocol_sha256": "protocol-sha", "annotation_sha256": sha256_file(annotation),
             "checkpoint_sha256": "checkpoint-sha", "native_mask_file_sha256": sha256_file(mask)}
    shard_path.write_text(json.dumps(shard), encoding="utf-8")
    assert _validate_cached_sample(shard_path, mask, protocol_sha="protocol-sha",
                                   sample=sample, annotation_path=annotation,
                                   checkpoint_sha="checkpoint-sha") == shard
    with pytest.raises(ValueError, match="provenance mismatch"):
        _validate_cached_sample(shard_path, mask, protocol_sha="wrong",
                                sample=sample, annotation_path=annotation,
                                checkpoint_sha="checkpoint-sha")
