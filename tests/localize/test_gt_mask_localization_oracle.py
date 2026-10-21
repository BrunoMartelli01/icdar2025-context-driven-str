import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from diagnostics.gt_mask_localization_oracle import (
    checked_native, compare_success, native_proposals, original_proposals,
    summarize, topology,
)
from diagnostics.repository_contours_train import digest, pack_mask
from localize.grouped_word_blocks import GroupedWordBlockParams
from localize.repository_contours import RepositoryContourParams
from localize.v1_local_rescue import LocalRescueParams
from segmentation.data import icdar13_task22_rgb_to_target_and_validity
from segmentation.evaluate import resize_binary_target_to_native_logits
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


def params():
    return (RepositoryContourParams(150, 6, 10),
            GroupedWordBlockParams(.35, .7, .5, 2.5, 6),
            LocalRescueParams(.4, 1.5, .5, .5, 2.5))


def test_gt_is_glyph_mask_not_filled_word_rectangle_and_dc_is_retained():
    rgb = np.full((128, 128, 3), 255, dtype=np.uint8)
    rgb[10:30, 10:20] = [1, 2, 3]
    rgb[10:30, 25:35] = [4, 5, 6]
    rgb[70:75, 70:75] = [7, 8, 9]
    target, validity = icdar13_task22_rgb_to_target_and_validity(
        Image.fromarray(rgb), '# 7 8 9 ignored\n1 2 3 "a"\n4 5 6 "b"')
    native = resize_binary_target_to_native_logits(target).numpy().astype(np.uint8)
    assert native[20, 22] == 0
    assert native[72, 72] == 1
    assert np.asarray(validity)[72, 72] == 0
    result = topology(native, (128, 128), [[10, 10, 35, 30]])
    assert result["words"][0]["components_intersecting_word"] == 2
    assert result["words"][0]["foreground_pixels_in_word"] == 400
    assert result["words"][0]["largest_fraction"] == .5


def test_mapping_exclusive_edges_and_empty_word():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[120:128, 120:128] = 1
    data = topology(mask, (256, 512), [[480, 240, 512, 256], [0, 0, 20, 20]])
    assert data["words"][0]["foreground_pixels_in_word"] == 64
    assert data["words"][1]["components_intersecting_word"] == 0
    assert data["words"][1]["largest_fraction"] is None


def test_native_target_identical_to_qualified_training_transform_without_augmentation():
    config = SegmentationTransformConfig(
        horizontal_flip_probability=0., rotation_probability=0., rotation_degrees=(0., 0.),
        image_interpolation="bilinear", mask_interpolation="bilinear", image_fill=(0, 0, 0),
        mask_fill=0, normalization="imagenet", resize_geometry="stretch",
        operation_order="flip_rotate_resize")
    transform = PairedSegmentationTransform(config, base_seed=202503)
    rng = np.random.default_rng(1)
    target = Image.fromarray((rng.random((173, 269)) > .6).astype(np.uint8) * 255)
    image = Image.new("RGB", target.size)
    _, expected, _ = transform.transform_with_validity(
        image, target, Image.new("L", target.size, 255), 0)
    actual = resize_binary_target_to_native_logits(target)
    assert np.array_equal(expected[0].numpy().astype(bool), actual.numpy())


def test_frozen_localizers_top10_and_inputs_not_mutated():
    mask = np.zeros((240, 240), dtype=np.uint8)
    for x in range(12):
        mask[15:35, 2 + x * 20:12 + x * 20] = 1
    before = mask.copy()
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result = original_proposals(mask, image, *params())
    assert np.array_equal(mask, before)
    assert len(result["CURRENT_ALG1"]["before_cap_boxes"]) == 12
    assert len(result["CURRENT_ALG1"]["boxes"]) == 10
    assert result == original_proposals(mask, image, *params())
    assert all(len(r["boxes"]) <= 10 for r in result.values())
    assert all(0 <= b[0] < b[2] <= 240 and 0 <= b[1] < b[3] <= 240
               for r in result.values() for b in r["boxes"])


def test_native_padding_is_not_original_padding():
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[0:20, 0:20] = 1
    result = native_proposals(mask, (256, 512), params()[0])
    assert result["boxes"] == [[0, 0, 104, 52]]


def test_summary_one_to_one_vs_best_iou_and_top10_loss():
    # One crop can cover two duplicate diagnostic GT boxes, but matches only one.
    proposal = {"boxes": [[0, 0, 20, 20]],
                "before_cap_boxes": [[0, 0, 20, 20], [40, 40, 60, 60]],
                "raw_count": 2, "merges": 0, "multi": 0}
    records = [{"sample_id": "100", "conditions": {"GT128": {"method": proposal}}}]
    words = {"100": [{"bbox": b, "text": "unused"} for b in
                     [[0, 0, 20, 20], [0, 0, 20, 20], [40, 40, 60, 60]]]}
    summary, rows = summarize(records, words, ["method"], ["GT128"])
    value = summary["method"]["GT128"]
    assert value["geometry"]["gt_with_proposal_iou_ge_0_50"] == 2
    assert value["geometry"]["true_positive_proposals_iou_ge_0_50"] == 1
    assert value["before_cap_gt_iou_ge_0_50"] == 3
    assert value["gt_losses_due_to_top10_at_0_50"] == 1
    assert value["wordness"]["MULTIWORD_MERGE"] == 1
    assert len(rows) == 3


def test_success_transition_accounts_for_non_monotonic_effects():
    assert compare_success([.2, .6, .8, .1], [.7, .3, .6, .4]) == {
        "recovered": 1, "lost": 1, "both_success": 1, "both_failure": 1}
    with pytest.raises(ValueError):
        compare_success([], [0.])


def test_checked_mask_fails_closed_on_wrong_checkpoint_and_corruption(tmp_path):
    mask = np.zeros((128, 128), dtype=np.uint8)
    value = {**pack_mask(mask), "sample_id": "100", "image_sha256": "image"}
    value["segmentation_metadata"] = {"native_mask_shape": [128, 128], "threshold": .5,
        "threshold_comparison": "greater_than_or_equal", "probability_activation": "sigmoid",
        "checkpoint": {"sha256": "checkpoint"}, "native_mask_sha256": value["array_sha256"]}
    path = tmp_path / "mask.json"
    data = {"contract": "contract", "payload_sha256": digest(value), "value": value}
    path.write_text(json.dumps(data))
    assert np.array_equal(checked_native(path, "contract", "100", "image", "checkpoint")[0], mask)
    with pytest.raises(RuntimeError, match="checkpoint"):
        checked_native(path, "contract", "100", "image", "other")
    data["value"]["sample_id"] = "101"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="checksum"):
        checked_native(path, "contract", "100", "image", "checkpoint")


def test_declared_scope_and_frozen_rescue():
    config = json.loads(Path("configs/approximate/gt_mask_localization_oracle_train_v1.json").read_text())
    assert config["scope"] == {"split": "TRAIN", "images": 229, "gt_words": 849}
    assert config["rescue_parameters"] == {
        "max_normalized_gap": .4, "max_gap_to_internal_ratio": 1.5,
        "min_vertical_overlap": .5, "max_center_y_ratio": .5, "max_height_ratio": 2.5}
    assert not config["ocr_run"] and not config["tuning"]
