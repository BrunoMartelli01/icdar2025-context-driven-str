from __future__ import annotations

import numpy as np
from PIL import Image

from scripts.evaluate_totaltext_highres_contour_padding import (
    _config, _input_tensor, contour_proposals, correct_gt_indices,
    crop_array_hash, padding_overlap, remap_native256_box,
)
from scripts.train_totaltext_v4_highres_ablation import _load_config, _transform


def test_predeclared_grid_and_native_remap() -> None:
    assert _config()["padding_native_pixels"] == [0, 2, 4, 6, 8, 12]
    assert remap_native256_box([0, 0, 1, 1], (333, 511)) == [0, 0, 2, 2]
    assert remap_native256_box([255, 255, 256, 256], (333, 511)) == [509, 331, 511, 333]


def test_contours_have_no_area_filter_or_top10_cap() -> None:
    mask = np.zeros((256, 256), dtype=np.uint8)
    for index in range(12):
        mask[20, 10 + index * 10] = 1
    data = contour_proposals(mask, (512, 768), [0, 2])
    assert data["0"]["raw_contours"] == 12
    assert len(data["0"]["proposals"]) == 12
    assert len(data["2"]["proposals"]) == 12
    assert all(item["bbox_native"] == item["padded_bbox_native"] for item in data["0"]["proposals"])
    first_zero = {item["contour_index"]: item for item in data["0"]["proposals"]}
    first_two = {item["contour_index"]: item for item in data["2"]["proposals"]}
    assert set(first_zero) == set(first_two)
    assert all(first_zero[key]["bbox_native"] == first_two[key]["bbox_native"] for key in first_zero)
    assert any(first_zero[key]["bbox"] != first_two[key]["bbox"] for key in first_zero)


def test_preprocessing_matches_frozen_validation_transform() -> None:
    rgb = np.arange(32 * 45 * 3, dtype=np.uint8).reshape(32, 45, 3)
    mask = Image.new("L", (45, 32), 0)
    expected, _ = _transform(_load_config(), augment=False, seed=202503)(Image.fromarray(rgb), mask, 0)
    actual = _input_tensor(rgb)
    assert actual.shape == (1, 3, 512, 512)
    assert np.array_equal(actual[0].numpy(), expected.numpy())


def test_crop_hash_contains_shape_and_pixels() -> None:
    first = np.zeros((2, 6, 3), dtype=np.uint8)
    second = np.zeros((3, 4, 3), dtype=np.uint8)
    assert crop_array_hash(first) != crop_array_hash(second)
    first[0, 0, 0] = 1
    assert crop_array_hash(first) != crop_array_hash(np.zeros_like(first))


def test_exact_word_ids_require_spatial_match_and_one_to_one() -> None:
    gt = [{"bbox": [0, 0, 20, 20], "text": "Book"},
          {"bbox": [30, 0, 50, 20], "text": "Book"}]
    preds = [{"bbox": [0, 0, 20, 20], "text": "book"},
             {"bbox": [0, 0, 20, 20], "text": "BOOK"},
             {"bbox": [30, 0, 50, 20], "text": "Wrong"}]
    assert correct_gt_indices(preds, gt) == {0}
    assert correct_gt_indices(preds, gt, casefold=False) == set()


def test_padding_intersection_counts_complementarity() -> None:
    correct = {"0": {"a", "b"}, "2": {"b", "c"}, "4": {"b"}}
    values = padding_overlap(correct, [0, 2, 4])
    assert values["correct_with_any_padding"] == 3
    assert values["gain_any_padding_vs_best_single"] == 1
    assert values["frequency_across_paddings"] == {"1": 2, "2": 0, "3": 1}
    assert values["unique_to_one_padding"] == {"0": 1, "2": 1, "4": 0}
    assert values["intersection_matrix"]["0"]["2"] == 1
