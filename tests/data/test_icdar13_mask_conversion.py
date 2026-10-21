from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch

from segmentation.data import (
    ICDAR13MaskConversionError,
    icdar13_task22_rgb_to_binary_mask,
    icdar13_task22_task1_boxes_to_target_and_validity,
)
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


def _convert(array: np.ndarray) -> np.ndarray:
    return np.asarray(icdar13_task22_rgb_to_binary_mask(Image.fromarray(array, mode="RGB")))


def _transform() -> PairedSegmentationTransform:
    return PairedSegmentationTransform(
        SegmentationTransformConfig(
            horizontal_flip_probability=0.0,
            rotation_probability=0.0,
            rotation_degrees=(0.0, 0.0),
            image_interpolation="bilinear",
            mask_interpolation="nearest",
            image_fill=(0, 0, 0),
            mask_fill=0,
            normalization="none",
            resize_geometry="stretch",
            operation_order="flip_rotate_resize",
        ),
        base_seed=202503,
    )


def test_pure_white_is_background() -> None:
    assert np.count_nonzero(_convert(np.full((3, 4, 3), 255, dtype=np.uint8))) == 0


@pytest.mark.parametrize("color", [(0, 0, 0), (255, 0, 0), (17, 63, 129), (254, 255, 255)])
def test_every_representative_non_white_rgb_is_foreground(color: tuple[int, int, int]) -> None:
    output = _convert(np.array([[color]], dtype=np.uint8))
    assert output.tolist() == [[255]]


def test_mixed_rgb_exact_conversion_is_binary_and_preserves_geometry() -> None:
    source = np.array(
        [
            [[255, 255, 255], [254, 255, 255], [255, 255, 254]],
            [[0, 0, 0], [255, 255, 255], [12, 34, 56]],
        ],
        dtype=np.uint8,
    )
    output = _convert(source)
    assert output.shape == source.shape[:2]
    assert output.tolist() == [[0, 255, 255], [255, 0, 255]]
    assert set(np.unique(output).tolist()) == {0, 255}


def test_rgb_behavior_is_unchanged() -> None:
    source = np.array([[[255, 255, 255], [254, 255, 255], [0, 0, 0]]], dtype=np.uint8)
    assert np.asarray(
        icdar13_task22_rgb_to_binary_mask(Image.fromarray(source, mode="RGB"))
    ).tolist() == [[0, 255, 255]]


def test_fully_opaque_rgba_matches_equivalent_rgb() -> None:
    rgb = np.array([[[255, 255, 255], [12, 34, 56]]], dtype=np.uint8)
    rgba = np.concatenate((rgb, np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)), axis=2)
    expected = icdar13_task22_rgb_to_binary_mask(Image.fromarray(rgb, mode="RGB"))
    actual = icdar13_task22_rgb_to_binary_mask(Image.fromarray(rgba, mode="RGBA"))
    assert np.array_equal(np.asarray(actual), np.asarray(expected))


def test_nonopaque_rgba_is_rejected() -> None:
    rgba = np.full((2, 2, 4), 255, dtype=np.uint8)
    rgba[1, 0, 3] = 254
    with pytest.raises(ICDAR13MaskConversionError, match="fully opaque"):
        icdar13_task22_rgb_to_binary_mask(Image.fromarray(rgba, mode="RGBA"))


def test_task1_boxes_ignore_only_complete_zero_overlap_colors() -> None:
    source = np.full((4, 6, 3), 255, dtype=np.uint8)
    source[1, 1] = (10, 20, 30)
    source[1, 4] = (10, 20, 30)
    source[3, 4] = (40, 50, 60)
    target, validity = icdar13_task22_task1_boxes_to_target_and_validity(
        Image.fromarray(source, mode="RGB"), '0,0,2,2,"care"\n'
    )
    assert np.asarray(target).tolist() == [
        [0, 0, 0, 0, 0, 0],
        [0, 255, 0, 0, 255, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 255, 0],
    ]
    assert np.asarray(validity)[1, 1] == 255
    assert np.asarray(validity)[1, 4] == 255
    assert np.asarray(validity)[3, 4] == 0
    assert np.count_nonzero(np.asarray(validity) == 0) == 1


def test_task1_box_outside_mask_fails_actionably() -> None:
    source = np.full((3, 4, 3), 255, dtype=np.uint8)
    with pytest.raises(ICDAR13MaskConversionError, match="outside mask geometry"):
        icdar13_task22_task1_boxes_to_target_and_validity(
            Image.fromarray(source, mode="RGB"), '0,0,4,2,"bad"\n'
        )


def test_nearest_resize_keeps_target_binary() -> None:
    source = np.full((7, 11, 3), 255, dtype=np.uint8)
    source[1:6, 3:9] = (254, 255, 255)
    mask = icdar13_task22_rgb_to_binary_mask(Image.fromarray(source, mode="RGB"))
    image = Image.new("RGB", (11, 7), (30, 40, 50))
    _, target = _transform()(image, mask, 0)
    assert tuple(target.shape) == (1, 128, 128)
    assert set(torch.unique(target).tolist()) == {0.0, 1.0}


@pytest.mark.parametrize("malformed, message", [
    (Image.new("L", (2, 2)), "mode RGB or RGBA"),
    (Image.new("CMYK", (2, 2)), "mode RGB or RGBA"),
    ("not-an-image", "must be a PIL image"),
])
def test_malformed_input_fails_actionably(malformed: object, message: str) -> None:
    with pytest.raises(ICDAR13MaskConversionError, match=message):
        icdar13_task22_rgb_to_binary_mask(malformed)  # type: ignore[arg-type]
