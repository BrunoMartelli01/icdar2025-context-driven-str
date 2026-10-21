from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


def _config(**overrides: object) -> SegmentationTransformConfig:
    values = {
        "horizontal_flip_probability": 0.5,
        "rotation_probability": 1.0,
        "rotation_degrees": (-12.0, 12.0),
        "image_interpolation": "bilinear",
        "mask_interpolation": "nearest",
        "image_fill": (0, 0, 0),
        "mask_fill": 0,
        "normalization": "none",
        "resize_geometry": "stretch",
        "operation_order": "flip_rotate_resize",
    }
    values.update(overrides)
    return SegmentationTransformConfig(**values)  # type: ignore[arg-type]


def _pair() -> tuple[Image.Image, Image.Image]:
    mask = np.zeros((80, 120), dtype=np.uint8)
    mask[20:55, 15:70] = 255
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    image[..., 0] = mask
    return Image.fromarray(image), Image.fromarray(mask)


def test_paper_sizes_binary_mask_and_explicit_metadata() -> None:
    transform = PairedSegmentationTransform(_config(), base_seed=41)
    image, mask = transform(*_pair(), 0)
    assert tuple(image.shape) == (3, 256, 256)
    assert tuple(mask.shape) == (1, 128, 128)
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})
    metadata = transform.metadata()
    assert metadata["paper_specified"]["image_size"] == [256, 256]  # type: ignore[index]
    assert metadata["approximate_choices"]["normalization"] == "none"  # type: ignore[index]


def test_same_seed_epoch_and_sample_are_deterministic() -> None:
    first = PairedSegmentationTransform(_config(), base_seed=73)
    second = PairedSegmentationTransform(_config(), base_seed=73)
    output_a = first(*_pair(), 5)
    output_b = second(*_pair(), 5)
    assert torch.equal(output_a[0], output_b[0])
    assert torch.equal(output_a[1], output_b[1])
    second.set_epoch(1)
    output_c = second(*_pair(), 5)
    assert not torch.equal(output_a[0], output_c[0])


def test_forced_flip_keeps_image_and_mask_geometry_aligned() -> None:
    transform = PairedSegmentationTransform(
        _config(
            horizontal_flip_probability=1.0,
            rotation_probability=0.0,
            rotation_degrees=(0.0, 0.0),
        ),
        base_seed=1,
    )
    image, mask = transform(*_pair(), 0)
    image_foreground = torch.nn.functional.interpolate(
        image[:1].unsqueeze(0), size=(128, 128), mode="nearest"
    ).squeeze(0) > 0.5
    disagreement = (image_foreground != mask.bool()).float().mean()
    assert disagreement < 0.02


def test_invalid_probabilities_sizes_and_ranges_fail() -> None:
    with pytest.raises(ValueError, match="horizontal_flip_probability"):
        _config(horizontal_flip_probability=1.1)
    with pytest.raises(ValueError, match="ordered"):
        _config(rotation_degrees=(5.0, -5.0))
    with pytest.raises(ValueError, match="Paper-specified"):
        _config(image_size=(128, 128))
