"""The curved cutout must isolate one mask component without moving its box."""

import numpy as np
import pytest

from localize.curved_component_crop import native_component_curved_crops
from localize.repository_contours import RepositoryContourParams


def test_curved_cutout_whitens_other_rgb_pixels_inside_tight_box():
    image = np.full((256, 256, 3), (25, 50, 75), dtype=np.uint8)
    native = np.zeros((128, 128), dtype=np.uint8)
    native[15:25, 10:30] = 1
    native[10:15, 10:15] = 1  # L-shaped component; native bbox area 300
    audit, proposals = native_component_curved_crops(
        native, image, RepositoryContourParams(150, 0, 10)
    )
    assert audit["raw_external_contours"] == 1
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.bbox_native == (10, 10, 30, 25)
    assert proposal.bbox_original == (20, 20, 60, 50)
    assert tuple(proposal.rectangular_rgb[0, -1]) == (25, 50, 75)
    assert tuple(proposal.curved_rgb[0, -1]) == (255, 255, 255)
    assert tuple(proposal.curved_rgb[-1, -1]) == (25, 50, 75)
    assert proposal.rectangular_rgb.shape == proposal.curved_rgb.shape


def test_second_component_inside_bbox_is_not_leaked_into_cutout():
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    native = np.zeros((128, 128), dtype=np.uint8)
    native[10:30, 10:16] = 1
    native[24:30, 10:30] = 1  # large L enclosing a disconnected small island
    native[14:18, 22:26] = 1
    audit, proposals = native_component_curved_crops(native, image, RepositoryContourParams(0, 0, 10))
    assert audit["raw_external_contours"] == 2
    large = next(item for item in proposals if item.component_pixels_native > 100)
    assert tuple(large.curved_rgb[10, 26]) == (255, 255, 255)


def test_zero_padding_is_enforced_and_border_clipped():
    image = np.full((257, 255, 3), 127, dtype=np.uint8)
    native = np.zeros((128, 128), dtype=np.uint8)
    native[0:10, 0:20] = 1
    _, proposals = native_component_curved_crops(native, image, RepositoryContourParams(150, 0, 10))
    assert proposals[0].bbox_original[:2] == (0, 0)
    with pytest.raises(ValueError, match="zero spatial padding"):
        native_component_curved_crops(native, image, RepositoryContourParams(150, 6, 10))
