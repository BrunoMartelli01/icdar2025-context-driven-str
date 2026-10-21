"""Algorithm 1 localization in the native 128x128 segmentation frame.

The paper specifies connected regions, true component-area filtering, padded
boxes, descending component-area order, at most ten crops, and cropping from
the original RGB image.  Numeric area/padding, connectivity, tie handling and
the native-coordinate interpretation remain explicit approximate choices.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from localize.mask_to_blocks import BlockParam, localize_mask
from localize.native_repository_contours import remap_native_box


@dataclass(frozen=True)
class NativeAlgorithm1Block:
    rank: int
    component_id: int
    component_area_native: int
    bbox_native: tuple[int, int, int, int]
    padded_bbox_native: tuple[int, int, int, int]
    bbox_original: tuple[int, int, int, int]
    padded_bbox_original: tuple[int, int, int, int]
    crop: np.ndarray


@dataclass(frozen=True)
class NativeAlgorithm1Result:
    parameters: BlockParam
    retained: tuple[NativeAlgorithm1Block, ...]
    audit: dict[str, object]


def localize_native_algorithm1(
    native_mask: np.ndarray,
    original_image: np.ndarray,
    params: BlockParam,
) -> NativeAlgorithm1Result:
    """Apply Algorithm 1 at 128x128 and crop mapped boxes from original RGB."""

    if native_mask.shape != (128, 128):
        raise ValueError("native Algorithm 1 requires an exactly 128x128 mask")
    if original_image.ndim not in (2, 3) or min(original_image.shape[:2]) < 1:
        raise ValueError("original image must be a nonempty HxW or HxWxC array")
    native_image = np.zeros((128, 128), dtype=np.uint8)
    localized = localize_mask(native_mask, native_image, params)
    image_hw = original_image.shape[:2]
    retained = []
    for block in localized.retained:
        bbox_original = tuple(remap_native_box(block.bbox, image_hw))
        padded_original = tuple(remap_native_box(block.padded_bbox, image_hw))
        x1, y1, x2, y2 = padded_original
        retained.append(
            NativeAlgorithm1Block(
                rank=block.rank,
                component_id=block.component_id,
                component_area_native=block.component_area,
                bbox_native=block.bbox,
                padded_bbox_native=block.padded_bbox,
                bbox_original=bbox_original,
                padded_bbox_original=padded_original,
                crop=original_image[y1:y2, x1:x2].copy(),
            )
        )
    audit = localized.as_dict()
    audit.update({
        "variant": "paper-algorithm1-native128-v1",
        "classification": "APPROXIMATE_PAPER_ALGORITHM1",
        "native_mask_shape": [128, 128],
        "original_image_shape": list(original_image.shape),
        "area_unit": "native_128_foreground_pixels",
        "padding_unit": "native_128_pixels",
        "bbox_mapping": "floor lower edges; ceil upper edges; xyxy exclusive",
        "final_crop_source": "original_rgb",
    })
    return NativeAlgorithm1Result(params, tuple(retained), audit)


__all__ = [
    "NativeAlgorithm1Block",
    "NativeAlgorithm1Result",
    "localize_native_algorithm1",
]
