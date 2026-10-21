"""Opt-in native128 coordinate-frame variant of frozen repository contours."""
from __future__ import annotations

import numpy as np

from .repository_contours import RepositoryContourParams, audit_repository_contours


def remap_native_box(box, image_hw):
    """Exclusive xyxy: floor lower edges, ceil upper edges; no mask resize."""
    h, w = image_hw
    x1, y1, x2, y2 = box
    if h < 1 or w < 1 or not (0 <= x1 < x2 <= 128 and 0 <= y1 < y2 <= 128):
        raise ValueError("invalid original dimensions or native128 box")
    return [x1 * w // 128, y1 * h // 128,
            (x2 * w + 127) // 128, (y2 * h + 127) // 128]


def localize_native_repository_contours(mask, image_hw, params: RepositoryContourParams):
    """Filter/pad/sort/top10 entirely at 128, then map retained padded boxes.

    min_area measures unpadded bbox area in native pixels squared (upstream
    repository semantics), NOT foreground area. pad measures native pixels.
    """
    if mask.shape != (128, 128):
        raise ValueError("requires native128 mask; resizing is not performed")
    if len(image_hw) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in image_hw):
        raise ValueError("original image shape must contain two positive integers")
    native = audit_repository_contours(mask, params)
    native["coordinate_convention"] = "native_128_xyxy_exclusive"
    return {
        "variant": "repository-contours-native128-v1",
        "reproduction_mode": "approximate",
        "classification": "ASSUMPTION",
        "rationale": "User-requested native coordinate frame; not established by the paper",
        "mask_resize": "none",
        "area_unit": "native128_bbox_pixels_squared",
        "padding_unit": "native128_pixels",
        "native_localization": native,
        "original_image_hw": list(image_hw),
        "retained": [
            {"rank": row["rank"], "contour_index": row["contour_index"],
             "native_bbox": row["bbox"], "native_padded_bbox": row["padded_bbox"],
             "native_bbox_area": row["bbox_area"], "native_padded_bbox_area": row["padded_bbox_area"],
             "bbox_original": remap_native_box(row["bbox"], image_hw),
             "padded_bbox_original": remap_native_box(row["padded_bbox"], image_hw)}
            for row in native["retained"]
        ],
    }
