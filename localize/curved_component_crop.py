"""Isolate a native-mask component's curved RGB support on a white canvas.

PRACTICAL_APPROXIMATION: this is a separate diagnostic, not paper Algorithm 1.
All proposal decisions use the input mask alone; the source RGB is only cropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .native_repository_contours import remap_native_box
from .repository_contours import RepositoryContourParams, audit_repository_contours


@dataclass(frozen=True)
class CurvedComponentProposal:
    rank: int
    contour_index: int
    bbox_native: tuple[int, int, int, int]
    bbox_original: tuple[int, int, int, int]
    bbox_area_native: int
    component_pixels_native: int
    foreground_fraction_in_crop: float
    rectangular_rgb: np.ndarray
    curved_rgb: np.ndarray


def native_component_curved_crops(
    native_mask: np.ndarray,
    image_rgb: np.ndarray,
    params: RepositoryContourParams,
) -> tuple[dict, tuple[CurvedComponentProposal, ...]]:
    """Return frozen native contour proposals plus shape-aware RGB cutouts.

    No spatial padding is allowed. Bilinear mask enlargement followed by a
    fixed >=0.5 cut preserves a soft-scale contour boundary; it is used only
    to select original RGB pixels, not to form, rank, or merge proposals.
    """
    if native_mask.shape != (128, 128) or image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
        raise ValueError("Expected native 128x128 mask and uint8 RGB image")
    if params.pad != 0:
        raise ValueError("Curved-component crop protocol requires zero spatial padding")
    if not np.isin(native_mask, (0, 1, 255)).all():
        raise ValueError("Expected a binary native mask")
    binary = np.ascontiguousarray((native_mask > 0).astype(np.uint8))
    audit = audit_repository_contours(binary, params)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != audit["raw_external_contours"]:
        raise RuntimeError("Contour enumeration differs from frozen localizer")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count < 1:
        raise RuntimeError("Invalid connected-component label count")
    height, width = image_rgb.shape[:2]
    proposals = []
    for item in audit["retained"]:
        contour = contours[item["contour_index"]]
        x, y = (int(value) for value in contour[0, 0])
        label = int(labels[y, x])
        if label <= 0:
            raise RuntimeError("Contour seed does not map to foreground")
        sx, sy, sw, sh, pixels = (int(value) for value in stats[label])
        bbox_native = tuple(int(value) for value in item["bbox"])
        if bbox_native != (sx, sy, sx + sw, sy + sh):
            raise RuntimeError("Contour and connected-component boxes disagree")
        bbox_original = tuple(remap_native_box(bbox_native, (height, width)))
        x1, y1, x2, y2 = bbox_original
        rectangular = np.ascontiguousarray(image_rgb[y1:y2, x1:x2])
        if rectangular.size == 0:
            raise RuntimeError("Empty remapped crop")
        # The single labelled component is enlarged independently: adjacent
        # words/components inside this bounding rectangle cannot leak in.
        enlarged = cv2.resize(
            (labels == label).astype(np.float32), (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        support = enlarged[y1:y2, x1:x2] >= 0.5
        if not support.any():
            raise RuntimeError("Component vanished while remapping to RGB")
        curved = rectangular.copy()
        curved[~support] = 255
        proposals.append(CurvedComponentProposal(
            rank=int(item["rank"]), contour_index=int(item["contour_index"]),
            bbox_native=bbox_native, bbox_original=bbox_original,
            bbox_area_native=int(item["bbox_area"]), component_pixels_native=pixels,
            foreground_fraction_in_crop=float(support.mean()),
            rectangular_rgb=rectangular, curved_rgb=curved,
        ))
    return audit, tuple(proposals)


__all__ = ["CurvedComponentProposal", "native_component_curved_crops"]
