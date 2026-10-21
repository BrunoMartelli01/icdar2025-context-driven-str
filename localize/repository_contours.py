"""Frozen author-repository localization, NOT the paper-component-area path.

This opt-in approximate variant follows upstream rectangle-area semantics.
Input must already be a binary mask in original-image coordinates. No resize,
grouping, morphology, threshold selection, OCR, or GT is performed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


SOURCE_COMMIT = "363cdeec0eda8999bf10d14c8f4ca6036dd422a2"
SOURCE_BLOB = "d9359417120c09a9730ddc529c59caa4c0690bb0"
SOURCE_SHA256 = "3e0d779ae761e038f48525c4a169e3e3be7e1849ddf85c73f309ce3d0cad7dbc"


@dataclass(frozen=True)
class RepositoryContourParams:
    """Explicit repository parameters; unlike upstream, invalid values fail early."""

    min_area: int
    pad: int
    nmax: int

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.min_area < 0 or self.pad < 0 or self.nmax < 1:
            raise ValueError("require min_area >= 0, pad >= 0, nmax >= 1")


def audit_repository_contours(
    mask: np.ndarray, params: RepositoryContourParams
) -> dict:
    """Return boxes and rejection provenance using the frozen upstream algorithm.

    Sorting is stable in OpenCV contour-return order, including exact area ties.
    RETR_EXTERNAL can omit foreground islands nested within an external contour.
    Neither bbox area nor contour polygon area is a true foreground-pixel count.
    """

    if mask.ndim != 2 or min(mask.shape) < 1:
        raise ValueError("mask must be a nonempty single-channel HxW array")
    if not np.isfinite(mask).all():
        raise ValueError("mask must contain finite values")
    if not np.isin(mask, (0, 1, 255)).all():
        raise ValueError("expected binary mask, not probabilities/logits")
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    accepted, rejected = [], []
    for contour_index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        item = {
            "contour_index": contour_index,
            "bbox": [x, y, x + w, y + h],
            "bbox_area": w * h,
        }
        if params.min_area > 0 and w * h < params.min_area:
            rejected.append({**item, "reason": "below_min_bbox_area"})
            continue
        padded = [max(0, x - params.pad), max(0, y - params.pad),
                  min(width, x + w + params.pad), min(height, y + h + params.pad)]
        item.update(padded_bbox=padded,
                    padded_bbox_area=(padded[2] - padded[0]) * (padded[3] - padded[1]))
        accepted.append(item)
    accepted.sort(key=lambda item: item["padded_bbox_area"], reverse=True)
    retained = [dict(item, rank=i + 1) for i, item in enumerate(accepted[:params.nmax])]
    rejected.extend(dict(item, reason="top_n_limit") for item in accepted[params.nmax:])
    return {
        "schema_version": 1,
        "variant": "repository-contours-v1",
        "reproduction_mode": "approximate",
        "result_class": "approximate reproduction",
        "classification": "REPOSITORY_SPECIFIED",
        "source_commit": SOURCE_COMMIT,
        "source_blob": SOURCE_BLOB,
        "source_sha256": SOURCE_SHA256,
        "opencv_version": cv2.__version__,
        "coordinate_convention": "original_image_xyxy_exclusive",
        "parameters": asdict(params),
        "mask_shape": list(binary.shape),
        "retrieval": "RETR_EXTERNAL",
        "approximation": "CHAIN_APPROX_SIMPLE",
        "filter_area": "none" if params.min_area == 0 else "unpadded_bbox_area",
        "sort_area": "clipped_padded_bbox_area",
        "tie_break": "stable_opencv_contour_order",
        "raw_external_contours": len(contours),
        "candidates_before_cap": len(accepted),
        "retained": retained,
        "rejected": rejected,
    }


def mask_to_blocks(
    mask: np.ndarray, params: RepositoryContourParams
) -> list[tuple[int, int, int, int]]:
    """Upstream-style bbox-only API; equivalent on validated binary inputs."""

    return [tuple(item["padded_bbox"]) for item in
            audit_repository_contours(mask, params)["retained"]]


def crop_repository_blocks(
    mask: np.ndarray, image: np.ndarray, params: RepositoryContourParams
) -> tuple[dict, list[np.ndarray]]:
    """Crop the original image without resizing or changing the default pipeline."""

    if image.ndim not in (2, 3) or image.shape[:2] != mask.shape:
        raise ValueError("mask and original image must have identical spatial dimensions")
    report = audit_repository_contours(mask, params)
    crops = [image[y1:y2, x1:x2].copy() for x1, y1, x2, y2 in
             (item["padded_bbox"] for item in report["retained"])]
    return report, crops
