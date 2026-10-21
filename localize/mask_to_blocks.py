"""Paper-specified mechanics of Algorithm 1 with explicit approximate parameters.

The paper specifies component-area filtering/sorting, padded boxes, original-image
crops, and ``nmax=10``. It does not specify numeric ``Amin``/``p``, connectivity,
or tie-breaking. This implementation therefore requires all four unresolved
choices from the caller and exposes them in every auditable result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np


ALGORITHM1_NMAX: Final[int] = 10
APPROXIMATE_TIE_BREAK: Final[str] = "top_to_bottom_then_left_to_right"


@dataclass(frozen=True)
class BlockParam:
    """Algorithm 1 inputs, including every unresolved localization choice."""

    min_area: int
    pad: int
    connectivity: int
    tie_break: str
    nmax: int = ALGORITHM1_NMAX

    def __post_init__(self) -> None:
        if self.min_area < 1:
            raise ValueError("Amin/min_area must be at least one foreground pixel")
        if self.pad < 0:
            raise ValueError("p/pad must be non-negative")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be explicitly set to 4 or 8")
        if self.tie_break != APPROXIMATE_TIE_BREAK:
            raise ValueError(
                "unsupported equal-area tie policy; expected "
                f"{APPROXIMATE_TIE_BREAK!r}"
            )
        if self.nmax != ALGORITHM1_NMAX:
            raise ValueError("Algorithm 1 fixes nmax=10")


@dataclass(frozen=True)
class LocalizedBlock:
    """One retained component and its crop, ordered by Algorithm 1 rank."""

    rank: int
    component_id: int
    component_area: int
    bbox: tuple[int, int, int, int]
    padded_bbox: tuple[int, int, int, int]
    crop: np.ndarray
    connectivity: int
    tie_break: str


@dataclass(frozen=True)
class RejectedComponent:
    """A component excluded by the area filter or the paper's top-10 cap."""

    component_id: int
    component_area: int
    bbox: tuple[int, int, int, int]
    padded_bbox: tuple[int, int, int, int] | None
    reason: str


@dataclass(frozen=True)
class LocalizationResult:
    """Auditable localization result retaining accepted and rejected components."""

    params: BlockParam
    retained: tuple[LocalizedBlock, ...]
    rejected: tuple[RejectedComponent, ...]

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-compatible manifest using exclusive ``xyxy`` boxes."""

        return {
            "schema_version": 1,
            "coordinate_convention": "xyxy_exclusive",
            "operation_order": [
                "extract_components",
                "filter_by_true_component_area",
                "compute_and_pad_bboxes",
                "sort_by_true_component_area_descending",
                "limit_to_top_10",
                "crop_original_image",
            ],
            "parameters": {
                "min_area": self.params.min_area,
                "pad": self.params.pad,
                "connectivity": self.params.connectivity,
                "tie_break": self.params.tie_break,
                "nmax": self.params.nmax,
            },
            "retained": [
                {
                    "rank": block.rank,
                    "component_id": block.component_id,
                    "component_area": block.component_area,
                    "bbox": list(block.bbox),
                    "padded_bbox": list(block.padded_bbox),
                    "crop_shape": list(block.crop.shape),
                }
                for block in self.retained
            ],
            "rejected": [
                {
                    "component_id": component.component_id,
                    "component_area": component.component_area,
                    "bbox": list(component.bbox),
                    "padded_bbox": (
                        list(component.padded_bbox)
                        if component.padded_bbox is not None
                        else None
                    ),
                    "reason": component.reason,
                }
                for component in self.rejected
            ],
        }


def _validate_inputs(mask: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, int, int]:
    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel HxW array")
    if image.ndim not in (2, 3):
        raise ValueError("image must be HxW or HxWxC")
    height, width = mask.shape
    if image.shape[:2] != (height, width):
        raise ValueError("mask and original image must have identical spatial dimensions")
    return (mask > 0).astype(np.uint8), height, width


def localize_mask(
    mask: np.ndarray,
    image: np.ndarray,
    params: BlockParam,
) -> LocalizationResult:
    """Localize text components and retain a complete Algorithm 1 audit trail.

    Bounding boxes use NumPy's exclusive ``xmax``/``ymax`` convention. Component
    area is the connected foreground-pixel count, never bounding-box area.
    Connectivity and the deterministic spatial tie policy are required explicit
    inputs because the paper does not specify either choice.
    """

    binary, height, width = _validate_inputs(mask, image)
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=params.connectivity
    )
    candidates: list[
        tuple[int, int, tuple[int, int, int, int], tuple[int, int, int, int]]
    ] = []
    rejected: list[RejectedComponent] = []
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        box_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        box_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component_area = int(stats[component_id, cv2.CC_STAT_AREA])
        bbox = (x, y, x + box_width, y + box_height)
        if component_area < params.min_area:
            rejected.append(
                RejectedComponent(
                    component_id=component_id,
                    component_area=component_area,
                    bbox=bbox,
                    padded_bbox=None,
                    reason="below_min_area",
                )
            )
            continue
        x1, y1, x2, y2 = bbox
        padded_bbox = (
            max(0, x1 - params.pad),
            max(0, y1 - params.pad),
            min(width, x2 + params.pad),
            min(height, y2 + params.pad),
        )
        candidates.append((component_id, component_area, bbox, padded_bbox))

    # Algorithm 1 specifies descending component area. The y/x keys implement
    # the documented approximate spatial tie rule; component_id is a final
    # total-order fallback consistent with the connected-component labels.
    candidates.sort(key=lambda item: (-item[1], item[2][1], item[2][0], item[0]))
    blocks: list[LocalizedBlock] = []
    for rank, (component_id, component_area, bbox, padded_bbox) in enumerate(
        candidates[: params.nmax], start=1
    ):
        px1, py1, px2, py2 = padded_bbox
        blocks.append(
            LocalizedBlock(
                rank=rank,
                component_id=component_id,
                component_area=component_area,
                bbox=bbox,
                padded_bbox=padded_bbox,
                crop=image[py1:py2, px1:px2].copy(),
                connectivity=params.connectivity,
                tie_break=params.tie_break,
            )
        )
    rejected.extend(
        RejectedComponent(
            component_id=component_id,
            component_area=component_area,
            bbox=bbox,
            padded_bbox=padded_bbox,
            reason="top_10_limit",
        )
        for component_id, component_area, bbox, padded_bbox in candidates[params.nmax :]
    )
    return LocalizationResult(params, tuple(blocks), tuple(rejected))


def mask_to_blocks(
    mask: np.ndarray,
    image: np.ndarray,
    params: BlockParam,
) -> list[LocalizedBlock]:
    """Compatibility API returning only retained Algorithm 1 crops."""

    return list(localize_mask(mask, image, params).retained)


__all__ = [
    "ALGORITHM1_NMAX",
    "APPROXIMATE_TIE_BREAK",
    "BlockParam",
    "LocalizationResult",
    "LocalizedBlock",
    "RejectedComponent",
    "localize_mask",
    "mask_to_blocks",
]
