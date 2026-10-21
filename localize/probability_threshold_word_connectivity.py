"""Native-CC implementation of frozen paper-like Algorithm 1 post-processing.

The binary mask is labelled at its native 128x128 resolution.  Component area
and bbox are then mapped exactly as PyTorch nearest-neighbour backprojection
would map the component pixels.  No morphology or component grouping occurs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK


NATIVE_SHAPE = (128, 128)


@dataclass(frozen=True)
class NativeComponent:
    component_id: int
    native_area: int
    mapped_area: int
    bbox_native: tuple[int, int, int, int]
    bbox_original: tuple[int, int, int, int]
    padded_bbox_original: tuple[int, int, int, int]


@dataclass(frozen=True)
class NativeComponentLocalization:
    labels_native: np.ndarray
    raw_components: tuple[NativeComponent, ...]
    candidates_before_cap: tuple[NativeComponent, ...]
    retained: tuple[NativeComponent, ...]


def nearest_axis_cells(source_size: int, output_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return exact half-open output intervals produced by torch nearest resize."""
    if source_size < 1 or output_size < 1:
        raise ValueError("nearest-axis mapping requires positive dimensions")
    source = torch.arange(source_size, dtype=torch.float32).reshape(1, 1, 1, source_size)
    mapped = F.interpolate(source, size=(1, output_size), mode="nearest")[0, 0, 0]
    indices = mapped.to(torch.int64).numpy()
    starts = np.full(source_size, -1, dtype=np.int64)
    ends = np.full(source_size, -1, dtype=np.int64)
    for index in range(source_size):
        positions = np.flatnonzero(indices == index)
        if len(positions):
            starts[index] = positions[0]
            ends[index] = positions[-1] + 1
    return starts, ends


def component_ids_intersecting_original_box(
    labels_native: np.ndarray,
    box: Sequence[int],
    *,
    x_cells: tuple[np.ndarray, np.ndarray],
    y_cells: tuple[np.ndarray, np.ndarray],
) -> tuple[int, ...]:
    """IDs with at least one backprojected foreground pixel inside ``box``."""
    if labels_native.shape != NATIVE_SHAPE:
        raise ValueError("native labels must be 128x128")
    x1, y1, x2, y2 = (int(value) for value in box)
    x_starts, x_ends = x_cells
    y_starts, y_ends = y_cells
    xs = np.flatnonzero((x_starts >= 0) & (x_ends > x1) & (x_starts < x2))
    ys = np.flatnonzero((y_starts >= 0) & (y_ends > y1) & (y_starts < y2))
    if not len(xs) or not len(ys):
        return ()
    return tuple(sorted(int(value) for value in np.unique(labels_native[np.ix_(ys, xs)]) if value))


def localize_native_components(
    mask_native: np.ndarray,
    *,
    output_width: int,
    output_height: int,
    min_area: int = 150,
    pad: int = 6,
    connectivity: int = 8,
    nmax: int = 10,
) -> NativeComponentLocalization:
    """Apply frozen area/filter/sort/cap semantics without grouping components."""
    if mask_native.shape != NATIVE_SHAPE or mask_native.dtype != np.uint8:
        raise ValueError("native mask must be uint8 128x128")
    if not np.isin(mask_native, (0, 1)).all():
        raise ValueError("native mask must be binary")
    if output_width < 1 or output_height < 1:
        raise ValueError("original image dimensions must be positive")
    if min_area < 1 or pad < 0 or connectivity not in (4, 8) or nmax != 10:
        raise ValueError("invalid frozen Algorithm1 post-processing parameters")

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_native, connectivity)
    x_starts, x_ends = nearest_axis_cells(128, output_width)
    y_starts, y_ends = nearest_axis_cells(128, output_height)
    x_sizes = np.maximum(0, x_ends - x_starts)
    y_sizes = np.maximum(0, y_ends - y_starts)
    weights = y_sizes[:, None] * x_sizes[None, :]
    mapped_areas = np.bincount(
        labels.ravel(), weights=weights.ravel(), minlength=count
    ).round().astype(np.int64)
    components = []
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        bbox_native = (x, y, x + width, y + height)
        component_pixels = labels == component_id
        surviving_pixels = (
            component_pixels
            & (y_sizes[:, None] > 0)
            & (x_sizes[None, :] > 0)
        )
        occupied_x = np.flatnonzero(surviving_pixels.any(axis=0))
        occupied_y = np.flatnonzero(surviving_pixels.any(axis=1))
        if len(occupied_x) and len(occupied_y):
            bbox_original = (
                int(x_starts[occupied_x[0]]), int(y_starts[occupied_y[0]]),
                int(x_ends[occupied_x[-1]]), int(y_ends[occupied_y[-1]]),
            )
        else:
            # The component is genuinely absent after nearest downsampling in
            # one original-image axis.  Its mapped area is zero, so Amin will
            # reject it; the empty box is retained only for raw-topology audit.
            bbox_original = (0, 0, 0, 0)
        x1, y1, x2, y2 = bbox_original
        components.append(NativeComponent(
            component_id=component_id,
            native_area=int(stats[component_id, cv2.CC_STAT_AREA]),
            mapped_area=int(mapped_areas[component_id]),
            bbox_native=bbox_native,
            bbox_original=bbox_original,
            padded_bbox_original=(
                max(0, x1 - pad), max(0, y1 - pad),
                min(output_width, x2 + pad), min(output_height, y2 + pad),
            ),
        ))
    candidates = [value for value in components if value.mapped_area >= min_area]
    candidates.sort(key=lambda value: (
        -value.mapped_area, value.bbox_original[1], value.bbox_original[0], value.component_id,
    ))
    return NativeComponentLocalization(
        labels_native=np.ascontiguousarray(labels, dtype=np.int32),
        raw_components=tuple(components),
        candidates_before_cap=tuple(candidates),
        retained=tuple(candidates[:nmax]),
    )


def algorithm1_contract() -> dict[str, object]:
    return {
        "connected_components_resolution": [128, 128],
        "connectivity": 8,
        "component_area": "foreground pixels after exact nearest backprojection",
        "min_area": 150,
        "bbox": "native component support remapped exactly to original image",
        "padding_original_pixels": 6,
        "sort": "mapped true component area descending",
        "tie_break": APPROXIMATE_TIE_BREAK,
        "nmax": 10,
        "grouping": False,
        "morphology": False,
    }


__all__ = [
    "NATIVE_SHAPE", "NativeComponent", "NativeComponentLocalization",
    "algorithm1_contract", "component_ids_intersecting_original_box",
    "localize_native_components", "nearest_axis_cells",
]
