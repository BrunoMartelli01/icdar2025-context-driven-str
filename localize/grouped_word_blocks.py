"""Practical graph-based remapping of glyph-like segmentation components.

``practical-remap-v1`` is intentionally separate from the paper-faithful
``localize.mask_to_blocks`` Algorithm 1 implementation.  It groups raw
connected components by declared geometric compatibility before cropping; it is
a PRACTICAL_APPROXIMATION and must never be selected by strict/paper-faithful
configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from .mask_to_blocks import ALGORITHM1_NMAX, APPROXIMATE_TIE_BREAK


PRACTICAL_REMAP_V1 = "practical-remap-v1"


@dataclass(frozen=True)
class GroupedWordBlockParams:
    """Fully explicit graph edge and post-group crop parameters."""

    horizontal_gap_ratio: float
    vertical_overlap_min: float
    max_center_y_ratio: float
    max_height_ratio: float
    pad: int
    connectivity: int = 8
    tie_break: str = APPROXIMATE_TIE_BREAK
    nmax: int = ALGORITHM1_NMAX

    def __post_init__(self) -> None:
        for name in (
            "horizontal_gap_ratio", "vertical_overlap_min", "max_center_y_ratio"
        ):
            if getattr(self, name) < 0:
                raise ValueError("{} must be non-negative".format(name))
        if self.max_height_ratio < 1:
            raise ValueError("max_height_ratio must be at least one")
        if self.pad < 0:
            raise ValueError("pad must be non-negative")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")
        if self.tie_break != APPROXIMATE_TIE_BREAK:
            raise ValueError("unsupported tie policy")
        if self.nmax != ALGORITHM1_NMAX:
            raise ValueError("practical-remap-v1 fixes nmax=10")


@dataclass(frozen=True)
class RawComponent:
    component_id: int
    foreground_area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


@dataclass(frozen=True)
class GroupedWordBlock:
    rank: int
    raw_component_ids: tuple[int, ...]
    foreground_area: int
    bbox: tuple[int, int, int, int]
    padded_bbox: tuple[int, int, int, int]
    crop: np.ndarray


@dataclass(frozen=True)
class GroupedWordBlocksResult:
    params: GroupedWordBlockParams
    raw_components: tuple[RawComponent, ...]
    graph_edges: tuple[dict[str, float | int], ...]
    groups_before_cap: tuple[dict[str, object], ...]
    retained: tuple[GroupedWordBlock, ...]
    rejected_group_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": PRACTICAL_REMAP_V1,
            "classification": "PRACTICAL_APPROXIMATION",
            "coordinate_convention": "xyxy_exclusive",
            "operation_order": [
                "extract_raw_connected_components",
                "construct_geometric_graph",
                "connected_components_of_graph",
                "union_raw_component_bboxes",
                "pad_and_clip_group_bbox",
                "sort_by_group_foreground_area_descending",
                "limit_to_top_10",
                "crop_original_image",
            ],
            "parameters": {
                "horizontal_gap_ratio": self.params.horizontal_gap_ratio,
                "vertical_overlap_min": self.params.vertical_overlap_min,
                "max_center_y_ratio": self.params.max_center_y_ratio,
                "max_height_ratio": self.params.max_height_ratio,
                "pad": self.params.pad,
                "connectivity": self.params.connectivity,
                "tie_break": self.params.tie_break,
                "nmax": self.params.nmax,
            },
            "raw_components": [
                {
                    "component_id": item.component_id,
                    "foreground_area": item.foreground_area,
                    "bbox": list(item.bbox),
                    "centroid": list(item.centroid),
                }
                for item in self.raw_components
            ],
            "graph_edges": [dict(item) for item in self.graph_edges],
            "groups_before_cap": [dict(item) for item in self.groups_before_cap],
            "retained": [
                {
                    "rank": item.rank,
                    "raw_component_ids": list(item.raw_component_ids),
                    "foreground_area": item.foreground_area,
                    "bbox": list(item.bbox),
                    "padded_bbox": list(item.padded_bbox),
                    "crop_shape": list(item.crop.shape),
                }
                for item in self.retained
            ],
            "rejected_group_ids": list(self.rejected_group_ids),
        }


def _validate(mask: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, int, int]:
    if mask.ndim != 2 or image.ndim not in (2, 3) or mask.shape != image.shape[:2]:
        raise ValueError("mask and image must have identical HxW geometry")
    binary = (mask > 0).astype(np.uint8)
    height, width = binary.shape
    return binary, height, width


def _extract(binary: np.ndarray, connectivity: int) -> tuple[RawComponent, ...]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity)
    rows = []
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        rows.append(
            RawComponent(
                component_id=component_id,
                foreground_area=int(stats[component_id, cv2.CC_STAT_AREA]),
                bbox=(x, y, x + width, y + height),
                centroid=(float(centroids[component_id][0]), float(centroids[component_id][1])),
            )
        )
    return tuple(rows)


def component_pair_metrics(left: RawComponent, right: RawComponent) -> dict[str, float]:
    """Metrics for a potential same-word edge, using mean box height as reference."""

    lx1, ly1, lx2, ly2 = left.bbox
    rx1, ry1, rx2, ry2 = right.bbox
    left_height, right_height = ly2 - ly1, ry2 - ry1
    reference_height = (left_height + right_height) / 2.0
    horizontal_gap = max(0, max(lx1, rx1) - min(lx2, rx2))
    vertical_overlap = max(0, min(ly2, ry2) - max(ly1, ry1))
    min_height = min(left_height, right_height)
    return {
        "horizontal_gap": float(horizontal_gap),
        "reference_height": float(reference_height),
        "horizontal_gap_over_reference_height": horizontal_gap / reference_height,
        "vertical_overlap_over_min_height": vertical_overlap / min_height,
        "center_y_distance_over_reference_height": abs(left.centroid[1] - right.centroid[1]) / reference_height,
        "height_ratio": max(left_height, right_height) / min_height,
    }


def compatible_components(
    left: RawComponent, right: RawComponent, params: GroupedWordBlockParams
) -> dict[str, float | int] | None:
    metrics = component_pair_metrics(left, right)
    if metrics["horizontal_gap_over_reference_height"] > params.horizontal_gap_ratio:
        return None
    # Vertical compatibility is intentionally conjunctive: large-gap/nearby-line
    # components cannot become a word group merely because their heights match.
    if metrics["vertical_overlap_over_min_height"] < params.vertical_overlap_min:
        return None
    if metrics["center_y_distance_over_reference_height"] > params.max_center_y_ratio:
        return None
    if metrics["height_ratio"] > params.max_height_ratio:
        return None
    return {
        "left_component_id": left.component_id,
        "right_component_id": right.component_id,
        **metrics,
    }


def _graph_groups(
    components: tuple[RawComponent, ...], edges: list[dict[str, float | int]]
) -> list[tuple[int, ...]]:
    parent = {component.component_id: component.component_id for component in components}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for edge in edges:
        union(int(edge["left_component_id"]), int(edge["right_component_id"]))
    grouped: dict[int, list[int]] = {}
    for component in components:
        grouped.setdefault(find(component.component_id), []).append(component.component_id)
    return [tuple(sorted(ids)) for _, ids in sorted(grouped.items())]


def group_word_blocks(
    mask: np.ndarray, image: np.ndarray, params: GroupedWordBlockParams
) -> GroupedWordBlocksResult:
    """Group raw components using only geometry, then retain at most ten crops."""

    binary, height, width = _validate(mask, image)
    components = _extract(binary, params.connectivity)
    by_id = {component.component_id: component for component in components}
    edges = []
    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            edge = compatible_components(left, right, params)
            if edge is not None:
                edges.append(edge)
    groups = []
    for group_index, component_ids in enumerate(_graph_groups(components, edges), start=1):
        members = [by_id[component_id] for component_id in component_ids]
        x1 = min(component.bbox[0] for component in members)
        y1 = min(component.bbox[1] for component in members)
        x2 = max(component.bbox[2] for component in members)
        y2 = max(component.bbox[3] for component in members)
        bbox = x1, y1, x2, y2
        padded = (
            max(0, x1 - params.pad), max(0, y1 - params.pad),
            min(width, x2 + params.pad), min(height, y2 + params.pad),
        )
        groups.append(
            {
                "group_id": group_index,
                "raw_component_ids": component_ids,
                "foreground_area": sum(component.foreground_area for component in members),
                "bbox": bbox,
                "padded_bbox": padded,
            }
        )
    groups.sort(
        key=lambda item: (
            -int(item["foreground_area"]),
            int(item["bbox"][1]), int(item["bbox"][0]), int(item["group_id"]),
        )
    )
    retained = []
    for rank, group in enumerate(groups[: params.nmax], start=1):
        x1, y1, x2, y2 = group["padded_bbox"]
        retained.append(
            GroupedWordBlock(
                rank=rank,
                raw_component_ids=tuple(group["raw_component_ids"]),
                foreground_area=int(group["foreground_area"]),
                bbox=tuple(group["bbox"]),
                padded_bbox=tuple(group["padded_bbox"]),
                crop=image[y1:y2, x1:x2].copy(),
            )
        )
    return GroupedWordBlocksResult(
        params=params,
        raw_components=components,
        graph_edges=tuple(edges),
        groups_before_cap=tuple(groups),
        retained=tuple(retained),
        rejected_group_ids=tuple(int(group["group_id"]) for group in groups[params.nmax:]),
    )


__all__: Final = [
    "PRACTICAL_REMAP_V1",
    "GroupedWordBlockParams",
    "GroupedWordBlock",
    "GroupedWordBlocksResult",
    "RawComponent",
    "compatible_components",
    "component_pair_metrics",
    "group_word_blocks",
]
