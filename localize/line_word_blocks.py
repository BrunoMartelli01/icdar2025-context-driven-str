"""Two-stage geometric word grouping for ``practical-remap-v2``.

This is deliberately separate from both paper Algorithm 1 and
``practical-remap-v1``.  It does not use OCR, transcriptions, GT boxes, or
morphology at inference.  The only input is a binary segmentation mask and the
source image used for the resulting crop.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Final, Iterable, Literal

import cv2
import numpy as np

from .mask_to_blocks import ALGORITHM1_NMAX, APPROXIMATE_TIE_BREAK


PRACTICAL_REMAP_V2: Final = "practical-remap-v2"
BoundaryFamily = Literal["absolute", "local_ratio", "hybrid"]


@dataclass(frozen=True)
class RawComponent:
    component_id: int
    foreground_area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]


@dataclass(frozen=True)
class Atom:
    atom_id: int
    raw_component_ids: tuple[int, ...]
    foreground_area: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class V2WordBlockParams:
    """Explicit v2 geometry, including the train-selected boundary rule."""

    family: BoundaryFamily
    absolute_gap_floor: float
    local_window: int
    local_gap_ratio: float
    local_fallback_absolute_floor: float
    line_vertical_overlap_min: float
    line_max_center_y_ratio: float
    line_max_height_ratio: float
    fragment_x_overlap_min: float
    fragment_max_vertical_gap_ratio: float
    fragment_small_height_ratio: float
    pad: int
    connectivity: int = 8
    tie_break: str = APPROXIMATE_TIE_BREAK
    nmax: int = ALGORITHM1_NMAX

    def __post_init__(self) -> None:
        if self.family not in ("absolute", "local_ratio", "hybrid"):
            raise ValueError("unsupported practical-remap-v2 boundary family")
        for name in (
            "absolute_gap_floor", "local_gap_ratio", "local_fallback_absolute_floor",
            "line_vertical_overlap_min", "line_max_center_y_ratio",
            "fragment_x_overlap_min", "fragment_max_vertical_gap_ratio",
            "fragment_small_height_ratio",
        ):
            if getattr(self, name) < 0:
                raise ValueError("{} must be non-negative".format(name))
        if self.local_window not in (2, 3):
            raise ValueError("local_window must be 2 or 3")
        if self.line_max_height_ratio < 1:
            raise ValueError("line_max_height_ratio must be at least one")
        if not 0 <= self.fragment_x_overlap_min <= 1:
            raise ValueError("fragment_x_overlap_min must be in [0, 1]")
        if not 0 <= self.fragment_small_height_ratio <= 1:
            raise ValueError("fragment_small_height_ratio must be in [0, 1]")
        if self.pad < 0 or self.connectivity not in (4, 8):
            raise ValueError("invalid pad or connectivity")
        if self.tie_break != APPROXIMATE_TIE_BREAK or self.nmax != ALGORITHM1_NMAX:
            raise ValueError("v2 fixes the shared approximate tie break and nmax=10")


@dataclass(frozen=True)
class V2LocalizedBlock:
    rank: int
    raw_component_ids: tuple[int, ...]
    atom_ids: tuple[int, ...]
    line_id: int
    foreground_area: int
    bbox: tuple[int, int, int, int]
    padded_bbox: tuple[int, int, int, int]
    crop: np.ndarray


@dataclass(frozen=True)
class V2PreparedMask:
    """Mask-derived stages shared by every train boundary-rule variant."""

    shape_hw: tuple[int, int]
    connectivity: int
    fragment_signature: tuple[float, float, float]
    line_signature: tuple[float, float, float]
    raw_components: tuple[RawComponent, ...]
    fragment_edges: tuple[dict[str, float | int], ...]
    atoms: tuple[Atom, ...]
    line_edges: tuple[dict[str, float | int], ...]
    line_candidates: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class V2GroupingResult:
    params: V2WordBlockParams
    raw_components: tuple[RawComponent, ...]
    fragment_edges: tuple[dict[str, float | int], ...]
    atoms: tuple[Atom, ...]
    line_edges: tuple[dict[str, float | int], ...]
    line_candidates: tuple[dict[str, object], ...]
    groups_before_cap: tuple[dict[str, object], ...]
    retained: tuple[V2LocalizedBlock, ...]
    rejected_group_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": PRACTICAL_REMAP_V2,
            "classification": "PRACTICAL_APPROXIMATION",
            "coordinate_convention": "xyxy_exclusive",
            "operation_order": [
                "extract_raw_connected_components",
                "optional_geometric_fragment_consolidation",
                "line_formation_from_vertical_geometry_only",
                "left_to_right_word_boundary_segmentation",
                "union_segment_bboxes",
                "pad_and_clip_group_bbox",
                "sort_by_group_foreground_area_descending",
                "limit_to_top_10",
                "crop_original_image",
            ],
            "parameters": self.params.__dict__.copy(),
            "raw_components": [
                {
                    "component_id": value.component_id,
                    "foreground_area": value.foreground_area,
                    "bbox": list(value.bbox),
                    "centroid": list(value.centroid),
                }
                for value in self.raw_components
            ],
            "fragment_edges": [dict(value) for value in self.fragment_edges],
            "atoms": [
                {
                    "atom_id": value.atom_id,
                    "raw_component_ids": list(value.raw_component_ids),
                    "foreground_area": value.foreground_area,
                    "bbox": list(value.bbox),
                }
                for value in self.atoms
            ],
            "line_edges": [dict(value) for value in self.line_edges],
            "line_candidates": [dict(value) for value in self.line_candidates],
            "groups_before_cap": [dict(value) for value in self.groups_before_cap],
            "retained": [
                {
                    "rank": value.rank,
                    "raw_component_ids": list(value.raw_component_ids),
                    "atom_ids": list(value.atom_ids),
                    "line_id": value.line_id,
                    "foreground_area": value.foreground_area,
                    "bbox": list(value.bbox),
                    "padded_bbox": list(value.padded_bbox),
                    "crop_shape": list(value.crop.shape),
                }
                for value in self.retained
            ],
            "rejected_group_ids": list(self.rejected_group_ids),
        }


def _validate(mask: np.ndarray, image: np.ndarray) -> tuple[np.ndarray, int, int]:
    if mask.ndim != 2 or image.ndim not in (2, 3) or mask.shape != image.shape[:2]:
        raise ValueError("mask and image must have identical HxW geometry")
    binary = (mask > 0).astype(np.uint8)
    return binary, *binary.shape


def _extract_raw(binary: np.ndarray, connectivity: int) -> tuple[RawComponent, ...]:
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity)
    values = []
    for component_id in range(1, count):
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        values.append(RawComponent(
            component_id=component_id,
            foreground_area=int(stats[component_id, cv2.CC_STAT_AREA]),
            bbox=(x, y, x + width, y + height),
            centroid=(float(centroids[component_id][0]), float(centroids[component_id][1])),
        ))
    return tuple(values)


def _union_groups(ids: Iterable[int], edges: Iterable[tuple[int, int]]) -> list[tuple[int, ...]]:
    parent = {value: value for value in ids}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for left, right in edges:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)
    grouped: dict[int, list[int]] = {}
    for value in parent:
        grouped.setdefault(find(value), []).append(value)
    return [tuple(sorted(values)) for _, values in sorted(grouped.items())]


def _height(box: tuple[int, int, int, int]) -> int:
    return box[3] - box[1]


def _width(box: tuple[int, int, int, int]) -> int:
    return box[2] - box[0]


def _bbox_union(boxes: Iterable[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    values = tuple(boxes)
    if not values:
        raise ValueError("cannot union an empty box list")
    return (
        min(value[0] for value in values), min(value[1] for value in values),
        max(value[2] for value in values), max(value[3] for value in values),
    )


def _fragment_edge(
    left: RawComponent, right: RawComponent, params: V2WordBlockParams
) -> dict[str, float | int] | None:
    left_box, right_box = left.bbox, right.bbox
    x_overlap = max(0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0]))
    x_overlap_ratio = x_overlap / min(_width(left_box), _width(right_box))
    vertical_gap = max(0, max(left_box[1], right_box[1]) - min(left_box[3], right_box[3]))
    vertical_gap_ratio = vertical_gap / max(_height(left_box), _height(right_box))
    small_height_ratio = min(_height(left_box), _height(right_box)) / max(_height(left_box), _height(right_box))
    if (
        x_overlap_ratio < params.fragment_x_overlap_min
        or vertical_gap_ratio > params.fragment_max_vertical_gap_ratio
        or small_height_ratio > params.fragment_small_height_ratio
    ):
        return None
    return {
        "left_component_id": left.component_id,
        "right_component_id": right.component_id,
        "x_overlap_over_min_width": x_overlap_ratio,
        "vertical_gap_over_max_height": vertical_gap_ratio,
        "small_height_ratio": small_height_ratio,
    }


def _atoms_from_fragments(
    raw: tuple[RawComponent, ...], params: V2WordBlockParams
) -> tuple[tuple[Atom, ...], tuple[dict[str, float | int], ...]]:
    edges = [
        edge
        for index, left in enumerate(raw)
        for right in raw[index + 1 :]
        if (edge := _fragment_edge(left, right, params)) is not None
    ]
    by_id = {value.component_id: value for value in raw}
    atoms = []
    for atom_id, component_ids in enumerate(
        _union_groups((value.component_id for value in raw),
                      ((int(edge["left_component_id"]), int(edge["right_component_id"])) for edge in edges)),
        start=1,
    ):
        members = [by_id[value] for value in component_ids]
        atoms.append(Atom(
            atom_id=atom_id,
            raw_component_ids=component_ids,
            foreground_area=sum(value.foreground_area for value in members),
            bbox=_bbox_union(value.bbox for value in members),
        ))
    return tuple(atoms), tuple(edges)


def _vertical_metrics(left: Atom, right: Atom) -> dict[str, float]:
    left_height, right_height = _height(left.bbox), _height(right.bbox)
    reference_height = (left_height + right_height) / 2.0
    vertical_overlap = max(0, min(left.bbox[3], right.bbox[3]) - max(left.bbox[1], right.bbox[1]))
    left_center_y = (left.bbox[1] + left.bbox[3]) / 2.0
    right_center_y = (right.bbox[1] + right.bbox[3]) / 2.0
    return {
        "vertical_overlap_over_min_height": vertical_overlap / min(left_height, right_height),
        "center_y_distance_over_reference_height": abs(left_center_y - right_center_y) / reference_height,
        "height_ratio": max(left_height, right_height) / min(left_height, right_height),
    }


def _line_edge(left: Atom, right: Atom, params: V2WordBlockParams) -> dict[str, float | int] | None:
    metrics = _vertical_metrics(left, right)
    if (
        metrics["vertical_overlap_over_min_height"] < params.line_vertical_overlap_min
        or metrics["center_y_distance_over_reference_height"] > params.line_max_center_y_ratio
        or metrics["height_ratio"] > params.line_max_height_ratio
    ):
        return None
    return {"left_atom_id": left.atom_id, "right_atom_id": right.atom_id, **metrics}


def _line_candidates(
    atoms: tuple[Atom, ...], params: V2WordBlockParams
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, float | int], ...]]:
    edges = [
        edge
        for index, left in enumerate(atoms)
        for right in atoms[index + 1 :]
        if (edge := _line_edge(left, right, params)) is not None
    ]
    by_id = {value.atom_id: value for value in atoms}
    candidates = []
    for line_id, atom_ids in enumerate(
        _union_groups((value.atom_id for value in atoms),
                      ((int(edge["left_atom_id"]), int(edge["right_atom_id"])) for edge in edges)),
        start=1,
    ):
        members = sorted((by_id[value] for value in atom_ids), key=lambda value: (value.bbox[0], value.bbox[1], value.atom_id))
        candidates.append({
            "line_id": line_id,
            "atom_ids": tuple(value.atom_id for value in members),
            "raw_component_ids": tuple(sorted(component_id for value in members for component_id in value.raw_component_ids)),
            "bbox": _bbox_union(value.bbox for value in members),
        })
    return tuple(candidates), tuple(edges)


def _normalized_gap(left: Atom, right: Atom) -> float:
    gap = max(0, right.bbox[0] - left.bbox[2])
    return gap / ((_height(left.bbox) + _height(right.bbox)) / 2.0)


def _local_reference(gaps: tuple[float, ...], index: int, window: int) -> float | None:
    surrounding = [
        gap for candidate, gap in enumerate(gaps)
        if candidate != index and abs(candidate - index) <= window and gap > 0
    ]
    return float(median(surrounding)) if surrounding else None


def _is_boundary(gap: float, local_reference: float | None, params: V2WordBlockParams) -> bool:
    if params.family == "absolute":
        return gap > params.absolute_gap_floor
    if local_reference is None:
        # Two-component lines and all-zero local neighborhoods still require an
        # explicit gap decision; this is fixed protocol behavior, not GT tuning.
        relative_boundary = gap > params.local_fallback_absolute_floor
    else:
        relative_boundary = gap > params.local_gap_ratio * local_reference
    if params.family == "local_ratio":
        return relative_boundary
    return gap > params.absolute_gap_floor and relative_boundary


def _line_word_segments(
    line: dict[str, object], atoms_by_id: dict[int, Atom], params: V2WordBlockParams
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    atoms = [atoms_by_id[int(value)] for value in line["atom_ids"]]
    if len(atoms) == 1:
        return [{"atom_ids": (atoms[0].atom_id,), "boundary_before": None}], []
    gaps = tuple(_normalized_gap(left, right) for left, right in zip(atoms, atoms[1:]))
    boundaries = []
    for index, gap in enumerate(gaps):
        reference = _local_reference(gaps, index, params.local_window)
        if _is_boundary(gap, reference, params):
            boundaries.append(index)
    starts = [0, *(index + 1 for index in boundaries)]
    ends = [*(index + 1 for index in boundaries), len(atoms)]
    segments = [
        {
            "atom_ids": tuple(value.atom_id for value in atoms[start:end]),
            "boundary_before": None if start == 0 else start - 1,
        }
        for start, end in zip(starts, ends)
    ]
    gap_records = [
        {
            "gap_index": index,
            "left_atom_id": atoms[index].atom_id,
            "right_atom_id": atoms[index + 1].atom_id,
            "normalized_gap": gap,
            "local_reference": _local_reference(gaps, index, params.local_window),
            "is_word_boundary": index in boundaries,
        }
        for index, gap in enumerate(gaps)
    ]
    return segments, gap_records


def prepare_word_blocks_v2(mask: np.ndarray, params: V2WordBlockParams) -> V2PreparedMask:
    """Prepare raw, fragment, and line geometry once for a mask.

    Calibration varies only word-boundary rules, so sharing this object makes
    every candidate operate on bit-identical raw components and text lines.
    """

    if mask.ndim != 2:
        raise ValueError("mask must be a single-channel HxW array")
    binary = (mask > 0).astype(np.uint8)
    raw = _extract_raw(binary, params.connectivity)
    atoms, fragment_edges = _atoms_from_fragments(raw, params)
    lines, line_edges = _line_candidates(atoms, params)
    return V2PreparedMask(
        shape_hw=tuple(binary.shape),
        connectivity=params.connectivity,
        fragment_signature=(
            params.fragment_x_overlap_min, params.fragment_max_vertical_gap_ratio,
            params.fragment_small_height_ratio,
        ),
        line_signature=(
            params.line_vertical_overlap_min, params.line_max_center_y_ratio,
            params.line_max_height_ratio,
        ),
        raw_components=raw,
        fragment_edges=fragment_edges,
        atoms=atoms,
        line_edges=line_edges,
        line_candidates=lines,
    )


def _validate_prepared(
    prepared: V2PreparedMask, image: np.ndarray, params: V2WordBlockParams
) -> tuple[int, int]:
    if image.ndim not in (2, 3) or image.shape[:2] != prepared.shape_hw:
        raise ValueError("prepared mask and source image must have identical HxW geometry")
    if (
        prepared.connectivity != params.connectivity
        or prepared.fragment_signature != (
            params.fragment_x_overlap_min, params.fragment_max_vertical_gap_ratio,
            params.fragment_small_height_ratio,
        )
        or prepared.line_signature != (
            params.line_vertical_overlap_min, params.line_max_center_y_ratio,
            params.line_max_height_ratio,
        )
    ):
        raise ValueError("prepared mask is incompatible with the fixed v2 fragment/line geometry")
    return prepared.shape_hw


def group_prepared_word_blocks_v2(
    prepared: V2PreparedMask, image: np.ndarray, params: V2WordBlockParams
) -> V2GroupingResult:
    """Apply only the selected word-boundary stage to a prepared mask."""

    height, width = _validate_prepared(prepared, image, params)
    raw, atoms, lines = prepared.raw_components, prepared.atoms, prepared.line_candidates
    fragment_edges, line_edges = prepared.fragment_edges, prepared.line_edges
    atoms_by_id = {value.atom_id: value for value in atoms}
    groups = []
    line_records = []
    group_id = 1
    for line in lines:
        segments, gaps = _line_word_segments(line, atoms_by_id, params)
        line_records.append({**line, "gaps": tuple(gaps)})
        for segment in segments:
            members = [atoms_by_id[value] for value in segment["atom_ids"]]
            bbox = _bbox_union(value.bbox for value in members)
            x1, y1, x2, y2 = bbox
            groups.append({
                "group_id": group_id,
                "line_id": line["line_id"],
                "atom_ids": tuple(segment["atom_ids"]),
                "raw_component_ids": tuple(sorted(component_id for value in members for component_id in value.raw_component_ids)),
                "foreground_area": sum(value.foreground_area for value in members),
                "bbox": bbox,
                "padded_bbox": (max(0, x1 - params.pad), max(0, y1 - params.pad), min(width, x2 + params.pad), min(height, y2 + params.pad)),
            })
            group_id += 1
    groups.sort(key=lambda value: (-int(value["foreground_area"]), int(value["bbox"][1]), int(value["bbox"][0]), int(value["group_id"])))
    retained = []
    for rank, group in enumerate(groups[:params.nmax], start=1):
        x1, y1, x2, y2 = group["padded_bbox"]
        retained.append(V2LocalizedBlock(
            rank=rank,
            raw_component_ids=tuple(group["raw_component_ids"]),
            atom_ids=tuple(group["atom_ids"]),
            line_id=int(group["line_id"]),
            foreground_area=int(group["foreground_area"]),
            bbox=tuple(group["bbox"]),
            padded_bbox=tuple(group["padded_bbox"]),
            crop=image[y1:y2, x1:x2].copy(),
        ))
    return V2GroupingResult(
        params=params,
        raw_components=raw,
        fragment_edges=fragment_edges,
        atoms=atoms,
        line_edges=line_edges,
        line_candidates=tuple(line_records),
        groups_before_cap=tuple(groups),
        retained=tuple(retained),
        rejected_group_ids=tuple(int(value["group_id"]) for value in groups[params.nmax:]),
    )


def group_word_blocks_v2(
    mask: np.ndarray, image: np.ndarray, params: V2WordBlockParams
) -> V2GroupingResult:
    """Apply fragment→line→relative-boundary grouping to one mask."""

    return group_prepared_word_blocks_v2(prepare_word_blocks_v2(mask, params), image, params)


__all__: Final = [
    "PRACTICAL_REMAP_V2", "RawComponent", "Atom", "V2WordBlockParams",
    "V2LocalizedBlock", "V2PreparedMask", "V2GroupingResult",
    "prepare_word_blocks_v2", "group_prepared_word_blocks_v2", "group_word_blocks_v2",
]
