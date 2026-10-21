"""Train-diagnostic nearest-neighbour word grouping from raw components only."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Literal, Sequence

import cv2
import numpy as np

from .line_word_blocks import Atom, V2WordBlockParams, _extract_raw, _line_candidates
from .mask_to_blocks import APPROXIMATE_TIE_BREAK


NNFamily = Literal["absolute", "local_median_ratio", "robust_mad"]


@dataclass(frozen=True)
class NNWordGroupingParams:
    family: NNFamily
    boundary_threshold: float
    local_window: int
    fallback_absolute_gap: float = 0.35
    line_vertical_overlap_min: float = 0.7
    line_max_center_y_ratio: float = 0.5
    line_max_height_ratio: float = 2.5
    pad: int = 6
    connectivity: int = 8
    nmax: int = 10


@dataclass(frozen=True)
class NNBlock:
    raw_component_ids: tuple[int, ...]
    foreground_area: int
    padded_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class NNResult:
    groups_before_cap: tuple[dict[str, object], ...]
    retained: tuple[NNBlock, ...]
    boundary_edges: tuple[dict[str, object], ...]


def _height(box: tuple[int, int, int, int]) -> int:
    return box[3] - box[1]


def _gap(left: Atom, right: Atom) -> float:
    return max(0, right.bbox[0] - left.bbox[2]) / ((_height(left.bbox) + _height(right.bbox)) / 2)


def _neighbors(gaps: Sequence[float], index: int, window: int) -> list[float]:
    return [float(value) for candidate, value in enumerate(gaps) if candidate != index and abs(candidate-index) <= window]


def _boundary(gap: float, neighbors: list[float], params: NNWordGroupingParams) -> tuple[bool, float | None]:
    if params.family == "absolute":
        return gap > params.boundary_threshold, gap
    if not neighbors:
        return gap > params.fallback_absolute_gap, None
    reference = float(median(neighbors))
    if params.family == "local_median_ratio":
        if reference <= 0:
            return gap > params.fallback_absolute_gap, None
        score = gap / reference
        return score > params.boundary_threshold, score
    deviations = [abs(value-reference) for value in neighbors]
    mad = float(median(deviations))
    if mad <= 0:
        return gap > params.fallback_absolute_gap, None
    score = (gap-reference) / (1.4826*mad)
    return score > params.boundary_threshold, score


def group_nn_words(mask: np.ndarray, image: np.ndarray, params: NNWordGroupingParams) -> NNResult:
    if mask.shape != image.shape[:2]:
        raise ValueError("mask/image geometry mismatch")
    raw = _extract_raw((mask > 0).astype(np.uint8), params.connectivity)
    atoms = tuple(Atom(v.component_id, (v.component_id,), v.foreground_area, v.bbox) for v in raw)
    line_params = V2WordBlockParams(
        family="absolute", absolute_gap_floor=0, local_window=2, local_gap_ratio=1,
        local_fallback_absolute_floor=0, line_vertical_overlap_min=params.line_vertical_overlap_min,
        line_max_center_y_ratio=params.line_max_center_y_ratio, line_max_height_ratio=params.line_max_height_ratio,
        fragment_x_overlap_min=1, fragment_max_vertical_gap_ratio=0,
        fragment_small_height_ratio=0, pad=params.pad, connectivity=params.connectivity,
    )
    lines, _ = _line_candidates(atoms, line_params)
    by_id = {value.atom_id: value for value in atoms}
    groups=[]; edges=[]
    for line in lines:
        members=[by_id[int(value)] for value in line["atom_ids"]]
        gaps=tuple(_gap(left,right) for left,right in zip(members,members[1:]))
        start=0
        for index,gap in enumerate(gaps):
            neighbor_values=_neighbors(gaps,index,params.local_window)
            is_boundary,score=_boundary(gap,neighbor_values,params)
            edges.append({"line_id":line["line_id"],"left_component_id":members[index].atom_id,"right_component_id":members[index+1].atom_id,"horizontal_gap_over_mean_height":gap,"local_values":neighbor_values,"decision_score":score,"is_word_boundary":is_boundary})
            if is_boundary:
                groups.append((line["line_id"],members[start:index+1]));start=index+1
        groups.append((line["line_id"],members[start:]))
    height,width=mask.shape; records=[]
    for group_id,(line_id,members) in enumerate(groups,1):
        x1=min(v.bbox[0] for v in members);y1=min(v.bbox[1] for v in members);x2=max(v.bbox[2] for v in members);y2=max(v.bbox[3] for v in members)
        records.append({"group_id":group_id,"line_id":line_id,"raw_component_ids":tuple(v.atom_id for v in members),"foreground_area":sum(v.foreground_area for v in members),"bbox":(x1,y1,x2,y2),"padded_bbox":(max(0,x1-params.pad),max(0,y1-params.pad),min(width,x2+params.pad),min(height,y2+params.pad))})
    records.sort(key=lambda v:(-int(v["foreground_area"]),v["bbox"][1],v["bbox"][0],v["group_id"]))
    retained=tuple(NNBlock(tuple(v["raw_component_ids"]),int(v["foreground_area"]),tuple(v["padded_bbox"])) for v in records[:params.nmax])
    return NNResult(tuple(records),retained,tuple(edges))


__all__ = ["NNWordGroupingParams", "NNResult", "group_nn_words"]
