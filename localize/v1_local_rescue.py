"""Local one-hop proposal rescue layered over frozen practical-remap-v1."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

import numpy as np

from .grouped_word_blocks import GroupedWordBlockParams, group_word_blocks


@dataclass(frozen=True)
class LocalRescueParams:
    max_normalized_gap: float
    max_gap_to_internal_ratio: float
    min_vertical_overlap: float
    max_center_y_ratio: float = 0.5
    max_height_ratio: float = 2.5


@dataclass(frozen=True)
class RescueBlock:
    raw_component_ids: tuple[int, ...]
    padded_bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class RescueResult:
    groups_before_cap: tuple[dict[str, object], ...]
    retained: tuple[RescueBlock, ...]


def _height(box: tuple[int,int,int,int]) -> int: return box[3]-box[1]


def _vertical(left: tuple[int,int,int,int],right: tuple[int,int,int,int]) -> tuple[float,float,float]:
    lh,rh=_height(left),_height(right); reference=(lh+rh)/2
    overlap=max(0,min(left[3],right[3])-max(left[1],right[1]))/min(lh,rh)
    center=abs((left[1]+left[3])/2-(right[1]+right[3])/2)/reference
    return overlap,center,max(lh,rh)/min(lh,rh)


def _union(boxes: list[tuple[int,int,int,int]]) -> tuple[int,int,int,int]:
    return min(v[0] for v in boxes),min(v[1] for v in boxes),max(v[2] for v in boxes),max(v[3] for v in boxes)


def _internal_gap(group: dict[str,object], raw_boxes: dict[int,tuple[int,int,int,int]]) -> float | None:
    boxes=sorted((raw_boxes[int(v)] for v in group["raw_component_ids"]),key=lambda v:(v[0],v[1]))
    gaps=[max(0,right[0]-left[2])/((_height(left)+_height(right))/2) for left,right in zip(boxes,boxes[1:])]
    positive=[v for v in gaps if v>0]
    return float(median(positive)) if positive else None


def _eligible(base: dict[str,object], neighbor: dict[str,object], raw_boxes: dict[int,tuple[int,int,int,int]], params: LocalRescueParams) -> tuple[bool,dict[str,float|None]]:
    left,right=sorted((tuple(base["bbox"]),tuple(neighbor["bbox"])),key=lambda v:v[0])
    gap=max(0,right[0]-left[2])/((_height(left)+_height(right))/2)
    overlap,center,height_ratio=_vertical(left,right); internal=_internal_gap(base,raw_boxes)
    relative=None if internal is None else gap/internal
    accepted=(gap<=params.max_normalized_gap and overlap>=params.min_vertical_overlap and center<=params.max_center_y_ratio and height_ratio<=params.max_height_ratio and (relative is None or relative<=params.max_gap_to_internal_ratio))
    return accepted,{"normalized_gap":gap,"vertical_overlap":overlap,"center_y_ratio":center,"height_ratio":height_ratio,"base_internal_gap":internal,"gap_to_internal_ratio":relative}


def local_rescue(mask: np.ndarray,image: np.ndarray,v1_params: GroupedWordBlockParams,params: LocalRescueParams,raw_boxes: dict[int,tuple[int,int,int,int]]) -> RescueResult:
    v1=group_word_blocks(mask,image,v1_params); bases=[dict(v) for v in v1.groups_before_cap]
    # Same-line relation uses only the fixed vertical geometry. Connected line
    # sets are used solely to define immediate left/right neighbours.
    parent=list(range(len(bases)))
    def find(v:int)->int:
        while parent[v]!=v: parent[v]=parent[parent[v]];v=parent[v]
        return v
    for i,left in enumerate(bases):
        for j in range(i+1,len(bases)):
            overlap,center,ratio=_vertical(tuple(left["bbox"]),tuple(bases[j]["bbox"]))
            if overlap>=params.min_vertical_overlap and center<=params.max_center_y_ratio and ratio<=params.max_height_ratio:
                a,b=find(i),find(j)
                if a!=b: parent[max(a,b)]=min(a,b)
    lines:dict[int,list[dict[str,object]]]={}
    for index,value in enumerate(bases):lines.setdefault(find(index),[]).append(value)
    proposals=[]
    for members in lines.values():
        members.sort(key=lambda v:(v["bbox"][0],v["bbox"][1],v["group_id"]))
        for index,base in enumerate(members):
            left=members[index-1] if index else None;right=members[index+1] if index+1<len(members) else None
            left_ok=left is not None and _eligible(base,left,raw_boxes,params)[0]
            right_ok=right is not None and _eligible(base,right,raw_boxes,params)[0]
            variants=[[base]]
            if left_ok:variants.append([left,base])
            if right_ok:variants.append([base,right])
            if left_ok and right_ok:variants.append([left,base,right])
            for variant in variants:
                ids=tuple(sorted({int(raw) for group in variant for raw in group["raw_component_ids"]}))
                bbox=_union([tuple(group["bbox"]) for group in variant]);x1,y1,x2,y2=bbox;height,width=mask.shape
                proposals.append({"raw_component_ids":ids,"foreground_area":sum(int(group["foreground_area"]) for group in variant),"bbox":bbox,"padded_bbox":(max(0,x1-v1_params.pad),max(0,y1-v1_params.pad),min(width,x2+v1_params.pad),min(height,y2+v1_params.pad))})
    unique={tuple(value["raw_component_ids"]):value for value in proposals};ordered=list(unique.values())
    ordered.sort(key=lambda v:(-v["foreground_area"],v["bbox"][1],v["bbox"][0],v["raw_component_ids"]))
    for index,value in enumerate(ordered,1):value["group_id"]=index
    retained=tuple(RescueBlock(tuple(v["raw_component_ids"]),tuple(v["padded_bbox"])) for v in ordered[:v1_params.nmax])
    return RescueResult(tuple(ordered),retained)


__all__=["LocalRescueParams","local_rescue"]
