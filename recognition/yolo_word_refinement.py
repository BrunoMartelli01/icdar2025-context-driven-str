"""GT-blind object-box crop variants and OCR-supported admission, no dataset rules."""
from __future__ import annotations
import math
import numpy as np
from localize.independent_word_slots import overlap_iou
from recognition.group_word_selection import normalize, reading_confidence


def ocr_quality(row: dict) -> float:
    if not normalize(row["raw_text"]):
        return 0.0
    return reading_confidence(row)["confidence_balanced"]


def joint_score(row: dict) -> float:
    return math.sqrt(row["detector_score"] * ocr_quality(row))


def nms(rows: list[dict], iou_threshold: float, *, same_text: bool = False,
        joint: bool = False) -> list[dict]:
    """Suppress overlapping boxes deterministically; optionally require equal reading."""
    if not 0 < iou_threshold < 1:
        raise ValueError("NMS IoU must be in (0,1)")
    selected = []
    for row in sorted(rows, key=lambda r: (-(joint_score(r) if joint else r["detector_score"]),
                                          r["candidate_id"])):
        if any(overlap_iou(row["bbox"], old["bbox"]) > iou_threshold
               and (not same_text or normalize(row["raw_text"]) == normalize(old["raw_text"]))
               for old in selected):
            continue
        selected.append(row)
    return selected


def crop_box(box: list[int], image_wh: tuple[int, int], fraction: float) -> list[int]:
    """Expand/shrink each axis proportionally; return exclusive integer coordinates."""
    if not -.5 < fraction < .5:
        raise ValueError("crop fraction must be in (-0.5,0.5)")
    x1,y1,x2,y2=box
    if not (0 <= x1 < x2 <= image_wh[0] and 0 <= y1 < y2 <= image_wh[1]):
        raise ValueError("invalid original box")
    dx,dy=(x2-x1)*fraction,(y2-y1)*fraction
    result=[max(0,math.floor(x1-dx)),max(0,math.floor(y1-dy)),
            min(image_wh[0],math.ceil(x2+dx)),min(image_wh[1],math.ceil(y2+dy))]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("crop variant has no area")
    return result


def tile_windows(image_wh: tuple[int,int], overlap: float) -> list[list[int]]:
    """Fixed 2x2 overlapping windows; fraction is overlap relative to tile extent."""
    if not 0 <= overlap < 1 or min(image_wh) < 2:
        raise ValueError("invalid tile geometry")
    def intervals(size):
        extent=min(size,math.ceil(size/(2-overlap)))
        return [(0,extent),(size-extent,size)]
    return [[x1,y1,x2,y2] for y1,y2 in intervals(image_wh[1])
            for x1,x2 in intervals(image_wh[0])]


def translated_slots(slots: list[dict], window: list[int], image_wh: tuple[int,int],
                     prefix: str) -> list[dict]:
    x,y,_,_=window
    rows=[]
    for slot in slots:
        a,b,c,d=slot["bbox"]
        bbox=[a+x,b+y,c+x,d+y]
        if not (0 <= bbox[0] < bbox[2] <= image_wh[0] and
                0 <= bbox[1] < bbox[3] <= image_wh[1]):
            raise ValueError("translated tile box outside image")
        rows.append({"candidate_id":f"{prefix}:{slot['source_index']}",
                     "bbox":bbox,"detector_score":slot["score"]})
    return rows


def arm_predictions(rows: list[dict], cfg: dict) -> dict[str,list[dict]]:
    arms={f"raw_{threshold}":[r for r in rows if r["detector_score"]>=threshold]
          for threshold in cfg["score_controls"]}
    raw=arms[f"raw_{cfg['reference_score']}"]
    arms["nonempty"]=[r for r in raw if normalize(r["raw_text"])]
    arms["geometry_nms"]=nms(raw,cfg["merge_iou"])
    for threshold in cfg["joint_controls"]:
        admitted=[r for r in rows if normalize(r["raw_text"]) and joint_score(r)>=threshold]
        arms[f"joint_{threshold}"]=admitted
        arms[f"joint_text_nms_{threshold}"]=nms(admitted,cfg["merge_iou"],same_text=True,joint=True)
    return arms
