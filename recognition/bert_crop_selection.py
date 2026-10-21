"""GT-blind BERT selection of existing crops, for APPROXIMATE experiments only.

No generated spelling, annotation, dataset lexicon or recursive OCR context.
Ranking within a group and admitting a group are deliberately separate decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from recognition.group_word_selection import box_area, geometric_mean
from recognition.scene_text_v1_evaluation import box_iou


CONDITIONS = ("local", "scene")
FAMILIES = ("confidence", "visual", "baseline_local_joint", "baseline_scene_joint",
            "local_visual", "scene_visual", "scene_joint")


@dataclass(frozen=True)
class CropPolicy:
    maximum_readings: int
    minimum_bbox_iou: float
    minimum_visual_ratio: float
    bert_weight: float
    minimum_gain: float
    anchor_confidence: float
    anchor_mask_mean: float
    anchor_minimum_word_length: int
    anchor_duplicate_iou: float
    neighbours_per_side: int
    minimum_vertical_overlap: float
    maximum_gap_heights: float
    maximum_caption_words: int
    resolution_height_cells: float
    language_log_midpoint: float
    language_log_scale: float
    admission_language_weight: float

    def __post_init__(self):
        for name in ("maximum_readings", "neighbours_per_side", "maximum_caption_words",
                     "anchor_minimum_word_length"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("minimum_bbox_iou", "minimum_visual_ratio", "minimum_vertical_overlap",
                     "anchor_duplicate_iou"):
            if not math.isfinite(getattr(self, name)) or not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0,1]")
        for name in ("anchor_confidence", "anchor_mask_mean", "admission_language_weight"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("maximum_gap_heights", "resolution_height_cells", "language_log_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("bert_weight", "minimum_gain"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.language_log_midpoint):
            raise ValueError("language midpoint must be finite")


def unit(value: float) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("expected a finite score in [0,1]")
    return float(value)


def physical_features(candidate: dict, mass, policy: CropPolicy) -> dict:
    """Absolute mask mean differs from relative group mask completeness."""
    bbox = candidate["bbox"]
    x1, y1, x2, y2 = bbox
    if (box_area(bbox) <= 0 or not 0 <= x1 < x2 <= mass.width
            or not 0 <= y1 < y2 <= mass.height):
        raise ValueError("candidate box must lie within the image")
    scale = mass.size / mass.side
    mean = mass(bbox) / (box_area(bbox) * scale ** 2)
    if not math.isfinite(mean) or not -1e-8 <= mean <= 1 + 1e-8:
        raise ValueError("invalid absolute mask average")
    mean = min(1., max(0., mean))
    confidence = unit(candidate["scores"]["confidence_balanced"])
    support = unit(candidate["scores"]["robust_mask_support"])
    resolution = min(1., (y2 - y1) * scale / policy.resolution_height_cells)
    return {"confidence": confidence, "mask_mean": mean, "mask_support": support,
            "height_cells": (y2 - y1) * scale, "resolution_factor": resolution,
            "ranking_visual": geometric_mean([confidence, support]),
            "admission_visual": geometric_mean([confidence, mean, resolution])}


def _assert_assignment(rows: list[dict]) -> None:
    if (len({r["group_id"] for r in rows}) != len(rows)
            or len({r["candidate_id"] for r in rows}) != len(rows)):
        raise ValueError("each group and original crop must occur at most once")


def observable_sample(baseline: list[dict], evidence: list[dict], rows: list[dict],
                      mass, caption: str, policy: CropPolicy) -> dict:
    """Shortlist distinct native readings and freeze common context before BERT."""
    _assert_assignment(baseline)
    groups = {g["group_id"]: g for g in evidence}
    source = {r["candidate_id"]: r for r in rows}
    if len(groups) != len(evidence) or len(source) != len(rows):
        raise ValueError("duplicate source identity")
    prepared = []
    for base in baseline:
        group = groups[base["group_id"]]
        candidates = group["candidates"]
        cid = base["candidate_id"]
        if any(base[k] != candidates[cid][k] for k in ("bbox", "raw_text", "word")):
            raise ValueError("baseline reading/geometry differs from group evidence")
        features = {k: physical_features(v, mass, policy) for k, v in candidates.items()}
        ordered = sorted(candidates, key=lambda k: (-features[k]["ranking_visual"], k))
        ids, readings = [cid], {candidates[cid]["raw_text"]}
        for other in ordered:
            if len(ids) >= policy.maximum_readings:
                break
            item = candidates[other]
            if (item["raw_text"] in readings or not item["word"]
                    or box_iou(item["bbox"], base["bbox"]) < policy.minimum_bbox_iou
                    or features[other]["ranking_visual"] <
                    policy.minimum_visual_ratio * features[cid]["ranking_visual"]):
                continue
            ids.append(other)
            readings.add(item["raw_text"])
        pool = []
        for other in ids:
            item = candidates[other]
            if (item["raw_text"] != source[other]["ocr"]["raw_text"]
                    or item["bbox"] != source[other]["bbox"]):
                raise ValueError("candidate is not a native PARSeq reading/crop")
            pool.append({"candidate_id": other, **{k: item[k] for k in
                         ("raw_text", "word", "bbox")},
                         "crop_rgb_sha256": source[other]["crop_rgb_sha256"],
                         "physical": features[other]})
        prepared.append({"group_id": base["group_id"], "baseline": dict(base),
                         "candidates": pool})
    anchors = []
    for group in sorted(prepared, key=lambda g: (
            -g["candidates"][0]["physical"]["confidence"], g["group_id"])):
        item = group["candidates"][0]
        p = item["physical"]
        if (len(item["word"]) < policy.anchor_minimum_word_length or p["confidence"] < policy.anchor_confidence
                or p["mask_mean"] < policy.anchor_mask_mean
                or any(box_iou(item["bbox"], old["bbox"]) >= policy.anchor_duplicate_iou for old in anchors)):
            continue
        anchors.append({"group_id": group["group_id"], **item})
    for group in prepared:
        tx1, ty1, tx2, ty2 = group["baseline"]["bbox"]
        left, right = [], []
        target_ids = set(groups[group["group_id"]]["candidates"])
        for anchor in anchors:
            if (anchor["group_id"] == group["group_id"]
                    or target_ids.intersection(groups[anchor["group_id"]]["candidates"])):
                continue
            x1, y1, x2, y2 = anchor["bbox"]
            overlap = max(0, min(y2, ty2) - max(y1, ty1))
            if overlap / min(y2 - y1, ty2 - ty1) < policy.minimum_vertical_overlap:
                continue
            max_gap = policy.maximum_gap_heights * max(y2 - y1, ty2 - ty1)
            if x2 <= tx1 and tx1 - x2 <= max_gap:
                left.append((tx1 - x2, x1, anchor["group_id"], anchor["raw_text"]))
            elif x1 >= tx2 and x1 - tx2 <= max_gap:
                right.append((x1 - tx2, x1, anchor["group_id"], anchor["raw_text"]))
        group["context"] = {"left": [r[3] for r in sorted(
            sorted(left)[:policy.neighbours_per_side], key=lambda r: (r[1], r[2]))],
            "right": [r[3] for r in sorted(
            sorted(right)[:policy.neighbours_per_side], key=lambda r: (r[1], r[2]))]}
    return {"groups": prepared, "anchor_count": len(anchors),
            "caption": " ".join(caption.split()[:policy.maximum_caption_words])}


def score_requests(scorer, requests: list[dict]) -> list[dict]:
    """Batch different group contexts with the existing exact masking algorithm."""
    import torch
    tasks, result = [], []
    for index, request in enumerate(requests):
        items, status = scorer.masked_tasks(request["text"], request["left"],
                                             request["right"], request["caption"])
        result.append({"status": status, "token_count": len(items),
                       "token_log_probabilities": [None] * len(items),
                       "sum_log_probability": None, "mean_log_probability": None})
        tasks.extend((index, position, item) for position, item in enumerate(items))
    tasks.sort(key=lambda row: (len(row[2]["input_ids"]), row[0], row[1]))
    for start in range(0, len(tasks), scorer.batch_size):
        batch = tasks[start:start + scorer.batch_size]
        features = [{k: task[k] for k in ("input_ids", "attention_mask", "token_type_ids")}
                    for _, _, task in batch]
        padded = scorer.tokenizer.pad(features, padding=True, return_tensors="pt")
        padded = {k: v.to(scorer.device) for k, v in padded.items()}
        with torch.inference_mode():
            logits = scorer.model(**padded).logits
            positions = torch.tensor([r[2]["position"] for r in batch], device=scorer.device)
            labels = torch.tensor([r[2]["label"] for r in batch], device=scorer.device)
            selected = logits[torch.arange(len(batch), device=scorer.device), positions].float()
            values = selected.log_softmax(-1).gather(1, labels[:, None]).squeeze(1).cpu().tolist()
        if any(not math.isfinite(v) for v in values):
            raise RuntimeError("BERT crop scoring produced non-finite scores")
        for (index, offset, _), value in zip(batch, values):
            result[index]["token_log_probabilities"][offset] = value
    for row in result:
        if row["status"] == "scored":
            values = row["token_log_probabilities"]
            if not values or any(v is None for v in values):
                raise RuntimeError("incomplete BERT crop token scores")
            row["sum_log_probability"] = sum(values)
            row["mean_log_probability"] = sum(values) / len(values)
    return result


def bert_choices(sample: dict, scores: dict, policy: CropPolicy, condition: str
                 ) -> tuple[list[dict], list[dict]]:
    if condition not in CONDITIONS:
        raise ValueError("unknown BERT context condition")
    if set(scores) != {g["group_id"] for g in sample["groups"]}:
        raise ValueError("BERT score/group population differs")
    proposed, trace = {}, {}
    for group in sample["groups"]:
        gid, pool = group["group_id"], group["candidates"]
        values = scores[gid][condition]
        if len(values) != len(pool):
            raise ValueError("BERT crop score cardinality differs")
        base_value = _language_log(values[0])
        chosen, gain = 0, policy.minimum_gain
        gains = [0.]
        for index, (candidate, value) in enumerate(zip(pool[1:], values[1:]), 1):
            language = _language_log(value)
            visual, base = candidate["physical"]["ranking_visual"], pool[0]["physical"]["ranking_visual"]
            if base_value is None or language is None or visual <= 0 or base <= 0:
                gains.append(None)
                continue
            delta = math.log(visual / base) + policy.bert_weight * (language - base_value)
            gains.append(delta)
            if delta > gain:
                chosen, gain = index, delta
        proposed[gid] = chosen
        trace[gid] = {"group_id": gid, "proposed_index": chosen,
                      "gains": gains, "conflict_reverted": False}
    for _ in range(len(proposed) + 1):
        owners = {}
        for group in sample["groups"]:
            gid = group["group_id"]
            owners.setdefault(group["candidates"][proposed[gid]]["candidate_id"], []).append(gid)
        collisions = [gids for gids in owners.values() if len(gids) > 1]
        if not collisions:
            break
        for gids in collisions:
            for gid in gids:
                proposed[gid] = 0
                trace[gid]["conflict_reverted"] = True
    else:
        raise RuntimeError("crop conflict failed to resolve")
    selected = []
    for group in sample["groups"]:
        gid = group["group_id"]
        item = group["candidates"][proposed[gid]]
        selected.append({"group_id": gid, **{k: item[k] for k in
                         ("candidate_id", "word", "raw_text", "bbox")},
                         "score": item["physical"]["ranking_visual"]})
        trace[gid]["selected_index"] = proposed[gid]
    _assert_assignment(selected)
    return selected, list(trace.values())


def _language_log(score: dict) -> float | None:
    value = score["mean_log_probability"]
    if value is None:
        if score["status"] == "scored":
            raise ValueError("scored BERT row lacks a likelihood")
        return None
    if score["status"] != "scored" or not math.isfinite(value) or value > 0:
        raise ValueError("invalid BERT log likelihood")
    return value


def admission_value(candidate: dict, score: dict | None, policy: CropPolicy) -> float:
    visual = unit(candidate["physical"]["admission_visual"])
    value = _language_log(score) if score is not None else None
    if value is None:
        return visual  # Explicit abstention: unknown/long inputs get no language penalty.
    scaled = (value - policy.language_log_midpoint) / policy.language_log_scale
    language = 1 / (1 + math.exp(-max(-700., min(700., scaled))))
    return visual * language ** policy.admission_language_weight


def selections(sample: dict, scores: dict, policy: CropPolicy) -> dict:
    local, local_trace = bert_choices(sample, scores, policy, "local")
    scene, scene_trace = bert_choices(sample, scores, policy, "scene")
    choices = {"baseline": [g["baseline"] for g in sample["groups"]],
               "local": local, "scene": scene}
    lookup = {g["group_id"]: g for g in sample["groups"]}
    admission = {}
    families = {"confidence": ("baseline", None), "visual": ("baseline", None),
                "baseline_local_joint": ("baseline", "local"),
                "baseline_scene_joint": ("baseline", "scene"),
                "local_visual": ("local", None), "scene_visual": ("scene", None),
                "scene_joint": ("scene", "scene")}
    for family, (selection, condition) in families.items():
        admission[family] = {}
        for selected in choices[selection]:
            group = lookup[selected["group_id"]]
            index = next(i for i, c in enumerate(group["candidates"])
                         if c["candidate_id"] == selected["candidate_id"])
            candidate = group["candidates"][index]
            language = scores[group["group_id"]][condition][index] if condition else None
            value = (candidate["physical"]["confidence"] if family == "confidence"
                     else admission_value(candidate, language, policy))
            admission[family][group["group_id"]] = value
    return {"choices": choices, "admission": admission,
            "trace": {"local": local_trace, "scene": scene_trace}, "ground_truth_use": "none"}


def apply_filter(selection: dict, family: str, threshold: float) -> list[dict]:
    if family not in FAMILIES or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("unknown admission family or invalid threshold")
    choice = ("local" if family == "local_visual" else "scene" if family in
              ("scene_visual", "scene_joint") else "baseline")
    values = selection["admission"][family]
    rows = selection["choices"][choice]
    if set(values) != {r["group_id"] for r in rows}:
        raise ValueError("admission population differs from selection")
    return [row for row in rows if unit(values[row["group_id"]]) >= threshold]
