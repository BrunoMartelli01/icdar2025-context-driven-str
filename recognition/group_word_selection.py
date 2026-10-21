"""Dataset-independent, training-free selection of observed crop readings.

No annotations, dictionaries, language priors or dataset-specific word rules.
Boxes are exclusive xyxy in a shared image coordinate system.
"""
from __future__ import annotations

from collections import defaultdict
import math
import unicodedata

import numpy as np
from scipy.optimize import linear_sum_assignment


CHEAP_METHODS = ("confidence_mean", "confidence_min", "confidence_balanced",
                 "consensus", "consensus_weighted", "stability",
                 "consensus_stability_confidence", "mask_completeness_confidence", "combined")


def normalize(text: str) -> str:
    return "".join(c for c in text.casefold() if unicodedata.category(c)[0] not in "PS")


def geometric_mean(values) -> float:
    values = list(values)
    if not values or any(not np.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("geometric mean requires finite scores in [0,1]")
    return math.exp(sum(math.log(v) for v in values) / len(values)) if all(values) else 0.


def box_area(box) -> float:
    return float(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))


def intersection(a, b) -> list[float]:
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def completeness(box, reference, mass=None) -> float:
    """Coverage and foreground purity relative to the group reference support.

    With mass=None this is a geometric control. A probability-mass callable
    supplies a soft mask version without a new binarization cutoff. Empty mask
    support uses the explicitly documented geometric control.
    """
    geometric = mass is None
    weight = box_area if geometric else mass
    target, total = weight(reference), weight(box)
    if target <= 0 or total <= 0:
        weight = box_area
        target, total = weight(reference), weight(box)
    if target <= 0 or total <= 0:
        raise ValueError("nonempty boxes required")
    common = weight(intersection(box, reference))
    return geometric_mean([min(1., max(0., common / target)), min(1., max(0., common / total))])


class ProbabilityMass:
    """Continuous rectangular integrals of a square letterboxed soft mask."""
    def __init__(self, probability: np.ndarray, image_wh: tuple[int, int]):
        p = np.asarray(probability, np.float64)
        if (p.ndim != 2 or p.shape[0] != p.shape[1] or not p.size
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)) or min(image_wh) <= 0):
            raise ValueError("invalid probability canvas or image size")
        self.size = p.shape[0]
        self.width, self.height = image_wh
        self.side = max(image_wh)
        self.pad_x = (self.side - self.width) // 2
        self.pad_y = (self.side - self.height) // 2
        self.integral = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    def _value(self, x, y):
        x, y = np.clip([x, y], 0, self.size)
        ix, iy = int(math.floor(x)), int(math.floor(y))
        jx, jy = min(ix + 1, self.size), min(iy + 1, self.size)
        dx, dy = x - ix, y - iy
        a = self.integral
        return ((1 - dx) * (1 - dy) * a[iy, ix] + dx * (1 - dy) * a[iy, jx]
                + (1 - dx) * dy * a[jy, ix] + dx * dy * a[jy, jx])

    def __call__(self, box):
        if box_area(box) == 0:
            return 0.
        x1, y1, x2, y2 = box
        scale = self.size / self.side
        x1, x2 = (x1 + self.pad_x) * scale, (x2 + self.pad_x) * scale
        y1, y2 = (y1 + self.pad_y) * scale, (y2 + self.pad_y) * scale
        return max(0., self._value(x2, y2) - self._value(x1, y2)
                   - self._value(x2, y1) + self._value(x1, y1))


def aligned_probability(whole: np.ndarray, tiles: list[tuple[np.ndarray, list[int]]],
                        image_wh: tuple[int, int]) -> np.ndarray:
    """Average available model views on the whole-image native canvas.

    Cell-center inverse letterbox mapping; linear interpolation, each view once.
    No mask threshold or annotation-derived separator is introduced.
    """
    import cv2
    ProbabilityMass(whole, image_wh)  # Validate shape/range.
    width, height = image_wh
    side, n = max(image_wh), whole.shape[0]
    coordinates = (np.arange(n, dtype=np.float32) + .5) * side / n - .5
    x, y = np.meshgrid(coordinates - (side - width) // 2,
                       coordinates - (side - height) // 2)
    result = whole.astype(np.float64).copy()
    count = np.ones(whole.shape, np.float64)
    for p, window in tiles:
        x1, y1, x2, y2 = window
        tw, th = x2 - x1, y2 - y1
        ProbabilityMass(p, (tw, th))
        if not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
            raise ValueError("tile window outside original image")
        native_side = max(tw, th)
        tx = ((x - x1 + (native_side - tw) // 2 + .5) * p.shape[0] / native_side - .5)
        ty = ((y - y1 + (native_side - th) // 2 + .5) * p.shape[0] / native_side - .5)
        valid = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
        projected = cv2.remap(p.astype(np.float32), tx.astype(np.float32), ty.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        result += np.where(valid, projected, 0)
        count += valid
    return (result / count).astype(np.float32)


def reading_confidence(ocr: dict) -> dict[str, float]:
    logs = np.asarray(ocr["token_log_probabilities"], np.float64)
    probs = np.asarray(ocr["token_probabilities"], np.float64)
    if (len(logs) != len(ocr["raw_text"]) or logs.shape != probs.shape
        or not np.isfinite(logs).all() or not np.isfinite(probs).all()
        or np.any(logs > 0) or np.any((probs < 0) | (probs > 1))):
        raise ValueError("invalid original PARSeq character scores")
    mean = float(np.exp(logs.mean())) if logs.size else 0.
    minimum = float(probs.min()) if probs.size else 0.
    return {"confidence_mean": mean, "confidence_min": minimum,
            "confidence_balanced": geometric_mean([mean, minimum])}


def view_origins(row: dict, allowed_methods: set[str]) -> list[tuple[str, float | None]]:
    result = set()
    for origin in row["origins"]:
        if origin["method"] not in allowed_methods:
            continue
        view = str(origin.get("tile_id", "whole"))
        threshold = origin.get("threshold")
        if threshold is not None and not 0 < threshold < 1:
            raise ValueError("invalid originating probability threshold")
        result.add((view, threshold))
    if not result:
        raise ValueError("candidate has no admissible generator provenance")
    return sorted(result, key=lambda value: (value[0], -1 if value[1] is None else value[1]))


def threshold_stability(grid: list[float], present: set[float]) -> float:
    """Longest contiguous support interval, normalized by the observed grid span.

    Voronoi interval weights avoid rewarding an artificially denser threshold
    grid. A singleton grid provides no contradiction and has neutral stability 1.
    """
    if not grid:
        return 1.
    grid = np.asarray(sorted(set(grid)), np.float64)
    if len(grid) == 1:
        return float(grid[0] in present)
    boundaries = np.r_[grid[0], (grid[1:] + grid[:-1]) / 2, grid[-1]]
    weights = np.diff(boundaries) / (grid[-1] - grid[0])
    best = current = 0.
    for threshold, weight in zip(grid, weights):
        current = current + weight if threshold in present else 0.
        best = max(best, current)
    return min(1., float(best))


def group_evidence(rows: list[dict], groups: list[dict], *, allowed_methods: set[str],
                   mass=None) -> list[dict]:
    """Inference-only boundary: word evidence and per-crop scores, no GT inputs."""
    by_id = {row["candidate_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate candidate IDs")
    confidence = {cid: reading_confidence(row["ocr"]) for cid, row in by_id.items()}
    origins = {cid: view_origins(row, allowed_methods) for cid, row in by_id.items()}
    output = []
    for group in sorted(groups, key=lambda value: value["group_id"]):
        if not set(group["candidate_ids"]) <= set(by_id):
            raise ValueError("group references missing candidate")
        words = defaultdict(list)
        grids, views = defaultdict(set), set()
        for cid in group["candidate_ids"]:
            for view, threshold in origins[cid]:
                views.add(view)
                if threshold is not None:
                    grids[view].add(threshold)
            word = normalize(by_id[cid]["ocr"]["raw_text"])
            if word:
                words[word].append(cid)
        word_features, candidates = {}, {}
        for word, cids in sorted(words.items()):
            per_view, present = {}, defaultdict(set)
            for cid in cids:
                for view, threshold in origins[cid]:
                    per_view[view] = max(per_view.get(view, 0.), confidence[cid]["confidence_balanced"])
                    if threshold is not None:
                        present[view].add(threshold)
            # Each view contributes at most once, regardless of its crop count.
            consensus = len(per_view) / len(views)
            weighted = sum(per_view.values()) / len(views)
            stability = sum(threshold_stability(sorted(grids[view]), present[view])
                            for view in per_view) / len(views)
            word_features[word] = {"consensus": consensus, "consensus_weighted": weighted,
                                   "stability": stability, "views": sorted(per_view)}
            for cid in sorted(cids):
                row, scores = by_id[cid], dict(confidence[cid])
                support = completeness(row["bbox"], group["reference_bbox"], mass)
                scores.update({key: word_features[word][key]
                               for key in ("consensus", "consensus_weighted", "stability")})
                scores["mask_completeness"] = support
                scores["consensus_stability_confidence"] = geometric_mean([
                    scores["confidence_balanced"], weighted, stability])
                scores["mask_completeness_confidence"] = geometric_mean([
                    scores["confidence_balanced"], support])
                scores["combined"] = geometric_mean([
                    scores["confidence_balanced"], weighted, stability, support])
                candidates[cid] = {"word": word, "raw_text": row["ocr"]["raw_text"],
                                   "bbox": row["bbox"], "scores": scores}
        output.append({"group_id": group["group_id"], "reference_bbox": group["reference_bbox"],
                       "reference_candidate_id": group["reference_candidate_id"],
                       "words": word_features, "candidates": candidates})
    return output


def ranked_candidates(group: dict, method: str) -> list[str]:
    def key(cid):
        scores = group["candidates"][cid]["scores"]
        value = scores[method]
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("selection score outside [0,1]")
        return (-value, -scores["mask_completeness"], -scores["confidence_balanced"], cid)
    return sorted((cid for cid, item in group["candidates"].items() if method in item["scores"]), key=key)


def select_groups(evidence: list[dict], method: str) -> list[dict]:
    """Maximum-score assignment: group<=1, original crop<=1; spelling can repeat.

    Every group has a zero-score abstention option. No GT-tuned rejection cutoff.
    All competing alternatives remain in the evidence artifact.
    """
    ordered = sorted(evidence, key=lambda group: group["group_id"])
    ranked = [ranked_candidates(group, method) for group in ordered]
    ids = sorted({cid for cids in ranked for cid in cids})
    if not ordered:
        return []
    columns = {cid: index for index, cid in enumerate(ids)}
    weights = np.full((len(ordered), len(ids) + len(ordered)), -np.inf)
    for i, (group, cids) in enumerate(zip(ordered, ranked)):
        weights[i, len(ids) + i] = 0.
        for cid in cids:
            weights[i, columns[cid]] = group["candidates"][cid]["scores"][method]
    row_indices, col_indices = linear_sum_assignment(weights, maximize=True)
    selected = []
    for i, column in zip(row_indices, col_indices):
        if column >= len(ids) or weights[i, column] <= 0:
            continue
        cid = ids[column]
        item = ordered[i]["candidates"][cid]
        selected.append({"group_id": ordered[i]["group_id"], "candidate_id": cid,
                         "word": item["word"], "raw_text": item["raw_text"],
                         "bbox": item["bbox"], "score": float(weights[i, column])})
    return selected


def cross_requests(evidence: list[dict], rows: list[dict], *, word_budget: int,
                   crop_budget: int) -> tuple[list[dict], dict]:
    """GT-blind shortlist; prioritize distinct views, then geometric diversity."""
    if word_budget < 1 or crop_budget < 1:
        raise ValueError("positive cross-scoring budgets required")
    by_id = {row["candidate_id"]: row for row in rows}
    requests, plans = {}, {}
    for group in evidence:
        ranked = ranked_candidates(group, "combined")
        words = {}
        for cid in ranked:
            item = group["candidates"][cid]
            words.setdefault(item["word"], item["raw_text"])
        words = dict(list(words.items())[:word_budget])
        remaining = sorted(group["candidates"], key=lambda cid: (
            -group["candidates"][cid]["scores"]["mask_completeness"],
            -group["candidates"][cid]["scores"]["confidence_balanced"], cid))
        chosen, used_views, used_pixels = [], set(), set()
        while remaining and len(chosen) < crop_budget:
            def priority(cid):
                row = by_id[cid]
                views = {str(origin.get("tile_id", "whole")) for origin in row["origins"]}
                box = np.array(row["bbox"], np.float64)
                distance = min((float(np.abs(box - np.array(by_id[old]["bbox"])).sum())
                                for old in chosen), default=0.)
                return (len(views - used_views), distance,
                        group["candidates"][cid]["scores"]["mask_completeness"],
                        group["candidates"][cid]["scores"]["confidence_balanced"])
            cid = max(remaining, key=priority)
            remaining.remove(cid)
            row = by_id[cid]
            if row["crop_rgb_sha256"] in used_pixels:
                continue
            chosen.append(cid)
            used_pixels.add(row["crop_rgb_sha256"])
            used_views.update(str(origin.get("tile_id", "whole")) for origin in row["origins"])
        pairs = {}
        for word, raw in words.items():
            pairs[word] = []
            for cid in chosen:
                crop_hash = by_id[cid]["crop_rgb_sha256"]
                key = (crop_hash, raw)
                request_id = f"{crop_hash}:{raw}"
                requests[key] = {"request_id": request_id, "candidate_id": cid,
                                 "crop_rgb_sha256": crop_hash, "raw_text": raw}
                pairs[word].append(request_id)
        plans[group["group_id"]] = {"words": words, "crops": chosen, "pairs": pairs}
    return sorted(requests.values(), key=lambda row: row["request_id"]), plans


def add_cross_scores(evidence: list[dict], plans: dict, scores: dict[str, float | None]) -> None:
    for group in evidence:
        pairs = plans[group["group_id"]]["pairs"]
        for item in group["candidates"].values():
            if item["word"] in pairs:
                # Equal contribution per representative crop; causal log likelihood
                # includes EOS and is normalized by its original raw token length.
                logs = [scores[key] for key in pairs[item["word"]]]
                if logs and all(value is None for value in logs):
                    item["scores"]["cross_parseq"] = 0.
                    item["cross_parseq_status"] = "UNSUPPORTED_LENGTH"
                    continue
                if not logs or any(value is None or not np.isfinite(value) or value > 0 for value in logs):
                    raise ValueError("invalid cross-crop log likelihood")
                item["scores"]["cross_parseq"] = float(math.exp(sum(logs) / len(logs)))


def add_clip_scores(evidence: list[dict], scores: dict) -> list[str]:
    methods = set()
    for group in evidence:
        for item in group["candidates"].values():
            word_scores = scores[group["group_id"]][item["word"]]
            for method, cosine in word_scores.items():
                if not np.isfinite(cosine) or not -1.000001 <= cosine <= 1.000001:
                    raise ValueError("invalid CLIP cosine")
                item["scores"][method] = float(np.clip((cosine + 1) / 2, 0, 1))
                methods.add(method)
    return sorted(methods)
