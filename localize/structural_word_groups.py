"""GT-blind structural ablations on frozen threshold/window crop proposals.

The four stages are separate approximate diagnostics, not paper Algorithm 1.
Masks are projected to a common image grid only to compare provenance; original
crop IDs, boxes and PARSeq readings are never changed.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json

import numpy as np


GRID = 256


@dataclass(frozen=True)
class TraceEvent:
    candidate_id: str
    method: str
    kind: str
    view: str
    pixels: frozenset[int]
    threshold: float | None
    support_key: tuple | None
    seed_ids: tuple[int, ...]
    border_cut: bool

    @property
    def area(self) -> int:
        return len(self.pixels)


class TraceCollector:
    """Collect source-mask footprints without modifying a crop recipe."""

    def __init__(self, image_wh: tuple[int, int], candidate_prefix: str):
        self.image_wh = image_wh
        self.prefix = candidate_prefix
        self.events: dict[str, list[TraceEvent]] = defaultdict(list)

    def __call__(self, cid: str, mask: np.ndarray, offset: tuple[int, int],
                 window: tuple[int, int, int, int], lineage: dict) -> None:
        if mask.ndim != 2:
            raise ValueError("invalid component mask")
        native = int(lineage["native_size"])
        if min(native, *self.image_wh) <= 0 or not np.any(mask):
            raise ValueError("empty or invalid component trace")
        yy, xx = np.nonzero(mask)
        xx = xx.astype(np.float64) + offset[0] + .5
        yy = yy.astype(np.float64) + offset[1] + .5
        x1, y1, x2, y2 = window
        view_w, view_h = x2 - x1, y2 - y1
        side = max(view_w, view_h)
        original_x = xx * side / native - (side - view_w) // 2 + x1
        original_y = yy * side / native - (side - view_h) // 2 + y1
        valid = ((original_x >= x1) & (original_x < x2)
                 & (original_y >= y1) & (original_y < y2)
                 & (original_x >= 0) & (original_x < self.image_wh[0])
                 & (original_y >= 0) & (original_y < self.image_wh[1]))
        if not valid.any():
            return
        gx = np.clip((original_x[valid] * GRID / self.image_wh[0]).astype(np.int32), 0, GRID - 1)
        gy = np.clip((original_y[valid] * GRID / self.image_wh[1]).astype(np.int32), 0, GRID - 1)
        pixels = frozenset(np.unique(gy * GRID + gx).tolist())
        if not pixels:
            return
        view = str(lineage.get("view", "whole"))
        local_x = original_x[valid] - x1
        local_y = original_y[valid] - y1
        native_step = side / native
        border_cut = view != "whole" and (local_x.min() <= native_step
             or local_x.max() >= view_w - native_step
             or local_y.min() <= native_step
             or local_y.max() >= view_h - native_step)
        support_key = None
        if "seed_low" in lineage:
            support_key = (lineage["seed_low"], lineage["seed_high"], lineage["support_index"])
        self.events[f"{self.prefix}:{cid}"].append(TraceEvent(
            candidate_id=f"{self.prefix}:{cid}", method=lineage["method"],
            kind=lineage["kind"], view=view, pixels=pixels,
            threshold=lineage.get("threshold"), support_key=support_key,
            seed_ids=tuple(lineage.get("seed_ids", ())), border_cut=border_cut))


def _overlap(a: TraceEvent, b: TraceEvent) -> tuple[float, float]:
    common = len(a.pixels & b.pixels)
    return common / min(a.area, b.area), min(a.area, b.area) / max(a.area, b.area)


def _source_events(group: dict, events: dict[str, list[TraceEvent]], view: str | None):
    result = []
    for cid in group["core_candidate_ids"]:
        for event in events.get(cid, ()):
            if event.kind == "seed_union" or (view is not None and event.view != view):
                continue
            result.append(event)
    # The same recipe can arise from several thresholds. Keep only distinct masks
    # and lineage keys; this bounds pairwise comparisons without changing support.
    distinct = {}
    for event in result:
        key = (event.view, event.kind, event.support_key, event.seed_ids, event.pixels)
        distinct.setdefault(key, event)
    return list(distinct.values())


def _lineage_score(left: dict, right: dict, events: dict[str, list[TraceEvent]]) -> float:
    best = 0.
    for a in _source_events(left, events, "whole"):
        for b in _source_events(right, events, "whole"):
            coverage, ratio = _overlap(a, b)
            if coverage >= .85 and ratio >= .5:
                best = max(best, coverage * ratio)
    return best


def _cross_view_score(left: dict, right: dict, events: dict[str, list[TraceEvent]]) -> float:
    best = 0.
    for a in _source_events(left, events, None):
        for b in _source_events(right, events, None):
            if a.view == b.view:
                continue
            coverage, ratio = _overlap(a, b)
            if coverage >= .8 and ratio >= .5:
                best = max(best, coverage * ratio)
            elif coverage >= .9 and ratio >= .25 and (a.border_cut or b.border_cut):
                best = max(best, coverage * ratio)
    return best


def _seed_runs(events: dict[str, list[TraceEvent]]) -> dict[tuple, dict[int, int]]:
    """Split a low-threshold support at spacing gaps measured in glyph height."""
    by_support: dict[tuple, dict[int, TraceEvent]] = defaultdict(dict)
    for proposals in events.values():
        for event in proposals:
            if event.kind == "seed_piece" and event.support_key is not None:
                by_support[event.support_key].setdefault(event.seed_ids[0], event)
    runs = {}
    for support, pieces in by_support.items():
        if len(pieces) < 2:
            continue
        centers = []
        for label, event in pieces.items():
            coords = np.fromiter(event.pixels, dtype=np.int32)
            xy = np.column_stack((coords % GRID, coords // GRID)).astype(np.float64)
            centers.append((label, xy.mean(0), xy))
        cloud = np.array([center for _, center, _ in centers])
        _, _, axes = np.linalg.svd(cloud - cloud.mean(0), full_matrices=False)
        axis = axes[0]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        side = np.array([-axis[1], axis[0]])
        ordered = []
        for label, _, xy in centers:
            primary, transverse = xy @ axis, xy @ side
            ordered.append((float(primary.mean()), label, float(primary.min()), float(primary.max()),
                            float(transverse.min()), float(transverse.max())))
        ordered.sort()
        word = 0
        assignment = {ordered[0][1]: word}
        for previous, current in zip(ordered, ordered[1:]):
            gap = max(0., current[2] - previous[3])
            thickness = max(1., min(previous[5] - previous[4], current[5] - current[4]))
            transverse_overlap = max(0., min(previous[5], current[5]) - max(previous[4], current[4]))
            if gap > .35 * thickness or transverse_overlap < .4 * thickness:
                word += 1
            assignment[current[1]] = word
        runs[support] = assignment
    return runs


def _boundary_score(left: dict, right: dict, events: dict[str, list[TraceEvent]],
                    runs: dict, raw_text: dict[str, str], unions: dict) -> float:
    from recognition.group_word_selection import normalize
    for a in _source_events(left, events, "whole"):
        if a.kind != "seed_piece" or a.support_key not in runs:
            continue
        for b in _source_events(right, events, "whole"):
            if (b.kind == "seed_piece" and b.support_key == a.support_key
                and a.seed_ids != b.seed_ids and
                runs[a.support_key][a.seed_ids[0]] == runs[b.support_key][b.seed_ids[0]]):
                first, second = normalize(raw_text[a.candidate_id]), normalize(raw_text[b.candidate_id])
                if not first or not second or " " in first or " " in second:
                    continue
                pair = {a.seed_ids[0], b.seed_ids[0]}
                if unions.get((a.support_key, frozenset(pair)), set()) & {first + second, second + first}:
                    return 1.
    return 0.


def _reference(cids: list[str], rows: dict[str, dict]) -> str:
    boxes = np.array([rows[cid]["bbox"] for cid in cids], np.float64)
    target = np.array([boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()])
    # A crop covering the whole proposal is preferable when letter fragments merge.
    covers = np.all(boxes[:, :2] <= target[:2], axis=1) & np.all(boxes[:, 2:] >= target[2:], axis=1)
    if covers.any():
        return min((cids[i] for i in np.flatnonzero(covers)), key=lambda cid: (
            (rows[cid]["bbox"][2] - rows[cid]["bbox"][0]) *
            (rows[cid]["bbox"][3] - rows[cid]["bbox"][1]), cid))
    median = np.median(boxes, axis=0)
    return min(cids, key=lambda cid: (float(np.abs(np.asarray(rows[cid]["bbox"]) - median).sum()), cid))


def _merge(base: list[dict], rows: dict[str, dict], events: dict[str, list[TraceEvent]],
           stage: int, raw_text: dict[str, str]):
    runs = _seed_runs(events) if stage >= 3 else {}
    unions = defaultdict(set)
    if stage >= 3:
        from recognition.group_word_selection import normalize
        for proposals in events.values():
            for event in proposals:
                if event.kind == "seed_union" and len(event.seed_ids) == 2:
                    unions[event.support_key, frozenset(event.seed_ids)].add(
                        normalize(raw_text[event.candidate_id]))
    links = {}
    for i, a in enumerate(base):
        for j in range(i + 1, len(base)):
            b = base[j]
            abox, bbox = a["reference_bbox"], b["reference_bbox"]
            if (max(abox[0], bbox[0]) > min(abox[2], bbox[2]) + max(abox[2]-abox[0], bbox[2]-bbox[0])
                or max(abox[1], bbox[1]) > min(abox[3], bbox[3]) + max(abox[3]-abox[1], bbox[3]-bbox[1])):
                continue
            lineage = _lineage_score(a, b, events)
            cross = _cross_view_score(a, b, events) if stage >= 2 else 0.
            boundary = _boundary_score(a, b, events, runs, raw_text, unions) if stage >= 3 else 0.
            score = max(lineage, cross, boundary)
            if score:
                links[i, j] = score
    clusters = [{i} for i in range(len(base))]
    owners = list(range(len(base)))
    for (i, j), _ in sorted(links.items(), key=lambda pair: (-pair[1], pair[0])):
        ci, cj = owners[i], owners[j]
        if ci == cj:
            continue
        # Complete-link protects against transitive bridges seen in the TRAIN
        # negative control. Boundary runs are already full-run links above.
        if not all((min(x, y), max(x, y)) in links for x in clusters[ci] for y in clusters[cj]):
            continue
        clusters[ci] |= clusters[cj]
        for member in clusters[cj]:
            owners[member] = ci
        clusters[cj] = set()
    output = []
    for cluster in (group for group in clusters if group):
        members = [base[i] for i in sorted(cluster)]
        core = sorted({cid for g in members for cid in g["core_candidate_ids"]})
        candidates = sorted({cid for g in members for cid in g["candidate_ids"]})
        reference = members[0]["reference_candidate_id"] if len(members) == 1 else _reference(core, rows)
        gid = sha256(json.dumps(core).encode()).hexdigest()
        output.append({"group_id": gid, "candidate_ids": candidates,
                       "core_candidate_ids": core, "shared_candidate_ids": sorted(set(candidates)-set(core)),
                       "reference_candidate_id": reference, "reference_bbox": rows[reference]["bbox"],
                       "parent_only": all(g["parent_only"] for g in members),
                       "source_group_ids": [g["group_id"] for g in members]})
    if stage >= 3:
        # A multiword parent is already shared by its child hypotheses. Remove
        # only a parent-only group whose every crop survives in another group.
        counts = defaultdict(int)
        for group in output:
            for cid in group["candidate_ids"]:
                counts[cid] += 1
        output = [g for g in output if not (g["parent_only"] and all(
            counts[cid] >= 2 for cid in g["candidate_ids"]))]
    return sorted(output, key=lambda group: group["group_id"])


def admit_groups(groups: list[dict], rows: dict[str, dict], raw_text: dict[str, str],
                 mean_probability: dict[str, float]) -> list[dict]:
    """Image-relative evidence gate, with an escape for single-view words."""
    from recognition.group_word_selection import normalize
    values = [mean_probability[g["group_id"]] for g in groups]
    median_mass = float(np.median(values)) if values else 0.
    kept = []
    for group in groups:
        views, levels, readings = set(), set(), defaultdict(set)
        for cid in group["candidate_ids"]:
            row = rows[cid]
            for origin in row["origins"]:
                view = str(origin.get("tile_id", "whole"))
                views.add(view)
                if "threshold" in origin:
                    levels.add((view, origin["threshold"]))
            word = normalize(raw_text[cid])
            if word:
                readings[word].add(cid)
        stable = any(len(cids) >= 2 for cids in readings.values())
        confirmed = len(views) >= 2 or (len(levels) >= 2 and stable)
        isolated = bool(readings) and mean_probability[group["group_id"]] >= median_mass
        if confirmed or isolated:
            kept.append(group)
    return kept


def structural_variants(base_groups: list[dict], rows: list[dict], collector: TraceCollector,
                        raw_text: dict[str, str], mean_probability: dict[str, float] | None = None):
    by_id = {row["candidate_id"]: row for row in rows}
    if set(by_id) != set(collector.events):
        missing = sorted(set(by_id) - set(collector.events))
        extra = sorted(set(collector.events) - set(by_id))
        raise RuntimeError(f"trace/crop identity mismatch; missing={missing[:3]} extra={extra[:3]}")
    result = {}
    for stage, name in ((1, "lineage"), (2, "lineage_footprint"),
                        (3, "lineage_footprint_boundaries")):
        result[name] = _merge(base_groups, by_id, collector.events, stage, raw_text)
    if mean_probability is not None:
        result["lineage_footprint_boundaries_admission"] = admit_groups(
            result["lineage_footprint_boundaries"], by_id, raw_text, mean_probability)
    return result
