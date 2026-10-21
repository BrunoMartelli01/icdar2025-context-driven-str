"""Opt-in geometric TRAIN diagnostic; frozen v1 and Algorithm1 stay unchanged.

Mutual immediate neighbours are linked first horizontally, then vertically,
then on the +45/-45 degree axes. Later passes see only unused singletons.
No RGB, glyph labels, word GT, OCR, learned model or morphology is used.
"""
from __future__ import annotations

import math
import numpy as np

from .grouped_word_blocks import GroupedWordBlockParams, RawComponent, _extract


AXES = {"H": (1., 0.), "V": (0., 1.),
        "D45": (math.sqrt(.5), math.sqrt(.5)),
        "D135": (-math.sqrt(.5), math.sqrt(.5))}


def project(component: RawComponent, axis: str):
    u = np.asarray(AXES[axis])
    v = np.array([-u[1], u[0]])
    x1, y1, x2, y2 = component.bbox
    corners = np.array([[x1, y1], [x1, y2], [x2, y1], [x2, y2]], dtype=float)
    along, across = corners @ u, corners @ v
    return (float(along.min()), float(along.max()), float(across.min()), float(across.max()),
            float(np.asarray(component.centroid) @ u), float(np.asarray(component.centroid) @ v))


def axis_edges(components, axis, params):
    """Nearest compatible forward/backward neighbour in a projected frame.

    The four v1 gates are transferred without retuning. Distance is longitudinal
    bbox gap/mean transverse extent, then centroid distance along the axis.
    Ties use raw IDs; only reciprocal choices become edges.
    """
    projected = {c.component_id: project(c, axis) for c in components}
    ordered = sorted(components, key=lambda c: (projected[c.component_id][4], c.component_id))
    forward, backward, metrics = {}, {}, {}
    for i, left in enumerate(ordered):
        l = projected[left.component_id]
        for right in ordered[i + 1:]:
            r = projected[right.component_id]
            if r[4] <= l[4]:
                continue
            lh, rh = l[3] - l[2], r[3] - r[2]
            ref = (lh + rh) / 2
            gap = max(0., r[0] - l[1]) / ref
            overlap = max(0., min(l[3], r[3]) - max(l[2], r[2])) / min(lh, rh)
            center = abs(l[5] - r[5]) / ref
            height = max(lh, rh) / min(lh, rh)
            if (gap > params.horizontal_gap_ratio or overlap < params.vertical_overlap_min
                    or center > params.max_center_y_ratio or height > params.max_height_ratio):
                continue
            a, b = left.component_id, right.component_id
            distance = (gap, (r[4] - l[4]) / ref)
            f, back = (*distance, b), (*distance, a)
            if a not in forward or f < forward[a]:
                forward[a] = f
            if b not in backward or back < backward[b]:
                backward[b] = back
            metrics[(a, b)] = {"axis": axis, "left_id": a, "right_id": b,
                              "normalized_gap": gap, "transverse_overlap": overlap,
                              "normalized_center_offset": center, "transverse_size_ratio": height}
    return [metrics[(a, choice[-1])] for a, choice in sorted(forward.items())
            if backward[choice[-1]][-1] == a]


def _components_from_edges(ids, edges):
    parent = {i: i for i in ids}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for e in edges:
        a, b = find(e["left_id"]), find(e["right_id"])
        parent[max(a, b)] = min(a, b)
    groups = {}
    for i in sorted(ids):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _snapshot(locked, available, raw, shape, params):
    groups = []
    h, w = shape
    for ids, axis in locked + [([i], "SINGLETON") for i in sorted(available)]:
        members = [raw[i] for i in ids]
        box = [min(c.bbox[0] for c in members), min(c.bbox[1] for c in members),
               max(c.bbox[2] for c in members), max(c.bbox[3] for c in members)]
        padded = [max(0, box[0] - params.pad), max(0, box[1] - params.pad),
                  min(w, box[2] + params.pad), min(h, box[3] + params.pad)]
        groups.append({"raw_component_ids": sorted(ids), "axis": axis, "bbox": box,
                       "padded_bbox": padded,
                       "foreground_area": sum(c.foreground_area for c in members)})
    groups.sort(key=lambda g: (-g["foreground_area"], g["bbox"][1], g["bbox"][0], g["raw_component_ids"]))
    return {"boxes": [g["padded_bbox"] for g in groups[:params.nmax]],
            "before_cap_boxes": [g["padded_bbox"] for g in groups], "groups": groups,
            "raw_count": len(raw), "merges": sum(len(g["raw_component_ids"]) - 1 for g in groups),
            "multi": sum(len(g["raw_component_ids"]) > 1 for g in groups)}


def directional_neighbors(mask: np.ndarray, params: GroupedWordBlockParams):
    """Return H/HV/HVD ablations from a single, deterministic priority pass."""
    if mask.ndim != 2 or min(mask.shape) < 1 or not np.isin(mask, [0, 1]).all():
        raise ValueError("requires nonempty binary 2D mask")
    components = _extract(mask.astype(np.uint8), params.connectivity)
    raw = {c.component_id: c for c in components}
    available = set(raw)
    locked, stages, results = [], {}, {}
    for axis in AXES:
        members = [raw[i] for i in sorted(available)]
        edges = axis_edges(members, axis, params)
        groups = _components_from_edges(available, edges)
        committed = [g for g in groups if len(g) >= 2]
        for ids in committed:
            locked.append((ids, axis))
            available.difference_update(ids)
        stages[axis] = {"eligible_singletons": len(members), "edges": edges,
                        "new_groups": len(committed), "consumed_components": sum(map(len, committed))}
        if axis in ("H", "V", "D135"):
            name = {"H": "DIRECTIONAL_H", "V": "DIRECTIONAL_HV", "D135": "DIRECTIONAL_HVD"}[axis]
            results[name] = _snapshot(locked, available, raw, mask.shape, params)
    return {"methods": results, "stages": stages}
