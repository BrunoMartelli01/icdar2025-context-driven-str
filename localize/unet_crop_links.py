"""Inference-only crop geometry and reciprocal pixel-link components."""

import math
import numpy as np
from PIL import Image
from scipy.ndimage import find_objects
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from localize.unet_word_research_v2 import NEIGHBORS, OPPOSITE
from segmentation.totaltext_image_frequency_augment import remap_letterbox_native_box
from segmentation.unet_word_research_v2 import neighbor_slices


def reciprocal_components(probabilities, links, canvas, post, options):
    """Same graph as V2 link_instances, with connected components computed in C."""
    p, links, canvas = np.asarray(probabilities), np.asarray(links), np.asarray(canvas)
    threshold = options["threshold"]
    if (p.ndim != 3 or p.shape[0] != 2 or links.shape != (8, *p.shape[1:])
            or canvas.shape != p.shape[1:] or canvas.dtype != bool or not 0 < threshold < 1
            or not np.isfinite(p).all() or not np.isfinite(links).all()
            or np.any((p < 0) | (p > 1)) or np.any((links < 0) | (links > 1))):
        raise ValueError("Invalid reciprocal-link evidence")
    foreground = (p[0] >= post["text_threshold"]) & canvas
    n = int(foreground.sum())
    labels = np.zeros(foreground.shape, np.int32)
    if not n:
        return labels, {"link_components": 0, "foreground_pixels": 0}
    indices = np.full(foreground.shape, -1, np.int32)
    indices[foreground] = np.arange(n)
    source, destination = [], []
    h, w = foreground.shape
    for direction in (4, 5, 6, 7):
        dy, dx = NEIGHBORS[direction]
        y, x, ny, nx = neighbor_slices(h, w, dy, dx)
        active = (foreground[y, x] & foreground[ny, nx]
                  & (links[direction, y, x] >= threshold)
                  & (links[OPPOSITE[direction], ny, nx] >= threshold))
        source.append(indices[y, x][active]); destination.append(indices[ny, nx][active])
    rows, cols = np.concatenate(source), np.concatenate(destination)
    graph = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(n, n)).tocsr()
    count, assignments = connected_components(graph, directed=False, return_labels=True)
    labels[foreground] = assignments.astype(np.int32) + 1
    return labels, {"link_components": int(count), "foreground_pixels": n}


def context_box(box, image_wh, options):
    w, h = image_wh
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError("Seed outside source RGB")
    fx = max(options["minimum_context_pixels"], math.ceil((x2 - x1) * options["context_fraction"]))
    fy = max(options["minimum_context_pixels"], math.ceil((y2 - y1) * options["context_fraction"]))
    return [max(0, x1 - fx), max(0, y1 - fy), min(w, x2 + fx), min(h, y2 + fy)]


def original_label_map(labels, image_wh):
    labels = np.asarray(labels)
    w, h = image_wh
    side = max(w, h); left, top = (side - w) // 2, (side - h) // 2
    square = Image.fromarray(labels.astype(np.int32), mode="I").resize((side, side), Image.Resampling.NEAREST)
    return np.asarray(square)[top:top + h, left:left + w].copy()


def patches_from_labels(labels, image_wh, *, offset=(0, 0), anchor=None):
    """Tight original-coordinate supports, retaining exclusive bbox conventions."""
    labels = np.asarray(labels)
    projected = original_label_map(labels, image_wh)
    patches, skipped = [], 0
    for label, region in enumerate(find_objects(labels), start=1):
        if region is None:
            continue
        y, x = region
        box = remap_letterbox_native_box([x.start, y.start, x.stop, y.stop], image_wh,
                                        native_size=labels.shape[0])
        x1, y1, x2, y2 = box
        mask = projected[y1:y2, x1:x2] == label
        if not mask.any():
            skipped += 1
            continue
        if anchor is not None and not np.any(mask & anchor[y1:y2, x1:x2]):
            continue
        patches.append({"bbox": [x1 + offset[0], y1 + offset[1], x2 + offset[0], y2 + offset[1]],
                        "mask": mask, "native_instance_id": label})
    return patches, skipped


def support_intersection(a, b):
    ax, ay, ar, ab = a["bbox"]; bx, by, br, bb = b["bbox"]
    x, y, r, bottom = max(ax, bx), max(ay, by), min(ar, br), min(ab, bb)
    if x >= r or y >= bottom:
        return 0
    return int(np.count_nonzero(a["mask"][y - ay:bottom - ay, x - ax:r - ax]
                               & b["mask"][y - by:bottom - by, x - bx:r - bx]))


def merge_duplicate_supports(patches, *, overlap_fraction):
    """Merge duplicate detections by support overlap, never by GT or OCR text."""
    if not 0 < overlap_fraction <= 1:
        raise ValueError("Invalid overlap fraction")
    merged = []
    for patch in patches:
        patch = {**patch, "mask": patch["mask"].copy()}
        while True:
            matches = [i for i, old in enumerate(merged)
                       if support_intersection(patch, old) >= overlap_fraction * min(
                           int(patch["mask"].sum()), int(old["mask"].sum()))]
            if not matches:
                break
            old = merged.pop(matches[0])
            x = min(patch["bbox"][0], old["bbox"][0]); y = min(patch["bbox"][1], old["bbox"][1])
            r = max(patch["bbox"][2], old["bbox"][2]); b = max(patch["bbox"][3], old["bbox"][3])
            union = np.zeros((b - y, r - x), bool)
            for item in (patch, old):
                ix, iy, ir, ib = item["bbox"]
                union[iy - y:ib - y, ix - x:ir - x] |= item["mask"]
            patch = {"bbox": [x, y, r, b], "mask": union}
        merged.append(patch)
    merged.sort(key=lambda p: (-int(p["mask"].sum()), p["bbox"][1], p["bbox"][0]))
    return merged
