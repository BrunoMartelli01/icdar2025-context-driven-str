"""GT-free, nucleus-preserving link growth and conservative crop replacement."""

import heapq

import cv2
import numpy as np

from localize.unet_crop_links import (
    merge_duplicate_supports, original_label_map, patches_from_labels,
    support_intersection,
)
from localize.unet_word_research_v2 import NEIGHBORS, OPPOSITE
from segmentation.unet_word_research_v2 import neighbor_slices


def seeded_link_instances(probabilities, links, canvas, post, options):
    """Soft reciprocal edges guide a minimax flood; they never create markers.

    Kernel components are immutable markers. Queue ties use path length, marker
    ID and coordinates. Missing-marker foreground components each survive as
    one fallback. All foreground survives, even when every link is zero.
    """
    p, links, canvas = map(np.asarray, (probabilities, links, canvas))
    if (p.ndim != 3 or p.shape[0] != 2 or links.shape != (8, *p.shape[1:])
            or canvas.shape != p.shape[1:] or canvas.dtype != bool
            or not np.isfinite(p).all() or not np.isfinite(links).all()
            or np.any((p < 0) | (p > 1)) or np.any((links < 0) | (links > 1))):
        raise ValueError("Invalid seeded-link probability/canvas evidence")
    lw, nw = options["link_weight"], options["landscape_weight"]
    weights = np.asarray(post["watershed_weights"][:2], np.float64)
    if (not np.isfinite([lw, nw, *weights]).all() or min(lw, nw) < 0
            or lw + nw <= 0 or np.any(weights < 0) or weights.sum() <= 0
            or any(not 0 < post[k] < 1 for k in ("text_threshold", "kernel_threshold"))):
        raise ValueError("Invalid seeded-link weights/thresholds")
    fg = (p[0] >= post["text_threshold"]) & canvas
    kernel = (p[1] >= post["kernel_threshold"]) & fg
    count, markers = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=8)
    labels = markers.copy()
    landscape = np.einsum("c,cyx->yx", weights / weights.sum(), p)
    h, w = fg.shape
    costs = np.zeros_like(links, dtype=np.float64)
    for d, (dy, dx) in enumerate(NEIGHBORS):
        y, x, ny, nx = neighbor_slices(h, w, dy, dx)
        reciprocal = np.minimum(links[d, y, x], links[OPPOSITE[d], ny, nx])
        node = 1 - np.minimum(landscape[y, x], landscape[ny, nx])
        costs[d, y, x] = (lw * (1 - reciprocal) + nw * node) / (lw + nw)
    queue = []
    # Only marker boundary pixels need to enter the queue.
    boundary = kernel & cv2.dilate((fg & ~kernel).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    for y, x in zip(*np.nonzero(boundary)):
        label = int(markers[y, x])
        for d, (dy, dx) in enumerate(NEIGHBORS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not labels[ny, nx]:
                queue.append((float(costs[d, y, x]), 1, label, int(ny), int(nx)))
    heapq.heapify(queue)
    while queue:
        cost, distance, label, y, x = heapq.heappop(queue)
        if labels[y, x]:
            continue
        labels[y, x] = label
        for d, (dy, dx) in enumerate(NEIGHBORS):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and not labels[ny, nx]:
                heapq.heappush(queue, (max(cost, float(costs[d, y, x])),
                                      distance + 1, label, ny, nx))
    fallback_count, fallback = cv2.connectedComponents((fg & (labels == 0)).astype(np.uint8), connectivity=8)
    missing = fg & (labels == 0)
    labels[missing] = fallback[missing] + count - 1
    if not np.array_equal(labels > 0, fg) or not np.array_equal(labels[kernel], markers[kernel]):
        raise RuntimeError("Seeded link growth changed foreground or nucleus identity")
    return labels, {"seed_components": count - 1, "unseeded_fallback_components": fallback_count - 1,
                    "instances": count - 1 + fallback_count - 1, "foreground_pixels": int(fg.sum())}


def seed_problem(probabilities, labels, seed_id, post, options):
    """First-pass evidence only: no GT, transcription, or second-pass scores."""
    p, labels = np.asarray(probabilities), np.asarray(labels)
    if (p.shape != (2, *labels.shape) or not np.isfinite(p).all()
            or np.any((p < 0) | (p > 1))):
        raise ValueError("Invalid first-pass gate evidence")
    support = labels == seed_id
    if not support.any():
        raise ValueError("Missing first-pass seed")
    yy, xx = np.nonzero(support)
    kernel = (p[1] >= post["kernel_threshold"]) & support
    cores, _ = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=8)
    metrics = {"mean_word_probability": float(p[0][support].mean()),
               "kernel_fraction": float(kernel.sum() / support.sum()),
               "kernel_components": cores - 1,
               "bbox_fill": float(support.sum() / ((yy.max() - yy.min() + 1) * (xx.max() - xx.min() + 1)))}
    reasons = []
    if metrics["mean_word_probability"] < options["minimum_mean_probability"]:
        reasons.append("low_word_confidence")
    if metrics["kernel_components"] != 1:
        reasons.append("missing_or_multiple_nuclei")
    if metrics["kernel_fraction"] < options["minimum_kernel_fraction"]:
        reasons.append("weak_nucleus")
    if metrics["bbox_fill"] < options["minimum_bbox_fill"]:
        reasons.append("sparse_support")
    return bool(reasons), {"metrics": metrics, "reasons": reasons}


def replacement_guard(children, original, other_seeds, options):
    """Reject incomplete, excessive, or neighboring-word replacements."""
    if not children:
        return False, {"reason": "no_child"}
    x = min(c["bbox"][0] for c in children); y = min(c["bbox"][1] for c in children)
    r = max(c["bbox"][2] for c in children); b = max(c["bbox"][3] for c in children)
    union = np.zeros((b - y, r - x), bool)
    for c in children:
        cx, cy, cr, cb = c["bbox"]
        union[cy - y:cb - y, cx - x:cr - x] |= c["mask"]
    combined = {"bbox": [x, y, r, b], "mask": union}
    area = int(original["mask"].sum())
    metrics = {"anchor_coverage": support_intersection(combined, original) / area,
               "support_growth": int(union.sum()) / area, "children": len(children)}
    reasons = []
    if metrics["children"] > options["maximum_children"]: reasons.append("too_many_children")
    if metrics["anchor_coverage"] < options["minimum_anchor_coverage"]: reasons.append("anchor_loss")
    if metrics["support_growth"] > options["maximum_support_growth"]: reasons.append("excessive_growth")
    for other in other_seeds:
        ox, oy, or_, ob = other["bbox"]
        for child in children:
            cx, cy, cr, cb = child["bbox"]
            ix, iy, ir, ib = max(ox, cx), max(oy, cy), min(or_, cr), min(ob, cb)
            # Measure foreground of the OTHER seed inside the proposed RGB box,
            # rather than bbox IoU or only overlap of the two predicted masks.
            covered = int(other["mask"][iy - oy:ib - oy, ix - ox:ir - ox].sum()) if ix < ir and iy < ib else 0
            if covered / int(other["mask"].sum()) > options["maximum_neighbor_coverage"]:
                reasons.append("neighbor_in_crop"); break
        if "neighbor_in_crop" in reasons: break
    return not reasons, {"metrics": metrics, "reason": reasons[0] if reasons else "accepted", "reasons": reasons}


def refine_from_evidence(first_pass, record, image_wh, load_crop, source_config,
                         config, *, decoder, selective):
    """Replay frozen crop maps; preserve every retained seed's exact support/box.

    load_crop(seed_id, decoder) returns already qualified arrays, including the
    decoded label map. Selective inference loads no evidence for retained seeds.
    There is no global duplicate merge in the selective path: it could otherwise
    modify a protected seed. This is recorded as part of the experimental rule.
    """
    labels = first_pass["labels"]
    anchor_map = original_label_map(labels, image_wh)
    originals, skipped = patches_from_labels(labels, image_wh)
    by_id = {p["native_instance_id"]: p for p in originals}
    result, decisions = [], []
    audit = {"crop_count": len(record["crops"]), "evaluated_crops": 0,
             "retained_seeds": 0, "accepted_seeds": 0, "rejected_seeds": 0,
             "mapped_empty_instances": skipped, "seed_fallbacks": 0}
    for crop in record["crops"]:
        sid = crop["seed_id"]
        original = by_id.get(sid)
        gate = {"reasons": ["whole_image_fallback"]}
        problem = True
        if selective and not crop["whole_image_fallback"]:
            if original is None:
                decisions.append({"seed_id": sid, "action": "projection_empty"}); continue
            problem, gate = seed_problem(first_pass["probabilities"], labels, sid,
                                         source_config["postprocessing"], config["gate"])
        if not problem:
            result.append(original); audit["retained_seeds"] += 1
            decisions.append({"seed_id": sid, "action": "retained", "gate": gate}); continue
        evidence = load_crop(sid, decoder)
        audit["evaluated_crops"] += 1
        x, y, r, b = crop["context_bbox"]
        anchor = None if crop["whole_image_fallback"] else anchor_map[y:b, x:r] == sid
        children, empty = patches_from_labels(evidence["decoded_labels"], (r - x, b - y), offset=(x, y), anchor=anchor)
        audit["mapped_empty_instances"] += empty
        accepted, guard = True, {"reason": "unconditional"}
        if selective and original is not None:
            accepted, guard = replacement_guard(children, original,
                [p for oid, p in by_id.items() if oid != sid], config["guard"])
        if (not children or not accepted) and original is not None:
            result.append(original); audit["seed_fallbacks"] += 1; audit["rejected_seeds"] += 1
            action = "rejected"
        else:
            result.extend(children); audit["accepted_seeds"] += 1; action = "accepted"
        decisions.append({"seed_id": sid, "action": action, "gate": gate, "guard": guard,
                          "decoder_audit": evidence.get("decoder_audit", {})})
    audit["proposals_before_duplicate_merge"] = len(result)
    if not selective:
        result = merge_duplicate_supports(result, overlap_fraction=source_config["crop"]["duplicate_overlap"])
    else:
        result.sort(key=lambda p: (-int(p["mask"].sum()), p["bbox"][1], p["bbox"][0]))
    audit["proposals_after_duplicate_merge"] = len(result)
    audit["decisions"] = decisions
    return result, audit
