"""APPROXIMATE local split/merge hypotheses for stretch-resized native128 masks.

No annotations enter this module. Coordinates are exclusive xyxy. Reuse the
repository's geodesic reconstruction and seam generators, without their
letterbox mapping (this U-Net uses a direct square resize).
"""
from __future__ import annotations

import json
import cv2
import numpy as np

from localize.crop_recovery import geodesic_partition, seam_partitions, tight_box
from localize.native_repository_contours import remap_native_box


METHODS = ("baseline", "threshold_split", "erosion_split", "seam_split", "merge")


def proposal(box, image_hw, *, padding, method, parents):
    box = list(map(int, box))
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= 128 and 0 <= y0 < y1 <= 128):
        raise ValueError("Invalid native128 proposal")
    crop = [max(0, x0-padding), max(0, y0-padding),
            min(128, x1+padding), min(128, y1+padding)]
    return {"bbox_native": box, "bbox": remap_native_box(box, image_hw),
            "padded_bbox_native": crop,
            "padded_bbox_original": remap_native_box(crop, image_hw),
            "method": method, "parents": list(parents)}


def baseline(probability, image_hw, config):
    p = np.asarray(probability)
    if (p.shape != (128, 128) or not np.isfinite(p).all()
            or np.any(p < 0) or np.any(p > 1)):
        raise ValueError("Expected finite native128 probabilities")
    contours, _ = cv2.findContours((p >= config["threshold"]).astype(np.uint8),
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for cid, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        row = proposal([x, y, x+w, y+h], image_hw,
                       padding=config["padding"], method="baseline", parents=[cid])
        row["contour_index"] = cid
        row["id"] = f"base:{cid}"
        rows.append(row)
    # Preserve historical padded-bbox sorting, including OpenCV tie order.
    rows.sort(key=lambda r: -((r["padded_bbox_native"][2]-r["padded_bbox_native"][0]) *
                              (r["padded_bbox_native"][3]-r["padded_bbox_native"][1])))
    return rows


def hypotheses(probability, rgb, config):
    """Return uncapped baseline, bounded local pool, and replacement groups."""
    p = np.asarray(probability)
    raw = baseline(p, rgb.shape[:2], config)
    base = raw[:config["baseline_cap"]]
    pool = {r["id"]: dict(r) for r in base}
    groups, seen = [], set()
    canvas = cv2.resize(rgb, (128, 128), interpolation=cv2.INTER_AREA)

    def add(pieces, offset, parents, method, origin):
        if not 1 <= len(pieces) <= config["maximum_pieces"]:
            return
        if method != "merge" and len(pieces) < 2:
            return
        boxes = []
        for piece in pieces:
            if np.count_nonzero(piece) < config["minimum_piece_area"]:
                return
            x0, y0, x1, y1 = tight_box(piece)
            boxes.append([x0+offset[0], y0+offset[1], x1+offset[0], y1+offset[1]])
        signature = (tuple(sorted(parents)), tuple(sorted(map(tuple, boxes))))
        if signature in seen or len(pool)+len(boxes) > config["maximum_pool"]:
            return
        seen.add(signature)
        ids = []
        gid = f"group:{len(groups)}"
        for j, box in enumerate(boxes):
            cid = f"{gid}:{j}"
            pool[cid] = {**proposal(box, rgb.shape[:2], padding=config["padding"],
                                   method=method, parents=parents), "id": cid}
            ids.append(cid)
        groups.append({"id": gid, "parents": list(parents), "children": ids,
                       "method": method, "origin": origin})

    binary = (p >= config["threshold"]).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for parent in base:
        x0, y0, x1, y1 = parent["bbox_native"]
        patch = p[y0:y1, x0:x1]
        # Use the actual component, not any neighboring foreground in its bbox.
        component = np.zeros_like(binary)
        cv2.drawContours(component, contours, parent["contour_index"], 1, cv2.FILLED)
        support = component[y0:y1, x0:x1].astype(bool) & binary[y0:y1, x0:x1].astype(bool)
        parents = [parent["id"]]
        for threshold in config["seed_thresholds"]:
            n, seeds = cv2.connectedComponents(((patch >= threshold) & support).astype(np.uint8), connectivity=8)
            counts = np.bincount(seeds.ravel(), minlength=n)
            labels = [i for i in range(1, n) if counts[i] >= config["minimum_seed_area"]]
            if not 2 <= len(labels) <= config["maximum_pieces"]:
                continue
            clean = np.zeros_like(seeds)
            for j, i in enumerate(labels, 1): clean[seeds == i] = j
            partition = geodesic_partition(patch, support, clean)
            add([partition == j for j in range(1, len(labels)+1)], (x0, y0), parents,
                "threshold_split", {"threshold": threshold})
        for radius in config["erosion_radii"]:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
            core = cv2.erode(support.astype(np.uint8), kernel,
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)
            n, seeds = cv2.connectedComponents(core, connectivity=8)
            if 2 <= n-1 <= config["maximum_pieces"]:
                partition = geodesic_partition(patch, support, seeds)
                add([partition == j for j in range(1, n)], (x0, y0), parents,
                    "erosion_split", {"radius": radius})
        if min(support.shape) >= config["seams"]["minimum_native_extent"]:
            by_cut = {}
            for piece, origin in seam_partitions(patch, support, canvas[y0:y1, x0:x1], config["seams"]):
                # All-seams unions overlap; they are not a valid replacement partition.
                if origin.get("all_seams"): continue
                key = json.dumps(origin, sort_keys=True)
                by_cut.setdefault(key, (origin, []))[1].append(piece)
            for origin, pieces in by_cut.values():
                add(pieces, (x0, y0), parents, "seam_split", origin)

    # Nearby fragment pairs/triples. Geometry gates are fixed before evaluation.
    for i, left in enumerate(base):
        a = left["bbox_native"]; ah = a[3]-a[1]
        neighbors = []
        for right in base[i+1:]:
            b = right["bbox_native"]; bh = b[3]-b[1]
            gap = max(a[0]-b[2], b[0]-a[2], 0)
            overlap = max(0, min(a[3], b[3])-max(a[1], b[1])) / min(ah, bh)
            if overlap >= config["merge_vertical_overlap"] and gap <= config["merge_gap_height"]*max(ah, bh):
                neighbors.append(right)
        for other in neighbors[:config["maximum_merge_neighbors"]]:
            b = other["bbox_native"]
            box = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
            add([np.ones((box[3]-box[1], box[2]-box[0]), bool)], box[:2],
                [left["id"], other["id"]], "merge", {})
    return raw, list(pool.values()), groups


def candidate_features(row, probability, ocr):
    x0, y0, x1, y1 = row["bbox_native"]
    w, h = x1-x0, y1-y0
    patch = probability[y0:y1, x0:x1]
    text = ocr["raw_text"]
    logs = ocr.get("token_log_probabilities", [])
    return [x0/128, y0/128, w/128, h/128, np.log1p(w/h),
            float(patch.mean()), float(patch.std()), float((patch >= .5).mean()),
            float((patch >= .8).mean()), len(text)/25,
            float(ocr["sequence_confidence_native"]),
            float(np.mean(logs)) if logs else -20.,
            min(ocr.get("token_probabilities", []), default=0.),
            len(row["parents"]), *(float(row["method"] == m) for m in METHODS)]


def select_groups(base, pool, groups, scores, *, cost, margin):
    """GT-free, non-conflicting replacements; original crops remain alternatives."""
    active = {p["id"]: p for p in base}
    lookup = {p["id"]: p for p in pool}
    options = []
    for group in groups:
        delta = (sum(scores[i] for i in group["children"])
                 - sum(scores[i] for i in group["parents"])
                 - cost*(len(group["children"])-len(group["parents"])))
        options.append((float(delta), group))
    chosen = []
    for gain, group in sorted(options, key=lambda r: (-r[0], r[1]["id"])):
        if gain <= margin or not all(i in active for i in group["parents"]):
            continue
        for cid in group["parents"]: del active[cid]
        for cid in group["children"]: active[cid] = lookup[cid]
        chosen.append({"group": group["id"], "expected_gain": gain})
    return list(active.values()), chosen
