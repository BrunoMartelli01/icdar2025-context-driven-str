"""Opt-in approximate contours guided by existing word-kernel/center maps.

GT never enters localization. The original contour localizer and the frozen
multi-task watershed remain unchanged. Refinement only relabels text pixels.
"""
from itertools import combinations

import cv2
import numpy as np

from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from localize.word_multitask import center_peaks, parseq_word_proposals, word_instances


METHODS = ("findcontours", "seeded", "refined")


def _inputs(probabilities, canvas, postprocessing):
    p, canvas = np.asarray(probabilities), np.asarray(canvas)
    if (p.ndim != 3 or p.shape[0] not in (1, 2, 3) or p.shape[1:] != canvas.shape
            or canvas.dtype != np.bool_ or not canvas.size or canvas.shape[0] != canvas.shape[1]
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Expected finite CxSxS probabilities and boolean canvas")
    threshold = postprocessing["text_threshold"]
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Invalid text threshold")
    return p, canvas, (p[0] >= threshold) & canvas


def findcontour_labels(foreground):
    """Use the frozen RETR_EXTERNAL localizer, preserving real pixel membership.

    Drawing filled contour polygons would invent pixels inside holes. Instead,
    map each retained contour to its existing 8-connected component. Nested
    islands omitted by RETR_EXTERNAL remain omitted, as in the original localizer.
    """
    if foreground.ndim != 2 or foreground.dtype != np.bool_ or not foreground.size:
        raise ValueError("Expected a nonempty 2D boolean mask")
    binary = foreground.astype(np.uint8)
    audit = audit_repository_contours(binary, RepositoryContourParams(0, 0, binary.size))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    selected = []
    for row in audit["retained"]:
        x, y = contours[row["contour_index"]][0, 0]
        selected.append(int(labels[y, x]))
    labels[~np.isin(labels, selected)] = 0
    return labels, {"external_contours": len(contours),
                    "nested_foreground_pixels_omitted": int(np.count_nonzero(foreground & (labels == 0)))}


def _seeded(p, canvas, postprocessing, cached_labels):
    if cached_labels is None:
        return word_instances(p, canvas, postprocessing)[0]
    labels = np.asarray(cached_labels)
    foreground = (p[0] >= postprocessing["text_threshold"]) & canvas
    if (labels.shape != canvas.shape or labels.dtype.kind not in "iu"
            or labels.min() < 0 or labels.max() > np.iinfo(np.int32).max
            or not np.array_equal(labels > 0, foreground)):
        raise ValueError("Cached seeded labels do not match text support")
    return labels.astype(np.int32, copy=True)


def refine_instances(probabilities, canvas, postprocessing, refinement, *, cached_labels=None):
    """Join fragmented masks using shared auxiliary evidence and center vetoes.

    A shared kernel may only join distinct original text components. A connected
    center region with exactly one retained peak may also join false partitions
    inside one text component. Endpoints must be within the declared Chebyshev
    foreground distance. Two distinct retained centers veto a union, including
    transitive unions. No new foreground pixels or forced fallback merges.
    """
    p, canvas, foreground = _inputs(probabilities, canvas, postprocessing)
    if len(p) < 2:
        raise ValueError("Guided refinement requires a kernel head")
    low = refinement["bridge_text_threshold"]
    distance = refinement["maximum_foreground_distance"]
    if (not np.isfinite(low) or not 0 < low <= postprocessing["text_threshold"]
            or isinstance(distance, bool) or not isinstance(distance, int) or distance < 1
            or any(type(refinement[key]) is not bool for key in ("kernel_merge", "center_merge"))):
        raise ValueError("Invalid declared refinement settings")
    for key in ("kernel_threshold", "center_threshold"):
        if not np.isfinite(postprocessing[key]) or not 0 < postprocessing[key] < 1:
            raise ValueError("Invalid auxiliary threshold")
    labels = _seeded(p, canvas, postprocessing, cached_labels)
    ids = [int(value) for value in np.unique(labels) if value]
    parent = {value: value for value in ids}
    _, original = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
    original_ids = {value: set(map(int, np.unique(original[labels == value]))) for value in ids}
    group_original_ids = {value: set(parts) for value, parts in original_ids.items()}
    low_support = (p[0] >= low) & canvas
    kernel = (p[1] >= postprocessing["kernel_threshold"]) & low_support
    peaks = (center_peaks(p[2], low_support, threshold=postprocessing["center_threshold"],
                         min_distance=postprocessing["center_min_distance"]) if len(p) == 3 else [])
    centers = {value: {index for index, (_, y, x) in enumerate(peaks) if labels[y, x] == value}
               for value in ids}
    candidates, rejected, unions = [], [], []

    def add_regions(support, reason, restrict_original):
        count, regions = cv2.connectedComponents(support.astype(np.uint8), connectivity=8)
        for region_id in range(1, count):
            region = regions == region_id
            peak_ids = {i for i, (_, y, x) in enumerate(peaks) if region[y, x]}
            if len(peak_ids) > 1:
                rejected.append({"reason": "multiple_center_peaks", "evidence": reason,
                                 "region_id": region_id, "center_count": len(peak_ids)})
                continue
            if reason == "single_center_region" and len(peak_ids) != 1:
                continue
            evidence = region & (kernel if reason == "single_center_region" else foreground)
            touched = {int(v) for v in np.unique(labels[evidence]) if v}
            if reason == "single_center_region":
                # A center rescue partition has no kernel pixels of its own.
                # Include only the partition containing this region's sole peak.
                touched.update(int(labels[peaks[i][1], peaks[i][2]]) for i in peak_ids
                               if labels[peaks[i][1], peaks[i][2]] > 0)
            touched = sorted(touched)
            for left, right in combinations(touched, 2):
                if restrict_original and original_ids[left] & original_ids[right]:
                    continue
                candidates.append((left, right, reason, region_id, peak_ids))

    if refinement["kernel_merge"]:
        add_regions(kernel, "shared_kernel_bridge", True)
    if len(p) == 3 and refinement["center_merge"]:
        add_regions((p[2] >= postprocessing["center_threshold"]) & low_support,
                    "single_center_region", False)

    def root(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    expanded = {}
    size = 2 * distance + 1
    for left, right, reason, region_id, peak_ids in candidates:
        a, b = root(left), root(right)
        if a == b:
            continue
        if reason == "shared_kernel_bridge" and group_original_ids[a] & group_original_ids[b]:
            rejected.append({"reason": "original_component_veto", "left": left, "right": right,
                             "evidence": reason, "region_id": region_id})
            continue
        if left not in expanded:
            expanded[left] = cv2.dilate((labels == left).astype(np.uint8), np.ones((size, size), np.uint8)) > 0
        if not np.any(expanded[left] & (labels == right)):
            rejected.append({"reason": "foreground_distance", "left": left, "right": right,
                             "evidence": reason, "region_id": region_id})
            continue
        combined_centers = centers[a] | centers[b] | peak_ids
        if len(combined_centers) > 1:
            rejected.append({"reason": "distinct_centers_in_union", "left": left, "right": right,
                             "evidence": reason, "region_id": region_id})
            continue
        keep, remove = min(a, b), max(a, b)
        parent[remove] = keep
        centers[keep] = combined_centers
        group_original_ids[keep] |= group_original_ids[remove]
        unions.append({"left": left, "right": right, "evidence": reason, "region_id": region_id})

    refined = np.zeros_like(labels)
    groups = {}
    for value in ids:
        representative = root(value)
        refined[labels == value] = representative
        groups.setdefault(representative, []).append(value)
    if not np.array_equal(refined > 0, foreground):
        raise RuntimeError("Refinement changed segmentation foreground")
    return refined, {"initial_instances": len(ids), "final_instances": len(groups),
                     "retained_center_peaks": len(peaks), "unions": unions, "rejected_unions": rejected,
                     "merged_groups": {str(k): v for k, v in groups.items() if len(v) > 1},
                     "foreground_pixels_preserved": True}


def word_contour_crops(probabilities, canvas, image_wh, postprocessing, refinement,
                       *, method, cached_labels=None):
    """Return PARSeq-compatible boxes, per-label contours and an instance map.

    Contours are extracted separately for each ID after refinement. Disconnected
    fragments with one ID produce one union box, while adjacent different IDs
    stay separate. Component area means true foreground pixels, including for
    filtering and sorting; it never means bounding-box or contour polygon area.
    """
    p, canvas, foreground = _inputs(probabilities, canvas, postprocessing)
    if method not in METHODS:
        raise ValueError("Unknown contour method")
    if method == "findcontours":
        labels, audit = findcontour_labels(foreground)
    elif method == "seeded":
        if len(p) < 2:
            raise ValueError("Seeded method requires auxiliary maps")
        labels = _seeded(p, canvas, postprocessing, cached_labels)
        audit = {"method": "existing_seeded_watershed", "cached_labels": cached_labels is not None}
    else:
        labels, audit = refine_instances(p, canvas, postprocessing, refinement, cached_labels=cached_labels)
    proposals = parseq_word_proposals(labels, image_wh, padding=postprocessing["padding"],
        min_area=postprocessing["min_area"], max_instances=postprocessing["max_instances"])
    retained = {row["instance_id"] for row in proposals}
    labels[~np.isin(labels, list(retained))] = 0
    for row in proposals:
        contours, _ = cv2.findContours((labels == row["instance_id"]).astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        row["contours_native"] = [contour[:, 0, :].tolist() for contour in contours]
        row["mask_fragment_contours"] = len(contours)
    return proposals, labels, audit
