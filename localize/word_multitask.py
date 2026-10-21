"""GT-blind instance separation using existing kernels and minimax watershed."""

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from localize.crop_recovery import geodesic_partition
from segmentation.totaltext_word_kernel import expand_word_kernels, instance_proposals
from segmentation.totaltext_image_frequency_augment import remap_letterbox_native_box


def center_peaks(center, foreground, *, threshold, min_distance):
    """Collapse plateaus, then deterministic Chebyshev NMS by value/y/x."""
    if not isinstance(min_distance, int) or min_distance < 1:
        raise ValueError("Center minimum distance must be a positive integer")
    # Background cannot suppress a foreground peak.
    values = np.where(foreground, center, -np.inf)
    candidates = foreground & (center >= threshold) & (values == maximum_filter(
        values, size=2 * min_distance + 1, mode="constant", cval=-np.inf))
    count, plateaus = cv2.connectedComponents(candidates.astype(np.uint8), connectivity=8)
    peaks = []
    for index in range(1, count):
        yy, xx = np.where(plateaus == index)
        best = int(np.argmax(center[yy, xx]))
        peaks.append((float(center[yy[best], xx[best]]), int(yy[best]), int(xx[best])))
    selected = []
    for peak in sorted(peaks, key=lambda p: (-p[0], p[1], p[2])):
        if all(max(abs(peak[1] - q[1]), abs(peak[2] - q[2])) > min_distance for q in selected):
            selected.append(peak)
    return sorted(selected, key=lambda p: (p[1], p[2]))


def _markers(foreground, kernel, center, config):
    count, components = cv2.connectedComponents(kernel.astype(np.uint8), connectivity=8)
    if center is None:
        return components, {"center_peaks": 0, "kernel_components_split_by_centers": 0,
                            "center_rescue_markers": 0}
    peaks = center_peaks(center, foreground, threshold=config["center_threshold"],
                         min_distance=config["center_min_distance"])
    markers = np.zeros(foreground.shape, np.int32)
    next_id, split = 1, 0
    for index in range(1, count):
        support = components == index
        inside = [p for p in peaks if support[p[1], p[2]]]
        if len(inside) < 2:
            markers[support] = next_id
            next_id += 1
        else:
            split += 1
            seeds = np.zeros_like(markers)
            for _, y, x in inside:
                seeds[y, x] = next_id
                next_id += 1
            local = geodesic_partition(center, support, seeds)
            markers[support] = local[support]
    rescued = 0
    for _, y, x in peaks:
        if markers[y, x]:
            continue
        yy, xx = np.where(markers > 0)
        distance = np.maximum(np.abs(yy - y), np.abs(xx - x))
        if len(distance) and int(distance.min()) <= config["center_min_distance"]:
            continue
        markers[y, x] = next_id
        next_id += 1
        rescued += 1
    return markers, {"center_peaks": len(peaks), "kernel_components_split_by_centers": split,
                     "center_rescue_markers": rescued}


def word_instances(probabilities, canvas, config):
    """Return distinct integer IDs, audit and text support. Never consumes GT.

    connected_components ignores auxiliary maps as a diagnostic control.
    kernel_bfs reuses the historical expander for two maps, and the existing
    geodesic flood at constant cost for labeled three-map markers.
    watershed uses the same flood on a configured confidence landscape.
    Every unseeded text component survives as a fallback instance.
    """
    p, canvas = np.asarray(probabilities), np.asarray(canvas)
    if (p.ndim != 3 or p.shape[0] not in (1, 2, 3) or p.shape[1:] != canvas.shape
        or canvas.dtype != np.bool_ or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Expected finite CxHxW probabilities and paired boolean RGB canvas")
    for key in ("text_threshold", "kernel_threshold", "center_threshold"):
        if not np.isfinite(config[key]) or not 0 < config[key] < 1:
            raise ValueError("Invalid threshold: " + key)
    method = config["method"]
    if method not in ("connected_components", "kernel_bfs", "watershed"):
        raise ValueError("Unknown word postprocessing method")
    foreground = (p[0] >= config["text_threshold"]) & canvas
    if len(p) == 1 or method == "connected_components":
        count, labels = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
        return labels, {"instances": count - 1, "seed_components": 0,
                        "unseeded_fallback_components": count - 1}, foreground
    kernel = (p[1] >= config["kernel_threshold"]) & foreground
    if len(p) == 2 and method == "kernel_bfs":
        labels, audit = expand_word_kernels(foreground, kernel)
        return labels, audit, foreground
    markers, audit = _markers(foreground, kernel, p[2] if len(p) == 3 else None, config)
    if method == "kernel_bfs":
        landscape = np.ones(foreground.shape, np.float32)
    else:
        weights = np.asarray(config["watershed_weights"][:len(p)], dtype=np.float64)
        if len(weights) != len(p) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Invalid watershed landscape weights")
        landscape = np.einsum("c,cyx->yx", weights / weights.sum(), p)
    labels = geodesic_partition(landscape, foreground, markers)
    missing = foreground & (labels == 0)
    count, fallback = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
    labels[missing] = fallback[missing] + int(markers.max())
    if not np.array_equal(labels > 0, foreground):
        raise RuntimeError("Instance partition changed text support")
    audit.update(instances=len(np.unique(labels[labels > 0])),
                 seed_components=len(np.unique(markers[markers > 0])),
                 unseeded_fallback_components=count - 1)
    return labels, audit, foreground


def parseq_word_proposals(labels, image_wh, *, padding, min_area, max_instances):
    """Exclusive original-coordinate xyxy boxes and true component areas.

    Output bbox matches existing PARSeq crop callers. Adjacent IDs stay separate.
    Filtering and padding are explicit experimental config choices.
    """
    if not isinstance(padding, int) or padding < 0 or min_area < 0:
        raise ValueError("Invalid word box padding/area")
    if max_instances is not None and (not isinstance(max_instances, int) or max_instances < 1):
        raise ValueError("Invalid instance limit")
    if labels.shape[0] != labels.shape[1]:
        raise ValueError("Existing letterbox remapping requires square native maps")
    size = labels.shape[0]
    proposals = [p for p in instance_proposals(labels) if p["foreground_area"] >= min_area]
    if max_instances is not None:
        proposals = proposals[:max_instances]
    for proposal in proposals:
        x1, y1, x2, y2 = proposal["bbox_native"]
        padded = [max(0, x1 - padding), max(0, y1 - padding),
                  min(size, x2 + padding), min(size, y2 + padding)]
        proposal["bbox_original"] = remap_letterbox_native_box(
            proposal["bbox_native"], image_wh, native_size=size)
        proposal["bbox"] = remap_letterbox_native_box(padded, image_wh, native_size=size)
    return proposals
