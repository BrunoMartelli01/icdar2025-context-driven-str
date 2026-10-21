"""GT-blind line markers and line/radius instance reconstruction."""

import cv2
import numpy as np

from localize.crop_recovery import geodesic_partition
from localize.word_multitask import word_instances
from segmentation.word_centerline import skeletonize_mask


METHODS = ("baseline", "line_markers", "line_radius")


def centerline_instances(evidence, canvas, post_config, method):
    """Decode probabilities/radii without access to GT or an owner map.

    line_markers preserves the frozen foreground with fallback for unseeded
    components. line_radius rasterizes predicted disks independently of that
    foreground. Overlapping disk owners compete by d/r; ties keep row-major
    component order. Labels stay distinct even when their reconstructed regions
    touch. Radii clipping is explicit and its count appears in the audit.
    """
    if method not in METHODS:
        raise ValueError("Unknown centerline decoding method")
    words, line, radius = (np.asarray(evidence[key]) for key in ("words", "line", "radius"))
    canvas = np.asarray(canvas)
    if (words.ndim != 3 or words.shape[0] != 2 or line.ndim != 2
            or words.shape[1:] != line.shape or radius.shape != line.shape or canvas.shape != line.shape
            or canvas.dtype != np.bool_ or not all(np.isfinite(value).all() for value in (words, line, radius))
            or np.any((words < 0) | (words > 1)) or np.any((line < 0) | (line > 1)) or np.any(radius < 0)):
        raise ValueError("Expected paired finite word/line probabilities, nonnegative radii and RGB canvas")
    threshold, floor, maximum = (post_config[key] for key in ("line_threshold", "radius_floor", "max_radius"))
    minimum = post_config["min_line_area"]
    if (not np.isfinite([threshold, floor, maximum]).all() or not 0 < threshold < 1
            or floor <= 0 or maximum < floor or type(minimum) is not int or minimum < 0):
        raise ValueError("Invalid line threshold/radius/area configuration")
    if method == "baseline":
        return word_instances(words, canvas, post_config)
    foreground = (words[0] >= post_config["text_threshold"]) & canvas
    mask = (line >= threshold) & canvas
    if method == "line_markers":
        mask &= foreground
    count, labels, stats, _ = cv2.connectedComponentsWithStatsWithAlgorithm(
        mask.astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)
    markers = np.zeros(line.shape, np.int32)
    next_id, filtered = 1, 0
    for component in range(1, count):
        if int(stats[component, cv2.CC_STAT_AREA]) < minimum:
            filtered += 1
        else:
            markers[labels == component] = next_id
            next_id += 1
    audit = {"method": method, "line_components_before_filter": count - 1,
             "filtered_line_components": filtered, "seed_components": next_id - 1,
             "line_pixels": int(np.count_nonzero(markers)), "unseeded_fallback_components": 0,
             "radius_clipped_pixels": 0}
    if method == "line_markers":
        instances = geodesic_partition(words[0], foreground, markers)
        missing = foreground & (instances == 0)
        missing_count, fallback = cv2.connectedComponentsWithAlgorithm(missing.astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)
        instances[missing] = fallback[missing] + next_id - 1
        if not np.array_equal(instances > 0, foreground):
            raise RuntimeError("Centerline marker expansion changed source foreground")
        audit["unseeded_fallback_components"] = missing_count - 1
    else:
        height, width = line.shape
        instances = np.zeros(line.shape, np.int32)
        best_distance = np.full(line.shape, np.inf, np.float64)
        skeleton_pixels = 0
        conflicts = np.zeros(line.shape, bool)
        for component in range(1, next_id):
            skeleton = skeletonize_mask(markers == component)
            skeleton_pixels += int(skeleton.sum())
            for y, x in np.argwhere(skeleton):
                raw_radius = float(radius[y, x])
                r = min(maximum, max(floor, raw_radius))
                audit["radius_clipped_pixels"] += int(raw_radius != r)
                extent = int(np.ceil(r))
                y0, y1 = max(0, int(y) - extent), min(height, int(y) + extent + 1)
                x0, x1 = max(0, int(x) - extent), min(width, int(x) + extent + 1)
                yy, xx = np.ogrid[y0:y1, x0:x1]
                distance = ((yy - y) ** 2 + (xx - x) ** 2) / r ** 2
                active = (distance <= 1) & canvas[y0:y1, x0:x1]
                previous = instances[y0:y1, x0:x1]
                conflicts[y0:y1, x0:x1] |= active & (previous > 0) & (previous != component)
                closer = active & (distance < best_distance[y0:y1, x0:x1])
                previous[closer] = component
                best_distance[y0:y1, x0:x1][closer] = distance[closer]
        audit.update(skeleton_pixels=skeleton_pixels, competing_disk_pixels=int(conflicts.sum()))
        foreground = instances > 0
    audit["instances"] = len(np.unique(instances[instances > 0]))
    return instances, audit, foreground
