"""GT-blind crop alternatives for frozen text-recall experiments (APPROXIMATE)."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from localize.center_guided_word_crops import center_guided_masks, validate_refinement
from localize.crop_recovery import geodesic_partition
from localize.word_guided_contours import findcontour_labels
from localize.word_multitask import center_peaks, parseq_word_proposals


METHODS = ("connected_components", "findcontours", "distance_watershed", "offset_watershed")


def distance_partition(foreground, config):
    """Per-component distance maxima seed the existing foreground-constrained flood.

    Every original foreground pixel survives. No annotation, inferred word count,
    or OCR result can choose markers. One fallback maximum protects tiny shapes.
    This can over-split curved/irregular single words; it is a diagnostic control.
    """
    if foreground.ndim != 2 or foreground.dtype != np.bool_ or not foreground.size:
        raise ValueError("Expected a nonempty boolean foreground map")
    for key in ("min_peak_radius", "relative_peak_height"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Invalid distance marker parameter: " + key)
    if (config["relative_peak_height"] > 1 or not np.isfinite(config["smoothing_sigma"])
        or config["smoothing_sigma"] < 0 or type(config["min_peak_distance"]) is not int
        or config["min_peak_distance"] < 1):
        raise ValueError("Invalid distance smoothing or peak spacing")
    count, components = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
    distance = distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1]
    distance = gaussian_filter(distance, sigma=config["smoothing_sigma"], mode="constant")
    seeds = np.zeros(foreground.shape, np.int32)
    landscape = np.zeros(foreground.shape, np.float32)
    next_id = 1
    for component in range(1, count):
        support = components == component
        maximum = float(distance[support].max())
        if maximum <= 0:
            raise RuntimeError("Nonempty component has no positive distance")
        landscape[support] = distance[support] / maximum
        peaks = center_peaks(distance, support,
            threshold=max(config["min_peak_radius"], config["relative_peak_height"] * maximum),
            min_distance=config["min_peak_distance"])
        if not peaks:
            yy, xx = np.where(support)
            best = int(np.argmax(distance[yy, xx]))
            peaks = [(maximum, int(yy[best]), int(xx[best]))]
        for _, y, x in peaks:
            seeds[y, x] = next_id
            next_id += 1
    labels = geodesic_partition(landscape, foreground, seeds)
    if not np.array_equal(labels > 0, foreground):
        raise RuntimeError("Distance watershed changed foreground support")
    return labels, {"base_components": count - 1, "markers": next_id - 1,
                    "instances": next_id - 1, "added_pixels": 0, "removed_pixels": 0}


def prepare_recipes(probability, offsets, canvas, image_wh, config, *, threshold):
    """Generate all recipes before scoring; deliberately accepts no GT arguments."""
    p, offsets, canvas = map(np.asarray, (probability, offsets, canvas))
    if (p.ndim != 2 or not p.size or offsets.shape != (2, *p.shape)
        or canvas.shape != p.shape or canvas.dtype != np.bool_
        or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
        or not np.isfinite(offsets).all() or np.any(np.abs(offsets) > 1)
        or not np.isfinite(threshold) or not 0 < threshold < 1):
        raise ValueError("Expected aligned finite text/offset/canvas maps and threshold")
    if config["reproduction_mode"] != "approximate" or tuple(config["methods"]) != METHODS:
        raise ValueError("Crop alternatives require explicit APPROXIMATE configuration")
    core = (p >= threshold) & canvas
    count, connected = cv2.connectedComponents(core.astype(np.uint8), connectivity=8)
    contours, contour_audit = findcontour_labels(core)
    distance, distance_audit = distance_partition(core, config["distance"])
    refinement = {**config["offset"], "text_threshold": float(threshold)}
    validate_refinement(refinement)
    if refinement["growth_distance"] != 0:
        raise ValueError("This matched crop study preserves foreground; no mask growth")
    joint = center_guided_masks(p, offsets, canvas, refinement)
    maps = {"connected_components": (connected, {"instances": count - 1}),
            "findcontours": (contours, contour_audit),
            "distance_watershed": (distance, distance_audit),
            "offset_watershed": (joint["preserved_labels"], joint["audit"])}
    return {method: {"labels": labels, "audit": audit,
        "proposals": parseq_word_proposals(labels, image_wh, **config["crop"])}
        for method, (labels, audit) in maps.items()}
