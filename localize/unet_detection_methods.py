"""GT-blind progressive expansion and original-RGB word rectification."""

import cv2
import numpy as np

from diagnostics.fixed_group_curve_geometry import fit, render
from localize.crop_recovery import geodesic_partition
from localize.word_multitask import word_instances


def detection_instances(probabilities, canvas, config):
    p = np.asarray(probabilities)
    if p.shape[0] == 2:
        return word_instances(p, canvas, config["postprocessing"])[:2]
    if (p.ndim != 3 or p.shape[0] != 4 or p.shape[1:] != canvas.shape or canvas.dtype != bool
            or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))):
        raise ValueError("Finite aligned text and three kernel probabilities required")
    pp = config["postprocessing"]
    foreground = (p[0] >= pp["text_threshold"]) & canvas
    smallest = (p[1] >= pp["kernel_threshold"]) & foreground
    count, labels = cv2.connectedComponents(smallest.astype(np.uint8), connectivity=8)
    for index in (2, 3):
        support = foreground & ((p[index] >= pp["kernel_threshold"]) | (labels > 0))
        labels = geodesic_partition(p[index], support, labels)
    weights = np.asarray(pp["watershed_weights"][:2], dtype=np.float64)
    if len(weights) != 2 or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Invalid progressive watershed weights")
    landscape = np.einsum("c,cyx->yx", weights / weights.sum(), p[[0, 3]])
    labels = geodesic_partition(landscape, foreground, labels)
    missing = foreground & (labels == 0)
    fallbacks, components = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
    labels[missing] = components[missing] + int(labels.max())
    if not np.array_equal(labels > 0, foreground):
        raise RuntimeError("Progressive expansion changed full-word foreground")
    return labels, {"instances": len(np.unique(labels[labels > 0])), "seed_components": count - 1,
                    "unseeded_fallback_components": fallbacks - 1, "kernel_scales": 3}


def rectified_word_crop(rgb, labels, proposal, kind, config):
    """Geometry/GT matching retains the original box for EVERY crop variant."""
    rgb = np.asarray(rgb)
    x0, y0, x1, y1 = proposal["bbox"]
    baseline = np.ascontiguousarray(rgb[y0:y1, x0:x1])
    if kind == "axis":
        return baseline, {"kind": kind, "reason": "baseline", "spec": None}
    if kind not in ("oriented", "bezier"):
        raise ValueError("Unknown word crop view")
    # Inverse letterbox first; resize only the cropped support instead of
    # materializing each entire original-resolution instance raster.
    bx0, by0, bx1, by1 = proposal["bbox_native"]
    own = (labels[by0:by1, bx0:bx1] == proposal["instance_id"]).astype(np.uint8)
    if not own.any() or x1 <= x0 or y1 <= y0:
        raise ValueError("Empty or inconsistent word instance crop")
    own = cv2.resize(own, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST).astype(bool)
    spec, reason = fit(own, (x0, y0), rgb.shape, kind.upper(), config["rectification"])
    if spec is None:
        return baseline, {"kind": kind, "reason": reason, "spec": None, "fallback": "axis"}
    return render(rgb, spec), {"kind": kind, "reason": reason, "spec": spec}
