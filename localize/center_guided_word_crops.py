"""Opt-in approximate shape/center fusion; no annotations enter this module."""

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from localize.crop_recovery import geodesic_partition
from localize.word_center_offsets import _splat
from localize.word_multitask import center_peaks, parseq_word_proposals


def validate_refinement(config):
    if config["reproduction_mode"] != "approximate":
        raise ValueError("Center-guided crop refinement is an approximate extension, forbidden in STRICT")
    for key in ("text_threshold", "growth_text_threshold", "smoothing_sigma", "min_peak_density", "assignment_radius"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Invalid center refinement parameter: " + key)
    if not 0 < config["growth_text_threshold"] <= config["text_threshold"] < 1:
        raise ValueError("Require 0 < growth threshold <= text threshold < 1")
    for key in ("min_center_distance", "min_center_support", "growth_distance"):
        if type(config[key]) is not int or config[key] < (0 if key == "growth_distance" else 1):
            raise ValueError("Invalid integer refinement parameter: " + key)


def _nearest_centers(vx, vy, centers, valid):
    distance = np.full(vx.shape, np.inf)
    ids = np.zeros(vx.shape, np.int32)
    for label, (_, y, x) in enumerate(centers, 1):
        candidate = (vx-x)**2 + (vy-y)**2
        closer = valid & (candidate < distance)
        distance[closer] = candidate[closer]
        ids[closer] = label
    return ids, distance


def center_guided_masks(text_probability, normalized_offsets, canvas, config):
    """Global endpoint consensus joins fragments, splits fused words, and bounds growth.

    High-confidence base text always survives. Low-confidence pixels may join
    only a reachable center identity supported by their own predicted endpoint,
    within the explicit distance from the original high-confidence support.
    A center is a hypothesis, not proof that a word exists.
    """
    validate_refinement(config)
    p, offsets, canvas = map(np.asarray, (text_probability, normalized_offsets, canvas))
    if (p.ndim != 2 or not p.size or offsets.shape != (2, *p.shape) or canvas.shape != p.shape
        or canvas.dtype != np.bool_ or not np.isfinite(p).all() or np.any((p < 0) | (p > 1))
        or not np.isfinite(offsets).all() or np.any(np.abs(offsets) > 1)):
        raise ValueError("Expected paired finite text probabilities, normalized xy offsets and boolean canvas")
    h, w = p.shape
    yy, xx = np.indices(p.shape)
    vx = xx + offsets[0].astype(np.float64) * w
    vy = yy + offsets[1].astype(np.float64) * h
    valid = (vx >= 0) & (vx <= w-1) & (vy >= 0) & (vy <= h-1)
    iy, ix = np.where(valid)
    valid[iy, ix] &= canvas[np.rint(vy[iy, ix]).astype(int), np.rint(vx[iy, ix]).astype(int)]
    core = (p >= config["text_threshold"]) & canvas
    voters = core & valid
    density = _splat(vx[voters], vy[voters], p[voters], p.shape)
    density = gaussian_filter(density, sigma=config["smoothing_sigma"], mode="constant", cval=0)
    density[~canvas] = 0
    peaks = sorted(center_peaks(density, canvas, threshold=config["min_peak_density"],
                   min_distance=config["min_center_distance"]), key=lambda peak: (-peak[0], peak[1], peak[2]))
    candidate_count = len(peaks)
    initial_ids, distance = _nearest_centers(vx, vy, peaks, valid)
    supported = core & valid & (distance <= config["assignment_radius"]**2)
    peaks = [peak for i, peak in enumerate(peaks, 1)
             if np.count_nonzero(supported & (initial_ids == i)) >= config["min_center_support"]]
    affinity, distance = _nearest_centers(vx, vy, peaks, valid)
    consistent = valid & (distance <= config["assignment_radius"]**2)
    seeds = np.where(core & consistent, affinity, 0).astype(np.int32)
    records = []
    for label, (mass, y, x) in enumerate(peaks, 1):
        sy, sx = np.where(seeds == label)
        if not len(sx):
            raise RuntimeError("Retained center lost all its high-confidence voters")
        nearest = int(np.argmin((sx-x)**2 + (sy-y)**2))
        records.append({"center_id": label, "vote_peak_xy_native": [x, y], "peak_density": mass,
                        "marker_xy_native": [int(sx[nearest]), int(sy[nearest])], "seed_pixels": len(sx)})
    preserved = geodesic_partition(p, core, seeds)
    missing = core & (preserved == 0)
    count, fallback = cv2.connectedComponentsWithAlgorithm(missing.astype(np.uint8), 8, cv2.CV_32S, cv2.CCL_WU)
    preserved[missing] = fallback[missing] + len(peaks)
    expanded = preserved.copy()
    eligible = (p >= config["growth_text_threshold"]) & canvas & consistent & ~core
    eligible &= distance_transform_edt(~core) <= config["growth_distance"]
    # Synchronous, identity-constrained growth. Different endpoint identities
    # never compete for a pixel; fallback regions without a cue do not expand.
    neighborhood = np.ones((3, 3), np.uint8)
    for _ in range(config["growth_distance"]):
        additions = np.zeros_like(expanded)
        for label in range(1, len(peaks)+1):
            reach = cv2.dilate((expanded == label).astype(np.uint8), neighborhood) > 0
            select = reach & eligible & (expanded == 0) & (affinity == label)
            additions[select] = label
        if not additions.any():
            break
        expanded[additions > 0] = additions[additions > 0]
    if not np.array_equal(preserved > 0, core) or np.any(core & (expanded == 0)):
        raise RuntimeError("Center refinement removed high-confidence base pixels")
    if np.any((expanded > 0) & ~canvas):
        raise RuntimeError("Refinement escaped the RGB canvas")
    return {"preserved_labels": preserved, "expanded_labels": expanded, "centers": records,
            "density": density, "core": core, "seeds": seeds,
            "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "audit": {"candidate_peaks": candidate_count,
                      "retained_centers": len(peaks), "unseeded_fallback_components": count-1,
                      "base_components": cv2.connectedComponents(core.astype(np.uint8), connectivity=8)[0]-1,
                      "rejected_core_endpoint_votes": int((core & ~valid).sum()),
                      "preserved_instances": len(np.unique(preserved[preserved > 0])),
                      "expanded_instances": len(np.unique(expanded[expanded > 0])),
                      "added_pixels": int(((expanded > 0) & ~core).sum()), "removed_core_pixels": 0}}


def center_guided_crops(labels, image_wh, crop_config):
    """One original RGB crop per ID, including disconnected same-center fragments."""
    return parseq_word_proposals(labels, image_wh, padding=crop_config["padding"],
        min_area=crop_config["min_area"], max_instances=crop_config["max_instances"])
