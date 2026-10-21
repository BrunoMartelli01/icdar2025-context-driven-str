"""Location cues from dense center voting, constrained by the classic U-Net mask."""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from localize.crop_recovery import geodesic_partition
from localize.word_multitask import center_peaks


def validate_vote_config(config):
    for key in ("text_threshold", "smoothing_sigma", "min_peak_density", "assignment_radius"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError("Invalid center voting setting: " + key)
    if config["text_threshold"] >= 1:
        raise ValueError("Text threshold must be in (0,1)")
    if (type(config["min_center_distance"]) is not int or config["min_center_distance"] < 1):
        raise ValueError("Center voting requires a positive integer NMS radius")


def _splat(vote_x, vote_y, weights, shape):
    """Bilinear subpixel votes; do not clamp out-of-canvas votes onto borders."""
    density = np.zeros(shape, np.float64)
    x0, y0 = np.floor(vote_x).astype(int), np.floor(vote_y).astype(int)
    fx, fy = vote_x - x0, vote_y - y0
    for dx, dy, fraction in ((0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                             (0, 1, (1-fx)*fy), (1, 1, fx*fy)):
        x, y = x0 + dx, y0 + dy
        valid = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
        np.add.at(density, (y[valid], x[valid]), weights[valid] * fraction[valid])
    return density


def word_center_look_signal(text_probability, normalized_offsets, canvas, config):
    """Return a vote map and foreground markers; this function produces no crops.

    Only pixels in the BASE network's text support vote. Component-local voting
    prevents one disconnected text region from borrowing another's centers.
    Peak positions are cues, possibly in a concavity; project each marker into
    its own text component with row-major distance ties. Keep unseeded regions.
    """
    validate_vote_config(config)
    text, offsets, canvas = map(np.asarray, (text_probability, normalized_offsets, canvas))
    if (text.ndim != 2 or not text.size or offsets.shape != (2, *text.shape)
        or canvas.shape != text.shape or canvas.dtype != np.bool_
        or not np.isfinite(text).all() or np.any((text < 0) | (text > 1))
        or not np.isfinite(offsets).all() or np.any(np.abs(offsets) > 1)):
        raise ValueError("Expected finite HxW text probabilities, normalized 2xHxW x/y offsets and boolean canvas")
    foreground = (text >= config["text_threshold"]) & canvas
    count, components = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
    height, width = text.shape
    density = np.zeros(text.shape, np.float64)
    markers = np.zeros(text.shape, np.int32)
    ownership_seeds = np.zeros(text.shape, np.int32)
    centers, next_id, rejected, unseeded, collisions = [], 1, 0, 0, 0
    for component in range(1, count):
        yy, xx = np.where(components == component)
        vx = xx + offsets[0, yy, xx].astype(np.float64) * width
        vy = yy + offsets[1, yy, xx].astype(np.float64) * height
        valid = (vx >= 0) & (vx <= width-1) & (vy >= 0) & (vy <= height-1)
        # Check the predicted endpoint against the RGB canvas only, never GT validity.
        index = np.flatnonzero(valid)
        valid[index] &= canvas[np.rint(vy[index]).astype(int), np.rint(vx[index]).astype(int)]
        rejected += int((~valid).sum())
        local = _splat(vx[valid], vy[valid], text[yy[valid], xx[valid]], text.shape)
        local = gaussian_filter(local, sigma=config["smoothing_sigma"], mode="constant", cval=0)
        local[~canvas] = 0
        density += local
        peaks = center_peaks(local, canvas, threshold=config["min_peak_density"],
                             min_distance=config["min_center_distance"])
        accepted, component_centers = 0, []
        # Prefer stronger cues if different peaks project to the same text pixel.
        for mass, y, x in sorted(peaks, key=lambda peak: (-peak[0], peak[1], peak[2])):
            nearest = int(np.argmin((xx-x)**2 + (yy-y)**2))
            my, mx = int(yy[nearest]), int(xx[nearest])
            if markers[my, mx]:
                collisions += 1
                continue
            markers[my, mx] = next_id
            centers.append({"marker_id": next_id, "base_component": component,
                            "vote_peak_xy_native": [x, y], "marker_xy_native": [mx, my],
                            "peak_density": mass})
            component_centers.append((next_id, x, y))
            next_id += 1
            accepted += 1
        # Dense agreement is additional guidance, not a new foreground contour.
        # Stream over centers rather than allocate an H*W*center_count tensor.
        nearest_distance = np.full(len(xx), np.inf)
        nearest_label = np.zeros(len(xx), np.int32)
        for marker_id, x, y in component_centers:
            distance = (vx-x)**2 + (vy-y)**2
            closer = valid & (distance < nearest_distance)
            nearest_distance[closer] = distance[closer]
            nearest_label[closer] = marker_id
        consistent = valid & (nearest_distance <= config["assignment_radius"]**2)
        ownership_seeds[yy[consistent], xx[consistent]] = nearest_label[consistent]
        unseeded += int(accepted == 0)
    ownership_seeds[markers > 0] = markers[markers > 0]
    return {"reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "vote_density": density, "markers": markers, "centers": centers,
            "ownership_seeds": ownership_seeds,
            "foreground": foreground,
            "audit": {"base_components": count-1, "center_markers": len(centers),
                      "rejected_endpoint_votes": rejected, "projected_marker_collisions": collisions,
                      "consistent_vote_seed_pixels": int((ownership_seeds > 0).sum()),
                      "unseeded_base_components": unseeded}}


def split_text_with_center_offsets(text_probability, normalized_offsets, canvas, config):
    """Optional split: base probabilities give shape/cost, voted centers give seeds.

    Reuse the deterministic foreground-constrained flood. Never threshold the
    vote map as a word contour. No text pixel is erased or added; absent cues
    retain the entire base component. Keep integer instance IDs for downstream
    cropping, because converting them to binary would undo the split.
    """
    signal = word_center_look_signal(text_probability, normalized_offsets, canvas, config)
    foreground, markers = signal["foreground"], signal["ownership_seeds"]
    labels = geodesic_partition(np.asarray(text_probability), foreground, markers)
    missing = foreground & (labels == 0)
    count, fallback = cv2.connectedComponents(missing.astype(np.uint8), connectivity=8)
    labels[missing] = fallback[missing] + int(markers.max())
    if not np.array_equal(labels > 0, foreground):
        raise RuntimeError("Center guidance changed the base U-Net text support")
    signal["audit"].update(instances=len(np.unique(labels[labels > 0])),
                           unseeded_fallback_components=count-1)
    return labels, signal

