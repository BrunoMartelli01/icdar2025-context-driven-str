"""Small GT-blind ablations on frozen word/kernel probabilities (approximate)."""

from math import ceil

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F

from localize.crop_recovery import geodesic_partition
from localize.word_guided_contours import refine_instances
from localize.word_multitask import parseq_word_proposals, word_instances


METHODS = ("baseline", "refined", "bounded_growth", "soft_fallback", "seeded_only")


def resize_probability(probability, size):
    value = np.asarray(probability)
    if (value.ndim != 2 or not value.size or not np.isfinite(value).all()
            or np.any((value < 0) | (value > 1)) or type(size) is not int or size < 1):
        raise ValueError("Expected finite probability map and positive size")
    if value.shape == (size, size):
        return value.astype(np.float32, copy=True)
    return F.interpolate(torch.from_numpy(value.astype(np.float32))[None, None],
                         (size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def filter_fallbacks(labels, kernel, *, seed_threshold, fallback_threshold):
    """Keep seeded labels plus fallback labels with declared near-kernel evidence."""
    if (labels.shape != kernel.shape or labels.dtype.kind not in "iu" or labels.min() < 0
            or not np.isfinite(kernel).all() or np.any((kernel < 0) | (kernel > 1))
            or not 0 < fallback_threshold <= seed_threshold < 1):
        raise ValueError("Invalid fallback probabilities/labels/thresholds")
    result = labels.copy()
    removed, retained_fallbacks = [], []
    for value in np.unique(labels[labels > 0]):
        values = kernel[labels == value]
        score = float(values.max())
        if not np.any(values >= fallback_threshold):
            result[labels == value] = 0
            removed.append({"instance_id": int(value), "maximum_kernel": score})
        elif not np.any(values >= seed_threshold):
            retained_fallbacks.append({"instance_id": int(value), "maximum_kernel": score})
    return result, {"removed_instances": removed, "retained_fallbacks": retained_fallbacks,
                    "seed_threshold": seed_threshold, "fallback_threshold": fallback_threshold}


def bounded_growth(probability, labels, canvas, *, low_threshold, radius):
    """Expand existing IDs on low-confidence support within a fixed native radius."""
    if (probability.shape != labels.shape or canvas.shape != labels.shape
            or canvas.dtype != np.bool_ or labels.dtype.kind not in "iu" or labels.min() < 0
            or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1))
            or not np.isfinite(radius) or radius <= 0 or not 0 < low_threshold < 1
            or np.any((labels > 0) & (~canvas | (probability < low_threshold)))):
        raise ValueError("Invalid growth support, seed labels, radius or threshold")
    if not np.any(labels):
        return labels.copy(), {"added_pixels": 0, "radius": radius}
    distance = distance_transform_edt(labels == 0)
    support = canvas & (probability >= low_threshold) & (distance <= radius)
    result = geodesic_partition(probability, support, labels)
    if not np.array_equal(result[labels > 0], labels[labels > 0]):
        raise RuntimeError("Growth changed an original word ID")
    return result, {"added_pixels": int(np.count_nonzero((result > 0) & (labels == 0))),
                    "radius": radius, "low_threshold": low_threshold}


def prepare_recipes(probabilities, canvas, image_wh, config, *, text_threshold):
    """Inference-only inputs: no targets, annotations, validity, strings or counts."""
    post = {**config["postprocessing"], "text_threshold": text_threshold}
    base, meta, foreground = word_instances(probabilities, canvas, post)
    scale = canvas.shape[0] / config["canonical_size"]
    refinement = {**config["refinement"], "maximum_foreground_distance":
                  ceil(config["refinement"]["maximum_foreground_distance"] * scale)}
    refined, refined_meta = refine_instances(probabilities, canvas, post, refinement,
                                             cached_labels=base)
    grown, growth_meta = bounded_growth(probabilities[0], base, canvas,
        low_threshold=config["growth"]["low_threshold"], radius=config["growth"]["radius_at256"] * scale)
    soft, soft_meta = filter_fallbacks(base, probabilities[1],
        seed_threshold=post["kernel_threshold"], fallback_threshold=config["fallback_kernel_threshold"])
    seeded, seeded_meta = filter_fallbacks(base, probabilities[1],
        seed_threshold=post["kernel_threshold"], fallback_threshold=post["kernel_threshold"])
    if not np.array_equal(refined > 0, foreground):
        raise RuntimeError("Refinement changed foreground")
    maps = {"baseline": (base, meta), "refined": (refined, refined_meta),
            "bounded_growth": (grown, growth_meta), "soft_fallback": (soft, soft_meta),
            "seeded_only": (seeded, seeded_meta)}
    return {method: {"labels": labels, "audit": audit,
                    "proposals": parseq_word_proposals(labels, image_wh, **config["crop"])}
            for method, (labels, audit) in maps.items()}
