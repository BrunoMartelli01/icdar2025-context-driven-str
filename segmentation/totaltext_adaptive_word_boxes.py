"""Filled, piecewise word-box TRAIN targets for Total-Text.

PRACTICAL_APPROXIMATION: original pixel GT positions the edges of broad
longitudinal strips; it is never dilated into letter-shaped targets. Every
strip stays inside the already disjoint v1 owner, so words cannot overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from PIL import Image
from scipy.ndimage import label


@dataclass(frozen=True)
class AdaptiveBoxAudit:
    words: int
    changed_words: int
    no_seed_fallbacks: int
    aspect_fallbacks: int
    thin_or_fragmented_fallbacks: int
    base_foreground_pixels: int
    box_foreground_pixels: int
    source_seed_pixels: int
    retained_source_seed_pixels: int
    mean_bins_per_changed_word: float


def _fit_one_box(owner: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Fit one filled strip envelope, or retain v1 if it would be unsafe.

    The fixed geometric safeguards precede evaluation: each strip is about a
    local word height wide; its fill is at least 70% of that height; a result
    keeping under 65% of the v1 area is rejected as letter-like.
    """

    rows, cols = owner.shape
    vertical = rows > cols
    if max(rows, cols) < 1.25 * min(rows, cols):
        return None
    region = owner.T if vertical else owner
    pixels = seed.T if vertical else seed
    thickness = np.count_nonzero(region, axis=0)
    positive = thickness[thickness > 0]
    if not positive.size or not pixels.any():
        return None
    reference_height = float(np.median(positive))
    length = region.shape[1]
    bin_count = min(12, max(4, int(ceil(length / max(3.0, reference_height)))))
    edges = np.rint(np.linspace(0, length, bin_count + 1)).astype(int)
    if len(set(edges)) != len(edges):
        return None
    top = np.full(bin_count, np.nan)
    bottom = np.full(bin_count, np.nan)
    margin = max(1, int(ceil(0.15 * reference_height)))
    minimum_height = max(2, int(ceil(0.70 * reference_height)))
    for index in range(bin_count):
        ys = np.flatnonzero(pixels[:, edges[index] : edges[index + 1]].any(axis=1))
        if not ys.size:
            continue
        lo, hi = int(ys.min()), int(ys.max()) + 1
        desired = max(minimum_height, hi - lo + 2 * margin)
        center = (lo + hi) / 2
        top[index] = int(np.floor(center - desired / 2))
        bottom[index] = int(np.ceil(center + desired / 2))
    known = np.flatnonzero(np.isfinite(top))
    if not known.size:
        return None
    for boundary in (top, bottom):
        boundary[:] = np.interp(np.arange(bin_count), known, boundary[known])
    result = np.zeros_like(region)
    for index in range(bin_count):
        lo = max(0, int(np.floor(top[index])))
        hi = min(region.shape[0], int(np.ceil(bottom[index])))
        result[lo:hi, edges[index] : edges[index + 1]] = True
    result &= region
    if np.any(pixels & ~result) or result.sum() < 0.65 * region.sum():
        return None
    structure = np.ones((3, 3), dtype=np.uint8)
    if label(result, structure)[1] > label(region, structure)[1]:
        return None
    return (result.T if vertical else result), bin_count


def fit_adaptive_word_boxes(
    base_owners: Image.Image, source_pixel_gt: Image.Image, validity: Image.Image
) -> tuple[Image.Image, Image.Image, AdaptiveBoxAudit]:
    """Return target, owner map and audit; never change v1 ownership."""

    base = np.asarray(base_owners, dtype=np.uint16)
    pixel = np.asarray(source_pixel_gt.convert("L")) > 0
    valid = np.asarray(validity.convert("L")) > 0
    if base.shape != pixel.shape or base.shape != valid.shape:
        raise ValueError("Owner, pixel GT and validity geometry must match")
    if np.any((base > 0) & ~valid):
        raise ValueError("Base owner map contains ignored pixels")
    output = base.copy()
    ids = np.unique(base)
    ids = ids[ids > 0]
    no_seed = aspect = unsafe = changed = bins = 0
    for word_id in ids:
        ys, xs = np.where(base == word_id)
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
        owner = base[y0:y1, x0:x1] == word_id
        seed = owner & pixel[y0:y1, x0:x1]
        if not seed.any():
            no_seed += 1
            continue
        if max(owner.shape) < 1.25 * min(owner.shape):
            aspect += 1
            continue
        fitted = _fit_one_box(owner, seed)
        if fitted is None:
            unsafe += 1
            continue
        region, count = fitted
        if not np.array_equal(region, owner):
            output[y0:y1, x0:x1][owner & ~region] = 0
            changed += 1
            bins += count
    source_seed = pixel & (base > 0)
    if np.any(source_seed & (output == 0)):
        raise ValueError("Adaptive word box lost source pixel-GT seeds")
    audit = AdaptiveBoxAudit(
        words=len(ids), changed_words=changed, no_seed_fallbacks=no_seed,
        aspect_fallbacks=aspect, thin_or_fragmented_fallbacks=unsafe,
        base_foreground_pixels=int((base > 0).sum()),
        box_foreground_pixels=int((output > 0).sum()),
        source_seed_pixels=int(source_seed.sum()),
        retained_source_seed_pixels=int((source_seed & (output > 0)).sum()),
        mean_bins_per_changed_word=bins / changed if changed else 0.0,
    )
    return (
        Image.fromarray((output > 0).astype(np.uint8) * 255, mode="L"),
        Image.fromarray(output, mode="I;16"), audit,
    )
