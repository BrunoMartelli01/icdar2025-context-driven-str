"""Smooth, filled Total-Text word polygons for a separate TRAIN target.

PRACTICAL_APPROXIMATION: interpolate the two annotated long sides of each
curved word as continuous PCHIP curves, then fill the polygon. A uniform
outward *box* buffer is used only when needed to retain source text pixels;
no letter-shaped dilation/erosion is used. V1 is the deterministic fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter1d
from scipy.spatial import cKDTree

from recognition.scene_text_datasets import SceneTextInstance
from segmentation.totaltext_word_regions import (
    _native_word_component_counts, make_word_regions,
)


SAMPLES_PER_SEGMENT = 6
MAX_BUFFER_FRACTION_OF_WORD_THICKNESS = 0.20
MAX_AREA_RATIO_TO_V1 = 1.30


@dataclass(frozen=True)
class CurvedBoxAudit:
    words: int
    curved_annotations: int
    changed_words: int
    geometric_fallbacks: int
    coverage_or_topology_fallbacks: int
    full_image_fallback: bool
    max_uniform_buffer_pixels: int
    base_foreground_pixels: int
    final_foreground_pixels: int
    v1_source_seed_pixels: int
    retained_v1_source_seed_pixels: int


def smooth_curved_polygon(
    points: tuple[tuple[int, int], ...], size_wh: tuple[int, int]
) -> tuple[tuple[int, int], ...]:
    """Interpolate upper/lower annotation sides; keep straight boxes unchanged.

    Total-Text curved polygons conventionally list the first long side and
    the reverse long side. Invalid/odd topology is not guessed.
    """

    n = len(points)
    if n < 6 or n % 2:
        return points
    source = np.asarray(points, dtype=np.float64)
    sides = []
    for side in (source[: n // 2], source[n // 2 :]):
        distance = np.linalg.norm(np.diff(side, axis=0), axis=1)
        if np.any(distance <= 0):
            return points
        arc = np.concatenate(([0.0], np.cumsum(distance)))
        locations = np.linspace(0.0, arc[-1], SAMPLES_PER_SEGMENT * (len(side) - 1) + 1)
        sides.append(np.column_stack((
            PchipInterpolator(arc, side[:, 0])(locations),
            PchipInterpolator(arc, side[:, 1])(locations),
        )))
    outline = np.vstack((sides[0][:-1], sides[1][:-1]))
    width, height = size_wh
    outline[:, 0] = np.clip(outline[:, 0], 0, width - 1)
    outline[:, 1] = np.clip(outline[:, 1], 0, height - 1)
    dense = tuple((int(round(x)), int(round(y))) for x, y in outline)
    return dense if len(set(dense)) >= 3 else points


def pixel_guided_curved_polygon(
    points: tuple[tuple[int, int], ...],
    size_wh: tuple[int, int],
    source_seeds: np.ndarray,
) -> tuple[tuple[int, int], ...]:
    """Fit two low-frequency curves around the whole word, not each glyph.

    The paired annotation sides define a curved centerline and local normal.
    Pixel GT contributes only an upper/lower envelope in overlapping windows
    along that centerline. The interval between the curves is filled later.
    """

    n = len(points)
    if n < 6 or n % 2 or not source_seeds.any():
        return smooth_curved_polygon(points, size_wh)
    source = np.asarray(points, dtype=np.float64)
    half = n // 2
    top = source[:half]
    bottom = source[half:][::-1]
    count = max(17, SAMPLES_PER_SEGMENT * (half - 1) + 1)
    t = np.linspace(0, half - 1, count)
    station = np.arange(half)
    upper_side = np.column_stack((
        PchipInterpolator(station, top[:, 0])(t),
        PchipInterpolator(station, top[:, 1])(t),
    ))
    lower_side = np.column_stack((
        PchipInterpolator(station, bottom[:, 0])(t),
        PchipInterpolator(station, bottom[:, 1])(t),
    ))
    center = (upper_side + lower_side) / 2
    vector = upper_side - center
    half_width = np.linalg.norm(vector, axis=1)
    if np.any(half_width < 1):
        return smooth_curved_polygon(points, size_wh)
    normal = vector / half_width[:, None]
    yy, xx = np.where(source_seeds)
    pixels = np.column_stack((xx, yy)).astype(np.float64)
    nearest = cKDTree(center).query(pixels, workers=1)[1]
    high = half_width.copy()
    low = -half_width.copy()
    margin = max(1.0, 0.12 * float(np.median(2 * half_width)))
    window = max(1, int(ceil(count / (2 * half))))
    for index in range(count):
        nearby = pixels[np.abs(nearest - index) <= window]
        if not len(nearby):
            continue
        projection = (nearby - center[index]) @ normal[index]
        high[index] = min(1.3 * half_width[index], max(0.0, float(projection.max()) + margin))
        low[index] = max(-1.3 * half_width[index], min(0.0, float(projection.min()) - margin))
        minimum = 1.4 * half_width[index]
        if high[index] - low[index] < minimum:
            high[index] = minimum / 2
            low[index] = -minimum / 2
    high = gaussian_filter1d(high, sigma=1.0, mode="nearest")
    low = gaussian_filter1d(low, sigma=1.0, mode="nearest")
    outline = np.vstack((center + normal * high[:, None],
                         (center + normal * low[:, None])[::-1]))
    width, height = size_wh
    outline[:, 0] = np.clip(outline[:, 0], 0, width - 1)
    outline[:, 1] = np.clip(outline[:, 1], 0, height - 1)
    dense = tuple((int(round(x)), int(round(y))) for x, y in outline)
    return dense if len(set(dense)) >= 3 else smooth_curved_polygon(points, size_wh)


def _uniformly_buffered_polygon(
    points: tuple[tuple[int, int], ...],
    *,
    size_wh: tuple[int, int],
    original_pixels: np.ndarray,
    source_seeds: np.ndarray,
) -> tuple[tuple[tuple[int, int], ...], int] | None:
    """Rasterize a spline-filled box and expand its entire contour if needed."""

    smooth = pixel_guided_curved_polygon(points, size_wh, source_seeds)
    if smooth == points:
        return None
    width, height = size_wh
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    thickness = min(max(xs) - min(xs), max(ys) - min(ys))
    max_buffer = max(1, int(ceil(MAX_BUFFER_FRACTION_OF_WORD_THICKNESS * thickness)))
    x0, x1 = max(0, min(xs) - max_buffer - 2), min(width, max(xs) + max_buffer + 3)
    y0, y1 = max(0, min(ys) - max_buffer - 2), min(height, max(ys) + max_buffer + 3)
    mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    local_points = np.asarray(smooth, dtype=np.int32) - np.array([x0, y0], dtype=np.int32)
    cv2.fillPoly(mask, [local_points], 1)
    if not mask.any():
        return None
    local_seed = source_seeds[y0:y1, x0:x1]
    if int(local_seed.sum()) != int(source_seeds.sum()):
        return None
    distance = distance_transform_edt(mask == 0)
    required = int(ceil(float(distance[local_seed].max()))) if local_seed.any() else 0
    if required > max_buffer:
        return None
    if required:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * required + 1,) * 2)
        mask = cv2.dilate(mask, kernel)
    if local_seed.any() and np.any(local_seed & (mask == 0)):
        return None
    original_area = int(original_pixels.sum())
    if original_area <= 0 or mask.sum() > MAX_AREA_RATIO_TO_V1 * original_area:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        return None
    contour = contours[0].reshape(-1, 2)
    if len(contour) < 3:
        return None
    return tuple((int(x + x0), int(y + y0)) for x, y in contour), required


def make_curved_word_boxes(
    size_wh: tuple[int, int],
    instances: tuple[SceneTextInstance, ...],
    base_owners: Image.Image,
    source_pixel_gt: Image.Image,
    base_separator: Image.Image,
    base_validity: Image.Image,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, CurvedBoxAudit]:
    """Fit curved boxes, restoring v1 words if source/native invariants fail."""

    base = np.asarray(base_owners, dtype=np.uint16)
    pixel = np.asarray(source_pixel_gt.convert("L")) > 0
    if base.shape != pixel.shape or base.shape[::-1] != size_wh:
        raise ValueError("Image, v1 owner and source pixel GT geometry differ")
    care = [item for item in instances if not item.ignore]
    if len(care) != int(base.max()):
        raise ValueError("Care annotation count and v1 owner labels differ")
    valid = np.asarray(base_validity.convert("L")) > 0
    if np.any((base > 0) & ~valid):
        raise ValueError("v1 owner intersects Don't Care")
    baseline = _native_word_component_counts(
        base, separator=base_separator, validity=base_validity
    )
    proposed = list(instances)
    care_positions = [i for i, item in enumerate(instances) if not item.ignore]
    geometric_fallbacks = max_buffer = 0
    for word_id, position in enumerate(care_positions, 1):
        item = instances[position]
        if len(item.polygon_xy) < 6 or len(item.polygon_xy) % 2:
            continue
        own = base == word_id
        fitted = _uniformly_buffered_polygon(
            item.polygon_xy, size_wh=size_wh, original_pixels=own,
            source_seeds=own & pixel,
        )
        if fitted is None:
            geometric_fallbacks += 1
            continue
        polygon, radius = fitted
        proposed[position] = replace(item, polygon_xy=polygon)
        max_buffer = max(max_buffer, radius)
    restored: set[int] = set()
    full_fallback = False
    while True:
        target, validity, separator, owners, _ = make_word_regions(size_wh, tuple(proposed))
        candidate = np.asarray(owners, dtype=np.uint16)
        native = _native_word_component_counts(
            candidate, separator=separator, validity=validity
        )
        violations = []
        for word_id in baseline:
            seeds = (base == word_id) & pixel
            count = native.get(word_id, 0)
            old = baseline[word_id]
            if (np.any(seeds & (candidate != word_id))
                    or (old > 0 and (count > old or count == 0))):
                violations.append(word_id)
        new = [word_id for word_id in violations if word_id not in restored]
        if not new:
            if violations:
                # Cross-word overlap after smoothing can defeat per-word
                # restoration. Only the exact v1 target is safe then.
                proposed = list(instances)
                full_fallback = True
                target, validity, separator, owners, _ = make_word_regions(
                    size_wh, tuple(proposed)
                )
                if not np.array_equal(np.asarray(owners), base):
                    raise ValueError("Full-image v1 fallback is not byte-identical")
            break
        for word_id in new:
            proposed[care_positions[word_id - 1]] = instances[care_positions[word_id - 1]]
            restored.add(word_id)
    final = np.asarray(owners, dtype=np.uint16)
    seeds = (base > 0) & pixel
    if np.any(seeds & (final != base)):
        raise ValueError("Curved box target lost original source text pixels")
    final_native = _native_word_component_counts(
        final, separator=separator, validity=validity
    )
    if any(
        baseline[word_id] > 0 and (
            final_native[word_id] > baseline[word_id] or final_native[word_id] == 0
        )
        for word_id in baseline
    ):
        raise ValueError("Curved box target worsened native word topology")
    audit = CurvedBoxAudit(
        words=len(care), curved_annotations=sum(len(item.polygon_xy) >= 6 for item in care),
        changed_words=sum(
            not np.array_equal(final == word_id, base == word_id)
            for word_id in baseline
        ),
        geometric_fallbacks=geometric_fallbacks,
        coverage_or_topology_fallbacks=len(restored),
        full_image_fallback=full_fallback,
        max_uniform_buffer_pixels=max_buffer,
        base_foreground_pixels=int((base > 0).sum()),
        final_foreground_pixels=int((final > 0).sum()),
        v1_source_seed_pixels=int(seeds.sum()),
        retained_v1_source_seed_pixels=int((seeds & (final == base)).sum()),
    )
    return target, validity, separator, owners, audit
