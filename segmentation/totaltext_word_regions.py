"""Disjoint curved word targets from Total-Text polygons (PRACTICAL_APPROXIMATION).

The official polygon, rather than its axis-aligned bounding rectangle, is the
word ribbon.  Contested pixels are assigned to the nearest exclusive pixel of
one of the polygons that actually covers them.  Adjacent different owners are
then separated by a black corridor; no text transcription is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, label

from recognition.scene_text_datasets import SceneTextInstance


@dataclass(frozen=True)
class WordRegionAudit:
    word_count: int
    curved_word_count: int
    contested_pixels: int
    removed_separator_pixels: int
    interword_gap_pixels: int
    retained_polygon_fraction: float
    empty_word_count: int


@dataclass(frozen=True)
class PreservedWordRegionAudit(WordRegionAudit):
    restored_contact_word_ids: tuple[int, ...]


@dataclass(frozen=True)
class RefinedWordAudit:
    word_count: int
    words_with_no_pixel_gt_seed: int
    words_with_disconnected_owner_fallback: int
    original_owner_pixels: int
    refined_owner_pixels: int
    pixel_gt_seed_pixels: int
    pixel_gt_seed_retained: int
    max_radius_pixels: int
    native_connectivity_reverted_words: int = 0


def refine_word_regions_with_pixel_gt(
    owners: Image.Image,
    source_pixel_gt: Image.Image,
    validity: Image.Image,
) -> tuple[Image.Image, Image.Image, RefinedWordAudit]:
    """Tightest connected source-polygon subregions around original GT pixels.

    The original pixel GT is used only to shape a TRAIN target. Every retained
    pixel belongs to exactly one already-disjoint word owner. No transcription
    or TEST label is used. The minimum envelope radius is one native-128 pixel
    expressed in source coordinates; larger radii are used only as needed to
    connect the seeded word region within its existing polygon.
    """

    owner = np.asarray(owners, dtype=np.uint16)
    pixels = np.asarray(source_pixel_gt.convert("L")) > 0
    valid = np.asarray(validity.convert("L")) > 0
    if owner.shape != pixels.shape or owner.shape != valid.shape:
        raise ValueError("Owner, pixel GT and validity geometry must match")
    if np.any((owner > 0) & ~valid):
        raise ValueError("Owner map contains ignored pixels")
    height, width = owner.shape
    base_radius = max(1, int(round(min(width, height) / 128)))
    refined = np.zeros_like(owner)
    absent_seeds = disconnected_fallbacks = largest_radius = 0
    seed_total = kept_seed_total = 0
    labels = np.unique(owner)
    labels = labels[labels > 0]
    structure = np.ones((3, 3), dtype=np.uint8)
    for index in labels:
        ys, xs = np.where(owner == index)
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
        local_owner = owner[y0:y1, x0:x1] == index
        local_seed = local_owner & pixels[y0:y1, x0:x1]
        seed_total += int(local_seed.sum())
        if not local_seed.any():
            absent_seeds += 1
            refined[y0:y1, x0:x1][local_owner] = index
            continue
        _, owner_cc = label(local_owner, structure=structure)
        if owner_cc != 1:
            disconnected_fallbacks += 1
            refined[y0:y1, x0:x1][local_owner] = index
            kept_seed_total += int(local_seed.sum())
            continue
        distance = distance_transform_edt(~local_seed)
        maximum = max(base_radius, int(np.ceil(distance[local_owner].max())))
        low, high = min(base_radius, maximum), maximum
        while low < high:
            radius = (low + high) // 2
            candidate = local_owner & (distance <= radius)
            _, count = label(candidate, structure=structure)
            if count == 1:
                high = radius
            else:
                low = radius + 1
        radius = low
        candidate = local_owner & (distance <= radius)
        _, count = label(candidate, structure=structure)
        if count != 1 or not np.all(candidate[local_seed]):
            raise ValueError(f"Failed to form a connected region for word owner {index}")
        refined[y0:y1, x0:x1][candidate] = index
        kept_seed_total += int((refined[y0:y1, x0:x1][local_seed] == index).sum())
        largest_radius = max(largest_radius, radius)
    if seed_total != kept_seed_total:
        raise ValueError("Refined word target lost source pixel-GT seeds")
    audit = RefinedWordAudit(
        word_count=len(labels),
        words_with_no_pixel_gt_seed=absent_seeds,
        words_with_disconnected_owner_fallback=disconnected_fallbacks,
        original_owner_pixels=int((owner > 0).sum()),
        refined_owner_pixels=int((refined > 0).sum()),
        pixel_gt_seed_pixels=seed_total,
        pixel_gt_seed_retained=kept_seed_total,
        max_radius_pixels=largest_radius,
    )
    return (
        Image.fromarray((refined > 0).astype(np.uint8) * 255, mode="L"),
        Image.fromarray(refined, mode="I;16"),
        audit,
    )


def _native_word_component_counts(
    owners: np.ndarray,
    *,
    separator: Image.Image,
    validity: Image.Image,
) -> dict[int, int]:
    """Count per-word native-128 CC using the qualified no-augmentation resize."""

    full_mask = Image.fromarray((owners > 0).astype(np.uint8) * 255, mode="L")
    full_owner = Image.fromarray(owners.astype(np.int32), mode="I")
    mask = np.asarray(full_mask.resize((128, 128), Image.Resampling.BILINEAR)) >= 128
    owner = np.asarray(full_owner.resize((128, 128), Image.Resampling.NEAREST))
    valid = np.asarray(validity.resize((128, 128), Image.Resampling.NEAREST)) > 0
    sep = np.asarray(separator.resize((128, 128), Image.Resampling.NEAREST)) > 0
    contact = np.zeros_like(mask)
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        y0a, y1a = max(0, -dy), 128 - max(0, dy)
        x0a, x1a = max(0, -dx), 128 - max(0, dx)
        y0b, y1b = max(0, dy), 128 - max(0, -dy)
        x0b, x1b = max(0, dx), 128 - max(0, -dx)
        left = owner[y0a:y1a, x0a:x1a]
        right = owner[y0b:y1b, x0b:x1b]
        different = (left > 0) & (right > 0) & (left != right)
        contact[y0a:y1a, x0a:x1a] |= different
        contact[y0b:y1b, x0b:x1b] |= different
    mask &= (owner > 0) & valid & ~sep & ~contact
    structure = np.ones((3, 3), dtype=np.uint8)
    return {
        int(index): int(label(mask & (owner == index), structure=structure)[1])
        for index in np.unique(owners)
        if index > 0
    }


def restore_worsened_native_words(
    base_owners: Image.Image,
    refined_owners: Image.Image,
    separator: Image.Image,
    validity: Image.Image,
) -> tuple[Image.Image, Image.Image, int]:
    """Revert only words whose native CC count worsens versus filled v1.

    This is a deterministic target-topology guard, not GT or test tuning.
    Different word labels remain disjoint because each refined word is a
    subset of exactly the corresponding v1 owner region.
    """

    base = np.asarray(base_owners, dtype=np.uint16)
    refined = np.asarray(refined_owners, dtype=np.uint16).copy()
    if base.shape != refined.shape or np.any((refined > 0) & (refined != base)):
        raise ValueError("Refined owner map must be a same-sized subset of base owners")
    baseline = _native_word_component_counts(base, separator=separator, validity=validity)
    restored: set[int] = set()
    while True:
        observed = _native_word_component_counts(refined, separator=separator, validity=validity)
        worsened = [
            index for index, count in baseline.items()
            if count > 0 and (observed[index] > count or observed[index] == 0)
        ]
        new = [index for index in worsened if index not in restored]
        if not new:
            if worsened:
                # Bilinear resizing is global: restoring one owner's pixels
                # can leave that owner's boundary different while neighbours
                # remain refined.  The only fail-closed reference is the
                # complete, already-qualified v1 image target.
                refined = base.copy()
                restored = set(baseline)
                if _native_word_component_counts(
                    refined, separator=separator, validity=validity
                ) != baseline:
                    raise ValueError("Native connectivity guard failed full-image fallback")
            break
        for index in new:
            refined[base == index] = index
            restored.add(index)
    target = Image.fromarray((refined > 0).astype(np.uint8) * 255, mode="L")
    return target, Image.fromarray(refined, mode="I;16"), len(restored)


def make_word_regions(
    size_wh: tuple[int, int], instances: tuple[SceneTextInstance, ...], *,
    preserve_contact_words: bool = False,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, WordRegionAudit]:
    """Return binary target, validity, separator, uint16 owner map and audit.

    A one-pixel 8-neighbour separator is enforced on the original raster.
    Unrepresentable words are reported in the audit. The optional SynthText
    policy restores existing owned pixels if the corridor erases a whole word;
    it never extends a polygon or changes the default Total-Text targets.
    """

    width, height = size_wh
    if width <= 0 or height <= 0 or not instances:
        raise ValueError("Word-region target requires a non-empty annotation and image")
    care = [item for item in instances if not item.ignore]
    if len(care) >= 65535:
        raise ValueError("Too many word instances for uint16 owner map")
    polygons = []
    validity = Image.new("L", size_wh, 255)
    validity_draw = ImageDraw.Draw(validity)
    for item in instances:
        if item.ignore:
            if len(item.polygon_xy) == 1:
                validity_draw.point(item.polygon_xy, fill=0)
            elif len(item.polygon_xy) == 2:
                validity_draw.line(item.polygon_xy, fill=0)
            else:
                validity_draw.polygon(item.polygon_xy, fill=0)
    for item in care:
        canvas = Image.new("1", size_wh, 0)
        ImageDraw.Draw(canvas).polygon(item.polygon_xy, fill=1)
        polygons.append(np.asarray(canvas, dtype=bool))
    if not polygons:
        raise ValueError("Word-region target has no care words")
    stack = np.stack(polygons, axis=0)
    count = stack.sum(axis=0)
    owner = np.where(count == 1, stack.argmax(axis=0) + 1, 0).astype(np.uint16)
    contested = count > 1
    # The nearest exclusive-pixel distance is evaluated only for participating
    # polygons.  A fully overlaid polygon gets a deterministic deepest-interior
    # seed; this keeps the annotation represented while recording contention.
    if contested.any():
        distances = []
        for index, polygon in enumerate(polygons, 1):
            exclusive = owner == index
            if not exclusive.any():
                interior = distance_transform_edt(polygon)
                y, x = np.unravel_index(int(interior.argmax()), interior.shape)
                if not polygon[y, x]:
                    raise ValueError(f"Care word {index} rasterized to an empty polygon")
                exclusive = exclusive.copy()
                exclusive[y, x] = True
            distances.append(distance_transform_edt(~exclusive)[contested])
        d = np.stack(distances, axis=0)
        d[~stack[:, contested]] = np.inf
        owner[contested] = (d.argmin(axis=0) + 1).astype(np.uint16)
    # Remove both sides of a contact.  This sacrifices only boundary pixels,
    # unlike discarding entire overlapping polygon areas.
    contact = np.zeros_like(owner, dtype=bool)
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        y0a, y1a = max(0, -dy), height - max(0, dy)
        x0a, x1a = max(0, -dx), width - max(0, dx)
        y0b, y1b = max(0, dy), height - max(0, -dy)
        x0b, x1b = max(0, dx), width - max(0, -dx)
        left = owner[y0a:y1a, x0a:x1a]
        right = owner[y0b:y1b, x0b:x1b]
        different = (left != 0) & (right != 0) & (left != right)
        contact[y0a:y1a, x0a:x1a] |= different
        contact[y0b:y1b, x0b:x1b] |= different
    before_contact = owner.copy() if preserve_contact_words else None
    owner[contact] = 0
    valid = np.asarray(validity) > 0
    owner[~valid] = 0
    restored = ()
    if preserve_contact_words:
        present_before = set(np.unique(before_contact[valid])) - {0}
        present_after = set(np.unique(owner)) - {0}
        restored = tuple(sorted(int(i) for i in present_before - present_after))
        for index in restored:
            pixels = (before_contact == index) & valid
            owner[pixels] = index
            contact[pixels] = False
    # Mark short background gaps reached by two different original word
    # regions. Radius is one native-128 pixel expressed in source coordinates,
    # not a dataset-tuned word-gap parameter. Only the auxiliary penalty map
    # changes; the binary target remains the disjoint polygon raster.
    radius = max(1, int(round(min(width, height) / 128)))
    reach = np.zeros_like(owner, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    for index in range(1, len(care) + 1):
        ys, xs = np.where(owner == index)
        if not len(xs):
            continue
        x0, x1 = max(0, int(xs.min()) - radius), min(width, int(xs.max()) + radius + 1)
        y0, y1 = max(0, int(ys.min()) - radius), min(height, int(ys.max()) + radius + 1)
        local = owner[y0:y1, x0:x1]
        dilation = cv2.dilate((local == index).astype(np.uint8), kernel) > 0
        local_reach = reach[y0:y1, x0:x1]
        reached = (local == 0) & dilation
        local_reach[reached] = np.minimum(local_reach[reached] + 1, 2)
    gap = (reach >= 2) & (owner == 0) & valid
    separator = (contact | gap) & valid
    retained = np.count_nonzero(owner)
    original = np.count_nonzero((count > 0) & valid)
    audit = WordRegionAudit(
        word_count=len(care),
        curved_word_count=sum(len(item.polygon_xy) > 4 for item in care),
        contested_pixels=int(contested.sum()),
        removed_separator_pixels=int(contact.sum()),
        interword_gap_pixels=int(gap.sum()),
        retained_polygon_fraction=float(retained / original) if original else 0.0,
        empty_word_count=sum(not np.any(owner == index) for index in range(1, len(care) + 1)),
    )
    if preserve_contact_words:
        from dataclasses import asdict
        audit = PreservedWordRegionAudit(**asdict(audit), restored_contact_word_ids=restored)
    return (
        Image.fromarray((owner > 0).astype(np.uint8) * 255, mode="L"),
        validity,
        Image.fromarray(separator.astype(np.uint8) * 255, mode="L"),
        Image.fromarray(owner, mode="I;16"),
        audit,
    )
