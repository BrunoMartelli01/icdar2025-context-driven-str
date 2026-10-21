"""Smaller filled word cores, an explicit APPROXIMATE supervision experiment."""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, find_objects, label

from segmentation.separated_word_gt import audit_word_owners


def native_training_owners(owners, validity, separators, reference_wh):
    """Replay the existing nonaugmented PIL target resize and owner separators.

    This checks effective target survival, not just nearest-resized owner IDs.
    It does not promise survival under future random rotation/augmentation.
    """
    own = np.asarray(Image.fromarray(owners.astype(np.int32), mode="I").resize(
        reference_wh, Image.Resampling.NEAREST)).copy()
    mask = np.asarray(Image.fromarray((owners > 0).astype(np.uint8) * 255).resize(
        reference_wh, Image.Resampling.BILINEAR)) >= 128
    valid = np.asarray(Image.fromarray(validity).resize(reference_wh, Image.Resampling.NEAREST)) > 0
    sep = np.asarray(Image.fromarray(separators).resize(reference_wh, Image.Resampling.NEAREST)) > 0
    contacts = np.zeros(own.shape, dtype=bool)
    h, w = own.shape
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        wa = (slice(max(0, -dy), h - max(0, dy)), slice(max(0, -dx), w - max(0, dx)))
        wb = (slice(max(0, dy), h - max(0, -dy)), slice(max(0, dx), w - max(0, -dx)))
        a, b = own[wa], own[wb]
        touch = (a > 0) & (b > 0) & (a != b)
        contacts[wa] |= touch
        contacts[wb] |= touch
    own[~mask | ~valid | sep | contacts] = 0
    return own


def compact_word_owners(owners, validity, separators, *, distance_fraction,
                        max_inset_reference, reference_wh, crop_padding_reference,
                        min_area_fraction, min_span_fraction, backoff_steps):
    """Shrink each word by its interior distance; retain a connected full core.

    Relative radius and a native-pixel cap prevent a fixed original-pixel
    erosion from disproportionately erasing small words. Back off when the core
    becomes too short/small or the unchanged crop padding cannot recover the
    original word bounds. Restore extra words lost by the real resize contract.
    No pixels are added, reassigned, or moved between word identities.
    """
    source = np.asarray(owners, dtype=np.uint16)
    valid = np.asarray(validity, dtype=np.uint8)
    original_sep = np.asarray(separators, dtype=np.uint8)
    if (source.ndim != 2 or source.shape != valid.shape or source.shape != original_sep.shape
        or not 0 < distance_fraction < 1 or max_inset_reference <= 0
        or min(reference_wh) <= 0 or crop_padding_reference < 0
        or not 0 < min_area_fraction <= 1 or not 0 < min_span_fraction <= 1
        or not isinstance(backoff_steps, int) or backoff_steps < 0):
        raise ValueError("Invalid compact-word geometry or explicit shrinking parameters")
    expected = int(source.max()) if source.size else 0
    before = audit_word_owners(source, expected)
    if before["missing_word_ids"] or before["interword_contact_pairs"]:
        raise ValueError("Source owners must preserve all separated word identities")
    h, w = source.shape
    sy, sx = reference_wh[1] / h, reference_wh[0] / w
    result = np.zeros_like(source)
    per_word = []
    for owner_id, window in enumerate(find_objects(source, max_label=expected), 1):
        word = source[window] == owner_id
        distances = distance_transform_edt(np.pad(word, 1), sampling=(sy, sx))[1:-1, 1:-1]
        requested = min(distance_fraction * float(distances.max()), max_inset_reference)
        source_y, source_x = np.where(word)
        source_bounds = (source_x.min(), source_y.min(), source_x.max(), source_y.max())
        x0, y0, x1, y1 = source_bounds
        span = max(x1 - x0 + 1, y1 - y0 + 1)
        chosen = word
        effective = 0.0
        fragments = 0
        for step in range(backoff_steps + 1):
            radius = requested / (2 ** step)
            candidate = word & (distances > radius)
            components, count = label(candidate, structure=np.ones((3, 3)))
            if not count:
                continue
            areas = np.bincount(components.ravel()); areas[0] = 0
            core = components == int(areas.argmax())
            ys, xs = np.where(core)
            cx0, cy0, cx1, cy1 = xs.min(), ys.min(), xs.max(), ys.max()
            crop_covers = ((cx0 - x0) * sx <= crop_padding_reference
                and (cy0 - y0) * sy <= crop_padding_reference
                and (x1 - cx1) * sx <= crop_padding_reference
                and (y1 - cy1) * sy <= crop_padding_reference)
            if (int(core.sum()) >= min_area_fraction * int(word.sum())
                and max(cx1 - cx0 + 1, cy1 - cy0 + 1) >= min_span_fraction * span
                and crop_covers):
                chosen, effective, fragments = core, radius, count
                break
        result[window][chosen] = owner_id
        per_word.append({"owner_id": owner_id, "before_pixels": int(word.sum()),
                         "requested_inset_reference": requested, "effective_inset_reference": effective,
                         "core_fragments_before_selection": fragments,
                         "geometry_restored": effective == 0.0, "native_survival_restored": False})

    baseline_native = native_training_owners(source, valid, original_sep, reference_wh)
    baseline_ids = set(np.unique(baseline_native)) - {0}
    restored = set()
    for _ in range(expected + 1):
        sep = original_sep.copy()
        sep[(source > 0) & (result == 0) & (valid > 0)] = 255
        after_native = native_training_owners(result, valid, sep, reference_wh)
        lost = baseline_ids - (set(np.unique(after_native)) - {0})
        if not lost:
            break
        if lost <= restored:
            raise ValueError("Native survival restoration did not converge")
        for owner_id in lost:
            result[source == owner_id] = owner_id
            restored.add(owner_id)
            per_word[owner_id - 1]["native_survival_restored"] = True
    else:
        raise ValueError("Native survival restoration exceeded the owner population")
    if np.any((result > 0) & (result != source)):
        raise ValueError("Compact targets added or reassigned source pixels")
    audit = audit_word_owners(result, expected)
    if audit["missing_word_ids"] or audit["interword_contact_pairs"]:
        raise ValueError("Compact targets lost words or joined distinct words")
    areas_after = np.bincount(result.ravel(), minlength=expected + 1)
    for row in per_word:
        after_pixels = int(areas_after[row["owner_id"]])
        row.update({"after_pixels": after_pixels, "removed_pixels": row["before_pixels"] - after_pixels,
                    "retained_area_fraction": after_pixels / row["before_pixels"]})
    audit.update({"distance_fraction": distance_fraction,
                  "max_inset_reference_pixels": max_inset_reference,
                  "crop_padding_reference_pixels": crop_padding_reference,
                  "reference_wh": list(reference_wh), "min_area_fraction": min_area_fraction,
                  "min_span_fraction": min_span_fraction, "backoff_steps": backoff_steps,
                  "source_foreground_pixels": int(np.count_nonzero(source)),
                  "foreground_pixels": int(np.count_nonzero(result)),
                  "removed_pixels": int(np.count_nonzero(source) - np.count_nonzero(result)),
                  "changed_words": sum(row["removed_pixels"] > 0 for row in per_word),
                  "native_survival_restored_ids": sorted(map(int, restored)),
                  "source_words_absent_at_training_resolution": sorted(set(range(1, expected + 1)) - baseline_ids),
                  "new_words_lost_at_training_resolution": [], "per_word": per_word})
    return result, sep, audit
