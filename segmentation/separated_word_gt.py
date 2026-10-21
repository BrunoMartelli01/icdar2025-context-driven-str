"""GT-derived, separated word regions for APPROXIMATE supervision only.

The original segmentation path is unchanged. Sources retain their word identity;
neither whitespace splitting of text lines nor detector predictions define GT.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, find_objects, label

from recognition.scene_text_datasets import SceneTextInstance
from segmentation.totaltext_word_regions import make_word_regions


def parse_icdar13_character_groups(path: Path) -> tuple[SceneTextInstance, ...]:
    """Derive filled word hulls from blank-separated Task 2.2 character groups.

    This grouping/hull rule is an ASSUMPTION supported by the physical files,
    not a claim that Task 2.2 distributes original word polygons. Ignore rows
    keep their own geometry and cannot erase the identity of a care group.
    """
    output: list[SceneTextInstance] = []
    corners: list[tuple[int, int]] = []
    characters: list[str] = []

    def finish_word() -> None:
        if not corners:
            return
        hull = cv2.convexHull(np.asarray(corners, dtype=np.int32)).reshape(-1, 2)
        if len(hull) < 3:
            raise ValueError(f"Degenerate word hull in {path}")
        output.append(SceneTextInstance(
            tuple(map(tuple, hull.tolist())), "".join(characters), False, "word"
        ))
        corners.clear()
        characters.clear()

    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        row = raw.strip()
        if not row:
            finish_word()
            continue
        ignored = row.startswith("#")
        fields = (row[1:] if ignored else row).strip().split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError(f"Expected 9 numbers and a character in {path}:{line_number}")
        values = [int(item) for item in fields[:9]]
        if any(value < 0 or value > 255 for value in values[:3]):
            raise ValueError(f"Invalid RGB in {path}:{line_number}")
        x0, y0, x1, y1 = values[5:9]
        if x0 > x1 or y0 > y1 or (not ignored and (x0 == x1 or y0 == y1)):
            raise ValueError(f"Invalid character box in {path}:{line_number}")
        text = fields[9]
        if len(text) < 2 or text[0] != '"' or text[-1] != '"':
            raise ValueError(f"Unquoted character in {path}:{line_number}")
        points = tuple(dict.fromkeys(((x0, y0), (x1, y0), (x1, y1), (x0, y1))))
        if ignored:
            # Mixed care/ignore blocks occur in the downloaded source (257).
            output.append(SceneTextInstance(points, "###", True, "word"))
        else:
            if not text[1:-1]:
                raise ValueError(f"Empty care character in {path}:{line_number}")
            corners.extend(points)
            characters.append(text[1:-1])
    finish_word()
    if not output:
        raise ValueError(f"No annotations in {path}")
    return tuple(output)


def audit_word_owners(owners: np.ndarray, expected_words: int) -> dict:
    """Check word survival, single components and zero 8-neighbour contacts."""
    if owners.ndim != 2 or not np.issubdtype(owners.dtype, np.integer):
        raise ValueError("Owners must be a two-dimensional integer array")
    ids = np.unique(owners[owners > 0]).tolist()
    missing = sorted(set(range(1, expected_words + 1)) - set(ids))
    unexpected = sorted(set(ids) - set(range(1, expected_words + 1)))
    contacts = 0
    height, width = owners.shape
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        a = owners[max(0, -dy):height - max(0, dy), max(0, -dx):width - max(0, dx)]
        b = owners[max(0, dy):height - max(0, -dy), max(0, dx):width - max(0, -dx)]
        contacts += int(np.count_nonzero((a > 0) & (b > 0) & (a != b)))
    components = {}
    for index, window in enumerate(find_objects(owners, max_label=expected_words), 1):
        components[str(index)] = (
            int(label(owners[window] == index, structure=np.ones((3, 3)))[1])
            if window is not None else 0
        )
    return {"words": expected_words, "missing_word_ids": missing,
            "unexpected_word_ids": unexpected, "interword_contact_pairs": contacts,
            "components_per_word": components,
            "fragmented_word_ids": [int(i) for i, n in components.items() if n > 1]}


def separated_word_regions(size_wh: tuple[int, int], instances: tuple[SceneTextInstance, ...],
                           *, ignore_policy: str = "ignore_region_precedence"):
    """Fill source word polygons, separate owners, retain one region per word.

    Only secondary disconnected islands are removed, with their pixel count
    recorded. Any missing word fails explicitly. All-ignore images remain in
    the dataset with a zero target and an explicit validity mask.
    """
    if ignore_policy not in {"ignore_region_precedence", "care_region_precedence"}:
        raise ValueError(f"Unknown ignore policy: {ignore_policy}")
    if any(item.granularity != "word" for item in instances):
        raise ValueError("Word GT requires word geometry; text-line annotations cannot be split implicitly")
    if not instances or min(size_wh) <= 0:
        raise ValueError("A nonempty annotation and image are required")
    care = [item for item in instances if not item.ignore]
    if not care:
        validity = Image.new("L", size_wh, 255)
        draw = ImageDraw.Draw(validity)
        for item in instances:
            if len(item.polygon_xy) >= 3:
                draw.polygon(item.polygon_xy, fill=0)
            elif len(item.polygon_xy) == 2:
                draw.line(item.polygon_xy, fill=0)
            else:
                draw.point(item.polygon_xy[0], fill=0)
        owners = np.zeros((size_wh[1], size_wh[0]), dtype=np.uint16)
        mask = Image.fromarray(owners.astype(np.uint8), mode="L")
        return mask, validity, mask.copy(), Image.fromarray(owners, mode="I;16"), {
            **audit_word_owners(owners, 0), "ignored_instances": len(instances),
            "secondary_island_pixels_removed": 0, "source_region_audit": None,
            "ignore_policy": ignore_policy, "care_ignore_overlap_pixels": 0}
    source_instances = tuple(care) if ignore_policy == "care_region_precedence" else instances
    _, validity, separator, owner_image, source_audit = make_word_regions(size_wh, source_instances)
    overlap_pixels = 0
    if ignore_policy == "care_region_precedence":
        # Coarse ignore boxes can enclose explicit care words. Full-word
        # supervision deliberately includes inter-character background pixels.
        union = Image.new("L", size_wh, 0)
        union_draw = ImageDraw.Draw(union)
        for item in care:
            union_draw.polygon(item.polygon_xy, fill=255)
        ignored = Image.new("L", size_wh, 0)
        ignore_draw = ImageDraw.Draw(ignored)
        for item in instances:
            if not item.ignore:
                continue
            if len(item.polygon_xy) >= 3:
                ignore_draw.polygon(item.polygon_xy, fill=255)
            elif len(item.polygon_xy) == 2:
                ignore_draw.line(item.polygon_xy, fill=255)
            else:
                ignore_draw.point(item.polygon_xy[0], fill=255)
        care_pixels = np.asarray(union) > 0
        ignore_pixels = np.asarray(ignored) > 0
        overlap_pixels = int(np.count_nonzero(care_pixels & ignore_pixels))
        validity = Image.fromarray((~ignore_pixels | care_pixels).astype(np.uint8) * 255, mode="L")
    owners = np.asarray(owner_image, dtype=np.uint16).copy()
    before = audit_word_owners(owners, len(care))
    if before["missing_word_ids"]:
        raise ValueError(f"Source geometry loses care words after separation: {before['missing_word_ids']}")
    removed = 0
    for index, window in enumerate(find_objects(owners, max_label=len(care)), 1):
        local = owners[window]
        components, count = label(local == index, structure=np.ones((3, 3)))
        if count > 1:
            areas = np.bincount(components.ravel())
            areas[0] = 0
            discard = (components > 0) & (components != int(areas.argmax()))
            removed += int(discard.sum())
            local[discard] = 0
    audit = audit_word_owners(owners, len(care))
    if audit["missing_word_ids"] or audit["fragmented_word_ids"] or audit["interword_contact_pairs"]:
        raise ValueError(f"Separated word topology failed: {audit}")
    audit.update({"ignored_instances": len(instances) - len(care),
                  "ignore_policy": ignore_policy, "care_ignore_overlap_pixels": overlap_pixels,
                  "secondary_island_pixels_removed": removed,
                  "fragmented_word_ids_before_cleanup": before["fragmented_word_ids"],
                  "source_region_audit": asdict(source_audit)})
    return (Image.fromarray((owners > 0).astype(np.uint8) * 255, mode="L"), validity,
            separator, Image.fromarray(owners, mode="I;16"), audit)


def widen_word_gaps(owners: Image.Image, *, clearance: float,
                   reference_wh: tuple[int, int], max_area_loss: float,
                   protected_pixels: Image.Image | None = None):
    """Retreat only boundaries near another word, keeping full connected bodies.

    Distances use reference-image pixel units, so the same gap request scales
    consistently with source resolution. Actual text-stroke GT, where present,
    is protected. A topology/area failure restores that word and is reported.
    This is an explicit ASSUMPTION; the input owner map is never modified.
    """
    if clearance <= 0 or min(reference_wh) <= 0 or not 0 < max_area_loss < 1:
        raise ValueError("Clearance, reference dimensions and area-loss bound must be valid")
    source = np.asarray(owners, dtype=np.uint16)
    height, width = source.shape
    sampling = (reference_wh[1] / height, reference_wh[0] / width)
    protected = (np.asarray(protected_pixels.convert("L")) > 0 if protected_pixels is not None
                 else np.zeros_like(source, dtype=bool))
    if protected.shape != source.shape:
        raise ValueError("Protected pixel GT and word owner geometry differ")
    output = source.copy()
    windows = find_objects(source)
    pad_y = int(np.ceil(clearance / sampling[0])) + 1
    pad_x = int(np.ceil(clearance / sampling[1])) + 1
    changed = restored = cleanup_pixels = 0
    inherited_fragmented = []
    per_word = []
    for index, window in enumerate(windows, 1):
        if window is None:
            raise ValueError(f"Missing source owner {index}")
        y0, y1 = max(0, window[0].start - pad_y), min(height, window[0].stop + pad_y)
        x0, x1 = max(0, window[1].start - pad_x), min(width, window[1].stop + pad_x)
        local = source[y0:y1, x0:x1]
        own = local == index
        rivals = (local > 0) & ~own
        row = {"owner_id": index, "before_pixels": int(own.sum()),
               "removed_pixels": 0, "restored": False, "protected_close_pixels": 0}
        seeds = protected[y0:y1, x0:x1] & own
        original_components, original_count = label(own, structure=np.ones((3, 3)))
        if original_count > 1:
            areas = np.bincount(original_components.ravel()); areas[0] = 0
            largest = original_components == int(areas.argmax())
            if np.any(seeds & ~largest):
                inherited_fragmented.append(index)
            else:
                removed_islands = own & ~largest
                cleanup_pixels += int(removed_islands.sum())
                output[y0:y1, x0:x1][removed_islands] = 0
                own = largest
        row["near_rival_pixels_before"] = 0
        if rivals.any():
            distance = distance_transform_edt(~rivals, sampling=sampling)
            near = own & (distance < clearance)
            row["near_rival_pixels_before"] = int(near.sum())
            row["protected_close_pixels"] = int((near & seeds).sum())
            discard = near & ~seeds
            candidate = own & ~discard
            components, count = label(candidate, structure=np.ones((3, 3)))
            if count > 1:
                areas = np.bincount(components.ravel()); areas[0] = 0
                largest = components == int(areas.argmax())
                if np.any(seeds & ~largest):
                    row["restored"] = True
                else:
                    candidate = largest
            loss = int(own.sum() - candidate.sum())
            if not candidate.any() or loss > max_area_loss * int(own.sum()):
                row["restored"] = True
            if row["restored"]:
                restored += 1
            else:
                row["removed_pixels"] = loss
                changed += int(loss > 0)
                output[y0:y1, x0:x1][own & ~candidate] = 0
        per_word.append(row)
    if np.any((output > 0) & (output != source)):
        raise ValueError("Gap operation added or reassigned source pixels")
    if np.any(protected & (source > 0) & (output != source)):
        raise ValueError("Gap operation lost protected text pixels")
    audit = audit_word_owners(output, len(windows))
    if audit["missing_word_ids"] or audit["interword_contact_pairs"]:
        raise ValueError(f"Gap operation failed owner survival/separation: {audit}")
    audit.update({"clearance_reference_pixels": clearance, "reference_wh": list(reference_wh),
                  "max_word_area_loss_fraction": max_area_loss, "changed_words": changed,
                  "restored_words": restored, "removed_pixels": int(np.count_nonzero(source) - np.count_nonzero(output)),
                  "protected_close_pixels": sum(r["protected_close_pixels"] for r in per_word),
                  "protected_seed_pixels": int(np.count_nonzero(protected & (source > 0))),
                  "unseeded_island_pixels_removed": cleanup_pixels,
                  "inherited_fragmented_word_ids": inherited_fragmented,
                  "per_word": per_word})
    return Image.fromarray((output > 0).astype(np.uint8) * 255, mode="L"), Image.fromarray(output, mode="I;16"), audit
