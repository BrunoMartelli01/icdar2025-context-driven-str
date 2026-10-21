"""Read-only SynthText word polygons; deterministic SHA-bound external subset.

No images/weights are downloaded. Only imnames, wordBB and txt from gt.mat
are read; character boxes are not interpreted or used as manufactured labels.
"""

from pathlib import Path
from dataclasses import asdict

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

from recognition.scene_text_datasets import SceneTextInstance
from scripts.prune_segmentation_checkpoints import sha256_file
from segmentation.totaltext_word_regions import make_word_regions
from segmentation.totaltext_word_kernel import native_owner_map, make_word_kernels
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.unet_detection_methods import progressive_targets

TARGET_POLICIES = ("legacy_fail_closed", "preserve_visible_words_v1")


def synthtext_word_targets(size_wh, words, *, policy, sample_id):
    """Audit an explicit synthetic-only policy; never clamp off-image words.

    Outside-image annotations have no observable training pixels and remain
    in the manifest with their original indices. All visible words must retain
    owners; malformed/ambiguous annotations still fail with sample context.
    """
    if policy not in TARGET_POLICIES:
        raise ValueError("Unknown SynthText target policy: " + str(policy))
    ignored, visible = [], []
    for index, word in enumerate(words, 1):
        points = tuple(tuple(p) for p in word["polygon_xy"])
        if policy == "preserve_visible_words_v1":
            width, height = size_wh
            xs, ys = zip(*points)
            outside = max(xs) < 0 or min(xs) >= width or max(ys) < 0 or min(ys) >= height
            if outside:
                ignored.append(index)
                continue
            x0, y0 = max(0, min(xs)), max(0, min(ys))
            x1, y1 = min(width - 1, max(xs)), min(height - 1, max(ys))
            canvas = Image.new("1", (x1 - x0 + 1, y1 - y0 + 1))
            ImageDraw.Draw(canvas).polygon(tuple((x - x0, y - y0) for x, y in points), fill=1)
            if not np.asarray(canvas).any():
                raise ValueError(f"SynthText {sample_id}: visible word {index} has an empty raster; "
                                 f"polygon={points}, image_size={size_wh}")
        visible.append((index, SceneTextInstance(points, word["transcription"], False, "word")))
    if not visible:
        raise ValueError(f"SynthText {sample_id}: no visible word annotations")
    try:
        maps = make_word_regions(size_wh, tuple(item for _, item in visible),
                                preserve_contact_words=policy == "preserve_visible_words_v1")
    except ValueError as exc:
        raise ValueError(f"SynthText {sample_id}, image_size={size_wh}: {exc}") from exc
    mask, valid, separation, owners, region_audit = maps
    if region_audit.empty_word_count:
        present = set(np.unique(np.asarray(owners)))
        missing = [index for owner_id, (index, _) in enumerate(visible, 1) if owner_id not in present]
        raise ValueError(f"SynthText {sample_id}: full-resolution rasterization lost words {missing}; "
                         f"image_size={size_wh}, policy={policy}")
    restored = [visible[index - 1][0] for index in getattr(region_audit, "restored_contact_word_ids", ())]
    audit = {"policy": policy, "annotation_words": len(words), "visible_words": len(visible),
             "outside_image_word_ids": ignored, "restored_contact_word_ids": restored,
             "region_audit": asdict(region_audit)}
    # JSON-compatible receipts also compare identically after manifest reload.
    if "restored_contact_word_ids" in audit["region_audit"]:
        audit["region_audit"]["restored_contact_word_ids"] = list(region_audit.restored_contact_word_ids)
    return mask, valid, separation, owners, audit


def matlab_text(value):
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.dtype.kind in "US":
        values = [str(x) for x in array.reshape(-1)]
        return ("" if all(len(x) <= 1 for x in values) else " ").join(values)
    if array.dtype == object:
        return " ".join(matlab_text(x) for x in array.reshape(-1))
    raise ValueError("Malformed MATLAB text cell")


def decode_word_boxes(boxes, text, coordinate_origin):
    if coordinate_origin not in (0, 1):
        raise ValueError("Explicit SynthText coordinate origin must be0 or1")
    value = np.asarray(boxes, dtype=np.float64)
    if value.shape == (2, 4):
        value = value[:, :, None]
    if value.ndim != 3 or value.shape[:2] != (2, 4) or not np.isfinite(value).all():
        raise ValueError("SynthText wordBB must be finite2x4xN")
    words = matlab_text(text).split()
    if len(words) != value.shape[2]:
        raise ValueError("SynthText word boxes and whitespace-separated transcription count differ")
    return [{"polygon_xy": np.rint(value[:, :, i].T - coordinate_origin).astype(int).tolist(),
             "transcription": word} for i, word in enumerate(words)]


def prepare_synthtext(root, *, max_images, seed, coordinate_origin, excluded_image_hashes,
                      materialize_image=None, target_policy="legacy_fail_closed"):
    """Build the same subset with optional on-demand acquisition before hashing.

    The callback supplies unchanged original RGB bytes at the resolved path.
    Selection, duplicate/overlap exclusions, annotation and image checks remain
    shared by ordinary preparation and storage-conserving ZIP extraction.
    """
    if target_policy not in TARGET_POLICIES:
        raise ValueError("Unknown SynthText target policy: " + str(target_policy))
    root = Path(root).resolve()
    mat = root / "gt.mat"
    if not mat.is_file():
        raise FileNotFoundError("SynthText requires --synthtext-root containing gt.mat and original images")
    if type(max_images) is not int or max_images < 1:
        raise ValueError("Positive predeclared SynthText subset size required")
    annotation_sha = sha256_file(mat)
    cells = loadmat(mat, variable_names=["imnames", "wordBB", "txt"])
    if any(k not in cells for k in ("imnames", "wordBB", "txt")):
        raise ValueError("SynthText gt.mat requires imnames, wordBB and txt")
    names, boxes, texts = [cells[k].reshape(-1) for k in ("imnames", "wordBB", "txt")]
    if len(names) != len(boxes) or len(names) != len(texts) or not len(names):
        raise ValueError("SynthText annotation cell populations differ")
    if max_images > len(names):
        raise ValueError("Requested SynthText subset exceeds dataset; change config explicitly")
    order = np.random.default_rng(seed).permutation(len(names))
    rows, excluded, duplicate_hashes = [], [], set()
    for index in order:
        relative = Path(matlab_text(names[index]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe SynthText image path")
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("SynthText image escapes its root")
        if materialize_image is not None:
            materialize_image(path)
        digest = sha256_file(path)  # A missing or invalid image is a failure.
        if digest in excluded_image_hashes:
            excluded.append({"image": str(relative), "reason": "TotalText_RGB_overlap"})
            continue
        if digest in duplicate_hashes:
            excluded.append({"image": str(relative), "reason": "duplicate_RGB_bytes"})
            continue
        words = decode_word_boxes(boxes[index], texts[index], coordinate_origin)
        with Image.open(path) as opened:
            size = opened.size
            opened.verify()
        row = {"sample_id": "synthtext:" + str(int(index)), "image_path": str(path),
               "image_sha256": digest, "words": words}
        if target_policy != "legacy_fail_closed":
            *_, row["target_audit"] = synthtext_word_targets(
                size, words, policy=target_policy, sample_id=row["sample_id"])
        rows.append(row)
        duplicate_hashes.add(digest)
        if len(rows) % 1000 == 0:
            print(f"[SynthText manifest] {len(rows)}/{max_images}", flush=True)
        if len(rows) == max_images:
            break
    if len(rows) != max_images:
        raise ValueError("Not enough eligible SynthText images after explicit overlap/duplicate exclusions")
    result = {"schema_version": 1, "reproduction_mode": "approximate",
            "result_class": "approximate reproduction", "dataset": "SynthText",
            "subset": "deterministic partial external training dataset", "seed": seed,
            "coordinate_origin": coordinate_origin, "gt_mat_sha256": annotation_sha,
            "root": str(root), "images": len(rows), "excluded": excluded, "samples": rows,
            "leakage_limitation": "Exact RGB-byte overlap excluded; perceptual near-duplicates unresolved"}
    if target_policy != "legacy_fail_closed":
        result["target_policy"] = target_policy
        result["target_summary"] = {
            "annotation_words": sum(r["target_audit"]["annotation_words"] for r in rows),
            "visible_words": sum(r["target_audit"]["visible_words"] for r in rows),
            "outside_image_words": sum(len(r["target_audit"]["outside_image_word_ids"]) for r in rows),
            "restored_contact_words": sum(len(r["target_audit"]["restored_contact_word_ids"]) for r in rows)}
    return result


class SynthTextWordDataset(Dataset):
    def __init__(self, manifest, transform, fractions, *, target_policy="legacy_fail_closed"):
        self.manifest, self.transform, self.fractions = manifest, transform, fractions
        if target_policy not in TARGET_POLICIES:
            raise ValueError("Unknown SynthText target policy: " + str(target_policy))
        if manifest.get("target_policy", "legacy_fail_closed") != target_policy:
            raise ValueError("SynthText manifest target policy differs; prepare a new audited manifest")
        self.target_policy = target_policy

    def __len__(self):
        return len(self.manifest["samples"])

    def __getitem__(self, index):
        row = self.manifest["samples"][index]
        path = Path(row["image_path"])
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError("SynthText RGB changed: " + row["sample_id"])
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        mask, valid, separation, owners, audit = synthtext_word_targets(
            image.size, row["words"], policy=self.target_policy, sample_id=row["sample_id"])
        if self.target_policy != "legacy_fail_closed" and row.get("target_audit") != audit:
            raise ValueError("SynthText target audit changed: " + row["sample_id"])
        x, target, validity, sep = self.transform.transform_with_validity_and_separation(
            image, mask, valid, separation, owners.convert("I"), index)
        native = native_owner_map(owners.convert("I"), self.transform, index, target, validity)
        kernel, _ = make_word_kernels(native, distance_fraction=self.fractions[0])
        return {"sample_id": row["sample_id"], "image_path": str(path), "image": x, "mask": target,
                "validity": validity, "separation": sep, "owners_native": torch.from_numpy(native),
                "kernel": torch.from_numpy((kernel > 0).astype(np.float32))[None],
                "kernels": torch.from_numpy(progressive_targets(native, self.fractions)),
                "canvas": letterbox_canvas_mask(image.size, native_size=target.shape[-1]),
                "native_words_lost": audit["visible_words"] - len(np.unique(native[native > 0])),
                "outside_image_words": len(audit["outside_image_word_ids"]),
                "restored_contact_words": len(audit["restored_contact_word_ids"])}
