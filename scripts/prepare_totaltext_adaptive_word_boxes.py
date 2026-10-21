"""Prepare filled adaptive word-box targets from frozen Total-Text TRAIN data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from io import BytesIO
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import yaml

from segmentation.scene_text_preparation import file_sha256
from segmentation.totaltext_adaptive_word_boxes import fit_adaptive_word_boxes
from segmentation.totaltext_word_regions import restore_worsened_native_words


def _link_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(source) != file_sha256(destination):
            raise ValueError(f"Existing file differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _save_immutable(image: Image.Image, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stream = BytesIO()
    image.save(stream, format="PNG")
    if destination.exists():
        if destination.read_bytes() != stream.getvalue():
            raise ValueError(f"Existing target differs: {destination}")
    else:
        destination.write_bytes(stream.getvalue())


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() != content:
        raise ValueError(f"Existing report differs: {path}")
    if not path.exists():
        path.write_bytes(content)


def prepare(base_root: Path, pixel_root: Path, output_root: Path) -> dict:
    base_root, pixel_root, output_root = (
        base_root.resolve(), pixel_root.resolve(), output_root.resolve()
    )
    if output_root in (base_root, pixel_root):
        raise ValueError("Adaptive boxes require an isolated output root")
    base_path = base_root / "manifests/total_text_train.yaml"
    pixel_path = pixel_root / "manifests/total_text_train.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    pixel = yaml.safe_load(pixel_path.read_text(encoding="utf-8"))
    if len(base["samples"]) != 1255 or len(pixel["samples"]) != 1255:
        raise ValueError("Expected exactly 1255 official TRAIN images")
    pixel_rows = {row["id"]: row for row in pixel["samples"]}
    if len(pixel_rows) != 1255:
        raise ValueError("Duplicate pixel-source sample ID")
    records, audits = [], []
    native_restored = 0
    for number, source in enumerate(base["samples"], 1):
        sample_id = source["id"]
        other = pixel_rows[sample_id]
        if source["image_sha256"] != other["image_sha256"]:
            raise ValueError(f"RGB mismatch: {sample_id}")
        pixel_file = pixel_root / other["mask"]
        if file_sha256(pixel_file) != other["mask_sha256"]:
            raise ValueError(f"Original pixel GT checksum mismatch: {sample_id}")
        for key in ("image", "validity_annotation", "separation_annotation"):
            _link_verified(base_root / source[key], output_root / source[key])
        annotation = Path("datasets/total_text/train/annotations") / (
            sample_id.split(":", 1)[1] + ".txt"
        )
        _link_verified(base_root / annotation, output_root / annotation)
        with Image.open(base_root / source["word_owner_annotation"]) as base_owner:
            with Image.open(pixel_file) as original_gt:
                with Image.open(base_root / source["validity_annotation"]) as validity:
                    with Image.open(base_root / source["separation_annotation"]) as sep:
                        _, owners, audit = fit_adaptive_word_boxes(
                            base_owner, original_gt, validity
                        )
                        target, owners, restored = restore_worsened_native_words(
                            base_owner, owners, sep, validity
                        )
                        source_seed = (np.asarray(original_gt.convert("L")) > 0) & (
                            np.asarray(base_owner) > 0
                        )
                        if np.any(source_seed & (np.asarray(owners) == 0)):
                            raise ValueError(f"Native guard lost source GT seeds: {sample_id}")
                        if np.any((np.asarray(owners) > 0) & (np.asarray(owners) != np.asarray(base_owner))):
                            raise ValueError(f"Word ownership changed: {sample_id}")
        native_restored += restored
        row = dict(source)
        _save_immutable(target, output_root / row["mask"])
        _save_immutable(owners, output_root / row["word_owner_annotation"])
        row["mask_sha256"] = file_sha256(output_root / row["mask"])
        row["word_owner_annotation_sha256"] = file_sha256(
            output_root / row["word_owner_annotation"]
        )
        records.append(row)
        details = asdict(audit)
        details["native_connectivity_restored_words"] = restored
        details["final_box_foreground_pixels"] = int((np.asarray(owners) > 0).sum())
        audits.append(details)
        if number % 50 == 0 or number == 1255:
            print(f"[adaptive filled boxes] {number}/1255", flush=True)
    manifest = dict(base)
    manifest["split"] = "official-train-adaptive-filled-word-boxes-v3"
    manifest["acquisition"] = {
        "classification": "ASSUMPTION",
        "variant_class": "PRACTICAL_APPROXIMATION",
        "base_manifest_sha256": file_sha256(base_path),
        "pixel_gt_manifest_sha256": file_sha256(pixel_path),
        "construction": "filled-per-word-longitudinal-strip-envelope-clipped-to-v1-owner",
    }
    manifest["annotation_policy"] = dict(base["annotation_policy"])
    manifest["annotation_policy"]["target_source"] = "totaltext_adaptive_filled_word_boxes_v3"
    manifest["annotation_policy"]["target_classification"] = "ASSUMPTION"
    manifest["samples"] = records
    manifest_path = output_root / "manifests/total_text_train.yaml"
    _write_immutable(manifest_path, yaml.safe_dump(manifest, sort_keys=False).encode("utf-8"))
    total = lambda key: sum(row[key] for row in audits)
    report = {
        "result_class": "PRACTICAL_APPROXIMATION; TRAIN targets only; not model accuracy",
        "images": 1255,
        "words": total("words"),
        "changed_words_before_native_guard": total("changed_words"),
        "no_seed_fallbacks": total("no_seed_fallbacks"),
        "aspect_fallbacks": total("aspect_fallbacks"),
        "thin_or_fragmented_fallbacks": total("thin_or_fragmented_fallbacks"),
        "native_connectivity_restored_words": native_restored,
        "base_foreground_pixels": total("base_foreground_pixels"),
        "final_box_foreground_pixels": total("final_box_foreground_pixels"),
        "source_pixel_gt_seed_pixels": total("source_seed_pixels"),
        "retained_source_pixel_gt_seed_pixels": total("retained_source_seed_pixels"),
        "base_manifest_sha256": file_sha256(base_path),
        "pixel_gt_manifest_sha256": file_sha256(pixel_path),
        "manifest_sha256": file_sha256(manifest_path),
        "per_image": {row["id"]: detail for row, detail in zip(records, audits)},
    }
    report_path = output_root / "qualification.json"
    _write_immutable(report_path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print("MANIFEST", manifest_path, file_sha256(manifest_path))
    print("QUALIFICATION", report_path, file_sha256(report_path))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--pixel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.base_root, args.pixel_root, args.output_root)


if __name__ == "__main__":
    main()
