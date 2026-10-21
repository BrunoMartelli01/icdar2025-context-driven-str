"""Prepare the filled curved-boundary v4 Total-Text TRAIN target."""

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

from recognition.scene_text_datasets import parse_totaltext_annotation
from segmentation.scene_text_preparation import file_sha256
from segmentation.totaltext_curved_box_v4 import make_curved_word_boxes


def _link_verified(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha256(source) != file_sha256(destination):
            raise ValueError(f"Existing linked input differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _save_immutable(image: Image.Image, destination: Path) -> None:
    stream = BytesIO()
    image.save(stream, format="PNG")
    content = stream.getvalue()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise ValueError(f"Existing v4 target differs: {destination}")
    else:
        destination.write_bytes(content)


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing v4 report differs: {path}")
    else:
        path.write_bytes(content)


def prepare(base_root: Path, pixel_root: Path, output_root: Path) -> dict:
    base_root, pixel_root, output_root = (
        base_root.resolve(), pixel_root.resolve(), output_root.resolve()
    )
    if output_root in (base_root, pixel_root):
        raise ValueError("v4 output must be isolated from source targets")
    base_path = base_root / "manifests/total_text_train.yaml"
    pixel_path = pixel_root / "manifests/total_text_train.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    pixel = yaml.safe_load(pixel_path.read_text(encoding="utf-8"))
    if len(base["samples"]) != 1255 or len(pixel["samples"]) != 1255:
        raise ValueError("Expected exactly 1255 official Total-Text TRAIN images")
    pixel_rows = {row["id"]: row for row in pixel["samples"]}
    if len(pixel_rows) != 1255:
        raise ValueError("Duplicate source pixel-mask ID")
    records, audits = [], []
    for number, source in enumerate(base["samples"], 1):
        sample_id = source["id"]
        other = pixel_rows[sample_id]
        if source["image_sha256"] != other["image_sha256"]:
            raise ValueError(f"Source RGB differs: {sample_id}")
        pixel_file = pixel_root / other["mask"]
        if file_sha256(pixel_file) != other["mask_sha256"]:
            raise ValueError(f"Source pixel GT checksum mismatch: {sample_id}")
        annotation = Path("datasets/total_text/train/annotations") / (
            sample_id.split(":", 1)[1] + ".txt"
        )
        _link_verified(base_root / annotation, output_root / annotation)
        _link_verified(base_root / source["image"], output_root / source["image"])
        _link_verified(
            base_root / source["validity_annotation"],
            output_root / source["validity_annotation"],
        )
        with Image.open(base_root / source["image"]) as rgb:
            image_wh = rgb.size
        instances = parse_totaltext_annotation(base_root / annotation)
        with Image.open(base_root / source["word_owner_annotation"]) as owner:
            with Image.open(pixel_file) as original_gt:
                with Image.open(base_root / source["separation_annotation"]) as sep:
                    with Image.open(base_root / source["validity_annotation"]) as valid:
                        target, new_valid, new_sep, new_owner, audit = make_curved_word_boxes(
                            image_wh, instances, owner, original_gt, sep, valid
                        )
                        if not np.array_equal(np.asarray(new_valid), np.asarray(valid)):
                            raise ValueError(f"Don't Care validity changed: {sample_id}")
        row = dict(source)
        for key, image in (
            ("mask", target),
            ("separation_annotation", new_sep),
            ("word_owner_annotation", new_owner),
        ):
            destination = output_root / row[key]
            _save_immutable(image, destination)
            row[f"{key}_sha256"] = file_sha256(destination)
        records.append(row)
        audits.append(asdict(audit))
        if number % 50 == 0 or number == 1255:
            print(f"[filled curved boxes v4] {number}/1255", flush=True)
    manifest = dict(base)
    manifest["split"] = "official-train-filled-curved-word-box-v4"
    manifest["acquisition"] = {
        "classification": "ASSUMPTION",
        "variant_class": "PRACTICAL_APPROXIMATION",
        "base_manifest_sha256": file_sha256(base_path),
        "source_pixel_gt_manifest_sha256": file_sha256(pixel_path),
        "construction": "pixel-guided-smooth-filled-two-sided-curved-word-polygon-v4",
    }
    manifest["annotation_policy"] = dict(base["annotation_policy"])
    manifest["annotation_policy"]["target_source"] = "totaltext_filled_curved_box_v4"
    manifest["annotation_policy"]["target_classification"] = "ASSUMPTION"
    manifest["samples"] = records
    manifest_path = output_root / "manifests/total_text_train.yaml"
    _write_immutable(manifest_path, yaml.safe_dump(manifest, sort_keys=False).encode())
    total = lambda key: sum(row[key] for row in audits)
    report = {
        "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "TRAIN target qualification only; not model accuracy",
        "images": 1255,
        "words": total("words"),
        "curved_annotations": total("curved_annotations"),
        "changed_words": total("changed_words"),
        "geometric_fallbacks": total("geometric_fallbacks"),
        "coverage_or_topology_fallbacks": total("coverage_or_topology_fallbacks"),
        "full_image_fallbacks": total("full_image_fallback"),
        "base_foreground_pixels": total("base_foreground_pixels"),
        "final_foreground_pixels": total("final_foreground_pixels"),
        "v1_source_seed_pixels": total("v1_source_seed_pixels"),
        "retained_v1_source_seed_pixels": total("retained_v1_source_seed_pixels"),
        "base_manifest_sha256": file_sha256(base_path),
        "pixel_gt_manifest_sha256": file_sha256(pixel_path),
        "manifest_sha256": file_sha256(manifest_path),
        "per_image": {row["id"]: audit for row, audit in zip(records, audits)},
    }
    report_path = output_root / "qualification.json"
    _write_immutable(report_path, (json.dumps(report, sort_keys=True, indent=2) + "\n").encode())
    print("MANIFEST", manifest_path, file_sha256(manifest_path), flush=True)
    print("QUALIFICATION", report_path, file_sha256(report_path), flush=True)
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
