"""Prepare a tighter, pixel-GT-guided curved-word TRAIN target (isolated v2)."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import shutil

from PIL import Image
import numpy as np
import yaml

from segmentation.scene_text_preparation import file_sha256
from segmentation.totaltext_word_regions import (
    refine_word_regions_with_pixel_gt, restore_worsened_native_words,
)


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


def _save_checked(image: Image.Image, destination: Path, *, allow_refresh: bool = False) -> None:
    from io import BytesIO

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        stream = BytesIO()
        image.save(stream, format="PNG")
        if destination.read_bytes() != stream.getvalue():
            if not allow_refresh:
                raise ValueError(f"Existing refined mask differs: {destination}")
            image.save(destination, format="PNG")
    else:
        image.save(destination, format="PNG")


def prepare(base_root: Path, pixel_root: Path, output_root: Path, *, refresh: bool = False) -> dict:
    base_root, pixel_root, output_root = base_root.resolve(), pixel_root.resolve(), output_root.resolve()
    base_manifest_path = base_root / "manifests/total_text_train.yaml"
    pixel_manifest_path = pixel_root / "manifests/total_text_train.yaml"
    base = yaml.safe_load(base_manifest_path.read_text(encoding="utf-8"))
    pixel = yaml.safe_load(pixel_manifest_path.read_text(encoding="utf-8"))
    if len(base["samples"]) != 1255 or len(pixel["samples"]) != 1255:
        raise ValueError("Refinement requires the 1255-image official TRAIN population")
    pixel_records = {row["id"]: row for row in pixel["samples"]}
    records = []
    audits = []
    for number, source in enumerate(base["samples"], 1):
        sample_id = source["id"]
        if sample_id not in pixel_records:
            raise ValueError(f"Pixel GT source lacks {sample_id}")
        pixel_record = pixel_records[sample_id]
        if source["image_sha256"] != pixel_record["image_sha256"]:
            raise ValueError(f"Source RGB differs for {sample_id}")
        if file_sha256(pixel_root / pixel_record["mask"]) != pixel_record["mask_sha256"]:
            raise ValueError(f"Pixel GT checksum mismatch for {sample_id}")
        for key in ("image", "validity_annotation", "separation_annotation"):
            _link_verified(base_root / source[key], output_root / source[key])
        annotation = Path("datasets/total_text/train/annotations") / f"{sample_id.split(':', 1)[1]}.txt"
        _link_verified(base_root / annotation, output_root / annotation)
        with Image.open(base_root / source["word_owner_annotation"]) as opened_owner:
            with Image.open(pixel_root / pixel_record["mask"]) as opened_pixel:
                with Image.open(base_root / source["validity_annotation"]) as opened_validity:
                    with Image.open(base_root / source["separation_annotation"]) as opened_sep:
                        target, owners, audit = refine_word_regions_with_pixel_gt(
                            opened_owner, opened_pixel, opened_validity
                        )
                        target, owners, restored = restore_worsened_native_words(
                            opened_owner, owners, opened_sep, opened_validity
                        )
                        audit = replace(
                            audit,
                            refined_owner_pixels=int((np.asarray(owners) > 0).sum()),
                            native_connectivity_reverted_words=restored,
                        )
        if audit.word_count <= 0 or audit.pixel_gt_seed_retained != audit.pixel_gt_seed_pixels:
            raise ValueError(f"Invalid refined region for {sample_id}")
        _save_checked(target, output_root / source["mask"], allow_refresh=refresh)
        _save_checked(owners, output_root / source["word_owner_annotation"], allow_refresh=refresh)
        row = dict(source)
        row["mask_sha256"] = file_sha256(output_root / source["mask"])
        row["word_owner_annotation_sha256"] = file_sha256(
            output_root / source["word_owner_annotation"]
        )
        records.append(row)
        audits.append(asdict(audit))
        if number % 100 == 0 or number == len(base["samples"]):
            print(f"[refined curved-word target] {number}/1255", flush=True)
    manifest = dict(base)
    manifest["split"] = "official-train-curved-word-regions-pixel-refined-v2"
    manifest["acquisition"] = {
        "classification": "ASSUMPTION",
        "base_word_region_manifest_sha256": file_sha256(base_manifest_path),
        "distributed_pixel_gt_manifest_sha256": file_sha256(pixel_manifest_path),
        "refinement": "minimal-connected-distance-envelope-within-disjoint-word-owner-with-native128-topology-guard",
    }
    manifest["annotation_policy"] = dict(base["annotation_policy"])
    manifest["annotation_policy"]["target_source"] = "totaltext_word_polygons_pixel_gt_refined_envelope_v2"
    manifest["annotation_policy"]["target_classification"] = "ASSUMPTION"
    manifest["annotation_policy"]["word_owner_cover_mode"] = "owner_equals_target"
    manifest["samples"] = records
    manifest_path = output_root / "manifests/total_text_train.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(manifest, sort_keys=False).encode("utf-8")
    if manifest_path.exists() and manifest_path.read_bytes() != content and not refresh:
        raise ValueError(f"Existing manifest differs: {manifest_path}")
    manifest_path.write_bytes(content)
    qualification = {
        "classification": "PRACTICAL_APPROXIMATION",
        "images": len(records),
        "words": sum(row["word_count"] for row in audits),
        "words_without_pixel_gt_seed": sum(row["words_with_no_pixel_gt_seed"] for row in audits),
        "words_with_disconnected_owner_fallback": sum(row["words_with_disconnected_owner_fallback"] for row in audits),
        "native_connectivity_reverted_words": sum(row["native_connectivity_reverted_words"] for row in audits),
        "original_owner_pixels": sum(row["original_owner_pixels"] for row in audits),
        "refined_owner_pixels": sum(row["refined_owner_pixels"] for row in audits),
        "pixel_gt_seed_pixels": sum(row["pixel_gt_seed_pixels"] for row in audits),
        "pixel_gt_seed_retained": sum(row["pixel_gt_seed_retained"] for row in audits),
        "maximum_radius_pixels_any_word": max(row["max_radius_pixels"] for row in audits),
        "base_manifest_sha256": file_sha256(base_manifest_path),
        "pixel_gt_manifest_sha256": file_sha256(pixel_manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "per_image": {row["id"]: audit for row, audit in zip(records, audits)},
    }
    report_path = output_root / "qualification.json"
    report_content = (json.dumps(qualification, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if report_path.exists() and report_path.read_bytes() != report_content and not refresh:
        raise ValueError(f"Existing qualification differs: {report_path}")
    report_path.write_bytes(report_content)
    print("MANIFEST", manifest_path, file_sha256(manifest_path))
    print("QUALIFICATION", report_path, file_sha256(report_path))
    return qualification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--pixel-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="Refresh only this generated v2 target")
    args = parser.parse_args()
    prepare(args.base_root, args.pixel_root, args.output_root, refresh=args.refresh)


if __name__ == "__main__":
    main()
