"""Audit isolated Total-Text refined TRAIN targets at native 128x128 resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from segmentation.scene_text_preparation import file_sha256
from segmentation.totaltext_word_regions import _native_word_component_counts


def qualify(base_root: Path, refined_root: Path, output: Path) -> dict:
    base_path = base_root / "manifests/total_text_train.yaml"
    refined_path = refined_root / "manifests/total_text_train.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    refined = yaml.safe_load(refined_path.read_text(encoding="utf-8"))
    if len(base["samples"]) != 1255 or len(refined["samples"]) != 1255:
        raise ValueError("Expected exactly 1255 TRAIN samples")
    distributions = {name: {"zero": 0, "one": 0, "multi": 0} for name in ("v1", "refined_v2")}
    worsened = improved = words = base_pixels = refined_pixels = 0
    for offset, (before, after) in enumerate(zip(base["samples"], refined["samples"]), 1):
        if before["id"] != after["id"] or before["image_sha256"] != after["image_sha256"]:
            raise ValueError(f"TRAIN alignment mismatch at index {offset}")
        paths = [
            (base_root, before, "word_owner_annotation"),
            (refined_root, after, "word_owner_annotation"),
            (refined_root, after, "mask"),
            (refined_root, after, "validity_annotation"),
            (refined_root, after, "separation_annotation"),
        ]
        for root, record, key in paths:
            source = root / record[key]
            if file_sha256(source) != record[f"{key}_sha256"]:
                raise ValueError(f"Checksum mismatch: {source}")
        with Image.open(base_root / before["word_owner_annotation"]) as base_image:
            base_owner = np.asarray(base_image, dtype=np.uint16).copy()
        with Image.open(refined_root / after["word_owner_annotation"]) as refined_image:
            new_owner = np.asarray(refined_image, dtype=np.uint16).copy()
        with Image.open(refined_root / after["mask"]) as mask_image:
            new_mask = np.asarray(mask_image.convert("L")) > 0
        if np.any((new_owner > 0) & (new_owner != base_owner)):
            raise ValueError(f"Refined owner escaped v1 polygon: {before['id']}")
        if not np.array_equal(new_mask, new_owner > 0):
            raise ValueError(f"Refined target differs from owner map: {before['id']}")
        with Image.open(refined_root / after["validity_annotation"]) as valid:
            with Image.open(refined_root / after["separation_annotation"]) as separator:
                old_counts = _native_word_component_counts(base_owner, separator=separator, validity=valid)
                new_counts = _native_word_component_counts(new_owner, separator=separator, validity=valid)
        if old_counts.keys() != new_counts.keys():
            raise ValueError(f"Word-owner labels changed: {before['id']}")
        base_pixels += int((base_owner > 0).sum())
        refined_pixels += int((new_owner > 0).sum())
        for index, count in old_counts.items():
            newer = new_counts[index]
            words += 1
            for name, value in (("v1", count), ("refined_v2", newer)):
                category = "zero" if value == 0 else "one" if value == 1 else "multi"
                distributions[name][category] += 1
            if count > 0 and (newer > count or newer == 0):
                worsened += 1
            if newer > 0 and (newer < count or count == 0):
                improved += 1
        if offset % 100 == 0 or offset == 1255:
            print(f"[native-128 qualification] {offset}/1255", flush=True)
    if worsened:
        raise ValueError(f"Native-128 topology worsened for {worsened} words")
    report = {
        "classification": "PRACTICAL_APPROXIMATION",
        "split": "Total-Text TRAIN",
        "images": len(base["samples"]),
        "words": words,
        "native_resolution": [128, 128],
        "base_manifest_sha256": file_sha256(base_path),
        "refined_manifest_sha256": file_sha256(refined_path),
        "distributions": distributions,
        "native_words_worsened": worsened,
        "native_words_improved": improved,
        "base_owner_pixels": base_pixels,
        "refined_owner_pixels": refined_pixels,
        "refined_area_fraction_of_base": refined_pixels / base_pixels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("QUALIFICATION", output, file_sha256(output), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--refined-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.base_root, args.refined_root, args.output)


if __name__ == "__main__":
    main()
