"""Audit curved-box v4 TRAIN targets and native-128 word connectivity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from segmentation.scene_text_preparation import file_sha256
from segmentation.totaltext_word_regions import _native_word_component_counts


def qualify(base_root: Path, box_root: Path, output: Path) -> dict:
    old_path = base_root / "manifests/total_text_train.yaml"
    new_path = box_root / "manifests/total_text_train.yaml"
    old = yaml.safe_load(old_path.read_text(encoding="utf-8"))
    new = yaml.safe_load(new_path.read_text(encoding="utf-8"))
    if len(old["samples"]) != 1255 or len(new["samples"]) != 1255:
        raise ValueError("Expected 1255 aligned TRAIN samples")
    distribution = {name: {"zero": 0, "one": 0, "multi": 0} for name in ("v1", "curved_box_v4")}
    words = improved = worsened = touch_count = 0
    for number, (before, after) in enumerate(zip(old["samples"], new["samples"]), 1):
        if before["id"] != after["id"] or before["image_sha256"] != after["image_sha256"]:
            raise ValueError(f"TRAIN alignment mismatch at {number}")
        for root, row, key in (
            (base_root, before, "word_owner_annotation"),
            (base_root, before, "separation_annotation"),
            (box_root, after, "word_owner_annotation"),
            (box_root, after, "mask"),
            (box_root, after, "validity_annotation"),
            (box_root, after, "separation_annotation"),
        ):
            path = root / row[key]
            if file_sha256(path) != row[f"{key}_sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
        with Image.open(base_root / before["word_owner_annotation"]) as image:
            old_owner = np.asarray(image, dtype=np.uint16).copy()
        with Image.open(box_root / after["word_owner_annotation"]) as image:
            new_owner = np.asarray(image, dtype=np.uint16).copy()
        with Image.open(box_root / after["mask"]) as image:
            mask = np.asarray(image.convert("L")) > 0
        with Image.open(box_root / after["validity_annotation"]) as valid:
            with Image.open(box_root / after["separation_annotation"]) as separator:
                with Image.open(base_root / before["separation_annotation"]) as old_sep:
                    valid_array = np.asarray(valid.convert("L")) > 0
                    sep_array = np.asarray(separator.convert("L")) > 0
                    old_counts = _native_word_component_counts(
                        old_owner, separator=old_sep, validity=valid
                    )
                    new_counts = _native_word_component_counts(
                        new_owner, separator=separator, validity=valid
                    )
        if not np.array_equal(mask, new_owner > 0):
            raise ValueError(f"Mask/owner mismatch: {after['id']}")
        if np.any((new_owner > 0) & (~valid_array | sep_array)):
            raise ValueError(f"Target includes ignored/separator pixels: {after['id']}")
        if old_counts.keys() != new_counts.keys():
            raise ValueError(f"Word labels changed: {after['id']}")
        for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
            a = new_owner[max(0, -dy):new_owner.shape[0]-max(0, dy),
                          max(0, -dx):new_owner.shape[1]-max(0, dx)]
            b = new_owner[max(0, dy):new_owner.shape[0]-max(0, -dy),
                          max(0, dx):new_owner.shape[1]-max(0, -dx)]
            touch_count += int(np.count_nonzero((a > 0) & (b > 0) & (a != b)))
        for word_id, count in old_counts.items():
            current = new_counts[word_id]
            words += 1
            for name, value in (("v1", count), ("curved_box_v4", current)):
                distribution[name]["zero" if value == 0 else "one" if value == 1 else "multi"] += 1
            worsened += int(count > 0 and (current > count or current == 0))
            improved += int(current > 0 and (current < count or count == 0))
        if number % 100 == 0 or number == 1255:
            print(f"[v4 native-128 qualification] {number}/1255", flush=True)
    if words != 9285 or worsened or touch_count:
        raise ValueError(f"v4 qualification failed: words={words} worse={worsened} contacts={touch_count}")
    report = {
        "classification": "PRACTICAL_APPROXIMATION",
        "split": "Total-Text TRAIN",
        "images": 1255,
        "words": words,
        "native_resolution": [128, 128],
        "distributions": distribution,
        "native_words_improved": improved,
        "native_words_worsened": worsened,
        "source_8_neighbor_interword_contacts": touch_count,
        "base_manifest_sha256": file_sha256(old_path),
        "curved_box_manifest_sha256": file_sha256(new_path),
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != content:
        raise ValueError(f"Existing qualification differs: {output}")
    if not output.exists():
        output.write_bytes(content)
    print("QUALIFICATION", output, file_sha256(output), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--box-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qualify(args.base_root, args.box_root, args.output)


if __name__ == "__main__":
    main()
