"""Verify filled v3 TRAIN masks and per-word topology at native 128x128."""

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
    base_path = base_root / "manifests/total_text_train.yaml"
    box_path = box_root / "manifests/total_text_train.yaml"
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    boxes = yaml.safe_load(box_path.read_text(encoding="utf-8"))
    if len(base["samples"]) != 1255 or len(boxes["samples"]) != 1255:
        raise ValueError("Expected exactly 1255 aligned TRAIN samples")
    distribution = {name: {"zero": 0, "one": 0, "multi": 0} for name in ("v1", "box_v3")}
    words = improved = worsened = 0
    for number, (old, new) in enumerate(zip(base["samples"], boxes["samples"]), 1):
        if old["id"] != new["id"] or old["image_sha256"] != new["image_sha256"]:
            raise ValueError(f"TRAIN alignment mismatch at row {number}")
        for root, row, key in (
            (base_root, old, "word_owner_annotation"),
            (box_root, new, "word_owner_annotation"),
            (box_root, new, "mask"),
            (box_root, new, "validity_annotation"),
            (box_root, new, "separation_annotation"),
        ):
            path = root / row[key]
            if file_sha256(path) != row[f"{key}_sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
        with Image.open(base_root / old["word_owner_annotation"]) as source:
            old_owner = np.asarray(source, dtype=np.uint16).copy()
        with Image.open(box_root / new["word_owner_annotation"]) as target:
            new_owner = np.asarray(target, dtype=np.uint16).copy()
        with Image.open(box_root / new["mask"]) as target:
            new_mask = np.asarray(target.convert("L")) > 0
        if np.any((new_owner > 0) & (new_owner != old_owner)):
            raise ValueError(f"Owner changed or boxes overlap for {new['id']}")
        if not np.array_equal(new_mask, new_owner > 0):
            raise ValueError(f"Mask/owner mismatch for {new['id']}")
        with Image.open(box_root / new["validity_annotation"]) as valid:
            with Image.open(box_root / new["separation_annotation"]) as separator:
                previous = _native_word_component_counts(old_owner, separator=separator, validity=valid)
                current = _native_word_component_counts(new_owner, separator=separator, validity=valid)
        if previous.keys() != current.keys():
            raise ValueError(f"Word IDs changed for {new['id']}")
        for word_id, before in previous.items():
            after = current[word_id]
            words += 1
            for name, value in (("v1", before), ("box_v3", after)):
                category = "zero" if value == 0 else "one" if value == 1 else "multi"
                distribution[name][category] += 1
            worsened += int(before > 0 and (after > before or after == 0))
            improved += int(after > 0 and (after < before or before == 0))
        if number % 100 == 0 or number == 1255:
            print(f"[v3 native-128 qualification] {number}/1255", flush=True)
    if words != 9285 or worsened:
        raise ValueError(f"Word count/topology failed: words={words}, worsened={worsened}")
    report = {
        "classification": "PRACTICAL_APPROXIMATION",
        "split": "Total-Text TRAIN",
        "images": 1255,
        "words": words,
        "native_resolution": [128, 128],
        "distributions": distribution,
        "native_words_improved": improved,
        "native_words_worsened": worsened,
        "base_manifest_sha256": file_sha256(base_path),
        "box_manifest_sha256": file_sha256(box_path),
    }
    content = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
    if output.exists() and output.read_bytes() != content:
        raise ValueError(f"Existing qualification differs: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
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
