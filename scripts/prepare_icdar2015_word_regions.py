"""Prepare and audit isolated ICDAR2015 TRAIN word masks using Total-Text operations."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
from PIL import Image
import yaml
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from recognition.scene_text_datasets import parse_icdar2015_annotation
from segmentation.data import file_sha256, load_segmentation_manifest, SegmentationDataset
from segmentation.icdar2015_transfer import load_config, make_targets, split_indices, transform
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


def target_topology(mask, owners, validity, count):
    """Audit all care words, including vanished words, with 8-connectivity."""
    mask = np.asarray(mask).astype(bool) & np.asarray(validity).astype(bool)
    owners = np.asarray(owners).copy()
    owners[~mask] = 0
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    associations = {}
    lost = fragmented = 0
    for word_id in range(1, count + 1):
        components = set(int(x) for x in np.unique(labels[owners == word_id]) if x > 0)
        lost += not bool(components)
        fragmented += len(components) > 1
        for component in components:
            associations.setdefault(component, set()).add(word_id)
    fused = [ids for ids in associations.values() if len(ids) > 1]
    return {"care_words": count, "lost_words": int(lost), "fragmented_words": int(fragmented),
            "components": int(labels.max()), "fused_components": len(fused),
            "words_in_fused_components": len(set().union(*fused)) if fused else 0}


def prepare(source_root, output_root, config_path):
    config = load_config(config_path)
    source_root, output_root = source_root.resolve(), output_root.resolve()
    if output_root == source_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Word-target output must be isolated from the source dataset")
    if output_root.exists():
        raise ValueError(f"Output already exists; preserve it and choose a new directory: {output_root}")
    manifest_path = source_root / config["source_manifest"]
    if file_sha256(manifest_path) != config["source_manifest_sha256"]:
        raise ValueError("Original ICDAR2015 TRAIN manifest SHA256 differs")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    train_ids, validation_ids = split_indices(manifest, config)
    # Verify every original RGB/target/validity hash before deriving any output.
    SegmentationDataset(manifest, source_root, verify_checksums=True)
    source = yaml.safe_load(manifest_path.read_text())
    paired = transform(config, augment=False)
    old = PairedSegmentationTransform(SegmentationTransformConfig(
        horizontal_flip_probability=0, rotation_probability=0, rotation_degrees=(-15, 15),
        image_interpolation="bilinear", mask_interpolation="bilinear", image_fill=(0, 0, 0),
        mask_fill=0, normalization="imagenet", resize_geometry="stretch",
        operation_order="flip_rotate_resize"), base_seed=config["training"]["seed"])
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-", dir=output_root.parent))
    records, audits, counts = [], {}, {}
    # A failed stage is preserved for diagnosis; no qualified output is published.
    for index, (sample, row) in enumerate(zip(manifest.samples, source["samples"])):
        annotation = Path(row["image"]).parent.parent / "annotations" / (sample.sample_id.split(":", 1)[1] + ".txt")
        annotation_path = source_root / annotation
        with Image.open(source_root / sample.image_path) as image:
            image = image.convert("RGB")
        instances = parse_icdar2015_annotation(annotation_path, image_size_wh=image.size)
        target, validity, separator, owners, audit = make_targets(image.size, instances)
        with Image.open(source_root / sample.validity_annotation_path) as original_valid:
            if not np.array_equal(np.asarray(validity), np.asarray(original_valid)):
                raise ValueError(f"Don't Care validity changed: {sample.sample_id}")
        count = sum(not item.ignore for item in instances)
        counts[sample.sample_id] = count
        record = dict(row)
        for rel in (Path(row["image"]), annotation):
            destination = stage / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Copies avoid mutable hard links back into the original dataset.
            shutil.copyfile(source_root / rel, destination)
        record["source_annotation"] = annotation.as_posix()
        record["source_annotation_sha256"] = file_sha256(annotation_path)
        stem = sample.sample_id.split(":", 1)[1] + ".png"
        for key, folder, value in (("mask", "masks", target),
                                  ("validity_annotation", "validity", validity),
                                  ("separation_annotation", "separators", separator),
                                  ("word_owner_annotation", "owners", owners)):
            relative = Path("datasets/icdar2015/train") / folder / stem
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            value.save(destination)
            record[key], record[key + "_sha256"] = relative.as_posix(), file_sha256(destination)
        owner_i = owners.convert("I")
        _, new_target, new_valid, _ = paired.transform_with_validity_and_separation(
            image, target, validity, separator, owner_i, index)
        _, new_owner, _ = paired._augment(image, owner_i, None, index)
        new_owner = np.asarray(TF.resize(new_owner, [256, 256], interpolation=InterpolationMode.NEAREST))
        with Image.open(source_root / sample.mask_path) as source_target:
            _, old_target, old_valid = old.transform_with_validity(image, source_target.convert("L"), validity, index)
        old_owner = np.asarray(TF.resize(owner_i, [128, 128], interpolation=InterpolationMode.NEAREST))
        audits[sample.sample_id] = {"source": asdict(audit),
            "original_native128_stretch": target_topology(old_target[0], old_owner, old_valid[0], count),
            "transferred_native256_letterbox": target_topology(new_target[0], new_owner, new_valid[0], count)}
        records.append(record)
        if (index + 1) % 50 == 0 or index + 1 == len(manifest.samples):
            print(f"[ICDAR2015 word targets] {index + 1}/{len(manifest.samples)}", flush=True)
    derived = dict(source)
    derived.update(split="official-train-disjoint-word-regions-totaltext-transfer-v1", samples=records)
    derived["acquisition"] = {"classification": "ASSUMPTION",
        "source_manifest_sha256": config["source_manifest_sha256"],
        "config_sha256": file_sha256(config_path), "care_words_by_id": counts}
    derived["annotation_policy"] = {**source["annotation_policy"],
        "target_source": "official_icdar2015_quadrilaterals_disjoint_totaltext_word_regions",
        "target_classification": "ASSUMPTION", "separation_map": "contact_and_short_interword_gaps",
        "word_owner_encoding": "uint16_png_background_zero"}
    destination_manifest = stage / config["source_manifest"]
    destination_manifest.parent.mkdir(parents=True, exist_ok=True)
    destination_manifest.write_text(yaml.safe_dump(derived, sort_keys=False))
    totals = {}
    for name in ("original_native128_stretch", "transferred_native256_letterbox"):
        totals[name] = {key: sum(row[name][key] for row in audits.values())
                        for key in next(iter(audits.values()))[name]}
    report = {"schema_version": 1, "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction", "result_class": "TRAIN target qualification only",
        "source_manifest_sha256": config["source_manifest_sha256"],
        "manifest_sha256": file_sha256(destination_manifest), "images": len(records),
        "care_words": sum(counts.values()), "aggregate": totals,
        "source_empty_words": sum(row["source"]["empty_word_count"] for row in audits.values()),
        "train_ids": [manifest.samples[i].sample_id for i in train_ids],
        "validation_ids": [manifest.samples[i].sample_id for i in validation_ids],
        "per_image": audits,
        "limitations": "Target topology audit only; resolution, aspect ratio and separation change together. Not model accuracy."}
    (stage / "qualification.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    stage.rename(output_root)
    print(json.dumps({k: v for k, v in report.items() if k not in {"per_image", "train_ids", "validation_ids"}}, sort_keys=True), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source_root, args.output_root, args.config)


if __name__ == "__main__":
    main()
