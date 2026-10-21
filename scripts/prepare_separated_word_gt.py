"""Export isolated, auditable masks of separated words from downloaded GT."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw
import yaml

from recognition.icdar13_annotations import parse_task1_annotation
from recognition.scene_text_datasets import (
    SceneTextInstance, parse_icdar2015_annotation, parse_totaltext_annotation,
)
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent, file_sha256
from segmentation.separated_word_gt import (
    audit_word_owners, parse_icdar13_character_groups, separated_word_regions, widen_word_gaps,
)


EXPECTED = {"icdar13": {"train": 229, "test": 233},
            "icdar2015": {"train": 1000, "test": 500},
            "total_text": {"train": 1255, "test": 300}}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_samples(download_root: Path, prepared_root: Path, dataset: str, split: str):
    if dataset == "icdar13":
        root = download_root / "icdar13/raw"
        images = sorted((root / f"{split}_images").glob("*.jpg"),
                        key=lambda p: int(p.stem.removeprefix("img_")))
        records = []
        for path in images:
            sample_id = path.stem.removeprefix("img_")
            annotation = (root / "train_segmentation_gt" / f"{sample_id}_GT.txt"
                          if split == "train" else root / "test_task1_gt" / f"gt_img_{sample_id}.txt")
            records.append((sample_id, path, annotation, None))
        return records, {"source": str(root), "classification": "REPOSITORY_SPECIFIED"}
    manifest_path = prepared_root / "manifests" / f"{dataset}_{split}.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if manifest["dataset"] != dataset or manifest["split_role"] != split:
        raise ValueError(f"Source manifest dataset/split mismatch: {manifest_path}")
    records = []
    for record in manifest["samples"]:
        sample_id = record["id"].split(":")[-1]
        annotation = prepared_root / "datasets" / dataset / split / "annotations" / f"{sample_id}.txt"
        records.append((sample_id, prepared_root / record["image"], annotation, record["image_sha256"]))
    return records, {"source_manifest": str(manifest_path),
                     "source_manifest_sha256": file_sha256(manifest_path),
                     "source_acquisition": manifest["acquisition"]}


def parse_instances(dataset: str, split: str, annotation: Path, size_wh: tuple[int, int]):
    if dataset == "icdar13":
        if split == "train":
            return parse_icdar13_character_groups(annotation)
        return tuple(SceneTextInstance(
            ((x0, y0), (x1, y0), (x1, y1), (x0, y1)), word.transcription, False, "word"
        ) for word in parse_task1_annotation(annotation, image_size_wh=size_wh)
            for x0, y0, x1, y1 in [word.bbox_xyxy])
    if dataset == "icdar2015":
        return parse_icdar2015_annotation(annotation)
    return parse_totaltext_annotation(annotation)


def preview(image_path: Path, owners: Image.Image, mask: Image.Image, destination: Path,
            title: str, baseline: Image.Image | None = None) -> None:
    with Image.open(image_path) as source:
        original = source.convert("RGB")
    labels = np.asarray(owners, dtype=np.uint16)
    colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for index in np.unique(labels[labels > 0]):
        # Deterministic high-contrast palette; visualization only.
        colored[labels == index] = (64 + (int(index) * 79) % 192,
                                   64 + (int(index) * 137) % 192,
                                   64 + (int(index) * 173) % 192)
    overlay = Image.blend(original, Image.fromarray(colored), 0.45)
    panels = [original, mask.convert("RGB"), overlay]
    names = ["Image", "Separated word mask", "One color per word"]
    if baseline is not None:
        panels.insert(1, baseline.convert("RGB"))
        names.insert(1, "V4 baseline")
    cell_width = 420
    cell_height = max(1, round(original.height * cell_width / original.width))
    output = Image.new("RGB", (len(panels) * cell_width, cell_height + 42), "white")
    draw = ImageDraw.Draw(output)
    draw.text((8, 4), title, fill="black")
    for offset, (panel, name) in enumerate(zip(panels, names)):
        draw.text((offset * cell_width + 8, 23), name, fill="black")
        output.paste(panel.resize((cell_width, cell_height), Image.Resampling.NEAREST),
                     (offset * cell_width, 42))
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination)


def prepare(download_root: Path, prepared_root: Path, output_root: Path, *,
            dataset: str, split: str, ignore_policy: str, v4_root: Path,
            clearance: float, reference_wh: tuple[int, int], max_area_loss: float,
            limit: int | None = None) -> dict:
    source, acquisition = source_samples(download_root, prepared_root, dataset, split)
    if len(source) != EXPECTED[dataset][split]:
        raise ValueError(f"{dataset}/{split}: expected {EXPECTED[dataset][split]} images, got {len(source)}")
    source = source[:limit] if limit is not None else source
    v4_rows = {}
    if dataset == "total_text" and split == "train":
        v4_manifest = v4_root / "manifests/total_text_train.yaml"
        v4 = yaml.safe_load(v4_manifest.read_text(encoding="utf-8"))
        if v4["annotation_policy"]["target_source"] != "totaltext_filled_curved_box_v4":
            raise ValueError("Expected frozen V4 full-word targets")
        v4_rows = {r["id"].split(":")[-1]: r for r in v4["samples"]}
        if len(v4_rows) != EXPECTED[dataset][split]:
            raise ValueError("V4 must contain all1255 TRAIN images")
        acquisition["v4_manifest"] = str(v4_manifest)
        acquisition["v4_manifest_sha256"] = file_sha256(v4_manifest)
    dataset_dir = output_root / dataset / split
    if dataset_dir.exists():
        raise ValueError(f"Refusing to overwrite generated masks: {dataset_dir}")
    dataset_dir.mkdir(parents=True)
    config = {"schema_version": 1, "reproduction_mode": "approximate",
              "claim_label": "approximate reproduction", "classification": "ASSUMPTION",
              "result_class": "smoke-test result" if limit is not None else "approximate reproduction",
              "dataset": dataset, "split": split, "expected_source_images": EXPECTED[dataset][split],
              "requested_images": len(source), "download_root": str(download_root),
              "prepared_root": str(prepared_root), "output_root": str(output_root),
              "algorithm": "v4-or-source-word-owners-guarded-neighbour-clearance-v1",
              "connectivity": 8, "resolution": "original decoded image resolution",
              "word_identity": "source annotation order, care words numbered from 1",
              "ignore_policy": "unchanged frozen V4 validity" if v4_rows else ignore_policy,
              "v4_root": str(v4_root), "clearance_reference_pixels": clearance,
              "reference_wh": list(reference_wh), "max_word_area_loss_fraction": max_area_loss,
              "protected_pixels": "source stroke GT for Total-Text and ICDAR13; unavailable for ICDAR2015",
              "island_tie_break": "first component in raster order",
              "ambiguity_id": "WORD-GT-001", "source_provenance": acquisition,
              "hardware": {"device": "cpu", "gpu_inference": False},
              "randomness": {"random_sampling": False, "seed": None,
                             "sample_order": "source manifest order; numeric ID for ICDAR13"},
              "software": {"python": platform.python_version(), "numpy": np.__version__},
              "code_sha256": {str(path): file_sha256(path) for path in [
                  Path(__file__), Path("segmentation/separated_word_gt.py"),
                  Path("segmentation/totaltext_word_regions.py"),
                  Path("segmentation/totaltext_curved_box_v4.py"), Path("segmentation/data.py"),
                  Path("recognition/scene_text_datasets.py"),
                  Path("recognition/icdar13_annotations.py")]}}
    write_json(dataset_dir / "resolved_config.json", config)
    records = []
    total_words = total_ignored = total_removed = total_gap_removed = total_gap_changed = total_fragmented = 0
    # Previews include first examples and the largest source overlaps (display only).
    overlap_previews = []
    for offset, (sample_id, image_path, annotation, frozen_image_hash) in enumerate(source, 1):
        try:
            image_hash = file_sha256(image_path)
            if frozen_image_hash is not None and image_hash != frozen_image_hash:
                raise ValueError(f"Source image changed: {image_path}")
            with Image.open(image_path) as opened:
                size = opened.size
            annotation_hash = file_sha256(annotation)
            instances, clipped = clip_instances_to_pixel_extent(
                parse_instances(dataset, split, annotation, size), size)
            protected = None
            if dataset == "total_text":
                pixel_path = prepared_root / "datasets" / dataset / split / "masks" / f"{sample_id}.png"
                with Image.open(pixel_path) as pixel:
                    protected = pixel.copy()
            elif dataset == "icdar13":
                from segmentation.data import (icdar13_task22_rgb_to_target_and_validity,
                                               icdar13_task22_rgb_to_binary_mask)
                if split == "train":
                    pixel_path = annotation.with_suffix(".bmp")
                    with Image.open(pixel_path) as pixel:
                        target, valid = icdar13_task22_rgb_to_target_and_validity(
                            pixel, annotation.read_text(encoding="utf-8-sig"))
                    protected = Image.fromarray(np.where(np.asarray(valid)>0, np.asarray(target), 0).astype(np.uint8))
                else:
                    pixel_path = download_root / "icdar13/raw/test_segmentation_gt" / f"gt_img_{sample_id}.png"
                    with Image.open(pixel_path) as pixel:
                        protected = icdar13_task22_rgb_to_binary_mask(pixel)
            baseline = None
            v4_provenance = None
            if v4_rows:
                row = v4_rows[sample_id]
                if image_hash != row["image_sha256"]:
                    raise ValueError(f"V4 source image differs: {sample_id}")
                if file_sha256(v4_root / "datasets/total_text/train/annotations" / f"{sample_id}.txt") != annotation_hash:
                    raise ValueError(f"V4 word annotation differs: {sample_id}")
                loaded = {}
                for key in ["mask", "word_owner_annotation", "validity_annotation", "separation_annotation"]:
                    path = v4_root / row[key]
                    if file_sha256(path) != row[key + "_sha256"]:
                        raise ValueError(f"V4 artifact checksum mismatch: {path}")
                    with Image.open(path) as item:
                        loaded[key] = item.copy()
                mask, owners = loaded["mask"], loaded["word_owner_annotation"]
                validity, separator = loaded["validity_annotation"], loaded["separation_annotation"]
                baseline = mask.copy()
                count = sum(not x.ignore for x in instances)
                audit = audit_word_owners(np.asarray(owners), count)
                audit.update({"ignored_instances": sum(x.ignore for x in instances),
                              "secondary_island_pixels_removed": 0, "source_region_audit": None})
                v4_provenance = {"owner_sha256": row["word_owner_annotation_sha256"],
                                 "mask_sha256": row["mask_sha256"]}
            else:
                mask, validity, separator, owners, audit = separated_word_regions(
                    size, instances, ignore_policy=ignore_policy)
            before_owners = np.asarray(owners).copy()
            mask, owners, gap_audit = widen_word_gaps(
                owners, clearance=clearance, reference_wh=reference_wh,
                max_area_loss=max_area_loss, protected_pixels=protected)
            removed = (before_owners > 0) & (np.asarray(owners) == 0)
            sep = np.asarray(separator).copy()
            sep[removed & (np.asarray(validity) > 0)] = 255
            separator = Image.fromarray(sep)
            native_before = audit_word_owners(np.asarray(Image.fromarray(before_owners.astype(np.int32), mode="I").resize(
                reference_wh, Image.Resampling.NEAREST)), audit["words"])
            native_after = audit_word_owners(np.asarray(owners.resize(reference_wh, Image.Resampling.NEAREST)), audit["words"])
            record = {"id": f"{dataset}:{sample_id}", "image": str(image_path),
                      "image_sha256": image_hash, "source_annotation": str(annotation),
                      "source_annotation_sha256": annotation_hash, "size_wh": list(size),
                      "clipped_source_instances": clipped, "audit": audit, "gap_audit": gap_audit,
                      "reference_resize_before": native_before, "reference_resize_after": native_after,
                      "v4_provenance": v4_provenance,
                      "protected_pixel_gt": str(pixel_path) if protected is not None else None,
                      "protected_pixel_gt_sha256": file_sha256(pixel_path) if protected is not None else None,
                      "instances": [{"owner_id": i, **asdict(instance)} for i, instance in
                                    enumerate([x for x in instances if not x.ignore], 1)],
                      "ignored_instances": [asdict(x) for x in instances if x.ignore]}
            for name, item in [("masks", mask), ("validity", validity),
                               ("separators", separator), ("owners", owners)]:
                path = dataset_dir / name / f"{sample_id}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                item.save(path, format="PNG")
                # Roundtrip ensures 16-bit ownership was preserved by the writer.
                with Image.open(path) as saved:
                    if not np.array_equal(np.asarray(saved), np.asarray(item)):
                        raise ValueError(f"PNG roundtrip changed target pixels: {path}")
                record[name] = path.relative_to(output_root).as_posix()
                record[name + "_sha256"] = file_sha256(path)
            write_json(dataset_dir / "annotations" / f"{sample_id}.json", record)
            records.append(record)
            total_words += audit["words"]
            total_ignored += audit["ignored_instances"]
            total_removed += audit["secondary_island_pixels_removed"]
            total_gap_removed += gap_audit["removed_pixels"]
            total_gap_changed += gap_audit["changed_words"]
            total_fragmented += len(gap_audit["fragmented_word_ids"])
            overlap = gap_audit["removed_pixels"]
            overlap_previews.append((overlap, sample_id, image_path))
            if offset <= 3:
                preview(image_path, owners, mask, dataset_dir / "previews" / f"{sample_id}.png",
                        f"{dataset} {split} {sample_id}: {audit['words']} words", baseline=baseline)
        except Exception as error:
            # Preserve and expose the failure; never continue with a missing sample.
            write_json(dataset_dir / "failure.json", {"sample_id": sample_id,
                       "completed_images": len(records), "requested_images": len(source),
                       "error_type": type(error).__name__, "error": str(error)})
            raise
        if offset % 50 == 0 or offset == len(source):
            print(f"{dataset}/{split}: {offset}/{len(source)} images, {total_words} words", flush=True)
    for _, sample_id, image_path in sorted(overlap_previews, key=lambda x: (-x[0], x[1]))[:3]:
        with Image.open(dataset_dir / "owners" / f"{sample_id}.png") as owners, \
                Image.open(dataset_dir / "masks" / f"{sample_id}.png") as mask:
            baseline = None
            if v4_rows:
                with Image.open(v4_root / v4_rows[sample_id]["mask"]) as old:
                    baseline = old.copy()
            preview(image_path, owners, mask, dataset_dir / "previews" / f"overlap_{sample_id}.png",
                    f"{dataset} {split} {sample_id}: enlarged gaps example", baseline=baseline)
    report = {"schema_version": 1, "claim_label": "approximate reproduction",
              "classification": "ASSUMPTION", "dataset": dataset, "split": split,
              "result_class": config["result_class"], "complete_official_split": limit is None,
              "images": len(records), "words": total_words, "ignored_instances": total_ignored,
              "missing_words": 0, "interword_contact_pairs": 0, "fragmented_words": total_fragmented,
              "gap_pixels_removed": total_gap_removed, "gap_changed_words": total_gap_changed,
              "secondary_island_pixels_removed": total_removed,
              "resolved_config_sha256": file_sha256(dataset_dir / "resolved_config.json"),
              "samples": records}
    write_json(dataset_dir / "manifest.json", report)
    return {k: v for k, v in report.items() if k != "samples"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["approximate"], required=True)
    parser.add_argument("--download-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=list(EXPECTED), nargs="+", required=True)
    parser.add_argument("--split", choices=["train", "test"], nargs="+", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ignore-policy", required=True,
                        choices=["care_region_precedence", "ignore_region_precedence"])
    parser.add_argument("--v4-root", type=Path, required=True)
    parser.add_argument("--clearance", type=float, required=True)
    parser.add_argument("--reference-size", type=int, nargs=2, required=True, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--max-word-area-loss", type=float, required=True)
    parser.add_argument("--zip", action="store_true", help="Package masks and metadata, without source images")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    args.output_root = args.output_root.resolve()
    args.download_root = args.download_root.resolve()
    args.prepared_root = args.prepared_root.resolve()
    args.v4_root = args.v4_root.resolve()
    if any(args.output_root == root or root in args.output_root.parents for root in
           [args.download_root, args.prepared_root]):
        parser.error("Output must be isolated from source dataset trees")
    summaries = []
    for dataset in args.dataset:
        for split in args.split:
            summaries.append(prepare(args.download_root, args.prepared_root, args.output_root,
                                     dataset=dataset, split=split, ignore_policy=args.ignore_policy,
                                     v4_root=args.v4_root, clearance=args.clearance,
                                     reference_wh=tuple(args.reference_size), max_area_loss=args.max_word_area_loss,
                                     limit=args.limit))
    write_json(args.output_root / "qualification.json", {"datasets": summaries})
    if args.zip:
        for dataset in args.dataset:
            destination = args.output_root / f"{dataset}-separated-word-gt.zip"
            if destination.exists():
                raise ValueError(f"Refusing to overwrite archive: {destination}")
            with ZipFile(destination, "w", ZIP_DEFLATED) as archive:
                for path in sorted((args.output_root / dataset).rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(args.output_root))
            print(f"ARCHIVE {destination} SHA256 {file_sha256(destination)}", flush=True)


if __name__ == "__main__":
    main()
