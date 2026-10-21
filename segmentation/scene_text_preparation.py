"""Prepare auditable binary segmentation targets for three scene-text datasets.

ICDAR2015 and CTW1500 expose polygon annotations, so their filled-polygon
targets are an explicit APPROXIMATE assumption.  Total-Text supplies pixel
ground truth; it is thresholded once into canonical lossless PNG targets.
"""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
from typing import Any, Iterable
from zipfile import ZipFile

from PIL import Image, ImageDraw, ImageOps
import numpy as np
import yaml

from recognition.scene_text_datasets import (
    SceneTextInstance,
    parse_icdar2015_annotation,
    parse_totaltext_annotation,
)


class SceneTextPreparationError(ValueError):
    """Raised when downloaded bytes cannot satisfy the frozen preparation contract."""


def _draw_ignored_geometry(draw: ImageDraw.ImageDraw, points: list[tuple[int, int]]) -> None:
    if len(points) == 1:
        draw.point(points, fill=0)
    elif len(points) == 2:
        draw.line(points, fill=0)
    else:
        draw.polygon(points, fill=0)


EXPECTED_COUNTS = {
    "ctw1500": {"train": 1000, "test": 500},
    "icdar2015": {"train": 1000, "test": 500},
    "total_text": {"train": 1255, "test": 300},
}

ARTIFACT_IDS = {
    "ctw1500": "ART-CTW1500-PRACTICAL-SPLIT",
    "icdar2015": "ART-ICDAR15-IST-OFFICIAL-SPLIT",
    "total_text": "ART-TOTALTEXT-OFFICIAL-SPLIT",
}


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_id(name: str, dataset: str) -> str:
    stem = Path(name).stem
    for prefix in ("gt_", "poly_gt_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix) :]
    if dataset == "icdar2015" and stem.lower().startswith("img_"):
        stem = stem[4:]
    if dataset == "total_text" and stem.lower().startswith("img"):
        stem = stem[3:]
    if not stem:
        raise SceneTextPreparationError(f"Cannot derive sample ID from {name}")
    return str(int(stem)) if stem.isascii() and stem.isdecimal() else stem


def _sort_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isascii() and value.isdecimal() else (1, value)


def rasterize_instances(
    size_wh: tuple[int, int], instances: Iterable[SceneTextInstance]
) -> tuple[Image.Image, Image.Image]:
    """Fill care polygons in the target and exclude ignore polygons from BCE."""

    target = Image.new("L", size_wh, 0)
    validity = Image.new("L", size_wh, 255)
    target_draw = ImageDraw.Draw(target)
    validity_draw = ImageDraw.Draw(validity)
    instance_count = 0
    for instance in instances:
        instance_count += 1
        points = list(instance.polygon_xy)
        if instance.ignore:
            _draw_ignored_geometry(validity_draw, points)
        else:
            target_draw.polygon(points, fill=255)
    if instance_count == 0:
        raise SceneTextPreparationError("Cannot rasterize an annotation with no instances")
    return target, validity


def normalize_totaltext_pixel_mask(
    mask: Image.Image,
    *,
    expected_size_wh: tuple[int, int],
    ignored_instances: Iterable[SceneTextInstance],
) -> tuple[Image.Image, Image.Image]:
    """Convert the distributed JPEG pixel mask to canonical binary PNG."""

    if mask.size != expected_size_wh:
        raise SceneTextPreparationError(
            f"Total-Text image/mask geometry mismatch: {expected_size_wh} != {mask.size}"
        )
    source = np.asarray(mask.convert("L"), dtype=np.uint8)
    target = Image.fromarray((source >= 128).astype(np.uint8) * 255, mode="L")
    validity = Image.new("L", expected_size_wh, 255)
    draw = ImageDraw.Draw(validity)
    for instance in ignored_instances:
        if instance.ignore:
            _draw_ignored_geometry(draw, list(instance.polygon_xy))
    return target, validity


def clip_instances_to_pixel_extent(
    instances: Iterable[SceneTextInstance], size_wh: tuple[int, int]
) -> tuple[tuple[SceneTextInstance, ...], int]:
    """Clip source edge coordinates to valid pixel indices and report changes."""

    width, height = size_wh
    output: list[SceneTextInstance] = []
    clipped_count = 0
    for instance in instances:
        points = tuple(
            (min(max(x, 0), width - 1), min(max(y, 0), height - 1))
            for x, y in instance.polygon_xy
        )
        if points != instance.polygon_xy:
            clipped_count += 1
        if not instance.ignore and len(set(points)) < 3:
            raise SceneTextPreparationError("Care polygon degenerates after border clipping")
        output.append(replace(instance, polygon_xy=points))
    return tuple(output), clipped_count


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    image.save(temporary, format="PNG", optimize=False, compress_level=9)
    temporary.replace(path)


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _manifest_record(
    *, root: Path, dataset: str, sample_id: str, image: Path, mask: Path, validity: Path
) -> dict[str, str]:
    return {
        "id": f"{dataset}:{sample_id}",
        "image": image.relative_to(root).as_posix(),
        "mask": mask.relative_to(root).as_posix(),
        "validity_annotation": validity.relative_to(root).as_posix(),
        "image_sha256": file_sha256(image),
        "mask_sha256": file_sha256(mask),
        "validity_annotation_sha256": file_sha256(validity),
    }


def _write_manifest(
    *,
    root: Path,
    dataset: str,
    split: str,
    records: list[dict[str, str]],
    acquisition: dict[str, Any],
    target_source: str,
    target_classification: str,
) -> Path:
    manifest = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "dataset": dataset,
        "split": f"official-{split}-prepared-segmentation-v1",
        "artifact_id": ARTIFACT_IDS[dataset],
        "split_role": "train" if split == "train" else "test",
        "split_policy": {
            "algorithm_id": "official-split-no-holdout-v1",
            "classification": "PAPER_SPECIFIED" if dataset == "total_text" else "ASSUMPTION",
            "validation_count": 0,
        },
        "acquisition": acquisition,
        "annotation_policy": {
            "target_conversion": "binary-mask-with-validity-v1",
            "target_source": target_source,
            "target_classification": target_classification,
            "masked_bce": True,
            "dont_care_training_behavior": "excluded_by_validity_mask",
            "mask_encoding": "lossless_png_values_0_or_255",
            "validity_encoding": "lossless_png_values_0_or_255",
            "training_ready": True,
        },
        "samples": records,
    }
    path = root / "manifests" / f"{dataset}_{split}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _zip_index(archive: ZipFile, suffix: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for name in archive.namelist():
        if name.endswith("/") or not name.lower().endswith(suffix):
            continue
        key = _canonical_id(name, "icdar2015")
        if key in output:
            raise SceneTextPreparationError(f"Duplicate archive sample ID {key}")
        output[key] = name
    return output


def _prepare_icdar2015(downloads: Path, root: Path) -> list[Path]:
    specifications = {
        "train": ("ch4_training_images.zip", "ch4_training_localization_transcription_gt.zip"),
        "test": ("ch4_test_images.zip", "Challenge4_Test_Task1_GT.zip"),
    }
    manifests: list[Path] = []
    archive_hashes = {name: file_sha256(downloads / name) for pair in specifications.values() for name in pair}
    for split, (image_name, annotation_name) in specifications.items():
        records: list[dict[str, str]] = []
        with ZipFile(downloads / image_name) as image_archive, ZipFile(
            downloads / annotation_name
        ) as annotation_archive:
            images = _zip_index(image_archive, ".jpg")
            annotations = _zip_index(annotation_archive, ".txt")
            if set(images) != set(annotations):
                raise SceneTextPreparationError(f"ICDAR2015 {split} image/GT IDs differ")
            if len(images) != EXPECTED_COUNTS["icdar2015"][split]:
                raise SceneTextPreparationError(f"ICDAR2015 {split} count mismatch")
            for sample_id in sorted(images, key=_sort_id):
                base = root / "datasets" / "icdar2015" / split
                image_path = base / "images" / f"{sample_id}.jpg"
                annotation_path = base / "annotations" / f"{sample_id}.txt"
                _write_bytes(image_path, image_archive.read(images[sample_id]))
                _write_bytes(annotation_path, annotation_archive.read(annotations[sample_id]))
                with Image.open(image_path) as opened:
                    size = opened.size
                    opened.verify()
                instances = parse_icdar2015_annotation(annotation_path, image_size_wh=size)
                target, validity = rasterize_instances(size, instances)
                mask_path = base / "masks" / f"{sample_id}.png"
                validity_path = base / "validity" / f"{sample_id}.png"
                _save_png(target, mask_path)
                _save_png(validity, validity_path)
                records.append(
                    _manifest_record(
                        root=root,
                        dataset="icdar2015",
                        sample_id=sample_id,
                        image=image_path,
                        mask=mask_path,
                        validity=validity_path,
                    )
                )
                if len(records) % 100 == 0 or len(records) == len(images):
                    print(
                        f"[prepare] icdar2015 {split}: {len(records)}/{len(images)}",
                        flush=True,
                    )
        manifests.append(
            _write_manifest(
                root=root,
                dataset="icdar2015",
                split=split,
                records=records,
                acquisition={
                    "classification": "UPSTREAM_PRIMARY_CHECKSUM_VERIFIED",
                    "archive_sha256": {name: archive_hashes[name] for name in (image_name, annotation_name)},
                },
                target_source="filled_official_word_quadrilaterals",
                target_classification="ASSUMPTION",
            )
        )
    return manifests


def _ctw_instance(
    raw: dict[str, Any], *, width: int, height: int
) -> tuple[SceneTextInstance, bool]:
    values = raw.get("polygon")
    if not isinstance(values, list) or len(values) < 6 or len(values) % 2:
        raise SceneTextPreparationError("Malformed CTW1500 MMOCR polygon")
    source_points = tuple(
        (int(values[index]), int(values[index + 1]))
        for index in range(0, len(values), 2)
    )
    points = tuple(
        (min(max(x, 0), width - 1), min(max(y, 0), height - 1))
        for x, y in source_points
    )
    if len(set(points)) < 3:
        raise SceneTextPreparationError("CTW1500 polygon degenerates after border clipping")
    return SceneTextInstance(
        polygon_xy=points,
        transcription="<detection-only>",
        ignore=bool(raw.get("ignore", 0)),
        granularity="text_line",
    ), points != source_points


def _prepare_ctw1500(downloads: Path, root: Path) -> list[Path]:
    archive_path = downloads / "ctw1500_mmocr.zip"
    manifests: list[Path] = []
    with ZipFile(archive_path) as archive:
        for split in ("train", "test"):
            metadata_name = f"ctw1500/textdet_{split}.json"
            metadata = json.loads(archive.read(metadata_name))
            data_list = metadata.get("data_list")
            if not isinstance(data_list, list) or len(data_list) != EXPECTED_COUNTS["ctw1500"][split]:
                raise SceneTextPreparationError(f"CTW1500 {split} metadata count mismatch")
            records: list[dict[str, str]] = []
            clipped_instances = 0
            ordered = sorted(
                data_list,
                key=lambda item: _sort_id(_canonical_id(str(item.get("img_path")), "ctw1500")),
            )
            for item in ordered:
                relative_image = str(item.get("img_path"))
                sample_id = _canonical_id(relative_image, "ctw1500")
                width, height = int(item.get("width", 0)), int(item.get("height", 0))
                if width < 1 or height < 1:
                    raise SceneTextPreparationError("CTW1500 metadata has invalid geometry")
                member = f"ctw1500/{relative_image}"
                base = root / "datasets" / "ctw1500" / split
                image_payload = archive.read(member)
                with Image.open(BytesIO(image_payload)) as opened:
                    if opened.size == (width, height):
                        image_path = base / "images" / f"{sample_id}.jpg"
                        _write_bytes(image_path, image_payload)
                    else:
                        transposed = ImageOps.exif_transpose(opened).convert("RGB")
                        if transposed.size != (width, height):
                            raise SceneTextPreparationError(
                                "CTW1500 declared/image geometry mismatch after EXIF transpose"
                            )
                        image_path = base / "images" / f"{sample_id}.png"
                        _save_png(transposed, image_path)
                raw_instances = item.get("instances")
                if not isinstance(raw_instances, list) or not raw_instances:
                    raise SceneTextPreparationError("CTW1500 image has no instances")
                converted = tuple(
                    _ctw_instance(raw, width=width, height=height) for raw in raw_instances
                )
                instances = tuple(instance for instance, _ in converted)
                clipped_instances += sum(was_clipped for _, was_clipped in converted)
                target, validity = rasterize_instances((width, height), instances)
                mask_path = base / "masks" / f"{sample_id}.png"
                validity_path = base / "validity" / f"{sample_id}.png"
                _save_png(target, mask_path)
                _save_png(validity, validity_path)
                records.append(
                    _manifest_record(
                        root=root,
                        dataset="ctw1500",
                        sample_id=sample_id,
                        image=image_path,
                        mask=mask_path,
                        validity=validity_path,
                    )
                )
                if len(records) % 100 == 0 or len(records) == len(ordered):
                    print(
                        f"[prepare] ctw1500 {split}: {len(records)}/{len(ordered)}",
                        flush=True,
                    )
            manifests.append(
                _write_manifest(
                    root=root,
                    dataset="ctw1500",
                    split=split,
                    records=records,
                    acquisition={
                        "classification": "SECONDARY_MIRROR_CHECKSUM_FROZEN",
                        "archive_sha256": file_sha256(archive_path),
                        "format": "MMOCR",
                        "border_clipping_policy": "clip_polygon_points_to_pixel_extent",
                        "border_clipped_instance_count": clipped_instances,
                    },
                    target_source="filled_official_text_line_polygons_from_mmocr_mirror",
                    target_classification="ASSUMPTION",
                )
            )
    return manifests


def _directory_index(
    directory: Path, dataset: str, *, suffixes: tuple[str, ...]
) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.encode("utf-8")):
        if (
            not path.is_file()
            or path.name.startswith("._")
            or path.suffix.lower() not in suffixes
        ):
            continue
        key = _canonical_id(path.name, dataset)
        if key in output:
            raise SceneTextPreparationError(f"Duplicate {dataset} sample ID {key}")
        output[key] = path
    return output


def _prepare_totaltext(downloads: Path, root: Path) -> list[Path]:
    source_root = downloads / "Total-Text-Dataset"
    manifests: list[Path] = []
    tree_manifest = downloads / "SHA256SUMS.downloaded.txt"
    for split, source_name in (("train", "Train"), ("test", "Test")):
        images = _directory_index(
            source_root / "Images" / source_name,
            "total_text",
            suffixes=(".jpg", ".jpeg", ".png"),
        )
        annotations = _directory_index(
            source_root / "txt_format" / source_name,
            "total_text",
            suffixes=(".txt",),
        )
        pixel_masks = _directory_index(
            source_root / "groundtruth_pixel" / source_name,
            "total_text",
            suffixes=(".jpg", ".jpeg", ".png"),
        )
        if set(images) != set(annotations) or set(images) != set(pixel_masks):
            raise SceneTextPreparationError(f"Total-Text {split} image/annotation/mask IDs differ")
        if len(images) != EXPECTED_COUNTS["total_text"][split]:
            raise SceneTextPreparationError(f"Total-Text {split} count mismatch")
        records: list[dict[str, str]] = []
        clipped_instances = 0
        for sample_id in sorted(images, key=_sort_id):
            base = root / "datasets" / "total_text" / split
            image_path = base / "images" / f"{sample_id}{images[sample_id].suffix.lower()}"
            annotation_path = base / "annotations" / f"{sample_id}.txt"
            _link_or_copy(images[sample_id], image_path)
            _link_or_copy(annotations[sample_id], annotation_path)
            with Image.open(image_path) as opened:
                size = opened.size
                opened.verify()
            source_instances = parse_totaltext_annotation(annotation_path)
            instances, sample_clipped = clip_instances_to_pixel_extent(
                source_instances, size
            )
            clipped_instances += sample_clipped
            with Image.open(pixel_masks[sample_id]) as opened_mask:
                target, validity = normalize_totaltext_pixel_mask(
                    opened_mask,
                    expected_size_wh=size,
                    ignored_instances=instances,
                )
            mask_path = base / "masks" / f"{sample_id}.png"
            validity_path = base / "validity" / f"{sample_id}.png"
            _save_png(target, mask_path)
            _save_png(validity, validity_path)
            records.append(
                _manifest_record(
                    root=root,
                    dataset="total_text",
                    sample_id=sample_id,
                    image=image_path,
                    mask=mask_path,
                    validity=validity_path,
                )
            )
            if len(records) % 100 == 0 or len(records) == len(images):
                print(
                    f"[prepare] total_text {split}: {len(records)}/{len(images)}",
                    flush=True,
                )
        manifests.append(
            _write_manifest(
                root=root,
                dataset="total_text",
                split=split,
                records=records,
                acquisition={
                    "classification": "SECONDARY_MIRROR_TREE_CHECKSUM_FROZEN",
                    "tree_manifest_sha256": file_sha256(tree_manifest),
                    "border_clipping_policy": "clip_polygon_points_to_pixel_extent",
                    "border_clipped_instance_count": clipped_instances,
                },
                target_source="distributed_totaltext_pixel_ground_truth_threshold_ge_128",
                target_classification="REPOSITORY_SPECIFIED",
            )
        )
    return manifests


def _write_combined_manifest(root: Path) -> Path:
    manifest = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "dataset": "ctw1500+icdar2015+total_text",
        "split": "combined-official-training-populations-v1",
        "artifact_id": "ART-SCENETEXT3-PRACTICAL-COMBINED-TRAIN",
        "split_role": "train",
        "split_policy": {
            "algorithm_id": "complete-training-union-no-validation-v1",
            "classification": "ASSUMPTION",
            "validation_count": 0,
        },
        "acquisition": {
            "classification": "MIXED_PRIMARY_AND_CHECKSUM_FROZEN_MIRRORS",
            "source_manifests": [
                "ctw1500_train.yaml",
                "icdar2015_train.yaml",
                "total_text_train.yaml",
            ],
        },
        "annotation_policy": {
            "target_conversion": "binary-mask-with-validity-v1",
            "target_source": "heterogeneous_pixel_and_filled_polygon_targets",
            "target_classification": "ASSUMPTION",
            "masked_bce": True,
            "dont_care_training_behavior": "excluded_by_validity_mask",
            "mask_encoding": "lossless_png_values_0_or_255",
            "validity_encoding": "lossless_png_values_0_or_255",
            "training_ready": True,
        },
        "source_manifests": [
            "ctw1500_train.yaml",
            "icdar2015_train.yaml",
            "total_text_train.yaml",
        ],
        "expected_sample_count": 3255,
        "sample_order": "source_manifest_order_then_source_sample_order",
    }
    path = root / "manifests" / "scene_text3_combined_train.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def prepare_scene_text_segmentation(download_root: Path, output_root: Path) -> dict[str, Any]:
    """Prepare all splits atomically and return a qualification summary."""

    download_root = download_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        qualification = output_root / "qualification.json"
        if not qualification.is_file():
            raise SceneTextPreparationError(
                f"Output exists without qualification record: {output_root}"
            )
        return json.loads(qualification.read_text(encoding="utf-8"))
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = output_root.parent / f".{output_root.name}.staging-{os.getpid()}"
    if stage.exists():
        raise SceneTextPreparationError(f"Unexpected staging path already exists: {stage}")
    stage.mkdir()
    try:
        manifests = []
        manifests.extend(_prepare_ctw1500(download_root / "ctw1500" / "downloads", stage))
        manifests.extend(_prepare_icdar2015(download_root / "icdar2015" / "downloads", stage))
        manifests.extend(_prepare_totaltext(download_root / "total_text" / "downloads", stage))
        manifests.append(_write_combined_manifest(stage))
        report = {
            "schema_version": 1,
            "status": "ready_for_training_qualification",
            "classification": "PRACTICAL_APPROXIMATION",
            "output_root": str(output_root),
            "manifests": {
                path.name: file_sha256(path) for path in sorted(manifests, key=lambda item: item.name)
            },
            "training_populations": {
                "ctw1500": 1000,
                "icdar2015": 1000,
                "total_text": 1255,
                "combined_unique": 3255,
            },
            "scientific_warning": (
                "ICDAR2015 and CTW1500 targets are filled polygons; Total-Text uses "
                "distributed pixel masks. They are not homogeneous target semantics."
            ),
        }
        (stage / "qualification.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        stage.rename(output_root)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


__all__ = [
    "SceneTextPreparationError",
    "clip_instances_to_pixel_extent",
    "normalize_totaltext_pixel_mask",
    "prepare_scene_text_segmentation",
    "rasterize_instances",
]
