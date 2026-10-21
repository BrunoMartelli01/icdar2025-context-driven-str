"""Deterministic manifests for ICDAR2015, Total-Text, and CTW1500.

The manifests freeze physical dataset identity but deliberately do not define
Table 5 matching, normalization, or accuracy semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .scene_text_datasets import (
    SceneTextAnnotationError,
    SceneTextInstance,
    parse_ctw1500_test_annotation,
    parse_ctw1500_train_xml,
    parse_icdar2015_annotation,
    parse_totaltext_annotation,
)


class SceneTextManifestError(ValueError):
    """Raised when a manifest or its physical dataset differs from its contract."""


@dataclass(frozen=True)
class SceneTextSample:
    sample_id: str
    image_path: Path
    annotation_path: Path
    image_size_wh: tuple[int, int]
    instances: tuple[SceneTextInstance, ...]


DATASET_CONTRACTS: dict[str, dict[str, Any]] = {
    "icdar2015": {
        "granularity": "word",
        "paper_role": "recognition_benchmark",
        "classification": "PAPER_SPECIFIED",
        "expected_counts": {"train": 1000, "test": 500},
        "parser": "icdar2015_quad_v1",
    },
    "total_text": {
        "granularity": "word",
        "paper_role": "segmentation_and_recognition_benchmark",
        "classification": "PAPER_SPECIFIED",
        "expected_counts": {"train": 1255, "test": 300},
        "parser": "total_text_polygon_txt_v1",
    },
    "ctw1500": {
        "granularity": "text_line",
        "paper_role": "out_of_paper_extension",
        "classification": "ASSUMPTION",
        "expected_counts": {"train": 1000, "test": 500},
        "parser": {"train": "ctw1500_xml_v1", "test": "ctw1500_relative14_txt_v1"},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_id(path: Path, dataset: str) -> str:
    stem = path.stem
    for prefix in ("gt_", "poly_gt_"):
        if stem.lower().startswith(prefix):
            stem = stem[len(prefix) :]
    if dataset == "icdar2015" and stem.lower().startswith("img_"):
        stem = stem[4:]
    if dataset == "total_text" and stem.lower().startswith("img"):
        stem = stem[3:]
    if not stem:
        raise SceneTextManifestError(f"Cannot derive sample ID from {path.name}")
    # CTW1500 test annotations commonly use 0001001.txt for image 1001.jpg.
    # Numeric canonicalization is therefore part of the pairing contract.
    return str(int(stem)) if stem.isascii() and stem.isdecimal() else stem


def _index(directory: Path, suffixes: tuple[str, ...], dataset: str) -> dict[str, Path]:
    if not directory.is_dir():
        raise SceneTextManifestError(f"Missing directory: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir(), key=lambda item: item.name.encode("utf-8")):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        sample_id = _canonical_id(path, dataset)
        if sample_id in result:
            raise SceneTextManifestError(
                f"Duplicate canonical ID {sample_id!r}: {result[sample_id]} and {path}"
            )
        result[sample_id] = path
    return result


def _sort_id(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isascii() and value.isdecimal() else (1, value)


def _parser(
    dataset: str, split: str, annotation_suffix: str | None = None
) -> Callable[..., tuple[SceneTextInstance, ...]]:
    if dataset == "icdar2015":
        return parse_icdar2015_annotation
    if dataset == "total_text":
        return parse_totaltext_annotation
    if dataset == "ctw1500":
        if split == "train" and annotation_suffix == ".xml":
            return parse_ctw1500_train_xml
        return parse_ctw1500_test_annotation
    raise SceneTextManifestError(f"Unsupported dataset: {dataset}")


def _ctw_txt_parser_id(path: Path) -> str:
    try:
        first = next(line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    except (OSError, UnicodeDecodeError, StopIteration) as error:
        raise SceneTextManifestError(f"Cannot identify CTW1500 annotation format: {path}") from error
    return (
        "ctw1500_e2e_absolute14_txt_v1"
        if first.isascii() and first.isdecimal()
        else "ctw1500_relative14_txt_v1"
    )


def build_scene_text_manifest(
    *,
    dataset: str,
    split: str,
    data_root: Path,
    image_directory: Path,
    annotation_directory: Path,
    output: Path,
    enforce_official_count: bool = True,
) -> dict[str, Any]:
    """Parse and checksum one explicit official split, then save JSON atomically."""

    if dataset not in DATASET_CONTRACTS or split not in {"train", "test"}:
        raise SceneTextManifestError("dataset/split must be a supported explicit value")
    root = data_root.resolve()
    image_directory = image_directory.resolve()
    annotation_directory = annotation_directory.resolve()
    for directory in (image_directory, annotation_directory):
        try:
            directory.relative_to(root)
        except ValueError as error:
            raise SceneTextManifestError(f"Dataset directory escapes data root: {directory}") from error

    images = _index(image_directory, (".jpg", ".jpeg", ".png"), dataset)
    annotation_suffixes = (".xml", ".txt") if dataset == "ctw1500" and split == "train" else (".txt",)
    annotations = _index(annotation_directory, annotation_suffixes, dataset)
    missing = sorted(images.keys() - annotations.keys(), key=_sort_id)
    extra = sorted(annotations.keys() - images.keys(), key=_sort_id)
    if missing or extra:
        raise SceneTextManifestError(
            f"Image/annotation mismatch: missing_annotations={missing[:20]}, extra_annotations={extra[:20]}"
        )
    expected = DATASET_CONTRACTS[dataset]["expected_counts"][split]
    if enforce_official_count and len(images) != expected:
        raise SceneTextManifestError(
            f"Expected official {dataset} {split} count {expected}; found {len(images)}"
        )

    annotation_formats = {path.suffix.lower() for path in annotations.values()}
    if len(annotation_formats) != 1:
        raise SceneTextManifestError(
            f"Annotation directory mixes formats: {sorted(annotation_formats)}"
        )
    annotation_suffix = next(iter(annotation_formats))
    parse = _parser(dataset, split, annotation_suffix)
    ctw_txt_parser_id = None
    if dataset == "ctw1500" and annotation_suffix == ".txt":
        parser_ids = {_ctw_txt_parser_id(path) for path in annotations.values()}
        if len(parser_ids) != 1:
            raise SceneTextManifestError(f"CTW1500 TXT annotations mix formats: {sorted(parser_ids)}")
        ctw_txt_parser_id = next(iter(parser_ids))
    samples: list[dict[str, Any]] = []
    total_instances = care_instances = 0
    for sample_id in sorted(images, key=_sort_id):
        image_path = images[sample_id]
        annotation_path = annotations[sample_id]
        try:
            with Image.open(image_path) as image:
                image_size = image.size
                image.verify()
            instances = parse(annotation_path, image_size_wh=image_size)
        except (OSError, SceneTextAnnotationError) as error:
            raise SceneTextManifestError(f"Invalid sample {sample_id}: {error}") from error
        total_instances += len(instances)
        care_instances += sum(not instance.ignore for instance in instances)
        samples.append(
            {
                "id": sample_id,
                "image": image_path.relative_to(root).as_posix(),
                "annotation": annotation_path.relative_to(root).as_posix(),
                "image_sha256": sha256_file(image_path),
                "annotation_sha256": sha256_file(annotation_path),
                "image_size_wh": list(image_size),
                "instance_count": len(instances),
                "care_instance_count": sum(not instance.ignore for instance in instances),
            }
        )

    contract = DATASET_CONTRACTS[dataset]
    report: dict[str, Any] = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction dataset manifest",
        "dataset": dataset,
        "split": split,
        "classification": contract["classification"],
        "paper_role": contract["paper_role"],
        "annotation_granularity": contract["granularity"],
        "parser": (
            ctw_txt_parser_id
            if dataset == "ctw1500" and annotation_suffix == ".txt"
            else contract["parser"][split]
            if isinstance(contract["parser"], dict)
            else contract["parser"]
        ),
        "sample_order": "numeric ID then UTF-8 lexical ID",
        "expected_official_sample_count": expected,
        "sample_count": len(samples),
        "instance_count": total_instances,
        "care_instance_count": care_instances,
        "table_5_claim": False,
        "evaluation_protocol": "UNRESOLVED",
        "blocking_ambiguity_ids": ["DS-003", "E-001"],
        "samples": samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def load_scene_text_manifest(
    manifest_path: Path, data_root: Path
) -> tuple[tuple[SceneTextSample, ...], dict[str, Any]]:
    """Verify every frozen byte and reparse each annotation."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SceneTextManifestError(f"Cannot read manifest {manifest_path}: {error}") from error
    dataset, split = manifest.get("dataset"), manifest.get("split")
    if manifest.get("schema_version") != 1 or dataset not in DATASET_CONTRACTS or split not in {"train", "test"}:
        raise SceneTextManifestError("Unsupported scene-text manifest contract")
    if manifest.get("table_5_claim") is not False or manifest.get("evaluation_protocol") != "UNRESOLVED":
        raise SceneTextManifestError("Manifest must not define or claim a Table 5 evaluation")
    root = data_root.resolve()
    output: list[SceneTextSample] = []
    for record in manifest.get("samples", []):
        paths: list[Path] = []
        for key in ("image", "annotation"):
            path = (root / str(record.get(key, ""))).resolve()
            try:
                path.relative_to(root)
            except ValueError as error:
                raise SceneTextManifestError(f"{key} path escapes data root") from error
            if path.is_symlink() or not path.is_file():
                raise SceneTextManifestError(f"Missing/non-regular {key}: {path}")
            paths.append(path)
        image_path, annotation_path = paths
        if dataset == "ctw1500" and annotation_path.suffix.lower() == ".txt":
            if manifest.get("parser") != _ctw_txt_parser_id(annotation_path):
                raise SceneTextManifestError(
                    f"CTW1500 parser/annotation format mismatch for {record.get('id')}"
                )
        parse = _parser(dataset, split, annotation_path.suffix.lower())
        if sha256_file(image_path) != record.get("image_sha256"):
            raise SceneTextManifestError(f"Image SHA256 mismatch for {record.get('id')}")
        if sha256_file(annotation_path) != record.get("annotation_sha256"):
            raise SceneTextManifestError(f"Annotation SHA256 mismatch for {record.get('id')}")
        with Image.open(image_path) as image:
            image_size = image.size
            image.verify()
        if list(image_size) != record.get("image_size_wh"):
            raise SceneTextManifestError(f"Image size mismatch for {record.get('id')}")
        instances = parse(annotation_path, image_size_wh=image_size)
        if len(instances) != record.get("instance_count"):
            raise SceneTextManifestError(f"Instance count mismatch for {record.get('id')}")
        output.append(
            SceneTextSample(
                sample_id=str(record["id"]),
                image_path=image_path,
                annotation_path=annotation_path,
                image_size_wh=image_size,
                instances=instances,
            )
        )
    if len(output) != manifest.get("sample_count"):
        raise SceneTextManifestError("Manifest sample count mismatch")
    return tuple(output), {
        "status": "qualified_dataset_identity",
        "manifest_sha256": sha256_file(manifest_path),
        "dataset": dataset,
        "split": split,
        "sample_count": len(output),
        "instance_count": sum(len(sample.instances) for sample in output),
        "table_5_claim": False,
    }


__all__ = [
    "DATASET_CONTRACTS",
    "SceneTextManifestError",
    "SceneTextSample",
    "build_scene_text_manifest",
    "load_scene_text_manifest",
    "sha256_file",
]
