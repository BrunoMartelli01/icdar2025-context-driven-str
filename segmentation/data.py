"""Checksum-aware segmentation manifests and paired image/mask loading."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from hashlib import sha256
from io import StringIO
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


class SegmentationManifestError(ValueError):
    """Raised when a manifest or a referenced sample is not reproducible."""


class ICDAR13MaskConversionError(ValueError):
    """Raised when a Task 2.2 mask cannot satisfy the frozen M9 RGB contract."""


ICDAR13_TASK22_MASK_CONVERSION = "icdar13-task2.2-exact-white-background-v1"
ICDAR13_TASK22_MASKED_CONVERSION = "icdar13-task2.2-author-validity-mask-v1"
ICDAR13_TASK22_TASK1_BOX_VALIDITY_CONVERSION = (
    "icdar13-task2.2-task1-box-zero-overlap-validity-v1"
)
BINARY_MASK_WITH_VALIDITY_CONVERSION = "binary-mask-with-validity-v1"


def icdar13_task22_rgb_to_binary_mask(mask: Image.Image) -> Image.Image:
    """Map exact RGB white to background, accepting only RGB or opaque RGBA."""

    if not isinstance(mask, Image.Image):
        raise ICDAR13MaskConversionError("ICDAR13 Task 2.2 mask must be a PIL image")
    if mask.mode not in ("RGB", "RGBA"):
        raise ICDAR13MaskConversionError(
            f"ICDAR13 Task 2.2 mask must have mode RGB or RGBA; got {mask.mode!r}"
        )
    mask_array = np.asarray(mask)
    if mask.mode == "RGBA":
        if (
            mask_array.dtype != np.uint8
            or mask_array.ndim != 3
            or mask_array.shape[2] != 4
        ):
            raise ICDAR13MaskConversionError(
                "ICDAR13 Task 2.2 RGBA mask must be an HxWx4 uint8 array"
            )
        if not np.all(mask_array[..., 3] == 255):
            raise ICDAR13MaskConversionError(
                "ICDAR13 Task 2.2 RGBA mask alpha must be fully opaque (255 everywhere)"
            )
        mask_array = mask_array[..., :3]
    if mask_array.dtype != np.uint8 or mask_array.ndim != 3 or mask_array.shape[2] != 3:
        raise ICDAR13MaskConversionError(
            "ICDAR13 Task 2.2 mask must be an HxWx3 uint8 RGB array"
        )
    if mask_array.shape[0] < 1 or mask_array.shape[1] < 1:
        raise ICDAR13MaskConversionError("ICDAR13 Task 2.2 mask geometry must be non-empty")
    foreground = ~np.all(mask_array == np.array((255, 255, 255), dtype=np.uint8), axis=2)
    return Image.fromarray(foreground.astype(np.uint8) * 255, mode="L")


def icdar13_task22_rgb_to_target_and_validity(
    mask: Image.Image, annotation_text: str
) -> Tuple[Image.Image, Image.Image]:
    """Build the binary target and zero-weight Don\'t Care validity mask.

    ICDAR13 Task 2.2 TXT records identify a BMP region by its first RGB
    triplet. Don\'t Care records are prefixed with ``#``. Pixels matching any
    such color are invalid for BCE; all other pixels are valid.
    """

    binary = icdar13_task22_rgb_to_binary_mask(mask)
    rgb = np.asarray(mask.convert("RGB"))
    ignored_colors = []
    for line_number, raw_line in enumerate(annotation_text.splitlines(), start=1):
        line = raw_line.lstrip("\ufeff ")
        if not line.startswith("#"):
            continue
        fields = line[1:].strip().split()
        if len(fields) < 3:
            raise ICDAR13MaskConversionError(
                "Malformed Don\'t Care record at line {}: missing RGB triplet".format(
                    line_number
                )
            )
        try:
            color = tuple(int(value) for value in fields[:3])
        except ValueError as error:
            raise ICDAR13MaskConversionError(
                "Malformed Don\'t Care RGB at line {}".format(line_number)
            ) from error
        if any(value < 0 or value > 255 for value in color):
            raise ICDAR13MaskConversionError(
                "Don\'t Care RGB values must be in [0,255] at line {}".format(line_number)
            )
        ignored_colors.append(color)

    valid = np.ones(rgb.shape[:2], dtype=np.bool_)
    for color in ignored_colors:
        matches = np.all(rgb == np.asarray(color, dtype=np.uint8), axis=2)
        if not np.any(matches):
            raise ICDAR13MaskConversionError(
                "Don\'t Care RGB {} is absent from the paired BMP".format(color)
            )
        valid &= ~matches
    validity = Image.fromarray(valid.astype(np.uint8) * 255, mode="L")
    return binary, validity


def icdar13_task22_task1_boxes_to_target_and_validity(
    mask: Image.Image, annotation_text: str
) -> Tuple[Image.Image, Image.Image]:
    """Conservatively infer ignored colors from official Task 1 care boxes.

    Task 1 TXT files contain care word boxes but no explicit Don't Care marker.
    A foreground color is ignored only when none of its pixels overlaps any
    Task 1 box. If even one pixel overlaps, the complete color remains valid.
    This is an APPROXIMATE diagnostic policy, not the author-confirmed Task 2
    ``#RGB`` mapping.
    """

    binary = icdar13_task22_rgb_to_binary_mask(mask)
    rgb = np.asarray(mask.convert("RGB"))
    height, width = rgb.shape[:2]
    care_boxes = np.zeros((height, width), dtype=np.bool_)
    record_count = 0
    for line_number, row in enumerate(csv.reader(StringIO(annotation_text)), start=1):
        if not row or all(not field.strip() for field in row):
            continue
        if len(row) < 5:
            raise ICDAR13MaskConversionError(
                "Malformed Task 1 record at line {}: expected box and transcription".format(
                    line_number
                )
            )
        try:
            left, top, right, bottom = (
                int(field.strip()) for field in row[:4]
            )
        except ValueError as error:
            raise ICDAR13MaskConversionError(
                "Malformed Task 1 coordinates at line {}".format(line_number)
            ) from error
        if (
            left < 0
            or top < 0
            or right < left
            or bottom < top
            or right >= width
            or bottom >= height
        ):
            raise ICDAR13MaskConversionError(
                "Task 1 box at line {} is outside mask geometry {}x{}".format(
                    line_number, width, height
                )
            )
        care_boxes[top : bottom + 1, left : right + 1] = True
        record_count += 1
    if record_count == 0:
        raise ICDAR13MaskConversionError("Task 1 annotation contains no care boxes")

    packed = (
        (rgb[..., 0].astype(np.uint32) << 16)
        | (rgb[..., 1].astype(np.uint32) << 8)
        | rgb[..., 2].astype(np.uint32)
    )
    foreground = np.asarray(binary, dtype=np.uint8) > 0
    care_colors = np.unique(packed[foreground & care_boxes])
    if care_colors.size == 0:
        raise ICDAR13MaskConversionError(
            "Task 1 care boxes do not overlap any foreground mask color"
        )
    ignored = foreground & ~np.isin(packed, care_colors)
    validity = Image.fromarray((~ignored).astype(np.uint8) * 255, mode="L")
    return binary, validity


@dataclass(frozen=True)
class SegmentationSample:
    sample_id: str
    image_path: PurePosixPath
    mask_path: PurePosixPath
    image_sha256: str | None
    mask_sha256: str | None
    validity_annotation_path: Optional[PurePosixPath] = None
    validity_annotation_sha256: Optional[str] = None
    separation_annotation_path: Optional[PurePosixPath] = None
    separation_annotation_sha256: Optional[str] = None
    word_owner_annotation_path: Optional[PurePosixPath] = None
    word_owner_annotation_sha256: Optional[str] = None


@dataclass(frozen=True)
class SegmentationManifest:
    schema_version: int
    reproduction_mode: str
    claim_label: str
    dataset: str
    split: str
    artifact_id: str | None
    split_role: str | None
    split_policy: Mapping[str, Any] | None
    acquisition: Mapping[str, Any] | None
    annotation_policy: Mapping[str, Any]
    samples: tuple[SegmentationSample, ...]
    source_manifests: Tuple[str, ...] = ()


def _relative_path(value: object, field: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise SegmentationManifestError(f"{field} must be a non-empty POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise SegmentationManifestError(f"{field} must stay within the supplied data root")
    return path


def _checksum(value: object, field: str, required: bool) -> str | None:
    if value in (None, ""):
        if required:
            raise SegmentationManifestError(f"{field} is required")
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise SegmentationManifestError(f"{field} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SegmentationManifestError(f"{field} is not hexadecimal") from error
    return value.lower()


def load_segmentation_manifest(
    path: str | Path, *, require_checksums: bool = True
) -> SegmentationManifest:
    """Load one explicit, ordered manifest without inferring files or splits."""

    manifest_path = Path(path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SegmentationManifestError(f"Cannot read manifest {manifest_path}: {error}") from error
    except yaml.YAMLError as error:
        raise SegmentationManifestError(f"Invalid YAML in {manifest_path}: {error}") from error
    if not isinstance(raw, dict):
        raise SegmentationManifestError("Manifest root must be a mapping")
    if raw.get("schema_version") != 1:
        raise SegmentationManifestError("Unsupported segmentation manifest schema_version")
    mode = raw.get("reproduction_mode")
    label = raw.get("claim_label")
    expected_labels = {"strict": "strict reproduction", "approximate": "approximate reproduction"}
    if mode not in expected_labels or label != expected_labels[mode]:
        raise SegmentationManifestError("Manifest reproduction_mode and claim_label are inconsistent")
    for field in ("dataset", "split"):
        if not isinstance(raw.get(field), str) or not raw[field]:
            raise SegmentationManifestError(f"Manifest field '{field}' is required")
    artifact_id = raw.get("artifact_id")
    if artifact_id is not None and (not isinstance(artifact_id, str) or not artifact_id):
        raise SegmentationManifestError("artifact_id must be a non-empty string when provided")
    split_role = raw.get("split_role")
    if split_role is not None and split_role not in ("train", "validation", "test"):
        raise SegmentationManifestError("split_role must be train, validation, or test")
    split_policy = raw.get("split_policy")
    if split_policy is not None and not isinstance(split_policy, dict):
        raise SegmentationManifestError("split_policy must be a mapping when provided")
    acquisition = raw.get("acquisition")
    if acquisition is not None and not isinstance(acquisition, dict):
        raise SegmentationManifestError("acquisition must be a mapping when provided")
    annotation_policy = raw.get("annotation_policy")
    if not isinstance(annotation_policy, dict):
        raise SegmentationManifestError("annotation_policy must explicitly describe mask semantics")
    records = raw.get("samples")
    source_manifest_values = raw.get("source_manifests")
    if source_manifest_values is not None:
        if records is not None:
            raise SegmentationManifestError("A composed manifest cannot also define samples")
        if not isinstance(source_manifest_values, list) or not source_manifest_values:
            raise SegmentationManifestError("source_manifests must be a non-empty list")
        source_manifests = []
        combined_samples = []
        for source_index, source_value in enumerate(source_manifest_values):
            source_relative = _relative_path(
                source_value, "source_manifests[{}]".format(source_index)
            )
            source_path = (manifest_path.parent / Path(source_relative)).resolve()
            if manifest_path.parent.resolve() not in source_path.parents:
                raise SegmentationManifestError("Composed source manifest escapes its directory")
            source = load_segmentation_manifest(
                source_path, require_checksums=require_checksums
            )
            source_manifests.append(source_relative.as_posix())
            combined_samples.extend(source.samples)
        if len({sample.sample_id for sample in combined_samples}) != len(combined_samples):
            raise SegmentationManifestError("Composed manifests contain duplicate sample IDs")
        if raw.get("sample_order") == "numeric_id_ascending":
            try:
                combined_samples.sort(key=lambda sample: int(sample.sample_id))
            except ValueError as error:
                raise SegmentationManifestError(
                    "numeric_id_ascending requires decimal sample IDs"
                ) from error
        expected_count = raw.get("expected_sample_count")
        if expected_count != len(combined_samples):
            raise SegmentationManifestError(
                "Composed manifest expected {} samples but resolved {}".format(
                    expected_count, len(combined_samples)
                )
            )
        annotation_template = raw.get("validity_annotation_path_template")
        if annotation_template is None:
            samples = tuple(combined_samples)
        else:
            if not isinstance(annotation_template, str) or "{id}" not in annotation_template:
                raise SegmentationManifestError(
                    "validity_annotation_path_template must contain {id}"
                )
            samples = tuple(
                replace(
                    sample,
                    validity_annotation_path=_relative_path(
                        annotation_template.format(id=sample.sample_id),
                        "validity_annotation_path_template",
                    ),
                )
                for sample in combined_samples
            )
        return SegmentationManifest(
            schema_version=1,
            reproduction_mode=mode,
            claim_label=label,
            dataset=raw["dataset"],
            split=raw["split"],
            artifact_id=artifact_id,
            split_role=split_role,
            split_policy=split_policy,
            acquisition=acquisition,
            annotation_policy=annotation_policy,
            samples=samples,
            source_manifests=tuple(source_manifests),
        )
    if not isinstance(records, list) or not records:
        raise SegmentationManifestError("samples must be a non-empty ordered list")

    samples: list[SegmentationSample] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise SegmentationManifestError(f"samples[{index}] must be a mapping")
        sample_id = record.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise SegmentationManifestError(f"samples[{index}].id is required")
        if sample_id in seen_ids:
            raise SegmentationManifestError(f"Duplicate sample ID '{sample_id}'")
        seen_ids.add(sample_id)
        samples.append(
            SegmentationSample(
                sample_id=sample_id,
                image_path=_relative_path(record.get("image"), f"samples[{index}].image"),
                mask_path=_relative_path(record.get("mask"), f"samples[{index}].mask"),
                image_sha256=_checksum(
                    record.get("image_sha256"), f"samples[{index}].image_sha256", require_checksums
                ),
                mask_sha256=_checksum(
                    record.get("mask_sha256"), f"samples[{index}].mask_sha256", require_checksums
                ),
                validity_annotation_path=(
                    _relative_path(
                        record.get("validity_annotation"),
                        f"samples[{index}].validity_annotation",
                    )
                    if record.get("validity_annotation") is not None
                    else None
                ),
                validity_annotation_sha256=_checksum(
                    record.get("validity_annotation_sha256"),
                    f"samples[{index}].validity_annotation_sha256",
                    False,
                ),
                separation_annotation_path=(
                    _relative_path(record["separation_annotation"], f"samples[{index}].separation_annotation")
                    if record.get("separation_annotation") is not None else None
                ),
                separation_annotation_sha256=_checksum(
                    record.get("separation_annotation_sha256"),
                    f"samples[{index}].separation_annotation_sha256",
                    require_checksums and record.get("separation_annotation") is not None,
                ),
                word_owner_annotation_path=(
                    _relative_path(record["word_owner_annotation"], f"samples[{index}].word_owner_annotation")
                    if record.get("word_owner_annotation") is not None else None
                ),
                word_owner_annotation_sha256=_checksum(
                    record.get("word_owner_annotation_sha256"),
                    f"samples[{index}].word_owner_annotation_sha256",
                    require_checksums and record.get("word_owner_annotation") is not None,
                ),
            )
        )
    return SegmentationManifest(
        schema_version=1,
        reproduction_mode=mode,
        claim_label=label,
        dataset=raw["dataset"],
        split=raw["split"],
        artifact_id=artifact_id,
        split_role=split_role,
        split_policy=split_policy,
        acquisition=acquisition,
        annotation_policy=annotation_policy,
        samples=tuple(samples),
        source_manifests=(),
    )


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


PairTransform = Callable[[Image.Image, Image.Image, int], Tuple[torch.Tensor, torch.Tensor]]


class SegmentationDataset(Dataset[Dict[str, Any]]):
    """Load exactly the pairs listed by a segmentation manifest."""

    def __init__(
        self,
        manifest: SegmentationManifest,
        data_root: str | Path,
        *,
        transform: PairTransform | None = None,
        verify_checksums: bool = True,
    ) -> None:
        self.manifest = manifest
        self.data_root = Path(data_root).resolve()
        self.transform = transform
        self.paths: list[Tuple[Path, ...]] = []
        for sample in manifest.samples:
            image_path = (self.data_root / Path(sample.image_path)).resolve()
            mask_path = (self.data_root / Path(sample.mask_path)).resolve()
            validity_path = (
                (self.data_root / Path(sample.validity_annotation_path)).resolve()
                if sample.validity_annotation_path is not None
                else None
            )
            separation_path = (
                (self.data_root / Path(sample.separation_annotation_path)).resolve()
                if sample.separation_annotation_path is not None else None
            )
            owner_path = (
                (self.data_root / Path(sample.word_owner_annotation_path)).resolve()
                if sample.word_owner_annotation_path is not None else None
            )
            if (
                self.data_root not in image_path.parents
                or self.data_root not in mask_path.parents
                or (validity_path is not None and self.data_root not in validity_path.parents)
                or (separation_path is not None and self.data_root not in separation_path.parents)
                or (owner_path is not None and self.data_root not in owner_path.parents)
            ):
                raise SegmentationManifestError(f"Sample '{sample.sample_id}' escapes data root")
            candidates = [
                ("image", image_path, sample.image_sha256),
                ("mask", mask_path, sample.mask_sha256),
            ]
            if validity_path is not None:
                candidates.append(
                    ("validity annotation", validity_path, sample.validity_annotation_sha256)
                )
            if separation_path is not None:
                candidates.append(("separation annotation", separation_path, sample.separation_annotation_sha256))
            if owner_path is not None:
                candidates.append(("word owner annotation", owner_path, sample.word_owner_annotation_sha256))
            for kind, candidate, expected in candidates:
                if not candidate.is_file():
                    raise SegmentationManifestError(
                        f"Sample '{sample.sample_id}' {kind} is missing: {candidate}"
                    )
                if verify_checksums and expected is not None and file_sha256(candidate) != expected:
                    raise SegmentationManifestError(
                        f"Sample '{sample.sample_id}' {kind} checksum mismatch"
                    )
            self.paths.append(
                (image_path, mask_path, validity_path, separation_path, owner_path)
            )

    def __len__(self) -> int:
        return len(self.manifest.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.manifest.samples[index]
        image_path, mask_path = self.paths[index][:2]
        validity_path = self.paths[index][2]
        separation_path = self.paths[index][3]
        owner_path = self.paths[index][4]
        with Image.open(image_path) as opened_image:
            image = opened_image.convert("RGB")
        with Image.open(mask_path) as opened_mask:
            conversion = self.manifest.annotation_policy.get("target_conversion")
            if conversion in (
                ICDAR13_TASK22_MASKED_CONVERSION,
                ICDAR13_TASK22_TASK1_BOX_VALIDITY_CONVERSION,
            ):
                if validity_path is None:
                    raise SegmentationManifestError(
                        "Author-protocol sample is missing its validity annotation"
                    )
                try:
                    annotation_text = validity_path.read_text(encoding="utf-8-sig")
                except (OSError, UnicodeError) as error:
                    raise SegmentationManifestError(
                        "Cannot read validity annotation for sample '{}'".format(
                            sample.sample_id
                        )
                    ) from error
                if conversion == ICDAR13_TASK22_MASKED_CONVERSION:
                    mask, validity = icdar13_task22_rgb_to_target_and_validity(
                        opened_mask, annotation_text
                    )
                else:
                    mask, validity = (
                        icdar13_task22_task1_boxes_to_target_and_validity(
                            opened_mask, annotation_text
                        )
                    )
            elif conversion == BINARY_MASK_WITH_VALIDITY_CONVERSION:
                if validity_path is None:
                    raise SegmentationManifestError(
                        "Binary-mask sample is missing its validity mask"
                    )
                mask_array = np.asarray(opened_mask.convert("L"))
                mask_values = set(np.unique(mask_array).tolist())
                if not mask_values.issubset({0, 255}):
                    raise SegmentationManifestError(
                        f"Sample '{sample.sample_id}' target mask is not canonical binary; "
                        f"values={sorted(mask_values)}"
                    )
                with Image.open(validity_path) as opened_validity:
                    validity_array = np.asarray(opened_validity.convert("L"))
                validity_values = set(np.unique(validity_array).tolist())
                if not validity_values.issubset({0, 255}):
                    raise SegmentationManifestError(
                        f"Sample '{sample.sample_id}' validity mask is not canonical binary; "
                        f"values={sorted(validity_values)}"
                    )
                if mask_array.shape != validity_array.shape:
                    raise SegmentationManifestError(
                        f"Sample '{sample.sample_id}' target/validity geometry mismatch"
                    )
                mask = Image.fromarray(mask_array, mode="L")
                validity = Image.fromarray(validity_array, mode="L")
            elif conversion == ICDAR13_TASK22_MASK_CONVERSION:
                mask = icdar13_task22_rgb_to_binary_mask(opened_mask)
            else:
                mask_array = np.asarray(opened_mask.convert("L"))
                values = set(np.unique(mask_array).tolist())
                if not values.issubset({0, 1, 255}):
                    raise SegmentationManifestError(
                        f"Sample '{sample.sample_id}' mask is not binary; "
                        f"values={sorted(values)}"
                    )
                mask = Image.fromarray((mask_array > 0).astype(np.uint8) * 255, mode="L")
        if conversion not in (
            ICDAR13_TASK22_MASKED_CONVERSION,
            ICDAR13_TASK22_TASK1_BOX_VALIDITY_CONVERSION,
            BINARY_MASK_WITH_VALIDITY_CONVERSION,
        ):
            validity = None
        separation = None
        owners = None
        if separation_path is not None:
            if validity is None:
                raise SegmentationManifestError("Separation annotation requires validity")
            with Image.open(separation_path) as opened_separation:
                separation_array = np.asarray(opened_separation.convert("L"))
            if not set(np.unique(separation_array).tolist()).issubset({0, 255}):
                raise SegmentationManifestError("Separation annotation must be binary 0/255")
            if separation_array.shape != np.asarray(mask).shape:
                raise SegmentationManifestError("Separation/target geometry mismatch")
            if np.any((separation_array > 0) & (np.asarray(mask) > 0)):
                raise SegmentationManifestError("Separation pixels must be zero in target")
            separation = Image.fromarray(separation_array, mode="L")
        if owner_path is not None:
            if separation is None:
                raise SegmentationManifestError("Word owner annotation requires separation mask")
            with Image.open(owner_path) as opened_owners:
                owner_array = np.asarray(opened_owners).astype(np.int32)
            if owner_array.shape != np.asarray(mask).shape or np.any(owner_array < 0):
                raise SegmentationManifestError("Word owner annotation geometry/values invalid")
            if not np.array_equal(owner_array > 0, np.asarray(mask) > 0):
                raise SegmentationManifestError("Word owner labels disagree with binary target")
            owners = Image.fromarray(owner_array, mode="I")
        if self.transform is None:
            image_out: Any = image
            mask_out: Any = mask
            validity_out: Any = validity
        elif validity is not None and separation is not None and owners is not None:
            transform_with_separation = getattr(self.transform, "transform_with_validity_and_separation", None)
            if transform_with_separation is None:
                raise SegmentationManifestError("Configured transform does not support separation masks")
            image_out, mask_out, validity_out, separation_out = transform_with_separation(
                image, mask, validity, separation, owners, index
            )
        elif separation is not None:
            raise SegmentationManifestError("Separation transform requires word owner labels")
        elif validity is not None:
            transform_with_validity = getattr(self.transform, "transform_with_validity", None)
            if transform_with_validity is None:
                raise SegmentationManifestError(
                    "Configured transform does not support per-pixel validity masks"
                )
            image_out, mask_out, validity_out = transform_with_validity(
                image, mask, validity, index
            )
        else:
            image_out, mask_out = self.transform(image, mask, index)
            validity_out = None
        result = {
            "sample_id": sample.sample_id,
            "image": image_out,
            "mask": mask_out,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "reproduction_mode": self.manifest.reproduction_mode,
            "claim_label": self.manifest.claim_label,
        }
        if validity_out is not None:
            result["validity"] = validity_out
            result["validity_annotation_path"] = str(validity_path)
        if separation is not None:
            if self.transform is None:
                separation_out = separation
            result["separation"] = separation_out
            result["separation_annotation_path"] = str(separation_path)
        return result


__all__: Sequence[str] = (
    "ICDAR13MaskConversionError",
    "ICDAR13_TASK22_MASK_CONVERSION",
    "ICDAR13_TASK22_MASKED_CONVERSION",
    "ICDAR13_TASK22_TASK1_BOX_VALIDITY_CONVERSION",
    "BINARY_MASK_WITH_VALIDITY_CONVERSION",
    "SegmentationDataset",
    "SegmentationManifest",
    "SegmentationManifestError",
    "SegmentationSample",
    "file_sha256",
    "icdar13_task22_rgb_to_binary_mask",
    "icdar13_task22_rgb_to_target_and_validity",
    "icdar13_task22_task1_boxes_to_target_and_validity",
    "load_segmentation_manifest",
)
