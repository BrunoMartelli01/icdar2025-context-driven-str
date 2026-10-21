"""Generate deterministic M9 manifests for the adopted ICDAR13 split assumption."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from PIL import Image
import yaml

from segmentation.data import ICDAR13_TASK22_MASK_CONVERSION, file_sha256


SOURCE_COUNT = 229
TRAIN_COUNT = 206
VALIDATION_COUNT = 23
SPLIT_SEED = 202503
ALGORITHM_ID = "icdar13-task2.2-sha256-ranked-holdout-v1"

POLICY: dict[str, Any] = {
    "algorithm_id": ALGORITHM_ID,
    "algorithm_version": 1,
    "classification": "ASSUMPTION",
    "ambiguity_ids": ["T-001", "T-002", "DS-003"],
    "source_population": {
        "count": SOURCE_COUNT,
        "image_pattern": "raw/train_images/<numeric_id>.jpg",
        "mask_pattern": "raw/train_segmentation_gt/<numeric_id>_GT.bmp",
        "canonical_sample_id": "str(int(numeric filename stem))",
        "canonical_order": "numeric sample_id ascending",
    },
    "seed": SPLIT_SEED,
    "ranking_expression": 'SHA256(UTF8("202503:" + sample_id))',
    "ranking": [
        "raw SHA-256 digest bytes ascending",
        "numeric sample_id ascending as total-order fallback",
    ],
    "selection": {
        "validation": "first 23 ranked samples",
        "training": "remaining 206 ranked samples",
        "training_count": TRAIN_COUNT,
        "validation_count": VALIDATION_COUNT,
    },
    "manifest_serialization_order": "numeric sample_id ascending within each manifest",
    "validation_semantics": {
        "source": "held-out subset of the official 229-image training source",
        "metrics_only": True,
        "selects_best_checkpoint": False,
        "checkpoint_selection": "last",
        "test_data_used": False,
    },
}

MASK_POLICY: dict[str, Any] = {
    "algorithm_id": ICDAR13_TASK22_MASK_CONVERSION,
    "algorithm_version": 1,
    "classification": "ASSUMPTION",
    "ambiguity_ids": ["DS-003"],
    "input": "HxWx3 uint8 RGB ICDAR13 Task 2.2 BMP",
    "conversion": {
        "classification": "REPOSITORY_SPECIFIED",
        "evidence_basis": "UPSTREAM_PRIMARY",
        "background": "RGB exactly (255,255,255)",
        "background_value": 0,
        "foreground": "every other RGB value",
        "foreground_value": 1,
        "exact_equality": True,
        "tolerance": None,
        "grayscale_threshold": None,
        "morphology": None,
        "palette_assumption": None,
        "color_remapping": None,
    },
    "dont_care_training": {
        "classification": "ASSUMPTION",
        "behavior": "foreground",
        "special_cased": False,
        "strict_blocker": True,
    },
    "txt_annotations": {
        "role": "audit/provenance only",
        "alters_bce_target": False,
    },
    "loss_contract": {
        "loss": "BCEWithLogitsLoss",
        "ignore_mask": False,
        "masked_bce": False,
        "ignore_index": None,
    },
    "physical_audit": {
        "bmp_count": 229,
        "txt_count": 229,
        "bad_bmp_count": 0,
        "nonbinary_after_conversion_count": 0,
        "near_white_non_white_pixel_count": 205264,
        "txt_files_with_dont_care": 76,
        "dont_care_line_count": 608,
    },
    "scope": "M9 training targets only; defines no M10 evaluation semantics",
}


class ICDAR13ManifestError(ValueError):
    """Raised when physical source bytes violate the frozen manifest contract."""


@dataclass(frozen=True)
class SourceSample:
    sample_id: str
    numeric_id: int
    image_path: Path
    mask_path: Path
    image_relative: str
    mask_relative: str
    image_sha256: str
    mask_sha256: str


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def policy_fingerprint() -> str:
    return sha256(_canonical_json(POLICY)).hexdigest()


def mask_policy_fingerprint() -> str:
    return sha256(_canonical_json(MASK_POLICY)).hexdigest()


def _numeric_id(filename_stem: str, *, kind: str) -> tuple[str, int]:
    if not filename_stem or not filename_stem.isascii() or not filename_stem.isdecimal():
        raise ICDAR13ManifestError(f"{kind} ID must be decimal numeric: {filename_stem!r}")
    numeric = int(filename_stem)
    return str(numeric), numeric


def _index_images(directory: Path) -> dict[str, tuple[int, Path]]:
    indexed: dict[str, tuple[int, Path]] = {}
    for path in sorted(directory.glob("*.jpg")):
        sample_id, numeric = _numeric_id(path.stem, kind="image")
        if sample_id in indexed:
            raise ICDAR13ManifestError(f"Duplicate canonical image ID: {sample_id}")
        indexed[sample_id] = (numeric, path)
    return indexed


def _index_masks(directory: Path) -> dict[str, tuple[int, Path]]:
    indexed: dict[str, tuple[int, Path]] = {}
    for path in sorted(directory.glob("*.bmp")):
        if not path.stem.endswith("_GT"):
            raise ICDAR13ManifestError(f"Mask does not match <numeric_id>_GT.bmp: {path.name}")
        sample_id, numeric = _numeric_id(path.stem[:-3], kind="mask")
        if sample_id in indexed:
            raise ICDAR13ManifestError(f"Duplicate canonical mask ID: {sample_id}")
        indexed[sample_id] = (numeric, path)
    return indexed


def discover_source_samples(data_root: str | Path) -> tuple[SourceSample, ...]:
    root = Path(data_root).resolve()
    image_directory = root / "raw" / "train_images"
    mask_directory = root / "raw" / "train_segmentation_gt"
    if not image_directory.is_dir() or not mask_directory.is_dir():
        raise ICDAR13ManifestError("ICDAR13 raw image/mask directories are missing")
    images = _index_images(image_directory)
    masks = _index_masks(mask_directory)
    missing_masks = sorted(images.keys() - masks.keys(), key=int)
    extra_masks = sorted(masks.keys() - images.keys(), key=int)
    if missing_masks or extra_masks:
        raise ICDAR13ManifestError(
            f"Image/mask ID mismatch: missing_masks={missing_masks}, extra_masks={extra_masks}"
        )
    if len(images) != SOURCE_COUNT:
        raise ICDAR13ManifestError(
            f"Expected exactly {SOURCE_COUNT} paired source samples; found {len(images)}"
        )

    samples: list[SourceSample] = []
    for sample_id in sorted(images, key=int):
        numeric, image_path = images[sample_id]
        mask_numeric, mask_path = masks[sample_id]
        if numeric != mask_numeric:
            raise ICDAR13ManifestError(f"Numeric image/mask ID mismatch for {sample_id}")
        try:
            with Image.open(image_path) as image:
                image.load()
                image_size = image.size
            with Image.open(mask_path) as mask:
                mask.load()
                mask_size = mask.size
        except OSError as error:
            raise ICDAR13ManifestError(f"Unreadable image/mask for {sample_id}: {error}") from error
        if image_size != mask_size:
            raise ICDAR13ManifestError(
                f"Image/mask geometry mismatch for {sample_id}: {image_size} != {mask_size}"
            )
        samples.append(
            SourceSample(
                sample_id=sample_id,
                numeric_id=numeric,
                image_path=image_path,
                mask_path=mask_path,
                image_relative=image_path.relative_to(root).as_posix(),
                mask_relative=mask_path.relative_to(root).as_posix(),
                image_sha256=file_sha256(image_path),
                mask_sha256=file_sha256(mask_path),
            )
        )
    resolved_paths = [
        resolved
        for sample in samples
        for resolved in (sample.image_path.resolve(), sample.mask_path.resolve())
    ]
    if len(set(resolved_paths)) != SOURCE_COUNT * 2:
        raise ICDAR13ManifestError("Duplicate resolved image/mask paths are forbidden")
    return tuple(samples)


def split_sample_ids(samples: tuple[SourceSample, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(samples) != SOURCE_COUNT or len({sample.sample_id for sample in samples}) != SOURCE_COUNT:
        raise ICDAR13ManifestError("Split requires exactly 229 unique canonical source IDs")
    ranked = sorted(
        samples,
        key=lambda sample: (
            sha256(f"{SPLIT_SEED}:{sample.sample_id}".encode("utf-8")).digest(),
            sample.numeric_id,
        ),
    )
    validation = {sample.sample_id for sample in ranked[:VALIDATION_COUNT]}
    validation_ids = tuple(
        sample.sample_id for sample in sorted(samples, key=lambda sample: sample.numeric_id)
        if sample.sample_id in validation
    )
    train_ids = tuple(
        sample.sample_id for sample in sorted(samples, key=lambda sample: sample.numeric_id)
        if sample.sample_id not in validation
    )
    if len(train_ids) != TRAIN_COUNT or len(validation_ids) != VALIDATION_COUNT:
        raise ICDAR13ManifestError("Frozen split cardinalities were not produced")
    return train_ids, validation_ids


def _manifest(
    samples: tuple[SourceSample, ...], *, split_role: str, split_name: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "artifact_id": "ART-ICDAR13-FST-OFFICIAL-SPLIT",
        "dataset": "ICDAR13-FST-Task2.2",
        "split": split_name,
        "split_role": split_role,
        "split_policy": {
            "artifact_id": "ART-ICDAR13-M9-APPROX-SPLIT-POLICY",
            "algorithm_id": ALGORITHM_ID,
            "classification": "ASSUMPTION",
            "seed": SPLIT_SEED,
            "policy_fingerprint_sha256": policy_fingerprint(),
        },
        "acquisition": {
            "official_upstream_artifact_id": "ART-ICDAR13-FST-OFFICIAL-SPLIT",
            "mirror_archive_artifact_ids": [
                "ART-ICDAR13-MIRROR-TRAIN-IMAGES-V084",
                "ART-ICDAR13-MIRROR-TASK2-GT-V084",
            ],
        },
        "annotation_policy": {
            "artifact_id": "ART-ICDAR13-M9-APPROX-TRAINING-MASK-POLICY",
            "classification": "ASSUMPTION",
            "source_format": "ICDAR13 Task 2.2 color-coded per-character RGB BMP",
            "target_conversion": ICDAR13_TASK22_MASK_CONVERSION,
            "policy_fingerprint_sha256": mask_policy_fingerprint(),
            "foreground_conversion": "exact RGB white -> 0; every other RGB -> 1",
            "conversion_classification": "REPOSITORY_SPECIFIED",
            "dont_care_training_behavior": "foreground",
            "dont_care_classification": "ASSUMPTION",
            "txt_role": "audit/provenance only; does not alter BCE targets",
            "ignore_mask": False,
            "masked_bce": False,
            "ignore_index": None,
            "ambiguity_ids": ["DS-003"],
            "training_ready": True,
        },
        "samples": [
            {
                "id": sample.sample_id,
                "image": sample.image_relative,
                "mask": sample.mask_relative,
                "image_sha256": sample.image_sha256,
                "mask_sha256": sample.mask_sha256,
            }
            for sample in samples
        ],
    }


def _render_yaml(value: dict[str, Any]) -> bytes:
    return yaml.safe_dump(
        value, sort_keys=False, allow_unicode=True, default_flow_style=False
    ).encode("utf-8")


def generate_icdar13_manifests(
    data_root: str | Path, output_directory: str | Path
) -> dict[str, Any]:
    samples = discover_source_samples(data_root)
    train_ids, validation_ids = split_sample_ids(samples)
    by_id = {sample.sample_id: sample for sample in samples}
    train_samples = tuple(by_id[sample_id] for sample_id in train_ids)
    validation_samples = tuple(by_id[sample_id] for sample_id in validation_ids)
    if set(train_ids) & set(validation_ids) or set(train_ids) | set(validation_ids) != set(by_id):
        raise ICDAR13ManifestError("Frozen split is not a disjoint cover of the source IDs")

    source_records = [
        {
            "id": sample.sample_id,
            "image": sample.image_relative,
            "mask": sample.mask_relative,
            "image_sha256": sample.image_sha256,
            "mask_sha256": sample.mask_sha256,
        }
        for sample in samples
    ]
    policy_record = {
        "schema_version": 1,
        "policy": POLICY,
        "policy_fingerprint_sha256": policy_fingerprint(),
        "source_population_sha256": sha256(_canonical_json(source_records)).hexdigest(),
        "train_ids_sha256": sha256(_canonical_json(list(train_ids))).hexdigest(),
        "validation_ids_sha256": sha256(_canonical_json(list(validation_ids))).hexdigest(),
        "train_ids": list(train_ids),
        "validation_ids": list(validation_ids),
    }
    mask_policy_record = {
        "schema_version": 1,
        "policy": MASK_POLICY,
        "policy_fingerprint_sha256": mask_policy_fingerprint(),
    }
    rendered = {
        "icdar13_segmentation_train.yaml": _render_yaml(
            _manifest(
                train_samples,
                split_role="train",
                split_name="official-training-source-approximate-train",
            )
        ),
        "icdar13_segmentation_validation.yaml": _render_yaml(
            _manifest(
                validation_samples,
                split_role="validation",
                split_name="official-training-source-approximate-validation",
            )
        ),
        "icdar13_segmentation_split_policy.json": (
            json.dumps(policy_record, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "icdar13_training_mask_policy.json": (
            json.dumps(mask_policy_record, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8"),
    }
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    for filename, content in rendered.items():
        path = destination / filename
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
    return {
        "source_count": len(samples),
        "train_count": len(train_ids),
        "validation_count": len(validation_ids),
        "validation_ids": list(validation_ids),
        "policy_fingerprint_sha256": policy_fingerprint(),
        "mask_policy_fingerprint_sha256": mask_policy_fingerprint(),
        "files": {
            filename: sha256(content).hexdigest() for filename, content in rendered.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, default=Path("manifests"))
    args = parser.parse_args()
    result = generate_icdar13_manifests(args.data_root, args.output_directory)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ALGORITHM_ID",
    "ICDAR13ManifestError",
    "MASK_POLICY",
    "POLICY",
    "SOURCE_COUNT",
    "SPLIT_SEED",
    "TRAIN_COUNT",
    "VALIDATION_COUNT",
    "discover_source_samples",
    "generate_icdar13_manifests",
    "mask_policy_fingerprint",
    "policy_fingerprint",
    "split_sample_ids",
]
