from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil

from PIL import Image
import pytest
import yaml

from segmentation.icdar13_manifests import (
    ICDAR13ManifestError,
    MASK_POLICY,
    POLICY,
    discover_source_samples,
    generate_icdar13_manifests,
    mask_policy_fingerprint,
    policy_fingerprint,
    split_sample_ids,
)


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_VALIDATION_IDS = [
    "113",
    "114",
    "115",
    "123",
    "124",
    "142",
    "150",
    "166",
    "168",
    "179",
    "218",
    "223",
    "245",
    "263",
    "280",
    "285",
    "288",
    "292",
    "299",
    "305",
    "310",
    "315",
    "328",
]


@pytest.fixture(scope="module")
def base_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("icdar13-source")
    images = root / "raw" / "train_images"
    masks = root / "raw" / "train_segmentation_gt"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for sample_id in range(100, 329):
        Image.new("RGB", (3, 2), (sample_id % 255, 20, 30)).save(
            images / f"{sample_id}.jpg"
        )
        Image.new("RGB", (3, 2), (255, sample_id % 255, 0)).save(
            masks / f"{sample_id}_GT.bmp"
        )
    return root


@pytest.fixture
def source_root(tmp_path: Path, base_source: Path) -> Path:
    destination = tmp_path / "data"
    shutil.copytree(base_source, destination)
    return destination


def test_frozen_counts_validation_ids_and_disjoint_cover(source_root: Path) -> None:
    samples = discover_source_samples(source_root)
    train_ids, validation_ids = split_sample_ids(samples)
    assert len(samples) == 229
    assert len(train_ids) == 206
    assert list(validation_ids) == EXPECTED_VALIDATION_IDS
    assert not set(train_ids) & set(validation_ids)
    assert set(train_ids) | set(validation_ids) == {str(value) for value in range(100, 329)}
    assert list(train_ids) == sorted(train_ids, key=int)
    assert list(validation_ids) == sorted(validation_ids, key=int)


def test_tracked_policy_and_manifests_match_frozen_contract() -> None:
    policy_path = ROOT / "manifests" / "icdar13_segmentation_split_policy.json"
    train_path = ROOT / "manifests" / "icdar13_segmentation_train.yaml"
    validation_path = ROOT / "manifests" / "icdar13_segmentation_validation.yaml"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    mask_policy = json.loads(
        (ROOT / "manifests" / "icdar13_training_mask_policy.json").read_text(
            encoding="utf-8"
        )
    )
    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    validation = yaml.safe_load(validation_path.read_text(encoding="utf-8"))
    assert policy["policy"] == POLICY
    assert policy["policy_fingerprint_sha256"] == policy_fingerprint()
    assert policy["validation_ids"] == EXPECTED_VALIDATION_IDS
    assert mask_policy["policy"] == MASK_POLICY
    assert mask_policy["policy_fingerprint_sha256"] == mask_policy_fingerprint()
    assert len(train["samples"]) == 206
    assert len(validation["samples"]) == 23
    train_ids = [sample["id"] for sample in train["samples"]]
    validation_ids = [sample["id"] for sample in validation["samples"]]
    assert not set(train_ids) & set(validation_ids)
    assert set(train_ids) | set(validation_ids) == set(policy["train_ids"]) | set(
        policy["validation_ids"]
    )
    for manifest in (train, validation):
        assert manifest["split_policy"]["classification"] == "ASSUMPTION"
        assert manifest["annotation_policy"]["training_ready"] is True
        assert manifest["annotation_policy"]["target_conversion"] == (
            "icdar13-task2.2-exact-white-background-v1"
        )
        assert manifest["annotation_policy"]["policy_fingerprint_sha256"] == (
            mask_policy_fingerprint()
        )
        assert manifest["annotation_policy"]["dont_care_training_behavior"] == "foreground"
        assert manifest["annotation_policy"]["ignore_mask"] is False
        for sample in manifest["samples"]:
            assert not Path(sample["image"]).is_absolute()
            assert not Path(sample["mask"]).is_absolute()
            assert "/home/" not in sample["image"] + sample["mask"]
            assert len(sample["image_sha256"]) == len(sample["mask_sha256"]) == 64


def test_regeneration_is_byte_for_byte_deterministic(source_root: Path, tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_report = generate_icdar13_manifests(source_root, first)
    second_report = generate_icdar13_manifests(source_root, second)
    assert first_report == second_report
    for filename in first_report["files"]:
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_non_numeric_id_is_rejected(source_root: Path) -> None:
    source = source_root / "raw" / "train_images" / "100.jpg"
    source.rename(source.with_name("not-numeric.jpg"))
    with pytest.raises(ICDAR13ManifestError, match="decimal numeric"):
        discover_source_samples(source_root)


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "extra", "geometry", "resolved_path_duplicate"]
)
def test_missing_duplicate_mismatched_pairs_are_rejected(
    source_root: Path, fault: str
) -> None:
    images = source_root / "raw" / "train_images"
    masks = source_root / "raw" / "train_segmentation_gt"
    if fault == "missing":
        (masks / "100_GT.bmp").unlink()
    elif fault == "duplicate":
        shutil.copyfile(images / "100.jpg", images / "0100.jpg")
    elif fault == "extra":
        shutil.copyfile(masks / "100_GT.bmp", masks / "999_GT.bmp")
    elif fault == "geometry":
        Image.new("RGB", (4, 2), (255, 0, 0)).save(masks / "100_GT.bmp")
    else:
        duplicate = images / "101.jpg"
        duplicate.unlink()
        duplicate.symlink_to("100.jpg")
    with pytest.raises(ICDAR13ManifestError):
        discover_source_samples(source_root)


def test_mirror_provenance_checksum_and_evidence_contract() -> None:
    sources = yaml.safe_load((ROOT / "provenance" / "SOURCES.yaml").read_text())["sources"]
    artifacts = yaml.safe_load((ROOT / "provenance" / "ARTIFACTS.yaml").read_text())[
        "artifacts"
    ]
    source_index = {item["id"]: item for item in sources}
    artifact_index = {item["id"]: item for item in artifacts}
    expected = {
        "ART-ICDAR13-MIRROR-TRAIN-IMAGES-V084": (
            "SRC-ICDAR13-KERASOCR-MIRROR-TRAIN-IMAGES-V084",
            "7a57d1699fbb92db3ad82c930202938562edaf72e1c422ddd923860d8ace8ded",
        ),
        "ART-ICDAR13-MIRROR-TASK2-GT-V084": (
            "SRC-ICDAR13-KERASOCR-MIRROR-TASK2-GT-V084",
            "4cedd5b1e33dc4354058f5967221ac85dbdf91a99b30f3ab1ecdf42786a9d027",
        ),
    }
    for artifact_id, (source_id, digest) in expected.items():
        source = source_index[source_id]
        artifact = artifact_index[artifact_id]
        assert source["evidence_classification"] == "SECONDARY"
        assert artifact["evidence_classification"] == "SECONDARY"
        assert digest in source["checkpoint_checksum"]
        assert digest in artifact["checkpoint_checksum"]
        assert "github.com/faustomorales/keras-ocr" in source[
            "source_repository_model_source"
        ]
    official = artifact_index["ART-ICDAR13-FST-OFFICIAL-SPLIT"]
    assert official["source_id"] == "SRC-ICDAR13-RRC"
    assert official["evidence_classification"] == "UPSTREAM_PRIMARY"


def test_policy_fingerprint_is_canonical_sha256() -> None:
    canonical = json.dumps(POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert policy_fingerprint() == sha256(canonical).hexdigest()
    mask_canonical = json.dumps(MASK_POLICY, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    assert mask_policy_fingerprint() == sha256(mask_canonical).hexdigest()


def test_training_mask_policy_provenance_contract() -> None:
    artifacts = yaml.safe_load((ROOT / "provenance" / "ARTIFACTS.yaml").read_text())[
        "artifacts"
    ]
    policy = next(
        item
        for item in artifacts
        if item["id"] == "ART-ICDAR13-M9-APPROX-TRAINING-MASK-POLICY"
    )
    assert policy["artifact_type"] == "dataset_annotation_policy"
    assert policy["decision_classification"] == "ASSUMPTION"
    assert policy["blocks_strict_reproduction"] is True
    assert mask_policy_fingerprint() in policy["checkpoint_checksum"]
