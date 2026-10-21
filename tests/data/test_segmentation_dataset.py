from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from segmentation.data import (
    SegmentationDataset,
    SegmentationManifestError,
    load_segmentation_manifest,
)


def _write_fixture(tmp_path: Path, *, bad_mask: bool = False) -> Path:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    for sample_id, value in (("b", 80), ("a", 160)):
        image = np.full((9, 13, 3), value, dtype=np.uint8)
        mask = np.zeros((9, 13), dtype=np.uint8)
        mask[2:6, 3:9] = 17 if bad_mask else 255
        Image.fromarray(image).save(image_dir / f"{sample_id}.png")
        Image.fromarray(mask).save(mask_dir / f"{sample_id}.png")
    samples = []
    for sample_id in ("b", "a"):
        image_bytes = (image_dir / f"{sample_id}.png").read_bytes()
        mask_bytes = (mask_dir / f"{sample_id}.png").read_bytes()
        samples.append(
            {
                "id": sample_id,
                "image": f"images/{sample_id}.png",
                "mask": f"masks/{sample_id}.png",
                "image_sha256": sha256(image_bytes).hexdigest(),
                "mask_sha256": sha256(mask_bytes).hexdigest(),
            }
        )
    manifest = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "dataset": "synthetic-smoke",
        "split": "train",
        "annotation_policy": {
            "classification": "ASSUMPTION",
            "foreground": "nonzero",
            "ignore": "unsupported-until-resolved",
        },
        "samples": samples,
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def test_manifest_preserves_order_and_loads_rgb_binary_pairs(tmp_path: Path) -> None:
    manifest = load_segmentation_manifest(_write_fixture(tmp_path))
    dataset = SegmentationDataset(manifest, tmp_path)
    assert [dataset[index]["sample_id"] for index in range(2)] == ["b", "a"]
    assert dataset[0]["image"].mode == "RGB"
    assert set(np.unique(dataset[0]["mask"])) == {0, 255}
    assert dataset[0]["claim_label"] == "approximate reproduction"


def test_checksum_mismatch_fails(tmp_path: Path) -> None:
    manifest = load_segmentation_manifest(_write_fixture(tmp_path))
    (tmp_path / "images" / "b.png").write_bytes(b"corrupt")
    with pytest.raises(SegmentationManifestError, match="checksum mismatch"):
        SegmentationDataset(manifest, tmp_path)


def test_nonbinary_mask_fails_at_sample_load(tmp_path: Path) -> None:
    manifest = load_segmentation_manifest(_write_fixture(tmp_path, bad_mask=True))
    dataset = SegmentationDataset(manifest, tmp_path)
    with pytest.raises(SegmentationManifestError, match="not binary"):
        dataset[0]


def test_manifest_rejects_traversal_duplicate_ids_and_missing_checksums(tmp_path: Path) -> None:
    path = _write_fixture(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["samples"][0]["image"] = "../escape.png"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SegmentationManifestError, match="within the supplied data root"):
        load_segmentation_manifest(path)

    path = _write_fixture(tmp_path / "second")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["samples"][1]["id"] = raw["samples"][0]["id"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SegmentationManifestError, match="Duplicate sample ID"):
        load_segmentation_manifest(path)

    path = _write_fixture(tmp_path / "third")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    del raw["samples"][0]["mask_sha256"]
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(SegmentationManifestError, match="mask_sha256 is required"):
        load_segmentation_manifest(path)
