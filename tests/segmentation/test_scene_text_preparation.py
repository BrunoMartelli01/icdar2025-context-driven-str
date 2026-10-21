from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from recognition.scene_text_datasets import SceneTextInstance
from segmentation.config import load_segmentation_config
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.scene_text_preparation import (
    _ctw_instance,
    clip_instances_to_pixel_extent,
    normalize_totaltext_pixel_mask,
    rasterize_instances,
)
from segmentation.train import DomainBalancedEpochSampler


ROOT = Path(__file__).resolve().parents[2]


def test_rasterize_instances_separates_target_and_ignore_validity() -> None:
    care = SceneTextInstance(((1, 1), (8, 1), (8, 5), (1, 5)), "word", False, "word")
    ignore = SceneTextInstance(((6, 3), (9, 3), (9, 7), (6, 7)), "###", True, "word")
    target, validity = rasterize_instances((10, 8), (care, ignore))
    target_array = np.asarray(target)
    validity_array = np.asarray(validity)
    assert target_array[2, 2] == 255
    assert validity_array[2, 2] == 255
    assert validity_array[4, 7] == 0
    assert set(np.unique(target_array)) == {0, 255}
    assert set(np.unique(validity_array)) == {0, 255}


def test_totaltext_jpeg_values_become_canonical_binary() -> None:
    source = Image.fromarray(np.array([[0, 1, 127, 128, 254, 255]], dtype=np.uint8), mode="L")
    target, validity = normalize_totaltext_pixel_mask(
        source, expected_size_wh=(6, 1), ignored_instances=()
    )
    assert np.asarray(target).tolist() == [[0, 0, 0, 255, 255, 255]]
    assert np.asarray(validity).tolist() == [[255, 255, 255, 255, 255, 255]]


def test_ctw_border_coordinates_are_clipped_explicitly() -> None:
    instance, clipped = _ctw_instance(
        {"polygon": [8, 2, 11, 2, 11, 6, 8, 6], "ignore": 0},
        width=10,
        height=8,
    )
    assert clipped is True
    assert max(x for x, _ in instance.polygon_xy) == 9


def test_totaltext_border_coordinates_are_clipped_explicitly() -> None:
    source = SceneTextInstance(((-2, 1), (10, 1), (10, 8), (0, 8)), "word", False, "word")
    (instance,), count = clip_instances_to_pixel_extent((source,), (10, 8))
    assert count == 1
    assert instance.polygon_xy == ((0, 1), (9, 1), (9, 7), (0, 7))


def _sample_manifest(root: Path, name: str, sample_id: str) -> Path:
    image = root / f"{name}.jpg"
    mask = root / f"{name}-mask.png"
    validity = root / f"{name}-validity.png"
    Image.new("RGB", (8, 6), "white").save(image)
    Image.new("L", (8, 6), 255).save(mask)
    Image.new("L", (8, 6), 255).save(validity)
    from segmentation.data import file_sha256

    payload = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "dataset": name,
        "split": "train",
        "artifact_id": f"ART-{name}",
        "split_role": "train",
        "annotation_policy": {
            "target_conversion": "binary-mask-with-validity-v1",
            "masked_bce": True,
        },
        "samples": [
            {
                "id": sample_id,
                "image": image.name,
                "mask": mask.name,
                "validity_annotation": validity.name,
                "image_sha256": file_sha256(image),
                "mask_sha256": file_sha256(mask),
                "validity_annotation_sha256": file_sha256(validity),
            }
        ],
    }
    path = root / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_binary_mask_validity_loader_and_generic_manifest_composition(tmp_path: Path) -> None:
    first = _sample_manifest(tmp_path, "first", "first:1")
    second = _sample_manifest(tmp_path, "second", "second:1")
    composed = {
        "schema_version": 1,
        "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction",
        "dataset": "first+second",
        "split": "train",
        "artifact_id": "ART-COMBINED",
        "split_role": "train",
        "annotation_policy": {
            "target_conversion": "binary-mask-with-validity-v1",
            "masked_bce": True,
        },
        "source_manifests": [first.name, second.name],
        "expected_sample_count": 2,
        "sample_order": "source_manifest_order_then_source_sample_order",
    }
    combined_path = tmp_path / "combined.yaml"
    combined_path.write_text(yaml.safe_dump(composed, sort_keys=False), encoding="utf-8")
    manifest = load_segmentation_manifest(combined_path)
    dataset = SegmentationDataset(manifest, tmp_path)
    assert [sample.sample_id for sample in manifest.samples] == ["first:1", "second:1"]
    assert dataset[0]["validity"].mode == "L"


def test_domain_balanced_sampler_is_deterministic_and_balanced() -> None:
    sample_ids = ["a:1", "a:2", "b:1", "b:2", "b:3", "c:1"]
    sampler = DomainBalancedEpochSampler(sample_ids, seed=17)
    epoch_zero = list(sampler)
    assert epoch_zero == list(DomainBalancedEpochSampler(sample_ids, seed=17))
    domains = Counter(sample_ids[index].split(":", 1)[0] for index in epoch_zero)
    assert domains == {"a": 3, "b": 3, "c": 3}
    sampler.set_epoch(1)
    assert list(sampler) != epoch_zero


def test_all_scene_text_training_configs_validate() -> None:
    names = (
        "segmentation_ctw1500_author_hyperparameters.yaml",
        "segmentation_icdar2015_author_hyperparameters.yaml",
        "segmentation_total_text_author_hyperparameters.yaml",
        "segmentation_scene_text3_balanced.yaml",
    )
    for name in names:
        config = load_segmentation_config(ROOT / "configs" / "approximate" / name)
        assert config.run.kind == "real_training"
