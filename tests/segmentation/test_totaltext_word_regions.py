from __future__ import annotations

import numpy as np
from pathlib import Path
from PIL import Image
import pytest
import torch
import yaml

from recognition.scene_text_datasets import SceneTextInstance
from segmentation.totaltext_word_regions import (
    _native_word_component_counts, make_word_regions, refine_word_regions_with_pixel_gt,
    restore_worsened_native_words,
)
from segmentation.data import SegmentationDataset, file_sha256, load_segmentation_manifest
from segmentation.training import separation_weighted_bce_with_logits
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


def _word(points: tuple[tuple[int, int], ...], text: str) -> SceneTextInstance:
    return SceneTextInstance(
        polygon_xy=points, transcription=text, ignore=False,
        granularity="word", orientation="c" if len(points) > 4 else "h",
    )


def test_curved_words_are_filled_and_disjoint_with_separator() -> None:
    curved = _word(((2, 9), (5, 6), (12, 5), (20, 7), (20, 12), (13, 10), (5, 12)), "CURVE")
    overlapping = _word(((16, 7), (28, 7), (28, 13), (16, 13)), "NEXT")
    target, validity, separator, owners, audit = make_word_regions(
        (32, 20), (curved, overlapping)
    )
    owner = np.asarray(owners)
    mask = np.asarray(target)
    sep = np.asarray(separator)
    assert audit.word_count == 2
    assert audit.curved_word_count == 1
    assert audit.contested_pixels > 0
    assert audit.empty_word_count == 0
    assert set(np.unique(owner)) == {0, 1, 2}
    assert np.array_equal(mask > 0, owner > 0)
    assert np.all(mask[sep > 0] == 0)
    assert np.all(np.asarray(validity) == 255)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if not dx and not dy:
                continue
            shifted = np.roll(owner, (dy, dx), axis=(0, 1))
            assert not np.any((owner > 0) & (shifted > 0) & (owner != shifted))


def test_tiny_ignore_point_does_not_break_rasterization() -> None:
    word = _word(((2, 2), (11, 2), (11, 9), (2, 9)), "TEXT")
    ignored = SceneTextInstance(
        polygon_xy=((15, 4),), transcription="###", ignore=True,
        granularity="word", orientation="#",
    )
    _, validity, _, _, _ = make_word_regions((20, 12), (word, ignored))
    assert np.asarray(validity)[4, 15] == 0


@pytest.mark.parametrize("ignore_whole_word", [False, True])
def test_optional_thin_word_preservation_never_restores_ignored_pixels(ignore_whole_word):
    large = _word(((8, 10), (40, 10), (40, 35), (8, 35)), "large")
    thin = _word(((12, 9), (32, 9), (32, 9), (12, 9)), "--")
    right = 32 if ignore_whole_word else 18
    ignored = SceneTextInstance(((12, 8), (right, 8), (right, 10), (12, 10)),
                                "###", True, "word")
    mask, validity, separator, owners, audit = make_word_regions(
        (50, 40), (large, thin, ignored), preserve_contact_words=True)
    invalid = np.asarray(validity) == 0
    assert not np.asarray(owners)[invalid].any()
    assert not np.asarray(mask)[invalid].any()
    assert not np.asarray(separator)[invalid].any()
    assert audit.restored_contact_word_ids == (() if ignore_whole_word else (2,))
    assert audit.empty_word_count == int(ignore_whole_word)


def test_pixel_gt_refinement_is_connected_tighter_and_preserves_seed() -> None:
    curved = _word(((2, 6), (8, 3), (17, 4), (26, 7), (26, 17), (17, 14), (8, 15), (2, 18)), "CURVE")
    target, validity, _, owner, _ = make_word_regions((32, 22), (curved,))
    seed = np.zeros((22, 32), dtype=np.uint8)
    seed[9:12, 5:8] = 255
    seed[8:11, 13:16] = 255
    seed[10:13, 22:25] = 255
    old = Image.fromarray(seed, mode="L")
    refined, labels, audit = refine_word_regions_with_pixel_gt(owner, old, validity)
    out = np.asarray(refined) > 0
    assert np.all(out[seed > 0])
    assert np.all(out <= (np.asarray(target) > 0))
    assert 0 < audit.refined_owner_pixels < audit.original_owner_pixels
    assert audit.pixel_gt_seed_retained == audit.pixel_gt_seed_pixels
    from scipy.ndimage import label
    assert label(out, structure=np.ones((3, 3)))[1] == 1
    assert np.array_equal(np.asarray(labels) > 0, out)


def test_native_topology_guard_restores_only_fragmented_word() -> None:
    base = np.zeros((256, 256), dtype=np.uint16)
    base[32:65, 24:100] = 1
    base[112:145, 24:100] = 2
    refined = np.zeros_like(base)
    refined[40:55, 30:45] = 1
    refined[40:55, 80:95] = 1
    refined[120:135, 30:95] = 2
    target, owners, restored = restore_worsened_native_words(
        Image.fromarray(base, mode="I;16"),
        Image.fromarray(refined, mode="I;16"),
        Image.new("L", (256, 256), 0),
        Image.new("L", (256, 256), 255),
    )
    result = np.asarray(owners)
    assert restored == 1
    assert np.array_equal(result[base == 1], base[base == 1])
    assert np.array_equal(result[base == 2], refined[base == 2])
    assert np.array_equal(np.asarray(target) > 0, result > 0)


def test_native_topology_guard_rejects_cross_word_pixels() -> None:
    base = np.zeros((256, 256), dtype=np.uint16)
    base[20:50, 20:50] = 1
    refined = base.copy()
    refined[20, 20] = 2
    with pytest.raises(ValueError, match="subset"):
        restore_worsened_native_words(
            Image.fromarray(base, mode="I;16"),
            Image.fromarray(refined, mode="I;16"),
            Image.new("L", (256, 256), 0),
            Image.new("L", (256, 256), 255),
        )


def test_native_topology_audit_matches_training_transform() -> None:
    from scipy.ndimage import label

    owner = np.zeros((256, 256), dtype=np.int32)
    owner[20:48, 20:40] = 1
    owner[20:48, 46:66] = 1
    owner[80:120, 30:110] = 2
    mask = Image.fromarray((owner > 0).astype(np.uint8) * 255, mode="L")
    owners = Image.fromarray(owner, mode="I")
    valid = Image.new("L", (256, 256), 255)
    separator = Image.new("L", (256, 256), 0)
    transform = PairedSegmentationTransform(
        SegmentationTransformConfig(
            horizontal_flip_probability=0.0,
            rotation_probability=0.0,
            rotation_degrees=(0.0, 0.0),
            image_interpolation="bilinear", mask_interpolation="bilinear",
            image_fill=(0, 0, 0), mask_fill=0, normalization="none",
            resize_geometry="stretch", operation_order="flip_rotate_resize",
        ), base_seed=5,
    )
    _, transformed, _, _ = transform.transform_with_validity_and_separation(
        Image.new("RGB", (256, 256), "white"), mask, valid, separator, owners, 0
    )
    native_owner = np.asarray(owners.resize((128, 128), Image.Resampling.NEAREST))
    native_mask = transformed.squeeze(0).numpy() > 0
    audited = _native_word_component_counts(owner, separator=separator, validity=valid)
    for index, count in audited.items():
        _, actual = label(
            native_mask & (native_owner == index),
            structure=np.ones((3, 3), dtype=np.uint8),
        )
        assert actual == count


def test_separation_weight_increases_false_bridge_penalty() -> None:
    logits = torch.tensor([[[[0.0, 0.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0]]]])
    valid = torch.ones_like(target)
    separation = torch.tensor([[[[0.0, 1.0]]]])
    baseline = separation_weighted_bce_with_logits(
        logits, target, valid, separation, separation_weight=1.0
    )
    weighted = separation_weighted_bce_with_logits(
        logits, target, valid, separation, separation_weight=4.0
    )
    assert float(weighted) == pytest.approx(float(baseline))
    weighted.backward()
    assert logits.grad[0, 0, 0, 1] > abs(logits.grad[0, 0, 0, 0])
    with pytest.raises(ValueError, match="zero-target"):
        separation_weighted_bce_with_logits(
            logits, target, valid, target, separation_weight=4.0
        )


def test_separation_manifest_and_transform_are_paired(tmp_path: Path) -> None:
    word_a = _word(((2, 3), (14, 3), (14, 14), (2, 14)), "A")
    word_b = _word(((12, 3), (24, 3), (24, 14), (12, 14)), "B")
    target, validity, separator, owners, _ = make_word_regions((32, 20), (word_a, word_b))
    files = {
        "image": Image.new("RGB", (32, 20), "white"),
        "mask": target,
        "validity_annotation": validity,
        "separation_annotation": separator,
        "word_owner_annotation": owners,
    }
    record = {"id": "total_text:sample"}
    for key, image in files.items():
        path = tmp_path / f"{key}.png"
        image.save(path)
        record[key] = path.name
        record[f"{key}_sha256"] = file_sha256(path)
    payload = {
        "schema_version": 1, "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction", "dataset": "total_text",
        "split": "train", "artifact_id": "ART-TOTALTEXT-OFFICIAL-SPLIT",
        "split_role": "train",
        "annotation_policy": {"target_conversion": "binary-mask-with-validity-v1", "masked_bce": True},
        "samples": [record],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    transform = PairedSegmentationTransform(
        SegmentationTransformConfig(
            horizontal_flip_probability=1.0,
            rotation_probability=0.0,
            rotation_degrees=(0.0, 0.0),
            image_interpolation="bilinear", mask_interpolation="bilinear",
            image_fill=(0, 0, 0), mask_fill=0, normalization="none",
            resize_geometry="stretch", operation_order="flip_rotate_resize",
        ), base_seed=5,
    )
    sample = SegmentationDataset(
        load_segmentation_manifest(manifest_path, require_checksums=True),
        tmp_path, transform=transform, verify_checksums=True,
    )[0]
    assert sample["mask"].shape == (1, 128, 128)
    assert sample["separation"].shape == (1, 128, 128)
    assert torch.count_nonzero(sample["separation"]) > 0
    assert torch.count_nonzero(sample["mask"] * sample["separation"]) == 0
