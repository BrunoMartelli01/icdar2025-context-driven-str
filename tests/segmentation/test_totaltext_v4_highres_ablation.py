"""Isolated high-resolution practical ablation contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import numpy as np
import torch

from models.unet_adv import AttentionGate, AuthorProtocolAGUNet, PracticalTotalTextHighResAGUNet
from scripts.train_totaltext_v4_highres_ablation import _load_config, _split_indices, word_topology_metrics
from segmentation.training import select_checkpoint_record
from segmentation.transforms import SegmentationTransformConfig


def _transform_config(*, classification: str, image_size: tuple[int, int], mask_size: tuple[int, int]) -> SegmentationTransformConfig:
    return SegmentationTransformConfig(
        horizontal_flip_probability=0.0, rotation_probability=0.0,
        rotation_degrees=(-15.0, 15.0), image_interpolation="bilinear",
        mask_interpolation="bilinear", image_fill=(0, 0, 0), mask_fill=0,
        normalization="imagenet", resize_geometry="stretch", operation_order="flip_rotate_resize",
        image_size=image_size, mask_size=mask_size, classification=classification,
    )


def test_highres_transform_requires_explicit_practical_classification() -> None:
    _transform_config(classification="PRACTICAL_APPROXIMATION_HIGH_RESOLUTION",
                      image_size=(512, 512), mask_size=(256, 256))
    with pytest.raises(ValueError, match="Transform sizes"):
        _transform_config(classification="ASSUMPTION", image_size=(512, 512), mask_size=(256, 256))
    with pytest.raises(ValueError, match="Transform sizes"):
        _transform_config(classification="PRACTICAL_APPROXIMATION_HIGH_RESOLUTION",
                          image_size=(256, 256), mask_size=(128, 128))


def test_residual_floor_is_parameter_free_and_preserves_default_gate() -> None:
    torch.manual_seed(11)
    gate = AttentionGate(8, 6, 4).eval()
    g = torch.randn(1, 8, 4, 4)
    x = torch.randn(1, 6, 4, 4)
    with torch.inference_mode():
        baseline = gate(g, x)
        gate.practical_skip_floor = 0.25
        residual = gate(g, x)
    torch.testing.assert_close(residual, 0.75 * baseline + 0.25 * x)
    assert sum(p.numel() for p in gate.parameters()) == 304


def test_highres_model_identity_and_parameter_count() -> None:
    original = AuthorProtocolAGUNet(encoder_pretrained=False)
    practical = PracticalTotalTextHighResAGUNet(encoder_pretrained=False, skip_floor=0.25)
    assert original.architecture_identity != practical.architecture_identity
    assert sum(p.numel() for p in original.parameters()) == 58155081
    assert sum(p.numel() for p in practical.parameters()) == 58155081
    assert sum(p.numel() for p in practical.parameters() if p.requires_grad) == 32598049
    assert all(stage.att.practical_skip_floor == 0.25 for stage in
               (practical.up3, practical.up2, practical.up1, practical.up0))


def test_train_validation_split_is_stable_and_disjoint() -> None:
    manifest = SimpleNamespace(samples=tuple(SimpleNamespace(sample_id=f"total_text:{i}") for i in range(1255)))
    train, validation, digest = _split_indices(manifest, 202503, 125)
    assert len(train) == 1130 and len(validation) == 125
    assert set(train).isdisjoint(validation)
    assert len(digest) == 64
    assert (train, validation, digest) == _split_indices(manifest, 202503, 125)


def test_three_predeclared_variants() -> None:
    config = _load_config()
    assert config["variants"] == {
        "highres_baseline": {"separation_weight": 1.0, "skip_floor": 0.0},
        "highres_separator": {"separation_weight": 3.0, "skip_floor": 0.0},
        "highres_separator_residual": {"separation_weight": 3.0, "skip_floor": 0.25},
    }


def test_word_topology_detects_fusion_and_split_without_text() -> None:
    owners = np.zeros((256, 256), dtype=np.int32)
    owners[20:30, 20:35] = 1
    owners[20:30, 45:60] = 2
    separated = owners > 0
    baseline = word_topology_metrics(separated, owners)
    assert baseline["gt_words"] == 2
    assert baseline["gt_with_predicted_bbox_iou_ge_0_5"] == 2
    assert baseline["predicted_components_touching_multiple_gt_words"] == 0
    fused = separated.copy()
    fused[25, 35:45] = True
    merged = word_topology_metrics(fused, owners)
    assert merged["predicted_components_touching_multiple_gt_words"] == 1
    assert merged["gt_words_touched_by_fused_components"] == 2
    split = separated.copy()
    split[20:30, 27] = False
    fragmented = word_topology_metrics(split, owners)
    assert fragmented["gt_words_split_across_predicted_components"] == 1
    assert word_topology_metrics(separated, owners, expected_owner_ids=(1, 2, 3))["gt_words_lost_in_native_target"] == 1


def test_validation_iou_checkpoint_selection_prefers_earliest_exact_tie() -> None:
    records = [
        {"epoch": 1, "validation": {"foreground_iou": 0.7}},
        {"epoch": 2, "validation": {"foreground_iou": 0.8}},
        {"epoch": 3, "validation": {"foreground_iou": 0.8}},
    ]
    assert select_checkpoint_record(records, "validation_foreground_iou")["epoch"] == 2
