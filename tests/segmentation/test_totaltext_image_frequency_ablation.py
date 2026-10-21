"""Focused CPU checks for isolated Total-Text image/frequency interventions."""

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from models.totaltext_wavelet import PracticalTotalTextWaveletHighResAGUNet
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct, _geometry_match, _tensor
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _geometry_differences
from scripts.train_totaltext_v4_image_frequency_ablation import (
    _completed_run_matches_budget, _effective_epoch_budget, proposal_metrics,
)
from segmentation.totaltext_image_frequency_augment import (
    PracticalTotalTextImageTransform, letterbox_canvas_mask, letterbox_geometry,
    mix_low_frequency_amplitude, remap_letterbox_native_box,
)
from segmentation.transforms import SegmentationTransformConfig


def _config():
    return SegmentationTransformConfig(
        horizontal_flip_probability=0.0, rotation_probability=0.0,
        rotation_degrees=(-15.0, 15.0), image_interpolation="bilinear",
        mask_interpolation="bilinear", image_fill=(0, 0, 0), mask_fill=0,
        normalization="imagenet", resize_geometry="stretch",
        operation_order="flip_rotate_resize", image_size=(512, 512),
        mask_size=(256, 256), classification="PRACTICAL_APPROXIMATION_HIGH_RESOLUTION",
    )


def test_short_budget_preserves_completed_full_run_and_rejects_false_full_claim():
    assert _effective_epoch_budget(500, 100) == 100
    assert _effective_epoch_budget(500, None) == 500
    with pytest.raises(ValueError, match="between 1"):
        _effective_epoch_budget(500, 0)
    with pytest.raises(ValueError, match="between 1"):
        _effective_epoch_budget(500, 501)
    historical_full = {"epochs": [{}] * 500}
    assert _completed_run_matches_budget(historical_full, 100, 500)
    short = {"epochs": [{}] * 100, "execution_budget": {
        "configured_epochs": 500, "requested_epochs": 100, "completed_epochs": 100}}
    assert _completed_run_matches_budget(short, 100, 500)
    assert not _completed_run_matches_budget(short, 500, 500)
    assert not _completed_run_matches_budget(short, 75, 500)


def test_geometry_batching_audit_detects_exact_per_image_drift():
    frozen = [{"sample_id": "a", "gt_words": 2, "matched_gt_iou_ge_50": 2},
              {"sample_id": "b", "gt_words": 1, "matched_gt_iou_ge_50": 1}]
    cached = [{"sample_id": "b", "gt_words": 1, "matched_gt_iou_ge_50": 0}, frozen[0]]
    differences = _geometry_differences(cached, frozen)
    assert len(differences) == 1
    assert differences[0]["sample_id"] == "b"
    assert differences[0]["frozen_batch_8"]["matched_gt_iou_ge_50"] == 1
    assert differences[0]["cached_batch_1"]["matched_gt_iou_ge_50"] == 0
    assert _geometry_differences(frozen, frozen) == []
    with pytest.raises(RuntimeError, match="sample IDs"):
        _geometry_differences(cached[:1], frozen)


def _pair():
    image = Image.new("RGB", (64, 32), (100, 120, 140))
    mask = Image.new("L", image.size, 0)
    mask_array = np.asarray(mask).copy()
    mask_array[8:24, 16:48] = 255
    mask = Image.fromarray(mask_array)
    validity = Image.new("L", image.size, 255)
    separation = Image.new("L", image.size, 0)
    owner = Image.fromarray((mask_array > 0).astype(np.int32), mode="I")
    return image, mask, validity, separation, owner


def test_letterbox_geometry_and_paired_owner_maps():
    image, mask, validity, separation, owner = _pair()
    transform = PracticalTotalTextImageTransform(_config(), base_seed=3,
        mode="letterbox", augment_images=True,
        parameters={"canvas_fill_rgb": [124, 116, 104]})
    rgb, target, valid, sep = transform.transform_with_validity_and_separation(
        image, mask, validity, separation, owner, 4)
    assert rgb.shape == (3, 512, 512)
    assert target.shape == valid.shape == sep.shape == (1, 256, 256)
    assert torch.count_nonzero(target[:, :64]) == 0
    assert torch.count_nonzero(valid[:, :64]) == 0
    assert torch.count_nonzero(target) > 0
    assert torch.equal(target, sep.logical_not().float() * target)
    assert letterbox_geometry(image, fill=(0, 0, 0)).size == (64, 64)


def test_letterbox_canvas_and_inverse_crop_coordinates():
    canvas = letterbox_canvas_mask((64, 32))
    assert canvas.shape == (256, 256)
    assert not canvas[:64].any() and canvas[64:192].all() and not canvas[192:].any()
    assert remap_letterbox_native_box((64, 64, 192, 192), (64, 32)) == [16, 0, 48, 32]
    with pytest.raises(ValueError, match="does not intersect"):
        remap_letterbox_native_box((0, 0, 16, 16), (64, 32))


def test_letterbox_inference_input_equals_validation_transform():
    pair = _pair()
    transform = PracticalTotalTextImageTransform(_config(), base_seed=4,
        mode="letterbox", augment_images=False,
        parameters={"canvas_fill_rgb": [124, 116, 104]})
    reference = transform.transform_with_validity_and_separation(*pair, 3)[0]
    observed = _tensor(np.asarray(pair[0]), transform)[0]
    assert torch.equal(reference, observed)


def test_photometric_changes_only_rgb_and_is_worker_independent():
    pair = _pair()
    transform = PracticalTotalTextImageTransform(_config(), base_seed=8,
        mode="photometric", augment_images=True,
        parameters={"probability": 1.0, "magnitude": 0.2,
                    "operations": ["brightness", "contrast", "saturation", "gamma",
                                   "gaussian_blur", "gaussian_noise"],
                    "blur_sigma_min": 0.1, "blur_sigma_max": 1.0,
                    "noise_std_max": 0.015})
    first = transform.transform_with_validity_and_separation(*pair, 9)
    second = transform.transform_with_validity_and_separation(*pair, 9)
    control = PracticalTotalTextImageTransform(_config(), base_seed=8,
        mode="none", augment_images=False).transform_with_validity_and_separation(*pair, 9)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert all(torch.equal(a, b) for a, b in zip(first[1:], control[1:]))
    assert not torch.equal(first[0], control[0])


def test_fourier_amplitude_changes_only_low_frequency_and_fails_without_train_pool(tmp_path: Path):
    source = torch.zeros(3, 32, 32)
    source[:, 8:24, 8:24] = 0.9
    donor = torch.full_like(source, 0.6)
    mixed = mix_low_frequency_amplitude(source, donor, fraction=0.05, strength=0.5)
    assert mixed.shape == source.shape and torch.isfinite(mixed).all()
    assert not torch.equal(mixed, source)
    assert torch.equal(mix_low_frequency_amplitude(source, donor, fraction=0.05, strength=0), source)
    with pytest.raises(ValueError, match="TRAIN images"):
        PracticalTotalTextImageTransform(_config(), base_seed=1,
            mode="fourier_amplitude", augment_images=True,
            parameters={"probability": 1.0, "fraction": 0.02, "strength": 0.5},
            donor_paths=(tmp_path / "one.png",), donor_indices=(0,))


def test_fourier_donor_is_from_train_pool_and_labels_are_unchanged(tmp_path: Path):
    donor_paths = (tmp_path / "train0.png", tmp_path / "train1.png")
    Image.new("RGB", (16, 16), (20, 30, 40)).save(donor_paths[0])
    Image.new("RGB", (16, 16), (220, 210, 200)).save(donor_paths[1])
    transform = PracticalTotalTextImageTransform(_config(), base_seed=3,
        mode="fourier_amplitude", augment_images=True,
        parameters={"probability": 1.0, "fraction": 0.02, "strength": 0.5},
        donor_paths=donor_paths, donor_indices=(0, 1))
    first = transform.transform_with_validity_and_separation(*_pair(), 0)
    second = transform.transform_with_validity_and_separation(*_pair(), 0)
    control = PracticalTotalTextImageTransform(_config(), base_seed=3,
        mode="none", augment_images=False).transform_with_validity_and_separation(*_pair(), 0)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert all(torch.equal(a, b) for a, b in zip(first[1:], control[1:]))
    assert not torch.equal(first[0], control[0])


def test_proposal_metrics_no_area_filter_or_top10():
    owners = np.zeros((256, 256), dtype=np.int32)
    prediction = np.zeros_like(owners, dtype=np.bool_)
    for index in range(12):
        top = index * 18
        owners[top:top+3, 2:5] = index + 1
        prediction[top:top+3, 2:5] = True
    result = proposal_metrics(prediction, owners)
    assert result["gt_words"] == 12
    assert result["raw_proposals"] == 12
    assert result["matched_gt_iou_ge_50"] == 12
    assert result["gt_words_touched_by_fused_components"] == 0


def test_ocr_evaluation_is_spatial_one_to_one_and_symbol_normalized():
    gt = [{"bbox": [0, 0, 20, 20], "text": "DON'T"},
          {"bbox": [25, 0, 45, 20], "text": "Stop!"}]
    predictions = [{"bbox": [0, 0, 20, 20], "text": "dont"},
                   {"bbox": [25, 0, 45, 20], "text": "STOP"},
                   {"bbox": [0, 0, 20, 20], "text": "dont"}]
    assert _geometry_match(predictions, gt, 0.5) == {0, 1}
    assert _correct(predictions, gt, normalize=False) == set()
    assert _correct(predictions, gt, normalize=True) == {0, 1}


def test_haar_adapter_is_zero_initialized_and_keeps_pretrained_rgb_stem():
    model = PracticalTotalTextWaveletHighResAGUNet(encoder_pretrained=False).eval()
    assert model.architecture_identity == "practical-totaltext-v4-highres-haar-adapter-v1"
    assert model.encoder.conv1.in_channels == 3
    assert model.wavelet_adapter.weight.numel() == 192
    assert torch.count_nonzero(model.wavelet_adapter.weight) == 0
    with torch.inference_mode():
        output = model(torch.randn(1, 3, 64, 64))
    assert output.shape == (1, 1, 32, 32)


def test_haar_adapter_receives_gradients_while_encoder_stays_frozen():
    model = PracticalTotalTextWaveletHighResAGUNet(encoder_pretrained=False).train()
    model(torch.randn(1, 3, 64, 64)).mean().backward()
    assert model.wavelet_adapter.weight.grad is not None
    assert torch.count_nonzero(model.wavelet_adapter.weight.grad) > 0
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
