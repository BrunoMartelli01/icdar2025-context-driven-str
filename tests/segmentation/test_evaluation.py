from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from segmentation.config import SegmentationConfigError, load_segmentation_config
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import (
    EXPECTED_CHECKPOINT_SHA256,
    SegmentationEvaluationError,
    _prepare_cuda_memory_tracking,
    _sample_record,
    aggregate_records,
    resize_binary_target_to_native_logits,
    resize_validity_to_native_logits,
    validate_checkpoint,
)


ROOT = Path(__file__).resolve().parents[2]


def _success(sample_id: str, tp: int, fp: int, fn: int) -> dict[str, object]:
    union = tp + fp + fn
    return {"sample_id": sample_id, "status": "success", "failure_type": None,
            "failure_message": None, "tp": tp, "fp": fp, "fn": fn,
            "foreground_iou": 1.0 if not union else tp / union,
            "foreground_f1": 1.0 if not union else 2 * tp / (2 * tp + fp + fn),
            "both_empty": not union}


def test_test_manifest_is_complete_ordered_and_checksum_frozen() -> None:
    path = ROOT / "manifests/icdar13_segmentation_test.yaml"
    manifest = yaml.safe_load(path.read_text())
    assert len(manifest["samples"]) == 233
    assert [item["id"] for item in manifest["samples"]] == [str(i) for i in range(1, 234)]
    assert all(len(item["image_sha256"]) == len(item["mask_sha256"]) == 64
               for item in manifest["samples"])
    assert manifest["annotation_policy"]["dont_care_evaluation_behavior"] == "foreground"
    assert manifest["annotation_policy"]["scope"].startswith("M10 evaluation only")


def test_metric_fixture_macro_primary_micro_diagnostic_and_empty() -> None:
    records = [_success("1", 1, 0, 0), _success("2", 1, 1, 1), _success("3", 0, 0, 0)]
    result = aggregate_records(records, expected_count=3)
    assert result["primary_per_image_macro"]["foreground_iou"] == pytest.approx((1 + 1/3 + 1)/3)
    assert result["primary_per_image_macro"]["foreground_f1"] == pytest.approx((1 + 1/2 + 1)/3)
    assert result["diagnostic_global_micro"]["foreground_iou"] == pytest.approx(1/2)
    assert result["both_empty_count"] == 1


def test_failures_are_retained_and_suppress_aggregates() -> None:
    failure = {**_success("2", 0, 0, 0), "status": "failure"}
    result = aggregate_records([_success("1", 1, 0, 0), failure], expected_count=2)
    assert result["failure_count"] == 1
    assert result["primary_per_image_macro"] is None
    assert result["diagnostic_global_micro"] is None
    with pytest.raises(SegmentationEvaluationError, match="No omitted"):
        aggregate_records([_success("1", 1, 0, 0)], expected_count=2)


def test_aggregation_is_deterministic() -> None:
    records = [_success(str(i), i, i % 2, (i + 1) % 2) for i in range(1, 8)]
    assert aggregate_records(records, expected_count=7) == aggregate_records(records, expected_count=7)


def test_checkpoint_hash_rejected_before_loading(tmp_path: Path) -> None:
    checkpoint = tmp_path / "wrong.pt"
    checkpoint.write_bytes(b"wrong")
    with pytest.raises(SegmentationEvaluationError, match="SHA256 mismatch"):
        validate_checkpoint(checkpoint, object(), EXPECTED_CHECKPOINT_SHA256)  # type: ignore[arg-type]


def test_evaluation_config_freezes_contract(tmp_path: Path) -> None:
    path = ROOT / "configs/approximate/segmentation_table3_icdar13.yaml"
    config = load_segmentation_config(path)
    assert config.evaluation.threshold_resolution == "native_128x128_logits"
    assert config.evaluation.primary_aggregation == "per_image_macro"
    raw = yaml.safe_load(path.read_text())
    raw["evaluation"]["probability_threshold"] = 0.6
    changed = tmp_path / "segmentation_table3_icdar13.yaml"
    changed.write_text(yaml.safe_dump(raw))
    with pytest.raises(SegmentationConfigError, match="probability_threshold"):
        load_segmentation_config(changed)


def test_author_evaluation_config_is_isolated_from_legacy_m9() -> None:
    path = ROOT / "configs/approximate/segmentation_icdar13_author_protocol_test.yaml"
    config = load_segmentation_config(path)
    assert config.architecture.implementation == "author_clarified_ag_unet"
    assert config.architecture.identity == "author-clarified-ag-unet-2048x1024-v1"
    assert config.architecture.freeze_encoder is True
    assert config.checkpoint.artifact_id == (
        "ART-ICDAR13-AUTHOR-PROTOCOL-APPROX-TRAINED-CHECKPOINT"
    )
    assert config.checkpoint.sha256 != EXPECTED_CHECKPOINT_SHA256
    assert config.data.test.manifest.endswith("icdar13_segmentation_test.yaml")
    assert config.evaluation.evaluation_resolution == "native_128x128"
    assert config.evaluation.evaluation_resolution_classification == "ASSUMPTION"
    assert config.evaluation.binary_mask_backprojection == "none"
    assert config.evaluation.gt_resize_interpolation == "bilinear"
    assert config.evaluation.gt_binarization_threshold == 0.5
    assert config.evaluation.validity_resize_interpolation == "nearest"


def test_task1_conservative_ignore_config_and_manifest_are_isolated() -> None:
    path = ROOT / (
        "configs/approximate/"
        "segmentation_icdar13_author_protocol_test_task1_ignore.yaml"
    )
    config = load_segmentation_config(path)
    assert config.evaluation.dont_care == "excluded_by_validity_mask"
    assert config.evaluation.gt_conversion == (
        "icdar13-task2.2-task1-box-zero-overlap-validity-v1"
    )
    assert config.evaluation.valid_mask == "task1_care_box_zero_overlap_colors"
    assert config.evaluation.dont_care_classification == "ASSUMPTION"
    manifest = load_segmentation_manifest(
        ROOT / str(config.data.test.manifest)
    )
    assert len(manifest.samples) == 233
    assert manifest.annotation_policy["classification"] == "ASSUMPTION"
    assert manifest.samples[0].validity_annotation_path.as_posix() == (
        "raw/test_task1_gt/gt_img_1.txt"
    )
    assert manifest.samples[-1].validity_annotation_path.as_posix() == (
        "raw/test_task1_gt/gt_img_233.txt"
    )


def test_native_evaluation_target_is_bilinear_then_thresholded() -> None:
    source = np.zeros((256, 256), dtype=np.uint8)
    source[:, 127:] = 255
    target = Image.fromarray(source, mode="L")
    actual = resize_binary_target_to_native_logits(target)
    expected = (
        torch.nn.functional.interpolate(
            torch.from_numpy(source).float()[None, None] / 255.0,
            size=(128, 128),
            mode="bilinear",
            align_corners=False,
        )
        >= 0.5
    )[0, 0]
    assert actual.shape == (128, 128)
    assert torch.equal(actual, expected)


def test_native_validity_uses_nearest_and_removes_ignored_metrics() -> None:
    validity_source = np.full((256, 256), 255, dtype=np.uint8)
    validity_source[:, 128:] = 0
    validity = resize_validity_to_native_logits(
        Image.fromarray(validity_source, mode="L")
    )
    assert validity.shape == (128, 128)
    assert validity[:, :64].all()
    assert not validity[:, 64:].any()

    prediction = torch.tensor([[True, True]])
    target = torch.tensor([[True, False]])
    record = _sample_record(
        "masked", prediction, target, torch.tensor([[True, False]])
    )
    assert record["tp"] == 1
    assert record["fp"] == record["fn"] == 0
    assert record["foreground_iou"] == 1.0
    assert record["ignored_pixel_count"] == 1


def test_cuda_peak_tracking_initializes_allocator_before_reset(monkeypatch) -> None:
    events = []
    device = torch.device("cuda:0")
    monkeypatch.setattr("torch.cuda.set_device", lambda value: events.append(("set", value)))
    monkeypatch.setattr("torch.empty", lambda *args, **kwargs: events.append(("empty", kwargs["device"])))
    monkeypatch.setattr("torch.cuda.synchronize", lambda value: events.append(("sync", value)))
    monkeypatch.setattr("torch.cuda.empty_cache", lambda: events.append(("empty_cache", None)))
    monkeypatch.setattr(
        "torch.cuda.reset_peak_memory_stats",
        lambda value: events.append(("reset", value)),
    )

    assert _prepare_cuda_memory_tracking(device) == 0
    assert [event[0] for event in events] == [
        "set", "empty", "sync", "empty_cache", "reset"
    ]
    assert events[-1] == ("reset", 0)
