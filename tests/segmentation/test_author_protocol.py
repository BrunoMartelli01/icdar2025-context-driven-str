from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from models.unet_adv import AuthorProtocolAGUNet
from segmentation.config import load_segmentation_config
from segmentation.data import (
    icdar13_task22_rgb_to_target_and_validity,
    load_segmentation_manifest,
)
from segmentation.training import (
    build_optimizer,
    masked_bce_with_logits,
    measure_foreground_iou,
    SegmentationTrainingConfig,
    select_checkpoint_record,
    train_one_epoch,
)
from segmentation.train import (
    _architecture_record,
    _invocation_final_epoch,
    _print_epoch_progress,
    _print_training_complete,
    _print_training_start,
    _verify_model_device,
)
from segmentation.transforms import PairedSegmentationTransform, SegmentationTransformConfig


ROOT = Path(__file__).resolve().parents[2]
AUTHOR_CONFIG = ROOT / "configs/approximate/segmentation_icdar13_author_protocol.yaml"
AUTHOR_MANIFEST = ROOT / "manifests/icdar13_segmentation_author_protocol.yaml"
HISTORICAL_CONFIG = ROOT / "configs/approximate/segmentation_icdar13.yaml"


class _FlushTrackingStream(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.flush_count = 0

    def flush(self) -> None:
        self.flush_count += 1
        super().flush()


def _author_transform() -> PairedSegmentationTransform:
    return PairedSegmentationTransform(
        SegmentationTransformConfig(
            horizontal_flip_probability=0.0,
            rotation_probability=0.0,
            rotation_degrees=(0.0, 0.0),
            image_interpolation="bilinear",
            mask_interpolation="bilinear",
            image_fill=(0, 0, 0),
            mask_fill=0,
            normalization="none",
            resize_geometry="stretch",
            operation_order="flip_rotate_resize",
            target_binarization_threshold=0.5,
            validity_interpolation="nearest",
        ),
        base_seed=1,
    )


def test_author_model_ends_at_128_and_uses_scalar_skip_preserving_attention() -> None:
    model = AuthorProtocolAGUNet(encoder_pretrained=False).eval()
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 256, 256))
    assert tuple(logits.shape) == (1, 1, 128, 128)
    assert model.up0.att.psi.out_channels == 1
    assert not hasattr(model.up0.att, "bn_psi")
    skip = torch.randn(2, 64, 128, 128)
    gate = torch.randn(2, 64, 128, 128)
    with torch.inference_mode():
        gated = model.up0.att(gate, skip)
    assert gated.shape == skip.shape


def test_author_encoder_parameters_and_batchnorm_state_stay_frozen() -> None:
    model = AuthorProtocolAGUNet(encoder_pretrained=False)
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    running_mean = model.encoder.bn1.running_mean.detach().clone()
    model.train()
    assert model.encoder.training is False
    logits = model(torch.randn(2, 3, 64, 64))
    logits.mean().backward()
    assert torch.equal(model.encoder.bn1.running_mean, running_mean)
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert model.bottleneck[0].weight.grad is not None
    assert model.head.weight.grad is not None


def test_target_downsampling_is_bilinear_then_thresholded_at_half() -> None:
    image = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8), mode="RGB")
    source = np.zeros((256, 256), dtype=np.uint8)
    source[:, 127:] = 255
    mask = Image.fromarray(source, mode="L")
    _, actual = _author_transform()(image, mask, 0)
    expected_float = torch.nn.functional.interpolate(
        torch.from_numpy(source).float()[None, None] / 255.0,
        size=(128, 128),
        mode="bilinear",
        align_corners=False,
    )
    expected = (expected_float >= 0.5).float().squeeze(0)
    assert torch.equal(actual, expected)
    assert set(torch.unique(actual).tolist()) == {0.0, 1.0}


def test_dont_care_color_becomes_invalid_without_changing_other_pixels() -> None:
    rgb = np.array(
        [[[255, 255, 255], [10, 20, 30], [40, 50, 60]]], dtype=np.uint8
    )
    target, validity = icdar13_task22_rgb_to_target_and_validity(
        Image.fromarray(rgb, mode="RGB"),
        '# 10 20 30 0 0 0 0 1 1 " "\n40 50 60 0 0 0 0 1 1 "T"\n',
    )
    assert np.asarray(target).tolist() == [[0, 255, 255]]
    assert np.asarray(validity).tolist() == [[255, 0, 255]]


def test_masked_bce_ignored_pixels_have_zero_loss_and_gradient() -> None:
    logits = torch.tensor([[[[0.0, 10.0]]]], requires_grad=True)
    targets = torch.tensor([[[[1.0, 0.0]]]])
    validity = torch.tensor([[[[1.0, 0.0]]]])
    loss = masked_bce_with_logits(logits, targets, validity)
    loss.backward()
    assert torch.allclose(loss, torch.nn.functional.softplus(torch.tensor(0.0)))
    assert logits.grad is not None
    assert logits.grad[0, 0, 0, 0].item() == -0.5
    assert logits.grad[0, 0, 0, 1].item() == 0.0


def test_author_protocol_config_uses_all_229_without_validation() -> None:
    config = load_segmentation_config(AUTHOR_CONFIG)
    manifest = load_segmentation_manifest(AUTHOR_MANIFEST, require_checksums=True)
    assert len(manifest.samples) == config.data.train.expected_sample_count == 229
    assert len({sample.sample_id for sample in manifest.samples}) == 229
    assert config.validation.enabled is False
    assert config.validation.held_out_count == 0
    assert config.architecture.freeze_encoder is True
    assert config.training.epochs == 100
    assert config.training.num_workers == 8


def test_qualification_step_limit_does_not_repeat_across_100_epochs() -> None:
    assert _invocation_final_epoch(
        configured_epochs=100,
        start_epoch=1,
        qualification_max_steps=2,
        max_epochs_this_invocation=None,
    ) == 1
    assert _invocation_final_epoch(
        configured_epochs=100,
        start_epoch=1,
        qualification_max_steps=None,
        max_epochs_this_invocation=None,
    ) == 100


def test_author_architecture_record_uses_only_real_author_metadata() -> None:
    config = load_segmentation_config(AUTHOR_CONFIG)
    model = AuthorProtocolAGUNet(encoder_pretrained=False)
    record = _architecture_record(model, config)
    assert "contract_artifact" not in record
    assert "state_dict_schema_fingerprint" not in record
    assert record["implementation"] == "author_clarified_ag_unet"
    assert record["identity"] == "author-clarified-ag-unet-2048x1024-v1"
    assert record["encoder"] == "resnet50"
    assert record["encoder_weights"] == "IMAGENET1K_V2"
    assert record["freeze_encoder"] is True
    assert record["frozen_batchnorm_state"] is True
    assert record["bottleneck_channels"] == 1024
    assert record["bottleneck_width_classification"] == "ASSUMPTION"
    assert record["decoder_channels"] == [512, 256, 128, 64]
    assert record["attention_intermediate_channels"] == "decoder_output_half_minimum_16"
    assert record["output_stride"] == 2
    assert record["parameter_count"] == 58_155_081
    assert record["trainable_parameter_count"] == 32_598_049
    assert record["frozen_parameter_count"] == 25_557_032


def test_historical_architecture_record_retains_legacy_contract_metadata() -> None:
    config = load_segmentation_config(HISTORICAL_CONFIG)
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    record = _architecture_record(model, config)
    assert record["contract_artifact"] == "provenance/AG_UNET_ARCHITECTURE_M8.json"
    assert record["state_dict_schema_fingerprint"] is None
    assert set(record) == {
        "identity",
        "contract_schema_version",
        "contract_artifact",
        "state_dict_schema_fingerprint",
        "parameter_count",
        "trainable_parameter_count",
        "encoder_weights",
        "encoder_weights_artifact_id",
        "freeze_encoder",
    }


def test_checkpoint_selection_uses_highest_training_foreground_iou() -> None:
    records = [
        {"epoch": 1, "training_selection": {"foreground_iou": 0.7}},
        {"epoch": 2, "training_selection": {"foreground_iou": 0.8}},
        {"epoch": 3, "training_selection": {"foreground_iou": 0.75}},
    ]
    selected = select_checkpoint_record(records, "training_foreground_iou")
    assert selected["epoch"] == 2


def test_live_training_messages_include_start_epoch_and_new_best() -> None:
    stream = _FlushTrackingStream()
    architecture = {
        "parameter_count": 100,
        "trainable_parameter_count": 40,
        "freeze_encoder": True,
    }
    with redirect_stdout(stream):
        _print_training_start(
            config_path=AUTHOR_CONFIG,
            protocol="author_clarified_icdar13_v1",
            device=torch.device("cpu"),
            gpu_name=None,
            train_images=229,
            batch_size=8,
            epochs=100,
            architecture=architecture,
            input_resolution=[256, 256],
            logits_resolution=[128, 128],
            checkpoint_selection="training_foreground_iou",
            compute_device="cpu",
        )
        _print_epoch_progress(
            epoch=1,
            epochs=100,
            train_loss=0.25,
            train_fg_iou=0.75,
            best_fg_iou=0.75,
            learning_rate=1e-4,
            elapsed_seconds=12.5,
            cuda_allocated=None,
            cuda_reserved=None,
            new_best_checkpoint="checkpoints/epoch-0001.pt",
            compute_device="cpu",
        )
        _print_training_complete(
            status="SUCCESS",
            best_epoch=1,
            best_fg_iou=0.75,
            checkpoint_path=Path("checkpoints/epoch-0001.pt"),
            elapsed_seconds=12.5,
            cuda_allocated=None,
            cuda_reserved=None,
        )
    output = stream.getvalue()
    assert "training_start device=cpu" in output
    assert "train_images=229 batch_size=8 epochs=100 compute_device=cpu" in output
    assert "parameters_total=100 parameters_trainable=40 parameters_frozen=60" in output
    assert "epoch=001/100 train_loss=0.250000 train_fgIoU=0.750000" in output
    assert "lr=0.0001 compute_device=cpu" in output
    assert "new_best_checkpoint=checkpoints/epoch-0001.pt" in output
    assert "training_complete status=SUCCESS best_epoch=1" in output
    assert stream.flush_count >= 4


def test_training_primitives_report_and_enforce_compute_device() -> None:
    model = torch.nn.Conv2d(3, 1, kernel_size=1)
    assert _verify_model_device(model, torch.device("cpu")) == "cpu"
    try:
        _verify_model_device(model, torch.device("meta"))
    except RuntimeError as error:
        assert "not fully on meta" in str(error)
    else:
        raise AssertionError("A model/device mismatch must fail before training")

    config = SegmentationTrainingConfig(
        epochs=1,
        seed=1,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        learning_rate_schedule="none",
        checkpoint_selection="last",
        deterministic_algorithms=True,
    )
    optimizer = build_optimizer(model, config)
    metrics = train_one_epoch(
        model,
        [
            {
                "image": torch.zeros(2, 3, 4, 4),
                "mask": torch.zeros(2, 1, 4, 4),
            }
        ],
        optimizer,
        device=torch.device("cpu"),
    )
    assert metrics["compute_device"] == "cpu"


def test_vectorized_training_iou_preserves_per_image_protocol() -> None:
    class FirstChannel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return images[:, :1]

    images = torch.tensor(
        [
            [[[10.0, -10.0], [-10.0, -10.0]]],
            [[[10.0, 10.0], [10.0, 10.0]]],
        ]
    )
    targets = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 0.0]]],
            [[[1.0, 0.0], [0.0, 0.0]]],
        ]
    )
    validity = torch.ones_like(targets)
    metrics = measure_foreground_iou(
        FirstChannel(),
        [{"image": images, "mask": targets, "validity": validity}],
        device=torch.device("cpu"),
        probability_threshold=0.5,
        aggregation="per_image_macro",
    )
    assert metrics["foreground_iou"] == (1.0 + 0.25) / 2
    assert metrics["intersection"] == 2
    assert metrics["union"] == 5
    assert metrics["compute_device"] == "cpu"
