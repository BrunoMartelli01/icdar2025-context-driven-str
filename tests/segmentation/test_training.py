from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from models.ag_unet_approx import ARCHITECTURE_IDENTITY, ApproximateAGUNet

from segmentation.config import StrictSegmentationBlocked, load_segmentation_config
from segmentation.training import (
    SegmentationTrainingConfig,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)


ROOT = Path(__file__).resolve().parents[2]


class TinySegmenter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.network(image)


def _config() -> SegmentationTrainingConfig:
    return SegmentationTrainingConfig(
        epochs=1,
        seed=202503,
        adam_betas=(0.9, 0.999),
        adam_epsilon=1e-8,
        learning_rate_schedule="none",
        checkpoint_selection="last",
        deterministic_algorithms=True,
    )


def test_segmentation_config_is_mode_isolated_and_strict_fails() -> None:
    config = load_segmentation_config(ROOT / "configs/approximate/segmentation_smoke.yaml")
    assert config.training.batch_size == 8
    assert config.claim_label == "approximate reproduction"
    with pytest.raises(StrictSegmentationBlocked, match="A-001"):
        load_segmentation_config(ROOT / "configs/strict/segmentation.yaml")


def test_one_batch_uses_paper_optimizer_values_and_bce() -> None:
    seed_everything(202503, deterministic_algorithms=True)
    model = TinySegmenter()
    optimizer = build_optimizer(model, _config())
    assert optimizer.defaults["lr"] == 1e-4
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    batches = [{"image": torch.rand(8, 3, 16, 16), "mask": torch.rand(8, 1, 16, 16)}]
    before = model.network.weight.detach().clone()
    metrics = train_one_epoch(model, batches, optimizer, device=torch.device("cpu"))
    assert metrics["samples"] == 8
    assert metrics["steps"] == 1
    assert metrics["loss"] > 0
    assert not torch.equal(before, model.network.weight)


def test_checkpoint_round_trip_is_strict_and_preserves_optimizer(tmp_path: Path) -> None:
    config = _config()
    model = TinySegmenter()
    optimizer = build_optimizer(model, config)
    train_one_epoch(
        model,
        [{"image": torch.rand(8, 3, 4, 4), "mask": torch.rand(8, 1, 4, 4)}],
        optimizer,
        device=torch.device("cpu"),
    )
    path = tmp_path / "checkpoint.pt"
    digest = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=1,
        config=config,
        resolved_config=OmegaConf.to_yaml(OmegaConf.create({"mode": "approximate"})),
        architecture="tiny-test-segmenter",
    )
    assert len(digest) == 64

    restored = TinySegmenter()
    restored_optimizer = build_optimizer(restored, config)
    payload = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        expected_architecture="tiny-test-segmenter",
    )
    assert payload["result_class"] == "smoke-test result"
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, restored.state_dict()[name])
    assert optimizer.state_dict()["param_groups"] == restored_optimizer.state_dict()["param_groups"]
    assert optimizer.state_dict()["state"].keys() == restored_optimizer.state_dict()["state"].keys()


def test_checkpoint_rejects_architecture_mismatch(tmp_path: Path) -> None:
    model = TinySegmenter()
    optimizer = build_optimizer(model, _config())
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
        config=_config(),
        resolved_config="mode: approximate\n",
        architecture="one",
    )
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(path, model=TinySegmenter(), expected_architecture="two")


def test_checkpoint_round_trip_uses_real_approximate_ag_unet_schema(tmp_path: Path) -> None:
    model = ApproximateAGUNet(encoder_weights=None, freeze_encoder=False)
    optimizer = build_optimizer(model, _config())
    expected_head = model.head.weight.detach().clone()
    path = tmp_path / "approximate-ag-unet.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=0,
        global_step=0,
        config=_config(),
        resolved_config="reproduction_mode: approximate\n",
        architecture=ARCHITECTURE_IDENTITY,
    )
    with torch.no_grad():
        model.head.weight.zero_()
    load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_architecture=ARCHITECTURE_IDENTITY,
    )
    assert torch.equal(model.head.weight, expected_head)
