import copy
import json
import sys

import numpy as np
import pytest
import torch
from torch import nn

from models.totaltext_partial_encoder import (
    PracticalTotalTextPartialEncoderAGUNet, build_partial_encoder_optimizer,
)
from models.unet_adv import PracticalTotalTextHighResAGUNet
from scripts.train_totaltext_partial_encoder_50ep import (
    PracticalPartialEncoderTrainingConfig, _config, evaluate, selection_key, training_config,
)
from scripts.train_totaltext_v4_image_frequency_ablation import _config as source_config
from segmentation.training import load_checkpoint, save_checkpoint


def tiny(branch):
    """Exercise the REAL _encode/train methods with inexpensive convolution fixtures."""
    model = PracticalTotalTextPartialEncoderAGUNet.__new__(PracticalTotalTextPartialEncoderAGUNet)
    nn.Module.__init__(model)
    model.branch = branch
    model.encoder = nn.Module()
    model.encoder.conv1 = nn.Conv2d(3, 4, 1)
    model.encoder.bn1 = nn.BatchNorm2d(4)
    model.encoder.relu = nn.ReLU()
    model.encoder.maxpool = nn.Identity()
    for name in ("layer1", "layer2", "layer3", "layer4"):
        setattr(model.encoder, name, nn.Sequential(nn.Conv2d(4, 4, 1), nn.BatchNorm2d(4), nn.ReLU()))
    model.encoder.requires_grad_(False)
    if branch == "partial_encoder":
        model.encoder.layer3.requires_grad_(True)
        model.encoder.layer4.requires_grad_(True)
    for module in model.encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(False)
    model.head = nn.Conv2d(4, 1, 1)
    return model


def test_only_late_encoder_convolutions_receive_gradients_and_bn_stays_fixed():
    model = tiny("partial_encoder").train()
    before = {name: tensor.clone() for name, tensor in model.encoder.named_buffers()}
    features = model._encode(torch.randn(2, 3, 8, 8))
    assert not features[2].requires_grad and features[3].requires_grad and features[4].requires_grad
    model.head(features[-1]).square().mean().backward()
    assert model.encoder.layer3[0].weight.grad is not None
    assert model.encoder.layer4[0].weight.grad is not None
    assert model.encoder.layer2[0].weight.grad is None
    for module in model.encoder.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert not module.training
            assert all(not p.requires_grad and p.grad is None for p in module.parameters())
    assert all(torch.equal(before[n], t) for n, t in model.encoder.named_buffers())


def test_control_remains_frozen_and_optimizer_has_disjoint_differential_groups():
    config = training_config(_config(), source_config())
    for branch in ("frozen_control", "partial_encoder"):
        model = tiny(branch).train()
        optimizer = build_partial_encoder_optimizer(model, config)
        features = model._encode(torch.randn(2, 3, 8, 8))
        assert features[-1].requires_grad == (branch == "partial_encoder")
        model.head(features[-1]).mean().backward()
        assert model.head.weight.grad is not None
        params = [p for group in optimizer.param_groups for p in group["params"]]
        assert len({id(p) for p in params}) == len(params)
        assert {id(p) for p in params} == {id(p) for p in model.parameters() if p.requires_grad}
        assert optimizer.param_groups[0]["lr"] == 1e-5
        if branch == "partial_encoder":
            assert optimizer.param_groups[1]["lr"] == 1e-6
        else:
            assert len(optimizer.param_groups) == 1


def test_full_architecture_parameter_keys_and_initial_outputs_preserve_baseline():
    torch.manual_seed(71)
    original = PracticalTotalTextHighResAGUNet(skip_floor=0.25, encoder_pretrained=False).eval()
    base_state = copy.deepcopy(original.state_dict())
    x = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        expected = original(x)
    assert expected.shape == (1, 1, 32, 32)
    baseline_count = sum(p.numel() for p in original.parameters())
    del original
    for branch in ("frozen_control", "partial_encoder"):
        model = PracticalTotalTextPartialEncoderAGUNet(branch=branch).eval()
        model.load_state_dict(base_state, strict=True)
        assert sum(p.numel() for p in model.parameters()) == baseline_count == 58155081
        with torch.inference_mode():
            assert torch.equal(model(x), expected)
        if branch == "partial_encoder":
            trainable_names = [name for name, p in model.encoder.named_parameters() if p.requires_grad]
            assert trainable_names and all(name.startswith(("layer3.", "layer4.")) for name in trainable_names)
        else:
            assert not any(p.requires_grad for p in model.encoder.parameters())
        del model


def test_unknown_branch_fails():
    with pytest.raises(ValueError, match="Unknown"):
        PracticalTotalTextPartialEncoderAGUNet(branch="test")


def test_native_validation_metrics_threshold_and_validity():
    class Fixed(nn.Module):
        def forward(self, x):
            return x[:, :1]
    owner = np.zeros((256, 256), np.int32)
    owner[2:6, 2:6] = 1
    logits = torch.full((1, 3, 256, 256), -8.0)
    logits[0, 0, 2:6, 2:6] = 8.0
    batch = {"image": logits, "mask": torch.from_numpy((owner > 0)[None, None].astype(np.float32)),
             "validity": torch.ones(1, 1, 256, 256), "sample_id": ["train:1"]}
    class Loader(list):
        dataset = [1]
    metrics = evaluate(Fixed(), Loader([batch]), {"train:1": (owner, (1,), np.ones((256, 256), bool))},
                       device=torch.device("cpu"), threshold=0.65)
    assert metrics["foreground_iou"] == 1
    assert metrics["matched_gt_iou_ge_50"] == 1
    assert metrics["proposal_f1_iou_50"] == 1
    assert metrics["gt_words_touched_by_fused_components"] == 0


def test_frozen_config_and_selection_ties():
    config = _config()
    assert config["additional_epochs"] == 50 and config["fork_checkpoint_epoch"] == 71
    assert config["split"] == "frozen_TRAIN_1130_validation_125"
    tc = training_config(config, source_config())
    assert tc.epochs == 50 and tc.learning_rate == 1e-5
    metric = dict(proposal_f1_iou_50=0.5, gt_words_touched_by_fused_components=3,
                  word_recall_iou_50=0.6, foreground_iou=0.7)
    assert selection_key({**metric, "gt_words_touched_by_fused_components": 2}) > selection_key(metric)
    assert not selection_key(metric) > selection_key(metric)  # earliest wins


def test_practical_learning_rate_does_not_relax_author_config():
    from dataclasses import asdict
    from segmentation.training import SegmentationTrainingConfig
    tc = training_config(_config(), source_config())
    with pytest.raises(ValueError, match="paper values"):
        SegmentationTrainingConfig(**asdict(tc))
    fields = asdict(tc)
    fields["learning_rate"] = 1e-4
    with pytest.raises(ValueError, match="schedule"):
        PracticalPartialEncoderTrainingConfig(**fields)


def test_strict_checkpoint_optimizer_rng_roundtrip(tmp_path):
    config = training_config(_config(), source_config())
    model = tiny("partial_encoder").train()
    optimizer = build_partial_encoder_optimizer(model, config)
    model.head(model._encode(torch.randn(2, 3, 8, 8))[-1]).square().mean().backward()
    optimizer.step()
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, epoch=2, global_step=4, config=config,
        resolved_config="{}", architecture=model.architecture_identity,
        result_class="approximate reproduction", run_contract={"branch": "partial_encoder"},
        train_metrics={"loss": 0.5}, validation_metrics={"foreground_iou": 0.7})
    restored = tiny("partial_encoder")
    restored_optimizer = build_partial_encoder_optimizer(restored, config)
    payload = load_checkpoint(path, model=restored, optimizer=restored_optimizer,
        expected_architecture=restored.architecture_identity,
        expected_run_contract={"branch": "partial_encoder"}, expected_training_config=config, restore_rng=True)
    assert payload["epoch"] == 2 and len(restored_optimizer.state) == len(optimizer.state)
    assert all(torch.equal(model.state_dict()[key], tensor) for key, tensor in restored.state_dict().items())
    assert [group["lr"] for group in restored_optimizer.param_groups] == [1e-5, 1e-6]
    with pytest.raises(ValueError, match="architecture"):
        load_checkpoint(path, model=tiny("frozen_control"), expected_architecture=tiny("frozen_control").architecture_identity)


def test_evaluator_native_mask_and_ocr_cache_fail_closed():
    from scripts.evaluate_totaltext_partial_encoder_parseq_val import native_prediction, _ocr_text
    from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
    logits = torch.full((1, 1, 256, 256), torch.logit(torch.tensor(0.65)))
    canvas = np.ones((256, 256), np.uint8)
    canvas[0] = 0
    mask = native_prediction(logits, canvas)
    assert mask[1:].all() and not mask[0].any()
    record = dict(crop_rgb_sha256="crop", parseq_checkpoint_sha256=PARSEQ_CHECKPOINT_SHA256,
                  parseq_source_archive_sha256=PARSEQ_RUNTIME_ARCHIVE_SHA256, raw_text="word")
    assert _ocr_text(record, "crop") == "word"
    with pytest.raises(ValueError, match="cache mismatch"):
        _ocr_text({**record, "crop_rgb_sha256": "different"}, "crop")


def test_summary_pairs_by_image_and_counts_gained_lost(tmp_path, monkeypatch):
    from scripts.summarize_totaltext_partial_encoder import main
    from scripts.train_totaltext_v4_highres_ablation import _json
    for branch in ("frozen_control", "partial_encoder"):
        protocol = dict(branch=branch, validation_ids_sha256="same", baseline_report_sha256="same",
                        parseq_checkpoint_sha256="same", segmentation="same", localization="same",
                        matching="same", device="cpu", parseq_batch_size=32)
        rows = []
        for index in range(125):
            ids = [0] if branch == "frozen_control" else [1]
            rows.append(dict(sample_id=str(index), ground_truth=["a", "b"],
                matched_gt_indices={metric: ids for metric in (
                    "geometry_iou_ge_50", "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")}))
        _json(tmp_path / (branch + "-validation-parseq") / "results.json",
              dict(protocol=protocol, aggregate={"gt_words": 983}, per_image=rows, frozen_letterbox_065={}))
    monkeypatch.setattr(sys, "argv", ["summary", "--output-root", str(tmp_path)])
    assert main() == 0
    summary = json.loads((tmp_path / "comparison.json").read_text())
    assert summary["partial_vs_control"]["geometry_iou_ge_50"] == dict(gained=125, lost=125, net=0)
