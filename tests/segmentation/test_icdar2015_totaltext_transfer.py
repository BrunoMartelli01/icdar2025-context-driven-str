"""Scientific boundaries for ICDAR2015 targets and the Total-Text U-Net transfer."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from models.icdar2015_unet import PracticalICDAR2015AGUNet
from models.unet_adv import AUTHOR_PROTOCOL_ARCHITECTURE_IDENTITY
from recognition.scene_text_datasets import SceneTextInstance
from scripts.prepare_icdar2015_word_regions import target_topology
from scripts.train_icdar2015_totaltext_transfer import training_config
from segmentation.data import file_sha256
from segmentation.icdar2015_transfer import (
    evaluate, load_config, make_targets, selection_key, split_indices, transform,
)
from segmentation.totaltext_image_frequency_augment import (
    letterbox_canvas_mask, remap_letterbox_native_box,
)


CONFIG = Path(__file__).resolve().parents[2] / "configs/approximate/icdar2015_totaltext_transfer_v1.json"


def word(points, *, ignore=False):
    return SceneTextInstance(tuple(points), "###" if ignore else "word", ignore, "word")


def test_touching_quadrilaterals_are_disjoint_and_ignore_is_excluded():
    instances = (word(((4, 4), (15, 4), (15, 16), (4, 16))),
                 word(((15, 4), (28, 4), (28, 16), (15, 16))),
                 word(((0, 0), (6, 0), (6, 6), (0, 6)), ignore=True))
    target, validity, separator, owners, audit = make_targets((32, 20), instances)
    foreground, valid, sep, owner = map(np.asarray, (target, validity, separator, owners))
    assert np.array_equal(foreground > 0, owner > 0)
    assert not np.any((foreground > 0) & ((valid == 0) | (sep > 0)))
    assert cv2.connectedComponents((foreground > 0).astype(np.uint8), connectivity=8)[0] == 3
    assert audit.word_count == 2 and audit.empty_word_count == 0
    assert audit.contested_pixels > 0 and audit.removed_separator_pixels > 0


def test_ignore_only_image_is_retained_as_negative_example():
    target, validity, separator, owners, audit = make_targets((20, 10), (
        word(((2, 2), (12, 2), (12, 7), (2, 7)), ignore=True),))
    assert not np.asarray(target).any() and not np.asarray(separator).any()
    assert not np.asarray(owners).any() and audit.word_count == 0
    assert np.asarray(validity)[3, 3] == 0 and np.asarray(validity)[0, 0] == 255


@pytest.mark.parametrize("augment", [False, True])
def test_native_targets_keep_paired_owners_separation_validity_and_canvas(augment):
    config = load_config(CONFIG)
    paired = transform(config, augment=augment)
    paired.set_epoch(7)
    image = Image.new("RGB", (320, 100), (124, 116, 104))
    target, valid, sep, owner, _ = make_targets(image.size, (
        word(((30, 20), (100, 20), (100, 70), (30, 70))),
        word(((100, 20), (160, 20), (160, 70), (100, 70))),))
    inputs = (image, target, valid, sep, owner.convert("I"), 11)
    first = paired.transform_with_validity_and_separation(*inputs)
    second = paired.transform_with_validity_and_separation(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    rgb, target_t, valid_t, sep_t = first
    assert rgb.shape == (3, 512, 512) and target_t.shape == (1, 256, 256)
    assert not torch.any(target_t.bool() & (sep_t.bool() | ~valid_t.bool()))
    assert not valid_t[:, :70].any() and not valid_t[:, 185:].any()
    if not augment:
        assert torch.equal(valid_t[0].bool(), letterbox_canvas_mask(image.size))


def test_inverse_letterbox_handles_wide_icdar_images_and_exclusive_edges():
    assert remap_letterbox_native_box([20, 61, 40, 81], (1280, 720)) == [100, 25, 200, 125]
    with pytest.raises(ValueError, match="does not intersect"):
        remap_letterbox_native_box([0, 0, 10, 10], (1280, 720))


def test_approximate_config_is_explicit_and_strict_cannot_inherit_it(tmp_path):
    config = load_config(CONFIG)
    assert config["reproduction_mode"] == "approximate"
    assert training_config(config).separation_weight == 3.0
    strict = tmp_path / "strict"
    strict.mkdir()
    candidate = strict / "transfer.json"
    candidate.write_text(CONFIG.read_text())
    with pytest.raises(ValueError, match="approximate"):
        load_config(candidate)
    approx = tmp_path / "approximate"
    approx.mkdir()
    candidate = approx / "transfer.json"
    changed = deepcopy(config)
    changed["reproduction_mode"] = "strict"
    candidate.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="reproduction_mode"):
        load_config(candidate)


def fake_manifest():
    return SimpleNamespace(dataset="icdar2015", split_role="train", samples=tuple(
        SimpleNamespace(sample_id=f"icdar2015:{i}", image_path=f"images/{i}.jpg",
                        image_sha256=f"{i:064x}") for i in range(1000)))


def test_heldout_split_is_deterministic_disjoint_and_rejects_test_and_duplicates():
    config, manifest = load_config(CONFIG), fake_manifest()
    train, val = split_indices(manifest, config)
    assert (train, val) == split_indices(manifest, config)
    assert len(train) == 900 and len(val) == 100 and not set(train) & set(val)
    manifest.split_role = "test"
    with pytest.raises(ValueError, match="TRAIN"):
        split_indices(manifest, config)
    manifest.split_role = "train"
    manifest.samples[val[0]].image_sha256 = manifest.samples[train[0]].image_sha256
    with pytest.raises(ValueError, match="leakage"):
        split_indices(manifest, config)


def test_word_selection_prefers_f1_then_less_fusion_and_earliest_epoch():
    metrics = {"proposal_f1_iou_50": 0.6, "gt_words_touched_by_fused_components": 2,
               "word_recall_iou_50": 0.7, "foreground_iou": 0.8}
    assert selection_key(metrics, 1) > selection_key(metrics, 2)
    assert selection_key({**metrics, "gt_words_touched_by_fused_components": 1}, 2) > selection_key(metrics, 1)
    assert selection_key({**metrics, "proposal_f1_iou_50": 0.7, "foreground_iou": 0.1}, 2) > selection_key(metrics, 1)


def test_lost_words_stay_in_denominator_and_incomplete_validation_is_an_error(tmp_path):
    mask = np.zeros((256, 256), dtype=bool)
    mask[10:20, 10:20] = True
    owners = mask.astype(np.int32)
    references = {"a": (owners, (1, 2), np.ones_like(mask))}
    batch = {"image": torch.zeros(1, 3, 512, 512), "sample_id": ["a"],
             "mask": torch.from_numpy(mask[None, None].copy()),
             "validity": torch.ones(1, 1, 256, 256)}

    class FixedModel(torch.nn.Module):
        def forward(self, image):
            return torch.from_numpy(np.where(mask, 20., -20.).astype(np.float32))[None, None]

    report = evaluate(FixedModel(), [batch], references, device=torch.device("cpu"),
                      probability_threshold=0.5, output_dir=tmp_path)
    result = report["aggregate"]
    assert result["gt_words"] == 2 and result["gt_words_lost_in_native_target"] == 1
    assert result["matched_gt_iou_ge_50"] == 1 and result["word_recall_iou_50"] == 0.5
    assert result["foreground_iou"] == 1 and result["proposal_f1_iou_50"] == pytest.approx(2 / 3)
    assert (tmp_path / "a.npz").is_file()
    assert report["per_image"][0]["intermediate_sha256"] == file_sha256(tmp_path / "a.npz")
    with pytest.raises(ValueError, match="incomplete"):
        evaluate(FixedModel(), [batch], {**references, "b": references["a"]}, device=torch.device("cpu"),
                 probability_threshold=0.5)
    audit = target_topology(mask, owners, np.ones_like(mask), 2)
    assert audit["lost_words"] == 1 and audit["care_words"] == 2


def test_transfer_model_has_distinct_identity_single_output_and_frozen_encoder():
    model = PracticalICDAR2015AGUNet(encoder_pretrained=False).train()
    assert model.architecture_identity != AUTHOR_PROTOCOL_ARCHITECTURE_IDENTITY
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(not module.training for module in model.encoder.modules())
    assert all(stage.att.practical_skip_floor == 0.25 for stage in (
        model.up0, model.up1, model.up2, model.up3))
    with torch.inference_mode():
        logits = model.eval()(torch.zeros(1, 3, 64, 64))
    assert logits.shape == (1, 1, 32, 32)
    assert sum(p.numel() for p in model.parameters()) == 58_155_081


def test_oom_is_saved_and_propagated_without_substitution(tmp_path, monkeypatch):
    import sys
    from scripts import train_icdar2015_totaltext_transfer as runner

    def fail(_):
        raise torch.cuda.OutOfMemoryError("Injected allocator failure")

    monkeypatch.setattr(runner, "run", fail)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(CONFIG), "--data-root", str(tmp_path),
        "--encoder-checkpoint", str(tmp_path / "encoder.pt"), "--output-dir", str(tmp_path), "--device", "cuda:0"])
    with pytest.raises(torch.cuda.OutOfMemoryError, match="Injected allocator failure"):
        runner.main()
    report = json.loads((tmp_path / "failure.json").read_text())
    assert report["oom"] is True and report["status"] == "FAILED"
    assert report["cuda_peak_allocated_bytes"] == 123 and report["cuda_peak_reserved_bytes"] == 456
