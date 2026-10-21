"""CPU fixtures for instance adjacency, paired crops, losses and provenance."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from models.totaltext_word_ablation import BRANCHES, PracticalTotalTextWordAGUNet
from models.unet_adv import PracticalTotalTextHighResAGUNet
from scripts.train_totaltext_v4_image_frequency_ablation import _config as source_config, _transform, proposal_metrics
from scripts.train_totaltext_word_ablation_50ep import (
    _config, labels_metrics, load_population, reconcile_epoch_records, training_config,
)
from segmentation.totaltext_word_ablation import (
    OFFSETS, REVERSE, WordTrainingTransform, ablation_loss, balanced_link_loss,
    choose_word_crop, crop_ignore_mask, deterministic_bilinear_resize, label_boxes, link_targets, owner_weights,
    pixel_link_labels, word_balanced_loss,
)
from segmentation.training import SegmentationTrainingConfig, load_checkpoint, save_checkpoint


def maps():
    owner = np.zeros((8, 12), np.int32)
    owner[2:6, 1:6] = 1
    owner[2:6, 6:11] = 2
    target = torch.from_numpy((owner > 0).astype(np.float32))[None, None]
    return torch.from_numpy(owner)[None, None], target, torch.ones_like(target)


def test_links_split_touching_words_using_only_predicted_neighbors():
    owners, target, valid = maps()
    links, allowed = link_targets(owners, target, valid)
    foreground = target[0, 0].numpy() > 0
    assert cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)[0] == 2
    labels = pixel_link_labels(foreground, links[0].numpy(), 0.5)
    assert len(label_boxes(labels)) == 2
    assert label_boxes(labels)[0]["bbox_native"] == [1, 2, 6, 6]
    assert label_boxes(labels)[1]["bbox_native"] == [6, 2, 11, 6]
    assert allowed[0, 4, 3, 5] == 1 and links[0, 4, 3, 5] == 0


def test_link_inference_requires_reciprocal_edges_and_preserves_singletons():
    fg = np.zeros((3, 4), bool); fg[1, 1:3] = True
    p = np.zeros((8, 3, 4), np.float32)
    p[4, 1, 1] = 0.9
    assert len(label_boxes(pixel_link_labels(fg, p, 0.5))) == 2
    p[REVERSE[4], 1, 2] = 0.5
    assert len(label_boxes(pixel_link_labels(fg, p, 0.5))) == 1
    assert not pixel_link_labels(np.zeros_like(fg), p, 0.5).any()


def test_border_directions_do_not_wrap_and_invalid_endpoints_have_zero_gradient():
    owners, target, valid = maps()
    valid[:, :, 3, 4] = 0
    labels, weights = link_targets(owners, target, valid)
    assert not weights[:, :, 3, 4].any()
    assert weights[0, 4, 3, 3] == 0
    assert not weights[0, 0, 0].any()
    logits = torch.zeros_like(labels, requires_grad=True)
    balanced_link_loss(logits, labels, weights).backward()
    assert not logits.grad[weights == 0].any()
    assert torch.isfinite(logits.grad).all()


def test_link_target_requires_valid_owner_and_paired_shapes():
    owners, target, valid = maps()
    owners[0, 0, 3, 3] = 0
    with pytest.raises(ValueError, match="no word owner"):
        link_targets(owners, target, valid)
    with pytest.raises(ValueError, match="paired"):
        link_targets(owners, target[:, :, :-1], valid)


def test_owner_weights_give_small_and_large_word_equal_total_weight():
    owner = np.zeros((8, 8), np.int32)
    owner[0, 0] = 1; owner[2:6, 2:6] = 2
    weight = owner_weights(owner, owner > 0, np.ones_like(owner))
    assert weight[owner == 1].sum() == pytest.approx(0.5)
    assert weight[owner == 2].sum() == pytest.approx(0.5)
    assert not weight[owner == 0].any()


def test_word_balanced_loss_masks_invalid_pixels_and_keeps_background_gradient():
    owners, target, valid = maps()
    valid[0, 0, 0, 0] = 0
    weight = torch.from_numpy(owner_weights(owners[0, 0], target[0, 0], valid[0, 0]))[None, None]
    logits = torch.zeros_like(target, requires_grad=True)
    loss = word_balanced_loss(logits, target, valid, torch.zeros_like(target), weight,
                              negative_ratio=3, separation_weight=3)
    assert loss.item() == pytest.approx(np.log(2), abs=1e-6)
    loss.backward()
    assert logits.grad[0, 0, 0, 0] == 0
    assert (logits.grad[target > 0] < 0).all()
    assert (logits.grad[target == 0] > 0).any()


def test_empty_foreground_and_invalid_only_word_loss():
    target = torch.zeros(1, 1, 3, 3)
    logits = torch.zeros_like(target, requires_grad=True)
    value = word_balanced_loss(logits, target, torch.ones_like(target), target, target,
                               negative_ratio=3, separation_weight=3)
    value.backward()
    assert (logits.grad > 0).all()
    with pytest.raises(ValueError, match="no valid"):
        word_balanced_loss(logits, target, target, target, target, negative_ratio=3, separation_weight=3)


def test_deterministic_bilinear_matches_reference_outputs_and_gradients():
    a = torch.randn(2, 1, 7, 9, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    resized = deterministic_bilinear_resize(a, (20, 27))
    reference = torch.nn.functional.interpolate(b, size=(20, 27), mode="bilinear", align_corners=False)
    assert torch.allclose(resized, reference, atol=3e-6, rtol=1e-5)
    resized.square().mean().backward(); reference.square().mean().backward()
    assert torch.allclose(a.grad, b.grad, atol=1e-6, rtol=1e-5)


def test_crop_anchor_is_whole_and_partial_other_words_are_identified():
    owner = np.zeros((40, 60), np.int32)
    owner[5:15, 5:15] = 1; owner[20:30, 25:55] = 2
    for seed in range(20):
        box = choose_word_crop(owner, torch.Generator().manual_seed(seed), (0.5, 0.75))
        x1, y1, x2, y2 = box
        piece = owner[y1:y2, x1:x2]
        assert any(np.count_nonzero(piece == k) == np.count_nonzero(owner == k) for k in (1, 2))
    ignored = crop_ignore_mask(owner, (0, 0, 35, 35))
    assert not ignored[5:15, 5:15].any()
    assert ignored[20:30, 25:35].all()
    assert choose_word_crop(np.zeros_like(owner), torch.Generator(), (0.5,)) is None


def test_nonzoom_transform_is_bit_identical_to_existing_qualified_transform(tmp_path):
    owner = np.zeros((40, 60), np.int32); owner[8:22, 8:24] = 1; owner[8:22, 28:40] = 2
    path = tmp_path / "owners.png"; Image.fromarray(owner, mode="I").save(path)
    image = Image.fromarray(np.full((40, 60, 3), 100, np.uint8))
    mask = Image.fromarray((owner > 0).astype(np.uint8) * 255)
    valid = Image.new("L", image.size, 255); separator = Image.new("L", image.size, 0)
    base = _transform(source_config(), "letterbox", train=True); base.set_epoch(9)
    paired = WordTrainingTransform(base, owner_paths=[path], zoom=False, zoom_probability=0.5, zoom_fractions=[0.5, 0.75])
    paired.set_epoch(9)
    expected = base.transform_with_validity_and_separation(image, mask, valid, separator, Image.fromarray(owner, mode="I"), 0)
    actual = paired.transform_with_validity_and_separation(image, mask, valid, separator, Image.fromarray(owner, mode="I"), 0)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    transformed = paired.transform_owners(Image.fromarray(owner, mode="I"), 0, actual[1], actual[2])
    assert not transformed[(actual[1] == 0) | (actual[2] == 0)].any()


def test_zoom_replays_same_crop_and_excludes_partially_cut_words(tmp_path):
    owner = np.zeros((20, 40), np.int32); owner[3:13, 5:15] = 1; owner[3:13, 22:38] = 2
    path = tmp_path / "owner.png"; Image.fromarray(owner, mode="I").save(path)
    base = _transform(source_config(), "letterbox", train=False)
    transform = WordTrainingTransform(base, owner_paths=[path], zoom=True, zoom_probability=1, zoom_fractions=[0.5])
    bbox = (0, 0, 30, 20)
    transform._crop_key = (0, 0); transform._crop_value = (bbox, crop_ignore_mask(owner, bbox))
    image = Image.new("RGB", (40, 20), (255, 255, 255))
    mask = Image.fromarray((owner > 0).astype(np.uint8) * 255)
    validity = Image.new("L", image.size, 255); separator = Image.new("L", image.size, 0)
    args = (image, mask, validity, separator, Image.fromarray(owner, mode="I"), 0)
    a = transform.transform_with_validity_and_separation(*args)
    b = transform.transform_with_validity_and_separation(*args)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    labels = transform.transform_owners(args[4], 0, a[1], a[2])
    assert set(torch.unique(labels).tolist()) == {0, 1}
    assert a[0].shape == (3, 512, 512) and a[1].shape == (1, 256, 256)


def test_label_metrics_equal_existing_baseline_and_preserve_instance_cuts():
    owners = np.zeros((256, 256), np.int32)
    owners[10:20, 10:20] = 1; owners[10:20, 20:30] = 2
    raw = cv2.connectedComponents((owners > 0).astype(np.uint8), connectivity=8)[1]
    assert labels_metrics(raw, owners, (1, 2)) == proposal_metrics(owners > 0, owners, expected_owner_ids=(1, 2))
    cut = labels_metrics(owners, owners, (1, 2))
    assert cut["matched_gt_iou_ge_50"] == 2 and cut["gt_words_touched_by_fused_components"] == 0
    assert labels_metrics(raw, owners, (1, 2))["gt_words_touched_by_fused_components"] == 2


@pytest.mark.parametrize("branch", BRANCHES)
def test_all_losses_backpropagate_without_invalid_pixel_gradient(branch):
    owners, target, valid = maps(); valid[0, 0, 0, 0] = 0
    foreground = torch.zeros_like(target, requires_grad=True)
    links = torch.zeros(1, 8, 8, 12, requires_grad=True)
    auxiliary = [torch.zeros(1, 1, 4, 6, requires_grad=True), torch.zeros(1, 1, 2, 3, requires_grad=True)]
    weights = torch.from_numpy(owner_weights(owners[0, 0], target[0, 0], valid[0, 0]))[None, None]
    evidence = {"foreground": foreground, "links": links, "auxiliary": auxiliary}
    batch = {"mask": target, "validity": valid, "separation": torch.zeros_like(target), "owners": owners, "word_weights": weights}
    value = ablation_loss(evidence, batch, branch, _config())
    value.backward()
    assert torch.isfinite(value) and foreground.grad[0, 0, 0, 0] == 0
    if branch == "pixel_link":
        assert links.grad is not None and torch.isfinite(links.grad).all()
    if branch == "deep_supervision":
        assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in auxiliary)


def test_full_models_preserve_base_output_and_have_declared_parameter_counts():
    torch.manual_seed(7)
    base = PracticalTotalTextHighResAGUNet(skip_floor=0.25, encoder_pretrained=False).eval()
    state = base.state_dict()
    x = torch.randn(1, 3, 64, 64)
    with torch.inference_mode(): expected = base(x)
    del base
    for branch in BRANCHES:
        model = PracticalTotalTextWordAGUNet(branch, link_initial_bias=2).eval()
        model.initialize_base_state(state)
        assert sum(p.numel() for p in model.parameters()) == 58155081 + (520 if branch == "pixel_link" else 386 if branch == "deep_supervision" else 0)
        assert not any(p.requires_grad for p in model.encoder.parameters())
        with torch.inference_mode():
            output = model.forward_evidence(x)
        assert torch.equal(output["foreground"], expected)
        assert output["auxiliary"] == ()
        if branch == "pixel_link":
            assert output["links"].shape == (1, 8, 32, 32)
            assert torch.equal(output["links"], torch.full_like(output["links"], 2.0))
        if branch == "deep_supervision":
            model.train()
            with torch.no_grad(): evidence = model.forward_evidence(x)
            assert [list(a.shape[-2:]) for a in evidence["auxiliary"]] == [[16, 16], [8, 8]]
            assert not model.encoder.training
        with pytest.raises(ValueError, match="checkpoint keys"):
            model.initialize_base_state({**state, "unexpected": torch.ones(1)})
        del model


def test_train_population_rejects_test_before_splitting(tmp_path, monkeypatch):
    import scripts.train_totaltext_word_ablation_50ep as trainer
    path = tmp_path / "manifest.yaml"; path.write_text("fixture")
    config = _config(); config["train_manifest"] = "manifest.yaml"; config["train_manifest_sha256"] = trainer.sha256_file(path)
    monkeypatch.setattr(trainer, "load_segmentation_manifest", lambda *a, **k: SimpleNamespace(dataset="total_text", split_role="test"))
    with pytest.raises(ValueError, match="TRAIN"):
        load_population(tmp_path, config)


def test_new_loss_config_does_not_relax_shared_paper_contract():
    from dataclasses import asdict
    for branch in BRANCHES:
        config = training_config(_config(), source_config(), branch)
        assert config.epochs == 50 and config.loss == branch
        with pytest.raises(ValueError, match="paper values"):
            SegmentationTrainingConfig(**asdict(config))


def test_strict_checkpoint_roundtrip_with_branch_and_optimizer(tmp_path):
    class Small(nn.Module):
        architecture_identity = "word-test"
        def __init__(self):
            super().__init__(); self.head = nn.Conv2d(1, 1, 1)
    model = Small(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    config = training_config(_config(), source_config(), "word_balanced")
    path = tmp_path / "last.pt"
    save_checkpoint(path, model=model, optimizer=optimizer, epoch=2, global_step=4,
        config=config, resolved_config="{}", architecture=model.architecture_identity,
        result_class="approximate reproduction", run_contract={"branch": "word_balanced"})
    fresh = Small()
    load_checkpoint(path, model=fresh, expected_architecture=fresh.architecture_identity,
        expected_run_contract={"branch": "word_balanced"}, expected_training_config=config)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), fresh.parameters()))
    with pytest.raises(ValueError, match="contract"):
        load_checkpoint(path, model=fresh, expected_architecture=fresh.architecture_identity,
            expected_run_contract={"branch": "pixel_link"})


def test_interruption_after_checkpoint_before_log_is_recovered_once():
    payload = {"epoch": 2, "global_step": 20, "train_metrics": {"loss": 1}, "validation_metrics": {"score": 2}}
    records = [{"epoch": 1, "last_checkpoint_sha256": "before"}]
    recovered = reconcile_epoch_records(records, payload, "last", 50)
    assert recovered[-1]["epoch"] == 2 and recovered[-1]["recovered_from_checkpoint"] is True
    assert records == [{"epoch": 1, "last_checkpoint_sha256": "before"}]
    assert reconcile_epoch_records(recovered, payload, "last", 50) == recovered
    with pytest.raises(ValueError, match="mismatch"):
        reconcile_epoch_records(recovered, payload, "changed", 50)
    with pytest.raises(ValueError, match="prefix"):
        reconcile_epoch_records([{"epoch": 2}], payload, "last", 50)
    with pytest.raises(ValueError, match="mismatch"):
        reconcile_epoch_records([], payload, "last", 50)


def test_paired_summary_rejects_changed_gt_and_counts_recovered_lost():
    from scripts.summarize_totaltext_word_ablation import paired_changes
    before = [{"sample_id": str(i), "ground_truth": ["one", "two"],
               "matched_gt_indices": {"exact": [0]}} for i in range(125)]
    after = [{**row, "matched_gt_indices": {"exact": [1]}} for row in before]
    assert paired_changes(before, after, "exact") == {"gained": 125, "lost": 125, "net": 0}
    after[0]["ground_truth"] = ["changed"]
    with pytest.raises(ValueError, match="GT"):
        paired_changes(before, after, "exact")


def test_ocr_reuse_requires_exact_crop_checkpoint_and_source_hashes():
    from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
    from scripts.evaluate_totaltext_word_ablation_parseq_val import _ocr_text
    record = {"raw_text": "word", "crop_rgb_sha256": "rgb",
              "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256}
    assert _ocr_text(record, "rgb") == "word"
    with pytest.raises(ValueError, match="cache mismatch"):
        _ocr_text({**record, "parseq_checkpoint_sha256": "other"}, "rgb")
