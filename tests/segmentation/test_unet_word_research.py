"""Scientific boundaries of paired glyph/word supervision and new heads."""

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import PurePosixPath
import json

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from localize.unet_word_research import embedding_instances, glyph_word_instances, limit_instances, mask_stability
from models.unet_word_research import MaskQualityHead, WordResearchAGUNet
from scripts.diagnose_unet_mask_stability import binary_auc
from scripts.run_unet_word_research import DEFAULT_CONFIG, checkpoint_key, read_config, training_config
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.data import SegmentationDataset, SegmentationManifest, SegmentationSample, file_sha256
from segmentation.totaltext_word_kernel import make_word_kernels
from segmentation.unet_word_research import (
    GlyphWordDataset, adaptive_threshold_targets, instance_embedding_loss, mask_quality_targets, research_loss,
)


@pytest.fixture
def cfg():
    return read_config(DEFAULT_CONFIG, "approximate")[0]


def test_strict_rejects_new_experiment():
    with pytest.raises(ValueError, match="APPROXIMATE"):
        read_config(DEFAULT_CONFIG, "strict")


def test_actual_training_values_are_preserved_in_checkpoint_metadata(cfg):
    cfg["batch_size"] = 2
    for branch in cfg["branches"]:
        tc = training_config(cfg, branch)
        assert asdict(tc)["learning_rate"] == 1e-5
        assert asdict(tc)["batch_size"] == 2
        assert tc.checkpoint_selection == ("validation_quality_loss" if branch == "quality" else "validation_word_f1")


def test_quality_selection_and_earliest_tie(cfg):
    assert checkpoint_key({"quality_loss": .1}, 1, "quality") > checkpoint_key({"quality_loss": .2}, 0, "quality")
    assert checkpoint_key({"quality_loss": .1}, 0, "quality") > checkpoint_key({"quality_loss": .1}, 1, "quality")


def test_paired_dataset_replays_flip_rotation_letterbox_and_reorders_source(tmp_path):
    _, source = read_config(DEFAULT_CONFIG, "approximate")
    source["training"].update(horizontal_flip_probability=1., rotation_probability=1.)
    transform = _transform(source, "letterbox", train=True)
    transform.set_epoch(3)
    roots = [tmp_path / "words", tmp_path / "glyphs"]
    samples = [[], []]
    for root in roots: root.mkdir()
    for index in range(2):
        rgb = np.zeros((48, 80, 3), np.uint8); rgb[4:40, 10:70] = [80 + index, 100, 120]
        owner = np.zeros((48, 80), np.uint16); owner[10:30, 8:33] = 1; owner[10:30, 38:70] = 2
        glyph = np.zeros((48, 80), np.uint8); glyph[12:28, 12:16] = 255; glyph[12:28, 44:48] = 255
        for j, root in enumerate(roots):
            Image.fromarray(rgb).save(root / f"{index}.png")
            Image.fromarray((owner > 0).astype(np.uint8) * 255 if j == 0 else glyph).save(root / f"{index}-mask.png")
            Image.fromarray(np.full((48, 80), 255, np.uint8)).save(root / f"{index}-valid.png")
            if j == 0:
                Image.fromarray(owner).save(root / f"{index}-owners.png")
                Image.fromarray(np.zeros((48, 80), np.uint8)).save(root / f"{index}-sep.png")
            sample = SegmentationSample(f"total_text:{index}", PurePosixPath(f"{index}.png"), PurePosixPath(f"{index}-mask.png"),
                file_sha256(root / f"{index}.png"), file_sha256(root / f"{index}-mask.png"),
                validity_annotation_path=PurePosixPath(f"{index}-valid.png"), validity_annotation_sha256=file_sha256(root / f"{index}-valid.png"),
                separation_annotation_path=PurePosixPath(f"{index}-sep.png") if j == 0 else None,
                separation_annotation_sha256=file_sha256(root / f"{index}-sep.png") if j == 0 else None,
                word_owner_annotation_path=PurePosixPath(f"{index}-owners.png") if j == 0 else None,
                word_owner_annotation_sha256=file_sha256(root / f"{index}-owners.png") if j == 0 else None)
            samples[j].append(sample)
    def manifest(items):
        return SegmentationManifest(1, "approximate", "approximate reproduction", "total_text", "train", None,
            "train", None, None, {"target_conversion": "binary-mask-with-validity-v1"}, tuple(items))
    word, original = manifest(samples[0]), manifest(list(reversed(samples[1])))
    base = SegmentationDataset(word, roots[0], transform=transform)
    dataset = GlyphWordDataset(base, original, roots[1], distance_fraction=.5)
    for index in range(2):
        batch = dataset[index]
        assert batch["glyph_mask"].shape == batch["mask"].shape == (1, 256, 256)
        assert batch["canvas"].dtype == torch.bool
        assert batch["glyph_mask"].sum() < batch["mask"].sum()
        assert torch.equal(batch["glyph_mask"], dataset[index]["glyph_mask"])
        assert torch.all(batch["glyph_validity"] <= batch["validity"])
    wrong = replace(original, samples=(replace(original.samples[0], image_sha256="0" * 64), original.samples[1]))
    with pytest.raises(ValueError, match="RGB checksum"):
        GlyphWordDataset(base, wrong, roots[1], distance_fraction=.5)


def embedding_batch():
    owners = torch.zeros(1, 6, 12, dtype=torch.int64)
    owners[:, 1:5, 1:5] = 1; owners[:, 1:5, 7:11] = 2
    kernels, _ = make_word_kernels(owners[0].numpy())
    return owners, torch.tensor(kernels > 0, dtype=torch.float32)[None, None], torch.ones(1, 1, 6, 12)


def test_embedding_loss_distinguishes_same_and_different_words_and_ignores_invalid():
    owners, kernels, valid = embedding_batch()
    correct = torch.zeros(1, 2, 6, 12, requires_grad=True)
    with torch.no_grad(): correct[:, 0, :, 6:] = 5.
    assert instance_embedding_loss(correct, owners, kernels, valid, pull_margin=.5, push_margin=3).item() == 0
    collapsed = torch.zeros_like(correct, requires_grad=True)
    assert instance_embedding_loss(collapsed, owners, kernels, valid, pull_margin=.5, push_margin=3).item() > 0
    torch.manual_seed(1)
    predicted = torch.randn_like(correct, requires_grad=True)
    valid[:, :, 0] = 0
    loss = instance_embedding_loss(predicted, owners, kernels, valid, pull_margin=.5, push_margin=3)
    loss.backward()
    assert torch.isfinite(predicted.grad).all() and not predicted.grad[:, :, 0].any()
    assert predicted.grad[:, :, 1:5].abs().sum() > 0


def test_embedding_empty_singleton_and_missing_core():
    vector = torch.randn(1, 2, 4, 4, requires_grad=True)
    owner = torch.zeros(1, 4, 4, dtype=torch.int64)
    core = torch.zeros(1, 1, 4, 4); valid = torch.ones_like(core)
    loss = instance_embedding_loss(vector, owner, core, valid, pull_margin=.5, push_margin=3)
    loss.backward(); assert not vector.grad.any()
    owner[0, 1, 1] = 1
    with pytest.raises(ValueError, match="no kernel"):
        instance_embedding_loss(vector, owner, core, valid, pull_margin=.5, push_margin=3)
    core[0, 0, 1, 1] = 1
    assert instance_embedding_loss(vector, owner, core, valid, pull_margin=.5, push_margin=3).item() == 0


def test_embedding_grouping_preserves_support_without_crossing_different_word_vectors(cfg):
    p = np.zeros((2, 12, 20), np.float32); p[0, 3:9, 2:18] = 1
    p[1, 5:7, 3:5] = 1; p[1, 5:7, 15:17] = 1
    e = np.zeros((4, 12, 20), np.float32); e[0, :, 10:] = 6
    labels, audit = embedding_instances(p, e, np.ones((12, 20), bool), cfg["postprocessing"])
    assert np.array_equal(labels > 0, p[0] > 0)
    assert labels[5, 4] != labels[5, 16]
    assert len(np.unique(labels[labels > 0])) == 2
    assert audit["unassigned_foreground_pixels"] == 0


def test_embedding_missing_foreground_keeps_unseeded_fallback(cfg):
    p = np.zeros((2, 12, 20), np.float32); p[0, 3:9, 2:6] = 1; p[0, 3:9, 12:17] = 1
    p[1, 5, 3] = 1
    labels, audit = embedding_instances(p, np.zeros((4, 12, 20)), np.ones((12, 20), bool), cfg["postprocessing"])
    assert labels[5, 3] != labels[5, 14] and not labels[:, 6:12].any()
    assert audit["fallback_components"] == 1


def test_glyph_attachment_does_not_join_words_or_erase_base_support():
    labels = np.zeros((10, 20), np.int32); labels[3:7, 2:7] = 1; labels[3:7, 12:17] = 2
    p = np.zeros_like(labels, np.float32); p[3:7, 1:5] = 1
    extended, audit = glyph_word_instances(labels, p, np.ones_like(labels, bool), threshold=.5, minimum_overlap=.7)
    assert extended[3, 1] == 1 and np.all(extended[labels > 0] == labels[labels > 0])
    assert audit["added_pixels"] == 4 and labels[3, 1] == 0
    p[5, 4:15] = 1
    extended, audit = glyph_word_instances(labels, p, np.ones_like(labels, bool), threshold=.5, minimum_overlap=.7)
    assert np.array_equal(extended, labels)
    assert audit["associations"][0]["reason"] == "ambiguous_or_weak_support"


def test_quality_targets_single_word_fragment_merge_and_ignored():
    owner = np.zeros((12, 20), np.int32); owner[2:10, 2:8] = 1; owner[2:10, 12:18] = 2
    valid = np.ones_like(owner, bool)
    ids, target = mask_quality_targets(owner, owner, valid)
    assert ids == [1, 2] and np.array_equal(target, np.ones((2, 3)))
    fragment = owner.copy(); fragment[6:] = 0
    _, target = mask_quality_targets(fragment, owner, valid)
    assert np.allclose(target[:, 1], .5) and np.allclose(target[:, 2], 1)
    merged = (owner > 0).astype(np.int32)
    _, target = mask_quality_targets(merged, owner, valid)
    assert target[0, 0] < .5 and target[0, 2] == .5
    ids, target = mask_quality_targets(owner, owner, np.zeros_like(valid))
    assert ids == [] and target.shape == (0, 3)


def test_quality_forward_no_gt_and_gradient():
    head = MaskQualityHead(channels=4)
    feature = torch.randn(2, 4, 16, 16)
    support = torch.zeros(2, 1, 16, 16); support[:, :, 3:9, 4:10] = 1
    result = head(feature, support)
    assert result.shape == (2, 3)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())
    with pytest.raises(ValueError, match="aligned"):
        head(feature, support[:, :, :5])


def test_top10_area_and_quality_ranking_same_candidate_set():
    labels = np.arange(1, 13, dtype=np.int32).reshape(3, 4)
    area = limit_instances(labels, 10)
    quality = limit_instances(labels, 10, {i: float(i) for i in range(1, 13)})
    assert len(np.unique(area[area > 0])) == len(np.unique(quality[quality > 0])) == 10
    assert area[0, 0] == 1 and quality[0, 0] == 0
    with pytest.raises(ValueError, match="every"):
        limit_instances(labels, 10, {1: 1})


def test_adaptive_targets_are_bounded_and_ignore_invalid(cfg):
    owner, _, valid = embedding_batch(); valid[:, :, 0] = 0
    targets, weights = adaptive_threshold_targets(owner, valid, **{k: cfg["adaptive"][k] for k in ("border_width", "threshold_range")})
    assert targets.min() >= .4 and targets.max() <= .9
    assert weights[:, :, 0].sum() == 0
    assert targets[0, 0, 1, 1] > targets[0, 0, 2, 2]


def test_auxiliary_losses_backpropagate_all_heads(cfg):
    owners, kernel, valid = embedding_batch()
    batch = {"mask": (owners > 0).float()[:, None], "kernel": kernel, "validity": valid,
             "owners_native": owners, "separation": torch.zeros_like(valid),
             "glyph_mask": kernel, "glyph_validity": valid}
    for branch, key in (("glyph_word", "glyph"), ("embedding", "embedding"), ("adaptive_threshold", "threshold")):
        evidence = {"words": torch.randn(1, 2, 6, 12, requires_grad=True), "glyph": None, "embedding": None, "threshold": None}
        evidence[key] = torch.randn(1, 4 if key == "embedding" else 1, 6, 12, requires_grad=True)
        model = type("Branch", (), {"branch": branch})()
        loss, terms, supervised = research_loss(model, evidence, batch, cfg)
        assert supervised and torch.isfinite(loss)
        loss.backward()
        assert evidence[key].grad.abs().sum() > 0 and evidence["words"].grad.abs().sum() > 0


def test_stability_has_no_gt_dependency_and_handles_fragmentation(cfg):
    p = np.zeros((2, 12, 20), np.float32)
    p[0, 3:9, 2:18] = .95; p[0, 3:9, 9:11] = .6; p[1, 5, 4] = 1
    labels, rows = mask_stability(p, np.ones((12, 20), bool), cfg["postprocessing"], cfg["stability_thresholds"])
    assert rows and any(r["risk"] > 0 for r in rows)
    assert np.array_equal(labels > 0, p[0] >= .65)
    assert not any("gt" in key for row in rows for key in row)


def test_auc_ties_empty_classes_and_order():
    assert binary_auc([0., 1.], [False, True]) == 1
    assert binary_auc([1., 0.], [False, True]) == 0
    assert binary_auc([1., 1.], [False, True]) == .5
    assert binary_auc([1.], [True]) is None


def test_quality_backbone_freezes_parameters_and_bn():
    model = WordResearchAGUNet("quality", skip_floor=.25, embedding_dimensions=4)
    model.train()
    assert all(not m.training for name, m in model.named_modules() if name and not name.startswith("quality"))
    assert model.quality.training
    assert all(p.requires_grad == name.startswith("quality.") for name, p in model.named_parameters())


def test_warm_start_strictly_preserves_kernel_outputs_and_rejects_missing_keys():
    from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet
    base = PracticalTotalTextWordKernelAGUNet(skip_floor=.25, encoder_pretrained=False).eval()
    model = WordResearchAGUNet("glyph_word", skip_floor=.25, embedding_dimensions=4).eval()
    receipt = model.initialize_kernel_state(base.state_dict())
    assert receipt["new_tensors"] == ["glyph.bias", "glyph.weight"]
    with torch.inference_mode():
        image = torch.randn(2, 3, 64, 64)
        assert torch.equal(base(image), model(image))
    state = dict(base.state_dict()); state.pop("head.bias")
    with pytest.raises(ValueError, match="full qualified"):
        model.initialize_kernel_state(state)
