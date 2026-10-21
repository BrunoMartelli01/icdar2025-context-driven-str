"""Corrective instance geometry, protected warmup and explicit label semantics."""

from copy import deepcopy
import numpy as np
import pytest
import torch

from localize.unet_word_research_v2 import (NEIGHBORS, OPPOSITE, link_instances,
    research_v2_instances, soft_embedding_instances)
from localize.word_multitask import word_instances
from models.unet_word_research_v2 import WordResearchV2AGUNet
from scripts.run_unet_word_research_v2 import DEFAULT_CONFIG, read_config, checkpoint_key
from segmentation.unet_word_research_v2 import (balanced_link_loss, build_research_v2_optimizer,
    db_raster_targets, glyph_weight, link_targets, research_v2_loss, gradient_diagnostics,
    train_research_v2_epoch)


@pytest.fixture
def cfg():
    return deepcopy(read_config(DEFAULT_CONFIG, "approximate")[0])


def word_maps():
    p = np.zeros((2, 12, 20), np.float32)
    p[0, 3:9, 2:18] = .95
    p[1, 5:7, 3:5] = .95; p[1, 5:7, 15:17] = .95
    return p, np.ones((12, 20), bool)


def test_soft_growth_never_fragments_rejected_pixels_and_preserves_seeds(cfg):
    p, canvas = word_maps(); e = np.zeros((4, 12, 20), np.float32)
    e[0, :, 10:] = 100
    labels, audit = soft_embedding_instances(p, e, canvas, cfg["postprocessing"], cfg["soft_embedding"])
    assert np.array_equal(labels > 0, p[0] >= .65)
    assert len(np.unique(labels[labels > 0])) == 2
    assert labels[5, 3] != labels[5, 15]
    assert audit["unseeded_fallback_components"] == 0


def test_zero_soft_weight_exactly_equals_watershed_and_is_deterministic(cfg):
    p, canvas = word_maps(); e = np.random.default_rng(12).normal(size=(4, 12, 20))
    options = {**cfg["soft_embedding"], "weight": 0}
    expected = word_instances(p, canvas, cfg["postprocessing"])[0]
    actual = soft_embedding_instances(p, e, canvas, cfg["postprocessing"], options)[0]
    assert np.array_equal(actual, expected)
    first = soft_embedding_instances(p, e, canvas, cfg["postprocessing"], cfg["soft_embedding"])[0]
    second = soft_embedding_instances(p, e, canvas, cfg["postprocessing"], cfg["soft_embedding"])[0]
    assert np.array_equal(first, second)


def test_soft_growth_retains_unseeded_islands_and_cannot_cross_background(cfg):
    p, canvas = word_maps(); p[0, :, 9:11] = 0; p[1, :, 12:] = 0
    labels, audit = soft_embedding_instances(p, np.zeros((4, 12, 20)), canvas, cfg["postprocessing"], cfg["soft_embedding"])
    assert audit["unseeded_fallback_components"] == 1
    assert not labels[:, 9:11].any() and labels[5, 3] != labels[5, 15]


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_soft_growth_rejects_nonfinite_embeddings(cfg, value):
    p, canvas = word_maps(); e = np.zeros((4, 12, 20)); e[0, 0, 0] = value
    with pytest.raises(ValueError, match="evidence"):
        soft_embedding_instances(p, e, canvas, cfg["postprocessing"], cfg["soft_embedding"])


def owner_batch():
    owner = torch.zeros(1, 8, 12, dtype=torch.int64)
    owner[0, 1:7, 1:5] = 1; owner[0, 1:7, 5:9] = 2
    valid = torch.ones(1, 1, 8, 12)
    return owner, valid


def test_links_use_word_ids_include_boundaries_and_never_wrap():
    owner, valid = owner_batch(); target, weight = link_targets(owner, valid)
    right = NEIGHBORS.index((0, 1))
    assert target[0, right, 3, 3] == 1
    assert target[0, right, 3, 4] == 0 and weight[0, right, 3, 4] > 0
    assert target[0, right, 3, 8] == 0 and weight[0, right, 3, 8] > 0
    assert not weight[:, :, 0].any() and not weight[:, right, :, -1].any()
    valid[:, :, :, 5] = 0
    _, weight = link_targets(owner, valid)
    assert weight[0, right, 3, 4] == 0


def test_link_class_balance_and_empty_supervision_gradients():
    owner, valid = owner_batch(); target, weights = link_targets(owner, valid)
    logits = torch.zeros_like(target, requires_grad=True)
    loss = balanced_link_loss(logits, target, weights)
    assert loss.item() == pytest.approx(2 * np.log(2))
    loss.backward(); assert logits.grad[weights > 0].abs().sum() > 0
    assert not logits.grad[weights == 0].any()
    logits = torch.zeros_like(target, requires_grad=True)
    loss = balanced_link_loss(logits, target, weights * 0); loss.backward()
    assert loss == 0 and not logits.grad.any()


def test_reciprocal_links_separate_touching_words_and_keep_all_foreground(cfg):
    owner, valid = owner_batch(); target, _ = link_targets(owner, valid)
    p = np.stack(((owner[0] > 0).numpy(), np.zeros((8, 12)))).astype(np.float32)
    labels, audit = link_instances(p, target[0].numpy(), np.ones((8, 12), bool), cfg["postprocessing"], cfg["links"])
    assert audit["link_components"] == 2
    assert labels[3, 3] != labels[3, 6]
    assert np.array_equal(labels > 0, owner[0].numpy() > 0)


def test_one_way_link_cannot_connect_two_pixels(cfg):
    p = np.zeros((2, 3, 4), np.float32); p[0, 1, 1:3] = 1
    links = np.zeros((8, 3, 4), np.float32); links[NEIGHBORS.index((0, 1)), 1, 1] = 1
    labels, _ = link_instances(p, links, np.ones((3, 4), bool), cfg["postprocessing"], cfg["links"])
    assert labels[1, 1] != labels[1, 2]


def test_db_core_shrinks_each_owner_and_respects_ignore(cfg):
    owner, valid = owner_batch(); valid[:, :, 0] = 0
    core, threshold, weight, fallback = db_raster_targets(owner, valid,
        **{k: cfg["db"][k] for k in ("shrink_ratio", "threshold_range")})
    assert core.sum() > 0 and core.sum() < (owner > 0).sum()
    assert not core[valid == 0].any() and not weight[valid == 0].any()
    assert threshold.min() >= .3 and threshold.max() <= .7
    for i in (1, 2):
        assert (core[0, 0][owner[0] == i] > 0).any()
    assert fallback >= 0


def test_db_one_pixel_and_empty_targets_are_explicit(cfg):
    owner = torch.zeros(1, 4, 4, dtype=torch.int64); owner[0, 1, 1] = 1
    valid = torch.ones(1, 1, 4, 4)
    core, _, weight, _ = db_raster_targets(owner, valid,
        **{k: cfg["db"][k] for k in ("shrink_ratio", "threshold_range")})
    assert core.sum() == 1 and weight[0, 0, 1, 1] == 1
    core, _, weight, fallback = db_raster_targets(owner * 0, valid,
        **{k: cfg["db"][k] for k in ("shrink_ratio", "threshold_range")})
    assert not core.any() and not weight.any() and fallback == 0


def test_glyph_schedule_has_protected_warmup_and_bounded_ramp(cfg):
    assert glyph_weight(cfg, 10) == 0
    assert glyph_weight(cfg, 11) == pytest.approx(.01)
    assert glyph_weight(cfg, 20) == pytest.approx(.1)
    assert glyph_weight(cfg, 50) == pytest.approx(.1)


@pytest.mark.parametrize("branch", ["glyph_isolated", "glyph_staged", "links_isolated", "threshold_frozen", "db_core"])
def test_private_heads_protect_all_original_weights_bn_and_shapes(cfg, branch):
    model = WordResearchV2AGUNet(branch, cfg)
    model.train(); model.set_research_epoch(1); model.train()
    assert all(not p.requires_grad for name, p in model.named_parameters() if name in model.original_trainability)
    assert not any(m.training for name, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d))
    before = {name: p.detach().clone() for name, p in model.named_parameters() if name.startswith("head.")}
    image = torch.randn(2, 3, 64, 64)
    with torch.no_grad(): evidence = model.forward_evidence(image)
    assert evidence["words"].shape == (2, 2, 32, 32)
    if branch == "glyph_staged":
        model.set_research_epoch(11); model.train()
        assert model.shared_trainable and model.head.weight.requires_grad
        assert not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())
    assert all(torch.equal(before[k], p) for k, p in model.named_parameters() if k in before)


def test_strict_warm_transfer_and_initial_threshold_core_geometry(cfg):
    source = WordResearchV2AGUNet("reference_control", cfg).eval()
    original = source.state_dict(); image = torch.randn(2, 3, 64, 64)
    with torch.inference_mode(): expected = source(image)
    for branch in ("glyph_isolated", "threshold_frozen", "db_core"):
        model = WordResearchV2AGUNet(branch, cfg).eval(); model.initialize_control_state(original)
        with torch.inference_mode(): evidence = model.forward_evidence(image)
        assert torch.equal(expected, evidence["words"])
        bad = dict(original); bad.pop("head.bias")
        with pytest.raises(ValueError, match="every original"):
            model.initialize_control_state(bad)


def test_checkpoint_selection_targets_glyph_learning_only_in_isolated_branch():
    a = {"glyph_pixel_f1": .8, "proposal_f1_iou_50": .5, "gt_words_touched_by_fused_components": 1,
         "word_recall_iou_50": .5, "foreground_iou": .7}
    b = {**a, "glyph_pixel_f1": .5, "proposal_f1_iou_50": .8}
    assert checkpoint_key(a, 2, "glyph_isolated") > checkpoint_key(b, 2, "glyph_isolated")
    assert checkpoint_key(a, 2, "glyph_staged") < checkpoint_key(b, 2, "glyph_staged")


def test_config_mode_isolation_and_v1_not_accepted():
    with pytest.raises(ValueError, match="APPROXIMATE"):
        read_config(DEFAULT_CONFIG, "strict")
    with pytest.raises(ValueError, match="contract"):
        read_config(DEFAULT_CONFIG.with_name("unet_word_research_v1.json"), "approximate")


@pytest.mark.parametrize("branch,key", [("glyph_isolated", "glyph"), ("links_isolated", "links"),
                                      ("threshold_frozen", "threshold"), ("db_core", "db_probability")])
def test_new_losses_update_private_evidence_without_word_gradients(cfg, branch, key):
    from types import SimpleNamespace
    owner, valid = owner_batch()
    batch = {"mask": (owner > 0).float()[:, None], "kernel": (owner > 0).float()[:, None],
        "validity": valid, "owners_native": owner, "separation": valid * 0,
        "glyph_mask": (owner > 0).float()[:, None], "glyph_validity": valid}
    evidence = {"words": torch.randn(1, 2, 8, 12, requires_grad=True), "glyph": None,
                "links": None, "threshold": None, "db_probability": None, "db_threshold": None}
    evidence[key] = torch.randn(1, 8 if key == "links" else 1, 8, 12, requires_grad=True)
    if branch == "db_core":
        evidence["db_threshold"] = torch.randn(1, 1, 8, 12, requires_grad=True)
    model = SimpleNamespace(branch=branch, shared_trainable=False, research_epoch=1)
    loss, terms, supervised = research_v2_loss(model, evidence, batch, cfg)
    assert supervised and torch.isfinite(loss)
    loss.backward()
    # Frozen word evidence must be detached in actual models; this fixture gives
    # it requires_grad to catch accidental direct word supervision in objectives.
    if branch in ("glyph_isolated", "links_isolated", "db_core"):
        assert not evidence["words"].grad.any()
    assert evidence[key].grad.abs().sum() > 0


def test_optimizer_retains_future_shared_parameters_across_warmup(cfg):
    model = WordResearchV2AGUNet("glyph_staged", cfg)
    optimizer = build_research_v2_optimizer(model, cfg)
    before = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert id(model.head.weight) in before and not model.head.weight.requires_grad
    assert [g["lr"] for g in optimizer.param_groups] == [cfg["learning_rate"], cfg["head_learning_rate"]]
    model.set_research_epoch(cfg["glyph"]["warmup_epochs"] + 1)
    assert model.head.weight.requires_grad
    assert before == [id(p) for group in optimizer.param_groups for p in group["params"]]


def test_unsupervised_link_batch_skips_weight_decay(cfg):
    class EmptyLinks(torch.nn.Module):
        branch = "links_isolated"
        shared_trainable = False
        research_epoch = 1
        def __init__(self):
            super().__init__(); self.value = torch.nn.Parameter(torch.tensor(.4))
        def forward_evidence(self, image):
            return {"words": torch.zeros(1, 2, 4, 4), "glyph": None,
                "links": self.value.expand(1, 8, 4, 4), "threshold": None,
                "db_probability": None, "db_threshold": None}
    model = EmptyLinks(); optimizer = torch.optim.Adam(model.parameters(), lr=.1, weight_decay=.5)
    batch = {"image": torch.zeros(1, 3, 8, 8), "owners_native": torch.zeros(1, 4, 4, dtype=torch.int64),
             "validity": torch.ones(1, 1, 4, 4), "sample_id": ["empty"]}
    before = model.value.detach().clone()
    result = train_research_v2_epoch(model, [batch], optimizer, torch.device("cpu"), cfg)
    assert result["skipped_unsupervised_batches"] == 1 and torch.equal(before, model.value)


def test_oom_is_propagated_without_alternate_masks(cfg):
    class Broken(torch.nn.Module):
        def forward_evidence(self, image):
            raise torch.cuda.OutOfMemoryError("qualify propagation")
    with pytest.raises(torch.cuda.OutOfMemoryError, match="propagation"):
        train_research_v2_epoch(Broken(), [{"image": torch.zeros(1, 3, 8, 8)}], None, torch.device("cpu"), cfg)
