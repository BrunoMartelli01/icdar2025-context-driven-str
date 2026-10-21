"""Predicted-window supervision, reciprocal connectivity, and crop coordinates."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch

from localize.unet_crop_links import (reciprocal_components, context_box, original_label_map,
    patches_from_labels, merge_duplicate_supports)
from localize.unet_word_research_v2 import link_instances
from models.unet_crop_links import CropLinkAGUNet
from models.unet_word_research_v2 import WordResearchV2AGUNet
from scripts.run_unet_crop_links import DEFAULT_CONFIG, read_config, crop_records
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.unet_crop_links import crop_supervision, PredictedCropDataset, crop_loss, train_crop_epoch
from segmentation.unet_word_research_v2 import link_targets


@pytest.fixture
def cfg():
    return deepcopy(read_config(DEFAULT_CONFIG, "approximate")[0])


@pytest.mark.parametrize("seed", range(8))
def test_sparse_links_equal_original_connectivity_and_ids(seed, cfg):
    rng = np.random.default_rng(seed)
    p = rng.random((2, 21, 31)).astype(np.float32)
    links = rng.random((8, 21, 31)).astype(np.float32)
    canvas = rng.random((21, 31)) > .1
    old = link_instances(p, links, canvas, cfg["postprocessing"], cfg["links"])
    new = reciprocal_components(p, links, canvas, cfg["postprocessing"], cfg["links"])
    assert np.array_equal(old[0], new[0]) and old[1] == new[1]


def test_perfect_links_separate_touching_words_and_retain_foreground(cfg):
    owners = torch.zeros(1, 10, 16, dtype=torch.int64)
    owners[:, 2:8, 2:7] = 1; owners[:, 2:8, 7:14] = 2
    valid = torch.ones(1, 1, 10, 16)
    targets, _ = link_targets(owners, valid)
    p = np.stack(((owners[0] > 0).numpy(), np.zeros((10, 16)))).astype(np.float32)
    labels, audit = reciprocal_components(p, targets[0].numpy(), np.ones((10, 16), bool), cfg["postprocessing"], cfg["links"])
    assert audit["link_components"] == 2
    assert np.array_equal(labels > 0, owners[0].numpy() > 0)
    assert labels[4, 4] != labels[4, 10]


def test_empty_and_nonfinite_links_are_explicit(cfg):
    p = np.zeros((2, 4, 8), np.float32); links = np.ones((8, 4, 8))
    labels, audit = reciprocal_components(p, links, np.ones((4, 8), bool), cfg["postprocessing"], cfg["links"])
    assert not labels.any() and audit["link_components"] == 0
    links[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="evidence"):
        reciprocal_components(p, links, np.ones((4, 8), bool), cfg["postprocessing"], cfg["links"])


def test_windows_use_predictions_and_are_clamped(cfg):
    proposals = [{"bbox": [1, 2, 11, 8], "instance_id": 5}]
    crops = crop_records(proposals, (20, 12), cfg)
    assert crops[0]["seed_id"] == 5 and crops[0]["context_bbox"] == [0, 0, 16, 12]
    assert not crops[0]["whole_image_fallback"]
    empty = crop_records([], (20, 12), cfg)
    assert empty == [{"seed_id": 0, "seed_bbox": [0, 0, 20, 12], "context_bbox": [0, 0, 20, 12], "whole_image_fallback": True}]
    with pytest.raises(ValueError): context_box([0, 0, 21, 8], (20, 12), cfg["crop"])


def test_truncated_neighbors_are_ignored_and_complete_words_remain():
    owner = np.zeros((12, 24), np.int32)
    owner[2:10, 2:8] = 3; owner[2:10, 10:22] = 7
    original = owner.copy()
    cropped, valid, ignored = crop_supervision(owner, np.ones_like(owner, bool), [0, 0, 15, 12], minimum_visible_fraction=.95)
    assert ignored == [7] and set(np.unique(cropped)) == {0, 3}
    assert not valid[2:10, 10:15].any() and valid[2:10, 2:8].all()
    assert np.array_equal(owner, original)


def test_visibility_threshold_is_inclusive_and_excludes_existing_ignore():
    owner = np.ones((2, 10), np.int32); valid = np.ones_like(owner, bool)
    cropped, v, ignored = crop_supervision(owner, valid, [0, 0, 9, 2], minimum_visible_fraction=.9)
    assert not ignored and cropped.sum() == 18 and v.all()
    valid[:, 9] = False
    _, _, ignored = crop_supervision(owner, valid, [0, 0, 9, 2], minimum_visible_fraction=1.)
    assert not ignored


def test_letterbox_projection_and_offsets_preserve_anchor_identity():
    labels = np.zeros((16, 16), np.int32)
    labels[6:10, 2:6] = 1; labels[6:10, 10:14] = 2
    projected = original_label_map(labels, (32, 16))
    patches, skipped = patches_from_labels(labels, (32, 16), offset=(7, 11), anchor=projected == 1)
    assert skipped == 0 and len(patches) == 1
    assert patches[0]["bbox"] == [11, 15, 19, 23]
    assert patches[0]["mask"].all()


def test_duplicate_supports_merge_but_neighbor_words_stay_separate():
    p = [{"bbox": [0, 0, 8, 4], "mask": np.ones((4, 8), bool)},
         {"bbox": [1, 0, 9, 4], "mask": np.ones((4, 8), bool)},
         {"bbox": [10, 0, 18, 4], "mask": np.ones((4, 8), bool)}]
    merged = merge_duplicate_supports(p, overlap_fraction=.6)
    assert len(merged) == 2 and merged[0]["bbox"] == [0, 0, 9, 4]
    assert merged[0]["mask"].sum() == 36
    assert p[0]["mask"].shape == (4, 8)


def test_overlapping_boxes_without_overlapping_support_do_not_merge():
    a = np.zeros((4, 8), bool); a[:, :3] = True
    b = np.zeros((4, 8), bool); b[:, 5:] = True
    assert len(merge_duplicate_supports([{"bbox": [0, 0, 8, 4], "mask": a},
        {"bbox": [0, 0, 8, 4], "mask": b}], overlap_fraction=.6)) == 2


def test_dataset_crop_sampling_and_augmented_owner_ids_are_reproducible(tmp_path, cfg):
    _, source = read_config(DEFAULT_CONFIG, "approximate")
    owners = np.zeros((32, 64), np.int32); owners[4:28, 4:20] = 3; owners[4:28, 40:60] = 7
    owner_path = tmp_path / "owners.tiff"; Image.fromarray(owners, mode="I").save(owner_path)
    item = {"image": Image.fromarray(np.zeros((32, 64, 3), np.uint8)),
        "mask": Image.fromarray((owners > 0).astype(np.uint8) * 255),
        "validity": Image.fromarray(np.full((32, 64), 255, np.uint8)),
        "separation": Image.fromarray(np.zeros((32, 64), np.uint8))}
    class Raw:
        manifest = SimpleNamespace(samples=[SimpleNamespace(sample_id="total_text:fixture")])
        paths = [(None, None, None, None, owner_path)]
        def __getitem__(self, _): return item
    rows = {"total_text:fixture": {"crops": [{"context_bbox": [0, 0, 64, 32]}]}}
    transform = _transform(source, "letterbox", train=False)
    dataset = PredictedCropDataset(Raw(), rows, [0], transform, cfg)
    first, second = dataset[0], dataset[0]
    assert torch.equal(first["image"], second["image"])
    assert torch.equal(first["owners_native"], second["owners_native"])
    assert set(torch.unique(first["owners_native"]).tolist()) == {0, 3, 7}
    assert first["image"].shape == (3, 512, 512) and first["mask"].shape == (1, 256, 256)
    assert torch.all(first["kernel"] <= first["mask"])


@pytest.mark.parametrize("variant", ["crop_word", "crop_links"])
def test_independent_models_strictly_copy_source_freeze_encoder_and_update_heads(cfg, variant):
    source = WordResearchV2AGUNet("reference_control", cfg)
    model = CropLinkAGUNet(variant, cfg)
    model.initialize_control_state(source.state_dict())
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in source.state_dict().items())
    assert all(not p.requires_grad for name, p in model.named_parameters() if name.startswith("encoder."))
    before = model.head.weight.detach().clone()
    private_before = model.links_private[-1].weight.detach().clone() if variant == "crop_links" else None
    image = torch.randn(2, 3, 64, 64)
    valid = torch.ones(2, 1, 32, 32)
    owners = torch.zeros(2, 32, 32, dtype=torch.int64); owners[:, 4:28, 3:15] = 1; owners[:, 4:28, 17:30] = 2
    batch = {"image": image, "mask": (owners > 0)[:, None].float(), "kernel": (owners > 0)[:, None].float(),
             "validity": valid, "separation": valid * 0, "owners_native": owners,
             "truncated_words": torch.zeros(2, dtype=torch.int64), "sample_id": ["a", "b"]}
    model.train(); evidence = model.forward_evidence(image)
    loss, terms, coefficient = crop_loss(evidence, batch, cfg, 1)
    assert torch.isfinite(loss)
    if variant == "crop_links":
        assert evidence["links"].shape == (2, 8, 32, 32) and coefficient == pytest.approx(.05)
    else: assert "links" not in terms and coefficient == 0
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    trained = train_crop_epoch(model, [batch], optimizer, torch.device("cpu"), cfg, 1)
    assert trained["samples"] == 2 and trained["steps"] == 1
    assert not torch.equal(before, model.head.weight)
    assert torch.equal(before, source.head.weight)
    if private_before is not None:
        assert not torch.equal(private_before, model.links_private[-1].weight)


def test_strict_warm_start_rejects_missing_original_tensor(cfg):
    model = CropLinkAGUNet("crop_links", cfg)
    state = WordResearchV2AGUNet("reference_control", cfg).state_dict()
    state.pop("head.weight")
    with pytest.raises(ValueError, match="every original"):
        model.initialize_control_state(state)
