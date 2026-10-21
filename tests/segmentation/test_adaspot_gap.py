"""Gap target geometry, paired augmentation, masked gradients and optimizer safety."""

from copy import deepcopy
from pathlib import PurePosixPath

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.run_adaspot_gap import DEFAULT_CONFIG, read_config
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.adaspot_gap import AdaSpotGapDataset, gap_loss, make_interword_gaps, train_gap_epoch
from segmentation.data import SegmentationDataset, SegmentationManifest, SegmentationSample, file_sha256
from segmentation.totaltext_word_kernel import make_word_kernels, WordKernelDataset


@pytest.fixture
def config():
    return read_config(DEFAULT_CONFIG, "approximate")[0]


def owners_pair():
    owners = np.zeros((12, 16), np.int32)
    owners[3:9, 2:6] = 1
    owners[3:9, 7:11] = 2
    return owners


@pytest.mark.parametrize("turn", range(4))
def test_gap_geometry_rotates_and_never_erases_words(turn):
    owners = owners_pair()
    expected, _ = make_interword_gaps(owners, np.ones_like(owners), radius=2)
    rotated = np.rot90(owners, turn)
    actual, audit = make_interword_gaps(rotated, np.ones_like(rotated), radius=2)
    assert np.array_equal(actual, np.rot90(expected, turn))
    assert actual.any() and not np.any(actual & (rotated > 0))
    assert audit["owners"] == 2


def test_single_owner_fragments_are_not_two_words_and_far_words_have_no_gap():
    owners = owners_pair()
    owners[owners == 2] = 1
    assert not make_interword_gaps(owners, np.ones_like(owners), radius=3)[0].any()
    owners[:] = 0; owners[1:3, 1:3] = 1; owners[8:10, 12:14] = 2
    assert not make_interword_gaps(owners, np.ones_like(owners), radius=2)[0].any()


def test_roi_gap_target_equals_independent_full_canvas_squared_distance():
    owners = np.zeros((15, 18), np.int32)
    owners[1:7, 1:5] = 1; owners[6:12, 7:10] = 2; owners[9:13, 12:16] = 3
    owners[9, 1] = 1  # Fragment: still the same distinct neighbor.
    yy, xx = np.indices(owners.shape)
    for radius in (1, 2, 3, 4):
        neighbors = np.zeros_like(owners)
        for word in (1, 2, 3):
            wy, wx = np.where(owners == word)
            minimum = ((yy[..., None] - wy) ** 2 + (xx[..., None] - wx) ** 2).min(axis=-1)
            neighbors += minimum <= radius ** 2
        expected = (owners == 0) & (neighbors >= 2)
        assert np.array_equal(make_interword_gaps(owners, np.ones_like(owners), radius=radius)[0], expected)


def test_gap_targets_clip_ignore_and_reject_invalid_owner_or_radius():
    owners = owners_pair()
    valid = np.ones_like(owners); valid[:, 6] = 0
    gap, _ = make_interword_gaps(owners, valid, radius=1)
    assert not gap[:, 6].any()
    valid[4, 3] = 0
    with pytest.raises(ValueError, match="outside valid"):
        make_interword_gaps(owners, valid, radius=2)
    for radius in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="radius"):
            make_interword_gaps(owners, np.ones_like(owners), radius=radius)


def loss_batch():
    owners = owners_pair()
    kernel, _ = make_word_kernels(owners)
    gap, _ = make_interword_gaps(owners, np.ones_like(owners), radius=2)
    return {"image": torch.zeros(1, 3, *owners.shape), "mask": torch.tensor(owners > 0)[None, None].float(),
        "kernel": torch.tensor(kernel > 0)[None, None].float(), "gap_target": torch.tensor(gap)[None, None].float(),
        "validity": torch.ones(1, 1, *owners.shape), "separation": torch.zeros(1, 1, *owners.shape),
        "sample_id": ["fixture:1"]}


def test_loss_masks_ignore_and_control_has_exactly_same_common_terms(config):
    batch = loss_batch()
    batch["validity"][:, :, 0] = 0
    words = torch.zeros(1, 2, 12, 16, requires_grad=True)
    gaps = torch.zeros(1, 1, 12, 16, requires_grad=True)
    loss, terms = gap_loss({"words": words, "gap": gaps}, batch, config["loss"])
    control, common = gap_loss({"words": words, "gap": None}, batch, config["loss"])
    assert set(common) == {"text", "kernel"}
    assert all(torch.equal(common[k], terms[k]) for k in common)
    assert torch.allclose(loss, control + terms["gap"])
    loss.backward()
    assert not words.grad[:, :, 0].any() and not gaps.grad[:, :, 0].any()
    assert words.grad[:, :, 1:].abs().sum() > 0 and gaps.grad[:, :, 1:].abs().sum() > 0


def test_loss_rejects_gap_over_text_and_nonbinary_gap(config):
    batch = loss_batch()
    evidence = {"words": torch.zeros(1, 2, 12, 16), "gap": torch.zeros(1, 1, 12, 16)}
    batch["gap_target"][0, 0, 4, 3] = 1
    with pytest.raises(ValueError, match="containment"):
        gap_loss(evidence, batch, config["loss"])
    batch = loss_batch(); batch["gap_target"][0, 0, 4, 6] = .2
    with pytest.raises(ValueError, match="binary"):
        gap_loss(evidence, batch, config["loss"])


class TinyGap(torch.nn.Module):
    branch = "gap"
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Conv2d(3, 2, 1)
        self.gap = torch.nn.Conv2d(3, 1, 1)
    def forward_evidence(self, image):
        return {"words": self.head(image), "gap": self.gap(image)}


def test_all_invalid_batch_does_not_apply_weight_decay(config):
    batch = loss_batch()
    for key in ("mask", "kernel", "gap_target", "validity"):
        batch[key].zero_()
    model = TinyGap()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.5)
    result = train_gap_epoch(model, [batch], optimizer, torch.device("cpu"), config, progress_every=0)
    assert result["steps"] == 0 and result["skipped_all_invalid_batches"] == 1
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_inference_error_and_oom_propagate(config):
    class Broken(TinyGap):
        def forward_evidence(self, image):
            raise torch.cuda.OutOfMemoryError("explicit fixture OOM")
    model = Broken()
    with pytest.raises(torch.cuda.OutOfMemoryError, match="explicit fixture"):
        train_gap_epoch(model, [loss_batch()], torch.optim.Adam(model.parameters()), torch.device("cpu"), config)


def test_paired_flip_rotation_letterbox_preserves_existing_targets(tmp_path):
    config, source = read_config(DEFAULT_CONFIG, "approximate")
    source = deepcopy(source)
    source["training"].update(horizontal_flip_probability=1., rotation_probability=1.)
    transform = _transform(source, "letterbox", train=True)
    transform.set_epoch(4)
    owners = np.zeros((48, 80), np.uint16); owners[10:30, 8:33] = 1; owners[10:30, 37:65] = 2
    images = {"rgb": Image.new("RGB", (80, 48), (40, 100, 120)),
        "mask": Image.fromarray((owners > 0).astype(np.uint8) * 255),
        "owners": Image.fromarray(owners), "valid": Image.new("L", (80, 48), 255),
        "sep": Image.new("L", (80, 48), 0)}
    for name, image in images.items():
        image.save(tmp_path / (name + ".png"))
    def pair(name):
        path = tmp_path / (name + ".png")
        return PurePosixPath(path.name), file_sha256(path)
    rgb, rgb_sha = pair("rgb"); mask, mask_sha = pair("mask")
    valid, valid_sha = pair("valid"); sep, sep_sha = pair("sep"); owner, owner_sha = pair("owners")
    sample = SegmentationSample("total_text:fixture", rgb, mask, rgb_sha, mask_sha,
        validity_annotation_path=valid, validity_annotation_sha256=valid_sha,
        separation_annotation_path=sep, separation_annotation_sha256=sep_sha,
        word_owner_annotation_path=owner, word_owner_annotation_sha256=owner_sha)
    manifest = SegmentationManifest(1, "approximate", "approximate reproduction", "total_text", "train", None,
        "train", None, None, {"target_conversion": "binary-mask-with-validity-v1"}, (sample,))
    base = SegmentationDataset(manifest, tmp_path, transform=transform)
    previous = WordKernelDataset(base, distance_fraction=.5)[0]
    current = AdaSpotGapDataset(base, distance_fraction=.5, gap_radius=3)[0]
    for key in ("image", "mask", "kernel", "validity", "separation", "owners_native"):
        assert torch.equal(previous[key], current[key])
    expected, _ = make_interword_gaps(current["owners_native"].numpy(), current["validity"][0].numpy(), radius=3)
    expected &= ~current["mask"][0].numpy().astype(bool)
    assert np.array_equal(current["gap_target"][0].numpy(), expected)
    assert not (current["gap_target"] * current["mask"]).any()
