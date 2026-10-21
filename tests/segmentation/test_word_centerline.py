"""Geometry, supervised gradients, and frozen-source checks for line targets."""

from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet
from models.word_centerline import CenterlineRadiusAGUNet
from segmentation.word_centerline import (WordCenterlineDataset, centerline_targets,
    centerline_loss, decoded_radius, skeletonize_mask, train_centerline_epoch)


TARGET = {"ribbon_fraction": .2, "min_half_width": 0.0}
LOSS = {"bce_weight": 1., "dice_weight": 1., "smooth": 1., "lambda_radius": 1.,
        "radius_floor": .5, "radius_smooth_l1_beta": 1.}
MODEL = {"skip_floor": .25, "private_head_channels": 32, "initial_line_bias": -2.,
         "initial_radius": 2., "loss": LOSS}


def owner_map():
    owners = np.zeros((64, 96), np.int32)
    owners[3:12, 3:43] = 1
    cv2.fillPoly(owners, [np.array([[52, 3], [86, 13], [84, 22], [50, 12]])], 2)
    cv2.ellipse(owners, (25, 39), (17, 14), 0, 0, 180, 3, thickness=7)
    owners[63, 95] = 4
    owners[34:37, 69:73] = owners[46:50, 76:79] = 5
    return owners


def batch_from_owners(owners, *, valid=None):
    valid = np.ones(owners.shape, np.float32) if valid is None else valid
    targets = centerline_targets(owners, valid, TARGET)
    batch = {key: torch.from_numpy(targets[key])[None, None]
             for key in ("line", "radius", "radius_weight", "line_skeleton")}
    batch["validity"] = torch.from_numpy(valid)[None, None]
    return batch


def test_native_targets_preserve_tilted_curved_tiny_and_fragmented_words():
    owners = owner_map()
    before = owners.copy()
    targets = centerline_targets(owners, np.ones_like(owners), TARGET)
    assert np.array_equal(owners, before)
    assert targets["audit"]["words"] == 5
    assert targets["audit"]["word_fragments"] == 6
    assert targets["audit"]["skeleton_fragments"] == 6
    assert targets["line"][63, 95] == 1
    assert targets["radius"][63, 95] == 1  # outside-image zero padding
    assert not targets["line"][owners == 0].any()
    assert np.all(targets["line_skeleton"] <= targets["line"])
    for owner in range(1, 6):
        support = owners == owner
        assert targets["line"][support].any()
        assert targets["radius_weight"][support].sum() == pytest.approx(1.)
    curve = (owners == 3) & (targets["line_skeleton"] > 0)
    yy, xx = np.where(curve)
    assert xx.max() - xx.min() > 20 and yy.max() - yy.min() > 8
    other = centerline_targets(owners, np.ones_like(owners), TARGET)
    assert targets["audit"] == other["audit"]
    for key in ("line", "radius", "radius_weight", "line_skeleton"):
        assert np.array_equal(targets[key], other[key])


def test_filled_image_and_compact_targets_have_defined_border_radius():
    owners = np.ones((8, 10), np.int32)
    target = centerline_targets(owners, np.ones_like(owners), TARGET)
    assert 0 < target["line"].sum() < owners.size
    assert target["radius"].max() <= 4
    for height, width in ((1, 1), (1, 5), (2, 2), (2, 7), (3, 3)):
        mask = np.ones((height, width), bool)
        skeleton = skeletonize_mask(mask)
        assert skeleton.any() and skeleton.dtype == bool
        assert cv2.connectedComponents(skeleton.astype(np.uint8), connectivity=8)[0] == 2


def test_empty_invalid_and_bad_targets_fail_explicitly():
    owners = np.zeros((8, 12), np.int32)
    targets = centerline_targets(owners, np.zeros_like(owners), TARGET)
    assert targets["audit"]["words"] == 0
    assert not targets["line"].any() and not targets["radius_weight"].any()
    owners[2, 2] = 1
    with pytest.raises(ValueError, match="invalid pixels"):
        centerline_targets(owners, np.zeros_like(owners), TARGET)
    for config in ({**TARGET, "ribbon_fraction": -1}, {**TARGET, "min_half_width": np.nan}):
        with pytest.raises(ValueError, match="configuration"):
            centerline_targets(owners, np.ones_like(owners), config)
    with pytest.raises(ValueError, match="boolean"):
        skeletonize_mask(owners)


def test_radius_gradients_only_on_valid_ribbon_and_per_owner_balance():
    owners = np.zeros((16, 32), np.int32)
    owners[2:10, 2:22] = 1
    owners[13, 28] = 2
    valid = np.ones_like(owners, np.float32)
    valid[0] = 0
    batch = batch_from_owners(owners, valid=valid)
    evidence = {"line": torch.zeros_like(batch["line"], requires_grad=True),
                "radius_raw": torch.zeros_like(batch["radius"], requires_grad=True)}
    total, terms, supervised = centerline_loss(evidence, batch, LOSS)
    assert supervised and total == terms["line"] + terms["radius"]
    total.backward()
    assert not evidence["line"].grad[batch["validity"] == 0].any()
    assert not evidence["radius_raw"].grad[batch["radius_weight"] == 0].any()
    assert torch.isfinite(evidence["radius_raw"].grad).all()
    predicted = decoded_radius(evidence["radius_raw"], .5)
    raw = torch.nn.functional.smooth_l1_loss(predicted.log(), batch["radius"].clamp_min(.5).log(), reduction="none")
    expected = sum(raw[0, 0][(torch.from_numpy(owners == owner)) & (batch["line"][0, 0] > 0)].mean()
                   for owner in (1, 2)) / 2
    assert terms["radius"].item() == pytest.approx(expected.item())


def test_all_invalid_loss_is_differentiable_zero_and_nonfinite_rejected():
    batch = batch_from_owners(np.zeros((8, 12), np.int32), valid=np.zeros((8, 12), np.float32))
    evidence = {"line": torch.zeros_like(batch["line"], requires_grad=True),
                "radius_raw": torch.zeros_like(batch["radius"], requires_grad=True)}
    loss, _, supervised = centerline_loss(evidence, batch, LOSS)
    loss.backward()
    assert not supervised and loss == 0
    assert not evidence["line"].grad.any() and not evidence["radius_raw"].grad.any()
    bad = {**evidence, "radius_raw": evidence["radius_raw"].detach() + float("nan")}
    with pytest.raises(ValueError, match="supervision"):
        centerline_loss(bad, batch, LOSS)


def test_dataset_reuses_paired_augmentation_and_does_not_modify_full_targets(tmp_path):
    from scripts.run_word_multitask import DEFAULT_CONFIG, read_config
    from scripts.train_totaltext_v4_image_frequency_ablation import _transform
    _, source = read_config(DEFAULT_CONFIG, "approximate")
    source["training"].update(horizontal_flip_probability=1., rotation_probability=1., rotation_degrees=[12., 12.])
    transform = _transform(source, "letterbox", train=True)
    transform.set_epoch(7)
    owner = np.zeros((20, 40), np.int32)
    owner[3:17, 3:14] = 1
    owner[3:17, 26:37] = 2
    Image.fromarray(owner).save(tmp_path / "owners.tiff")
    rgb = Image.new("RGB", (40, 20), (120, 120, 120))
    image, mask, valid, separation = transform.transform_with_validity_and_separation(
        rgb, Image.fromarray((owner > 0).astype(np.uint8) * 255), Image.new("L", (40, 20), 255),
        Image.new("L", (40, 20), 0), Image.fromarray(owner), 0)
    class Base:
        data_root = tmp_path
        manifest = SimpleNamespace(samples=[SimpleNamespace(word_owner_annotation_path="owners.tiff")])
        def __len__(self): return 1
        def __getitem__(self, index):
            return {"image": image, "mask": mask, "validity": valid, "separation": separation, "sample_id": "word:1"}
    base = Base()
    base.transform = transform
    dataset = WordCenterlineDataset(base, distance_fraction=.5, target_config=TARGET)
    a, b = dataset[0], dataset[0]
    assert torch.equal(a["mask"], mask) and torch.equal(a["line"], b["line"])
    assert a["line"].shape == (1, 256, 256)
    assert not a["line"][a["mask"] == 0].any()
    assert not a["radius_weight"][a["validity"] == 0].any()


def test_model_strict_transfer_training_preserves_every_source_tensor_and_output():
    torch.set_num_threads(1)
    source = PracticalTotalTextWordKernelAGUNet(skip_floor=.25, encoder_pretrained=False).eval()
    model = CenterlineRadiusAGUNet(MODEL)
    report = model.initialize_control_state(source.state_dict())
    assert report["new_tensors"] and all(key.startswith("centerline_private.") for key in report["new_tensors"])
    model.train()
    assert all(parameter.requires_grad == key.startswith("centerline_private.") for key, parameter in model.named_parameters())
    assert not any(module.training for module in model.modules() if isinstance(module, torch.nn.BatchNorm2d))
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        before = model.forward_evidence(image)
        expected = source(image)
    assert torch.equal(expected, before["words"])
    assert before["line"].shape == before["radius_raw"].shape == (1, 1, 32, 32)
    assert torch.all(before["line"] == -2.)
    assert torch.allclose(decoded_radius(before["radius_raw"], .5), torch.full((1, 1, 32, 32), 2.))
    owner = np.zeros((32, 32), np.int32)
    owner[8:24, 3:29] = 1
    batch = batch_from_owners(owner)
    batch.update(image=image, sample_id=["word:1"])
    optimizer = torch.optim.Adam([parameter for parameter in model.parameters() if parameter.requires_grad], lr=.001)
    result = train_centerline_epoch(model, [batch], optimizer, "cpu", {"loss": LOSS})
    assert result["steps"] == 1 and result["samples"] == 1
    with torch.no_grad():
        after = model.forward_evidence(image)
    assert torch.equal(expected, after["words"])
    assert not torch.equal(before["line"], after["line"])
    for key, value in source.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    broken = dict(source.state_dict())
    broken.pop("head.bias")
    with pytest.raises(ValueError, match="every original"):
        model.initialize_control_state(broken)
    broken = dict(source.state_dict())
    broken["head.bias"] = broken["head.bias"].double()
    with pytest.raises(ValueError, match="shape/dtype"):
        model.initialize_control_state(broken)
