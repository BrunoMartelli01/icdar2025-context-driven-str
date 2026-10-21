import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from models.unet_adv import PracticalTotalTextHighResAGUNet
from models.word_center_offsets import WordCenterOffsetAGUNet
from scripts.run_word_multitask import DEFAULT_CONFIG, read_config
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.training import save_checkpoint, load_checkpoint
from segmentation.word_center_offsets import (
    load_center_offset_config, make_word_center_offsets, word_center_offset_loss,
    WordCenterOffsetDataset, WordCenterOffsetTrainingConfig, train_word_center_offset_epoch,
)


CONFIG = Path(__file__).resolve().parents[2] / "configs/approximate/word_center_offsets_v1.json"


def test_dense_targets_touching_words_fractional_centers_ignore_and_single_pixel():
    owners = np.zeros((8, 20), np.int32)
    owners[2:6, 2:8] = 1
    owners[2:6, 8:18] = 2
    owners[7, 19] = 3
    validity = np.ones_like(owners, bool)
    validity[2, 2:8] = False
    original = owners.copy()
    offsets, centers = make_word_center_offsets(owners, validity)
    assert np.array_equal(owners, original)
    assert offsets.shape == (2, 8, 20) and offsets.dtype == np.float32
    assert not offsets[:, ~validity].any() and not offsets[:, owners == 0].any()
    for row in centers:
        yy, xx = np.where((owners == row["owner_id"]) & validity)
        assert np.allclose(xx + offsets[0, yy, xx] * 20, row["center_xy_native"][0])
        assert np.allclose(yy + offsets[1, yy, xx] * 8, row["center_xy_native"][1])
    assert centers[-1]["center_xy_native"] == [19., 7.] and not offsets[:, 7, 19].any()


def test_empty_and_curved_targets_center_is_a_cue_not_a_contour():
    owners = np.zeros((12, 20), np.int32)
    assert not make_word_center_offsets(owners, np.ones_like(owners, bool))[0].any()
    owners[2:10, 2:5] = owners[2:10, 15:18] = owners[8:10, 2:18] = 1
    offsets, centers = make_word_center_offsets(owners, np.ones_like(owners, bool))
    cx, cy = centers[0]["center_xy_native"]
    assert owners[round(cy), round(cx)] == 0
    assert np.isfinite(offsets).all()
    assert not make_word_center_offsets(owners, np.zeros_like(owners, bool))[1]


def test_word_balanced_loss_reference_and_zero_background_ignore_gradient():
    raw = torch.zeros(1, 2, 2, 6, requires_grad=True)
    owners = torch.zeros(1, 2, 6, dtype=torch.int32)
    owners[0, 0, :4] = 1
    owners[0, 1, 0] = 2
    target = torch.zeros_like(raw)
    target[:, :, 0, :4] = .2
    target[:, :, 1, 0] = -.6
    valid = torch.ones(1, 1, 2, 6)
    valid[:, :, 0, 3] = 0
    loss = word_center_offset_loss(raw, target, owners, valid, beta=.1)
    assert loss.item() == pytest.approx((.15 + .55) / 2)
    loss.backward()
    assert not raw.grad[0, :, 0, 3:].any() and not raw.grad[0, :, 1, 1:].any()
    assert torch.all(raw.grad[0, :, 0, :3] < 0) and torch.all(raw.grad[0, :, 1, 0] > 0)
    zero_raw = torch.randn_like(raw, requires_grad=True)
    zero = word_center_offset_loss(zero_raw, target, owners, valid * 0, beta=.1)
    zero.backward()
    assert zero.item() == 0 and not zero_raw.grad.any()


def test_loss_per_image_mean_and_empty_image_not_diluting_supervision():
    raw = torch.zeros(3, 2, 2, 2, requires_grad=True)
    target = torch.full_like(raw, .2)
    owners = torch.ones(3, 2, 2, dtype=torch.int64)
    owners[2] = 0
    target[1] = .6
    value = word_center_offset_loss(raw, target, owners, torch.ones(3, 1, 2, 2), beta=.1)
    assert value.item() == pytest.approx(.35)


@pytest.mark.parametrize("problem", ["nan", "range", "owners", "shape", "beta"])
def test_invalid_loss_inputs_fail(problem):
    raw, target = torch.zeros(1, 2, 2, 2), torch.zeros(1, 2, 2, 2)
    owners, valid, beta = torch.ones(1, 2, 2, dtype=torch.int32), torch.ones(1, 1, 2, 2), .01
    if problem == "nan": raw[0, 0, 0, 0] = float("nan")
    if problem == "range": target[0, 0, 0, 0] = 1
    if problem == "owners": owners = owners.float()
    if problem == "shape": raw = raw[:, :1]
    if problem == "beta": beta = 0
    with pytest.raises(ValueError):
        word_center_offset_loss(raw, target, owners, valid, beta=beta)


def test_dataset_replays_rotation_flip_letterbox_before_vector_construction(tmp_path):
    _, source = read_config(DEFAULT_CONFIG, "approximate")
    source["training"].update(horizontal_flip_probability=1., rotation_probability=1., rotation_degrees=[12., 12.])
    transform = _transform(source, "letterbox", train=True)
    transform.set_epoch(7)
    owners = np.zeros((20, 40), np.int32)
    owners[3:17, 3:20] = 1; owners[3:17, 20:37] = 2
    Image.fromarray(owners).save(tmp_path / "owners.tiff")
    rgb, target = Image.new("RGB", (40, 20), (120, 120, 120)), Image.fromarray((owners > 0).astype(np.uint8)*255)
    image, mask, validity = transform.transform_with_validity(rgb, target, Image.new("L", (40, 20), 255), 0)
    class Base:
        data_root = tmp_path
        manifest = SimpleNamespace(samples=[SimpleNamespace(word_owner_annotation_path=Path("owners.tiff"))])
        def __len__(self): return 1
        def __getitem__(self, index):
            return {"image": image, "mask": mask, "validity": validity, "sample_id": "word:1"}
    base = Base(); base.transform = transform
    dataset = WordCenterOffsetDataset(base)
    a, b = dataset[0], dataset[0]
    assert torch.equal(a["mask"], mask) and torch.equal(a["center_offsets"], b["center_offsets"])
    assert not a["center_offsets"][:, validity[0] == 0].any()
    for owner in (1, 2):
        yy, xx = np.where(a["owners_native"].numpy() == owner)
        assert np.ptp(xx + a["center_offsets"][0, yy, xx].numpy()*mask.shape[-1]) < 1e-4
        assert np.ptp(yy + a["center_offsets"][1, yy, xx].numpy()*mask.shape[-2]) < 1e-4


def test_model_strict_warm_start_identity_shapes_and_unchanged_base():
    torch.manual_seed(37)
    base = PracticalTotalTextHighResAGUNet(skip_floor=.25, encoder_pretrained=False).eval()
    auxiliary = WordCenterOffsetAGUNet(skip_floor=.25, encoder_pretrained=False).eval()
    sample = torch.randn(1, 3, 32, 64)
    with torch.inference_mode(): before = base(sample).clone()
    receipt = auxiliary.initialize_from_text_model(base)
    for key, value in base.state_dict().items():
        if not key.startswith("head."):
            assert torch.equal(value, auxiliary.state_dict()[key])
    assert receipt["reset_tensors"] == ["head.weight", "head.bias"]
    assert sum(p.numel() for p in auxiliary.parameters()) == 58155146
    with torch.inference_mode():
        predicted = auxiliary(sample)
        assert predicted.shape == (1, 2, 16, 32) and not predicted.any()
        assert torch.equal(before, base(sample))
    auxiliary.train()
    assert not auxiliary.encoder.training
    assert all(not p.requires_grad for p in auxiliary.encoder.parameters())
    with pytest.raises(ValueError, match="single-output"):
        auxiliary.initialize_from_text_model(auxiliary)
    with pytest.raises(RuntimeError):
        auxiliary.load_state_dict(base.state_dict(), strict=True)


def test_training_checkpoint_roundtrip_empty_batches_and_head_update(tmp_path):
    model = torch.nn.Conv2d(3, 2, 1)
    owners = np.zeros((8, 12), np.int32); owners[1:7, 2:10] = 1
    target, _ = make_word_center_offsets(owners, np.ones_like(owners, bool))
    batch = {"image": torch.randn(1, 3, 8, 12), "sample_id": ["word:1"],
             "owners_native": torch.from_numpy(owners)[None],
             "center_offsets": torch.from_numpy(target)[None], "validity": torch.ones(1, 1, 8, 12)}
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=.01)
    before = model.weight.detach().clone()
    empty = {**batch, "owners_native": batch["owners_native"] * 0, "sample_id": ["empty:1"]}
    result = train_word_center_offset_epoch(model, [empty], optimizer, torch.device("cpu"), beta=.01)
    assert result["empty_batches_without_update"] == 1 and result["samples"] == 1
    assert torch.equal(before, model.weight)
    trained = train_word_center_offset_epoch(model, [empty, batch], optimizer, torch.device("cpu"), beta=.01)
    assert trained["steps"] == 1 and trained["samples"] == 2 and trained["supervised_images"] == 1
    assert torch.all((model.weight-before).flatten(1).abs().sum(1) > 0)
    config = WordCenterOffsetTrainingConfig(epochs=1, seed=37, adam_betas=(.9, .999), adam_epsilon=1e-8,
        learning_rate_schedule="none", checkpoint_selection="last", deterministic_algorithms=True, offset_beta=.01)
    resolved = load_center_offset_config(CONFIG, mode="approximate")
    save_checkpoint(tmp_path / "smoke.pt", model=model, optimizer=optimizer, epoch=1, global_step=1,
        config=config, resolved_config=json.dumps(resolved), architecture="synthetic-offset-test",
        result_class="smoke-test result", run_contract={"variant": "center_offsets"})
    copy = torch.nn.Conv2d(3, 2, 1)
    payload = load_checkpoint(tmp_path / "smoke.pt", model=copy, expected_architecture="synthetic-offset-test",
        expected_run_contract={"variant": "center_offsets"}, expected_training_config=config)
    assert torch.equal(model.weight, copy.weight) and payload["training_config"]["loss"] == config.loss


def test_config_namespace_modes_and_reproducibility_contract():
    config = load_center_offset_config(CONFIG, mode="approximate")
    assert config["claim_label"] == "approximate reproduction"
    with pytest.raises(ValueError, match="approximate"):
        load_center_offset_config(CONFIG, mode="strict")
    with pytest.raises(ValueError, match="configs/approximate"):
        load_center_offset_config(Path("/tmp/offsets.json"), mode="approximate")


def test_smoke_propagates_oom_and_records_failure(tmp_path, monkeypatch):
    import scripts.profile_word_center_offsets as script
    output = tmp_path / "failed.json"
    monkeypatch.setattr("sys.argv", ["profile", "--mode", "approximate", "--config", str(CONFIG),
        "--device", "cpu", "--batch-size", "2", "--height", "64", "--width", "96", "--seed", "37",
        "--output", str(output)])
    def fail(*args, **kwargs): raise torch.cuda.OutOfMemoryError("synthetic OOM propagation test")
    monkeypatch.setattr(script, "profile", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic"):
        script.main()
    report = json.loads(output.read_text())
    assert report["status"] == "FAIL" and report["oom"] is True
    assert report["model_set"] == "auxiliary offset U-Net only"
    assert report["input_shape"] == [2, 3, 64, 96]
    assert report["result_class"] == "smoke-test result" and report["resolved_config"]["reproduction_mode"] == "approximate"
