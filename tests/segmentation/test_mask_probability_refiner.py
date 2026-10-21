import inspect

import numpy as np
import pytest
import torch

from models.mask_probability_refiner import MaskProbabilityRefiner
from segmentation.mask_probability_refiner import (paired_transform, corrupt_probabilities,
    refiner_loss, RefinerDataset, pixel_counts)
from scripts.run_mask_probability_refiner import read_config, refine_maps, validate_population, REPO


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("architecture,steps", [("shallow", 1), ("dilated", 1), ("unet", 1), ("unet", 3)])
def test_exact_identity_including_saturated_and_threshold_values_and_trainability(architecture, steps):
    torch.manual_seed(17)
    model = MaskProbabilityRefiner(architecture, width=8, steps=steps)
    rgb, base = torch.rand(2, 3, 32, 32), torch.rand(2, 2, 32, 32)
    base[:, :, :1, :4] = torch.tensor([0., 1., .5, .65])
    canvas = torch.ones(2, 1, 32, 32, dtype=torch.bool)
    outputs = model(rgb, base, canvas)
    assert all(torch.equal(o["probabilities"], base) for o in outputs)
    target = (base < .5).float()
    loss = refiner_loss(outputs, base, target, canvas)
    loss.backward()
    assert model.head.weight.grad.norm() > 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    optimizer.step()
    optimizer.zero_grad()
    after = model(rgb, base, canvas)
    assert not torch.equal(after[-1]["probabilities"], base)
    assert torch.isfinite(after[-1]["probabilities"]).all()
    refiner_loss(after, base, target, canvas).backward()
    assert any(p.grad is not None and p.grad.norm() > 0 for p in model.features.parameters())
    replacement = MaskProbabilityRefiner(architecture, width=8, steps=steps)
    replacement.load_state_dict(model.state_dict(), strict=True)
    assert torch.equal(replacement(rgb, base, canvas)[-1]["probabilities"], after[-1]["probabilities"])


def test_exact_paired_augmentation_and_deterministic_sampling():
    row = {"rgb": torch.arange(3*32*32).reshape(3, 32, 32).to(torch.uint8),
           "probabilities": torch.rand(2, 32, 32), "target": torch.rand(2, 32, 32) > .5,
           "validity": torch.ones(1, 32, 32, dtype=torch.bool), "canvas": torch.ones(1, 32, 32, dtype=torch.bool)}
    transformed = paired_transform(row, flip=True, quarter_turns=1)
    assert all(torch.equal(v, torch.rot90(row[k].flip(-1), 1, (-2, -1))) for k, v in transformed.items())
    left = RefinerDataset({"a": row}, ["a"], seed=1, epoch=3, corruption_probability=1.)[0]
    right = RefinerDataset({"a": row}, ["a"], seed=1, epoch=3, corruption_probability=1.)[0]
    assert all(torch.equal(left[k], right[k]) for k in row)
    assert left["corrupted"] and row["rgb"].dtype == torch.uint8


def test_ignore_pixels_do_not_influence_loss_or_gradient():
    base, target = torch.rand(1, 2, 16, 16), torch.zeros(1, 2, 16, 16)
    target[:, :, 4:8, 4:8] = 1
    valid = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    valid[:, :, :3] = False
    a = torch.randn_like(base, requires_grad=True)
    b = a.detach().clone()
    b[:, :, :3] += 50
    b.requires_grad_()
    def loss(logits):
        return refiner_loss([{"logits": logits, "probabilities": logits.sigmoid()}], base, target, valid, error_focus=True)
    loss_a, loss_b = loss(a), loss(b)
    assert torch.equal(loss_a, loss_b)
    loss_a.backward()
    loss_b.backward()
    assert torch.equal(a.grad, b.grad)
    assert not a.grad[:, :, :3].any()


@pytest.mark.parametrize("foreground", [False, True])
def test_missing_classes_finite_and_all_invalid_explicit_failure(foreground):
    p = torch.full((1, 2, 16, 16), .5, requires_grad=True)
    valid = torch.ones(1, 1, 16, 16, dtype=torch.bool)
    output = [{"logits": torch.logit(p), "probabilities": p}]
    target = torch.full_like(p, float(foreground))
    loss = refiner_loss(output, p.detach(), target, valid)
    assert torch.isfinite(loss)
    loss.backward()
    assert p.grad.norm() > 0
    with pytest.raises(ValueError, match="No valid supervision"):
        refiner_loss(output, p.detach(), target, valid & False)


def test_inference_interface_does_not_accept_gt_and_baseline_is_exact():
    assert list(inspect.signature(refine_maps).parameters) == ["model", "rgb", "probabilities", "canvas", "device"]
    p = torch.rand(2, 16, 16)
    assert np.array_equal(refine_maps(None, None, p, None, "cpu"), p.numpy())


def test_pixels_preserve_threshold_boundaries_ignored_pixels_and_empty_cases():
    p = np.asarray([[[.65, .64, 1]], [[.5, .49, 1]]])
    target = np.asarray([[[1, 0, 0]], [[1, 0, 0]]], dtype=bool)
    counts = pixel_counts(p, target, np.asarray([[1, 1, 0]], dtype=bool), [.65, .5])
    assert counts == [{"tp": 1, "fp": 0, "fn": 0}]*2


def test_approximate_namespace_and_population_enforcement(tmp_path):
    config = REPO/"configs/approximate/mask_probability_refiner_v1.json"
    c = read_config(config, "approximate")
    assert c["epochs"] == 80 and len(c["arms"]) == 6
    with pytest.raises(ValueError, match="namespace"):
        read_config(config, "strict")
    p = tmp_path/"config.json"
    p.write_text(config.read_text())
    with pytest.raises(ValueError, match="namespace"):
        read_config(p, "approximate")
    with pytest.raises(RuntimeError, match="denominator"):
        validate_population({"per_image": [{"sample_id": "x"}], "aggregate": {"gt_words": 982}}, 983, 1)


def test_corruption_uses_predicted_maps_and_preserves_range():
    p = torch.zeros(2, 16, 16)
    p[:, 6:10, 6:10] = .9
    outputs = [corrupt_probabilities(p, np.random.default_rng(i)) for i in range(20)]
    assert any(o.sum() > p.sum() for o in outputs)
    assert any(o.sum() < p.sum() for o in outputs)
    assert all(o.shape == p.shape and o.min() >= 0 and o.max() <= 1 for o in outputs)


def test_interrupted_resume_equals_uninterrupted_training(tmp_path, monkeypatch):
    from scripts import run_mask_probability_refiner as runner
    c = read_config(REPO/"configs/approximate/mask_probability_refiner_v1.json", "approximate")
    c.update(epochs=2, calibration_epochs=[0, 1, 2], batch_size=1)
    torch.manual_seed(31)
    records = {sid: {"rgb": torch.randint(256, (3, 16, 16), dtype=torch.uint8),
        "probabilities": torch.rand(2, 16, 16), "target": torch.rand(2, 16, 16) > .5,
        "validity": torch.ones(1, 16, 16, dtype=torch.bool), "canvas": torch.ones(1, 16, 16, dtype=torch.bool)}
        for sid in ("one", "two")}
    splits = {"fit": ["one", "two"], "calibration": [], "evaluation": []}
    baseline = {"aggregate": {"ocr_correct": 0, "ocr_hmean": 0., "geometry_hmean": 0.}}
    monkeypatch.setattr(runner, "evaluate", lambda *a, **kw: baseline)
    monkeypatch.setattr(runner, "memory", lambda *a: {})
    monkeypatch.setattr(runner, "status", lambda *a, **kw: None)
    complete, resumed = tmp_path/"complete", tmp_path/"resumed"
    for root in (complete, resumed):
        root.mkdir()
        (root/"run_contract.json").write_text('{"fixture": true}\n')
    def train(root):
        return runner.train_arm(None, records, splits, "shallow_balanced", None, c,
                                torch.device("cpu"), root, baseline, tmp_path, False)
    train(complete)
    def interrupt(root, stage, **details):
        if details.get("epoch") == 1 and stage == "training":
            raise RuntimeError("simulated interruption after durable checkpoint")
    monkeypatch.setattr(runner, "status", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        train(resumed)
    monkeypatch.setattr(runner, "status", lambda *a, **kw: None)
    train(resumed)
    left = torch.load(complete/"shallow_balanced/last.pt", weights_only=False)
    right = torch.load(resumed/"shallow_balanced/last.pt", weights_only=False)
    assert left["epoch"] == right["epoch"] == 2
    assert left["total_steps"] == right["total_steps"] == 4
    assert left["scheduler"] == right["scheduler"]
    assert all(torch.equal(left["model"][k], right["model"][k]) for k in left["model"])
    assert [r["mean_loss"] for r in left["history"]] == [r["mean_loss"] for r in right["history"]]
