from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import scripts.train_word_center_offsets as runner
from segmentation.training import build_optimizer, seed_everything
from segmentation.word_center_offsets import make_word_center_offsets


def batch():
    owners = np.zeros((8, 12), np.int32)
    owners[1:7, 2:10] = 1
    target, _ = make_word_center_offsets(owners, np.ones_like(owners, bool))
    generator = torch.Generator().manual_seed(42)
    return {"image": torch.randn(2, 3, 8, 12, generator=generator), "sample_id": ["word:1", "word:2"],
            "owners_native": torch.from_numpy(owners)[None].repeat(2, 1, 1),
            "center_offsets": torch.from_numpy(target)[None].repeat(2, 1, 1, 1),
            "validity": torch.ones(2, 1, 8, 12), "expected_word_count": torch.tensor([1, 1])}


def config():
    cfg, _ = runner.read_config(runner.DEFAULT_CONFIG, "approximate")
    cfg["training"].update(epochs=3, batch_size=2)
    cfg["dataset"].update(images=4, validation_images=2, validation_expected_words=2)
    return cfg


class Tiny(torch.nn.Conv2d):
    architecture_identity = "synthetic-offset-launcher-test"

    def __init__(self):
        super().__init__(3, 2, 1)


class EpochState:
    def set_epoch(self, value): self.epoch = value


def execute(path, cfg, *, stop_after_epoch=None):
    seed_everything(37, deterministic_algorithms=True)
    model = Tiny()
    optimizer = build_optimizer(model, runner.training_config(cfg))
    data = batch()
    path.mkdir(exist_ok=True)
    result = runner.train(path, model, optimizer, [data], [data], EpochState(), EpochState(),
        cfg, {"variant": "synthetic_center_offsets", "config": cfg}, torch.device("cpu"), stop_after_epoch=stop_after_epoch)
    return result, model


def test_configuration_strict_namespace_and_real_protocol():
    cfg, _ = runner.read_config(runner.DEFAULT_CONFIG, "approximate")
    assert cfg["training"]["epochs"] == 100 and cfg["training"]["batch_size"] == 8
    assert runner.training_config(cfg).checkpoint_selection == "validation_offset_loss"
    with pytest.raises(ValueError, match="approximate"):
        runner.read_config(runner.DEFAULT_CONFIG, "strict")
    with pytest.raises(ValueError, match="configs/approximate"):
        runner.read_config(Path("/tmp/not_a_training_profile.json"), "approximate")


@pytest.mark.parametrize("problem", ["batch", "epochs", "beta"])
def test_invalid_training_settings_fail(problem):
    cfg = config()
    if problem == "batch": cfg["training"]["batch_size"] = 0
    if problem == "epochs": cfg["training"]["epochs"] = 0
    if problem == "beta": cfg["loss"]["beta"] = 0
    with pytest.raises(ValueError): runner.training_config(cfg)


def test_validation_reference_endpoint_scaling_and_empty_image_counts(tmp_path):
    data = batch()
    class Ideal(torch.nn.Module):
        def forward(self, image): return torch.atanh(data["center_offsets"])
    data["owners_native"][1] = 0
    data["expected_word_count"][1] = 1
    result = runner.evaluate_offsets(Ideal(), [data], torch.device("cpu"), beta=.01, export_dir=tmp_path)
    stats = result["aggregate"]
    assert stats["offset_loss"] < 1e-12 and stats["endpoint_error_native_pixels"] < 1e-6
    assert stats["samples"] == 2 and stats["supervised_images"] == 1 and stats["empty_images"] == 1
    assert stats["expected_words"] == 2 and stats["surviving_words"] == 1 and stats["lost_words"] == 1
    assert result["per_image"][1]["offset_loss"] is None
    with np.load(tmp_path / "word__1/offsets.npz") as exported:
        assert exported["normalized_offsets"].shape == (2, 8, 12)
        assert np.allclose(exported["normalized_offsets"], data["center_offsets"][0])


def test_partial_budget_resume_matches_uninterrupted_training(tmp_path):
    cfg = config()
    full, full_model = execute(tmp_path / "full", cfg)
    partial, _ = execute(tmp_path / "resume", cfg, stop_after_epoch=1)
    assert partial["status"] == "IN_PROGRESS" and partial["completed_epochs"] == 1
    assert not (tmp_path / "resume/run_manifest.json").exists()
    resumed, model = execute(tmp_path / "resume", cfg)
    assert full["validation"] == resumed["validation"] and resumed["completed_epochs"] == 3
    assert full["checkpoint"]["epoch"] == resumed["checkpoint"]["epoch"]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in full_model.state_dict().items())
    record = torch.load(tmp_path / "resume/last.pt", weights_only=False)
    assert record["training_config"]["loss"] == "word_balanced_center_offset_SmoothL1"
    assert record["training_config"]["batch_size"] == 2


def test_interruption_after_checkpoint_recovers_log_without_retraining(tmp_path, monkeypatch):
    cfg = config()
    full, full_model = execute(tmp_path / "full", cfg)
    append = runner._append_epoch
    def interrupt(path, row): raise RuntimeError("synthetic interruption after atomic checkpoint")
    monkeypatch.setattr(runner, "_append_epoch", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        execute(tmp_path / "resume", cfg)
    monkeypatch.setattr(runner, "_append_epoch", append)
    resumed, model = execute(tmp_path / "resume", cfg)
    assert full["validation"] == resumed["validation"]
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in full_model.state_dict().items())
    rows = runner._epoch_log(tmp_path / "resume/epochs.jsonl")
    assert rows[0]["recovered_from_checkpoint"] is True
    assert [row["epoch"] for row in rows] == [1, 2, 3]


def test_resume_rejects_incompatible_provenance(tmp_path):
    cfg = config()
    execute(tmp_path / "run", cfg, stop_after_epoch=1)
    changed = deepcopy(cfg); changed["loss"]["beta"] = .02
    with pytest.raises(ValueError, match="contract"):
        execute(tmp_path / "run", changed)


def test_loss_ties_keep_earliest_checkpoint(tmp_path, monkeypatch):
    cfg = config()
    evaluator = runner.evaluate_offsets
    def tied(*args, **kwargs):
        result = evaluator(*args, **kwargs)
        result["aggregate"]["offset_loss"] = .5
        result["aggregate"]["endpoint_error_native_pixels"] = 1.
        return result
    monkeypatch.setattr(runner, "evaluate_offsets", tied)
    result, _ = execute(tmp_path / "run", cfg)
    assert result["checkpoint"]["epoch"] == 1


def test_cli_records_and_propagates_oom(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["train", "--mode", "approximate", "--target-root", str(tmp_path),
        "--output-root", str(tmp_path), "--device", "cpu", "--action", "smoke"])
    def fail(args):
        args._owns_output = True
        args._resolved_cfg, _ = runner.read_config(runner.DEFAULT_CONFIG, "approximate")
        raise torch.cuda.OutOfMemoryError("synthetic launcher OOM")
    monkeypatch.setattr(runner, "run", fail)
    with pytest.raises(torch.cuda.OutOfMemoryError): runner.main()
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["oom"] is True and failure["precision"] == "float32"
    assert failure["batch_size"] == 8 and failure["result_class"] == "smoke-test result"
