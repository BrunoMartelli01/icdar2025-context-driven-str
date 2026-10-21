"""Exact checkpoint/RNG resume, isolation, denominator and failure boundaries."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import run_adaspot_gap as module
from scripts.run_adaspot_gap import DEFAULT_CONFIG, read_config, train_branch, training_config
from segmentation.training import seed_everything


@pytest.fixture
def config():
    value = read_config(DEFAULT_CONFIG, "approximate")[0]
    value.update(epochs=2, batch_size=2)
    return value


class Tiny(torch.nn.Module):
    branch = "control"
    architecture_identity = "unit-test-adaspot-resume"
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))


def prepare(monkeypatch):
    def train(model, batches, optimizer, device, config):
        loss = (model.weight - torch.rand(())).square()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        return {"loss": float(loss.detach()), "steps": 1, "samples": 2}
    def evaluate(model, loader, references, config, device):
        value = float(model.weight.detach())
        return {"aggregate": {"proposal_f1_iou_50": value, "gt_words_touched_by_fused_components": 0,
            "word_recall_iou_50": value, "foreground_iou": value, "gt_words": config["expected_validation_words"]}}
    monkeypatch.setattr(module, "train_gap_epoch", train)
    monkeypatch.setattr(module, "evaluate", evaluate)
    monkeypatch.setattr(module, "experiment_environment", lambda _: {"fixture": True})
    return SimpleNamespace(set_epoch=lambda _: None)


def invoke(path, model, config, setter, stop=None, contract=None):
    path.mkdir(exist_ok=True)
    optimizer = module.build_optimizer(model, config)
    return train_branch(path, model, optimizer, [], [], {}, setter, setter, config,
                        contract or {"fixture": True}, torch.device("cpu"), stop)


def test_resume_is_identical_with_rng_and_optimizer_restored(tmp_path, monkeypatch, config):
    setter = prepare(monkeypatch)
    seed_everything(12, deterministic_algorithms=True)
    continuous = Tiny(); full = invoke(tmp_path / "full", continuous, config, setter)
    seed_everything(12, deterministic_algorithms=True)
    partial = invoke(tmp_path / "resume", Tiny(), config, setter, stop=1)
    torch.rand(31)
    resumed = Tiny(); result = invoke(tmp_path / "resume", resumed, config, setter)
    assert partial["status"] == "IN_PROGRESS" and not partial["training_complete"]
    assert full["training_complete"] and result["training_complete"]
    assert torch.equal(continuous.weight, resumed.weight)


def test_checkpoint_log_interruption_recovers_without_repeating_training(tmp_path, monkeypatch, config):
    setter = prepare(monkeypatch); directory = tmp_path / "run"
    invoke(directory, Tiny(), config, setter, stop=1)
    (directory / "epochs.jsonl").unlink()
    result = invoke(directory, Tiny(), config, setter)
    rows = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2] and rows[0]["recovered_from_checkpoint"]
    assert result["completed_epochs"] == 2


def test_changed_resume_contract_and_changed_denominator_fail(tmp_path, monkeypatch, config):
    setter = prepare(monkeypatch); directory = tmp_path / "run"
    invoke(directory, Tiny(), config, setter, stop=1)
    with pytest.raises(ValueError, match="contract"):
        invoke(directory, Tiny(), config, setter, contract={"fixture": "changed"})
    monkeypatch.setattr(module, "evaluate", lambda *args: {"aggregate": {"gt_words": 0}})
    with pytest.raises(ValueError, match="denominator"):
        invoke(tmp_path / "bad_denominator", Tiny(), config, setter)


def test_config_isolation_and_true_training_metadata(config, tmp_path):
    with pytest.raises(ValueError, match="APPROXIMATE"):
        read_config(DEFAULT_CONFIG, "strict")
    with pytest.raises(ValueError, match="namespace"):
        read_config(tmp_path / "config.json", "approximate")
    assert asdict(training_config(config))["learning_rate"] == 1e-5
    assert asdict(training_config(config))["batch_size"] == 2
    assert "gap" in asdict(training_config(config))["loss"]


def test_unsupported_metric_metadata_fails_before_execution(monkeypatch):
    raw = json.loads(DEFAULT_CONFIG.read_text()); raw["metric_protocol"]["geometry"] = "official_polygon"
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw:
                        json.dumps(raw) if self == DEFAULT_CONFIG else original(self, *a, **kw))
    with pytest.raises(ValueError, match="Metric protocol"):
        read_config(DEFAULT_CONFIG, "approximate")


@pytest.mark.parametrize("key,value", [("gap_learning_rate", float("nan")), ("epochs", 0), ("precision", "float16")])
def test_bad_production_config_fails(monkeypatch, key, value):
    raw = json.loads(DEFAULT_CONFIG.read_text()); raw[key] = value
    original = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **kw:
                        json.dumps(raw) if self == DEFAULT_CONFIG else original(self, *a, **kw))
    with pytest.raises(ValueError):
        read_config(DEFAULT_CONFIG, "approximate")


def test_production_cpu_training_is_rejected_before_dataset_loading(tmp_path):
    args = SimpleNamespace(config=DEFAULT_CONFIG, mode="approximate", word_root=tmp_path / "data",
                           output_root=tmp_path / "run", device="cpu", action="train")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        module.run(args)


def test_new_head_uses_declared_lr_and_common_parameters_keep_control_lr(config):
    model = Tiny()
    model.gap = torch.nn.Linear(2, 1)
    optimizer = module.build_optimizer(model, config)
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 1e-4]


def test_invalid_source_checkpoint_fails_closed(tmp_path, config):
    (tmp_path / "best.pt").write_bytes(b"wrong weights")
    (tmp_path / "run_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        module.load_source_state(tmp_path, config)
