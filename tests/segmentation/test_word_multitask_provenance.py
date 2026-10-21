"""Contract serialization and complete launcher-stage boundary regressions."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_word_multitask import DEFAULT_CONFIG, read_config, run_contract
from segmentation.word_multitask_provenance import (
    LEGACY_RUNNER_SHA256, PROVENANCE_SOURCE, RUNNER_SOURCE,
    ensure_branch_contract, json_contract, reusable_smoke,
)


def current_contract():
    cfg, source = read_config(DEFAULT_CONFIG, "approximate")
    return run_contract(cfg, source, "text", "test-encoder-sha", cfg["validation_ids_sha256"])


def legacy_smoke(directory, current):
    old = deepcopy(current)
    old["source_sha256"][RUNNER_SOURCE] = LEGACY_RUNNER_SHA256
    old["source_sha256"].pop(PROVENANCE_SOURCE)
    directory.mkdir(exist_ok=True)
    (directory / "run_contract.json").write_text(json.dumps(old, indent=2) + "\n")
    report = {"result_class": "smoke-test result", "run_contract": old,
              "train": {"steps": 1, "samples": old["config"]["batch_size"]},
              "memory": {"oom": False, "device": "cpu"}}
    (directory / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    return old


def test_actual_contract_is_equal_after_json_roundtrip():
    contract = current_contract()
    assert contract["training_config"]["adam_betas"] == [0.9, 0.999]
    assert json.loads(json.dumps(contract)) == contract
    assert json_contract({"nested": {"betas": (0.9, 0.999)}}) == {"nested": {"betas": [0.9, 0.999]}}
    with pytest.raises(ValueError):
        json_contract({"nonfinite": float("nan")})


def test_same_contract_stages_do_not_rewrite_provenance(tmp_path):
    contract = current_contract()
    ensure_branch_contract(tmp_path, contract)
    original = (tmp_path / "run_contract.json").read_bytes()
    for _ in range(3):
        ensure_branch_contract(tmp_path, contract)
    assert (tmp_path / "run_contract.json").read_bytes() == original
    assert not (tmp_path / "contract_transition.json").exists()


def test_known_legacy_smoke_is_archived_and_reused_without_relabeling(tmp_path):
    contract = current_contract()
    old = legacy_smoke(tmp_path, contract)
    old_bytes = (tmp_path / "run_contract.json").read_bytes()
    smoke_bytes = (tmp_path / "smoke.json").read_bytes()
    ensure_branch_contract(tmp_path, contract)
    assert (tmp_path / "smoke_run_contract.json").read_bytes() == old_bytes
    assert (tmp_path / "smoke.json").read_bytes() == smoke_bytes
    assert json.loads((tmp_path / "run_contract.json").read_text()) == contract
    assert reusable_smoke(tmp_path, contract, "cpu")["run_contract"] == old
    receipt = (tmp_path / "contract_transition.json").read_bytes()
    ensure_branch_contract(tmp_path, contract)
    assert (tmp_path / "contract_transition.json").read_bytes() == receipt
    with pytest.raises(ValueError, match="different device"):
        reusable_smoke(tmp_path, contract, "cuda:0")


@pytest.mark.parametrize("change", ["loss", "source", "unknown_runner", "training", "wrong_smoke", "missing_smoke"])
def test_legacy_repair_rejects_scientific_changes_or_training_artifacts(tmp_path, change):
    contract = current_contract()
    old = legacy_smoke(tmp_path, contract)
    if change == "loss":
        old["config"]["loss"]["lambda_center"] = 2.0
    elif change == "source":
        old["source_sha256"]["segmentation/word_multitask.py"] = "unknown"
    elif change == "unknown_runner":
        old["source_sha256"][RUNNER_SOURCE] = "unknown"
    elif change == "training":
        (tmp_path / "last.pt").write_bytes(b"training-artifact-fixture")
    elif change == "wrong_smoke":
        report = json.loads((tmp_path / "smoke.json").read_text())
        report["run_contract"]["variant"] = "text_kernel"
        (tmp_path / "smoke.json").write_text(json.dumps(report))
    else:
        (tmp_path / "smoke.json").unlink()
    (tmp_path / "run_contract.json").write_text(json.dumps(old))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        ensure_branch_contract(tmp_path, contract)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_contract_repair_recovers_after_archive_before_active_commit(tmp_path, monkeypatch):
    import segmentation.word_multitask_provenance as provenance
    contract = current_contract()
    old = legacy_smoke(tmp_path, contract)
    write = provenance._json
    def interrupt(path, value):
        if path.name == "run_contract.json":
            raise RuntimeError("interrupted before contract commit")
        write(path, value)
    monkeypatch.setattr(provenance, "_json", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        ensure_branch_contract(tmp_path, contract)
    assert json.loads((tmp_path / "run_contract.json").read_text()) == old
    monkeypatch.setattr(provenance, "_json", write)
    ensure_branch_contract(tmp_path, contract)
    assert json.loads((tmp_path / "run_contract.json").read_text()) == contract
    assert reusable_smoke(tmp_path, contract, "cpu") is not None


@pytest.mark.parametrize("artifact", ["smoke.json", "contract_transition.json"])
def test_legacy_smoke_tamper_after_repair_fails(tmp_path, artifact):
    contract = current_contract()
    legacy_smoke(tmp_path, contract)
    ensure_branch_contract(tmp_path, contract)
    path = tmp_path / artifact
    if artifact == "smoke.json":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        receipt = json.loads(path.read_text())
        receipt["classification"] = "PAPER_SPECIFIED"
        path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="checksum"):
        reusable_smoke(tmp_path, contract, "cpu")


def test_complete_smoke_train_evaluate_and_reuse_stages(tmp_path, monkeypatch):
    """Exercise actual run(), optimizer/checkpoint I/O and exports on a tiny CPU fixture."""
    import scripts.run_word_multitask as runner
    cfg, source = read_config(DEFAULT_CONFIG, "approximate")
    cfg = deepcopy(cfg)
    cfg.update(epochs=1, num_workers=0, expected_validation_words=8)
    root, output = tmp_path / "data", tmp_path / "run"
    (root / "manifests").mkdir(parents=True)
    manifest_path = root / cfg["train_manifest"]
    manifest_path.write_text("synthetic launcher-boundary fixture\n")
    cfg["train_manifest_sha256"] = sha256_file(manifest_path)
    encoder_path = tmp_path / "encoder.pt"
    torch.save(torch.nn.Linear(1, 1).state_dict(), encoder_path)
    cfg["encoder_sha256_prefix"] = sha256_file(encoder_path)[:8]
    samples = []
    for index in range(16):
        image_path = "image{}.png".format(index)
        Image.new("RGB", (32, 32)).save(root / image_path)
        samples.append(SimpleNamespace(sample_id="fixture:{}".format(index), image_path=image_path))
    manifest = SimpleNamespace(dataset="total_text", split_role="train", samples=samples)
    owners = np.zeros((16, 16), np.int32)
    owners[4:12, 4:12] = 1
    target = torch.from_numpy(owners > 0)[None].float()
    kernel = torch.zeros_like(target); kernel[:, 6:10, 6:10] = 1
    center = kernel * 0.3; center[:, 8, 8] = 1
    class Dataset(torch.utils.data.Dataset):
        def __len__(self): return 16
        def __getitem__(self, index):
            return {"image": torch.ones(3, 32, 32), "mask": target.clone(), "kernel": kernel.clone(),
                    "center": center.clone(), "validity": torch.ones_like(target),
                    "separation": torch.zeros_like(target), "sample_id": samples[index].sample_id}
    class Transform:
        def set_epoch(self, epoch): self.epoch = epoch
    class Model(torch.nn.Module):
        def __init__(self, variant):
            super().__init__()
            self.encoder = torch.nn.Linear(1, 1)
            self.head = torch.nn.Conv2d(3, runner.VARIANTS[variant], 1, stride=2)
            self.architecture_identity = "synthetic-stage-fixture-" + variant
        def forward(self, image): return self.head(image)
    monkeypatch.setattr(runner, "read_config", lambda *args: (deepcopy(cfg), deepcopy(source)))
    monkeypatch.setattr(runner, "load_segmentation_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(runner, "_split_indices", lambda *args: (list(range(8)), list(range(8, 16)), cfg["validation_ids_sha256"]))
    monkeypatch.setattr(runner, "_transform", lambda *args, **kwargs: Transform())
    monkeypatch.setattr(runner, "SegmentationDataset", lambda *args, **kwargs: Dataset())
    monkeypatch.setattr(runner, "WordMultitaskDataset", lambda base, **kwargs: base)
    monkeypatch.setattr(runner, "_validation_reference", lambda *args: {
        s.sample_id: (owners, (1,), np.ones((16, 16), bool)) for s in samples[8:]})
    monkeypatch.setattr(runner, "build_word_model", lambda variant, **kwargs: Model(variant))
    args = SimpleNamespace(mode="approximate", config=DEFAULT_CONFIG, target_root=root, output_root=output,
        encoder_checkpoint=encoder_path, encoder_sha256=sha256_file(encoder_path), num_workers=None,
        device="cpu", variant="all", action="smoke")
    for action in ("smoke", "train", "evaluate", "smoke", "train", "evaluate"):
        args.action = action
        result = runner.run(args)
        assert result["branches"] == 3
    for variant in runner.VARIANTS:
        branch = output / variant
        contract = json.loads((branch / "run_contract.json").read_text())
        payload = torch.load(branch / "best.pt", weights_only=False)
        assert contract["training_config"]["adam_betas"] == [0.9, 0.999]
        assert payload["run_contract"] == contract
        assert payload["training_config"]["adam_betas"] == (0.9, 0.999)
        assert payload["epoch"] == 1
        assert json.loads((branch / "validation.json").read_text())["aggregate"]["gt_words"] == 8
