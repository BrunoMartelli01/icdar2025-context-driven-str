"""Phase-aware resume, config gating, source qualification and frozen OCR."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from scripts.run_unet_word_research_v2 import (DEFAULT_CONFIG, read_config, train_branch,
                                            training_config, load_source_state)
from segmentation.unet_word_research_v2 import build_research_v2_optimizer
from segmentation.training import seed_everything


@pytest.fixture
def cfg():
    c = deepcopy(read_config(DEFAULT_CONFIG, "approximate")[0])
    c.update(epochs=3, batch_size=2)
    c["glyph"].update(warmup_epochs=1, ramp_epochs=1)
    return c


class PhaseModel(torch.nn.Module):
    branch = "glyph_staged"
    architecture_identity = "test-corrective-phase-model"

    def __init__(self):
        super().__init__()
        self.word = torch.nn.Parameter(torch.tensor(.2))
        self.glyph_private = torch.nn.Parameter(torch.tensor(.1))
        self.original_trainability = {"word": True}
        self.research_epoch = 0
        self.set_research_epoch(0)

    def set_research_epoch(self, epoch):
        self.research_epoch = epoch
        self.word.requires_grad_(epoch > 1)


def install_training(monkeypatch):
    def train(model, loader, optimizer, device, config):
        loss = (model.glyph_private - torch.rand(())).square()
        if model.word.requires_grad:
            loss = loss + (model.word - torch.rand(())).square()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        return {"loss": float(loss.detach()), "samples": 2, "steps": 1, "phase_epoch": model.research_epoch}

    def evaluate(model, *args, **kwargs):
        score = float(model.word.detach() + model.glyph_private.detach())
        return {"aggregate": {"proposal_f1_iou_50": score, "gt_words_touched_by_fused_components": 0,
            "word_recall_iou_50": score, "foreground_iou": score, "gt_words": 983}}

    monkeypatch.setattr("scripts.run_unet_word_research_v2.train_research_epoch", train)
    monkeypatch.setattr("scripts.run_unet_word_research_v2.evaluate", evaluate)
    monkeypatch.setattr("scripts.run_unet_word_research_v2.experiment_environment", lambda _: {"kind": "test"})
    return SimpleNamespace(set_epoch=lambda _: None)


def invoke(path, model, cfg, setter, stop=None, contract=None):
    path.mkdir(exist_ok=True)
    optimizer = build_research_v2_optimizer(model, cfg)
    return train_branch(path, model, optimizer, [], [], {}, setter, setter, cfg,
                        contract or {"test": True}, torch.device("cpu"), stop)


def test_resume_crosses_frozen_to_joint_phase_with_identical_state(tmp_path, monkeypatch, cfg):
    setter = install_training(monkeypatch)
    seed_everything(19, deterministic_algorithms=True); full = PhaseModel()
    result = invoke(tmp_path / "full", full, cfg, setter)
    seed_everything(19, deterministic_algorithms=True); partial = PhaseModel()
    paused = invoke(tmp_path / "resume", partial, cfg, setter, stop=1)
    assert paused["status"] == "IN_PROGRESS" and partial.word == .2
    torch.rand(23); resumed = PhaseModel()
    finished = invoke(tmp_path / "resume", resumed, cfg, setter)
    assert finished["training_complete"] and result["completed_epochs"] == 3
    assert torch.equal(full.word, resumed.word) and torch.equal(full.glyph_private, resumed.glyph_private)
    history = [json.loads(x) for x in (tmp_path / "resume/epochs.jsonl").read_text().splitlines()]
    assert [r["train"]["phase_epoch"] for r in history] == [1, 2, 3]


def test_phase_checkpoint_before_log_recovery_and_contract_rejection(tmp_path, monkeypatch, cfg):
    setter = install_training(monkeypatch)
    path = tmp_path / "run"; invoke(path, PhaseModel(), cfg, setter, stop=1)
    (path / "epochs.jsonl").unlink()
    finished = invoke(path, PhaseModel(), cfg, setter)
    rows = [json.loads(x) for x in (path / "epochs.jsonl").read_text().splitlines()]
    assert rows[0]["recovered_from_checkpoint"] and finished["completed_epochs"] == 3
    with pytest.raises(ValueError, match="contract"):
        invoke(path, PhaseModel(), cfg, setter, contract={"test": "changed"})


def test_source_checksum_and_architecture_are_not_relaxed(tmp_path, cfg):
    (tmp_path / "best.pt").write_bytes(b"wrong checkpoint")
    (tmp_path / "run_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_source_state(tmp_path, cfg)


def test_frozen_ocr_absolute_checkpoint_boundary_cache_and_resume(tmp_path, monkeypatch, cfg):
    import scripts.evaluate_unet_word_research_v2_ocr as m
    from scripts.prune_segmentation_checkpoints import sha256_file as digest
    from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
    from PIL import Image

    root, output = tmp_path / "data", tmp_path / "runs"
    root.mkdir(); output.mkdir()
    manifest_path = root / "manifest.yaml"; manifest_path.write_text("fixture")
    cfg.update(train_manifest="manifest.yaml", word_manifest_sha256=digest(manifest_path),
               validation_count=2, expected_validation_words=2)
    samples = []
    for i in range(2):
        p = root / f"{i}.png"; Image.fromarray(np.full((12, 20, 3), 40 + i, np.uint8)).save(p)
        samples.append(SimpleNamespace(sample_id=f"total_text:{i}", image_path=Path(p.name), image_sha256=digest(p)))
    monkeypatch.setattr(m, "read_config", lambda *_: (cfg, {}))
    monkeypatch.setattr(m, "load_segmentation_manifest", lambda *_, **__: SimpleNamespace(dataset="total_text", split_role="train", samples=samples))
    monkeypatch.setattr(m, "_split_indices", lambda *_: ([], [0, 1], cfg["validation_ids_sha256"]))
    monkeypatch.setattr(m, "_train_gt", lambda *_: ([{"bbox": [0, 0, 20, 12], "text": "word"}], "fixture"))
    weights, archive, source = tmp_path / "weights", tmp_path / "archive", tmp_path / "source.pt"
    source.write_bytes(b"source checkpoint")
    monkeypatch.setattr(m, "sha256_file", lambda p: PARSEQ_CHECKPOINT_SHA256 if Path(p) == weights
        else PARSEQ_RUNTIME_ARCHIVE_SHA256 if Path(p) == archive else digest(Path(p)))
    created = []
    class Recognizer:
        def __init__(self, config): self.config = config; created.append(True)
        def recognize_candidate_batch(self, images):
            return [SimpleNamespace(to_dict=lambda: {"raw_text": "word"}) for _ in images]
        def release(self): return {}
    monkeypatch.setattr(m, "ParseqApproxRecognizer", Recognizer)
    for arm in cfg["embedding_arms"]:
        directory = output / arm; directory.mkdir()
        (directory / "run_manifest.json").write_text(json.dumps({"status": "SUCCESS", "training_complete": True,
            "run_contract": {"test": True}, "checkpoint": {"path": str(source), "sha256": digest(source)}}))
        rows = []
        for sample in samples:
            folder = directory / "validation_exports" / sample.sample_id.replace(":", "__"); folder.mkdir(parents=True)
            p = folder / "proposals.json"
            p.write_text(json.dumps({"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "proposals": [{"instance_id": 1, "bbox": [0, 0, 20, 12], "foreground_area": 240}], "quality_scores": {}}))
            rows.append({"sample_id": sample.sample_id, "proposals_sha256": digest(p), "ignored_instance_ids": []})
        (directory / "validation.json").write_text(json.dumps({"run_contract": {"test": True},
            "checkpoint_sha256": digest(source), "per_image": rows}))
    args = SimpleNamespace(config=DEFAULT_CONFIG, mode="approximate", subset="postprocess", word_root=root,
        run_root=output, parseq_checkpoint=weights, parseq_source_archive=archive, parseq_source=tmp_path / "runtime",
        batch_size=2, device="cpu")
    first = m.run(args); repeated = m.run(args)
    assert first["arms"] == repeated["arms"] and len(created) == 1
    assert first["protocol"]["comparison_baseline"] == "embedding50_hard"
    assert all(x["aggregate"]["exact_casefold_iou_ge_50"] == 2 for x in first["arms"].values())
    args.batch_size = 1
    with pytest.raises(ValueError, match="backend"):
        m.run(args)
