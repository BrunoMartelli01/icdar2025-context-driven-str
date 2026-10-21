"""Resume, provenance, empty outputs and OCR protocol qualification."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.evaluate_unet_word_research_ocr import cache_record, cached_text, score_predictions
from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from scripts.run_unet_word_research import DEFAULT_CONFIG, load_source_state, read_config, train_branch, training_config
from segmentation.training import build_optimizer, seed_everything
from segmentation.unet_word_research import train_research_epoch


@pytest.fixture
def cfg():
    value = read_config(DEFAULT_CONFIG, "approximate")[0]
    value.update(epochs=2, batch_size=2)
    return value


class TinyModel(torch.nn.Module):
    branch = "control"
    architecture_identity = "unit-test-unet-research"

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))


def prepare_training(monkeypatch, cfg):
    def train(model, loader, optimizer, device, config):
        loss = (model.weight - torch.rand(())).square()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        return {"loss": float(loss.detach()), "steps": 1, "samples": 2}

    def evaluate(model, loader, references, config, device):
        value = float(model.weight.detach())
        return {"aggregate": {"proposal_f1_iou_50": value, "gt_words_touched_by_fused_components": 0,
            "word_recall_iou_50": value, "foreground_iou": value, "gt_words": config["expected_validation_words"]}}

    monkeypatch.setattr("scripts.run_unet_word_research.train_research_epoch", train)
    monkeypatch.setattr("scripts.run_unet_word_research.evaluate", evaluate)
    monkeypatch.setattr("scripts.run_unet_word_research.experiment_environment", lambda _: {"kind": "unit test"})
    return SimpleNamespace(set_epoch=lambda _: None)


def invoke(directory, model, cfg, setter, stop=None, contract=None):
    directory.mkdir(exist_ok=True)
    optimizer = build_optimizer(model, training_config(cfg, "control"))
    return train_branch(directory, model, optimizer, [], [], {}, setter, setter, cfg,
                        contract or {"unit_test": True}, torch.device("cpu"), stop)


def test_partial_resume_preserves_rng_optimizer_and_exact_final_state(tmp_path, monkeypatch, cfg):
    setter = prepare_training(monkeypatch, cfg)
    seed_everything(12, deterministic_algorithms=True)
    continuous = TinyModel()
    full = invoke(tmp_path / "full", continuous, cfg, setter)
    seed_everything(12, deterministic_algorithms=True)
    interrupted = TinyModel()
    partial = invoke(tmp_path / "resume", interrupted, cfg, setter, stop=1)
    assert partial["status"] == "IN_PROGRESS" and not partial["training_complete"]
    torch.rand(37)  # Simulate RNG consumption by a fresh execution environment.
    resumed = TinyModel()
    finished = invoke(tmp_path / "resume", resumed, cfg, setter)
    assert full["training_complete"] and finished["training_complete"]
    assert torch.equal(resumed.weight, continuous.weight)
    assert finished["completed_epochs"] == 2


def test_checkpoint_before_log_recovers_without_repeating_epoch(tmp_path, monkeypatch, cfg):
    setter = prepare_training(monkeypatch, cfg)
    directory = tmp_path / "run"
    partial = invoke(directory, TinyModel(), cfg, setter, stop=1)
    (directory / "epochs.jsonl").unlink()
    finished = invoke(directory, TinyModel(), cfg, setter)
    rows = [json.loads(line) for line in (directory / "epochs.jsonl").read_text().splitlines()]
    assert [r["epoch"] for r in rows] == [1, 2]
    assert rows[0]["recovered_from_checkpoint"] is True and finished["completed_epochs"] == 2


def test_resume_rejects_changed_contract_and_noncontiguous_history(tmp_path, monkeypatch, cfg):
    setter = prepare_training(monkeypatch, cfg)
    directory = tmp_path / "run"
    invoke(directory, TinyModel(), cfg, setter, stop=1)
    with pytest.raises(ValueError, match="contract"):
        invoke(directory, TinyModel(), cfg, setter, contract={"unit_test": "changed"})
    row = json.loads((directory / "epochs.jsonl").read_text()); row["epoch"] = 5
    (directory / "epochs.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Noncontiguous"):
        invoke(directory, TinyModel(), cfg, setter)


def test_source_artifact_hash_mismatch_is_explicit(tmp_path, cfg):
    (tmp_path / "best.pt").write_bytes(b"not qualified")
    (tmp_path / "run_manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        load_source_state(tmp_path, cfg)


def test_empty_recognition_retains_all_gt_words_and_duplicate_predictions_match_once():
    gt = [{"bbox": [0, 0, 10, 10], "text": "abc"}, {"bbox": [20, 0, 30, 10], "text": "def"}]
    metrics, matched = score_predictions([], gt)
    assert metrics["gt_words"] == 2 and metrics["exact_casefold_iou_ge_50"] == 0
    predictions = [{"bbox": [0, 0, 10, 10], "text": "ABC"}] * 2
    metrics, matched = score_predictions(predictions, gt)
    assert metrics["proposals"] == 2 and metrics["exact_casefold_iou_ge_50"] == 1
    assert matched["exact_casefold_iou_ge_50"] == {0}


def test_cached_ocr_requires_same_weights_decode_and_crop():
    record = {"crop_rgb_sha256": "a" * 64, "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "provenance": {"decode_ar": True, "refine_iters": 1}, "raw_text": "sample"}
    assert cached_text(record, "a" * 64) == "sample"
    for change in ({"crop_rgb_sha256": "b" * 64}, {"provenance": {}}, {"raw_text": None}):
        with pytest.raises(ValueError, match="provenance"):
            cached_text({**record, **change}, "a" * 64)


def test_batched_ocr_diagnostics_receive_explicit_verified_decode_provenance():
    prediction = SimpleNamespace(to_dict=lambda: {"raw_text": "word", "predicted_length": 4})
    config = SimpleNamespace(decode_ar=True, refine_iters=1, runtime_source_type="github_commit_archive", runtime_source=Path("runtime"))
    record = cache_record(prediction, "a" * 64, config)
    assert cached_text(record, "a" * 64) == "word"
    assert record["provenance"]["strict_state_dict_load"] is True
    config.decode_ar = False
    with pytest.raises(ValueError, match="provenance"):
        cache_record(prediction, "a" * 64, config)


def test_real_inference_errors_are_not_replaced_by_masks(cfg):
    class Broken(TinyModel):
        def forward_evidence(self, image):
            raise torch.cuda.OutOfMemoryError("synthetic OOM for propagation test")
    model = Broken()
    batch = {"image": torch.zeros(1, 3, 32, 32)}
    with pytest.raises(torch.cuda.OutOfMemoryError, match="propagation"):
        train_research_epoch(model, [batch], build_optimizer(model, training_config(cfg, "control")), torch.device("cpu"), cfg)


def test_frozen_crop_ocr_stage_all_branches_cache_resume_and_export_tampering(tmp_path, monkeypatch, cfg):
    """Small fixture exercises the full file boundary; real PARSeq has its smoke."""
    from PIL import Image
    from types import SimpleNamespace
    import scripts.evaluate_unet_word_research_ocr as module
    from scripts.prune_segmentation_checkpoints import sha256_file as real_hash
    from models.unet_word_research import BRANCHES
    root, output = tmp_path / "dataset", tmp_path / "runs"
    root.mkdir(); output.mkdir()
    manifest_path = root / "manifest.yaml"; manifest_path.write_text("fixture")
    cfg.update(train_manifest="manifest.yaml", word_manifest_sha256=real_hash(manifest_path),
               validation_count=2, expected_validation_words=2)
    samples = []
    for i in range(2):
        path = root / f"{i}.png"
        Image.fromarray(np.full((12, 20, 3), 40 + i, np.uint8)).save(path)
        samples.append(SimpleNamespace(sample_id=f"total_text:{i}", image_path=Path(path.name), image_sha256=real_hash(path)))
    monkeypatch.setattr(module, "read_config", lambda *_: (cfg, {}))
    monkeypatch.setattr(module, "load_segmentation_manifest", lambda *_, **__: SimpleNamespace(dataset="total_text", split_role="train", samples=samples))
    monkeypatch.setattr(module, "_split_indices", lambda *_: ([], [0, 1], cfg["validation_ids_sha256"]))
    monkeypatch.setattr(module, "_train_gt", lambda *_: ([{"bbox": [0, 0, 20, 12], "text": "word"}], "annotation-fixture"))
    weights, archive = tmp_path / "weights", tmp_path / "archive"
    monkeypatch.setattr(module, "sha256_file", lambda path: PARSEQ_CHECKPOINT_SHA256 if Path(path) == weights
                        else PARSEQ_RUNTIME_ARCHIVE_SHA256 if Path(path) == archive else real_hash(Path(path)))
    constructed, allocator = [], []
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.torch.cuda, "max_memory_allocated", lambda _: 0)
    monkeypatch.setattr(module.torch.cuda, "max_memory_reserved", lambda _: 0)
    monkeypatch.setattr(module, "_prepare_cuda_memory_tracking", lambda device: allocator.append(str(device)))
    class FixtureRecognizer:
        def __init__(self, config):
            assert allocator == ["cuda:0"]  # Allocator qualification precedes model transfer.
            self.config = config; constructed.append(True)
        def recognize_candidate_batch(self, images):
            return [SimpleNamespace(to_dict=lambda: {"raw_text": "word"}) for _ in images]
        def release(self):
            return {"test_fixture": True}
    monkeypatch.setattr(module, "ParseqApproxRecognizer", FixtureRecognizer)
    for branch in BRANCHES:
        directory = output / branch; directory.mkdir()
        (directory / "best.pt").write_bytes(b"fixture checkpoint")
        digest = real_hash(directory / "best.pt")
        contract = {"test_fixture": True}
        (directory / "run_manifest.json").write_text(json.dumps({"status": "SUCCESS", "training_complete": True,
            "run_contract": contract, "checkpoint": {"sha256": digest}}))
        rows = []
        for sample in samples:
            folder = directory / "validation_exports" / sample.sample_id.replace(":", "__")
            folder.mkdir(parents=True)
            path = folder / "proposals.json"
            path.write_text(json.dumps({"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "proposals": [{"instance_id": 1, "bbox": [0, 0, 20, 12], "foreground_area": 240}],
                "quality_scores": {1: .9} if branch == "quality" else {}}))
            rows.append({"sample_id": sample.sample_id, "proposals_sha256": real_hash(path), "ignored_instance_ids": []})
        (directory / "validation.json").write_text(json.dumps({"run_contract": contract, "checkpoint_sha256": digest,
            "per_image": rows}))
    args = SimpleNamespace(config=DEFAULT_CONFIG, mode="approximate", word_root=root, run_root=output,
        parseq_checkpoint=weights, parseq_source_archive=archive, parseq_source=tmp_path / "runtime", batch_size=2, device="cuda:0")
    result = module.run(args)
    assert len(result["arms"]) == 7 and len(constructed) == 1
    assert all(v["aggregate"]["exact_casefold_iou_ge_50"] == 2 for v in result["arms"].values())
    assert len(list((output / "research_ocr_cache").glob("*.json"))) == 2
    allocator.clear()
    repeated = module.run(args)
    assert repeated["arms"] == result["arms"] and len(constructed) == 1
    path = output / "control" / "validation_exports" / "total_text__0" / "proposals.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="proposal receipt changed"):
        module.run(args)
