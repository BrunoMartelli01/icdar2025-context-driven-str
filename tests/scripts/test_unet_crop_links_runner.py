"""Config isolation, exact checkpoint resume, paired reporting and fallback behavior."""

from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
from PIL import Image
import pytest
import torch

from scripts.run_unet_crop_links import (DEFAULT_CONFIG, read_config, train_branch,
    bind_contract, refined_patches, qualify_global)
from scripts.summarize_unet_crop_links import paired_changes
from segmentation.training import seed_everything


@pytest.fixture
def cfg():
    c = deepcopy(read_config(DEFAULT_CONFIG, "approximate")[0]); c.update(epochs=3, batch_size=2)
    return c


def test_approximate_namespace_and_mode_are_required(tmp_path):
    with pytest.raises(ValueError, match="APPROXIMATE"): read_config(DEFAULT_CONFIG, "strict")
    fake = tmp_path / "config.json"; fake.write_bytes(DEFAULT_CONFIG.read_bytes())
    with pytest.raises(ValueError, match="APPROXIMATE"): read_config(fake, "approximate")


def test_run_contract_and_incomplete_global_dependency_fail(tmp_path, cfg):
    p = tmp_path / "contract.json"; bind_contract(p, {"seed": 1}); bind_contract(p, {"seed": 1})
    with pytest.raises(ValueError, match="contract changed"): bind_contract(p, {"seed": 2})
    with pytest.raises(ValueError, match="not completed"): qualify_global(tmp_path, cfg)


class TinyModel(torch.nn.Module):
    architecture_identity = "test-learned-crop"
    crop_variant = "crop_word"
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.25))
    def set_research_epoch(self, epoch): self.epoch = epoch


def setup_train(monkeypatch):
    def train(model, loader, optimizer, device, config, epoch):
        loss = (model.weight - torch.rand(())).square()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        return {"loss": float(loss.detach()), "steps": 1, "samples": 2}
    monkeypatch.setattr("scripts.run_unet_crop_links.train_crop_epoch", train)
    monkeypatch.setattr("scripts.run_unet_crop_links.experiment_environment", lambda _: {"kind": "test"})
    setter = SimpleNamespace(set_epoch=lambda _: None)
    def execute(path, model, c, stop=None):
        path.mkdir(exist_ok=True)
        def evaluate():
            p = float(model.weight.detach())
            return {"aggregate": {"proposal_f1_iou_50": p, "gt_words_touched_by_fused_components": 0,
                "word_recall_iou_50": p, "foreground_iou": p, "gt_words": 983}}
        return train_branch(path, model, torch.optim.Adam(model.parameters(), lr=.02), [], setter, setter,
                            evaluate, c, {"fixture": True}, torch.device("cpu"), stop)
    return execute


def test_resume_restores_rng_optimizer_and_replays_same_updates(tmp_path, monkeypatch, cfg):
    execute = setup_train(monkeypatch)
    seed_everything(19, deterministic_algorithms=True); full = TinyModel()
    a = execute(tmp_path / "full", full, cfg)
    seed_everything(19, deterministic_algorithms=True); partial = TinyModel()
    paused = execute(tmp_path / "resume", partial, cfg, 1)
    assert paused["status"] == "IN_PROGRESS"
    torch.rand(23); resumed = TinyModel()
    b = execute(tmp_path / "resume", resumed, cfg)
    assert torch.equal(full.weight, resumed.weight) and a["training_complete"] and b["training_complete"]


def test_checkpoint_saved_before_log_is_recovered(tmp_path, monkeypatch, cfg):
    execute = setup_train(monkeypatch); path = tmp_path / "run"
    execute(path, TinyModel(), cfg, 1); (path / "epochs.jsonl").unlink()
    execute(path, TinyModel(), cfg)
    rows = [json.loads(s) for s in (path / "epochs.jsonl").read_text().splitlines()]
    assert rows[0]["recovered_from_checkpoint"] and [r["epoch"] for r in rows] == [1, 2, 3]


def test_zero_child_preserves_first_pass_seed_and_exports_raw_evidence(cfg, tmp_path):
    labels = np.zeros((256, 256), np.int32); labels[80:120, 70:130] = 1
    row = {"crops": [{"seed_id": 1, "seed_bbox": [70, 80, 130, 120],
                     "context_bbox": [40, 40, 160, 160], "whole_image_fallback": False}]}
    class Empty:
        def forward_evidence(self, images):
            return {"words": torch.full((len(images), 2, 256, 256), -100.), "links": None}
    def transform(image, *args): return torch.zeros(3, 512, 512), None
    patches, audit = refined_patches(Empty(), Image.new("RGB", (256, 256)), row, labels, transform, cfg,
                                    torch.device("cpu"), evidence_dir=tmp_path)
    assert len(patches) == 1 and patches[0]["bbox"] == [70, 80, 130, 120]
    assert audit["seed_fallbacks"] == 1
    assert len(audit["crop_evidence"]) == 1
    with np.load(tmp_path / "1.npz", allow_pickle=False) as saved:
        assert saved["probabilities"].shape == (2, 256, 256) and not saved["labels"].any()


def test_oom_is_propagated_without_dummy_crop_output(cfg):
    class OOM:
        def forward_evidence(self, _): raise torch.cuda.OutOfMemoryError("synthetic boundary test")
    row = {"crops": [{"seed_id": 0, "seed_bbox": [0, 0, 64, 64],
                     "context_bbox": [0, 0, 64, 64], "whole_image_fallback": True}]}
    with pytest.raises(torch.cuda.OutOfMemoryError):
        refined_patches(OOM(), Image.new("RGB", (64, 64)), row, np.zeros((256, 256), np.int32),
            lambda *_: (torch.zeros(3, 512, 512), None), cfg, torch.device("cpu"))


def test_paired_summary_counts_words_even_when_image_total_is_unchanged():
    gt = [{"text": "A"}, {"text": "B"}]
    old = [{"sample_id": "x", "ground_truth": gt, "matched_gt_indices": {"exact_casefold_iou_ge_50": [0]}}]
    new = [{"sample_id": "x", "ground_truth": gt, "matched_gt_indices": {"exact_casefold_iou_ge_50": [1]}}]
    result = paired_changes(new, old)
    assert result["gained"] == result["lost"] == 1 and result["net"] == 0
    assert result["gained_words"][0]["text"] == "B"
    with pytest.raises(ValueError, match="population"): paired_changes([], old)


def test_ocr_four_arm_original_rgb_boundary_cache_resume_and_partial_rejection(tmp_path, monkeypatch, cfg):
    import scripts.evaluate_unet_crop_links_ocr as m
    from scripts.prune_segmentation_checkpoints import sha256_file as digest
    from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
    root, output = tmp_path / "data", tmp_path / "runs"
    root.mkdir(); output.mkdir()
    manifest = root / "manifest.yaml"; manifest.write_text("fixture")
    cfg.update(train_manifest="manifest.yaml", word_manifest_sha256=digest(manifest), validation_count=2, expected_validation_words=2)
    samples = []
    for i in range(2):
        path = root / f"{i}.png"
        Image.fromarray(np.full((12, 20, 3), 40 + i, np.uint8)).save(path)
        samples.append(SimpleNamespace(sample_id=f"total_text:{i}", image_path=path.name, image_sha256=digest(path)))
    monkeypatch.setattr(m, "read_config", lambda *_: (cfg, {}))
    monkeypatch.setattr(m, "load_segmentation_manifest", lambda *_, **__: SimpleNamespace(dataset="total_text", split_role="train", samples=samples))
    monkeypatch.setattr(m, "_split_indices", lambda *_: ([], [0, 1], cfg["validation_ids_sha256"]))
    monkeypatch.setattr(m, "_train_gt", lambda *_: ([{"bbox": [0, 0, 20, 12], "text": "word"}], "fixture"))
    weights, archive, checkpoint = tmp_path / "weights", tmp_path / "archive", tmp_path / "source.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    monkeypatch.setattr(m, "sha256_file", lambda p: PARSEQ_CHECKPOINT_SHA256 if p == weights else
        PARSEQ_RUNTIME_ARCHIVE_SHA256 if p == archive else digest(p))
    created = []
    class Recognizer:
        def __init__(self, config): self.config = config; created.append(True)
        def recognize_candidate_batch(self, crops):
            assert all(c.shape == (12, 20, 3) for c in crops)
            return [SimpleNamespace(to_dict=lambda: {"raw_text": "word"}) for _ in crops]
        def release(self): return {}
    monkeypatch.setattr(m, "ParseqApproxRecognizer", Recognizer)
    for name in cfg["branches"]:
        directory = output / name; directory.mkdir()
        receipt = {"status": "SUCCESS", "training_complete": True, "completed_epochs": cfg["epochs"] if name.startswith("crop_") else 0,
            "run_contract": {"fixture": True}, "checkpoint": {"path": str(checkpoint), "sha256": digest(checkpoint)}}
        (directory / "run_manifest.json").write_text(json.dumps(receipt)); rows = []
        for sample in samples:
            folder = directory / "validation_exports" / sample.sample_id.replace(":", "__"); folder.mkdir(parents=True)
            maps = folder / "maps.npz"; maps.write_bytes(b"fixture masks")
            proposals = folder / "proposals.json"
            proposals.write_text(json.dumps({"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "maps_sha256": digest(maps), "quality_scores": {},
                "proposals": [{"instance_id": 1, "bbox": [0, 0, 20, 12], "foreground_area": 240}]}))
            rows.append({"sample_id": sample.sample_id, "proposals_sha256": digest(proposals),
                         "maps_sha256": digest(maps), "ignored_instance_ids": []})
        (directory / "validation.json").write_text(json.dumps({"run_contract": {"fixture": True},
            "checkpoint_sha256": digest(checkpoint), "per_image": rows}))
    args = SimpleNamespace(config=DEFAULT_CONFIG, mode="approximate", subset="all", word_root=root, run_root=output,
        parseq_checkpoint=weights, parseq_source_archive=archive, parseq_source=tmp_path / "runtime", batch_size=2, device="cpu")
    first = m.run(args); repeated = m.run(args)
    assert first["arms"] == repeated["arms"] and len(created) == 1
    assert first["protocol"]["comparison_baseline"] == "control"
    assert all(a["aggregate"]["exact_casefold_iou_ge_50"] == 2 for a in first["arms"].values())
    receipt_path = output / "crop_links/run_manifest.json"
    receipt = json.loads(receipt_path.read_text()); receipt["completed_epochs"] -= 1
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Incomplete"): m.run(args)
    receipt["completed_epochs"] += 1; receipt_path.write_text(json.dumps(receipt))
    (output / "crop_links/validation_exports/total_text__0/maps.npz").write_bytes(b"changed supports")
    with pytest.raises(ValueError, match="supports changed"): m.run(args)
    args.batch_size = 1
    with pytest.raises(ValueError, match="backend"): m.run(args)
