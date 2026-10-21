from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.run_unet_detection_methods import (DEFAULT_CONFIG, bind, read_config, train_branch,
    training_config, verify_completed, verify_exports, source_state)
from segmentation.training import seed_everything


@pytest.fixture
def cfg():
    value = read_config(DEFAULT_CONFIG, "approximate")[0]
    value.update(epochs=3, batch_size=2)
    value["synthtext"]["epochs"] = 3
    return value


class TinyModel(torch.nn.Module):
    branch = "control"
    architecture_identity = "test-detection-method-resume"
    def __init__(self):
        super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.2))


def install(monkeypatch):
    def train(model, *args):
        optimizer = args[1]
        loss = (model.weight - torch.rand(())).square()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        return {"steps": 1, "loss": float(loss.detach()), "samples": 2}
    monkeypatch.setattr("scripts.run_unet_detection_methods.train_detection_epoch", train)
    monkeypatch.setattr("scripts.run_unet_detection_methods.experiment_environment", lambda _: {"fixture": True})
    return SimpleNamespace(set_epoch=lambda _: None)


def invoke(path, model, cfg, setter, stop=None, contract=None, synthetic=False):
    path.mkdir(exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    def validation():
        score = float(model.weight.detach())
        return {"native": {"gt_words_touched_by_fused_components": 0},
                "original": {"geometry_f1": score, "geometry_recall_all_gt": score, "gt_words": 983},
                "pixel": {"foreground_iou_macro": score}}
    return train_branch(path, model, optimizer, [], validation, [setter], cfg, contract or {"fixture": True},
                        torch.device("cpu"), synthetic, stop)


def test_epoch_boundary_resume_restores_optimizer_rng_and_same_final_weights(tmp_path, monkeypatch, cfg):
    setter = install(monkeypatch)
    seed_everything(24, deterministic_algorithms=True); full = TinyModel()
    result = invoke(tmp_path / "full", full, cfg, setter)
    seed_everything(24, deterministic_algorithms=True); partial = TinyModel()
    paused = invoke(tmp_path / "resume", partial, cfg, setter, stop=1)
    assert paused["status"] == "IN_PROGRESS"
    torch.rand(29)
    resumed = TinyModel(); finished = invoke(tmp_path / "resume", resumed, cfg, setter)
    assert torch.equal(full.weight, resumed.weight) and finished["training_complete"]
    assert finished["checkpoint"]["epoch"] == result["checkpoint"]["epoch"]


def test_checkpoint_before_epoch_log_is_recovered_without_retraining(tmp_path, monkeypatch, cfg):
    setter = install(monkeypatch); path = tmp_path / "run"
    invoke(path, TinyModel(), cfg, setter, stop=1)
    (path / "epochs.jsonl").unlink()
    result = invoke(path, TinyModel(), cfg, setter)
    rows = [json.loads(line) for line in (path / "epochs.jsonl").read_text().splitlines()]
    assert rows[0]["recovered_from_checkpoint"] and result["completed_epochs"] == 3
    with pytest.raises(ValueError, match="contract"):
        invoke(path, TinyModel(), cfg, setter, contract={"fixture": "changed"})


def test_synthetic_stage_uses_last_checkpoint_without_target_validation_selection(tmp_path, monkeypatch, cfg):
    setter = install(monkeypatch); model = TinyModel()
    result = invoke(tmp_path / "synth", model, cfg, setter, synthetic=True)
    assert result["checkpoint"]["epoch"] == cfg["synthtext"]["epochs"]
    assert result["validation"] == {"synthetic_last_epoch": 3}
    assert training_config(cfg, True).checkpoint_selection == "last"


def test_contracts_canonicalize_tuples_and_preserve_existing_artifacts(tmp_path):
    path = tmp_path / "receipt.json"
    bind(path, {"betas": (.9, .999)}); before = path.read_bytes()
    bind(path, {"betas": [.9, .999]}); assert path.read_bytes() == before
    with pytest.raises(ValueError, match="NEW"): bind(path, {"betas": [0., 1.]})
    assert path.read_bytes() == before


def test_configuration_namespaces_and_training_loss_metadata():
    with pytest.raises(ValueError, match="APPROXIMATE"): read_config(DEFAULT_CONFIG, "strict")
    with pytest.raises(ValueError, match="APPROXIMATE"): read_config(Path("configs/demo/unet_detection_methods_v1.json"), "approximate")
    cfg = read_config(DEFAULT_CONFIG, "approximate")[0]
    assert asdict(training_config(cfg))["loss"] == "experimental_detection_BCE_Dice_nested_kernels_v1"


def test_synthetic_target_fix_is_an_explicit_isolated_configuration():
    baseline = read_config(DEFAULT_CONFIG, "approximate")[0]
    revised = read_config(DEFAULT_CONFIG.with_name("unet_detection_methods_synthtext_targets_v2.json"), "approximate")[0]
    assert "target_policy" not in baseline["synthtext"]
    assert revised["synthtext"]["target_policy"] == "preserve_visible_words_v1"
    assert revised["scientific_choices"][-1]["classification"] == "ASSUMPTION"
    assert revised["scientific_choices"][:-1] == baseline["scientific_choices"]
    revised["synthtext"].pop("target_policy")
    revised["scientific_choices"].pop()
    assert revised == baseline
    with pytest.raises(ValueError, match="APPROXIMATE"):
        read_config(DEFAULT_CONFIG.with_name("unet_detection_methods_synthtext_targets_v2.json"), "strict")


def test_partial_training_and_partial_exports_cannot_be_final_results(tmp_path):
    with pytest.raises(ValueError, match="partial"):
        verify_completed(tmp_path, {"status": "IN_PROGRESS"}, {}, 50)
    with pytest.raises(ValueError, match="Incomplete"):
        verify_exports(tmp_path, {"run_contract": {}, "status": "SUCCESS", "per_image": []}, {})


def test_no_pretrained_branch_can_use_a_missing_or_partial_synthetic_checkpoint(tmp_path, monkeypatch, cfg):
    monkeypatch.setattr("scripts.run_unet_detection_methods.load_source_state", lambda *_: {})
    args = SimpleNamespace(v1_root=tmp_path, output_root=tmp_path)
    with pytest.raises(FileNotFoundError): source_state(args, cfg, "pretrained_control")
    directory = tmp_path / "synthtext_pretrain"; directory.mkdir()
    (directory / "run_manifest.json").write_text(json.dumps({"status": "IN_PROGRESS"}))
    with pytest.raises(ValueError, match="completed"): source_state(args, cfg, "pretrained_control")


def test_completed_synthetic_handoff_is_pinned_and_can_use_a_separate_smoke_root(tmp_path, monkeypatch, cfg):
    import scripts.run_unet_detection_methods as m
    monkeypatch.setattr(m, "load_source_state", lambda *_: {})
    directory = tmp_path / "synthetic/synthtext_pretrain"; directory.mkdir(parents=True)
    manifest = directory.parent / "synthtext_manifest.json"; manifest.write_text('{"fixture":true}')
    contract = {"phase": "SynthText_domain_pretrain", "resolved_config": cfg,
                "synthetic_manifest_sha256": m.sha256_file(manifest),
                "source_sha256": {f: m.sha256_file(m.REPO / f) for f in m.SCIENCE_FILES}}
    payload = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
               "epoch": cfg["synthtext"]["epochs"], "run_contract": contract,
               "architecture": "experimental-unet-detection-methods-v1:control",
               "training_config": asdict(training_config(cfg, True)),
               "model_state_dict": {"fixture_weight": torch.tensor(.7)}}
    path = directory / "best.pt"; torch.save(payload, path)
    receipt = {"status": "SUCCESS", "training_complete": True, "completed_epochs": cfg["synthtext"]["epochs"],
               "checkpoint": {"path": "best.pt", "epoch": cfg["synthtext"]["epochs"], "sha256": m.sha256_file(path)},
               "run_contract": contract}
    (directory / "run_manifest.json").write_text(json.dumps(receipt))
    args = SimpleNamespace(v1_root=tmp_path, output_root=tmp_path / "smoke", pretrain_root=directory.parent)
    state, provenance = source_state(args, cfg, "pretrained_control")
    assert torch.equal(state["fixture_weight"], torch.tensor(.7)) and provenance["sha256"] == m.sha256_file(path)
    manifest.write_text('{"fixture":"changed"}')
    with pytest.raises(ValueError, match="provenance"): source_state(args, cfg, "pretrained_control")


def test_ocr_one_to_one_repeated_words_empty_predictions_and_punctuation():
    from scripts.evaluate_unet_word_research_v2_ocr import score_predictions
    gt = [{"bbox": [0, 0, 10, 10], "text": "WORD"}, {"bbox": [20, 0, 30, 10], "text": "WORD!"}]
    predicted = [{"bbox": [0, 0, 10, 10], "text": "word"},
                 {"bbox": [0, 0, 10, 10], "text": "word"},
                 {"bbox": [20, 0, 30, 10], "text": "word"},
                 {"bbox": [40, 0, 50, 10], "text": ""}]
    count, matches = score_predictions(predicted, gt)
    assert count["proposals"] == 4 and count["gt_words"] == 2
    assert count["exact_casefold_iou_ge_50"] == 1 and count["exact_no_symbols_iou_ge_50"] == 2


def test_crop_cache_refuses_changed_checkpoint_or_pixel_identity():
    from scripts.evaluate_unet_word_research_v2_ocr import cached_text
    from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
    record = {"crop_rgb_sha256": "fixture", "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256, "raw_text": "word",
              "provenance": {"decode_ar": True, "refine_iters": 1}}
    assert cached_text(record, "fixture") == "word"
    with pytest.raises(ValueError): cached_text(record, "other pixels")
    with pytest.raises(ValueError): cached_text({**record, "parseq_checkpoint_sha256": "bad"}, "fixture")


def test_oom_propagates_and_records_failure_without_alternate_model(tmp_path, monkeypatch, cfg):
    import scripts.run_unet_detection_methods as m
    from PIL import Image
    manifest_path = tmp_path / cfg["train_manifest"]
    manifest_path.parent.mkdir(parents=True); manifest_path.write_text("fixture manifest")
    cfg.update(word_manifest_sha256=m.sha256_file(manifest_path), expected_validation_words=1, num_workers=0)
    Image.new("RGB", (40, 24), "white").save(tmp_path / "image.png")
    sample = SimpleNamespace(sample_id="total_text:fixture", image_path=Path("image.png"))
    manifest = SimpleNamespace(dataset="total_text", split_role="train", samples=[sample])
    monkeypatch.setattr(m, "read_config", lambda *_: (cfg, {}))
    monkeypatch.setattr(m, "load_segmentation_manifest", lambda *_a, **_kw: manifest)
    monkeypatch.setattr(m, "_split_indices", lambda *_: ([0], [0], cfg["validation_ids_sha256"]))
    monkeypatch.setattr(m, "_transform", lambda *_a, **_kw: None)
    class EmptyDataset(torch.utils.data.Dataset):
        def __len__(self): return 1
        def __getitem__(self, i): raise AssertionError("No dataset/model fallback may run")
    monkeypatch.setattr(m, "SegmentationDataset", lambda *_a, **_kw: EmptyDataset())
    monkeypatch.setattr(m, "DetectionWordDataset", lambda x, _: x)
    monkeypatch.setattr(m, "_validation_reference", lambda *_: {})
    monkeypatch.setattr(m, "_train_gt", lambda *_: ([{"bbox": [0, 0, 10, 10], "text": "word"}], "fixture"))
    monkeypatch.setattr(m, "source_state", lambda *_: ({}, {"fixture": True}))
    class OOMModel(torch.nn.Module):
        def initialize_source(self, _): pass
        def to(self, _): raise torch.cuda.OutOfMemoryError("injected qualification OOM")
    monkeypatch.setattr(m, "DetectionMethodsAGUNet", lambda *_a, **_kw: OOMModel())
    args = SimpleNamespace(config=DEFAULT_CONFIG, mode="approximate", batch_size=None, num_workers=None,
        device="cpu", word_root=tmp_path, v1_root=tmp_path, output_root=tmp_path / "output",
        limit_images=1, action="smoke", branch="control", stop_after_epoch=None)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="injected"):
        m.run(args)
    failure = json.loads((args.output_root / "control/failure.json").read_text())
    assert failure["memory"]["oom"] and failure["error_type"] == "OutOfMemoryError"
    assert not (args.output_root / "control/best.pt").exists()
