"""Audit boundaries: dataset denominator, cache backend, smoke mode and OOM."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_seeded_selective import ARMS, DEFAULT_CONFIG, crop_loader, read_config, score_image


def test_missing_crop_evidence_fails_before_decoding():
    records = {"crops": [{"seed_id": 1}]}
    reports = {a: {"x": {"audit": {"crop_evidence": []}}} for a in ("crop_word", "crop_links")}
    with pytest.raises(ValueError, match="per-crop evidence"):
        crop_loader(None, "x", records, reports, {}, {})


def test_geometry_empty_predictions_keep_every_gt_word_and_ignore_is_explicit():
    owners = np.zeros((8, 10), np.int32); owners[2:5, 2:5] = 1
    batch = {"validity": np.ones((8, 10), np.uint8)}
    gt = [{"bbox": [2, 2, 5, 5], "text": "word"}]
    metrics, _, _, _, _, iou, f1 = score_image([], batch, owners, gt, [.25, .5, .75])
    assert metrics["gt_words"] == 1 and metrics["raw_proposals"] == 0 and iou == f1 == 0.
    batch["validity"][0:2, 7:9] = 0
    patches = [{"bbox": [7, 0, 9, 2], "mask": np.ones((2, 2), bool)}]
    metrics, proposals, ignored, *_ = score_image(patches, batch, owners, gt, [.25, .5, .75])
    assert len(proposals) == 1 and ignored == [1] and metrics["raw_proposals"] == 0 and metrics["gt_words"] == 1


@pytest.fixture
def ocr_fixture(tmp_path, monkeypatch):
    import scripts.evaluate_unet_seeded_selective_ocr as m
    from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
    config, source = deepcopy(read_config(DEFAULT_CONFIG, "approximate"))
    root, output = tmp_path / "data", tmp_path / "run"; root.mkdir(); output.mkdir()
    manifest = root / "manifest.yaml"; manifest.write_text("fixture")
    source.update(train_manifest="manifest.yaml", word_manifest_sha256=sha256_file(manifest))
    path = root / "image.png"; Image.fromarray(np.full((12, 20, 3), 42, np.uint8)).save(path)
    sample = SimpleNamespace(sample_id="total_text:0", image_path=path.name, image_sha256=sha256_file(path))
    contract = {"resolved_config": config, "source_config": source, "source_sha256": {},
                "runtime_inputs": {"word_root": str(root)}, "result_class": "smoke-test result",
                "validation_ids_sha256": source["validation_ids_sha256"], "indices": [0]}
    (output / "run_contract.json").write_text(json.dumps(contract))
    monkeypatch.setattr(m, "read_config", lambda *_: (config, source))
    monkeypatch.setattr(m, "load_segmentation_manifest", lambda *_, **__: SimpleNamespace(dataset="total_text", split_role="train", samples=[sample]))
    monkeypatch.setattr(m, "_split_indices", lambda *_: ([], [0], source["validation_ids_sha256"]))
    monkeypatch.setattr(m, "_train_gt", lambda *_: ([{"bbox": [0, 0, 20, 12], "text": "word"}], "annotation"))
    weights, archive = tmp_path / "weights", tmp_path / "archive"
    monkeypatch.setattr(m, "sha256_file", lambda p: PARSEQ_CHECKPOINT_SHA256 if p == weights else
                        PARSEQ_RUNTIME_ARCHIVE_SHA256 if p == archive else sha256_file(p))
    calls = []
    class Recognizer:
        def __init__(self, c): self.config = c; calls.append("load")
        def recognize_candidate_batch(self, crops):
            calls.append("recognize")
            assert len(crops) == 1 and crops[0].shape == (12, 20, 3)
            return [SimpleNamespace(to_dict=lambda: {"raw_text": "word"})]
        def release(self): calls.append("release")
    monkeypatch.setattr(m, "ParseqApproxRecognizer", Recognizer)
    for arm, source_arm in ARMS.items():
        directory = output / arm; folder = directory / "validation_exports/total_text__0"; folder.mkdir(parents=True)
        maps = folder / "maps.npz"; np.savez_compressed(maps, support_1=np.ones((12, 20), bool))
        props = folder / "proposals.json"
        props.write_text(json.dumps({"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
            "maps_sha256": sha256_file(maps), "proposals": [{"instance_id": 1, "bbox": [0, 0, 20, 12], "foreground_area": 240}]}))
        report = {"run_contract": contract, "result_class": "smoke-test result",
            "checkpoint_sha256": config["source_artifacts"][source_arm]["checkpoint_sha256"],
            "aggregate": {"gt_words": 1}, "per_image": [{"sample_id": sample.sample_id, "ignored_instance_ids": [],
                "maps_sha256": sha256_file(maps), "proposals_sha256": sha256_file(props)}]}
        rp = directory / "validation.json"; rp.write_text(json.dumps(report))
        (directory / "run_manifest.json").write_text(json.dumps({"status": "SUCCESS", "execution_kind": "frozen_postprocess",
            "run_contract": contract, "result_class": "smoke-test result", "source_checkpoint": config["source_artifacts"][source_arm],
            "validation_sha256": sha256_file(rp)}))
    args = SimpleNamespace(config=DEFAULT_CONFIG, mode="approximate", word_root=root, run_root=output,
        parseq_checkpoint=weights, parseq_source_archive=archive, parseq_source=tmp_path / "runtime",
        device="cpu", batch_size=8, cache_source=None, smoke=True)
    return m, args, output, calls, Recognizer


def test_six_arms_share_rgb_cache_and_resume_without_reloading_model(ocr_fixture):
    m, args, _, calls, _ = ocr_fixture
    first = m.run(args); again = m.run(args)
    assert first["arms"] == again["arms"] and calls == ["load", "recognize", "release"]
    assert first["protocol"]["result_class"] == "smoke-test result"
    assert first["new_unique_crops_recognized"] == 1 and again["new_unique_crops_recognized"] == 0
    assert all(a["aggregate"]["exact_casefold_iou_ge_50"] == 1 for a in first["arms"].values())


def test_smoke_results_cannot_be_reported_as_final_evaluation(ocr_fixture):
    m, args, *_ = ocr_fixture; args.smoke = False
    with pytest.raises(ValueError, match="Partial smoke"): m.run(args)


def test_modified_supports_fail_checksum_validation(ocr_fixture):
    m, args, output, *_ = ocr_fixture
    (output / "crop_word_selective/validation_exports/total_text__0/maps.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum changed"): m.run(args)


def test_cpu_gpu_cache_mixing_is_explicitly_rejected(ocr_fixture):
    m, args, output, *_ = ocr_fixture; m.run(args)
    external = output.parent / "external"; external.mkdir()
    execution = json.loads((output / "corrective_ocr_cache/execution.json").read_text()); execution["device"] = "cuda:0"
    (external / "execution.json").write_text(json.dumps(execution)); args.cache_source = external
    with pytest.raises(ValueError, match="backend differs"): m.run(args)


def test_oom_propagates_with_failure_receipt_and_model_release(ocr_fixture, monkeypatch):
    m, args, output, calls, recognizer = ocr_fixture
    def oom(*_): raise torch.cuda.OutOfMemoryError("synthetic OOM boundary")
    monkeypatch.setattr(recognizer, "recognize_candidate_batch", oom)
    with pytest.raises(torch.cuda.OutOfMemoryError): m.run(args)
    assert json.loads((output / "ocr_failure.json").read_text())["oom"] is True and calls[-1] == "release"
