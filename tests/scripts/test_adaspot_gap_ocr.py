"""Frozen three-arm crops, exact matching, resume and exported-map integrity."""

import json
from pathlib import PurePosixPath
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts import evaluate_adaspot_gap_ocr as module
from scripts.prune_segmentation_checkpoints import sha256_file


def test_precision_recall_hmean_and_full_denominator():
    rows = [{"gt_words": 3, "proposals": 4, "geometry_iou_ge_50": 2,
             "exact_casefold_iou_ge_50": 1, "exact_no_symbols_iou_ge_50": 2}]
    result = module.aggregate_rows(rows, 3)
    assert result["exact_casefold_precision"] == .25
    assert result["exact_casefold_recall"] == 1 / 3
    assert result["exact_casefold_hmean"] == 2 / 7
    with pytest.raises(ValueError, match="denominator"):
        module.aggregate_rows(rows, 4)


def test_empty_outputs_and_duplicates_remain_in_denominator():
    gt = [{"bbox": [0, 0, 10, 10], "text": "word"}]
    result, matches = module.score_predictions([], gt)
    assert result["gt_words"] == 1 and result["exact_casefold_iou_ge_50"] == 0
    result, matches = module.score_predictions([{"bbox": [0, 0, 10, 10], "text": "WORD"}] * 2, gt)
    assert result["proposals"] == 2 and result["exact_casefold_iou_ge_50"] == 1


def test_unmatched_sources_or_incomplete_control_provenance_are_rejected():
    contract = {key: {} for key in ("resolved_config", "resolved_preprocessing", "validation_ids_sha256",
        "runtime_inputs", "execution_device", "backend", "gpu_name", "training_config", "source_sha256")}
    module.compare_matched_contracts(contract, contract)
    with pytest.raises(ValueError, match="source_sha256"):
        module.compare_matched_contracts(contract, {**contract, "source_sha256": {"changed": "hash"}})
    with pytest.raises(ValueError, match="incomplete"):
        module.compare_matched_contracts(contract, {"resolved_config": {}})


def test_full_file_boundary_cache_reuse_gap_off_ignore_and_tampering(tmp_path, monkeypatch):
    """Explicit two-image fixture; no real benchmark or PARSeq result is claimed."""
    root, output = tmp_path / "data", tmp_path / "run"
    root.mkdir(); output.mkdir()
    config = {"train_manifest": "manifest.yaml", "word_manifest_sha256": "manifest-fixture",
        "seed": 202503, "validation_count": 2, "validation_ids_sha256": "split-fixture",
        "expected_validation_words": 2, "epochs": 2, "branches": ["gap", "control"]}
    (root / "manifest.yaml").write_text("fixture")
    weights, archive = tmp_path / "weights", tmp_path / "archive"
    weights.write_bytes(b"fixture weights"); archive.write_bytes(b"fixture archive")
    samples = []
    for index in range(2):
        name = "{}.png".format(index)
        Image.new("RGB", (20, 12), (20 + index, 50, 80)).save(root / name)
        samples.append(SimpleNamespace(sample_id="total_text:" + str(index), image_path=PurePosixPath(name),
                                       image_sha256=sha256_file(root / name)))
    manifest = SimpleNamespace(dataset="total_text", split_role="train", samples=samples)
    monkeypatch.setattr(module, "read_config", lambda *_: (config, {}))
    monkeypatch.setattr(module, "load_segmentation_manifest", lambda *a, **kw: manifest)
    monkeypatch.setattr(module, "_split_indices", lambda *_: ([], [0, 1], "split-fixture"))
    monkeypatch.setattr(module, "sha256_file", lambda p:
        "manifest-fixture" if str(p).endswith("manifest.yaml") else
        module.PARSEQ_CHECKPOINT_SHA256 if p == weights else
        module.PARSEQ_RUNTIME_ARCHIVE_SHA256 if p == archive else sha256_file(p))
    monkeypatch.setattr(module, "_train_gt", lambda *_: ([{"bbox": [0, 0, 20, 12], "text": "word"}], "annotation-fixture"))
    constructed = []
    class FixtureRecognizer:
        def __init__(self, cfg):
            self.config = cfg; constructed.append(True)
        def recognize_candidate_batch(self, crops):
            return [SimpleNamespace(to_dict=lambda: {"raw_text": "word"}) for crop in crops]
        def release(self):
            return {"fixture": True}
    monkeypatch.setattr(module, "ParseqApproxRecognizer", FixtureRecognizer)
    for branch in config["branches"]:
        directory = output / branch; directory.mkdir()
        (directory / "best.pt").write_bytes(b"fixture checkpoint")
        digest = sha256_file(directory / "best.pt")
        contract = {"resolved_config": config, "resolved_preprocessing": {},
            "validation_ids_sha256": "split-fixture", "runtime_inputs": {"word_root": str(root)},
            "execution_device": "cpu", "backend": {"fixture": True}, "gpu_name": None,
            "training_config": {"fixture": True}, "source_sha256": {"fixture": "same-source"}}
        (directory / "run_manifest.json").write_text(json.dumps({"status": "SUCCESS", "training_complete": True,
            "completed_epochs": 2, "run_contract": contract, "checkpoint": {"sha256": digest}}))
        rows = []
        for sample in samples:
            folder = directory / "validation_exports" / sample.sample_id.replace(":", "__")
            folder.mkdir(parents=True)
            np.savez_compressed(folder / "maps.npz", labels=np.ones((4, 4), np.int32))
            maps_sha = sha256_file(folder / "maps.npz")
            proposal = {"instance_id": 1, "bbox": [0, 0, 20, 12], "foreground_area": 16}
            path = folder / "proposals.json"
            path.write_text(json.dumps({"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "maps_sha256": maps_sha, "proposals": [proposal], "gap_disabled_proposals": [proposal]}))
            # The disabled arm has its OWN ignored-ID list, independent of gap labels.
            rows.append({"sample_id": sample.sample_id, "maps_sha256": maps_sha,
                "proposals_sha256": sha256_file(path), "ignored_instance_ids": [],
                "gap_disabled_ignored_instance_ids": [1] if sample.sample_id.endswith(":0") else []})
        (directory / "validation.json").write_text(json.dumps({"run_contract": contract,
            "checkpoint_sha256": digest, "per_image": rows}))
    args = SimpleNamespace(config=module.DEFAULT_CONFIG, mode="approximate", word_root=root, run_root=output,
        parseq_checkpoint=weights, parseq_source_archive=archive, parseq_source=tmp_path / "runtime",
        device="cpu", batch_size=2)
    result = module.run(args)
    assert result["arms"]["gap"]["aggregate"]["exact_casefold_iou_ge_50"] == 2
    assert result["arms"]["gap_disabled"]["aggregate"]["exact_casefold_iou_ge_50"] == 1
    assert result["arms"]["gap_disabled"]["aggregate"]["gt_words"] == 2
    assert len(constructed) == 1 and len(list((output / "parseq_ocr_cache").glob("*.json"))) == 2
    repeated = module.run(args)
    assert result["arms"] == repeated["arms"] and len(constructed) == 1
    path = output / "control" / "validation_exports" / "total_text__0" / "maps.npz"
    path.write_bytes(b"changed export")
    with pytest.raises(ValueError, match="maps changed"):
        module.run(args)
