import json
from pathlib import Path

import pytest

from recognition.parseq import PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256
from scripts.evaluate_text_recall_crops import CONFIG, completed_source, prepare_ocr_provider, read_config
from scripts.prune_segmentation_checkpoints import sha256_file


def test_new_crop_config_does_not_enter_strict_or_fall_back(tmp_path):
    assert read_config(CONFIG, "approximate")["common_threshold"] == .5
    with pytest.raises(ValueError, match="APPROXIMATE"): read_config(CONFIG, "strict")
    copy = tmp_path / "config.json"; copy.write_bytes(CONFIG.read_bytes())
    with pytest.raises(ValueError, match="namespace"): read_config(copy, "approximate")


def test_source_checkpoint_mutation_fails(tmp_path):
    checkpoint = tmp_path / "best.pt"; checkpoint.write_bytes(b"fixture")
    contract = {"source_sha256": {}}
    manifest = {"status": "SUCCESS", "result_class": "approximate reproduction", "run_contract": contract,
                "checkpoint": {"path": "best.pt", "sha256": sha256_file(checkpoint)}}
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "validation.json").write_text(json.dumps({"run_contract": contract}))
    assert completed_source(tmp_path)[0] == manifest
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed source checkpoint"): completed_source(tmp_path)


def source_cache(root, text="word"):
    key = "crop"
    record = {"crop_rgb_sha256": key, "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
              "decode_ar": True, "refine_iters": 1, "raw_text": text, "predicted_length": len(text)}
    path = root / "ocr" / (key+".json"); path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    protocol = {"parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                "recognition": {"decode_ar": True, "refine_iters": 1}}
    report = {"status": "SUCCESS", "result_class": "approximate reproduction", "protocol": protocol,
              "ocr_records_sha256": {key: sha256_file(path)}}
    (root / "recognition_results.json").write_text(json.dumps(report))
    return path


def geometry():
    return {"branches": {"test": {"findcontours": {"per_image": [
        {"proposals": [{"crop_rgb_sha256": "crop"}]}]}}}}


def test_completed_ocr_import_is_byte_identical_and_rejects_mutation(tmp_path):
    root = tmp_path / "source"; path = source_cache(root)
    provider = prepare_ocr_provider(tmp_path / "out", [root], geometry())
    assert (provider / "crop.json").read_bytes() == path.read_bytes()
    assert json.loads((provider.parent / "protocol.json").read_text())["records"]["crop"]["sha256"] == sha256_file(path)
    path.write_text('{}')
    with pytest.raises(ValueError, match="Historical OCR record changed"):
        prepare_ocr_provider(tmp_path / "out2", [root], geometry())


def test_conflicting_same_rgb_transcriptions_are_not_silently_reused(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    source_cache(first, "a"); source_cache(second, "b")
    with pytest.raises(ValueError, match="Conflicting exact-crop"):
        prepare_ocr_provider(tmp_path / "out", [first, second], geometry())
