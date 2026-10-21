"""TRAIN-only threshold diagnostic invariants; no CUDA inference."""

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import (
    EXPECTED_THRESHOLDS, _config, _freeze_probability, _load_probability,
    _matched, _overlap, threshold_key,
)


def test_predeclared_train_only_configuration():
    config = _config()
    assert config["split"] == "TRAIN_heldout_125"
    assert config["thresholds"] == list(EXPECTED_THRESHOLDS)
    assert config["padding_native_pixels"] == 0
    assert config["area_filter"] == "disabled"
    assert config["maximum_proposals"] == "disabled"
    assert threshold_key(0.5) == "0.50"
    with pytest.raises(ValueError):
        threshold_key(0.525)


def test_correct_word_overlap_tracks_new_and_lost_word_identities():
    correct = {"0.35": {"im:1#0", "im:1#2"},
               "0.50": {"im:1#0", "im:1#1"},
               "0.65": {"im:1#1", "im:1#3"}}
    result = _overlap(correct)
    assert result["correct_any_threshold"] == 4
    assert result["gain_any_threshold_vs_0_50"] == 2
    assert result["by_threshold"]["0.35"]["new_vs_0_50"] == 1
    assert result["by_threshold"]["0.35"]["lost_vs_0_50"] == 1
    assert result["intersection_matrix"]["0.35"]["0.65"] == 0
    assert result["new_word_ids_vs_0_50"]["0.65"] == ["im:1#3"]


def test_native_probability_cache_is_float32_and_hash_verified(tmp_path: Path):
    sample = SimpleNamespace(sample_id="total_text:1", image_sha256="a" * 64)
    native = np.full((256, 256), np.float32(0.5), dtype=np.float32)
    native[0, 0] = np.float32(0.49)
    path = tmp_path / "native.npy"
    metadata = tmp_path / "native.json"
    _freeze_probability(path, metadata, native, sample=sample, protocol_sha="protocol")
    import json
    frozen = json.loads(metadata.read_text(encoding="utf-8"))
    restored = _load_probability(path, frozen, sample=sample, protocol_sha="protocol")
    assert np.array_equal(restored >= 0.5, native >= 0.5)
    assert sha256(restored.tobytes()).hexdigest() == frozen["array_sha256"]
    with pytest.raises(ValueError, match="provenance"):
        _load_probability(path, frozen, sample=sample, protocol_sha="other")
    with pytest.raises(ValueError, match="native"):
        _freeze_probability(path, metadata, native.astype(np.float16),
                            sample=sample, protocol_sha="protocol")


def test_geometric_and_ocr_matching_remain_one_to_one():
    predictions = [{"bbox": [0, 0, 10, 10], "text": "Hi!"},
                   {"bbox": [1, 0, 11, 10], "text": "Hi!"}]
    gt = [{"bbox": [0, 0, 10, 10], "text": "hi"}]
    matched = _matched(predictions, gt)
    assert matched["geometry_iou_ge_50"] == {0}
    assert matched["exact_casefold_iou_ge_50"] == set()
    assert matched["exact_no_symbols_iou_ge_50"] == {0}
