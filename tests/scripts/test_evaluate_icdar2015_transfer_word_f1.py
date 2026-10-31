"""ICDAR2015 TEST proposal, coordinate and evaluation configuration checks."""

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts.evaluate_icdar2015_totaltext_transfer_word_f1 import (
    crop_hash, ground_truth, load_config, proposals,
)


CONFIG = Path(__file__).resolve().parents[2] / "configs/approximate/icdar2015_totaltext_transfer_word_f1_v1.json"


def test_all_components_use_pixel_area_order_and_inverse_letterbox():
    mask = np.zeros((256, 256), dtype=bool)
    mask[60, 20] = mask[61, 21] = True
    mask[80:82, 80:82] = True
    rows = proposals(mask, (1280, 720), load_config(CONFIG))
    assert len(rows) == 2 and [r["component_area"] for r in rows] == [4, 2]
    assert rows[1]["bbox_native"] == [20, 60, 22, 62]
    assert rows[1]["bbox"] == [100, 20, 110, 30]
    assert proposals(np.zeros_like(mask), (1280, 720), load_config(CONFIG)) == []


def test_crop_identity_includes_shape_and_pixels():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    assert crop_hash(rgb) == crop_hash(rgb.copy())
    assert crop_hash(rgb) != crop_hash(rgb.reshape(6, 4, 3))
    changed = rgb.copy()
    changed[0, 0, 0] = 1
    assert crop_hash(rgb) != crop_hash(changed)
    with pytest.raises(ValueError, match="nonempty"):
        crop_hash(rgb[:0])


def test_gt_preserves_transcription_and_ignored_denominator_is_explicit(tmp_path):
    root = tmp_path / "datasets/icdar2015/test"
    (root / "annotations").mkdir(parents=True)
    (root / "annotations/1.txt").write_text("0,0,9,0,9,9,0,9,Café!\n10,0,20,0,20,9,10,9,###\n")
    sample = SimpleNamespace(sample_id="icdar2015:1", image_path=Path("datasets/icdar2015/test/images/1.jpg"))
    words, ignored, digest = ground_truth(sample, tmp_path, (20, 10))
    assert ignored == 1 and len(words) == 1 and words[0]["text"] == "Café!"
    assert words[0]["bbox"] == [0, 0, 10, 10] and len(digest) == 64
    assert json.loads(json.dumps(words)) == words


def test_strict_configuration_cannot_load_approximate_evaluator(tmp_path):
    config = load_config(CONFIG)
    assert config["text_normalization"] == "unicode_casefold_then_isalnum"
    assert config["expected_images"] == 500 and config["expected_care_words"] == 2077
    strict = tmp_path / "strict"
    strict.mkdir()
    candidate = strict / "config.json"
    candidate.write_text(CONFIG.read_text())
    with pytest.raises(ValueError, match="approximate"):
        load_config(candidate)
