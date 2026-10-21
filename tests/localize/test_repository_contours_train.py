import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from diagnostics import repository_contours_train as run
from localize.grouped_word_blocks import GroupedWordBlockParams
from localize.repository_contours import RepositoryContourParams


def test_native_cache_roundtrip_and_integrity():
    rng = np.random.default_rng(14)
    mask = rng.integers(0, 2, (128, 128), dtype=np.uint8)
    value = run.pack_mask(mask)
    np.testing.assert_array_equal(mask, run.unpack_mask(value))
    assert len(value["bits"]) < 2800
    with pytest.raises(ValueError, match="checksum"):
        run.unpack_mask({**value, "array_sha256": "wrong"})
    with pytest.raises(ValueError):
        run.unpack_mask({**value, "bits": "AA=="})
    with pytest.raises(ValueError):
        run.pack_mask(np.ones((128, 128), dtype=float))


def test_resumable_shard_guard(tmp_path):
    path = tmp_path / "cache" / "sample.json"
    run.save_shard(path, "frozen", {"text": "word"})
    assert run.read_shard(path, "frozen") == {"text": "word"}
    with pytest.raises(RuntimeError):
        run.read_shard(path, "different")
    data = json.loads(path.read_text())
    data["value"]["text"] = "corrupted"
    path.write_text(json.dumps(data))
    with pytest.raises(RuntimeError):
        run.read_shard(path, "frozen")


def test_cache_sample_identity(tmp_path):
    path = tmp_path / "sample.json"
    run.save_shard(path, "contract", {"sample_id": "100", "image_sha256": "image"})
    good = SimpleNamespace(sample_id="100", image_sha256="image")
    assert run.read_sample(path, "contract", good)["sample_id"] == "100"
    with pytest.raises(RuntimeError, match="identity"):
        run.read_sample(path, "contract", SimpleNamespace(sample_id="101", image_sha256="image"))


def test_proposals_same_mask_no_mutation():
    mask = np.zeros((60, 80), np.uint8)
    mask[2:22, 2] = 1
    mask[21, 2:22] = 1
    mask[30:40, 40:60] = 1
    before = mask.copy()
    rgb = np.zeros((60, 80, 3), np.uint8)
    groups = GroupedWordBlockParams(.35, .7, .5, 2.5, pad=6, connectivity=8, nmax=10)
    result = run.proposals(mask, rgb, RepositoryContourParams(150, 6, 10), groups)
    assert len(result["CURRENT_ALG1"]["boxes"]) == 1
    assert len(result["REPOSITORY_CONTOURS"]["boxes"]) == 2
    assert len(result["PRACTICAL_REMAP_V1"]["boxes"]) > 0
    np.testing.assert_array_equal(mask, before)


def test_posthoc_assignment_uses_geometry_not_ocr(monkeypatch):
    gt = [SimpleNamespace(bbox=(0, 0, 10, 10)), SimpleNamespace(bbox=(20, 0, 30, 10))]
    monkeypatch.setattr(run, "train_task22_word_boxes", lambda *args: gt)
    monkeypatch.setattr(run, "_texts", lambda *args: ("ONE", "TWO"))
    best = [0, 0, 10, 10]
    worse = [0, 0, 8, 8]
    methods = {m: {"boxes": [best, worse], "precap": 2, "merges": 0, "multi": 0} for m in run.METHODS}
    record = {"methods": methods, "texts": {json.dumps(best): "wrong", json.dumps(worse): "ONE"}}
    sample = SimpleNamespace(sample_id="fixture", mask_path="gt", validity_annotation_path="annotation")
    summary, rows, crops = run.evaluate([record], Path("/unused"), [sample], expected_gt_words=2)
    for method in run.METHODS:
        assert summary[method]["ocr"]["exact_casefold"] == 0
        assert summary[method]["ocr"]["available_predictions"] == 1
        assert summary[method]["geometry"]["gt_words"] == 2
        assert rows[0]["methods"][method]["text"] == "wrong"
        assert rows[1]["methods"][method]["text"] is None
    assert len(crops) == 6
    with pytest.raises(RuntimeError, match="849"):
        run.evaluate([record], Path("/unused"), [sample])
