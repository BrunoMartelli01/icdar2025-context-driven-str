import json

import numpy as np
from PIL import Image
import pytest

from scripts.export_train_v1_parseq_inputs import (
    _candidate_shard_valid, _ocr_shard_valid, _params, _sha,
)


def test_frozen_v1_config_resolves_expected_parameters():
    params = _params(__import__("pathlib").Path(
        "configs/approximate/pipeline_icdar13_practical_remap_v1.yaml"
    ))
    assert params.horizontal_gap_ratio == 0.35
    assert params.vertical_overlap_min == 0.7
    assert params.pad == 6
    assert params.nmax == 10


def test_candidate_and_ocr_shards_are_restart_verifiable(tmp_path):
    crop = tmp_path / "crops" / "0001" / "v1-001.png"
    crop.parent.mkdir(parents=True)
    Image.fromarray(np.zeros((4, 7, 3), dtype=np.uint8), mode="RGB").save(crop)
    candidate = {
        "candidate_id": "v1:0001:001", "rank": 1,
        "crop_path": "crops/0001/v1-001.png", "crop_sha256": _sha(crop),
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps({
        "sample_id": "1", "image_sha256": "a" * 64,
        "candidates": [candidate],
    }))
    value = _candidate_shard_valid(
        candidate_path, sample_id="1", image_sha="a" * 64, root=tmp_path,
    )
    assert value["candidates"][0]["candidate_id"] == "v1:0001:001"

    ocr_path = tmp_path / "ocr.json"
    from recognition.parseq import (
        PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256, PARSEQ_RUNTIME_REVISION,
    )
    ocr_path.write_text(json.dumps({
        "candidate_ids": ["v1:0001:001"],
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION,
        "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "results": [{"candidate_id": "v1:0001:001", "raw_text": "word"}],
    }))
    assert _ocr_shard_valid(
        ocr_path, candidate_ids=["v1:0001:001"]
    )["results"][0]["raw_text"] == "word"

    crop.write_bytes(b"corrupted")
    with pytest.raises(RuntimeError, match="crop SHA mismatch"):
        _candidate_shard_valid(
            candidate_path, sample_id="1", image_sha="a" * 64, root=tmp_path,
        )
