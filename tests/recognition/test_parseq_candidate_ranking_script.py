import json
from pathlib import Path

import pytest

from recognition.parseq import (
    PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256,
    PARSEQ_RUNTIME_REVISION,
)
from scripts.diagnose_parseq_candidate_ranking_train import (
    _sha256_file,
    _task22_word_texts,
    _validate_ocr_shard,
)


def test_task22_transcriptions_are_grouped_only_at_blank_records(tmp_path: Path) -> None:
    path = tmp_path / "annotation.txt"
    path.write_text(
        '# 255 0 0 "ignored"\n10 20 30 40 "W"\n11 21 31 41 "o"\n\n'
        '12 22 32 42 "r"\n13 23 33 43 "d"\n',
        encoding="utf-8",
    )
    assert _task22_word_texts(path) == ("Wo", "rd")


def test_task22_transcription_parser_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "annotation.txt"
    path.write_text("malformed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        _task22_word_texts(path)


def test_ocr_shard_requires_exact_candidate_and_runtime_contract(tmp_path: Path) -> None:
    adapter = Path(__file__).resolve().parents[2] / "recognition/parseq.py"
    path = tmp_path / "batch.json"
    shard = {
        "candidate_ids": ["0001:00001"],
        "batch_size": 64,
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION,
        "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "parseq_adapter_sha256": _sha256_file(adapter),
        "results": [{"candidate_id": "0001:00001"}],
    }
    path.write_text(json.dumps(shard), encoding="utf-8")
    assert _validate_ocr_shard(path, ["0001:00001"], 64) == shard["results"]
    shard["candidate_ids"] = ["drift"]
    path.write_text(json.dumps(shard), encoding="utf-8")
    with pytest.raises(RuntimeError, match="provenance/cardinality"):
        _validate_ocr_shard(path, ["0001:00001"], 64)
