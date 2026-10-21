from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import numpy as np
import pytest
import torch

import recognition.parseq as parseq_module
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE,
    PARSEQ_RUNTIME_REVISION,
    ParseqApproxRecognizer,
    verify_parseq_runtime_source,
)


def _make_archive_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    source = tmp_path / "extracted" / ("parseq-" + PARSEQ_RUNTIME_REVISION)
    source.mkdir(parents=True)
    (source / "README.md").write_text("qualified source\n", encoding="utf-8")
    package = source / "strhub"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")

    archive = tmp_path / "parseq.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(source, arcname=source.name)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setattr(parseq_module, "PARSEQ_RUNTIME_ARCHIVE_SHA256", digest)
    return source, archive


class _Tokenizer:
    def decode(self, probabilities):
        return ["ab"], [torch.tensor([0.9, 0.8])]


class _Model:
    tokenizer = _Tokenizer()
    decode_ar = True
    refine_iters = 1

    def __call__(self, tensor):
        assert tensor.shape == (1, 3, 32, 128)
        logits = torch.full((1, 3, 4), -10.0)
        logits[0, 0, 1] = 10
        logits[0, 1, 2] = 10
        logits[0, 2, 0] = 10
        return logits


class _BatchTokenizer:
    def decode(self, probabilities):
        assert probabilities.shape[0] == 2
        return ["ab", "a"], [
            torch.tensor([0.9, 0.8, 0.7]),
            torch.tensor([0.6, 0.5]),
        ]


class _BatchModel:
    tokenizer = _BatchTokenizer()

    def __call__(self, tensor):
        assert tensor.shape == (2, 3, 32, 128)
        logits = torch.full((2, 3, 4), -10.0)
        logits[0, 0, 1] = 10
        logits[0, 1, 2] = 10
        logits[0, 2, 0] = 10
        logits[1, 0, 1] = 10
        logits[1, 1:, 0] = 10
        return logits


def test_real_sequence_contract_preserves_tokens_confidence_and_role():
    recognizer = ParseqApproxRecognizer._from_test_components(
        _Model(), lambda _: torch.zeros(3, 32, 128)
    )
    result = recognizer.recognize(
        np.zeros((17, 91, 3), dtype=np.uint8), input_role="T1_full_image"
    )
    assert result.raw_text == "ab"
    assert result.token_ids == (1, 2)
    assert result.token_confidences == (pytest.approx(0.9), pytest.approx(0.8))
    assert result.confidence == pytest.approx(0.85)
    assert result.preprocessing["full_image_T1_semantics"] == "UNRESOLVED"


def test_candidate_batch_exposes_three_uncombined_confidence_scores():
    recognizer = ParseqApproxRecognizer._from_test_components(
        _BatchModel(), lambda _: torch.zeros(3, 32, 128)
    )
    results = recognizer.recognize_candidate_batch((
        np.zeros((17, 91, 3), dtype=np.uint8),
        np.zeros((23, 52, 3), dtype=np.uint8),
    ))

    assert [value.raw_text for value in results] == ["ab", "a"]
    assert results[0].token_ids == (1, 2)
    assert results[0].token_probabilities == (pytest.approx(0.9), pytest.approx(0.8))
    assert results[0].sequence_confidence_native == pytest.approx(0.9 * 0.8 * 0.7)
    assert results[0].minimum_token_probability == pytest.approx(0.8)
    assert results[0].mean_token_log_probability == pytest.approx(
        sum(results[0].token_log_probabilities) / 2
    )
    assert results[0].predicted_length == 2
    assert results[1].sequence_confidence_native == pytest.approx(0.6 * 0.5)
    assert results[1].predicted_length == 1


def test_candidate_batch_rejects_empty_or_invalid_input():
    recognizer = ParseqApproxRecognizer._from_test_components(
        _BatchModel(), lambda _: torch.zeros(3, 32, 128)
    )
    with pytest.raises(ValueError, match="cannot be empty"):
        recognizer.recognize_candidate_batch([])
    with pytest.raises(ValueError, match="RGB uint8"):
        recognizer.recognize_candidate_batch([np.zeros((17, 91), dtype=np.uint8)])


def test_archive_source_identity_verifies_every_extracted_file(tmp_path, monkeypatch):
    source, archive = _make_archive_fixture(tmp_path, monkeypatch)

    result = verify_parseq_runtime_source(
        source,
        source_type=PARSEQ_ARCHIVE_SOURCE,
        source_archive=archive,
    )

    assert result["runtime_revision"] == PARSEQ_RUNTIME_REVISION
    assert result["source_tree_matches_archive"] is True
    assert result["source_archive_files_verified"] == 2


def test_archive_source_identity_rejects_modified_extraction(tmp_path, monkeypatch):
    source, archive = _make_archive_fixture(tmp_path, monkeypatch)
    (source / "README.md").write_text("modified\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="differs from archive"):
        verify_parseq_runtime_source(
            source,
            source_type=PARSEQ_ARCHIVE_SOURCE,
            source_archive=archive,
        )


def test_archive_source_identity_rejects_untracked_file(tmp_path, monkeypatch):
    source, archive = _make_archive_fixture(tmp_path, monkeypatch)
    (source / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected file"):
        verify_parseq_runtime_source(
            source,
            source_type=PARSEQ_ARCHIVE_SOURCE,
            source_archive=archive,
        )


def test_archive_source_identity_requires_frozen_archive_hash(tmp_path, monkeypatch):
    source, archive = _make_archive_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(parseq_module, "PARSEQ_RUNTIME_ARCHIVE_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="archive SHA256 mismatch"):
        verify_parseq_runtime_source(
            source,
            source_type=PARSEQ_ARCHIVE_SOURCE,
            source_archive=archive,
        )
