import pytest
import sys

from caption.blip2_captioner import (
    DEFAULT_GENERATION, PRACTICAL_ANTI_REPETITION_GENERATION,
)
from scripts.export_train_blip2_captions import _args, _caption_contract, _validate_shard


def _valid():
    contract = _caption_contract()
    return {
        "schema_version": 1,
        "sample_id": "1",
        "image_sha256": "a" * 64,
        "caption_contract": contract,
        "result": {
            "raw_caption": "a street sign",
            "model_id": contract["model_id"],
            "revision": contract["revision"],
            "prompt_hash": contract["prompt_sha256"],
            "generation_config": dict(DEFAULT_GENERATION),
            "generation_config_hash": contract["generation_sha256"],
            "device": "cuda:0",
            "dtype": "float16",
            "placement": "single_device",
            "quantization": "none",
            "success": True,
        },
    }


def test_caption_shard_accepts_exact_frozen_contract():
    _validate_shard(_valid(), sample_id="1", image_sha256="a" * 64)


@pytest.mark.parametrize("field,value", [("dtype", "float32"), ("device", "cpu")])
def test_caption_shard_rejects_runtime_drift(field, value):
    shard = _valid()
    shard["result"][field] = value
    with pytest.raises(RuntimeError, match="contract mismatch"):
        _validate_shard(shard, sample_id="1", image_sha256="a" * 64)


def test_caption_shard_rejects_empty_caption():
    shard = _valid()
    shard["result"]["raw_caption"] = "  "
    with pytest.raises(RuntimeError, match="contract mismatch"):
        _validate_shard(shard, sample_id="1", image_sha256="a" * 64)


def test_practical_antirepetition_contract_is_separate_from_repository_contract():
    baseline = _caption_contract()
    practical = _caption_contract("practical-antirepetition-v1")
    assert "generation_profile" not in baseline
    assert practical["generation_profile"] == "practical-antirepetition-v1"
    assert practical["generation"] == PRACTICAL_ANTI_REPETITION_GENERATION
    shard = _valid()
    shard["caption_contract"] = practical
    shard["result"]["generation_config"] = dict(PRACTICAL_ANTI_REPETITION_GENERATION)
    shard["result"]["generation_config_hash"] = practical["generation_sha256"]
    _validate_shard(
        shard, sample_id="1", image_sha256="a" * 64,
        profile="practical-antirepetition-v1",
    )


def test_practical_caption_cli_does_not_require_parseq_artifacts(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "export_train_blip2_captions",
        "--train-manifest", "train.yaml",
        "--data-root", "data",
        "--hf-cache", "hf",
        "--device", "cuda:0",
        "--generation-profile", "practical-antirepetition-v1",
        "--output-dir", "persistent-output",
    ])
    args = _args()
    assert args.source_report is None
    assert args.candidate_generation_manifest is None
