from __future__ import annotations

import socket
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from pipeline.config import (
    ConfigValidationError,
    get_required,
    load_config,
    parse_config,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]
STRICT = ROOT / "configs" / "strict" / "pipeline.yaml"
APPROXIMATE = ROOT / "configs" / "approximate" / "pipeline.yaml"
DEMO = ROOT / "configs" / "demo" / "pipeline.yaml"
TABLE5_SMOKE = ROOT / "configs" / "approximate" / "pipeline_icdar13_table5_smoke.yaml"


def _strict_skeleton() -> dict:
    return {
        "reproduction_mode": "strict",
        "claim_label": "strict reproduction",
        "recognition": {"primary": {"implementation": "parseq_s"}},
        "fallback": {"enabled": True, "implementation": "deepsolo"},
        "segmentation": {},
        "caption": {},
        "scoring": {},
    }


def test_strict_config_parses() -> None:
    assert parse_config(STRICT).reproduction_mode == "strict"


def test_approximate_config_parses() -> None:
    assert parse_config(APPROXIMATE).reproduction_mode == "approximate"


def test_table5_smoke_config_is_approximate_and_cannot_claim_table5() -> None:
    cfg = load_config(TABLE5_SMOKE)
    assert cfg.run.table_5_claim is False
    assert cfg.segmentation.probability_threshold == 0.5
    assert cfg.segmentation.validity_mask == "evaluation_only_not_applied_downstream"
    assert (cfg.scoring.alpha, cfg.scoring.beta, cfg.scoring.tau) == (0.6, 0.4, 0.8)
    assert cfg.fallback.enabled is True
    assert cfg.fallback.transport == "local_python_subprocess"
    assert cfg.fallback.runtime_artifact_id == "ART-DEEPSOLO-PAPER-HOST-RUNTIME-CANDIDATE"


def test_demo_config_parses() -> None:
    assert parse_config(DEMO).reproduction_mode == "demo"


def test_nested_config_retrieval() -> None:
    cfg = OmegaConf.create({"one": {"two": {"value": 42}}})
    assert get_required(cfg, "one.two.value") == 42
    with pytest.raises(ConfigValidationError, match="missing or empty"):
        get_required(cfg, "one.missing")


def test_missing_strict_values_fail() -> None:
    cfg = OmegaConf.create(_strict_skeleton())
    with pytest.raises(ConfigValidationError, match="segmentation.architecture_artifact_id"):
        validate_config(cfg)


def test_strict_mode_rejects_approximate_parseq_checkpoint_and_runtime() -> None:
    data = _strict_skeleton()
    data["recognition"]["primary"].update(
        {
            "checkpoint_artifact_id": "ART-PARSEQ-V100-PARSEQ-CANDIDATE",
            "runtime_artifact_id": "ART-PARSEQ-PYTORCH2-RUNTIME-CANDIDATE",
        }
    )
    with pytest.raises(ConfigValidationError, match="requires approximate mode"):
        validate_config(OmegaConf.create(data))


def test_approximate_mode_accepts_qualified_parseq_pairing() -> None:
    cfg = load_config(APPROXIMATE)
    assert cfg.recognition.primary.checkpoint_artifact_id == "ART-PARSEQ-V100-PARSEQ-CANDIDATE"
    assert cfg.recognition.primary.runtime_artifact_id == "ART-PARSEQ-PYTORCH2-RUNTIME-CANDIDATE"


def test_checkpoint_and_runtime_provenance_remain_separate() -> None:
    cfg = load_config(APPROXIMATE)
    assert cfg.recognition.primary.checkpoint_artifact_id != cfg.recognition.primary.runtime_artifact_id


def test_strict_mode_rejects_demo_trocr() -> None:
    data = _strict_skeleton()
    data["recognition"]["primary"]["implementation"] = "trocr"
    with pytest.raises(ConfigValidationError, match="requires PARSeq-S"):
        validate_config(OmegaConf.create(data))


def test_strict_mode_cannot_disable_deepsolo() -> None:
    data = _strict_skeleton()
    data["fallback"]["enabled"] = False
    with pytest.raises(ConfigValidationError, match="cannot disable"):
        validate_config(OmegaConf.create(data))


def test_unknown_artifact_id_fails() -> None:
    cfg = parse_config(APPROXIMATE)
    cfg.recognition.primary.checkpoint_artifact_id = "ART-DOES-NOT-EXIST"
    with pytest.raises(ConfigValidationError, match="Unknown artifact ID"):
        validate_config(cfg)


def test_blocking_artifact_fails_in_strict_mode() -> None:
    data = _strict_skeleton()
    safe = "ART-PAPER-PDF-V2"
    data["segmentation"].update(
        {
            "architecture_artifact_id": safe,
            "checkpoint_artifact_id": "ART-AGUNET-AUTHOR-CHECKPOINT",
        }
    )
    data["recognition"]["primary"].update(
        {"checkpoint_artifact_id": safe, "runtime_artifact_id": safe}
    )
    data["caption"]["checkpoint_artifact_id"] = safe
    data["scoring"]["embedder_artifact_id"] = safe
    data["fallback"]["checkpoint_artifact_id"] = safe
    with pytest.raises(ConfigValidationError, match="BLOCKING"):
        validate_config(OmegaConf.create(data))


def test_config_validation_never_uses_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_socket(*args, **kwargs):
        raise AssertionError("configuration validation attempted network access")

    monkeypatch.setattr(socket, "socket", blocked_socket)
    load_config(APPROXIMATE)
    load_config(DEMO)
    with pytest.raises(ConfigValidationError):
        load_config(STRICT)
