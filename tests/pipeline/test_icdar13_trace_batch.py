from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_icdar13_trace_batch import _summarize_result


CONFIG_SHA = "a" * 64
IMAGE_SHA = "b" * 64


def write_result(path: Path, **overrides) -> None:
    value = {
        "success": True,
        "reproduction_mode": "APPROXIMATE",
        "claims": {"table_5": False, "table_6": False, "table_7": False},
        "image": {"file_sha256": IMAGE_SHA},
        "configuration": {"resolved_yaml_sha256": CONFIG_SHA},
        "segmentation": {
            "mask_artifact": {"path": "mask.png"},
            "localized_blocks": [{"rank": 1}, {"rank": 2}],
        },
        "scores": {"C": 0.1},
        "fallback": {
            "needs_fallback": True,
            "invoked": True,
            "recognized_strings": ["one", "two"],
        },
        "final_prediction": "one",
    }
    value.update(overrides)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_batch_summary_preserves_routing_without_accuracy(tmp_path: Path) -> None:
    result = tmp_path / "result.json"
    write_result(result)
    summary = _summarize_result(
        result,
        expected_image_sha256=IMAGE_SHA,
        expected_config_sha256=CONFIG_SHA,
    )
    assert summary["success"] is True
    assert summary["confidence"] == 0.1
    assert summary["fallback_invoked"] is True
    assert summary["fallback_string_count"] == 2
    assert summary["localized_block_count"] == 2
    assert summary["intermediates_recorded"] is True
    assert "accuracy" not in summary


@pytest.mark.parametrize(
    "override,error",
    [
        ({"claims": {"table_5": True}}, "claim Table 5"),
        ({"reproduction_mode": "STRICT"}, "not explicitly APPROXIMATE"),
        ({"image": {"file_sha256": "c" * 64}}, "image SHA256 mismatch"),
        ({"configuration": {"resolved_yaml_sha256": "d" * 64}}, "configuration SHA256 mismatch"),
    ],
)
def test_batch_summary_fails_closed_on_identity_or_claim_drift(
    override: dict, error: str, tmp_path: Path
) -> None:
    result = tmp_path / "result.json"
    write_result(result, **override)
    with pytest.raises(RuntimeError, match=error):
        _summarize_result(
            result,
            expected_image_sha256=IMAGE_SHA,
            expected_config_sha256=CONFIG_SHA,
        )
