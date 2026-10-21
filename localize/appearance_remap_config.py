"""Explicit approximate-only configuration for the appearance remap prototype."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .appearance_word_remap import MODE, AppearanceRemapParams
from .grouped_word_blocks import GroupedWordBlockParams


@dataclass(frozen=True)
class AppearanceRemapConfig:
    source: Path
    resolved: dict[str, Any]
    v1: GroupedWordBlockParams
    remap: AppearanceRemapParams

    @property
    def resolved_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.resolved, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_appearance_config(path: str | Path) -> AppearanceRemapConfig:
    path = Path(path).resolve()
    if path.parent.name != "approximate" or path.parent.parent.name != "configs":
        raise ValueError("appearance remap requires an explicit configs/approximate/ file")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    expected = {
        "schema_version": 1, "experiment_kind": "appearance_remap_train_diagnostic",
        "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
        "result_class": "PRACTICAL_APPROXIMATION", "table_5_accuracy": None,
        "status": "prototype_not_calibrated_or_promoted",
    }
    for key, item in expected.items():
        if key not in value or value[key] != item:
            raise ValueError(f"invalid or missing {key}; strict/default fallback is forbidden")
    if set(value) != set(expected) | {"localization", "qualification", "scientific_choices"}:
        raise ValueError("unknown/missing root configuration keys; no implicit inheritance")
    localization = value["localization"]
    if set(localization) != {"mode", "v1", "appearance"} or localization["mode"] != MODE:
        raise ValueError("invalid localization mode/configuration")
    v1 = GroupedWordBlockParams(**localization["v1"])
    for key, number in localization["v1"].items():
        if key != "tie_break" and (isinstance(number, bool) or not isinstance(number, (int, float))):
            raise ValueError(f"invalid v1 parameter {key}")
    for key in ("pad", "connectivity", "nmax"):
        if type(localization["v1"][key]) is not int:
            raise ValueError(f"v1 {key} must be an integer")
    remap = AppearanceRemapParams(**localization["appearance"])
    qualification = value["qualification"]
    if set(qualification) != {
        "cache_manifest_sha256", "checkpoint_sha256", "v1_calibration_sha256",
        "split", "sample_count", "gt_word_count", "significant_gt_coverage",
    }:
        raise ValueError("invalid qualification contract")
    for key in ("cache_manifest_sha256", "checkpoint_sha256", "v1_calibration_sha256"):
        digest = qualification[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid {key}")
    if (qualification["split"] != "official-training-all-229-author-protocol"
            or qualification["sample_count"] != 229 or qualification["gt_word_count"] != 849):
        raise ValueError("only official TRAIN-229 / 849 derived care words is supported")
    if not 0 < qualification["significant_gt_coverage"] <= 1:
        raise ValueError("invalid diagnostic significant_gt_coverage")
    if not isinstance(value["scientific_choices"], list) or not value["scientific_choices"]:
        raise ValueError("scientific choices must be a nonempty list")
    for decision in value["scientific_choices"]:
        if decision.get("classification") not in {"PAPER_SPECIFIED", "REPOSITORY_SPECIFIED", "ASSUMPTION", "HARDWARE_DEVIATION", "UNRESOLVED"}:
            raise ValueError("scientific choices require exactly one source classification")
    json.dumps(value, allow_nan=False)
    return AppearanceRemapConfig(path, value, v1, remap)
