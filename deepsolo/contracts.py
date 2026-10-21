"""Dependency-free JSON contract shared by the main and DeepSolo environments."""

from __future__ import annotations

import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from .candidates import CANDIDATES, sha256_file
from .reduction import REDUCTION_POLICIES


CONTRACT_SCHEMA_VERSION = 1


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("{} must be a lowercase SHA256".format(field))
    return value


@dataclass(frozen=True)
class DeepSoloRequest:
    schema_version: int
    request_id: str
    image_path: str
    image_sha256: str
    candidate_id: str
    inference_options: Dict[str, Any]
    reduction_policy: str
    reproduction_mode: str = "APPROXIMATE"

    @classmethod
    def create(
        cls,
        image_path: Path,
        candidate_id: str,
        inference_options: Mapping[str, Any],
        reduction_policy: str,
        request_id: str = "",
    ) -> "DeepSoloRequest":
        path = image_path.resolve()
        if not path.is_file():
            raise ValueError("DeepSolo input image does not exist: {}".format(path))
        return cls(
            schema_version=CONTRACT_SCHEMA_VERSION,
            request_id=request_id or str(uuid.uuid4()),
            image_path=str(path),
            image_sha256=sha256_file(path),
            candidate_id=candidate_id,
            inference_options=dict(inference_options),
            reduction_policy=reduction_policy,
        ).validated()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeepSoloRequest":
        required = {
            "schema_version", "request_id", "image_path", "image_sha256",
            "candidate_id", "inference_options", "reduction_policy", "reproduction_mode",
        }
        if set(value) != required:
            raise ValueError("DeepSolo request fields do not match schema version 1")
        options = value["inference_options"]
        if not isinstance(options, Mapping):
            raise ValueError("inference_options must be an object")
        return cls(
            schema_version=value["schema_version"],
            request_id=value["request_id"],
            image_path=value["image_path"],
            image_sha256=value["image_sha256"],
            candidate_id=value["candidate_id"],
            inference_options=dict(options),
            reduction_policy=value["reduction_policy"],
            reproduction_mode=value["reproduction_mode"],
        ).validated()

    def validated(self) -> "DeepSoloRequest":
        if self.schema_version != CONTRACT_SCHEMA_VERSION:
            raise ValueError("Unsupported DeepSolo request schema_version")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("request_id must be a non-empty string")
        if self.candidate_id not in CANDIDATES:
            raise ValueError("Unknown provenance-defined DeepSolo candidate")
        _require_sha256(self.image_sha256, "image_sha256")
        if self.reduction_policy not in REDUCTION_POLICIES:
            raise ValueError("An explicit named DeepSolo reduction policy is required")
        if self.reproduction_mode != "APPROXIMATE":
            raise ValueError("DeepSolo candidate execution is available only in APPROXIMATE mode")
        allowed = {"device", "score_threshold", "lexicon_type"}
        if set(self.inference_options) - allowed:
            raise ValueError("Unknown DeepSolo inference option")
        device = self.inference_options.get("device")
        if device not in ("cpu", "cuda", "cuda:0"):
            raise ValueError("DeepSolo device must be cpu, cuda, or cuda:0")
        threshold = self.inference_options.get("score_threshold")
        if threshold is not None and (
            not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0
        ):
            raise ValueError("score_threshold must be null or in [0,1]")
        lexicon = self.inference_options.get("lexicon_type")
        if lexicon is not None and (not isinstance(lexicon, int) or lexicon not in (1, 2, 3)):
            raise ValueError("lexicon_type must be null, 1, 2, or 3")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeepSoloResponse:
    payload: Dict[str, Any]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeepSoloResponse":
        payload = dict(value)
        required = {
            "schema_version", "request_id", "success", "detections",
            "recognized_strings", "selected_output", "reduction_policy",
            "artifact", "latency", "placement", "memory", "error",
        }
        if set(payload) != required:
            raise ValueError("DeepSolo response fields do not match schema version 1")
        if payload["schema_version"] != CONTRACT_SCHEMA_VERSION:
            raise ValueError("Unsupported DeepSolo response schema_version")
        if not isinstance(payload["success"], bool):
            raise ValueError("DeepSolo response success must be boolean")
        if not isinstance(payload["detections"], list):
            raise ValueError("DeepSolo response detections must be a list")
        for detection in payload["detections"]:
            if not isinstance(detection, Mapping):
                raise ValueError("Each DeepSolo detection must be an object")
            required_detection = {
                "source_index", "text", "score", "score_type", "geometry", "raw",
            }
            if set(detection) != required_detection:
                raise ValueError("DeepSolo detection fields do not match schema version 1")
            if not isinstance(detection["source_index"], int):
                raise ValueError("DeepSolo detection source_index must be an integer")
            if not isinstance(detection["text"], str):
                raise ValueError("DeepSolo detection text must be a string")
            if not isinstance(detection["score"], (int, float)):
                raise ValueError("DeepSolo detection score must be numeric")
            if not isinstance(detection["geometry"], Mapping):
                raise ValueError("DeepSolo detection geometry must be an object")
            if not isinstance(detection["raw"], Mapping):
                raise ValueError("DeepSolo raw detection must be an object")
        if not isinstance(payload["recognized_strings"], list):
            raise ValueError("DeepSolo recognized_strings must be a list")
        if not all(isinstance(value, str) for value in payload["recognized_strings"]):
            raise ValueError("Each DeepSolo recognized string must be text")
        if payload["reduction_policy"] not in REDUCTION_POLICIES:
            raise ValueError("DeepSolo response reduction policy is invalid")
        if payload["success"]:
            if payload["error"] is not None or not isinstance(payload["selected_output"], str):
                raise ValueError("Successful DeepSolo response has inconsistent error/output")
        elif not isinstance(payload["error"], Mapping):
            raise ValueError("Failed DeepSolo response requires a structured error")
        for field in ("artifact", "latency", "placement", "memory"):
            if not isinstance(payload[field], Mapping):
                raise ValueError("DeepSolo response {} must be an object".format(field))
        return cls(payload)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.payload)

    @property
    def success(self) -> bool:
        return bool(self.payload["success"])

    @property
    def selected_output(self) -> str:
        return str(self.payload["selected_output"])
