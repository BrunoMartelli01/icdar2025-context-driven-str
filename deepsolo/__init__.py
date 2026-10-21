"""Versioned process boundary for the isolated APPROXIMATE DeepSolo runtime."""

from .adapter import DeepSoloDockerAdapter, DeepSoloProcessError, DeepSoloSubprocessAdapter
from .candidates import CANDIDATES, DEEPSOLO_SOURCE_REVISION, DeepSoloCandidate
from .contracts import CONTRACT_SCHEMA_VERSION, DeepSoloRequest, DeepSoloResponse
from .reduction import REDUCTION_POLICIES, reduce_detections

__all__ = [
    "CANDIDATES",
    "CONTRACT_SCHEMA_VERSION",
    "DEEPSOLO_SOURCE_REVISION",
    "DeepSoloCandidate",
    "DeepSoloDockerAdapter",
    "DeepSoloProcessError",
    "DeepSoloRequest",
    "DeepSoloResponse",
    "DeepSoloSubprocessAdapter",
    "REDUCTION_POLICIES",
    "reduce_detections",
]
