from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import numpy as np

from deepsolo import candidates
from deepsolo.candidates import (
    CANDIDATES, DEEPSOLO_SOURCE_REVISION, QUALIFIED_CANDIDATE_ID,
    QUALIFIED_CHECKPOINT_SHA256, QUALIFIED_CONFIG_SHA256, inspect_candidate_source,
    resolve_runtime_candidate, sha256_file, verify_host_source_identity,
)
from deepsolo.reduction import reduce_detections
from deepsolo.worker import _serialize_instances


def test_r50_and_vitae_candidates_remain_separately_named():
    backbones = {candidate.backbone for candidate in CANDIDATES.values()}
    assert backbones == {"ResNet-50", "ViTAEv2-S"}
    assert len(CANDIDATES) == 8
    assert all(candidate.artifact_id == artifact_id for artifact_id, candidate in CANDIDATES.items())
    totaltext = CANDIDATES["ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE"]
    ic15 = CANDIDATES["ART-DEEPSOLO-R50-IC15-CANDIDATE"]
    assert totaltext.training_variant == "Synth150K+MLT17+IC13+IC15"
    assert ic15.training_variant == "Synth150K+Total-Text+MLT17+IC13"
    qualified = CANDIDATES[QUALIFIED_CANDIDATE_ID]
    assert qualified.qualified_config_sha256 == QUALIFIED_CONFIG_SHA256
    assert qualified.qualified_checkpoint_sha256 == QUALIFIED_CHECKPOINT_SHA256
    assert sum(
        candidate.qualified_checkpoint_sha256 is not None
        for candidate in CANDIDATES.values()
    ) == 1


@pytest.mark.parametrize("artifact_id", tuple(CANDIDATES))
def test_candidate_source_config_and_architecture_pairing(monkeypatch, tmp_path: Path, artifact_id: str):
    candidate = CANDIDATES[artifact_id]
    config = tmp_path / candidate.config_relative_path
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('_BASE_: "../Base_det.yaml"\n', encoding="utf-8")
    base = config.parent.parent / "Base_det.yaml"
    builder = (
        'NAME: "build_resnet_backbone"' if candidate.backbone == "ResNet-50"
        else 'NAME: "build_vitaev2_backbone"'
    )
    base.write_text(builder + "\n", encoding="utf-8")
    monkeypatch.setattr(candidates, "source_revision", lambda root: DEEPSOLO_SOURCE_REVISION)
    result = inspect_candidate_source(tmp_path, artifact_id)
    assert result["status"] == "source_and_config_identity_verified"
    assert result["checkpoint_load"].startswith("PENDING")


def _runtime_manifest(path: Path, source_root: str = "/workspace/DeepSolo") -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "source_root": source_root,
        "source_revision": DEEPSOLO_SOURCE_REVISION,
        "candidates": {},
        "runtime": {},
    }), encoding="utf-8")
    return path


def test_host_source_revision_verification_attests_runtime_manifest(monkeypatch, tmp_path: Path):
    source = tmp_path / "DeepSolo"
    source.mkdir()
    manifest = _runtime_manifest(tmp_path / "runtime.json")
    def run(command, **kwargs):
        assert command == ["git", "-C", str(source.resolve()), "rev-parse", "HEAD"]
        return subprocess.CompletedProcess(
            command, 0, stdout=DEEPSOLO_SOURCE_REVISION + "\n", stderr=""
        )
    monkeypatch.setattr(candidates.subprocess, "run", run)
    verified = verify_host_source_identity(manifest, source)
    verification = verified["runtime"]["source_identity_verification"]
    assert verification["source_type"] == "git_checkout"
    assert verification["source_revision"] == DEEPSOLO_SOURCE_REVISION
    assert verification["source_revision_evidence"] == {
        "method": "host_git_rev_parse",
        "git_head": DEEPSOLO_SOURCE_REVISION,
    }
    assert verification["verified"] is True
    assert verification["host_source_root"] == str(source.resolve())


def test_host_source_revision_verification_rejects_wrong_revision(monkeypatch, tmp_path: Path):
    source = tmp_path / "DeepSolo"
    source.mkdir()
    manifest = _runtime_manifest(tmp_path / "runtime.json")
    monkeypatch.setattr(
        candidates.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="0" * 40 + "\n", stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="source revision mismatch"):
        verify_host_source_identity(manifest, source)


def test_host_source_revision_verification_fails_clearly_when_git_is_unavailable(
    monkeypatch, tmp_path: Path,
):
    source = tmp_path / "DeepSolo"
    source.mkdir()
    manifest = _runtime_manifest(tmp_path / "runtime.json")
    monkeypatch.setattr(
        candidates.subprocess,
        "run",
        lambda command, **kwargs: (_ for _ in ()).throw(FileNotFoundError("git")),
    )
    with pytest.raises(RuntimeError, match="Could not verify.*git checkout revision"):
        verify_host_source_identity(manifest, source)


def test_worker_candidate_resolution_uses_required_host_attestation_without_git(
    monkeypatch, tmp_path: Path,
):
    artifact_id = "ART-DEEPSOLO-R50-TOTALTEXT-TEXTOCR-CANDIDATE"
    candidate = CANDIDATES[artifact_id]
    config = tmp_path / candidate.config_relative_path
    config.parent.mkdir(parents=True)
    config.write_text('_BASE_: "../Base_det.yaml"\n', encoding="utf-8")
    (config.parent.parent / "Base_det.yaml").write_text(
        'NAME: "build_resnet_backbone"\n', encoding="utf-8"
    )
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"checkpoint")
    manifest = {
        "schema_version": 1,
        "source_root": str(tmp_path),
        "source_revision": DEEPSOLO_SOURCE_REVISION,
        "candidates": {
            artifact_id: {
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
            },
        },
        "runtime": {
            "source_identity_verification": {
                "method": "host_git_rev_parse",
                "revision": DEEPSOLO_SOURCE_REVISION,
                "verified": True,
                "host_source_root": "/host/verified/checkout",
            },
        },
    }
    monkeypatch.setattr(
        candidates,
        "source_revision",
        lambda root: (_ for _ in ()).throw(AssertionError("worker must not execute git")),
    )
    resolved = resolve_runtime_candidate(manifest, artifact_id)
    assert resolved["source_revision"] == DEEPSOLO_SOURCE_REVISION
    assert resolved["checkpoint_sha256"] == sha256_file(checkpoint)


def test_worker_rejects_manifest_without_host_source_attestation(tmp_path: Path):
    with pytest.raises(RuntimeError, match="not verified by the host adapter"):
        resolve_runtime_candidate(
            {
                "schema_version": 1,
                "source_root": str(tmp_path),
                "source_revision": DEEPSOLO_SOURCE_REVISION,
                "candidates": {},
                "runtime": {},
            },
            "ART-DEEPSOLO-R50-TOTALTEXT-TEXTOCR-CANDIDATE",
        )


def detections():
    return [
        {"text": "RIGHT", "score": 0.9, "geometry": {"boundary_points": [[20, 1], [30, 5]]}},
        {"text": "LOW", "score": 0.7, "geometry": {"boundary_points": [[0, 20], [5, 25]]}},
        {"text": "LEFT", "score": 0.9, "geometry": {"boundary_points": [[0, 1], [10, 5]]}},
    ]


def test_named_reduction_policies_are_distinct_and_deterministic():
    assert reduce_detections(detections(), "highest_score") == "RIGHT"
    assert reduce_detections(detections(), "source_order_space_join") == "RIGHT LOW LEFT"
    assert reduce_detections(detections(), "reading_order_space_join") == "LEFT RIGHT LOW"
    assert reduce_detections([], "highest_score") == ""


def test_reading_order_does_not_silently_accept_missing_geometry():
    with pytest.raises(ValueError, match="geometry"):
        reduce_detections([{"text": "WORD", "score": 1.0}], "reading_order_space_join")


def test_upstream_output_fields_serialize_raw_text_confidence_and_geometry():
    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)
        def tolist(self):
            return self.value.tolist()
        def numpy(self):
            return self.value

    class Instances:
        scores = Tensor([0.9])
        pred_classes = Tensor([4])
        recs = Tensor([[0, 0, 36, 1]])
        rec_scores = Tensor(np.eye(37)[[[0, 0, 36, 1]]])
        bd = Tensor([[[0, 0, 0, 2], [2, 0, 2, 2]]])
        ctrl_points = Tensor([[0, 1, 2, 1]])
        def to(self, device):
            assert device == "cpu"
            return self
        def has(self, field):
            return hasattr(self, field)

    transformer = type("Transformer", (), {"VOC_SIZE": 37, "CUSTOM_DICT": ""})()
    config = type("Config", (), {"MODEL": type("Model", (), {"TRANSFORMER": transformer})()})()
    detection = _serialize_instances(Instances(), config)[0]
    assert detection["text"] == "ab"
    assert detection["score"] == pytest.approx(0.9)
    assert detection["raw"]["predicted_class"] == 4
    assert detection["raw"]["recognition_token_confidences"] == [1.0] * 4
    assert detection["geometry"]["boundary_points"] == [
        [0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0],
    ]
