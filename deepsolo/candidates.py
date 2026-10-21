"""Provenance-defined DeepSolo candidate identities; none is paper-selected."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


DEEPSOLO_SOURCE_REVISION = "dbadae995035246bad3376c7a44c015c69e9b313"
DETECTRON2_REVISION = "d1e04565d3bec8719335b88be9e9b961bf3ec464"
QUALIFIED_CANDIDATE_ID = "ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE"
QUALIFIED_CONFIG_SHA256 = "0da913239e4b3f11ae1bad19e0b104996f7c3f3607aa2a7f4b1f52e3838b702d"
QUALIFIED_CHECKPOINT_SHA256 = "ac2a01b3d8df0ee42e4632ff57a0d3a20a5706fb0ac1c5ae33b2455d0d90da68"
QUALIFIED_SOURCE_ARCHIVE_SHA256 = "828c63e1c19f07d03ae094e4dd39a1eb093b0660a604bef75324f8e716693bfb"
HOST_SOURCE_VERIFICATION_METHOD = "host_git_rev_parse"
ARCHIVE_SOURCE_VERIFICATION_METHOD = "github_commit_archive_marker_and_sha256"
GIT_CHECKOUT_SOURCE_TYPE = "git_checkout"
GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE = "github_commit_archive"
SOURCE_TYPES = (GIT_CHECKOUT_SOURCE_TYPE, GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE)


@dataclass(frozen=True)
class DeepSoloCandidate:
    artifact_id: str
    config_relative_path: str
    backbone: str
    benchmark: str
    training_variant: str
    qualified_config_sha256: Optional[str] = None
    qualified_checkpoint_sha256: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _candidate(artifact_id: str, backbone: str, benchmark: str, textocr: bool) -> DeepSoloCandidate:
    backbone_dir = "R_50" if backbone == "ResNet-50" else "ViTAEv2_S"
    dataset_dir = "TotalText" if benchmark == "Total-Text" else "IC15"
    suffix = "_textocr" if textocr else ""
    pretraining = (
        "Synth150K+MLT17+IC13+IC15"
        if benchmark == "Total-Text" else "Synth150K+Total-Text+MLT17+IC13"
    )
    qualified = artifact_id == QUALIFIED_CANDIDATE_ID
    return DeepSoloCandidate(
        artifact_id=artifact_id,
        config_relative_path=(
            "configs/{}/{}/finetune_150k_tt_mlt_13_15{}.yaml".format(
                backbone_dir, dataset_dir, suffix
            )
        ),
        backbone=backbone,
        benchmark=benchmark,
        training_variant=pretraining + ("+TextOCR" if textocr else ""),
        qualified_config_sha256=QUALIFIED_CONFIG_SHA256 if qualified else None,
        qualified_checkpoint_sha256=QUALIFIED_CHECKPOINT_SHA256 if qualified else None,
    )


CANDIDATES = {
    item.artifact_id: item
    for item in (
        _candidate("ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE", "ResNet-50", "Total-Text", False),
        _candidate("ART-DEEPSOLO-R50-TOTALTEXT-TEXTOCR-CANDIDATE", "ResNet-50", "Total-Text", True),
        _candidate("ART-DEEPSOLO-VITAEV2S-TOTALTEXT-CANDIDATE", "ViTAEv2-S", "Total-Text", False),
        _candidate("ART-DEEPSOLO-VITAEV2S-TOTALTEXT-TEXTOCR-CANDIDATE", "ViTAEv2-S", "Total-Text", True),
        _candidate("ART-DEEPSOLO-R50-IC15-CANDIDATE", "ResNet-50", "ICDAR15", False),
        _candidate("ART-DEEPSOLO-R50-IC15-TEXTOCR-CANDIDATE", "ResNet-50", "ICDAR15", True),
        _candidate("ART-DEEPSOLO-VITAEV2S-IC15-CANDIDATE", "ViTAEv2-S", "ICDAR15", False),
        _candidate("ART-DEEPSOLO-VITAEV2S-IC15-TEXTOCR-CANDIDATE", "ViTAEv2-S", "ICDAR15", True),
    )
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_revision(source_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_source_identity(
    source_root: Path,
    *,
    source_type: str = GIT_CHECKOUT_SOURCE_TYPE,
    declared_revision: Optional[str] = None,
    source_commit_marker: Optional[Path] = None,
    source_archive: Optional[Path] = None,
    source_archive_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify one supported immutable source form without inferring identity."""
    source_root = source_root.resolve()
    if not source_root.is_dir():
        raise ValueError("DeepSolo source root is missing: {}".format(source_root))
    if source_type == GIT_CHECKOUT_SOURCE_TYPE:
        if any(
            value is not None
            for value in (source_commit_marker, source_archive, source_archive_sha256)
        ):
            raise ValueError("Archive metadata is invalid for a git_checkout source")
        try:
            revision = source_revision(source_root)
        except (OSError, subprocess.CalledProcessError) as error:
            raise RuntimeError(
                "Could not verify the DeepSolo git checkout revision at {}".format(
                    source_root
                )
            ) from error
        if declared_revision is not None and declared_revision != revision:
            raise RuntimeError(
                "Declared DeepSolo source revision does not match Git HEAD"
            )
        if revision != DEEPSOLO_SOURCE_REVISION:
            raise RuntimeError(
                "DeepSolo source revision mismatch: expected {}, got {}".format(
                    DEEPSOLO_SOURCE_REVISION, revision
                )
            )
        return {
            "source_type": GIT_CHECKOUT_SOURCE_TYPE,
            "source_revision": revision,
            "source_revision_evidence": {
                "method": HOST_SOURCE_VERIFICATION_METHOD,
                "git_head": revision,
            },
            "source_archive": None,
            "source_archive_sha256": None,
        }
    if source_type != GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE:
        raise ValueError("Unsupported DeepSolo source type: {}".format(source_type))
    if declared_revision is None:
        raise ValueError("github_commit_archive requires an explicit source revision")
    if declared_revision != DEEPSOLO_SOURCE_REVISION:
        raise RuntimeError(
            "DeepSolo source revision mismatch: expected {}, got {}".format(
                DEEPSOLO_SOURCE_REVISION, declared_revision
            )
        )
    if source_commit_marker is None:
        raise ValueError("github_commit_archive requires a source commit marker")
    if source_archive is None:
        raise ValueError("github_commit_archive requires the source archive")
    if source_archive_sha256 is None:
        raise ValueError("github_commit_archive requires an explicit archive SHA256")
    if not re.fullmatch(r"[0-9a-f]{64}", source_archive_sha256):
        raise ValueError("Source archive SHA256 must be 64 lowercase hexadecimal characters")
    if source_archive_sha256 != QUALIFIED_SOURCE_ARCHIVE_SHA256:
        raise RuntimeError("DeepSolo source archive SHA256 is not the qualified digest")
    marker_path = source_commit_marker.resolve()
    archive_path = source_archive.resolve()
    if not marker_path.is_file():
        raise ValueError("DeepSolo source commit marker is missing: {}".format(marker_path))
    if not archive_path.is_file():
        raise ValueError("DeepSolo source archive is missing: {}".format(archive_path))
    marker_bytes = marker_path.read_bytes()
    allowed_marker_bytes = (
        declared_revision.encode("ascii"),
        (declared_revision + "\n").encode("ascii"),
    )
    if marker_bytes not in allowed_marker_bytes:
        raise RuntimeError(
            "DeepSolo source commit marker does not exactly match the declared revision"
        )
    actual_archive_sha256 = sha256_file(archive_path)
    if actual_archive_sha256 != source_archive_sha256:
        raise RuntimeError(
            "DeepSolo source archive SHA256 mismatch: expected {}, got {}".format(
                source_archive_sha256, actual_archive_sha256
            )
        )
    return {
        "source_type": GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE,
        "source_revision": declared_revision,
        "source_revision_evidence": {
            "method": ARCHIVE_SOURCE_VERIFICATION_METHOD,
            "commit_marker_path": str(marker_path),
            "commit_marker_sha256": sha256_file(marker_path),
            "commit_marker_revision": declared_revision,
        },
        "source_archive": str(archive_path),
        "source_archive_sha256": actual_archive_sha256,
    }


def verify_host_source_identity(
    runtime_manifest: Path, host_source_root: Path
) -> Dict[str, Any]:
    """Verify the checkout on the host and return an invocation-scoped manifest."""
    manifest_path = runtime_manifest.resolve()
    source_root = host_source_root.resolve()
    if not manifest_path.is_file():
        raise ValueError("DeepSolo runtime manifest is missing: {}".format(manifest_path))
    if not source_root.is_dir():
        raise ValueError("DeepSolo host source root is missing: {}".format(source_root))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("DeepSolo runtime manifest is not valid JSON") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise ValueError("DeepSolo runtime manifest schema_version must be 1")
    if manifest.get("source_revision") != DEEPSOLO_SOURCE_REVISION:
        raise RuntimeError("Runtime manifest does not pin the qualified DeepSolo revision")
    source_type = str(manifest.get("source_type", GIT_CHECKOUT_SOURCE_TYPE))
    evidence = manifest.get("source_revision_evidence", {})
    if not isinstance(evidence, Mapping):
        raise ValueError("DeepSolo source revision evidence must be an object")
    identity = verify_source_identity(
        source_root,
        source_type=source_type,
        declared_revision=str(manifest["source_revision"]),
        source_commit_marker=(
            Path(str(evidence["commit_marker_path"]))
            if "commit_marker_path" in evidence else None
        ),
        source_archive=(
            Path(str(manifest["source_archive"]))
            if manifest.get("source_archive") else None
        ),
        source_archive_sha256=(
            str(manifest["source_archive_sha256"])
            if manifest.get("source_archive_sha256") else None
        ),
    )
    verified = deepcopy(dict(manifest))
    runtime = verified.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ValueError("DeepSolo runtime manifest runtime metadata must be an object")
    verified["runtime"] = {
        **dict(runtime),
        "source_identity_verification": {
            **identity,
            "verified": True,
            "host_source_root": str(source_root),
        },
    }
    return verified


def _verified_source_identity_from_manifest(
    manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping):
        raise RuntimeError("DeepSolo runtime manifest has no runtime metadata")
    verification = runtime.get("source_identity_verification")
    if not isinstance(verification, Mapping):
        raise RuntimeError(
            "DeepSolo source identity was not verified by the host adapter"
        )
    source_type = str(verification.get("source_type", GIT_CHECKOUT_SOURCE_TYPE))
    revision = str(
        verification.get("source_revision", verification.get("revision", ""))
    )
    evidence = verification.get("source_revision_evidence")
    if evidence is None and verification.get("method") is not None:
        evidence = {"method": verification.get("method")}
    expected_method = (
        HOST_SOURCE_VERIFICATION_METHOD
        if source_type == GIT_CHECKOUT_SOURCE_TYPE
        else ARCHIVE_SOURCE_VERIFICATION_METHOD
    )
    if (
        source_type not in SOURCE_TYPES
        or verification.get("verified") is not True
        or revision != DEEPSOLO_SOURCE_REVISION
        or not isinstance(evidence, Mapping)
        or evidence.get("method") != expected_method
    ):
        raise RuntimeError("DeepSolo host source identity verification is invalid")
    archive_sha256 = verification.get("source_archive_sha256")
    if source_type == GITHUB_COMMIT_ARCHIVE_SOURCE_TYPE and not (
        isinstance(archive_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", archive_sha256)
    ):
        raise RuntimeError("DeepSolo archive source identity verification is invalid")
    return {
        "source_type": source_type,
        "source_revision": revision,
        "source_revision_evidence": dict(evidence),
        "source_archive": verification.get("source_archive"),
        "source_archive_sha256": archive_sha256,
    }


def inspect_candidate_source(
    source_root: Path,
    artifact_id: str,
    *,
    verified_source_revision: Optional[str] = None,
    verified_source_identity: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Verify immutable source/config identity without importing legacy packages."""
    candidate = CANDIDATES.get(artifact_id)
    if candidate is None:
        raise ValueError("Unknown provenance-defined DeepSolo candidate: {}".format(artifact_id))
    source_root = source_root.resolve()
    if verified_source_revision is not None and verified_source_identity is not None:
        raise ValueError("Pass only one form of verified DeepSolo source identity")
    if verified_source_identity is not None:
        revision = str(verified_source_identity.get("source_revision", ""))
        source_identity = dict(verified_source_identity)
    elif verified_source_revision is not None:
        revision = verified_source_revision
        source_identity = {
            "source_type": GIT_CHECKOUT_SOURCE_TYPE,
            "source_revision": revision,
            "source_revision_evidence": {"method": HOST_SOURCE_VERIFICATION_METHOD},
            "source_archive": None,
            "source_archive_sha256": None,
        }
    else:
        source_identity = verify_source_identity(source_root)
        revision = str(source_identity["source_revision"])
    if revision != DEEPSOLO_SOURCE_REVISION:
        raise RuntimeError("DeepSolo source revision mismatch: {}".format(revision))
    config_path = source_root / candidate.config_relative_path
    if not config_path.is_file():
        raise RuntimeError("Candidate config is missing: {}".format(config_path))
    config_text = config_path.read_text(encoding="utf-8")
    base_match = re.search(r'_BASE_:\s*"([^"]+)"', config_text)
    if not base_match:
        raise RuntimeError("Candidate config does not declare its official base config")
    base_path = (config_path.parent / base_match.group(1)).resolve()
    base_text = base_path.read_text(encoding="utf-8")
    expected_builder = (
        'NAME: "build_resnet_backbone"'
        if candidate.backbone == "ResNet-50" else 'NAME: "build_vitaev2_backbone"'
    )
    if expected_builder not in base_text:
        raise RuntimeError("Candidate config/backbone pairing does not match provenance")
    return {
        "status": "source_and_config_identity_verified",
        "candidate": candidate.to_dict(),
        **source_identity,
        "source_revision": revision,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "base_config_path": str(base_path),
        "base_config_sha256": sha256_file(base_path),
        "checkpoint_load": "PENDING: checkpoint and legacy runtime required",
        "inference_output": "PENDING: checkpoint and legacy runtime required",
    }


def resolve_runtime_candidate(
    manifest: Mapping[str, Any], artifact_id: str
) -> Dict[str, Any]:
    """Resolve an explicit checkpoint while preventing config/candidate substitution."""
    if manifest.get("schema_version") != 1:
        raise ValueError("DeepSolo runtime manifest schema_version must be 1")
    verified_identity = _verified_source_identity_from_manifest(manifest)
    source_root = Path(str(manifest.get("source_root", ""))).resolve()
    inspected = inspect_candidate_source(
        source_root,
        artifact_id,
        verified_source_identity=verified_identity,
    )
    if manifest.get("source_revision") != DEEPSOLO_SOURCE_REVISION:
        raise RuntimeError("Runtime manifest does not pin the qualified DeepSolo revision")
    candidate_entries = manifest.get("candidates")
    if not isinstance(candidate_entries, Mapping) or artifact_id not in candidate_entries:
        raise RuntimeError("Runtime manifest has no checkpoint for {}".format(artifact_id))
    entry = candidate_entries[artifact_id]
    if not isinstance(entry, Mapping):
        raise ValueError("Runtime candidate entry must be an object")
    checkpoint = Path(str(entry.get("checkpoint_path", ""))).resolve()
    expected_digest = str(entry.get("checkpoint_sha256", ""))
    if not checkpoint.is_file():
        raise RuntimeError("DeepSolo checkpoint is missing: {}".format(checkpoint))
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("Runtime manifest requires an immutable lowercase checkpoint SHA256")
    actual_digest = sha256_file(checkpoint)
    if actual_digest != expected_digest:
        raise RuntimeError("DeepSolo checkpoint SHA256 mismatch")
    candidate = CANDIDATES[artifact_id]
    if (
        candidate.qualified_config_sha256 is not None
        and inspected["config_sha256"] != candidate.qualified_config_sha256
    ):
        raise RuntimeError("DeepSolo config SHA256 does not match the qualified candidate")
    if (
        candidate.qualified_checkpoint_sha256 is not None
        and actual_digest != candidate.qualified_checkpoint_sha256
    ):
        raise RuntimeError("DeepSolo checkpoint SHA256 does not match the qualified candidate")
    return {
        **inspected,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_digest,
        "runtime_metadata": dict(manifest.get("runtime", {})),
    }
