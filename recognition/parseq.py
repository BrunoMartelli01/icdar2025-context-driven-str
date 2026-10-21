"""Provenance-locked APPROXIMATE PARSeq sequence recognition.

This adapter uses the already-qualified official runtime/checkpoint pairing.  It
does not identify the paper's unknown PARSeq-S artifact, and applying a
single-instance recognizer to the full scene (T1) remains unresolved.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
from PIL import Image


PARSEQ_CHECKPOINT_NAME = "parseq-bb5792a6.pt"
PARSEQ_CHECKPOINT_SHA256 = "bb5792a68e367476abca029cbf8699abc805f3d3dc7e57aae45c8ec4f7b7cd00"
PARSEQ_RUNTIME_REVISION = "1902db043c029a7e03a3818c616c06600af574be"
PARSEQ_RUNTIME_ARCHIVE_SHA256 = (
    "f57289cb1fc7fe41f488a52ea48f8a7f114252ef405b0b64efc1d528c29b48ef"
)
PARSEQ_GIT_SOURCE = "git_checkout"
PARSEQ_ARCHIVE_SOURCE = "github_commit_archive"
PARSEQ_SOURCE_TYPES = (PARSEQ_GIT_SOURCE, PARSEQ_ARCHIVE_SOURCE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_source_identity(source: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Could not verify the PARSeq Git checkout at {}".format(source)
        ) from error
    if revision != PARSEQ_RUNTIME_REVISION:
        raise RuntimeError("PARSeq runtime revision mismatch: {}".format(revision))
    if dirty:
        raise RuntimeError("PARSeq runtime Git checkout is not clean")
    return {
        "source_type": PARSEQ_GIT_SOURCE,
        "runtime_revision": revision,
        "source_clean": True,
        "source_tree_matches_archive": None,
        "source_archive": None,
        "source_archive_sha256": None,
    }


def _archive_member_path(name: str) -> tuple[str, ...]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("PARSeq source archive contains an unsafe member path")
    parts = path.parts
    expected_root = "parseq-{}".format(PARSEQ_RUNTIME_REVISION)
    if not parts or parts[0] != expected_root:
        raise RuntimeError("PARSeq source archive has an unexpected root directory")
    return tuple(parts[1:])


def _archive_source_identity(source: Path, archive: Path | None) -> dict[str, Any]:
    if archive is None:
        raise ValueError("github_commit_archive requires --parseq-source-archive")
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError("PARSeq source archive is missing: {}".format(archive))
    archive_digest = _sha256(archive)
    if archive_digest != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise RuntimeError("PARSeq source archive SHA256 mismatch")

    archive_files: dict[tuple[str, ...], tuple[str, bool]] = {}
    archive_directories: set[tuple[str, ...]] = {()}
    with tarfile.open(archive, mode="r:gz") as stream:
        for member in stream.getmembers():
            relative = _archive_member_path(member.name)
            if member.isdir():
                archive_directories.add(relative)
                continue
            if not member.isfile():
                raise RuntimeError(
                    "PARSeq source archive contains unsupported non-file members"
                )
            if relative in archive_files:
                raise RuntimeError("PARSeq source archive contains duplicate members")
            extracted = stream.extractfile(member)
            if extracted is None:
                raise RuntimeError("Could not read a PARSeq source archive member")
            digest = hashlib.sha256()
            for block in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(block)
            archive_files[relative] = (digest.hexdigest(), bool(member.mode & 0o111))

    for relative, (expected_digest, expected_executable) in archive_files.items():
        local = source.joinpath(*relative)
        if not local.is_file() or local.is_symlink():
            raise RuntimeError(
                "PARSeq extracted source file is missing or not regular: {}".format(local)
            )
        if _sha256(local) != expected_digest:
            raise RuntimeError("PARSeq extracted source differs from archive: {}".format(local))
        if bool(local.stat().st_mode & 0o111) != expected_executable:
            raise RuntimeError(
                "PARSeq extracted source mode differs from archive: {}".format(local)
            )

    for local in source.rglob("*"):
        relative = local.relative_to(source)
        parts = relative.parts
        if "__pycache__" in parts or local.suffix == ".pyc":
            continue
        if local.is_symlink():
            raise RuntimeError("PARSeq extracted source contains an unexpected symlink")
        key = tuple(parts)
        if local.is_dir():
            if key not in archive_directories:
                raise RuntimeError(
                    "PARSeq extracted source has an unexpected directory: {}".format(local)
                )
        elif key not in archive_files:
            raise RuntimeError(
                "PARSeq extracted source has an unexpected file: {}".format(local)
            )

    return {
        "source_type": PARSEQ_ARCHIVE_SOURCE,
        "runtime_revision": PARSEQ_RUNTIME_REVISION,
        "source_clean": None,
        "source_tree_matches_archive": True,
        "source_archive": str(archive),
        "source_archive_sha256": archive_digest,
        "source_archive_files_verified": len(archive_files),
    }


def verify_parseq_runtime_source(
    source: Path,
    *,
    source_type: str = PARSEQ_GIT_SOURCE,
    source_archive: Path | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError("PARSeq source directory is missing: {}".format(source))
    if source_type == PARSEQ_GIT_SOURCE:
        if source_archive is not None:
            raise ValueError("git_checkout does not accept a PARSeq source archive")
        return _git_source_identity(source)
    if source_type == PARSEQ_ARCHIVE_SOURCE:
        return _archive_source_identity(source, source_archive)
    raise ValueError("Unsupported PARSeq source type: {}".format(source_type))


@dataclass(frozen=True)
class ParseqApproxConfig:
    runtime_source: Path
    checkpoint: Path
    device: str = "cpu"
    decode_ar: bool = True
    refine_iters: int = 1
    runtime_source_type: Literal["git_checkout", "github_commit_archive"] = PARSEQ_GIT_SOURCE
    runtime_source_archive: Path | None = None


@dataclass(frozen=True)
class ParseqResult:
    raw_text: str
    token_ids: tuple[int, ...]
    token_confidences: tuple[float, ...]
    confidence: float | None
    confidence_aggregation: str
    input_role: Literal["T1_full_image", "T3_localized_crop"]
    preprocessing: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ParseqCandidateDiagnosticResult:
    """Unaggregated PARSeq path statistics for proposal-ranking diagnostics."""

    raw_text: str
    token_ids: tuple[int, ...]
    token_probabilities: tuple[float, ...]
    token_log_probabilities: tuple[float, ...]
    sequence_confidence_native: float
    sequence_confidence_native_aggregation: str
    mean_token_log_probability: float | None
    minimum_token_probability: float | None
    predicted_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ParseqApproxRecognizer:
    """Real official PARSeq preprocessing, forward pass, and tokenizer decode."""

    def __init__(self, config: ParseqApproxConfig) -> None:
        import torch

        source, checkpoint = config.runtime_source.resolve(), config.checkpoint.resolve()
        if not source.is_dir() or not checkpoint.is_file():
            raise RuntimeError("PARSeq runtime source and checkpoint must exist")
        source_identity = verify_parseq_runtime_source(
            source,
            source_type=config.runtime_source_type,
            source_archive=config.runtime_source_archive,
        )
        digest = _sha256(checkpoint)
        if checkpoint.name != PARSEQ_CHECKPOINT_NAME or digest != PARSEQ_CHECKPOINT_SHA256:
            raise RuntimeError("PARSeq checkpoint name or SHA256 does not match the qualified artifact")

        sys.path.insert(0, str(source))
        try:
            from strhub.data.module import SceneTextDataModule
            from strhub.models.utils import create_model

            model = create_model("parseq", pretrained=False)
            if (bool(model.model.decode_ar) != config.decode_ar or
                    int(model.model.refine_iters) != config.refine_iters):
                raise RuntimeError("Qualified PARSeq runtime decoding defaults changed")
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            loaded = model.model.load_state_dict(state, strict=True)
            if loaded.missing_keys or loaded.unexpected_keys:
                raise RuntimeError(f"Strict PARSeq load was incomplete: {loaded}")
            transform = SceneTextDataModule.get_transform(tuple(model.hparams.img_size))
        finally:
            sys.path.remove(str(source))

        self.config = config
        self.model = model.eval().to(config.device)
        self.transform = transform
        self._torch = torch
        self._provenance = {
            "status": "approximate reproduction",
            "checkpoint": PARSEQ_CHECKPOINT_NAME,
            "checkpoint_sha256": digest,
            **source_identity,
            "strict_state_dict_load": True,
            "decode_ar": config.decode_ar,
            "refine_iters": config.refine_iters,
        }

    @classmethod
    def _from_test_components(cls, model: Any, transform: Any) -> "ParseqApproxRecognizer":
        """Construct with fakes for unit tests; never used by experiment configuration."""
        import torch

        obj = cls.__new__(cls)
        obj.config = ParseqApproxConfig(Path("test-runtime"), Path("test-checkpoint"))
        obj.model, obj.transform, obj._torch = model, transform, torch
        obj._provenance = {"status": "test fixture", "strict_state_dict_load": False}
        return obj

    def recognize(
        self,
        image_rgb: np.ndarray,
        *,
        input_role: Literal["T1_full_image", "T3_localized_crop"],
    ) -> ParseqResult:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("PARSeq expects an RGB uint8 HxWx3 array")
        tensor = self.transform(Image.fromarray(image_rgb, mode="RGB")).unsqueeze(0)
        with self._torch.inference_mode():
            logits = self.model(tensor.to(self.config.device))
            probabilities = logits.softmax(-1)
            decoded, decoded_confidences = self.model.tokenizer.decode(probabilities)

        ids = probabilities.argmax(-1)[0].detach().cpu().tolist()
        # EOS is id 0 in this qualified runtime. Keep only decoded character ids.
        if 0 in ids:
            ids = ids[: ids.index(0)]
        raw_conf = decoded_confidences[0].detach().cpu().tolist()
        token_confidences = tuple(float(value) for value in raw_conf[: len(ids)])
        # Sequence confidence is not paper-specified. Arithmetic mean is exposed
        # as an ASSUMPTION while the unaggregated character confidences are kept.
        confidence = (
            sum(token_confidences) / len(token_confidences) if token_confidences else None
        )
        return ParseqResult(
            raw_text=str(decoded[0]),
            token_ids=tuple(int(value) for value in ids),
            token_confidences=token_confidences,
            confidence=confidence,
            confidence_aggregation="ASSUMPTION: arithmetic mean of decoded character probabilities",
            input_role=input_role,
            preprocessing={
                "color": "RGB",
                "input_shape": list(tensor.shape),
                "resize": "32x128 bicubic (qualified official runtime transform)",
                "tensor_scale": "ToTensor [0,1]",
                "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                "full_image_T1_semantics": "UNRESOLVED" if input_role == "T1_full_image" else None,
            },
            provenance=dict(self._provenance),
        )

    def recognize_candidate_batch(
        self, images_rgb: list[np.ndarray] | tuple[np.ndarray, ...]
    ) -> tuple[ParseqCandidateDiagnosticResult, ...]:
        """Decode one proposal batch and retain native confidence diagnostics.

        The sequence score follows upstream PARSeq evaluation: product of the
        tokenizer-filtered greedy probabilities, including EOS when present.
        Character-level scores exclude EOS and match ``predicted_length``.
        """
        if not images_rgb:
            raise ValueError("PARSeq candidate batch cannot be empty")
        tensors = []
        for image_rgb in images_rgb:
            if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
                raise ValueError("PARSeq expects every batch item to be RGB uint8 HxWx3")
            tensors.append(self.transform(Image.fromarray(image_rgb, mode="RGB")))
        batch = self._torch.stack(tensors, dim=0)
        with self._torch.inference_mode():
            logits = self.model(batch.to(self.config.device))
            probabilities = logits.softmax(-1)
            log_probabilities = logits.log_softmax(-1)
            decoded, decoded_confidences = self.model.tokenizer.decode(probabilities)

        greedy_ids = probabilities.argmax(-1).detach().cpu()
        greedy_logs = log_probabilities.gather(
            -1, probabilities.argmax(-1, keepdim=True)
        ).squeeze(-1).detach().cpu()
        results = []
        for index, (text, decoded_probability_tensor) in enumerate(
            zip(decoded, decoded_confidences)
        ):
            ids = greedy_ids[index].tolist()
            if 0 in ids:
                ids = ids[: ids.index(0)]
            character_count = len(ids)
            if len(str(text)) != character_count:
                raise RuntimeError("PARSeq decoded text/token length mismatch")
            decoded_probabilities = tuple(
                float(value) for value in decoded_probability_tensor.detach().cpu().tolist()
            )
            token_probabilities = decoded_probabilities[:character_count]
            token_logs = tuple(
                float(value) for value in greedy_logs[index, :character_count].tolist()
            )
            if len(token_probabilities) != character_count or len(token_logs) != character_count:
                raise RuntimeError("PARSeq candidate confidence length mismatch")
            results.append(ParseqCandidateDiagnosticResult(
                raw_text=str(text),
                token_ids=tuple(int(value) for value in ids),
                token_probabilities=token_probabilities,
                token_log_probabilities=token_logs,
                sequence_confidence_native=float(decoded_probability_tensor.prod().item()),
                sequence_confidence_native_aggregation=(
                    "upstream PARSeq greedy decoded-probability product; includes EOS when present"
                ),
                mean_token_log_probability=(
                    sum(token_logs) / character_count if character_count else None
                ),
                minimum_token_probability=(
                    min(token_probabilities) if token_probabilities else None
                ),
                predicted_length=character_count,
            ))
        if len(results) != len(images_rgb):
            raise RuntimeError("PARSeq candidate batch cardinality mismatch")
        return tuple(results)

    def release(self) -> dict[str, Any]:
        """Drop runtime references for the M6 one-shot sequential lifecycle."""
        had_model = self.model is not None
        self.model = None
        self.transform = None
        return {"model_reference_released": had_model, "transform_reference_released": True}
