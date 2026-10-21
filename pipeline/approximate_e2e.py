"""One-shot APPROXIMATE M6 integration through conditional DeepSolo fallback."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Protocol

import numpy as np

from caption.blip2_captioner import CaptionResult
from deepsolo.contracts import DeepSoloResponse
from localize.mask_to_blocks import BlockParam, localize_mask
from recognition.parseq import ParseqResult
from scoring.lexical import LexicalMode, lexical_similarity
from scoring.strategies import Strategy, decide

from .sequential_lifecycle import SequentialLifecycle


T3Selection = Literal[
    "rank_1_largest_component",
    "repository_max_paired_context_confidence_all_crops",
]


class ReleasableSegmenter(Protocol):
    def __call__(self, image_rgb: np.ndarray) -> Dict[str, np.ndarray]: ...
    def release(self) -> Dict[str, Any]: ...


class ReleasableRecognizer(Protocol):
    def recognize(self, image_rgb: np.ndarray, *, input_role: str) -> ParseqResult: ...
    def release(self) -> Dict[str, Any]: ...


class ReleasableCaptioner(Protocol):
    def caption(self, image_rgb: np.ndarray) -> CaptionResult: ...
    def release(self) -> Dict[str, Any]: ...


class ReleasableSemantic(Protocol):
    def compare_candidates(self, t1: str, t3: str, caption: str) -> tuple[float, float]: ...
    def release(self) -> Dict[str, Any]: ...


class FallbackClient(Protocol):
    def infer(self, **kwargs: Any) -> DeepSoloResponse: ...


@dataclass(frozen=True)
class ApproximateE2EConfig:
    localization: BlockParam
    segmentation_source: str
    scoring_strategy: Strategy
    lexical_mode: LexicalMode
    deepsolo_candidate: str
    deepsolo_reduction_policy: str
    deepsolo_inference_options: Dict[str, Any]
    alpha: float = 0.6
    beta: float = 0.4
    tau: float = 0.8
    reproduction_mode: str = "APPROXIMATE"
    result_class: str = "approximate reproduction engineering smoke test"
    t3_selection: T3Selection = "rank_1_largest_component"
    t3_selection_classification: Literal["ASSUMPTION", "REPOSITORY_SPECIFIED"] = (
        "ASSUMPTION"
    )

    def __post_init__(self) -> None:
        if self.reproduction_mode != "APPROXIMATE":
            raise ValueError("M6 candidate integration is forbidden in STRICT mode")
        if (self.alpha, self.beta, self.tau) != (0.6, 0.4, 0.8):
            raise ValueError("M6 scoring is qualified only for alpha=0.6, beta=0.4, tau=0.8")
        if not self.segmentation_source:
            raise ValueError("An explicit segmentation source/fixture label is required")
        if self.t3_selection not in (
            "rank_1_largest_component",
            "repository_max_paired_context_confidence_all_crops",
        ):
            raise ValueError("Unknown T3 selection policy: {}".format(self.t3_selection))
        if (
            self.t3_selection == "repository_max_paired_context_confidence_all_crops"
            and self.scoring_strategy != "repository_paired_threshold"
        ):
            raise ValueError(
                "Repository max-C3 crop selection requires repository_paired_threshold"
            )
        if (
            self.t3_selection == "repository_max_paired_context_confidence_all_crops"
            and self.t3_selection_classification != "REPOSITORY_SPECIFIED"
        ):
            raise ValueError("Repository max-C3 selection must be classified explicitly")


def _pixel_hash(image: np.ndarray) -> str:
    return hashlib.sha256(
        str((image.shape, image.dtype.str)).encode("utf-8") + image.tobytes()
    ).hexdigest()


def _placement(component: Any, fallback: str) -> str:
    config = getattr(component, "config", None)
    return str(getattr(config, "device", fallback))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_png_atomic(array: np.ndarray, path: Path, *, mode: str) -> dict[str, Any]:
    from PIL import Image

    with tempfile.NamedTemporaryFile(
        suffix=".png", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        Image.fromarray(array, mode=mode).save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "path": str(path.resolve()),
        "file_sha256": _file_sha256(path),
        "decoded_pixel_sha256": _pixel_hash(array),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "encoding": "lossless PNG",
    }


class ApproximateE2EPipeline:
    """Run one image while ensuring BLIP-2 is released before DeepSolo starts."""

    STRICT_BLOCKERS = (
        "Exact paper PARSeq-S artifact/runtime is unresolved.",
        "Exact paper BLIP-2 and MPNet artifacts/protocols are unresolved.",
        "Exact DeepSolo checkpoint/config per benchmark is unresolved.",
        "DeepSolo multi-output reduction is unresolved.",
        "The frozen segmentation checkpoint is APPROXIMATE, not paper-identical.",
    )

    def __init__(
        self,
        config: ApproximateE2EConfig,
        *,
        segmenter: ReleasableSegmenter,
        recognizer: ReleasableRecognizer,
        captioner: ReleasableCaptioner,
        semantic: ReleasableSemantic,
        fallback: FallbackClient,
    ) -> None:
        self.config = config
        self.segmenter = segmenter
        self.recognizer = recognizer
        self.captioner = captioner
        self.semantic = semantic
        self.fallback = fallback
        self._used = False

    def run(
        self,
        image_rgb: np.ndarray,
        *,
        image_path: Path,
        artifact_root: Path | None = None,
    ) -> Dict[str, Any]:
        if self._used:
            raise RuntimeError("Sequential M6 pipeline instances are one-shot after model release")
        self._used = True
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("Pipeline expects RGB uint8 HxWx3 input")
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise ValueError("Pipeline image_path must exist for the DeepSolo boundary")
        lifecycle = SequentialLifecycle()
        latencies: Dict[str, float] = {}
        placements: Dict[str, str] = {}

        started = time.perf_counter()
        segmentation = self.segmenter(image_rgb)
        segmentation_metadata = dict(segmentation.get("metadata", {}))
        latencies["segmentation"] = time.perf_counter() - started
        placements["segmentation"] = _placement(self.segmenter, "fixture/CPU")
        lifecycle.stage_completed("segmentation", latencies["segmentation"], placements["segmentation"])
        lifecycle.release("segmentation", self.segmenter)

        localization_started = time.perf_counter()
        mask = np.asarray(segmentation["mask"])
        localization = localize_mask(mask, image_rgb, self.config.localization)
        blocks = localization.retained
        mask_artifact = None
        crop_artifacts: dict[int, dict[str, Any]] = {}
        if artifact_root is not None:
            artifact_root = artifact_root.resolve()
            artifact_root.mkdir(parents=True, exist_ok=False)
            mask_artifact = _write_png_atomic(
                (mask * 255).astype(np.uint8, copy=False),
                artifact_root / "segmentation-mask.png",
                mode="L",
            )
            for block in blocks:
                crop_artifacts[block.rank] = _write_png_atomic(
                    block.crop,
                    artifact_root / "crop-rank-{:02d}.png".format(block.rank),
                    mode="RGB",
                )
        latencies["localization"] = time.perf_counter() - localization_started
        placements["localization"] = "CPU"
        lifecycle.stage_completed("localization", latencies["localization"], "CPU")

        recognition_started = time.perf_counter()
        t1 = self.recognizer.recognize(image_rgb, input_role="T1_full_image")
        t3_all = [
            self.recognizer.recognize(block.crop, input_role="T3_localized_crop")
            for block in blocks
        ]
        empty_t3 = ParseqResult(
            "", (), (), None, "not available: no localized component",
            "T3_localized_crop", {}, {"status": "empty no-crop result"},
        )
        latencies["parseq"] = time.perf_counter() - recognition_started
        placements["parseq"] = _placement(self.recognizer, "unknown")
        lifecycle.stage_completed("parseq", latencies["parseq"], placements["parseq"])
        lifecycle.release("parseq", self.recognizer)

        caption_started = time.perf_counter()
        caption = self.captioner.caption(image_rgb)
        latencies["blip2"] = time.perf_counter() - caption_started
        placements["blip2"] = "{} via {}".format(caption.device, caption.placement)
        lifecycle.stage_completed("blip2", latencies["blip2"], placements["blip2"])
        lifecycle.release("blip2", self.captioner)

        semantic_started = time.perf_counter()
        t3_semantic_scores: list[float] = []
        if self.config.t3_selection == "repository_max_paired_context_confidence_all_crops":
            S1, _ = self.semantic.compare_candidates(t1.raw_text, "", caption.raw_caption)
            for candidate in t3_all:
                _, candidate_S3 = self.semantic.compare_candidates(
                    "", candidate.raw_text, caption.raw_caption
                )
                t3_semantic_scores.append(candidate_S3)
        else:
            t3 = t3_all[0] if t3_all else empty_t3
            S1, S3 = self.semantic.compare_candidates(
                t1.raw_text, t3.raw_text, caption.raw_caption
            )
            t3_semantic_scores.append(S3)
        semantic_metadata = dict(getattr(self.semantic, "metadata", {}))
        latencies["mpnet"] = time.perf_counter() - semantic_started
        placements["mpnet"] = _placement(self.semantic, "unknown")
        lifecycle.stage_completed("mpnet", latencies["mpnet"], placements["mpnet"])
        lifecycle.release("mpnet", self.semantic)

        scoring_started = time.perf_counter()
        lexical1 = lexical_similarity(t1.raw_text, caption.raw_caption, self.config.lexical_mode)
        t3_candidate_scores: list[dict[str, Any]] = []
        selected_rank: int | None
        if self.config.t3_selection == "repository_max_paired_context_confidence_all_crops":
            for rank, (candidate, candidate_S3) in enumerate(
                zip(t3_all, t3_semantic_scores), start=1
            ):
                candidate_lexical = lexical_similarity(
                    candidate.raw_text, caption.raw_caption, self.config.lexical_mode
                )
                candidate_C3 = (
                    self.config.alpha * candidate_S3
                    + self.config.beta * candidate_lexical.score
                )
                t3_candidate_scores.append(
                    {
                        "rank": rank,
                        "raw_text": candidate.raw_text,
                        "S3": candidate_S3,
                        "L3": candidate_lexical.score,
                        "C3": candidate_C3,
                        "evaluated_for_selection": True,
                    }
                )
            if t3_candidate_scores:
                selected_offset = max(
                    range(len(t3_candidate_scores)),
                    key=lambda offset: t3_candidate_scores[offset]["C3"],
                )
                selected_rank = selected_offset + 1
                t3 = t3_all[selected_offset]
                S3 = t3_semantic_scores[selected_offset]
                lexical3 = lexical_similarity(
                    t3.raw_text, caption.raw_caption, self.config.lexical_mode
                )
            else:
                selected_rank = None
                t3 = empty_t3
                S3 = -1.0
                lexical3 = lexical_similarity(
                    t3.raw_text, caption.raw_caption, self.config.lexical_mode
                )
        else:
            selected_rank = 1 if t3_all else None
            lexical3 = lexical_similarity(
                t3.raw_text, caption.raw_caption, self.config.lexical_mode
            )
            for rank, candidate in enumerate(t3_all, start=1):
                selected = rank == 1
                t3_candidate_scores.append(
                    {
                        "rank": rank,
                        "raw_text": candidate.raw_text,
                        "S3": S3 if selected else None,
                        "L3": lexical3.score if selected else None,
                        "C3": (
                            self.config.alpha * S3
                            + self.config.beta * lexical3.score
                        ) if selected else None,
                        "evaluated_for_selection": selected,
                    }
                )
        decision = decide(
            self.config.scoring_strategy,
            T1=t1.raw_text, T2=caption.raw_caption, T3=t3.raw_text,
            S1=S1, S3=S3, L1=lexical1.score, L3=lexical3.score,
        )
        latencies["scoring"] = time.perf_counter() - scoring_started
        placements["scoring"] = "CPU"
        lifecycle.stage_completed("scoring", latencies["scoring"], "CPU")

        fallback_response = None
        final_prediction = decision.selected_output
        fallback_reason = None
        if decision.needs_fallback:
            fallback_reason = "PAPER_SPECIFIED trigger: C < tau"
            fallback_started = time.perf_counter()
            fallback_response = self.fallback.infer(
                image_path=image_path,
                candidate_id=self.config.deepsolo_candidate,
                inference_options=self.config.deepsolo_inference_options,
                reduction_policy=self.config.deepsolo_reduction_policy,
            )
            latencies["deepsolo"] = time.perf_counter() - fallback_started
            placements["deepsolo"] = "isolated context-str-deepsolo subprocess"
            lifecycle.stage_completed(
                "deepsolo", latencies["deepsolo"], placements["deepsolo"]
            )
            final_prediction = fallback_response.selected_output

        image_file_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        response_dict = fallback_response.to_dict() if fallback_response else None
        hardware_deviations = [
            "HARDWARE_DEVIATION: sequential model lifecycle",
        ]
        if caption.hardware_deviation:
            hardware_deviations.append(caption.hardware_deviation)
        if fallback_response is not None:
            hardware_deviations.append(
                "HARDWARE_DEVIATION: DeepSolo executes in an isolated legacy subprocess"
            )
        trace = {
            "schema_version": 1,
            "reproduction_mode": self.config.reproduction_mode,
            "result_class": self.config.result_class,
            "claims": {"table_5": False, "table_6": False, "table_7": False},
            "image": {
                "path": str(image_path),
                "file_sha256": image_file_hash,
                "decoded_pixel_sha256": _pixel_hash(image_rgb),
            },
            "artifacts": {
                "segmentation": self.config.segmentation_source,
                "parseq": dict(t1.provenance),
                "blip2": {
                    "model_id": caption.model_id,
                    "revision": caption.revision,
                    "prompt_hash": caption.prompt_hash,
                    "generation_config_hash": caption.generation_config_hash,
                },
                "mpnet": semantic_metadata,
                "deepsolo_candidate": self.config.deepsolo_candidate,
            },
            "segmentation": {
                "source": self.config.segmentation_source,
                "inference": segmentation_metadata,
                "mask_shape": list(mask.shape),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "mask_artifact": mask_artifact,
                "localization_audit": localization.as_dict(),
                "localized_blocks": [
                    {
                        "rank": block.rank,
                        "component_id": block.component_id,
                        "component_area": block.component_area,
                        "bbox": block.bbox,
                        "padded_bbox": block.padded_bbox,
                        "crop_artifact": crop_artifacts.get(block.rank),
                    }
                    for block in blocks
                ],
            },
            "T1": t1.to_dict(),
            "T2": caption.to_dict(),
            "T3": {
                "selected": t3.to_dict(),
                "all": [item.to_dict() for item in t3_all],
                "selected_rank": selected_rank,
                "candidate_scores": t3_candidate_scores,
                "selection_policy": "{}: {}".format(
                    self.config.t3_selection_classification,
                    self.config.t3_selection,
                ),
            },
            "scores": {
                "S1": decision.S1, "S3": decision.S3,
                "L1": decision.L1, "L3": decision.L3,
                "S": decision.S, "L": decision.L, "C": decision.C,
                "alpha": decision.alpha, "beta": decision.beta, "tau": decision.tau,
                "semantic_winner": decision.semantic_winner,
                "confidence_winner": decision.confidence_winner,
                "strategy": decision.strategy,
                "lexical_mode": self.config.lexical_mode,
            },
            "fallback": {
                "needs_fallback": decision.needs_fallback,
                "invoked": fallback_response is not None,
                "reason": fallback_reason,
                "selected_pre_fallback_prediction": decision.selected_output,
                "candidate": self.config.deepsolo_candidate,
                "raw_outputs": response_dict["detections"] if response_dict else [],
                "recognized_strings": response_dict["recognized_strings"] if response_dict else [],
                "reduction_policy": self.config.deepsolo_reduction_policy,
                "response": response_dict,
            },
            "final_prediction": final_prediction,
            "per_stage_latency_seconds": latencies,
            "placement": placements,
            "lifecycle": lifecycle.events,
            "hardware_deviations": hardware_deviations,
            "warnings": [
                "An engineering smoke test is not a Table 5 experiment result.",
                "DeepSolo reduction policies are assumptions, not paper-exact behavior.",
            ],
            "strict_blockers": list(self.STRICT_BLOCKERS),
        }
        return trace

    @staticmethod
    def write_json(trace: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, encoding="utf-8"
        ) as stream:
            json.dump(trace, stream, indent=2, sort_keys=True)
            stream.write("\n")
            temporary = Path(stream.name)
        os.replace(temporary, path)
