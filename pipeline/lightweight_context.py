"""APPROXIMATE M5 pipeline that deliberately stops before DeepSolo fallback."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from caption.blip2_captioner import CaptionResult
from localize.mask_to_blocks import BlockParam, localize_mask
from recognition.parseq import ParseqResult
from scoring.lexical import LexicalMode, lexical_similarity
from scoring.strategies import STRATEGIES, decide


class Segmenter(Protocol):
    def __call__(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]: ...


class Recognizer(Protocol):
    def recognize(self, image_rgb: np.ndarray, *, input_role: str) -> ParseqResult: ...


class Captioner(Protocol):
    def caption(self, image_rgb: np.ndarray) -> CaptionResult: ...


class SemanticScorer(Protocol):
    def compare_candidates(self, t1: str, t3: str, caption: str) -> tuple[float, float]: ...


@dataclass(frozen=True)
class LightweightConfig:
    localization: BlockParam
    lexical_mode: LexicalMode = "raw"
    t3_selection: str = "ASSUMPTION: rank-1/largest component"
    result_class: str = "approximate reproduction smoke test"


class LightweightContextPipeline:
    """Run through ``needs_fallback`` and never import or call DeepSolo."""

    def __init__(
        self, config: LightweightConfig, *, segmenter: Segmenter, recognizer: Recognizer,
        captioner: Captioner, semantic: SemanticScorer,
    ) -> None:
        self.config, self.segmenter, self.recognizer = config, segmenter, recognizer
        self.captioner, self.semantic = captioner, semantic

    def run(self, image_rgb: np.ndarray, *, source_id: str = "") -> dict[str, Any]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("Pipeline expects RGB uint8 HxWx3 input")
        segmentation = self.segmenter(image_rgb)
        mask = np.asarray(segmentation["mask"])
        localization = localize_mask(mask, image_rgb, self.config.localization)
        blocks = localization.retained
        t1 = self.recognizer.recognize(image_rgb, input_role="T1_full_image")
        t3_results = [
            self.recognizer.recognize(block.crop, input_role="T3_localized_crop")
            for block in blocks
        ]
        # L-004 is unresolved. The smoke path uses the largest component solely
        # as an explicit approximate interface choice and retains every result.
        t3 = t3_results[0] if t3_results else ParseqResult(
            "", (), (), None, "not available: no localized component",
            "T3_localized_crop", {}, {"status": "empty no-crop result"},
        )
        caption = self.captioner.caption(image_rgb)
        S1, S3 = self.semantic.compare_candidates(t1.raw_text, t3.raw_text, caption.raw_caption)
        L1 = lexical_similarity(t1.raw_text, caption.raw_caption, self.config.lexical_mode)
        L3 = lexical_similarity(t3.raw_text, caption.raw_caption, self.config.lexical_mode)
        decisions = [
            decide(name, T1=t1.raw_text, T2=caption.raw_caption, T3=t3.raw_text,
                   S1=S1, S3=S3, L1=L1.score, L3=L3.score).to_dict()
            for name in STRATEGIES
        ]
        image_hash = hashlib.sha256(
            str((image_rgb.shape, image_rgb.dtype.str)).encode() + image_rgb.tobytes()
        ).hexdigest()
        return {
            "schema_version": 1,
            "result_class": self.config.result_class,
            "source_id": source_id,
            "image_pixel_hash": image_hash,
            "segmentation": {
                "mask_shape": list(mask.shape),
                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                "localization_audit": localization.as_dict(),
                "localized_blocks": [
                    {
                        "rank": block.rank, "component_id": block.component_id,
                        "component_area": block.component_area, "bbox": block.bbox,
                        "padded_bbox": block.padded_bbox,
                    } for block in blocks
                ],
            },
            "recognition": {
                "T1": t1.to_dict(), "T3_selected": t3.to_dict(),
                "T3_all": [result.to_dict() for result in t3_results],
                "T3_selection": self.config.t3_selection,
            },
            "caption": caption.to_dict(),
            "similarities": {
                "S1": S1, "S3": S3, "L1": asdict(L1), "L3": asdict(L3),
            },
            "scoring": decisions,
            "fallback": {
                "invoked": False,
                "needs_fallback_by_strategy": {
                    decision["strategy"]: decision["needs_fallback"] for decision in decisions
                },
            },
        }

    @staticmethod
    def write_json(output: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
            json.dump(output, stream, indent=2, sort_keys=True)
            stream.write("\n")
            temporary = Path(stream.name)
        os.replace(temporary, path)
