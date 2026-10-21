from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from caption.blip2_captioner import CaptionResult
from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam
from pipeline.lightweight_context import LightweightConfig, LightweightContextPipeline
from recognition.parseq import ParseqResult


class Segmenter:
    def __call__(self, image):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[2:6, 3:8] = 1
        return {"mask": mask}


class Recognizer:
    def recognize(self, image, *, input_role):
        text = "FULL" if input_role == "T1_full_image" else "CROP"
        return ParseqResult(text, (1,), (0.9,), 0.9, "fixture", input_role, {}, {"status": "fixture"})


class Captioner:
    def caption(self, image):
        return CaptionResult("A sign says CROP", "model", "revision", "ph", {}, "gh",
                             "float32", "cpu", "single_device", "none", 0.01, None, None)


class Semantic:
    def compare_candidates(self, t1, t3, caption):
        return 0.4, 0.9


def test_tiny_pipeline_persists_all_intermediates_and_stops_before_fallback(tmp_path: Path):
    pipeline = LightweightContextPipeline(
        LightweightConfig(
            BlockParam(
                min_area=1,
                pad=1,
                connectivity=8,
                tie_break=APPROXIMATE_TIE_BREAK,
            )
        ),
        segmenter=Segmenter(), recognizer=Recognizer(), captioner=Captioner(), semantic=Semantic(),
    )
    output = pipeline.run(np.zeros((10, 12, 3), dtype=np.uint8), source_id="synthetic")
    assert output["recognition"]["T1"]["raw_text"] == "FULL"
    assert output["recognition"]["T3_selected"]["raw_text"] == "CROP"
    assert output["segmentation"]["localization_audit"]["parameters"]["connectivity"] == 8
    assert len(output["scoring"]) == 3
    assert output["fallback"]["invoked"] is False
    path = tmp_path / "intermediate.json"
    pipeline.write_json(output, path)
    assert json.loads(path.read_text())["schema_version"] == 1
