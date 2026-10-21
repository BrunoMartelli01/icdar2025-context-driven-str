from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from caption.blip2_captioner import CaptionResult
from deepsolo.contracts import DeepSoloResponse
from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam
from pipeline.approximate_e2e import ApproximateE2EConfig, ApproximateE2EPipeline
from recognition.parseq import ParseqResult
from scripts.run_table5_smoke import _failure_record


CANDIDATE = "ART-DEEPSOLO-R50-TOTALTEXT-CANDIDATE"


class Segmenter:
    config = type("Config", (), {"device": "cuda:0"})()
    def __init__(self, events):
        self.events = events
    def __call__(self, image):
        self.events.append("segmentation")
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[2:6, 3:8] = 1
        return {"mask": mask}
    def release(self):
        self.events.append("release_segmentation")
        return {"fixture": True}


class Recognizer:
    config = type("Config", (), {"device": "cuda:0"})()
    def __init__(self, events):
        self.events = events
    def recognize(self, image, *, input_role):
        self.events.append(input_role)
        text = "FULL" if input_role == "T1_full_image" else "CROP"
        return ParseqResult(text, (1,), (0.9,), 0.9, "fixture", input_role, {}, {
            "status": "fixture", "artifact_id": "PARSEQ-FIXTURE",
        })
    def release(self):
        self.events.append("release_parseq")
        return {"fixture": True}


class Captioner:
    def __init__(self, events, caption):
        self.events, self.raw_caption = events, caption
    def caption(self, image):
        self.events.append("blip2")
        return CaptionResult(
            self.raw_caption, "fixture-model", "fixture-revision", "prompt-hash", {},
            "generation-hash", "float16", "cuda:0", "cpu_offload", "none",
            0.01, 1, 2, hardware_deviation="HARDWARE_DEVIATION: fixture",
        )
    def release(self):
        self.events.append("release_blip2")
        return {"fixture": True}


class Semantic:
    config = type("Config", (), {"device": "cpu"})()
    def __init__(self, events, scores):
        self.events, self.scores = events, scores
    @property
    def metadata(self):
        return {"status": "fixture", "revision": "fixture-mpnet"}
    def compare_candidates(self, t1, t3, caption):
        self.events.append("mpnet")
        return self.scores
    def release(self):
        self.events.append("release_mpnet")
        return {"fixture": True}


class RepositorySemantic(Semantic):
    def compare_candidates(self, t1, t3, caption):
        self.events.append("mpnet")
        if t1 == "FULL":
            return 0.2, 0.0
        return 0.0, {"LOW": 0.3, "HIGH": 0.95, "HIGH_TIE": 0.95}.get(t3, 0.0)


class Fallback:
    def __init__(self, events):
        self.events, self.calls = events, 0
    def infer(self, **kwargs):
        self.events.append("deepsolo")
        self.calls += 1
        assert kwargs["image_path"].is_file()
        assert kwargs["candidate_id"] == CANDIDATE
        return DeepSoloResponse.from_dict({
            "schema_version": 1,
            "request_id": "fixture",
            "success": True,
            "detections": [
                {"source_index": 0, "text": "FALLBACK", "score": 0.95,
                 "score_type": "fixture instance score",
                 "geometry": {"boundary_points": [[0, 0], [1, 1]]},
                 "raw": {"fixture": True}},
            ],
            "recognized_strings": ["FALLBACK"],
            "selected_output": "FALLBACK",
            "reduction_policy": kwargs["reduction_policy"],
            "artifact": {"status": "fixture"},
            "latency": {"total_seconds": 0.01},
            "placement": {"process_boundary": "fixture subprocess"},
            "memory": {},
            "error": None,
        })


def make_pipeline(events, fallback, *, caption, scores):
    return ApproximateE2EPipeline(
        ApproximateE2EConfig(
            localization=BlockParam(
                min_area=1,
                pad=1,
                connectivity=8,
                tie_break=APPROXIMATE_TIE_BREAK,
            ),
            segmentation_source="synthetic segmentation fixture",
            scoring_strategy="selected_candidate_paired",
            lexical_mode="raw",
            deepsolo_candidate=CANDIDATE,
            deepsolo_reduction_policy="highest_score",
            deepsolo_inference_options={
                "device": "cuda:0", "score_threshold": None, "lexicon_type": None,
            },
        ),
        segmenter=Segmenter(events), recognizer=Recognizer(events),
        captioner=Captioner(events, caption), semantic=Semantic(events, scores),
        fallback=fallback,
    )


def image_fixture(tmp_path: Path):
    array = np.zeros((10, 12, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(array).save(path)
    return array, path


def test_no_fallback_path_releases_models_and_persists_complete_trace(tmp_path: Path):
    events = []
    fallback = Fallback(events)
    pipeline = make_pipeline(events, fallback, caption="FULL", scores=(0.95, 0.1))
    image, path = image_fixture(tmp_path)
    trace = pipeline.run(image, image_path=path, artifact_root=tmp_path / "artifacts")
    assert fallback.calls == 0
    assert trace["fallback"]["needs_fallback"] is False
    assert trace["fallback"]["invoked"] is False
    assert trace["final_prediction"] == "FULL"
    assert events.index("release_blip2") < events.index("mpnet")
    assert trace["claims"] == {"table_5": False, "table_6": False, "table_7": False}
    assert trace["segmentation"]["inference"] == {}
    mask_artifact = trace["segmentation"]["mask_artifact"]
    assert Path(mask_artifact["path"]).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert mask_artifact["shape"] == [10, 12]
    crop_artifact = trace["segmentation"]["localized_blocks"][0]["crop_artifact"]
    assert Path(crop_artifact["path"]).is_file()
    assert crop_artifact["encoding"] == "lossless PNG"
    assert trace["hardware_deviations"] == [
        "HARDWARE_DEVIATION: sequential model lifecycle",
        "HARDWARE_DEVIATION: fixture",
    ]
    assert trace["segmentation"]["localization_audit"]["parameters"] == {
        "min_area": 1,
        "pad": 1,
        "connectivity": 8,
        "tie_break": APPROXIMATE_TIE_BREAK,
        "nmax": 10,
    }
    output = tmp_path / "trace.json"
    pipeline.write_json(trace, output)
    assert json.loads(output.read_text(encoding="utf-8"))["T1"]["raw_text"] == "FULL"


def test_constructed_fallback_invokes_deepsolo_only_after_blip_release(tmp_path: Path):
    events = []
    fallback = Fallback(events)
    pipeline = make_pipeline(events, fallback, caption="unrelated scene", scores=(0.0, 0.0))
    image, path = image_fixture(tmp_path)
    trace = pipeline.run(image, image_path=path)
    assert fallback.calls == 1
    assert trace["fallback"]["needs_fallback"] is True
    assert trace["fallback"]["selected_pre_fallback_prediction"] == "FULL"
    assert trace["fallback"]["raw_outputs"][0]["text"] == "FALLBACK"
    assert trace["final_prediction"] == "FALLBACK"
    assert trace["hardware_deviations"][-1] == (
        "HARDWARE_DEVIATION: DeepSolo executes in an isolated legacy subprocess"
    )
    assert events.index("release_blip2") < events.index("deepsolo")
    release_stages = [
        event["stage"] for event in trace["lifecycle"]
        if event["event"] == "component_released"
    ]
    assert release_stages == ["segmentation", "parseq", "blip2", "mpnet"]


def test_table5_smoke_failure_record_is_explicit_and_forbids_claims(tmp_path: Path):
    paths = {
        name: tmp_path / name
        for name in (
            "config", "image", "segmentation_checkpoint", "parseq_source",
            "parseq_source_archive", "parseq_checkpoint", "deepsolo_python",
            "deepsolo_runtime_manifest", "deepsolo_source_root",
            "output",
        )
    }
    record = _failure_record(Namespace(**paths), RuntimeError("fixture failure"))
    assert record["success"] is False
    assert record["claims"] == {
        "table_5": False,
        "table_6": False,
        "table_7": False,
    }
    assert record["error"]["type"] == "RuntimeError"
    assert record["error"]["message"] == "fixture failure"
    assert record["error"]["deepsolo_response"] is None
    assert "alternate model" in record["warnings"][1]


def test_pipeline_refuses_to_overwrite_intermediate_artifacts(tmp_path: Path):
    events = []
    pipeline = make_pipeline(
        events, Fallback(events), caption="FULL", scores=(0.95, 0.1)
    )
    image, path = image_fixture(tmp_path)
    artifact_root = tmp_path / "existing"
    artifact_root.mkdir()
    with pytest.raises(FileExistsError):
        pipeline.run(image, image_path=path, artifact_root=artifact_root)


def test_repository_policy_scores_all_crops_and_selects_first_maximum(tmp_path: Path):
    events = []

    class ThreeBlockSegmenter(Segmenter):
        def __call__(self, image):
            self.events.append("segmentation")
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[1:5, 1:5] = 1
            mask[7:9, 1:3] = 1
            mask[7:9, 9:11] = 1
            return {"mask": mask}

    class ThreeCropRecognizer(Recognizer):
        def __init__(self, events):
            super().__init__(events)
            self.crop_index = 0

        def recognize(self, image, *, input_role):
            self.events.append(input_role)
            if input_role == "T1_full_image":
                text = "FULL"
            else:
                text = ("LOW", "HIGH", "HIGH_TIE")[self.crop_index]
                self.crop_index += 1
            return ParseqResult(text, (1,), (0.9,), 0.9, "fixture", input_role, {}, {
                "status": "fixture", "artifact_id": "PARSEQ-FIXTURE",
            })

    fallback = Fallback(events)
    pipeline = ApproximateE2EPipeline(
        ApproximateE2EConfig(
            localization=BlockParam(
                min_area=1,
                pad=0,
                connectivity=8,
                tie_break=APPROXIMATE_TIE_BREAK,
            ),
            segmentation_source="synthetic segmentation fixture",
            scoring_strategy="repository_paired_threshold",
            lexical_mode="raw",
            deepsolo_candidate=CANDIDATE,
            deepsolo_reduction_policy="highest_score",
            deepsolo_inference_options={"device": "cuda:0"},
            t3_selection="repository_max_paired_context_confidence_all_crops",
            t3_selection_classification="REPOSITORY_SPECIFIED",
        ),
        segmenter=ThreeBlockSegmenter(events),
        recognizer=ThreeCropRecognizer(events),
        captioner=Captioner(events, "HIGH"),
        semantic=RepositorySemantic(events, (0.0, 0.0)),
        fallback=fallback,
    )
    image, path = image_fixture(tmp_path)
    trace = pipeline.run(image, image_path=path)

    assert fallback.calls == 0
    assert trace["T3"]["selected"]["raw_text"] == "HIGH"
    assert trace["T3"]["selected_rank"] == 2
    assert [item["raw_text"] for item in trace["T3"]["candidate_scores"]] == [
        "LOW", "HIGH", "HIGH_TIE",
    ]
    assert all(item["evaluated_for_selection"] for item in trace["T3"]["candidate_scores"])
    assert trace["scores"]["strategy"] == "repository_paired_threshold"
    assert trace["final_prediction"] == "HIGH"


def test_repository_crop_selection_requires_repository_strategy() -> None:
    with pytest.raises(ValueError, match="requires repository_paired_threshold"):
        ApproximateE2EConfig(
            localization=BlockParam(
                min_area=1,
                pad=0,
                connectivity=8,
                tie_break=APPROXIMATE_TIE_BREAK,
            ),
            segmentation_source="fixture",
            scoring_strategy="selected_candidate_paired",
            lexical_mode="raw",
            deepsolo_candidate=CANDIDATE,
            deepsolo_reduction_policy="highest_score",
            deepsolo_inference_options={},
            t3_selection="repository_max_paired_context_confidence_all_crops",
            t3_selection_classification="REPOSITORY_SPECIFIED",
        )
