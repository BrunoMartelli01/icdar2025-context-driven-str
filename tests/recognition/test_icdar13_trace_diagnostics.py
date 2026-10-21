from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recognition.icdar13_annotations import ICDAR13RecognitionSample, ICDAR13Word
from recognition.icdar13_trace_diagnostics import (
    ICDAR13TraceDiagnosticError,
    diagnose_trace_index,
)
from recognition.parseq_deepsolo_alignment import (
    ParseqDeepSoloAlignmentError,
    diagnose_parseq_deepsolo_alignment,
)


CONFIG_SHA = "a" * 64
IMAGE_SHA = "b" * 64
ANNOTATION_SHA = "c" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sample(sample_id: str, references: list[str]) -> ICDAR13RecognitionSample:
    return ICDAR13RecognitionSample(
        sample_id=sample_id,
        image_path=Path("image.jpg"),
        image_sha256=IMAGE_SHA,
        annotation_path=Path("gt.txt"),
        annotation_sha256=ANNOTATION_SHA,
        image_size_wh=(10, 10),
        words=tuple(ICDAR13Word((0, 0, 1, 1), value) for value in references),
    )


def _write_fixture(
    root: Path,
    *,
    final_prediction: str = "alpha",
    claims: dict | None = None,
    index_accuracy=None,
) -> Path:
    contract = {
        "reproduction_mode": "APPROXIMATE",
        "claims": {"table_5": False, "table_6": False, "table_7": False},
        "configuration": {"resolved_yaml_sha256": CONFIG_SHA},
        "scientific_boundary": {"accuracy": None, "evaluation_status": "UNRESOLVED"},
    }
    contract_path = root / "batch-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    result_path = root / "samples" / "0001" / "result.json"
    result_path.parent.mkdir(parents=True)
    intermediate_root = result_path.parent / "intermediates"
    intermediate_root.mkdir()
    mask_path = intermediate_root / "segmentation-mask.png"
    crop_path = intermediate_root / "crop-rank-01.png"
    mask_path.write_bytes(b"mask")
    crop_path.write_bytes(b"crop")
    result = {
        "success": True,
        "reproduction_mode": "APPROXIMATE",
        "claims": claims or {"table_5": False, "table_6": False, "table_7": False},
        "image": {"file_sha256": IMAGE_SHA},
        "configuration": {"resolved_yaml_sha256": CONFIG_SHA},
        "segmentation": {
            "mask_artifact": {
                "path": str(mask_path.resolve()),
                "file_sha256": _sha256(mask_path),
            },
            "localized_blocks": [
                {
                    "rank": 1,
                    "padded_bbox": [0, 0, 5, 5],
                    "crop_artifact": {
                        "path": str(crop_path.resolve()),
                        "file_sha256": _sha256(crop_path),
                    }
                }
            ],
        },
        "T1": {"raw_text": "noise"},
        "T3": {
            "selected": {"raw_text": "Beta"},
            "selected_rank": 1,
            "all": [{"raw_text": "Beta", "confidence": 0.9}],
        },
        "fallback": {
            "invoked": True,
            "recognized_strings": ["alpha", "BETA", "extra"],
            "raw_outputs": [
                {
                    "source_index": 0,
                    "text": "BETA",
                    "score": 0.9,
                    "geometry": {
                        "boundary_points": [[1, 1], [4, 1], [4, 4], [1, 4]],
                    },
                },
                {
                    "source_index": 1,
                    "text": "far",
                    "score": 0.8,
                    "geometry": {
                        "boundary_points": [[6, 6], [9, 6], [9, 9], [6, 9]],
                    },
                },
            ],
        },
        "final_prediction": final_prediction,
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    record = {
        "sample_id": "1",
        "success": True,
        "image_sha256": IMAGE_SHA,
        "annotation_sha256": ANNOTATION_SHA,
        "ground_truth_word_count": 2,
        "result": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
    }
    index = {
        "schema_version": 1,
        "reproduction_mode": "APPROXIMATE",
        "claims": {"table_5": False, "table_6": False, "table_7": False},
        "accuracy": index_accuracy,
        "evaluation_status": "UNRESOLVED",
        "batch_contract": {
            "path": str(contract_path.resolve()),
            "sha256": _sha256(contract_path),
        },
        "recorded_sample_count": 1,
        "successful_sample_count": 1,
        "failed_sample_count": 0,
        "fallback_invoked_count": 1,
        "no_fallback_count": 0,
        "samples": [record],
    }
    index_path = root / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index_path


def test_diagnostics_keep_candidate_interpretations_separate(tmp_path: Path) -> None:
    index = _write_fixture(tmp_path)
    report = diagnose_trace_index(index, [_sample("1", ["Alpha", "Beta"])])

    assert report["accuracy"] is None
    assert report["claims"]["table_5"] is False
    assert report["population"]["complete_official_test_233"] is False
    assert report["population"]["verified_intermediate_file_count"] == 2
    final = report["diagnostics"]["literal_final_string_membership"]
    assert final["case_sensitive_hit_count"] == 0
    assert final["casefold_hit_count"] == 1
    fallback = report["diagnostics"]["literal_fallback_multiset_overlap"]
    assert fallback["case_sensitive_one_to_one_match_count"] == 0
    assert fallback["casefold_one_to_one_match_count"] == 2
    assert fallback["casefold_gt_fraction"] == 1.0


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("accuracy", "keep accuracy null"),
        ("claim", "reject Table 5/6/7 claims"),
        ("result_hash", "Result file identity mismatch"),
    ],
)
def test_diagnostics_fail_closed_on_claim_or_identity_drift(
    tmp_path: Path, mutation: str, error: str
) -> None:
    index = _write_fixture(tmp_path, index_accuracy=0.9 if mutation == "accuracy" else None)
    if mutation == "claim":
        result_path = tmp_path / "samples" / "0001" / "result.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["claims"]["table_5"] = True
        result_path.write_text(json.dumps(result), encoding="utf-8")
        stored = json.loads(index.read_text(encoding="utf-8"))
        stored["samples"][0]["result_sha256"] = _sha256(result_path)
        index.write_text(json.dumps(stored), encoding="utf-8")
    elif mutation == "result_hash":
        result_path = tmp_path / "samples" / "0001" / "result.json"
        result_path.write_text(result_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ICDAR13TraceDiagnosticError, match=error):
        diagnose_trace_index(index, [_sample("1", ["Alpha", "Beta"])])


def test_parseq_deepsolo_alignment_keeps_geometry_and_text_separate(
    tmp_path: Path,
) -> None:
    index = _write_fixture(tmp_path)
    report = diagnose_parseq_deepsolo_alignment(
        index, [_sample("1", ["Alpha", "Beta"])]
    )

    assert report["accuracy"] is None
    assert report["claims"]["table_5"] is False
    assert report["population"]["parseq_crop_count"] == 1
    assert report["population"]["deepsolo_detection_count"] == 2
    spatial = report["spatial_envelope_diagnostics"]
    assert spatial["detections_with_positive_overlap"] == 1
    assert spatial["detections_best_coverage_ge_0_8"] == 1
    assert spatial["crops_best_coverage_ge_0_5"] == 0
    text = report["text_diagnostics"]
    assert text["spatially_best_positive_pairs"]["casefold_exact_count"] == 1
    assert text["selected_T3_vs_any_deepsolo_detection"]["casefold_exact_count"] == 1
    assert text["T1_vs_any_deepsolo_detection"]["casefold_exact_count"] == 0
    pairs = report["per_sample"][0]["pairwise_alignment"]
    assert len(pairs) == 2
    assert pairs[0]["deepsolo_envelope_coverage"] == 1.0
    assert pairs[0]["parseq_crop_coverage"] == 0.36
    assert pairs[1]["intersection_area"] == 0.0


def test_parseq_deepsolo_alignment_rejects_crop_prediction_drift(
    tmp_path: Path,
) -> None:
    index = _write_fixture(tmp_path)
    result_path = tmp_path / "samples" / "0001" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["T3"]["all"] = []
    result_path.write_text(json.dumps(result), encoding="utf-8")
    stored_index = json.loads(index.read_text(encoding="utf-8"))
    stored_index["samples"][0]["result_sha256"] = _sha256(result_path)
    index.write_text(json.dumps(stored_index), encoding="utf-8")

    with pytest.raises(ParseqDeepSoloAlignmentError, match="cardinality mismatch"):
        diagnose_parseq_deepsolo_alignment(index, [_sample("1", ["Alpha", "Beta"])])
