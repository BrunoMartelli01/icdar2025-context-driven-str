"""Explicit non-Table-5 diagnostics over frozen ICDAR13 APPROXIMATE traces.

The paper does not define recognition accuracy, normalization, denominator, or the
mapping from multiple outputs to ground truth.  This module therefore reports several
literal observations side by side and deliberately leaves ``accuracy`` null.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from recognition.icdar13_annotations import ICDAR13RecognitionSample


class ICDAR13TraceDiagnosticError(ValueError):
    """Raised when a trace index or result violates the diagnostic contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_false_claims(value: dict[str, Any], *, label: str) -> None:
    claims = value.get("claims")
    expected = {"table_5": False, "table_6": False, "table_7": False}
    if claims != expected:
        raise ICDAR13TraceDiagnosticError(
            "{} must explicitly reject Table 5/6/7 claims".format(label)
        )


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ICDAR13TraceDiagnosticError("{} must be a string".format(label))
    return value


def _any_exact(prediction: str, references: tuple[str, ...], *, casefold: bool) -> bool:
    if not prediction:
        return False
    if casefold:
        prediction = prediction.casefold()
        references = tuple(reference.casefold() for reference in references)
    return prediction in references


def _multiset_intersection_count(
    predictions: Iterable[str], references: tuple[str, ...], *, casefold: bool
) -> int:
    if casefold:
        predictions = (prediction.casefold() for prediction in predictions)
        references = tuple(reference.casefold() for reference in references)
    return sum((Counter(predictions) & Counter(references)).values())


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _verify_file_artifact(
    artifact: Any, *, expected_root: Path, label: str
) -> None:
    if not isinstance(artifact, dict):
        raise ICDAR13TraceDiagnosticError("{} metadata is missing".format(label))
    path = Path(str(artifact.get("path", ""))).resolve()
    try:
        path.relative_to(expected_root)
    except ValueError as error:
        raise ICDAR13TraceDiagnosticError(
            "{} path escapes the sample intermediates directory".format(label)
        ) from error
    if not path.is_file() or _sha256_file(path) != artifact.get("file_sha256"):
        raise ICDAR13TraceDiagnosticError("{} file identity mismatch".format(label))


def diagnose_trace_index(
    index_path: Path,
    samples: Iterable[ICDAR13RecognitionSample],
) -> dict[str, Any]:
    """Validate an APPROXIMATE batch index and expose named literal diagnostics.

    The returned fractions are assumption-based observations, not recognition accuracy.
    Any failed sample suppresses aggregate fractions so it cannot disappear silently.
    """

    index_path = index_path.resolve()
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema_version") != 1:
        raise ICDAR13TraceDiagnosticError("Unsupported trace-index schema")
    if index.get("reproduction_mode") != "APPROXIMATE":
        raise ICDAR13TraceDiagnosticError("Trace index is not explicitly APPROXIMATE")
    _require_false_claims(index, label="Trace index")
    if index.get("accuracy") is not None:
        raise ICDAR13TraceDiagnosticError("Trace index must keep accuracy null")
    if index.get("evaluation_status") != "UNRESOLVED":
        raise ICDAR13TraceDiagnosticError("Trace index must keep evaluation UNRESOLVED")

    contract_record = index.get("batch_contract", {})
    contract_path = Path(str(contract_record.get("path", ""))).resolve()
    expected_contract_path = (index_path.parent / "batch-contract.json").resolve()
    if contract_path != expected_contract_path or not contract_path.is_file():
        raise ICDAR13TraceDiagnosticError("Batch contract path is missing or unexpected")
    contract_sha256 = _sha256_file(contract_path)
    if contract_sha256 != contract_record.get("sha256"):
        raise ICDAR13TraceDiagnosticError("Batch contract SHA256 mismatch")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("reproduction_mode") != "APPROXIMATE":
        raise ICDAR13TraceDiagnosticError("Batch contract is not APPROXIMATE")
    _require_false_claims(contract, label="Batch contract")
    boundary = contract.get("scientific_boundary", {})
    if boundary.get("accuracy") is not None or boundary.get("evaluation_status") != "UNRESOLVED":
        raise ICDAR13TraceDiagnosticError("Batch contract scientific boundary drifted")

    sample_by_id = {sample.sample_id: sample for sample in samples}
    records = index.get("samples")
    if not isinstance(records, list):
        raise ICDAR13TraceDiagnosticError("Trace index samples must be a list")
    ids = [str(record.get("sample_id")) for record in records]
    try:
        numeric_order = sorted(ids, key=int)
    except ValueError as error:
        raise ICDAR13TraceDiagnosticError("Trace sample IDs must be decimal integers") from error
    if len(ids) != len(set(ids)) or ids != numeric_order:
        raise ICDAR13TraceDiagnosticError("Trace sample IDs must be unique and numeric-order sorted")
    if index.get("recorded_sample_count") != len(records):
        raise ICDAR13TraceDiagnosticError("Recorded sample count mismatch")

    per_sample: list[dict[str, Any]] = []
    successful_count = 0
    failed_count = 0
    fallback_count = 0
    no_fallback_count = 0
    verified_intermediate_count = 0
    counters = {
        "final_case_sensitive": 0,
        "final_casefold": 0,
        "t1_case_sensitive": 0,
        "t1_casefold": 0,
        "t3_case_sensitive": 0,
        "t3_casefold": 0,
        "fallback_raw_prediction_count": 0,
        "fallback_ground_truth_count": 0,
        "fallback_multiset_case_sensitive": 0,
        "fallback_multiset_casefold": 0,
    }

    expected_results_root = (index_path.parent / "samples").resolve()
    expected_config_sha256 = contract.get("configuration", {}).get("resolved_yaml_sha256")
    for record in records:
        sample_id = str(record.get("sample_id"))
        sample = sample_by_id.get(sample_id)
        if sample is None:
            raise ICDAR13TraceDiagnosticError("Unknown dataset sample {}".format(sample_id))
        if record.get("image_sha256") != sample.image_sha256:
            raise ICDAR13TraceDiagnosticError("Image identity mismatch for sample {}".format(sample_id))
        if record.get("annotation_sha256") != sample.annotation_sha256:
            raise ICDAR13TraceDiagnosticError(
                "Annotation identity mismatch for sample {}".format(sample_id)
            )
        if record.get("ground_truth_word_count") != len(sample.words):
            raise ICDAR13TraceDiagnosticError(
                "Ground-truth count mismatch for sample {}".format(sample_id)
            )

        if not bool(record.get("success")):
            failed_count += 1
            per_sample.append(
                {
                    "sample_id": sample_id,
                    "success": False,
                    "error": record.get("error"),
                    "diagnostics": None,
                }
            )
            continue

        successful_count += 1
        result_path = Path(str(record.get("result", ""))).resolve()
        expected_result_path = (
            expected_results_root / "{:04d}".format(int(sample_id)) / "result.json"
        ).resolve()
        if result_path != expected_result_path:
            raise ICDAR13TraceDiagnosticError(
                "Result path is unexpected for sample {}".format(sample_id)
            )
        if not result_path.is_file() or _sha256_file(result_path) != record.get("result_sha256"):
            raise ICDAR13TraceDiagnosticError(
                "Result file identity mismatch for sample {}".format(sample_id)
            )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("success") is not True or result.get("reproduction_mode") != "APPROXIMATE":
            raise ICDAR13TraceDiagnosticError(
                "Successful result contract mismatch for sample {}".format(sample_id)
            )
        _require_false_claims(result, label="Result {}".format(sample_id))
        if result.get("image", {}).get("file_sha256") != sample.image_sha256:
            raise ICDAR13TraceDiagnosticError(
                "Result image identity mismatch for sample {}".format(sample_id)
            )
        if result.get("configuration", {}).get("resolved_yaml_sha256") != expected_config_sha256:
            raise ICDAR13TraceDiagnosticError(
                "Result configuration mismatch for sample {}".format(sample_id)
            )

        expected_intermediate_root = (result_path.parent / "intermediates").resolve()
        segmentation = result.get("segmentation", {})
        _verify_file_artifact(
            segmentation.get("mask_artifact"),
            expected_root=expected_intermediate_root,
            label="Sample {} segmentation mask".format(sample_id),
        )
        verified_intermediate_count += 1
        localized_blocks = segmentation.get("localized_blocks")
        if not isinstance(localized_blocks, list):
            raise ICDAR13TraceDiagnosticError(
                "Localized blocks are missing for sample {}".format(sample_id)
            )
        for block_index, block in enumerate(localized_blocks, start=1):
            _verify_file_artifact(
                block.get("crop_artifact") if isinstance(block, dict) else None,
                expected_root=expected_intermediate_root,
                label="Sample {} crop {}".format(sample_id, block_index),
            )
            verified_intermediate_count += 1

        references = tuple(word.transcription for word in sample.words)
        final_prediction = _string(result.get("final_prediction"), label="final_prediction")
        t1 = _string(result.get("T1", {}).get("raw_text"), label="T1.raw_text")
        t3 = _string(
            result.get("T3", {}).get("selected", {}).get("raw_text"),
            label="T3.selected.raw_text",
        )
        fallback = result.get("fallback", {})
        invoked = fallback.get("invoked")
        if not isinstance(invoked, bool):
            raise ICDAR13TraceDiagnosticError("Fallback invocation must be boolean")
        raw_outputs = fallback.get("recognized_strings", [])
        if not isinstance(raw_outputs, list):
            raise ICDAR13TraceDiagnosticError("Fallback recognized strings must be a list")
        raw_outputs = [
            _string(value, label="fallback.recognized_strings") for value in raw_outputs
        ]
        if not invoked and raw_outputs:
            raise ICDAR13TraceDiagnosticError(
                "Non-invoked fallback returned strings for sample {}".format(sample_id)
            )

        flags = {
            "final_exact_any_gt_case_sensitive": _any_exact(
                final_prediction, references, casefold=False
            ),
            "final_exact_any_gt_casefold": _any_exact(
                final_prediction, references, casefold=True
            ),
            "t1_exact_any_gt_case_sensitive": _any_exact(t1, references, casefold=False),
            "t1_exact_any_gt_casefold": _any_exact(t1, references, casefold=True),
            "selected_t3_exact_any_gt_case_sensitive": _any_exact(
                t3, references, casefold=False
            ),
            "selected_t3_exact_any_gt_casefold": _any_exact(t3, references, casefold=True),
        }
        counters["final_case_sensitive"] += int(flags["final_exact_any_gt_case_sensitive"])
        counters["final_casefold"] += int(flags["final_exact_any_gt_casefold"])
        counters["t1_case_sensitive"] += int(flags["t1_exact_any_gt_case_sensitive"])
        counters["t1_casefold"] += int(flags["t1_exact_any_gt_casefold"])
        counters["t3_case_sensitive"] += int(flags["selected_t3_exact_any_gt_case_sensitive"])
        counters["t3_casefold"] += int(flags["selected_t3_exact_any_gt_casefold"])

        if invoked:
            fallback_count += 1
            counters["fallback_raw_prediction_count"] += len(raw_outputs)
            counters["fallback_ground_truth_count"] += len(references)
            counters["fallback_multiset_case_sensitive"] += _multiset_intersection_count(
                raw_outputs, references, casefold=False
            )
            counters["fallback_multiset_casefold"] += _multiset_intersection_count(
                raw_outputs, references, casefold=True
            )
        else:
            no_fallback_count += 1

        per_sample.append(
            {
                "sample_id": sample_id,
                "success": True,
                "ground_truth": list(references),
                "T1": t1,
                "selected_T3": t3,
                "final_prediction": final_prediction,
                "fallback_invoked": invoked,
                "fallback_recognized_strings": raw_outputs,
                "diagnostics": flags,
            }
        )

    if index.get("successful_sample_count") != successful_count:
        raise ICDAR13TraceDiagnosticError("Successful sample count mismatch")
    if index.get("failed_sample_count") != failed_count:
        raise ICDAR13TraceDiagnosticError("Failed sample count mismatch")
    if index.get("fallback_invoked_count") != fallback_count:
        raise ICDAR13TraceDiagnosticError("Fallback count mismatch")
    if index.get("no_fallback_count") != no_fallback_count:
        raise ICDAR13TraceDiagnosticError("No-fallback count mismatch")

    fractions_available = failed_count == 0
    denominator = successful_count if fractions_available else 0
    diagnostics = {
        "aggregation_guard": {
            "fractions_available": fractions_available,
            "rule": (
                "Fractions are suppressed when any recorded sample failed; failures are never omitted."
            ),
        },
        "literal_final_string_membership": {
            "unit": "one successful source image",
            "denominator": denominator if fractions_available else None,
            "case_sensitive_hit_count": counters["final_case_sensitive"],
            "case_sensitive_fraction": (
                _fraction(counters["final_case_sensitive"], denominator)
                if fractions_available else None
            ),
            "casefold_hit_count": counters["final_casefold"],
            "casefold_fraction": (
                _fraction(counters["final_casefold"], denominator)
                if fractions_available else None
            ),
        },
        "literal_primary_string_membership": {
            "unit": "one successful source image",
            "denominator": denominator if fractions_available else None,
            "T1_case_sensitive_hit_count": counters["t1_case_sensitive"],
            "T1_case_sensitive_fraction": (
                _fraction(counters["t1_case_sensitive"], denominator)
                if fractions_available else None
            ),
            "T1_casefold_hit_count": counters["t1_casefold"],
            "T1_casefold_fraction": (
                _fraction(counters["t1_casefold"], denominator)
                if fractions_available else None
            ),
            "selected_T3_case_sensitive_hit_count": counters["t3_case_sensitive"],
            "selected_T3_case_sensitive_fraction": (
                _fraction(counters["t3_case_sensitive"], denominator)
                if fractions_available else None
            ),
            "selected_T3_casefold_hit_count": counters["t3_casefold"],
            "selected_T3_casefold_fraction": (
                _fraction(counters["t3_casefold"], denominator)
                if fractions_available else None
            ),
        },
        "literal_fallback_multiset_overlap": {
            "scope": "successful samples on which fallback was invoked",
            "fallback_sample_count": fallback_count,
            "raw_prediction_count": counters["fallback_raw_prediction_count"],
            "ground_truth_word_count": counters["fallback_ground_truth_count"],
            "case_sensitive_one_to_one_match_count": counters[
                "fallback_multiset_case_sensitive"
            ],
            "case_sensitive_gt_fraction": (
                _fraction(
                    counters["fallback_multiset_case_sensitive"],
                    counters["fallback_ground_truth_count"],
                ) if fractions_available else None
            ),
            "casefold_one_to_one_match_count": counters["fallback_multiset_casefold"],
            "casefold_gt_fraction": (
                _fraction(
                    counters["fallback_multiset_casefold"],
                    counters["fallback_ground_truth_count"],
                ) if fractions_available else None
            ),
        },
    }

    expected_complete_ids = [str(value) for value in range(1, 234)]
    return {
        "schema_version": 1,
        "reproduction_mode": "APPROXIMATE",
        "result_class": "approximate recognition diagnostics; not a Table 5 result",
        "claims": {"table_5": False, "table_6": False, "table_7": False},
        "accuracy": None,
        "evaluation_status": "UNRESOLVED",
        "scientific_boundary": {
            "classification": "ASSUMPTION",
            "metric_defined": False,
            "normalization_defined": False,
            "denominator_defined": False,
            "multiple_output_mapping_defined": False,
            "blocking_ambiguity_ids": [
                "R-003", "S-001", "S-003", "D-002", "DS-002", "E-001"
            ],
            "prohibition": "No diagnostic fraction may be reported as Table 5 accuracy.",
        },
        "source": {
            "index_path": str(index_path),
            "index_sha256": _sha256_file(index_path),
            "batch_contract_path": str(contract_path),
            "batch_contract_sha256": contract_sha256,
        },
        "population": {
            "recorded_sample_ids": ids,
            "recorded_sample_count": len(ids),
            "successful_sample_count": successful_count,
            "failed_sample_count": failed_count,
            "complete_official_test_233": ids == expected_complete_ids,
            "verified_intermediate_file_count": verified_intermediate_count,
        },
        "routing": {
            "fallback_invoked_count": fallback_count,
            "no_fallback_count": no_fallback_count,
        },
        "diagnostics": diagnostics,
        "per_sample": per_sample,
    }


__all__ = ["ICDAR13TraceDiagnosticError", "diagnose_trace_index"]
