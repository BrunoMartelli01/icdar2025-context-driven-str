#!/usr/bin/env python3
"""TRAIN-only evaluation of frozen practical word-candidate selectors.

All selector decisions are computed before Task-2.2 GT is opened.  The script
reuses frozen probability, candidate, PARSeq, BLIP/MPNet and v1 artifacts; it
does not run segmentation, OCR, captioning, or model training.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Sequence

import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.multihypothesis_word_proposals import THRESHOLDS, enumerate_native_hypotheses
from localize.practical_candidate_selection import (
    VARIANTS, attach_family_consensus, attach_variant_scores, box_iou,
    candidate_density, intersection_area, select_candidates, selector_contract,
    span_boundary_score, v1_support,
)
from localize.practical_remap_protocol import proposal_geometry, train_task22_word_boxes
from pipeline.frozen_segmentation import backproject_binary_mask
from scripts.diagnose_blip_context_candidate_ranking_train import (
    EXPECTED_CANDIDATE_METADATA_SHA256, EXPECTED_CANDIDATES,
    EXPECTED_GENERATION_MANIFEST_SHA256, EXPECTED_OCR_MANIFEST_SHA256,
    EXPECTED_SOURCE_REPORT_SHA256, _load_candidates_without_gt,
)
from scripts.diagnose_multihypothesis_word_proposal_oracle_train import (
    EXPECTED_CHECKPOINT_SHA256, EXPECTED_PROBABILITY_MANIFEST_SHA256,
    EXPECTED_V1_SHA256, _load_probability,
)
from segmentation.data import load_segmentation_manifest


MODE = "PRACTICAL_CANDIDATE_SELECTION_TRAIN_V1"
EXPECTED_CONTEXT_REPORT_SHA256 = "26e06c4a7f83d7d1dde7c17a8a424e1397a3b9de194b5260b5fe220dc6662a76"
EXPECTED_MASK_MANIFEST_SHA256 = "e9a7ce5badfbc3d99885fef5e655b20f7f5f167f29defa2d3f3c7e91c6d5851a"
EXPECTED_V1_OCR_REPORT_SHA256 = "68c6a8279949a4de141cca0d3ed363b52a01c6a53cac62943cbe687fdc4fe462"
EXPECTED_IMAGES = 229
EXPECTED_WORDS = 849
SIGNIFICANT_GT_COVERAGE = 0.25
TASK22_LINE = re.compile(r'^\s*(\d+)\s+(\d+)\s+(\d+).*"(.*)"\s*$')


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--v1-ocr-report", type=Path, required=True)
    parser.add_argument("--parseq-source-report", type=Path, required=True)
    parser.add_argument("--context-report", type=Path, required=True)
    parser.add_argument("--candidate-artifact-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    return parser.parse_args()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("artifact path must be relative")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact escapes declared root")
    return path


def _write_exclusive(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content); stream.flush(); os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _task22_texts(path: Path) -> tuple[str, ...]:
    words: list[str] = []
    current: list[str] = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        if not raw.strip():
            if current:
                words.append("".join(current)); current = []
            continue
        if raw.lstrip().startswith("#"):
            continue
        match = TASK22_LINE.match(raw)
        if match is None:
            raise ValueError(f"malformed Task-2.2 annotation: {path}")
        current.append(match.group(4))
    if current:
        words.append("".join(current))
    return tuple(words)


def _wordness(box: Sequence[int], gt_boxes: Sequence[Sequence[int]]) -> tuple[str, int | None, float]:
    if not gt_boxes:
        return "UNMATCHED", None, 0.0
    overlaps = []
    for index, gt in enumerate(gt_boxes):
        intersection = intersection_area(box, gt)
        gt_area = max(1, (gt[2] - gt[0]) * (gt[3] - gt[1]))
        overlaps.append((box_iou(box, gt), intersection / gt_area, index))
    best = max(overlaps, key=lambda value: (value[0], value[1], -value[2]))
    significant = sum(value[1] >= SIGNIFICANT_GT_COVERAGE for value in overlaps)
    category = (
        "MULTIWORD_MERGE" if significant >= 2 else
        "WORD_GOOD" if best[0] >= 0.5 else
        "WORD_PARTIAL" if significant == 1 else "UNMATCHED"
    )
    return category, best[2], best[0]


def _load_mask(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = _contained(root, record["native_mask_path"])
    if _sha(path) != record["native_mask_file_sha256"]:
        raise RuntimeError(f"native mask file SHA mismatch: {record['sample_id']}")
    with Image.open(path) as opened:
        value = (np.asarray(opened.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
    if value.shape != (128, 128) or hashlib.sha256(value.tobytes()).hexdigest() != record["native_mask_array_sha256"]:
        raise RuntimeError(f"native mask content mismatch: {record['sample_id']}")
    return np.ascontiguousarray(value)


def _enrich_candidates(
    rows: list[dict[str, Any]], *, probability: np.ndarray,
    image_width: int, image_height: int, anchors: Sequence[Sequence[int]],
    context_by_id: dict[str, float],
) -> None:
    hypotheses = {
        threshold: enumerate_native_hypotheses(
            np.ascontiguousarray((probability >= threshold).astype(np.uint8)), max_components=12,
        )
        for threshold in THRESHOLDS
    }
    for row in rows:
        candidate_id = str(row["candidate_id"])
        threshold = float(row["source_threshold"])
        hypothesis = hypotheses[threshold]
        raw_by_id = {value.component_id: value for value in hypothesis.raw_components}
        line_id = int(row["source_line_ids"][0])
        line = next(value for value in hypothesis.base_lines if int(value["line_id"]) == line_id)
        boxes = [raw_by_id[int(value)].bbox for value in line["raw_component_ids"]]
        foreground_area = sum(
            raw_by_id[int(value)].foreground_area for value in row["source_raw_component_ids"]
        )
        row["foreground_area_native"] = foreground_area
        row["foreground_density"] = candidate_density(foreground_area, row["source_bbox_native"])
        row["boundary_score"] = span_boundary_score(
            component_boxes=boxes,
            start=int(row["source_span_start"]), end=int(row["source_span_end"]),
        )
        row["v1_support"] = v1_support(row["bbox"], anchors)
        row["context_score"] = context_by_id[candidate_id]
    attach_family_consensus(rows)
    attach_variant_scores(rows)


def _evaluate_variant(
    *, variant: str, selections: dict[str, tuple[dict[str, Any], ...]],
    pool_by_sample: dict[str, list[dict[str, Any]]], gt_by_sample: dict[str, list[dict[str, Any]]],
    v1_success: dict[tuple[str, int], bool],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    geometry_input = []
    wordness = Counter()
    per_word = []
    selected_total = 0
    failure = Counter()
    recovered = lost = 0
    for sample_id, words in gt_by_sample.items():
        selected = selections[sample_id]
        selected_boxes = [row["bbox"] for row in selected]
        gt_boxes = [row["bbox"] for row in words]
        selected_total += len(selected)
        for candidate in selected:
            category, _, _ = _wordness(candidate["bbox"], gt_boxes)
            wordness[category] += 1
        geometry_input.append((selected_boxes, gt_boxes, len(selected), 0, 0))
        pool = pool_by_sample[sample_id]
        for index, word in enumerate(words, start=1):
            best_selected = max(selected, key=lambda row: box_iou(row["bbox"], word["bbox"]), default=None)
            best_iou = box_iou(best_selected["bbox"], word["bbox"]) if best_selected else 0.0
            available = best_selected is not None and intersection_area(best_selected["bbox"], word["bbox"]) > 0
            predicted = str(best_selected["ocr"]["raw_text"]) if available else None
            fuzzy = ratio(predicted.casefold(), word["text"].casefold()) / 100 if predicted is not None else None
            pool_best = max((box_iou(row["bbox"], word["bbox"]) for row in pool), default=0.0)
            success = best_iou >= 0.5
            if success:
                category = "SUCCESS"
            elif pool_best < 0.5:
                category = "GENERATOR_LIMIT"
            else:
                intersecting = [
                    row for row in selected if intersection_area(row["bbox"], word["bbox"]) > 0
                ]
                multi = any(_wordness(row["bbox"], gt_boxes)[0] == "MULTIWORD_MERGE" for row in intersecting)
                category = "SELECTOR_OVERMERGE" if multi else "SELECTOR_UNDERMERGE_OR_MISS"
            failure[category] += 1
            was_v1 = v1_success[(sample_id, index)]
            recovered += int(success and not was_v1)
            lost += int(was_v1 and not success)
            per_word.append({
                "sample_id": sample_id, "gt_word_index": index,
                "gt_bbox": word["bbox"], "gt_text": word["text"],
                "selected_candidate_id": best_selected["candidate_id"] if available else None,
                "selected_bbox": best_selected["bbox"] if available else None,
                "selected_text": predicted, "best_iou": best_iou,
                "exact_casefold": predicted is not None and predicted.casefold() == word["text"].casefold(),
                "rapidfuzz_casefold": fuzzy,
                "rapidfuzz_casefold_ge_0_8": fuzzy is not None and fuzzy >= 0.8,
                "pool_best_iou": pool_best, "selection_outcome": category,
            })
    geometry = proposal_geometry(geometry_input).as_dict()
    exact = sum(row["exact_casefold"] for row in per_word)
    fuzzy = sum(row["rapidfuzz_casefold_ge_0_8"] for row in per_word)
    available = sum(row["selected_text"] is not None for row in per_word)
    summary = {
        "variant": variant,
        "geometry": geometry,
        "ocr": {
            "gt_words": len(per_word), "available_predictions": available,
            "exact_casefold": exact, "exact_rate_all_gt": exact / len(per_word),
            "rapidfuzz_casefold_ge_0_8": fuzzy, "fuzzy_rate_all_gt": fuzzy / len(per_word),
            "median_rapidfuzz_available": float(np.median([
                row["rapidfuzz_casefold"] for row in per_word if row["rapidfuzz_casefold"] is not None
            ])),
        },
        "wordness": {"total_selected": selected_total, **{
            name: wordness[name] for name in
            ("WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED")
        }},
        "selection_failure_breakdown": dict(sorted(failure.items())),
        "vs_v1_per_word_geometry": {"failures_recovered": recovered, "successes_lost": lost},
    }
    return summary, per_word


def main() -> int:
    args = _args()
    paths = {name: value.resolve() for name, value in vars(args).items() if isinstance(value, Path)}
    expected = {
        "probability_manifest": EXPECTED_PROBABILITY_MANIFEST_SHA256,
        "mask_manifest": EXPECTED_MASK_MANIFEST_SHA256,
        "v1_calibration": EXPECTED_V1_SHA256,
        "v1_ocr_report": EXPECTED_V1_OCR_REPORT_SHA256,
        "parseq_source_report": EXPECTED_SOURCE_REPORT_SHA256,
        "context_report": EXPECTED_CONTEXT_REPORT_SHA256,
    }
    for name, digest in expected.items():
        if _sha(paths[name]) != digest:
            raise RuntimeError(f"frozen {name} SHA256 mismatch")
    if paths["output"].exists() or paths["csv_output"].exists():
        raise FileExistsError("refusing to overwrite candidate-selection diagnostic")

    probability_report = json.loads(paths["probability_manifest"].read_text(encoding="utf-8"))
    mask_report = json.loads(paths["mask_manifest"].read_text(encoding="utf-8"))
    v1_report = json.loads(paths["v1_calibration"].read_text(encoding="utf-8"))
    context_report = json.loads(paths["context_report"].read_text(encoding="utf-8"))
    v1_ocr = json.loads(paths["v1_ocr_report"].read_text(encoding="utf-8"))
    manifest = load_segmentation_manifest(paths["train_manifest"])
    if (
        len(manifest.samples) != EXPECTED_IMAGES or manifest.split_role != "train"
        or probability_report.get("test_data_loaded") is not False
        or probability_report.get("gt_loaded") is not False
        or probability_report.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or mask_report.get("test_data_loaded") is not False
        or len(context_report.get("candidate_scores", [])) != EXPECTED_CANDIDATES
    ):
        raise RuntimeError("TRAIN-only frozen artifact contract mismatch")

    candidates = _load_candidates_without_gt(paths["candidate_artifact_dir"])
    context_by_id = {
        str(row["candidate_id"]): float(row["fixed_context_score"])
        for row in context_report["candidate_scores"]
    }
    if set(context_by_id) != {str(row["candidate_id"]) for row in candidates}:
        raise RuntimeError("context and candidate populations differ")
    by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_sample.setdefault(str(row["sample_id"]), []).append(row)
    probability_by_id = {str(row["sample_id"]): row for row in probability_report["samples"]}
    mask_by_id = {str(row["sample_id"]): row for row in mask_report["samples"]}
    params = GroupedWordBlockParams(**v1_report["calibration"]["selected"]["parameters"])

    # Inference-only boundary.  No GT path is opened before all three selector
    # variants have produced their final per-image outputs.
    selections: dict[str, dict[str, tuple[dict[str, Any], ...]]] = {
        variant: {} for variant in VARIANTS
    }
    anchors_by_sample: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
    option_diagnostics = []
    for sample_number, sample in enumerate(manifest.samples, start=1):
        sample_id = str(sample.sample_id)
        image_path = _contained(paths["data_root"], sample.image_path)
        with Image.open(image_path) as opened:
            width, height = opened.size
        probability = _load_probability(paths["probability_manifest"].parent, probability_by_id[sample_id])
        native_mask = _load_mask(paths["mask_manifest"].parent, mask_by_id[sample_id])
        if not np.array_equal(native_mask, (probability >= 0.5).astype(np.uint8)):
            raise RuntimeError(f"probability/mask cache mismatch: {sample_id}")
        projected = backproject_binary_mask(native_mask, (height, width))
        v1 = group_word_blocks(projected, np.zeros((height, width), dtype=np.uint8), params)
        anchors = tuple(tuple(int(value) for value in block.padded_bbox) for block in v1.retained)
        anchors_by_sample[sample_id] = anchors
        rows = [dict(row) for row in by_sample[sample_id]]
        _enrich_candidates(
            rows, probability=probability, image_width=width, image_height=height,
            anchors=anchors, context_by_id=context_by_id,
        )
        for variant in VARIANTS:
            selections[variant][sample_id] = select_candidates(rows, anchors=anchors, variant=variant)
        option_diagnostics.append({
            "sample_id": sample_id, "pool_candidates": len(rows), "v1_anchors": len(anchors),
            "selected": {variant: len(selections[variant][sample_id]) for variant in VARIANTS},
        })
        if sample_number == 1 or sample_number % 25 == 0 or sample_number == EXPECTED_IMAGES:
            print(f"[selector/no-GT] {sample_number}/{EXPECTED_IMAGES} sample_id={sample_id}", flush=True)

    # Post-hoc evaluation boundary: GT geometry/text becomes visible only here.
    gt_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in v1_ocr["per_word"]:
        gt_by_sample.setdefault(str(row["sample_id"]), []).append({
            "bbox": list(row["gt_bbox"]), "text": str(row["gt_text"]),
            "gt_word_index": int(row["gt_index"]),
        })
    for sample_id, words in gt_by_sample.items():
        words.sort(key=lambda row: row["gt_word_index"])
        if [row["gt_word_index"] for row in words] != list(range(1, len(words) + 1)):
            raise RuntimeError(f"frozen TRAIN GT ordering mismatch: {sample_id}")
    if sum(map(len, gt_by_sample.values())) != EXPECTED_WORDS:
        raise RuntimeError("TRAIN GT cardinality mismatch")

    v1_success = {
        (str(row["sample_id"]), int(row["gt_index"])):
            float(row["PRACTICAL_REMAP_V1"]["bbox_iou"]) >= 0.5
        for row in v1_ocr["per_word"]
    }
    summaries = {}
    per_word_by_variant = {}
    for variant in VARIANTS:
        summary, rows = _evaluate_variant(
            variant=variant, selections=selections[variant], pool_by_sample=by_sample,
            gt_by_sample=gt_by_sample, v1_success=v1_success,
        )
        summaries[variant] = summary
        per_word_by_variant[variant] = rows
    chosen = max(VARIANTS, key=lambda variant: (
        summaries[variant]["ocr"]["exact_casefold"],
        summaries[variant]["geometry"]["gt_with_proposal_iou_ge_0_50"],
        -summaries[variant]["wordness"]["MULTIWORD_MERGE"],
        -VARIANTS.index(variant),
    ))

    baseline = {
        "geometry": v1_ocr["PRACTICAL_REMAP_V1"]["geometry"],
        "ocr": {key: v1_ocr["PRACTICAL_REMAP_V1"][key] for key in (
            "gt_words_total", "available_predictions", "exact_casefold",
            "exact_percent_of_all_gt", "rapidfuzz_casefold_ge_0_8",
            "fuzzy_percent_of_all_gt", "median_rapidfuzz_available",
        )},
        "wordness": v1_ocr["PRACTICAL_REMAP_V1"]["wordness"],
    }
    independent = json.loads(paths["parseq_source_report"].read_text(encoding="utf-8"))["ranking"]
    report = {
        "schema_version": 1, "mode": MODE,
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC", "table_5_claim": False,
        "scope": {"split": "TRAIN", "images": EXPECTED_IMAGES, "gt_words": EXPECTED_WORDS},
        "scientific_boundary": {
            "segmentation_rerun": False, "cuda_used": False, "test_loaded": False,
            "training": False, "gt_used_by_selector": False,
            "gt_first_loaded_after_all_selections_frozen": True,
            "paper_faithful_algorithm1_changed": False,
            "practical_remap_v1_changed": False, "practical_remap_v2_changed": False,
        },
        "selector_contract": selector_contract(),
        "variant_selection_rule_declared_before_evaluation": [
            "maximize TRAIN exact-casefold over all 849 GT words",
            "then maximize GT words with selected proposal IoU>=0.50",
            "then minimize selected MULTIWORD_MERGE crops",
            "then prefer declared variant order",
        ],
        "chosen_variant": chosen,
        "baseline_practical_remap_v1": baseline,
        "variants": summaries,
        "frozen_pool_context": {
            "candidates": EXPECTED_CANDIDATES,
            "oracle_gt_with_candidate_iou_ge_0_50": 735,
            "oracle_gt_with_iou_ge_0_50_and_parseq_exact": 667,
            "independent_mean_token_log_probability": independent["mean_token_log_probability"],
        },
        "per_image_option_diagnostics": option_diagnostics,
        "per_word_chosen_variant": per_word_by_variant[chosen],
        "provenance": {
            "probability_manifest_sha256": EXPECTED_PROBABILITY_MANIFEST_SHA256,
            "mask_manifest_sha256": EXPECTED_MASK_MANIFEST_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "v1_calibration_sha256": EXPECTED_V1_SHA256,
            "v1_ocr_report_sha256": EXPECTED_V1_OCR_REPORT_SHA256,
            "parseq_source_report_sha256": EXPECTED_SOURCE_REPORT_SHA256,
            "candidate_generation_manifest_sha256": EXPECTED_GENERATION_MANIFEST_SHA256,
            "candidate_metadata_sha256": EXPECTED_CANDIDATE_METADATA_SHA256,
            "candidate_ocr_manifest_sha256": EXPECTED_OCR_MANIFEST_SHA256,
            "context_report_sha256": EXPECTED_CONTEXT_REPORT_SHA256,
            "config_sha256": _sha(paths["config"]),
            "selector_source_sha256": _sha(Path(__file__).resolve().parents[1] / "localize/practical_candidate_selection.py"),
        },
    }
    _write_exclusive(paths["output"], json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    csv_fields = (
        "sample_id", "gt_word_index", "gt_text", "selected_candidate_id", "selected_text",
        "best_iou", "exact_casefold", "rapidfuzz_casefold", "selection_outcome",
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=paths["csv_output"].parent, delete=False) as stream:
        temporary = Path(stream.name); writer = csv.DictWriter(stream, fieldnames=csv_fields); writer.writeheader()
        for row in per_word_by_variant[chosen]:
            writer.writerow({key: row[key] for key in csv_fields})
        stream.flush(); os.fsync(stream.fileno())
    try:
        if paths["csv_output"].exists():
            raise FileExistsError(f"refusing to overwrite {paths['csv_output']}")
        os.link(temporary, paths["csv_output"])
    finally:
        temporary.unlink()
    print(json.dumps({"chosen_variant": chosen, "baseline": baseline, "variants": summaries}, indent=2), flush=True)
    print(f"JSON {paths['output']} SHA256 {_sha(paths['output'])}", flush=True)
    print(f"CSV {paths['csv_output']} SHA256 {_sha(paths['csv_output'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
