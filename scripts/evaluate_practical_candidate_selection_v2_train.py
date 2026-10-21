#!/usr/bin/env python3
"""Evaluate anchor-complete proposal selection on frozen ICDAR13 TRAIN artifacts.

The only model call is PARSeq on the exact practical-remap-v1 crop population.
Candidate alternatives reuse previously frozen OCR.  GT is opened only after
all candidate features and selections are complete.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio
import yaml

from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.multihypothesis_word_proposals import (
    BBOX_VARIANTS, DEDUPLICATION_IOU, enumerate_native_hypotheses,
    deduplicate_candidates_with_provenance, materialize_bbox_variants,
)
from localize.practical_candidate_selection import (
    box_iou, candidate_density, intersection_area, span_boundary_score,
)
from localize.practical_candidate_selection_v2 import (
    VARIANTS, form_anchor_lines, select_anchor_complete, selector_contract,
)
from localize.practical_remap_protocol import proposal_geometry
from pipeline.frozen_segmentation import backproject_binary_mask
from recognition.parseq import (
    PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256, PARSEQ_RUNTIME_REVISION,
)
from scripts.diagnose_blip_context_candidate_ranking_train import (
    EXPECTED_CANDIDATES, _load_candidates_without_gt,
)
from scripts.diagnose_multihypothesis_word_proposal_oracle_train import (
    EXPECTED_CHECKPOINT_SHA256, EXPECTED_PROBABILITY_MANIFEST_SHA256,
    EXPECTED_V1_SHA256, _load_probability,
)
from scripts.diagnose_parseq_candidate_ranking_train import _run_or_resume_ocr
from segmentation.data import load_segmentation_manifest


MODE = "PRACTICAL_CANDIDATE_SELECTION_V2_TRAIN"
EXPECTED_MASK_MANIFEST_SHA256 = "e9a7ce5badfbc3d99885fef5e655b20f7f5f167f29defa2d3f3c7e91c6d5851a"
EXPECTED_V1_OCR_REPORT_SHA256 = "68c6a8279949a4de141cca0d3ed363b52a01c6a53cac62943cbe687fdc4fe462"
EXPECTED_CONTEXT_REPORT_SHA256 = "26e06c4a7f83d7d1dde7c17a8a424e1397a3b9de194b5260b5fe220dc6662a76"
EXPECTED_IMAGES = 229
EXPECTED_WORDS = 849
THRESHOLD_ORDER = (0.50, 0.40, 0.30, 0.20)
SIGNIFICANT_GT_COVERAGE = 0.25


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probability-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--v1-reference-report", type=Path, required=True)
    parser.add_argument("--context-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidate-artifact-dir", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path)
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
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
        raise ValueError("relative artifact path required")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes declared root")
    return path


def _write_atomic(path: Path, value: Any, *, exclusive: bool) -> None:
    if exclusive and path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    if exclusive:
        try:
            os.link(temporary, path)
        finally:
            temporary.unlink()
    else:
        os.replace(temporary, path)


def _load_native_mask(root: Path, record: dict[str, Any]) -> np.ndarray:
    path = _contained(root, record["native_mask_path"])
    if _sha(path) != record["native_mask_file_sha256"]:
        raise RuntimeError(f"native-mask file SHA mismatch: {record['sample_id']}")
    with Image.open(path) as opened:
        value = (np.asarray(opened.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
    if value.shape != (128, 128) or hashlib.sha256(value.tobytes()).hexdigest() != record["native_mask_array_sha256"]:
        raise RuntimeError(f"native-mask array mismatch: {record['sample_id']}")
    return np.ascontiguousarray(value)


def _prepare_exact_v1_artifacts(
    *, artifact_dir: Path, manifest: Any, data_root: Path,
    probability_manifest: Path, probability_report: dict[str, Any],
    mask_manifest: Path, mask_report: dict[str, Any], params: GroupedWordBlockParams,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metadata_path = artifact_dir / "v1-candidates.jsonl"
    generation_path = artifact_dir / "generation-manifest.json"
    contract = {
        "probability_manifest_sha256": _sha(probability_manifest),
        "mask_manifest_sha256": _sha(mask_manifest),
        "v1_parameters": params.__dict__, "sample_count": EXPECTED_IMAGES,
        "gt_loaded": False, "test_loaded": False,
    }
    if artifact_dir.exists():
        if not metadata_path.is_file() or not generation_path.is_file():
            raise RuntimeError("existing v2 artifact directory is incomplete")
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if generation.get("contract") != contract or _sha(metadata_path) != generation.get("metadata_sha256"):
            raise RuntimeError("existing exact-v1 artifact contract mismatch")
        rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines()]
        if len(rows) != generation.get("candidate_count"):
            raise RuntimeError("exact-v1 artifact cardinality mismatch")
        print(f"[v1 crop preparation] reusable candidates={len(rows)}", flush=True)
        return rows, generation

    staging = Path(tempfile.mkdtemp(prefix="candidate-selection-v2-", dir=artifact_dir.parent))
    probability_by_id = {str(row["sample_id"]): row for row in probability_report["samples"]}
    mask_by_id = {str(row["sample_id"]): row for row in mask_report["samples"]}
    rows = []
    samples = []
    try:
        crops_root = staging / "crops"; crops_root.mkdir()
        with (staging / "v1-candidates.jsonl").open("w", encoding="utf-8") as stream:
            for number, sample in enumerate(manifest.samples, start=1):
                sample_id = str(sample.sample_id)
                image_path = _contained(data_root, sample.image_path)
                if _sha(image_path) != probability_by_id[sample_id]["image_sha256"]:
                    raise RuntimeError(f"TRAIN image SHA mismatch: {sample_id}")
                with Image.open(image_path) as opened:
                    rgb = np.asarray(opened.convert("RGB"))
                probability = _load_probability(probability_manifest.parent, probability_by_id[sample_id])
                native = _load_native_mask(mask_manifest.parent, mask_by_id[sample_id])
                if not np.array_equal(native, (probability >= 0.5).astype(np.uint8)):
                    raise RuntimeError(f"TRAIN probability/mask mismatch: {sample_id}")
                projected = backproject_binary_mask(native, rgb.shape[:2])
                result = group_word_blocks(projected, rgb, params)
                sample_dir = crops_root / f"{int(sample_id):04d}"; sample_dir.mkdir()
                sample_rows = []
                for block in result.retained:
                    candidate_id = f"v1:{int(sample_id):04d}:{block.rank:03d}"
                    crop_path = sample_dir / f"v1-{block.rank:03d}.png"
                    Image.fromarray(block.crop, mode="RGB").save(crop_path)
                    row = {
                        "candidate_id": candidate_id, "sample_id": sample_id,
                        "candidate_index": block.rank, "source": "v1_exact",
                        "bbox": list(block.padded_bbox), "tight_bbox": list(block.bbox),
                        "raw_component_ids": list(block.raw_component_ids),
                        "foreground_area_projected_mask": int(block.foreground_area),
                        "foreground_density": candidate_density(block.foreground_area, block.bbox),
                        "crop_path": str(crop_path.relative_to(staging)),
                        "crop_sha256": _sha(crop_path), "crop_shape_hwc": list(block.crop.shape),
                    }
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    rows.append(row); sample_rows.append(row)
                samples.append({"sample_id": sample_id, "candidate_count": len(sample_rows)})
                print(f"[v1 crop preparation/no-GT] {number}/{EXPECTED_IMAGES} sample_id={sample_id} crops={len(sample_rows)}", flush=True)
            stream.flush(); os.fsync(stream.fileno())
        generation = {
            "schema_version": 1, "artifact": "practical-candidate-selection-v2-exact-v1-crops",
            "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
            "contract": contract, "candidate_count": len(rows), "samples": samples,
            "metadata_sha256": _sha(staging / "v1-candidates.jsonl"),
        }
        _write_atomic(staging / "generation-manifest.json", generation, exclusive=False)
        os.replace(staging, artifact_dir)
        return rows, generation
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _candidate_provenance_and_features(
    *, candidates: list[dict[str, Any]], probability: np.ndarray,
    width: int, height: int, context_by_id: dict[str, float],
) -> list[dict[str, Any]]:
    master = []
    hypotheses = {}
    for threshold in THRESHOLD_ORDER:
        hypothesis = enumerate_native_hypotheses(
            np.ascontiguousarray((probability >= threshold).astype(np.uint8)), max_components=12,
        )
        hypotheses[threshold] = hypothesis
        master.extend(
            row for row in materialize_bbox_variants(
                hypothesis, threshold=threshold, width=width, height=height,
            ) if row["source_kind"] == "same_line"
        )
    provenance = deduplicate_candidates_with_provenance(master, threshold=DEDUPLICATION_IOU)
    if len(provenance) != len(candidates):
        raise RuntimeError("provenance-preserving dedup changed candidate count")
    output = []
    for frozen, source in zip(candidates, provenance):
        if list(source["bbox_original"]) != list(frozen["bbox"]):
            raise RuntimeError(f"provenance-preserving dedup changed candidate order: {frozen['candidate_id']}")
        hypothesis = hypotheses[float(frozen["source_threshold"])]
        raw_by_id = {row.component_id: row for row in hypothesis.raw_components}
        line_id = int(frozen["source_line_ids"][0])
        line = next(row for row in hypothesis.base_lines if int(row["line_id"]) == line_id)
        component_boxes = [raw_by_id[int(value)].bbox for value in line["raw_component_ids"]]
        foreground_area = sum(
            raw_by_id[int(value)].foreground_area for value in frozen["source_raw_component_ids"]
        )
        threshold_support = len({float(row["threshold"]) for row in source["dedup_sources"]})
        stability = float(np.mean([
            box_iou(source["bbox_original"], row["bbox_original"])
            for row in source["dedup_sources"]
        ]))
        output.append({
            **frozen, "source": "multihypothesis_alternative",
            "dedup_sources": source["dedup_sources"],
            "consensus_threshold_support": threshold_support,
            "consensus_stability": stability,
            "consensus_score": (threshold_support / len(THRESHOLD_ORDER)) * stability,
            "foreground_area_native": foreground_area,
            "foreground_density": candidate_density(foreground_area, frozen["source_bbox_native"]),
            "boundary_score": span_boundary_score(
                component_boxes=component_boxes,
                start=int(frozen["source_span_start"]), end=int(frozen["source_span_end"]),
            ),
            "context_score": context_by_id[frozen["candidate_id"]],
        })
    return output


def _attach_base_features(anchors: list[dict[str, Any]], alternatives: list[dict[str, Any]]) -> None:
    lines = form_anchor_lines(anchors)
    position = {anchor: (line_id, offset) for line_id, line in enumerate(lines) for offset, anchor in enumerate(line)}
    for anchor_index, anchor in enumerate(anchors):
        matches = sorted(
            alternatives,
            key=lambda row: (-box_iou(row["bbox"], anchor["bbox"]), row["candidate_id"]),
        )
        best = matches[0] if matches and box_iou(matches[0]["bbox"], anchor["bbox"]) >= 0.5 else None
        line_id, offset = position[anchor_index]
        line_boxes = [anchors[value]["tight_bbox"] for value in lines[line_id]]
        anchor["boundary_score"] = span_boundary_score(
            component_boxes=line_boxes, start=offset, end=offset + 1,
        )
        anchor["consensus_score"] = float(best["consensus_score"]) if best else 0.0
        anchor["context_score"] = float(best["context_score"]) if best else 0.0


def _wordness(box: Sequence[int], gt_boxes: Sequence[Sequence[int]]) -> str:
    overlaps = []
    for gt in gt_boxes:
        intersection = intersection_area(box, gt)
        gt_area = max(1, (gt[2] - gt[0]) * (gt[3] - gt[1]))
        overlaps.append((box_iou(box, gt), intersection / gt_area))
    best = max((value[0] for value in overlaps), default=0.0)
    significant = sum(value[1] >= SIGNIFICANT_GT_COVERAGE for value in overlaps)
    return (
        "MULTIWORD_MERGE" if significant >= 2 else "WORD_GOOD" if best >= 0.5
        else "WORD_PARTIAL" if significant == 1 else "UNMATCHED"
    )


def _evaluate(
    selections: dict[str, tuple[dict[str, Any], ...]],
    gt_by_sample: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    geometry_input = []; wordness = Counter(); per_word = []
    for sample_id, words in gt_by_sample.items():
        selected = selections[sample_id]
        boxes = [row["bbox"] for row in selected]; gt_boxes = [row["bbox"] for row in words]
        geometry_input.append((boxes, gt_boxes, len(boxes), 0, 0))
        wordness.update(_wordness(row["bbox"], gt_boxes) for row in selected)
        for index, word in enumerate(words, start=1):
            best = max(selected, key=lambda row: box_iou(row["bbox"], word["bbox"]), default=None)
            best_iou = box_iou(best["bbox"], word["bbox"]) if best else 0.0
            available = best is not None and intersection_area(best["bbox"], word["bbox"]) > 0
            text = str(best["ocr"]["raw_text"]) if available else None
            fuzzy = ratio(text.casefold(), word["text"].casefold()) / 100 if text is not None else None
            per_word.append({
                "sample_id": sample_id, "gt_word_index": index, "gt_text": word["text"],
                "gt_bbox": word["bbox"], "candidate_id": best["candidate_id"] if available else None,
                "candidate_source": best["source"] if available else None,
                "candidate_bbox": best["bbox"] if available else None, "best_iou": best_iou,
                "text": text, "exact_casefold": text is not None and text.casefold() == word["text"].casefold(),
                "rapidfuzz_casefold": fuzzy,
                "rapidfuzz_casefold_ge_0_8": fuzzy is not None and fuzzy >= 0.8,
            })
    geometry = proposal_geometry(geometry_input).as_dict()
    exact = sum(row["exact_casefold"] for row in per_word)
    fuzzy = sum(row["rapidfuzz_casefold_ge_0_8"] for row in per_word)
    summary = {
        "geometry": geometry,
        "ocr": {
            "gt_words": len(per_word), "available_predictions": sum(row["text"] is not None for row in per_word),
            "exact_casefold": exact, "exact_rate": exact / len(per_word),
            "rapidfuzz_casefold_ge_0_8": fuzzy, "fuzzy_rate": fuzzy / len(per_word),
        },
        "wordness": {"total": sum(wordness.values()), **{
            name: wordness[name] for name in ("WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED")
        }},
        "selection": {
            "exact_v1_candidates": sum(
                row["source"] == "v1_exact" for selected in selections.values() for row in selected
            ),
            "alternative_candidates": sum(
                row["source"] != "v1_exact" for selected in selections.values() for row in selected
            ),
        },
    }
    return summary, per_word


def main() -> int:
    args = _args()
    probability_manifest = args.probability_manifest.resolve()
    mask_manifest = args.mask_manifest.resolve()
    train_manifest = args.train_manifest.resolve()
    data_root = args.data_root.resolve()
    v1_path = args.v1_calibration.resolve()
    reference_path = args.v1_reference_report.resolve()
    context_path = args.context_report.resolve()
    config_path = args.config.resolve()
    artifact_dir = args.artifact_dir.resolve()
    output = args.output.resolve()
    expected = {
        probability_manifest: EXPECTED_PROBABILITY_MANIFEST_SHA256,
        mask_manifest: EXPECTED_MASK_MANIFEST_SHA256,
        v1_path: EXPECTED_V1_SHA256,
        reference_path: EXPECTED_V1_OCR_REPORT_SHA256,
        context_path: EXPECTED_CONTEXT_REPORT_SHA256,
    }
    for path, digest in expected.items():
        if _sha(path) != digest:
            raise RuntimeError(f"frozen artifact SHA mismatch: {path}")
    resolved_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        resolved_config.get("mode") != "practical-candidate-selection-v2"
        or resolved_config.get("classification") != "PRACTICAL_APPROXIMATION"
        or resolved_config.get("selector", {}).get("variants") != list(VARIANTS)
        or resolved_config.get("generator", {}).get("thresholds") != [0.20, 0.30, 0.40, 0.50]
    ):
        raise RuntimeError("candidate-selection-v2 config contract mismatch")
    probability_report = json.loads(probability_manifest.read_text(encoding="utf-8"))
    mask_report = json.loads(mask_manifest.read_text(encoding="utf-8"))
    v1_report = json.loads(v1_path.read_text(encoding="utf-8"))
    manifest = load_segmentation_manifest(train_manifest)
    if (
        len(manifest.samples) != EXPECTED_IMAGES or manifest.split_role != "train"
        or probability_report.get("gt_loaded") is not False
        or probability_report.get("test_data_loaded") is not False
        or probability_report.get("checkpoint", {}).get("sha256") != EXPECTED_CHECKPOINT_SHA256
        or mask_report.get("test_data_loaded") is not False
    ):
        raise RuntimeError("requires frozen TRAIN-229 caches")
    params = GroupedWordBlockParams(**v1_report["calibration"]["selected"]["parameters"])
    anchors, generation = _prepare_exact_v1_artifacts(
        artifact_dir=artifact_dir, manifest=manifest, data_root=data_root,
        probability_manifest=probability_manifest, probability_report=probability_report,
        mask_manifest=mask_manifest, mask_report=mask_report, params=params,
    )
    if args.prepare_only:
        print(json.dumps({
            "status": "READY_FOR_PARSEQ_GPU", "artifact_dir": str(artifact_dir),
            "exact_v1_crops": len(anchors), "generation_manifest_sha256": _sha(artifact_dir / "generation-manifest.json"),
        }, indent=2), flush=True)
        return 0
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if args.device != "cuda:0" or not all((args.parseq_source, args.parseq_source_archive, args.parseq_checkpoint)):
        raise ValueError("evaluation requires qualified PARSeq arguments and --device cuda:0")
    if _sha(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256:
        raise RuntimeError("PARSeq checkpoint SHA mismatch")

    ocr_by_id, ocr_manifest = _run_or_resume_ocr(
        artifact_dir=artifact_dir, candidates=anchors, batch_size=args.batch_size,
        parseq_source=args.parseq_source.resolve(),
        parseq_source_archive=args.parseq_source_archive.resolve(),
        parseq_checkpoint=args.parseq_checkpoint.resolve(), device=args.device,
    )
    for anchor in anchors:
        anchor["ocr"] = ocr_by_id[anchor["candidate_id"]]

    frozen_candidates = _load_candidates_without_gt(args.candidate_artifact_dir.resolve())
    context_report = json.loads(context_path.read_text(encoding="utf-8"))
    context_by_id = {str(row["candidate_id"]): float(row["fixed_context_score"]) for row in context_report["candidate_scores"]}
    if len(frozen_candidates) != EXPECTED_CANDIDATES or set(context_by_id) != {row["candidate_id"] for row in frozen_candidates}:
        raise RuntimeError("frozen candidate/context populations differ")
    probability_by_id = {str(row["sample_id"]): row for row in probability_report["samples"]}
    anchors_by_sample: dict[str, list[dict[str, Any]]] = {}
    alternatives_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in anchors:
        anchors_by_sample.setdefault(str(row["sample_id"]), []).append(row)
    frozen_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in frozen_candidates:
        frozen_by_sample.setdefault(str(row["sample_id"]), []).append(row)

    selections = {variant: {} for variant in VARIANTS}
    control = {}
    support_diagnostics = Counter()
    for number, sample in enumerate(manifest.samples, start=1):
        sample_id = str(sample.sample_id)
        with Image.open(_contained(data_root, sample.image_path)) as opened:
            width, height = opened.size
        probability = _load_probability(probability_manifest.parent, probability_by_id[sample_id])
        alternatives = _candidate_provenance_and_features(
            candidates=frozen_by_sample[sample_id], probability=probability,
            width=width, height=height, context_by_id=context_by_id,
        )
        sample_anchors = anchors_by_sample[sample_id]
        _attach_base_features(sample_anchors, alternatives)
        control[sample_id] = tuple(sample_anchors)
        for row in alternatives:
            support_diagnostics[f"threshold_support_{row['consensus_threshold_support']}"] += 1
        for variant in VARIANTS:
            selections[variant][sample_id] = select_anchor_complete(
                [dict(row) for row in sample_anchors], [dict(row) for row in alternatives], variant=variant,
            )
        print(f"[v2 selection/no-GT] {number}/{EXPECTED_IMAGES} sample_id={sample_id}", flush=True)

    # GT is accessed only after every selection is fixed.
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    gt_by_sample: dict[str, list[dict[str, Any]]] = {}
    for row in reference["per_word"]:
        gt_by_sample.setdefault(str(row["sample_id"]), []).append({
            "gt_word_index": int(row["gt_index"]), "bbox": list(row["gt_bbox"]), "text": str(row["gt_text"]),
        })
    for rows in gt_by_sample.values():
        rows.sort(key=lambda row: row["gt_word_index"])
    if sum(map(len, gt_by_sample.values())) != EXPECTED_WORDS:
        raise RuntimeError("TRAIN GT cardinality mismatch")
    control_summary, control_rows = _evaluate(control, gt_by_sample)
    if (
        control_summary["geometry"]["gt_with_proposal_iou_ge_0_50"] != 568
        or control_summary["ocr"]["exact_casefold"] != 502
        or control_summary["ocr"]["rapidfuzz_casefold_ge_0_8"] != 578
    ):
        raise RuntimeError("exact-v1 control failed to reproduce frozen geometry/OCR")
    summaries = {}; rows_by_variant = {}
    for variant in VARIANTS:
        summaries[variant], rows_by_variant[variant] = _evaluate(selections[variant], gt_by_sample)
    chosen = max(VARIANTS, key=lambda variant: (
        summaries[variant]["ocr"]["exact_casefold"],
        summaries[variant]["geometry"]["gt_with_proposal_iou_ge_0_50"],
        -summaries[variant]["wordness"]["MULTIWORD_MERGE"], -VARIANTS.index(variant),
    ))
    promote = (
        summaries[chosen]["ocr"]["exact_casefold"] > control_summary["ocr"]["exact_casefold"]
        and summaries[chosen]["geometry"]["gt_with_proposal_iou_ge_0_50"]
        >= control_summary["geometry"]["gt_with_proposal_iou_ge_0_50"]
        and summaries[chosen]["wordness"]["MULTIWORD_MERGE"]
        <= control_summary["wordness"]["MULTIWORD_MERGE"]
    )
    report = {
        "schema_version": 1, "mode": MODE,
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC", "table_5_claim": False,
        "scope": {"split": "TRAIN", "images": EXPECTED_IMAGES, "gt_words": EXPECTED_WORDS},
        "scientific_boundary": {
            "segmentation_rerun": False, "test_loaded": False, "training": False,
            "gt_used_by_selector": False, "gt_first_loaded_after_selection": True,
        },
        "selector_contract": selector_contract(),
        "deduplication_provenance": dict(sorted(support_diagnostics.items())),
        "v1_exact_control": control_summary, "variants": summaries,
        "chosen_variant": chosen,
        "promotion_rule": "exact>v1 AND IoU>=.50 recall>=v1 AND MULTIWORD<=v1",
        "decision": "PROMOTE" if promote else "REJECT",
        "per_word_chosen_variant": rows_by_variant[chosen],
        "provenance": {
            "probability_manifest_sha256": EXPECTED_PROBABILITY_MANIFEST_SHA256,
            "mask_manifest_sha256": EXPECTED_MASK_MANIFEST_SHA256,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            "v1_calibration_sha256": EXPECTED_V1_SHA256,
            "v1_reference_report_sha256": EXPECTED_V1_OCR_REPORT_SHA256,
            "context_report_sha256": EXPECTED_CONTEXT_REPORT_SHA256,
            "resolved_config": resolved_config,
            "config_sha256": _sha(config_path),
            "exact_v1_generation_manifest_sha256": _sha(artifact_dir / "generation-manifest.json"),
            "exact_v1_ocr": ocr_manifest,
            "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
            "parseq_runtime_revision": PARSEQ_RUNTIME_REVISION,
            "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
            "selector_source_sha256": _sha(Path(__file__).resolve().parents[1] / "localize/practical_candidate_selection_v2.py"),
            "diagnostic_source_sha256": _sha(Path(__file__).resolve()),
        },
    }
    _write_atomic(output, report, exclusive=True)
    print(json.dumps({
        "v1_exact_control": control_summary, "variants": summaries,
        "chosen_variant": chosen, "decision": report["decision"],
    }, indent=2), flush=True)
    print(f"JSON {output}\nSHA256 {_sha(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
