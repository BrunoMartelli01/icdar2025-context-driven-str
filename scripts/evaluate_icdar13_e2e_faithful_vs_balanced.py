#!/usr/bin/env python3
"""Run a resumable ICDAR13 TEST E2E diagnostic for Algorithm1 and balanced mean.

The comparison shares every model output that is independent of localization.
It deliberately reports no Table-5 accuracy: the paper's recognition unit and
matching rule are unresolved.  Explicit image-membership and one-to-one text
multiset diagnostics are emitted instead.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch
import yaml

from caption.blip2_captioner import BLIP2ApproxConfig, BLIP2Captioner
from deepsolo.adapter import DeepSoloSubprocessAdapter
from deepsolo.candidates import QUALIFIED_CANDIDATE_ID
from localize.grouped_word_blocks import GroupedWordBlockParams, group_word_blocks
from localize.mask_to_blocks import APPROXIMATE_TIE_BREAK, BlockParam, localize_mask
from localize.multihypothesis_word_proposals import (
    DEDUPLICATION_IOU, deduplicate_candidates_with_provenance,
    enumerate_native_hypotheses, materialize_bbox_variants,
)
from localize.practical_candidate_selection import candidate_density, span_boundary_score
from localize.practical_candidate_selection_v2 import (
    form_anchor_lines, select_anchor_complete,
)
from pipeline.e2e_comparison_diagnostic import (
    CLASSIFICATION, PAPER_ICDAR13_R50_ACCURACY_PERCENT, multiset_exact,
    repository_route, word_crop_diagnostic,
)
from pipeline.frozen_segmentation import FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig
from recognition.icdar13_annotations import load_icdar13_recognition_samples
from recognition.parseq import (
    PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    PARSEQ_RUNTIME_REVISION, ParseqApproxConfig, ParseqApproxRecognizer,
)
from scoring.lexical import lexical_similarity
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from segmentation.data import load_segmentation_manifest


EXPECTED_CHECKPOINT_SHA256 = "b450f19e79f649f8873a8b8db0d9a25ab0a7f144ac5f8676a2a80d5c1d8fd2f2"
EXPECTED_V1_CALIBRATION_SHA256 = "6c0e4cb36fe49f2c1d2306fc7226a5a7d7938208b7b223b4490a6dae377a659f"
EXPECTED_BALANCED_TRAIN_SHA256 = "8db4dac11c9d08e956d1f11c982b97c5e538e009f3dff9d0b5bd0e393b7a94e5"
EXPECTED_DEEPSOLO_RUNTIME_SHA256 = "c52a65b7d844cd35bcf6e38e3cb58c5194851c8327cbdaba9b313869ec2c7ffa"
EXPECTED_SEGMENTATION_MANIFEST_SHA256 = "8bffaa558f687d5ae6a6e41a30a4a51db682124da99d4c2496aa3fa4a2a001d8"
EXPECTED_RECOGNITION_MANIFEST_SHA256 = "d8ab8863fa2dd00cdbf0667fb36c20a28432ba070164b85d9dbff0827dfb20bd"
EXPECTED_IMAGES = 233
EXPECTED_WORDS = 1095
THRESHOLD_ORDER = (0.50, 0.40, 0.30, 0.20)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-manifest", type=Path, required=True)
    parser.add_argument("--recognition-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--evaluation-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--v1-calibration", type=Path, required=True)
    parser.add_argument("--balanced-train-report", type=Path, required=True)
    parser.add_argument("--comparison-config", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--deepsolo-python", type=Path, required=True)
    parser.add_argument("--deepsolo-runtime-manifest", type=Path, required=True)
    parser.add_argument("--deepsolo-source-root", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def _contained(root: Path, relative: str | Path) -> Path:
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("relative artifact path required")
    value = (root / relative).resolve()
    if not value.is_relative_to(root.resolve()):
        raise ValueError("artifact path escapes data root")
    return value


def _release_cuda(component: Any) -> dict[str, Any]:
    record = component.release()
    del component
    gc.collect()
    torch.cuda.empty_cache()
    return record


def _crop_key(box: Sequence[int]) -> str:
    return "bbox:" + ",".join(str(int(value)) for value in box)


def _load_shard(path: Path, *, contract: str, sample_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("contract_sha256") != contract or value.get("sample_id") != sample_id:
        raise RuntimeError(f"stale or invalid resumable shard: {path}")
    return value


def _proposal_features(probability: np.ndarray, *, width: int, height: int, sample_id: str) -> list[dict[str, Any]]:
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
    retained = deduplicate_candidates_with_provenance(master, threshold=DEDUPLICATION_IOU)
    output = []
    for index, source in enumerate(retained, start=1):
        threshold = float(source["threshold"])
        hypothesis = hypotheses[threshold]
        raw_by_id = {row.component_id: row for row in hypothesis.raw_components}
        line_id = int(source["source_line_ids"][0])
        line = next(row for row in hypothesis.base_lines if int(row["line_id"]) == line_id)
        component_boxes = [raw_by_id[int(value)].bbox for value in line["raw_component_ids"]]
        foreground_area = sum(
            raw_by_id[int(value)].foreground_area for value in source["raw_component_ids"]
        )
        threshold_support = len({float(row["threshold"]) for row in source["dedup_sources"]})
        stability = float(np.mean([
            _box_iou(source["bbox_original"], row["bbox_original"])
            for row in source["dedup_sources"]
        ]))
        box = [int(value) for value in source["bbox_original"]]
        output.append({
            "candidate_id": f"alternative:{int(sample_id):04d}:{index:05d}",
            "sample_id": sample_id, "candidate_index": index,
            "source": "multihypothesis_alternative", "bbox": box,
            "source_threshold": threshold,
            "source_bbox_native": [int(value) for value in source["bbox_native"]],
            "source_raw_component_ids": [int(value) for value in source["raw_component_ids"]],
            "source_line_ids": [int(value) for value in source["source_line_ids"]],
            "source_span_start": int(source["start"]), "source_span_end": int(source["end"]),
            "bbox_variant": str(source["bbox_variant"]),
            "dedup_sources": source["dedup_sources"],
            "consensus_threshold_support": threshold_support,
            "consensus_stability": stability,
            "consensus_score": (threshold_support / len(THRESHOLD_ORDER)) * stability,
            "foreground_area_native": int(foreground_area),
            "foreground_density": candidate_density(foreground_area, source["bbox_native"]),
            "boundary_score": span_boundary_score(
                component_boxes=component_boxes, start=int(source["start"]), end=int(source["end"]),
            ),
            "ocr_key": _crop_key(box),
        })
    return output


def _box_iou(left: Sequence[int], right: Sequence[int]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    b = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = a + b - intersection
    return intersection / union if union else 0.0


def _build_proposal_shards(
    *, args: argparse.Namespace, samples: Sequence[Any], params: GroupedWordBlockParams,
    contract: str, root: Path,
) -> dict[str, Any]:
    stage = root / "01-proposals"; stage.mkdir(parents=True, exist_ok=True)
    pending = [sample for sample in samples if not (stage / f"{int(sample.sample_id):04d}.json").exists()]
    segmenter = None
    release = {"model_reference_released": False}
    try:
        if pending:
            segmenter = FrozenAuthorProtocolSegmenter(FrozenSegmentationConfig(
                evaluation_config=args.evaluation_config.resolve(),
                checkpoint=args.checkpoint.resolve(), device=args.device,
            ))
        for number, sample in enumerate(samples, start=1):
            path = stage / f"{int(sample.sample_id):04d}.json"
            existing = _load_shard(path, contract=contract, sample_id=sample.sample_id)
            if existing is not None:
                print(f"[segmentation/proposals] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} cached", flush=True)
                continue
            image_path = _contained(args.data_root.resolve(), sample.image_path)
            if _sha(image_path) != sample.image_sha256:
                raise RuntimeError(f"TEST image SHA mismatch: {sample.sample_id}")
            with Image.open(image_path) as opened:
                rgb = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
            segmentation = segmenter(rgb)
            projected = segmentation["mask"]
            faithful_result = localize_mask(projected, rgb, BlockParam(
                min_area=150, pad=6, connectivity=8,
                tie_break=APPROXIMATE_TIE_BREAK, nmax=10,
            ))
            grouped = group_word_blocks(projected, rgb, params)
            faithful = [{
                "candidate_id": f"faithful:{int(sample.sample_id):04d}:{row.rank:03d}",
                "source": "Algorithm1", "rank": row.rank,
                "bbox": list(row.padded_bbox), "tight_bbox": list(row.bbox),
                "component_area": row.component_area, "ocr_key": _crop_key(row.padded_bbox),
            } for row in faithful_result.retained]
            anchors = [{
                "candidate_id": f"v1:{int(sample.sample_id):04d}:{row.rank:03d}",
                "sample_id": sample.sample_id, "candidate_index": row.rank,
                "source": "v1_exact", "bbox": list(row.padded_bbox),
                "tight_bbox": list(row.bbox), "raw_component_ids": list(row.raw_component_ids),
                "foreground_area_projected_mask": row.foreground_area,
                "foreground_density": candidate_density(row.foreground_area, row.bbox),
                "ocr_key": _crop_key(row.padded_bbox),
            } for row in grouped.retained]
            alternatives = _proposal_features(
                segmentation["probability_native"], width=rgb.shape[1],
                height=rgb.shape[0], sample_id=sample.sample_id,
            )
            _write_atomic(path, {
                "schema_version": 1, "contract_sha256": contract,
                "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "segmentation": segmentation["metadata"],
                "faithful": faithful, "v1_anchors": anchors, "alternatives": alternatives,
            })
            print(
                f"[segmentation/proposals] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} "
                f"faithful={len(faithful)} v1={len(anchors)} alternatives={len(alternatives)}",
                flush=True,
            )
    finally:
        if segmenter is not None:
            release = _release_cuda(segmenter)
    return release


def _run_ocr_shards(
    *, args: argparse.Namespace, samples: Sequence[Any], contract: str, root: Path,
) -> dict[str, Any]:
    proposals_root, stage = root / "01-proposals", root / "02-parseq"
    stage.mkdir(parents=True, exist_ok=True)
    pending = [sample for sample in samples if not (stage / f"{int(sample.sample_id):04d}.json").exists()]
    recognizer = None
    release = {"model_reference_released": False}
    try:
        if pending:
            recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                runtime_source=args.parseq_source.resolve(), checkpoint=args.parseq_checkpoint.resolve(),
                device=args.device, runtime_source_type="github_commit_archive",
                runtime_source_archive=args.parseq_source_archive.resolve(),
            ))
        for number, sample in enumerate(samples, start=1):
            path = stage / f"{int(sample.sample_id):04d}.json"
            existing = _load_shard(path, contract=contract, sample_id=sample.sample_id)
            if existing is not None:
                print(f"[PARSeq] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} cached", flush=True)
                continue
            proposal_path = proposals_root / f"{int(sample.sample_id):04d}.json"
            proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
            image_path = _contained(args.data_root.resolve(), sample.image_path)
            with Image.open(image_path) as opened:
                rgb = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
            t1 = recognizer.recognize(rgb, input_role="T1_full_image").to_dict()
            boxes: dict[str, list[int]] = {}
            for family in ("faithful", "v1_anchors", "alternatives"):
                for row in proposal[family]:
                    boxes.setdefault(row["ocr_key"], [int(value) for value in row["bbox"]])
            keys = sorted(boxes)
            results: dict[str, Any] = {}
            for offset in range(0, len(keys), args.batch_size):
                batch_keys = keys[offset:offset + args.batch_size]
                crops = []
                for key in batch_keys:
                    x1, y1, x2, y2 = boxes[key]
                    crops.append(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
                values = recognizer.recognize_candidate_batch(crops)
                results.update({key: value.to_dict() for key, value in zip(batch_keys, values)})
            _write_atomic(path, {
                "schema_version": 1, "contract_sha256": contract,
                "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "proposal_shard_sha256": _sha(proposal_path), "T1": t1,
                "unique_crop_count": len(results), "crops": results,
            })
            print(
                f"[PARSeq] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} "
                f"unique_crops={len(results)} T1={t1['raw_text']!r}", flush=True,
            )
    finally:
        if recognizer is not None:
            release = _release_cuda(recognizer)
    return release


def _run_caption_shards(
    *, args: argparse.Namespace, samples: Sequence[Any], contract: str, root: Path,
) -> dict[str, Any]:
    stage = root / "03-blip2"; stage.mkdir(parents=True, exist_ok=True)
    pending = [sample for sample in samples if not (stage / f"{int(sample.sample_id):04d}.json").exists()]
    captioner = None
    release = {"model_reference_released": False, "processor_reference_released": False}
    try:
        if pending:
            captioner = BLIP2Captioner(BLIP2ApproxConfig(
                device=args.device, dtype="float16", placement="single_device", quantization="none",
            ))
        for number, sample in enumerate(samples, start=1):
            path = stage / f"{int(sample.sample_id):04d}.json"
            existing = _load_shard(path, contract=contract, sample_id=sample.sample_id)
            if existing is not None:
                print(f"[BLIP-2] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} cached", flush=True)
                continue
            image_path = _contained(args.data_root.resolve(), sample.image_path)
            with Image.open(image_path) as opened:
                rgb = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
            result = captioner.caption(rgb).to_dict()
            _write_atomic(path, {
                "schema_version": 1, "contract_sha256": contract,
                "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "caption": result,
            })
            print(
                f"[BLIP-2] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} "
                f"caption={result['raw_caption']!r}", flush=True,
            )
    finally:
        if captioner is not None:
            release = _release_cuda(captioner)
    return release


def _candidate_rows(rows: list[dict[str, Any]], ocr: dict[str, Any]) -> list[dict[str, Any]]:
    return [{**row, "ocr": ocr[row["ocr_key"]]} for row in rows]


def _attach_anchor_features(
    anchors: list[dict[str, Any]], alternatives: list[dict[str, Any]],
) -> None:
    """Mirror the frozen TRAIN selector's GT-blind v1-anchor features."""
    lines = form_anchor_lines(anchors)
    position = {
        anchor: (line_id, offset)
        for line_id, line in enumerate(lines)
        for offset, anchor in enumerate(line)
    }
    for anchor_index, anchor in enumerate(anchors):
        matches = sorted(
            alternatives,
            key=lambda row: (-_box_iou(row["bbox"], anchor["bbox"]), row["candidate_id"]),
        )
        best = matches[0] if matches and _box_iou(matches[0]["bbox"], anchor["bbox"]) >= 0.5 else None
        line_id, offset = position[anchor_index]
        line_boxes = [anchors[value]["tight_bbox"] for value in lines[line_id]]
        anchor["boundary_score"] = span_boundary_score(
            component_boxes=line_boxes, start=offset, end=offset + 1,
        )
        anchor["consensus_score"] = float(best["consensus_score"]) if best else 0.0
        anchor["context_score"] = float(best["context_score"]) if best else 0.0


def _score_and_select_shards(
    *, samples: Sequence[Any], contract: str, root: Path,
) -> dict[str, Any]:
    stage = root / "04-routing"; stage.mkdir(parents=True, exist_ok=True)
    scorer = MPNetSemanticSimilarity(MPNetApproxConfig(device="cpu", precision="float32"))
    try:
        for number, sample in enumerate(samples, start=1):
            path = stage / f"{int(sample.sample_id):04d}.json"
            existing = _load_shard(path, contract=contract, sample_id=sample.sample_id)
            if existing is not None:
                print(f"[MPNet/routing] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} cached", flush=True)
                continue
            proposal = json.loads((root / "01-proposals" / f"{int(sample.sample_id):04d}.json").read_text())
            ocr_shard = json.loads((root / "02-parseq" / f"{int(sample.sample_id):04d}.json").read_text())
            caption_shard = json.loads((root / "03-blip2" / f"{int(sample.sample_id):04d}.json").read_text())
            crops_ocr = ocr_shard["crops"]
            caption = str(caption_shard["caption"]["raw_caption"])
            t1 = str(ocr_shard["T1"]["raw_text"])
            faithful = _candidate_rows(proposal["faithful"], crops_ocr)
            anchors = _candidate_rows(proposal["v1_anchors"], crops_ocr)
            alternatives = _candidate_rows(proposal["alternatives"], crops_ocr)
            all_rows = faithful + anchors + alternatives
            texts = [t1] + [str(row["ocr"]["raw_text"]) for row in all_rows] + [caption]
            embeddings = scorer.embeddings(texts)
            caption_embedding = embeddings[-1]
            similarities = np.clip(embeddings[:-1] @ caption_embedding, -1.0, 1.0)
            s1 = float(similarities[0])
            for row, semantic in zip(all_rows, similarities[1:]):
                row["semantic_similarity"] = float(semantic)
                lexical = lexical_similarity(str(row["ocr"]["raw_text"]), caption, "raw").score
                row["context_score"] = 0.6 * float(semantic) + 0.4 * float(lexical)
            faithful_count, anchor_count = len(faithful), len(anchors)
            faithful = all_rows[:faithful_count]
            anchors = all_rows[faithful_count:faithful_count + anchor_count]
            alternatives = all_rows[faithful_count + anchor_count:]
            _attach_anchor_features(anchors, alternatives)
            balanced = list(select_anchor_complete(anchors, alternatives, variant="BALANCED_ANCHOR_MEAN"))

            def routed(rows: list[dict[str, Any]]) -> dict[str, Any]:
                candidates = [{
                    "candidate_id": row["candidate_id"], "bbox": row["bbox"],
                    "text": row["ocr"]["raw_text"],
                    "semantic_similarity": row["semantic_similarity"],
                } for row in rows]
                return repository_route(t1=t1, caption=caption, s1=s1, candidates=candidates)

            faithful_route, balanced_route = routed(faithful), routed(balanced)
            _write_atomic(path, {
                "schema_version": 1, "contract_sha256": contract,
                "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "T1": t1, "T2": caption,
                "faithful": {"proposals": [{
                    "candidate_id": row["candidate_id"], "bbox": row["bbox"],
                    "text": row["ocr"]["raw_text"],
                } for row in faithful], "route": faithful_route},
                "balanced": {"proposals": [{
                    "candidate_id": row["candidate_id"], "bbox": row["bbox"],
                    "text": row["ocr"]["raw_text"], "source": row["source"],
                } for row in balanced], "route": balanced_route},
            })
            print(
                f"[MPNet/routing] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} "
                f"fallback faithful={faithful_route['needs_fallback']} balanced={balanced_route['needs_fallback']}",
                flush=True,
            )
    finally:
        release = scorer.release()
    return release


def _run_deepsolo_shards(
    *, args: argparse.Namespace, samples: Sequence[Any], contract: str, root: Path,
) -> None:
    routing_root, stage = root / "04-routing", root / "05-deepsolo"
    stage.mkdir(parents=True, exist_ok=True)
    adapter = DeepSoloSubprocessAdapter(
        args.deepsolo_python, args.deepsolo_runtime_manifest,
        host_source_root=args.deepsolo_source_root,
    )
    for number, sample in enumerate(samples, start=1):
        path = stage / f"{int(sample.sample_id):04d}.json"
        existing = _load_shard(path, contract=contract, sample_id=sample.sample_id)
        if existing is not None:
            print(f"[DeepSolo] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} cached", flush=True)
            continue
        routing = json.loads((routing_root / f"{int(sample.sample_id):04d}.json").read_text())
        needed = routing["faithful"]["route"]["needs_fallback"] or routing["balanced"]["route"]["needs_fallback"]
        response = None
        if needed:
            response = adapter.infer(
                image_path=_contained(args.data_root.resolve(), sample.image_path),
                candidate_id=QUALIFIED_CANDIDATE_ID,
                inference_options={"device": "cuda:0", "score_threshold": None, "lexicon_type": None},
                reduction_policy="highest_score",
            ).to_dict()
        _write_atomic(path, {
            "schema_version": 1, "contract_sha256": contract,
            "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
            "invoked": needed, "response": response,
        })
        print(
            f"[DeepSolo] {number}/{EXPECTED_IMAGES} sample_id={sample.sample_id} "
            f"invoked={needed} strings={0 if response is None else len(response['recognized_strings'])}",
            flush=True,
        )


def _aggregate_final(
    *, branch: str, rows: Sequence[dict[str, Any]], gt_by_id: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    image_membership = 0
    multiset_matches = 0
    prediction_count = 0
    reference_count = 0
    fallback_count = 0
    per_image = []
    for row in rows:
        route = row[branch]["route"]
        gt = gt_by_id[row["sample_id"]]
        deep = row["deepsolo"]
        if route["needs_fallback"]:
            fallback_count += 1
            final = str(deep["selected_output"])
            population = [str(value) for value in deep["recognized_strings"]]
        else:
            final = str(route["pre_fallback_output"])
            population = [final] if final else []
        references = [str(value["text"]) for value in gt]
        image_membership += int(final.casefold() in {value.casefold() for value in references})
        per_image_multiset = multiset_exact(population, references)
        multiset_matches += int(per_image_multiset["matches"])
        prediction_count += int(per_image_multiset["predictions"])
        reference_count += int(per_image_multiset["references"])
        per_image.append({
            "sample_id": row["sample_id"], "fallback": bool(route["needs_fallback"]),
            "final_prediction": final, "final_is_gt_member_casefold": final.casefold() in {
                value.casefold() for value in references
            }, "raw_output_population": population,
        })
    precision = multiset_matches / prediction_count if prediction_count else 0.0
    recall = multiset_matches / reference_count if reference_count else 0.0
    multiset = {
        "matches": multiset_matches, "predictions": prediction_count,
        "references": reference_count, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "aggregation": "sum of per-image casefolded one-to-one multiset matches",
    }
    return {
        "images": len(rows), "gt_words": reference_count,
        "fallback_images": fallback_count, "fallback_rate": fallback_count / len(rows),
        "no_fallback_images": len(rows) - fallback_count,
        "singular_final_gt_membership_images": image_membership,
        "singular_final_gt_membership_rate": image_membership / len(rows),
        "one_to_one_casefold_multiset": multiset,
        "per_image": per_image,
    }


def _finalize(
    *, args: argparse.Namespace, samples: Sequence[Any], contract: str, root: Path,
    stage_releases: dict[str, Any], input_hashes: dict[str, str], config: dict[str, Any],
) -> dict[str, Any]:
    # Scientific boundary: recognition GT is opened only after every inference,
    # selection, routing and fallback output has been frozen in a shard.
    recognition_samples, recognition_record = load_icdar13_recognition_samples(
        args.recognition_manifest.resolve(), args.data_root.resolve()
    )
    if len(recognition_samples) != EXPECTED_IMAGES or sum(len(row.words) for row in recognition_samples) != EXPECTED_WORDS:
        raise RuntimeError("ICDAR13 TEST recognition cardinality mismatch")
    gt_by_id = {row.sample_id: [{
        "bbox": list(word.bbox_xyxy), "text": word.transcription,
    } for word in row.words] for row in recognition_samples}
    rows = []
    for sample in samples:
        sample_id = sample.sample_id
        routing = json.loads((root / "04-routing" / f"{int(sample_id):04d}.json").read_text())
        deep = json.loads((root / "05-deepsolo" / f"{int(sample_id):04d}.json").read_text())
        if routing.get("contract_sha256") != contract or deep.get("contract_sha256") != contract:
            raise RuntimeError("finalization encountered a stale shard")
        response = deep["response"]
        rows.append({
            **routing,
            "deepsolo": None if response is None else {
                "selected_output": response["selected_output"],
                "recognized_strings": response["recognized_strings"],
                "detections": response["detections"],
            },
        })
    proposal_gt = [gt_by_id[row.sample_id] for row in recognition_samples]
    faithful_proposals = [row["faithful"]["proposals"] for row in rows]
    balanced_proposals = [row["balanced"]["proposals"] for row in rows]
    faithful_final = _aggregate_final(branch="faithful", rows=rows, gt_by_id=gt_by_id)
    balanced_final = _aggregate_final(branch="balanced", rows=rows, gt_by_id=gt_by_id)
    report = {
        "schema_version": 1,
        "classification": CLASSIFICATION,
        "scope": {"split": "official_test_233", "images": EXPECTED_IMAGES, "gt_words": EXPECTED_WORDS},
        "shared_execution": {
            "same_segmentation_output_per_image": True,
            "same_T1_caption_and_deepsolo_output_when_invoked": True,
            "gt_loaded_after_all_inference": True,
            "stage_releases": stage_releases,
        },
        "branches": {
            "faithful_available_approximate": {
                "localization": "Algorithm1; repository/assumption parameters 150/6/8/top10",
                "crop_diagnostic": word_crop_diagnostic(faithful_proposals, proposal_gt),
                "end_to_end_diagnostics": faithful_final,
            },
            "balanced_anchor_mean": {
                "localization": "rejected TRAIN BALANCED_ANCHOR_MEAN candidate; explicit TEST diagnostic only",
                "crop_diagnostic": word_crop_diagnostic(balanced_proposals, proposal_gt),
                "end_to_end_diagnostics": balanced_final,
            },
        },
        "paper_comparison": {
            "paper_table_5_icdar13_deepsolo_r50_accuracy_percent": PAPER_ICDAR13_R50_ACCURACY_PERCENT,
            "paper_reported_no_fallback": "168/229 (73%); conflicts with official TEST-233 used here",
            "comparable_accuracy": None,
            "percentage_point_gap": None,
            "reason": "UNRESOLVED E-001/DS-002/D-002: paper omits recognition unit, matching, normalization and multi-output reduction; local diagnostics must not be relabeled as Table 5 accuracy.",
        },
        "provenance": {
            **input_hashes, "run_contract_sha256": contract,
            "resolved_comparison_config": config,
            "source_sha256": _sha(Path(__file__).resolve()),
        },
        "scientific_boundary": {
            "table_5_claim": False, "accuracy": None,
            "strict_reproduction": False,
            "faithful_label": "available repository-aligned approximate path, not STRICT",
            "balanced_train_decision": "REJECT",
        },
    }
    _write_atomic(args.output.resolve(), report)
    return report


def main() -> int:
    args = _args()
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("this physical qualification must run on PAPER with --device cuda:0")
    if "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("expected PAPER NVIDIA RTX 3090")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    output, root = args.output.resolve(), args.artifact_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite final report: {output}")
    paths = {
        "segmentation_manifest_sha256": args.segmentation_manifest.resolve(),
        "recognition_manifest_sha256": args.recognition_manifest.resolve(),
        "segmentation_checkpoint_sha256": args.checkpoint.resolve(),
        "v1_calibration_sha256": args.v1_calibration.resolve(),
        "balanced_train_report_sha256": args.balanced_train_report.resolve(),
        "comparison_config_sha256": args.comparison_config.resolve(),
        "segmentation_evaluation_config_sha256": args.evaluation_config.resolve(),
        "parseq_source_archive_sha256": args.parseq_source_archive.resolve(),
        "parseq_checkpoint_sha256": args.parseq_checkpoint.resolve(),
        "deepsolo_runtime_manifest_sha256": args.deepsolo_runtime_manifest.resolve(),
    }
    input_hashes = {name: _sha(path) for name, path in paths.items()}
    required = {
        "segmentation_manifest_sha256": EXPECTED_SEGMENTATION_MANIFEST_SHA256,
        "recognition_manifest_sha256": EXPECTED_RECOGNITION_MANIFEST_SHA256,
        "segmentation_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "v1_calibration_sha256": EXPECTED_V1_CALIBRATION_SHA256,
        "balanced_train_report_sha256": EXPECTED_BALANCED_TRAIN_SHA256,
        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "deepsolo_runtime_manifest_sha256": EXPECTED_DEEPSOLO_RUNTIME_SHA256,
    }
    for name, expected in required.items():
        if input_hashes[name] != expected:
            raise RuntimeError(f"frozen input SHA mismatch: {name}")
    balanced_train = json.loads(args.balanced_train_report.resolve().read_text())
    if balanced_train.get("chosen_variant") != "BALANCED_ANCHOR_MEAN" or balanced_train.get("decision") != "REJECT":
        raise RuntimeError("balanced TRAIN result identity/decision changed")
    config = yaml.safe_load(args.comparison_config.resolve().read_text())
    if config.get("table_5_claim") is not False or config.get("balanced_branch", {}).get("train_decision") != "REJECT":
        raise RuntimeError("comparison config must retain diagnostic/REJECT boundary")
    manifest = load_segmentation_manifest(args.segmentation_manifest.resolve())
    if manifest.split_role != "test" or len(manifest.samples) != EXPECTED_IMAGES:
        raise RuntimeError("requires official TEST-233 image population")
    calibration = json.loads(args.v1_calibration.resolve().read_text())
    params = GroupedWordBlockParams(**calibration["calibration"]["selected"]["parameters"])
    contract = _json_hash({
        "schema": 1, "inputs": input_hashes, "device": args.device,
        "batch_size": args.batch_size, "parseq_revision": PARSEQ_RUNTIME_REVISION,
        "selector": "BALANCED_ANCHOR_MEAN", "classification": CLASSIFICATION,
        "sources": {
            "runner": _sha(Path(__file__).resolve()),
            "selector": _sha(Path(__file__).resolve().parents[1] / "localize/practical_candidate_selection_v2.py"),
            "proposal_generator": _sha(Path(__file__).resolve().parents[1] / "localize/multihypothesis_word_proposals.py"),
        },
    })
    root.mkdir(parents=True, exist_ok=True)
    stage_releases = {}
    stage_releases["segmentation"] = _build_proposal_shards(
        args=args, samples=manifest.samples, params=params, contract=contract, root=root,
    )
    stage_releases["parseq"] = _run_ocr_shards(
        args=args, samples=manifest.samples, contract=contract, root=root,
    )
    stage_releases["blip2"] = _run_caption_shards(
        args=args, samples=manifest.samples, contract=contract, root=root,
    )
    stage_releases["mpnet"] = _score_and_select_shards(
        samples=manifest.samples, contract=contract, root=root,
    )
    _run_deepsolo_shards(args=args, samples=manifest.samples, contract=contract, root=root)
    report = _finalize(
        args=args, samples=manifest.samples, contract=contract, root=root,
        stage_releases=stage_releases, input_hashes=input_hashes, config=config,
    )
    print(json.dumps({
        "faithful": report["branches"]["faithful_available_approximate"],
        "balanced": report["branches"]["balanced_anchor_mean"],
        "paper_comparison": report["paper_comparison"],
    }, indent=2), flush=True)
    print(f"JSON {output}\nSHA256 {_sha(output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
