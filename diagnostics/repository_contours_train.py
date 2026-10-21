"""Frozen TRAIN comparison of current Algorithm1, upstream contours and v1.

No selection/tuning: GPU work is run manually on PAPER. Shared native masks
and OCR are persisted as small, atomic JSON shards; crops remain implicit ROIs.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rapidfuzz.fuzz import ratio
import torch

from localize.grouped_word_blocks import group_word_blocks
from localize.mask_to_blocks import BlockParam, localize_mask
from localize.practical_remap_protocol import proposal_geometry, train_task22_word_boxes
from localize.repository_contours import RepositoryContourParams, audit_repository_contours, SOURCE_SHA256
from pipeline.frozen_segmentation import (
    FrozenAuthorProtocolSegmenter, FrozenSegmentationConfig, backproject_binary_mask,
)
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer, verify_parseq_runtime_source,
)
from scripts.diagnose_train_gtbox_vs_v1_parseq import _texts, _wordness, _iou, _inter
from scripts.export_train_v1_parseq_inputs import (
    EXPECTED_CHECKPOINT_SHA256, EXPECTED_TRAIN_MANIFEST_SHA256, EXPECTED_SPLIT,
    _sha, _write_atomic, _contained, _params, _candidate_shard_valid, _ocr_shard_valid,
)
from segmentation.data import load_segmentation_manifest

METHODS = ("CURRENT_ALG1", "REPOSITORY_CONTOURS", "PRACTICAL_REMAP_V1")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def pixel_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pack_mask(mask):
    if mask.shape != (128, 128) or mask.dtype != np.uint8 or not np.isin(mask, (0, 1)).all():
        raise ValueError("expected binary uint8 native mask128")
    return {"bits": base64.b64encode(np.packbits(mask).tobytes()).decode(),
            "array_sha256": pixel_sha(mask)}


def unpack_mask(value):
    packed = base64.b64decode(value["bits"], validate=True)
    if len(packed) != 2048:
        raise ValueError("invalid packed native mask size")
    mask = np.unpackbits(np.frombuffer(packed, dtype=np.uint8)).reshape(128, 128)
    if pixel_sha(mask) != value["array_sha256"]:
        raise ValueError("native mask checksum mismatch")
    return mask


def save_shard(path, contract, value):
    _write_atomic(path, {"contract": contract, "payload_sha256": digest(value), "value": value})


def read_shard(path, contract):
    data = json.loads(path.read_text())
    if data["contract"] != contract or data["payload_sha256"] != digest(data["value"]):
        raise RuntimeError(f"cache contract/checksum mismatch: {path}")
    return data["value"]


def read_sample(path, contract, sample):
    value = read_shard(path, contract)
    if value.get("sample_id") != sample.sample_id or value.get("image_sha256") != sample.image_sha256:
        raise RuntimeError(f"cached sample identity mismatch: {path}")
    return value


def proposals(mask, rgb, contour_params, v1_params):
    current = localize_mask(mask, rgb, BlockParam(
        contour_params.min_area, contour_params.pad, 8, "top_to_bottom_then_left_to_right",
        contour_params.nmax))
    repo = audit_repository_contours(mask, contour_params)
    v1 = group_word_blocks(mask, rgb, v1_params)
    groups = v1.groups_before_cap
    return {
        "CURRENT_ALG1": {"boxes": [list(b.padded_bbox) for b in current.retained],
            "precap": len(current.retained) + sum(b.reason == "top_10_limit" for b in current.rejected),
            "raw_count": len(current.retained) + len(current.rejected), "merges": 0, "multi": 0},
        "REPOSITORY_CONTOURS": {"boxes": [b["padded_bbox"] for b in repo["retained"]],
            "precap": repo["candidates_before_cap"], "raw_count": repo["raw_external_contours"],
            "merges": 0, "multi": 0},
        "PRACTICAL_REMAP_V1": {"boxes": [list(b.padded_bbox) for b in v1.retained],
            "precap": len(groups), "raw_count": len(current.retained) + len(current.rejected),
            "merges": sum(len(g["raw_component_ids"]) - 1 for g in groups),
            "multi": sum(len(g["raw_component_ids"]) > 1 for g in groups)},
    }


def evaluate(records, data_root, samples, *, expected_gt_words=849):
    """GT enters only after all masks/proposals/OCR have been generated."""
    geometry = {m: [] for m in METHODS}
    categories = {m: Counter() for m in METHODS}
    ocr = {m: Counter() for m in METHODS}
    rows = []
    crop_rows = []
    for sample, record in zip(samples, records):
        boxes = train_task22_word_boxes(data_root / sample.mask_path, data_root / sample.validity_annotation_path)
        texts = _texts(data_root / sample.validity_annotation_path)
        if len(boxes) != len(texts):
            raise RuntimeError("GT geometry/text population mismatch")
        words = [{"bbox": b.bbox, "text": t} for b, t in zip(boxes, texts)]
        for method in METHODS:
            group = record["methods"][method]
            candidates = group["boxes"]
            geometry[method].append((candidates, [w["bbox"] for w in words],
                group["precap"], group["merges"], group["multi"]))
            for rank, box in enumerate(candidates, 1):
                category, best, iou = _wordness(box, words)
                categories[method][category] += 1
                crop_rows.append({"sample_id": sample.sample_id, "method": method, "rank": rank,
                    "bbox": box, "text": record["texts"][json.dumps(box)],
                    "category": category, "best_gt_index": best, "best_gt_iou": iou})
        for index, word in enumerate(words):
            row = {"sample_id": sample.sample_id, "gt_index": index, **word, "methods": {}}
            for method in METHODS:
                candidates = record["methods"][method]["boxes"]
                best = max(candidates, key=lambda b: _iou(b, word["bbox"]), default=None)
                available = best is not None and _inter(best, word["bbox"]) > 0
                text = record["texts"][json.dumps(best)] if available else None
                exact = available and text.casefold() == word["text"].casefold()
                fuzzy = ratio(text.casefold(), word["text"].casefold()) / 100 if available else None
                ocr[method].update(gt_words=1, available_predictions=int(available),
                    exact_casefold=int(exact), fuzzy_ge_0_8=int(fuzzy is not None and fuzzy >= .8))
                row["methods"][method] = {"bbox": best, "text": text,
                    "iou": _iou(best, word["bbox"]) if available else 0,
                    "exact": bool(exact), "fuzzy": fuzzy}
            rows.append(row)
    if len(rows) != expected_gt_words:
        raise RuntimeError(f"expected {expected_gt_words} TRAIN words, got {len(rows)}")
    summary = {}
    for method in METHODS:
        summary[method] = {"geometry": proposal_geometry(geometry[method]).as_dict(),
            "ocr": {**ocr[method], "exact_rate": ocr[method]["exact_casefold"] / expected_gt_words,
                    "fuzzy_rate": ocr[method]["fuzzy_ge_0_8"] / expected_gt_words},
            "wordness": {c: categories[method][c] for c in
                         ("WORD_GOOD", "WORD_PARTIAL", "MULTIWORD_MERGE", "UNMATCHED")}}
    return summary, rows, crop_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("train-manifest", "data-root", "checkpoint", "evaluation-config",
                 "repository-config", "v1-config", "compact-v1", "parseq-source",
                 "parseq-source-archive", "parseq-checkpoint", "output-dir"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--check", action="store_true")
    a = p.parse_args()
    if _sha(a.train_manifest) != EXPECTED_TRAIN_MANIFEST_SHA256 or _sha(a.checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("frozen TRAIN manifest/checkpoint mismatch")
    if _sha(a.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256 or _sha(a.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise RuntimeError("frozen PARSeq artifacts mismatch")
    verify_parseq_runtime_source(a.parseq_source, source_type=PARSEQ_ARCHIVE_SOURCE,
                                source_archive=a.parseq_source_archive)
    manifest = load_segmentation_manifest(a.train_manifest)
    if manifest.split != EXPECTED_SPLIT or manifest.split_role != "train" or len(manifest.samples) != 229:
        raise RuntimeError("requires TRAIN-229")
    config = json.loads(a.repository_config.read_text())
    if config["reproduction_mode"] != "approximate" or config["source_sha256"] != SOURCE_SHA256:
        raise RuntimeError("require frozen approximate repository config")
    contour_params = RepositoryContourParams(**config["parameters"])
    if config["parameters"] != {"min_area": 150, "pad": 6, "nmax": 10}:
        raise RuntimeError("no parameter tuning allowed")
    v1_params = _params(a.v1_config)
    dataset_files = {}
    for sample in manifest.samples:
        for path, sha in ((sample.image_path, sample.image_sha256), (sample.mask_path, sample.mask_sha256),
                          (sample.validity_annotation_path, sample.validity_annotation_sha256)):
            observed = _sha(_contained(a.data_root, path))
            if sha is not None and observed != sha:
                raise RuntimeError(f"dataset checksum mismatch: {path}")
            dataset_files[str(path)] = {"sha256": observed, "prior_manifest_hash_available": sha is not None}
    compact_path = a.compact_v1 / "manifest.json"
    compact = json.loads(compact_path.read_text())
    if compact["segmentation"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("compact segmentation mismatch")
    refs = {r["sample_id"]: r for r in compact["samples"]}
    if set(refs) != {s.sample_id for s in manifest.samples}:
        raise RuntimeError("compact population mismatch")
    for sample in manifest.samples:
        ref = refs[sample.sample_id]
        cp, op = (_contained(a.compact_v1, ref[k]) for k in ("candidate_shard", "ocr_shard"))
        if _sha(cp) != ref["candidate_shard_sha256"] or _sha(op) != ref["ocr_shard_sha256"]:
            raise RuntimeError("compact shard hash mismatch")
        cached = _candidate_shard_valid(cp, sample_id=sample.sample_id,
                    image_sha=sample.image_sha256, root=a.compact_v1)
        _ocr_shard_valid(op, candidate_ids=[r["candidate_id"] for r in cached["candidates"]])
    source_files = [Path(__file__), Path("localize/repository_contours.py"),
        Path("localize/mask_to_blocks.py"), Path("localize/grouped_word_blocks.py"),
        Path("pipeline/frozen_segmentation.py"), Path("recognition/parseq.py"),
        Path("scripts/export_train_v1_parseq_inputs.py"), Path("scripts/diagnose_train_gtbox_vs_v1_parseq.py"),
        Path("localize/practical_remap_protocol.py")]
    protocol = {"mode": "APPROXIMATE", "classification": "PRACTICAL_APPROXIMATION",
        "scope": {"split": "TRAIN", "images": 229, "gt_words": 849},
        "files": {str(f.resolve()): _sha(f) for f in source_files + [a.train_manifest, a.checkpoint,
            a.evaluation_config, a.repository_config, a.v1_config, a.parseq_checkpoint, compact_path]},
        "dataset_files": dataset_files,
        "device": a.device, "torch": torch.__version__, "opencv": cv2.__version__,
        "shared_masks": True, "threshold": .5, "methods": METHODS,
        "gt_usage": "post-hoc only; best IoU crop per word, positive intersection for OCR availability; casefold",
        "wordness": "MULTIWORD first if >=2 GT boxes covered >=.25; otherwise GOOD IoU>=.5, PARTIAL one covered GT, UNMATCHED",
        "batch_size": 1, "precision": "float32", "test_loaded": False, "tuning": False}
    contract = digest(protocol)
    out = a.output_dir
    old_protocol = out / "protocol.json"
    if old_protocol.exists() and json.loads(old_protocol.read_text()) != json.loads(json.dumps(protocol)):
        raise RuntimeError("output protocol changed: choose a new output directory")
    print("[check] TRAIN-229 images/targets, checkpoint, configs and PARSeq source PASS", flush=True)
    print("[check] annotation hashes frozen now where absent from original manifest", flush=True)
    print("[check] compact v1 crop/OCR cache hashes PASS", flush=True)
    if a.check:
        print("[check] no inference performed; masks will be shared and cached", flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    _write_atomic(old_protocol, protocol)
    segmenter = None
    try:
        for n, sample in enumerate(manifest.samples, 1):
            path = out / "masks" / f"{sample.sample_id}.json"
            if path.exists():
                unpack_mask(read_sample(path, contract, sample))
                action = "reuse"
            else:
                if segmenter is None:
                    segmenter = FrozenAuthorProtocolSegmenter(FrozenSegmentationConfig(a.evaluation_config, a.checkpoint, a.device))
                with Image.open(a.data_root / sample.image_path) as source:
                    rgb = np.array(source.convert("RGB"))
                seg = segmenter(rgb)
                value = {**pack_mask(seg["mask_native"]), "sample_id": sample.sample_id,
                    "image_sha256": sample.image_sha256, "segmentation_metadata": seg["metadata"],
                    "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(a.device)) if a.device.startswith("cuda") else None,
                    "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(a.device)) if a.device.startswith("cuda") else None}
                save_shard(path, contract, value)
                action = "generated"
            print(f"[shared mask] {n}/229 {sample.sample_id} {action}", flush=True)
    finally:
        if segmenter is not None:
            segmenter.release()
            del segmenter
    records = []
    recognizer = None
    try:
        for n, sample in enumerate(manifest.samples, 1):
            target = out / "samples" / f"{sample.sample_id}.json"
            if target.exists():
                records.append(read_sample(target, contract, sample))
                print(f"[crops/OCR] {n}/229 reuse", flush=True)
                continue
            with Image.open(a.data_root / sample.image_path) as source:
                rgb = np.array(source.convert("RGB"))
            native = unpack_mask(read_sample(out / "masks" / f"{sample.sample_id}.json", contract, sample))
            mask = backproject_binary_mask(native, rgb.shape[:2])
            methods = proposals(mask, rgb, contour_params, v1_params)
            ref = refs[sample.sample_id]
            cp, op = (_contained(a.compact_v1, ref[k]) for k in ("candidate_shard", "ocr_shard"))
            if _sha(cp) != ref["candidate_shard_sha256"] or _sha(op) != ref["ocr_shard_sha256"]:
                raise RuntimeError("compact shard hash mismatch")
            cached = _candidate_shard_valid(cp, sample_id=sample.sample_id,
                       image_sha=sample.image_sha256, root=a.compact_v1)
            cached_ocr = _ocr_shard_valid(op, candidate_ids=[r["candidate_id"] for r in cached["candidates"]])
            reused = {}
            for row, result in zip(cached["candidates"], cached_ocr["results"]):
                with Image.open(_contained(a.compact_v1, row["crop_path"])) as source:
                    reused[(json.dumps(row["bbox"]), pixel_sha(np.array(source.convert("RGB"))))] = result["raw_text"]
            texts, generated, cached_count = {}, 0, 0
            for method in METHODS:
                for box in methods[method]["boxes"]:
                    key = json.dumps(box)
                    if key in texts:
                        continue
                    x1, y1, x2, y2 = box
                    crop = rgb[y1:y2, x1:x2]
                    reuse_key = (key, pixel_sha(crop))
                    if reuse_key in reused:
                        texts[key] = reused[reuse_key]
                        cached_count += 1
                    else:
                        if recognizer is None:
                            recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                                runtime_source=a.parseq_source, checkpoint=a.parseq_checkpoint, device=a.device,
                                runtime_source_type=PARSEQ_ARCHIVE_SOURCE, runtime_source_archive=a.parseq_source_archive))
                        texts[key] = recognizer.recognize(crop, input_role="T3_localized_crop").raw_text
                        generated += 1
            record = {"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "native_sha256": pixel_sha(native),
                "projected_sha256": pixel_sha(mask), "methods": methods, "texts": texts,
                "mask_matches_previous_v1": pixel_sha(native) == cached["native_mask_sha256"],
                "new_ocr": generated, "reused_ocr": cached_count,
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(a.device)) if generated and a.device.startswith("cuda") else None,
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(a.device)) if generated and a.device.startswith("cuda") else None}
            save_shard(target, contract, record)
            records.append(record)
            print(f"[crops/OCR] {n}/229 new={generated} reused={cached_count}", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
    summary, rows, crop_rows = evaluate(records, a.data_root, manifest.samples)
    report = {"protocol": protocol, "contract_sha256": contract, "summary": summary,
        "per_word": rows, "per_crop": crop_rows,
        "masks_matching_historical_v1": sum(r["mask_matches_previous_v1"] for r in records),
        "new_ocr": sum(r["new_ocr"] for r in records), "reused_ocr": sum(r["reused_ocr"] for r in records),
        "table5_accuracy": None, "fallback_executed": False,
        "shards": {str(p.relative_to(out)): _sha(p) for directory in ("masks", "samples") for p in sorted((out / directory).glob("*.json"))}}
    target = out / "results.json"
    if target.exists() and json.loads(target.read_text()) != json.loads(json.dumps(report)):
        raise RuntimeError("refusing to replace a different completed report")
    _write_atomic(target, report)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"JSON {target} SHA256 {_sha(target)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
