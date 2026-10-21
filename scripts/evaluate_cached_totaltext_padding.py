"""CPU-only, provenance-checked padding diagnostic on frozen Total-Text masks.

Approximate, exploratory TEST analysis. No segmentation forward pass, training,
GT-conditioned proposals, or changes to the running experiment are performed.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import platform
from pathlib import Path
import random
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from localize.native_repository_contours import remap_native_box
from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from recognition.scene_text_v1_evaluation import box_iou, evaluate_images, maximum_matching
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json, _gt, _sample_key
from segmentation.data import file_sha256, load_segmentation_manifest


def crop_hash(rgb: np.ndarray) -> str:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
        raise ValueError("expected nonempty RGB uint8 crop")
    digest = sha256(np.asarray(rgb.shape, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _seal(path: Path, value: dict) -> None:
    if path.exists():
        if _json(path) != value:
            raise ValueError(f"existing artifact belongs to another protocol: {path}")
    else:
        _atomic_json(path, value)


def _artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing/unsafe cache artifact: {relative}")
    return path


def _checked_artifact(root: Path, record: dict) -> Path:
    path = _artifact(root, record["file"])
    if file_sha256(path) != record["sha256"]:
        raise ValueError(f"artifact checksum mismatch: {path}")
    return path


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        return np.ascontiguousarray(opened.convert("RGB"))


def proposals_for_padding(mask: np.ndarray, image_hw: tuple[int, int], plan: dict) -> dict:
    """Keep the source filter/cap/sort; padding can change top10 identities."""
    if mask.shape != (128, 128):
        raise ValueError("the source protocol requires native128 masks")
    variants = {}
    for pad in plan["padding_native_pixels"]:
        audit = audit_repository_contours(mask, RepositoryContourParams(
            min_area=plan["min_unpadded_bbox_area_native"], pad=pad,
            nmax=plan["max_proposals_per_image"],
        ))
        variants[str(pad)] = {
            "raw_contours": audit["raw_external_contours"],
            "candidates_before_cap": audit["candidates_before_cap"],
            "rejected": audit["rejected"],
            "proposals": [{
                "rank": item["rank"], "contour_index": item["contour_index"],
                "bbox_area_native": item["bbox_area"],
                "bbox_native": item["bbox"], "padded_bbox_native": item["padded_bbox"],
                "bbox": remap_native_box(item["bbox"], image_hw),
                "padded_bbox_original": remap_native_box(item["padded_bbox"], image_hw),
            } for item in audit["retained"]],
        }
    if len({len(value["proposals"]) for value in variants.values()}) != 1:
        raise RuntimeError("padding changed proposal count under the frozen filter/cap")
    return variants


def assert_baseline_proposals(actual: list[dict], saved: list[dict]) -> None:
    keys = ("rank", "contour_index", "bbox_area_native", "bbox_native",
            "padded_bbox_native", "bbox", "padded_bbox_original")
    if [{key: row[key] for key in keys} for row in actual] != [
        {key: row[key] for key in keys} for row in saved
    ]:
        raise ValueError("padding6 proposals do not reproduce the frozen source cache")


def evaluation_rows(shards: list[dict], gt: dict, ocr: dict, pad: int,
                    *, padded: bool) -> list[dict]:
    bbox_key = "padded_bbox_original" if padded else "bbox"
    return [{
        "sample_id": shard["sample_id"], "ground_truth": gt[shard["sample_id"]],
        "predictions": [{
            "rank": item["rank"], "contour_index": item["contour_index"],
            "bbox": item[bbox_key], "text": ocr[item["crop_array_sha256"]]["raw_text"],
            "confidence": ocr[item["crop_array_sha256"]]["sequence_confidence_native"],
        } for item in shard["variants"][str(pad)]["proposals"]],
    } for shard in shards]


def correct_word_ids(rows: list[dict]) -> set[str]:
    output = set()
    for row in rows:
        pred, gt = row["predictions"], row["ground_truth"]
        edges = [(p, g) for p, item in enumerate(pred) for g, word in enumerate(gt)
                 if box_iou(item["bbox"], word["bbox"]) >= 0.5
                 and item["text"].casefold() == word["text"].casefold()]
        output.update(f"{row['sample_id']}:{g}" for g in maximum_matching(len(pred), len(gt), edges))
    return output


def _plan(path: Path) -> dict:
    plan = _json(path)
    if (plan.get("variant") != "totaltext-separated-v1-cached-padding-test-v1"
            or plan.get("reproduction_mode") != "approximate"):
        raise ValueError("explicit APPROXIMATE cached-padding config required")
    pads = plan["padding_native_pixels"]
    if (not pads or any(type(p) is not int or p < 0 for p in pads)
            or pads != sorted(set(pads)) or plan["baseline_padding_native_pixels"] not in pads):
        raise ValueError("padding grid must be sorted unique nonnegative integers containing baseline")
    runtime = plan["runtime"]
    if runtime["device"] != "cpu" or runtime["precision"] != "float32":
        raise ValueError("this concurrent diagnostic requires CPU FP32")
    for key in ("batch_size", "torch_threads", "torch_interop_threads"):
        if type(runtime[key]) is not int or runtime[key] < 1:
            raise ValueError(f"runtime.{key} must be a positive integer")
    return plan


def _prepare_sample(sample, source: Path, data_root: Path, output: Path,
                    plan: dict, protocol_sha: str, source_protocol_sha: str) -> dict:
    key = _sample_key(sample.sample_id)
    source_shard_path = _artifact(source, f"samples/{key}.json")
    cached = _json(source_shard_path)
    if cached["protocol_sha256"] != source_protocol_sha or cached["sample_id"] != sample.sample_id:
        raise ValueError("source shard identity/protocol mismatch")
    image_path = data_root / sample.image_path
    if file_sha256(image_path) != sample.image_sha256 or cached["image_sha256"] != sample.image_sha256:
        raise ValueError(f"image checksum mismatch: {sample.sample_id}")
    image = _rgb(image_path)
    mask_path = _checked_artifact(source, cached["native_mask"])
    with Image.open(mask_path) as opened:
        raw_mask = np.asarray(opened)
    if not np.isin(raw_mask, (0, 255)).all():
        raise ValueError("source mask must be binary PNG")
    mask = np.ascontiguousarray((raw_mask > 0).astype(np.uint8))
    if sha256(mask.tobytes()).hexdigest() != cached["native_mask_array_sha256"]:
        raise ValueError("source native mask array checksum mismatch")
    variants = proposals_for_padding(mask, image.shape[:2], plan)
    baseline = variants[str(plan["baseline_padding_native_pixels"])]["proposals"]
    assert_baseline_proposals(baseline, cached["proposals"])
    if any(value["raw_contours"] != cached["raw_contours"] or
           value["candidates_before_cap"] != cached["candidates_before_cap"]
           for value in variants.values()):
        raise ValueError("source contour populations changed")
    for variant in variants.values():
        for item in variant["proposals"]:
            x1, y1, x2, y2 = item["padded_bbox_original"]
            crop = np.ascontiguousarray(image[y1:y2, x1:x2])
            digest = crop_hash(crop)
            relative = f"crops/{digest}.png"
            path = output / relative
            if path.exists():
                if crop_hash(_rgb(path)) != digest:
                    raise ValueError(f"corrupt crop cache: {path}")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop).save(path, compress_level=1)
            item.update(crop_array_sha256=digest, file=relative, sha256=file_sha256(path))
    for actual, old in zip(baseline, cached["proposals"]):
        if crop_hash(_rgb(_checked_artifact(source, old))) != actual["crop_array_sha256"]:
            raise ValueError("baseline6 RGB pixels differ from source crops")
    shard = {
        "schema_version": 1, "protocol_sha256": protocol_sha,
        "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
        "image_hw": list(image.shape[:2]),
        "source_shard_sha256": file_sha256(source_shard_path),
        "native_mask": {"path": str(mask_path), "sha256": file_sha256(mask_path)},
        "native_mask_array_sha256": cached["native_mask_array_sha256"],
        "variants": variants,
    }
    _seal(output / f"proposals/{key}.json", shard)
    return shard


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    plan = _plan(args.config.resolve())
    source, output = args.source_run.resolve(), args.output_dir.resolve()
    data_root = args.data_root.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("source and output trees must be disjoint")
    source_path = source / "results.json"
    if file_sha256(source_path) != plan["source_results_sha256"]:
        raise ValueError("source results SHA256 does not match this separated-v1 experiment")
    report, source_protocol = _json(source_path), _json(source / "protocol.json")
    if report["protocol"] != source_protocol:
        raise ValueError("source report and protocol disagree")
    loc = source_protocol["localization"]
    if (source_protocol["segmentation_checkpoint_sha256"] != plan["source_checkpoint_sha256"]
            or loc["min_bbox_area_native"] != plan["min_unpadded_bbox_area_native"]
            or loc["padding_native"] != plan["baseline_padding_native_pixels"]
            or loc["max_proposals_per_image"] != plan["max_proposals_per_image"]):
        raise ValueError("source checkpoint/localization does not match the explicit plan")
    manifest_path = args.test_manifest.resolve()
    if file_sha256(manifest_path) != source_protocol["test_manifest_sha256"]:
        raise ValueError("TEST manifest differs from frozen source")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if (manifest.dataset != "total_text" or manifest.split_role != "test"
            or len(manifest.samples) != plan["expected_images"]):
        raise ValueError("qualified complete official Total-Text TEST manifest required")
    samples = list(manifest.samples)
    if args.limit_images is not None:
        if not 1 <= args.limit_images <= len(samples):
            raise ValueError("invalid explicit smoke-test image limit")
        samples = samples[:args.limit_images]
    if file_sha256(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256 or file_sha256(
        args.parseq_source_archive
    ) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError("PARSeq artifact hashes do not match the qualified runtime")
    runtime = plan["runtime"]
    torch.set_num_threads(runtime["torch_threads"])
    torch.set_num_interop_threads(runtime["torch_interop_threads"])
    random.seed(runtime["seed"]); np.random.seed(runtime["seed"]); torch.manual_seed(runtime["seed"])
    torch.use_deterministic_algorithms(True)
    code_paths = [Path(__file__).resolve(), Path("localize/repository_contours.py").resolve(),
                  Path("localize/native_repository_contours.py").resolve(),
                  Path("recognition/parseq.py").resolve(),
                  Path("recognition/scene_text_v1_evaluation.py").resolve(),
                  Path("scripts/evaluate_totaltext_v1_findcontours_parseq.py").resolve(),
                  Path("recognition/scene_text_datasets.py").resolve(),
                  Path("segmentation/data.py").resolve()]
    protocol = {
        "schema_version": 1, "reproduction_mode": "approximate",
        "result_class": plan["result_class"] if args.limit_images is None else "smoke-test; approximate reproduction",
        "plan": plan, "config_sha256": file_sha256(args.config),
        "source_run": str(source), "source_protocol": source_protocol,
        "source_protocol_sha256": file_sha256(source / "protocol.json"),
        "data_root": str(data_root), "test_manifest_sha256": file_sha256(manifest_path),
        "sample_ids": [s.sample_id for s in samples],
        "parseq_source": str(args.parseq_source.resolve()),
        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_decode_ar": True, "parseq_refine_iters": 1,
        "parseq_input_shape": [runtime["batch_size"], 3, 32, 128],
        "deterministic_algorithms": True,
        "software": {"python": platform.python_version(), "pytorch": torch.__version__,
                     "numpy": np.__version__, "opencv": cv2.__version__},
        "hardware": {"device": "cpu", "machine": platform.machine(),
                     "processor": platform.processor(), "peak_cuda_allocated_bytes": None,
                     "peak_cuda_reserved_bytes": None, "cuda_used": False},
        "code_sha256": {str(p): file_sha256(p) for p in code_paths},
        "ground_truth_use": "all padding proposals and OCR frozen before loading annotations; evaluation only",
        "matching": "one-to-one IoU>=0.50; casefold exact; care GT; punctuation retained; empty outputs included",
        "metrics": ["padded_crop_box", "unpadded_detection_box", "image_level_lexical_multiset"],
        "selection": "post-hoc TEST exploration; no independently validated padding selection",
        "table_5_claim": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    _seal(output / "protocol.json", protocol)
    protocol_sha = file_sha256(output / "protocol.json")
    shards = []
    for i, sample in enumerate(samples, 1):
        shards.append(_prepare_sample(sample, source, data_root, output, plan,
                                      protocol_sha, protocol["source_protocol_sha256"]))
        if i % 25 == 0 or i == len(samples):
            print(f"[crops] images={i}/{len(samples)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    crops = {item["crop_array_sha256"]: item for shard in shards
             for variant in shard["variants"].values() for item in variant["proposals"]}
    ocr, pending = {}, []
    for digest, item in crops.items():
        path = output / f"ocr/{digest}.json"
        if path.exists():
            value = _json(path)
            if (value["protocol_sha256"] != protocol_sha or value["crop_array_sha256"] != digest
                    or value["crop_file_sha256"] != item["sha256"]):
                raise ValueError(f"OCR cache provenance mismatch: {path}")
            ocr[digest] = value
        else:
            pending.append((digest, item))
    print(f"[PARSeq CPU] unique_crops={len(crops)} pending={len(pending)}", flush=True)
    if pending:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=args.parseq_source.resolve(), checkpoint=args.parseq_checkpoint.resolve(),
            device="cpu", runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive.resolve(),
        ))
        try:
            for start in range(0, len(pending), runtime["batch_size"]):
                batch = pending[start:start + runtime["batch_size"]]
                images = [_rgb(_checked_artifact(output, item)) for _, item in batch]
                if any(crop_hash(rgb) != digest for rgb, (digest, _) in zip(images, batch)):
                    raise ValueError("OCR input pixel hash mismatch")
                decoded = recognizer.recognize_candidate_batch(images)
                if len(decoded) != len(batch):
                    raise RuntimeError("PARSeq batch cardinality mismatch")
                for (digest, item), value in zip(batch, decoded):
                    record = {"schema_version": 1, "protocol_sha256": protocol_sha,
                              "crop_array_sha256": digest, "crop_file_sha256": item["sha256"],
                              **value.to_dict()}
                    _atomic_json(output / f"ocr/{digest}.json", record)
                    ocr[digest] = record
                if start % (runtime["batch_size"] * 8) == 0 or start + len(batch) == len(pending):
                    print(f"[PARSeq CPU] completed={start+len(batch)}/{len(pending)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        finally:
            recognizer.release()
    # Only now load the original care-word annotations and source OCR texts.
    gt, annotation_records = {}, []
    for sample, shard in zip(samples, shards):
        image_path = data_root / sample.image_path
        annotation = image_path.parent.parent / "annotations" / f"{sample.sample_id.partition(':')[2]}.txt"
        h, w = shard["image_hw"]
        gt[sample.sample_id] = _gt(annotation, (w, h))
        cached = _json(source / f"samples/{_sample_key(sample.sample_id)}.json")
        if cached["ground_truth"] != gt[sample.sample_id]:
            raise ValueError("original GT differs from frozen source evaluation")
        annotation_records.append({"sample_id": sample.sample_id, "path": str(annotation),
                                   "sha256": file_sha256(annotation)})
    baseline = plan["baseline_padding_native_pixels"]
    baseline_ids = {p["sample_id"]: {v["contour_index"] for v in p["variants"][str(baseline)]["proposals"]}
                    for p in shards}
    summaries, outputs, word_ids = {}, {}, {}
    for pad in plan["padding_native_pixels"]:
        summaries[str(pad)], outputs[str(pad)], word_ids[str(pad)] = {}, {}, {}
        for name, padded in (("padded_crop_box", True), ("unpadded_detection_box", False)):
            rows = evaluation_rows(shards, gt, ocr, pad, padded=padded)
            summaries[str(pad)][name] = evaluate_images(rows, text_available=True)
            outputs[str(pad)][name] = rows
            word_ids[str(pad)][name] = correct_word_ids(rows)
        changes = [s["sample_id"] for s in shards if baseline_ids[s["sample_id"]] !=
                   {v["contour_index"] for v in s["variants"][str(pad)]["proposals"]}]
        summaries[str(pad)]["selection_changed_images_vs_padding6"] = changes
    for pad in plan["padding_native_pixels"]:
        for name in ("padded_crop_box", "unpadded_detection_box"):
            ids, base = word_ids[str(pad)][name], word_ids[str(baseline)][name]
            summaries[str(pad)][name]["vs_padding6"] = {
                "gained_correct_gt_ids": sorted(ids - base),
                "lost_correct_gt_ids": sorted(base - ids), "net_correct_words": len(ids)-len(base),
            }
    changed_texts, confidence_diffs = [], []
    source_protocol_sha = protocol["source_protocol_sha256"]
    for shard in shards:
        key = _sample_key(shard["sample_id"])
        cached = _json(source / f"samples/{key}.json")
        for item, old in zip(shard["variants"][str(baseline)]["proposals"], cached["proposals"]):
            saved = _json(source / f"ocr/{key}__{old['rank']:02d}.json")
            if saved["protocol_sha256"] != source_protocol_sha or saved["crop_sha256"] != old["sha256"]:
                raise ValueError("source OCR provenance mismatch")
            cpu = ocr[item["crop_array_sha256"]]
            if cpu["raw_text"] != saved["raw_text"]:
                changed_texts.append({"sample_id": shard["sample_id"], "rank": item["rank"],
                                      "cpu": cpu["raw_text"], "saved_gpu": saved["raw_text"]})
            confidence_diffs.append(abs(cpu["sequence_confidence_native"]-saved["sequence_confidence_native"]))
    result = {
        "schema_version": 1, "reproduction_mode": "approximate", "result_class": protocol["result_class"],
        "protocol_sha256": protocol_sha, "images": len(samples),
        "care_words": sum(len(v) for v in gt.values()), "unique_rgb_crops": len(crops),
        "elapsed_seconds": time.monotonic()-started,
        "cpu_vs_saved_gpu_padding6": {"crop_count": len(confidence_diffs),
            "changed_text_count": len(changed_texts), "changed_texts": changed_texts,
            "max_abs_sequence_confidence_difference": max(confidence_diffs, default=0)},
        "annotation_provenance": annotation_records, "summary": summaries,
        "outputs": outputs,
        "proposal_shards": [{"file": f"proposals/{_sample_key(s['sample_id'])}.json",
                             "sha256": file_sha256(output / f"proposals/{_sample_key(s['sample_id'])}.json")}
                            for s in shards],
        "note": "TEST padding exploration; crop-box IoU gain is distinct from actual OCR gain; not a Table5 result",
    }
    if (output / "results.json").exists():
        old = _json(output / "results.json")
        old.pop("elapsed_seconds", None)
        stable = {k: v for k, v in result.items() if k != "elapsed_seconds"}
        if old != stable:
            raise ValueError("completed result replay differs")
    else:
        _atomic_json(output / "results.json", result)
    print(json.dumps({"images": len(samples), "care_words": result["care_words"],
                      "padding6_cpu_gpu_parity": result["cpu_vs_saved_gpu_padding6"],
                      "results": str(output / "results.json")}, ensure_ascii=False), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source-run", "test-manifest", "data-root", "parseq-source",
                 "parseq-source-archive", "parseq-checkpoint", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--limit-images", type=int, help="explicit separate smoke-test output only")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as error:
        source, output = args.source_run.resolve(), args.output_dir.resolve()
        if source != output and source not in output.parents and output not in source.parents:
            output.mkdir(parents=True, exist_ok=True)
            failure = output / "failure.json"
            if not failure.exists():
                _atomic_json(failure, {"status": "failed", "error_type": type(error).__name__,
                                      "error": str(error), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
