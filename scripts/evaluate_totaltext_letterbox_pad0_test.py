"""Frozen letterbox + pad-0 PARSeq on official Total-Text TEST-300.

This is a restartable PRACTICAL_APPROXIMATION. TEST GT is read only after
segmentation masks, contour boxes, and crop identities have been frozen.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from scripts.evaluate_totaltext_highres_contour_padding import (
    _annotation_path, _ground_truth, _rgb, _save_mask, contour_proposals,
    crop_array_hash,
)
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import (
    _correct, _geometry_match, _model_and_run, _tensor,
)
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import (
    CONFIG_PATH as TRAIN_CONFIG_PATH, _config as train_config,
)
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_image_frequency_augment import (
    letterbox_canvas_mask, remap_letterbox_native_box,
)


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_letterbox_pad0_test_v1.json"


def _config(config_path: Path = CONFIG_PATH, *, expected_threshold: float = 0.5) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (config.get("reproduction_mode") != "approximate"
        or config.get("classification") != "PRACTICAL_APPROXIMATION"
        or config.get("selected_variant") != "letterbox"
        or config.get("test_images") != 300
        or config.get("test_care_words") != 2204
        or config.get("input_size") != [512, 512]
        or config.get("native_mask_size") != [256, 256]
        or config.get("probability_threshold") != expected_threshold
        or config.get("inference_batch_size") != 1
        or config.get("padding_native_pixels") != 0
        or config.get("area_filter") != "disabled"
        or config.get("maximum_proposals") != "disabled"
        or config.get("matching") != "one_to_one_bbox_iou_ge_0_5"):
        raise ValueError("Frozen letterbox TEST protocol changed")
    if config["letterbox_canvas_fill_rgb"] != train_config()["variants"]["letterbox"]["parameters"]["canvas_fill_rgb"]:
        raise ValueError("Letterbox canvas fill differs from qualified TRAIN configuration")
    return config


def _sample_metrics(predictions: list[dict], gt: list[dict]) -> dict:
    matched = {"geometry_iou_ge_25": _geometry_match(predictions, gt, 0.25),
               "geometry_iou_ge_50": _geometry_match(predictions, gt, 0.5),
               "geometry_iou_ge_75": _geometry_match(predictions, gt, 0.75),
               "exact_casefold_iou_ge_50": _correct(predictions, gt, normalize=False),
               "exact_no_symbols_iou_ge_50": _correct(predictions, gt, normalize=True)}
    return {"gt_words": len(gt), "proposals": len(predictions),
            **{key: len(value) for key, value in matched.items()},
            "matched_gt_indices": {key: sorted(value) for key, value in matched.items()}}


def _validate_cached_sample(shard_path: Path, mask_path: Path, *, protocol_sha: str,
                            sample, annotation_path: Path, checkpoint_sha: str) -> dict:
    value = json.loads(shard_path.read_text(encoding="utf-8"))
    if (value.get("protocol_sha256") != protocol_sha
        or value.get("sample_id") != sample.sample_id
        or value.get("image_sha256") != sample.image_sha256
        or value.get("annotation_sha256") != sha256_file(annotation_path)
        or value.get("checkpoint_sha256") != checkpoint_sha
        or sha256_file(mask_path) != value.get("native_mask_file_sha256")):
        raise ValueError(f"Cached TEST shard provenance mismatch: {shard_path}")
    return value


def _refresh_failed_threshold_protocol(path: Path, protocol: dict) -> None:
    """Atomically refresh only the source hash of an otherwise empty failed run."""
    previous_bytes = path.read_bytes()
    previous = json.loads(previous_bytes)
    ignored = {"evaluator_source_sha256"}
    previous_stable = {k: v for k, v in previous.items() if k not in ignored}
    current_stable = {k: v for k, v in protocol.items() if k not in ignored}
    if ("train_threshold_selection_report_sha256" not in protocol
        or previous_stable != current_stable
        or any(entry != path for entry in path.parent.iterdir())):
        raise ValueError("Existing TEST output belongs to a different protocol")
    encoded = (json.dumps(protocol, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if path.read_bytes() != previous_bytes or any(
            entry not in (path, temporary) for entry in path.parent.iterdir()
        ):
            raise ValueError("Failed TEST protocol changed during atomic refresh")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args, *, config_path: Path = CONFIG_PATH, expected_threshold: float = 0.5) -> dict:
    config = _config(config_path, expected_threshold=expected_threshold)
    training = train_config()
    test_root = args.data_root.resolve()
    train_root = args.target_root.resolve()
    test_manifest_path = args.test_manifest.resolve()
    if sha256_file(test_manifest_path) != config["test_manifest_sha256"]:
        raise ValueError("Official Total-Text TEST manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(test_manifest_path, require_checksums=True)
    if (manifest.dataset != "total_text" or manifest.split_role != "test"
        or len(manifest.samples) != config["test_images"]):
        raise ValueError("Expected qualified official Total-Text TEST-300")
    train_manifest_path = train_root / training["train_manifest"]
    if sha256_file(train_manifest_path) != training["train_manifest_sha256"]:
        raise ValueError("Qualified Total-Text v4 TRAIN manifest SHA256 mismatch")
    train_manifest = load_segmentation_manifest(train_manifest_path, require_checksums=True)
    if train_manifest.dataset != "total_text" or train_manifest.split_role != "train":
        raise ValueError("Frozen letterbox selection must originate from TRAIN")
    _, val_ids, val_sha = _split_indices(train_manifest, training["training"]["seed"], 125)
    if len(val_ids) != 125:
        raise RuntimeError("Frozen TRAIN validation split changed")
    selected_report_path = args.training_runs.resolve() / "validation-pad0-parseq-v1/results.json"
    if sha256_file(selected_report_path) != config["selection_report_sha256"]:
        raise ValueError("TRAIN-only selection report SHA256 mismatch")
    selected_report = json.loads(selected_report_path.read_text(encoding="utf-8"))
    if (selected_report.get("images") != 125
        or selected_report.get("protocol", {}).get("validation_ids_sha256") != val_sha
        or selected_report.get("variants", {}).get("letterbox", {}).get("aggregate", {}).get("gt_words") != 983):
        raise ValueError("TRAIN-only letterbox selection evidence changed")
    threshold_report_sha = config.get("threshold_selection_report_sha256")
    if threshold_report_sha is not None:
        threshold_report_path = getattr(args, "threshold_selection_report", None)
        if threshold_report_path is None or sha256_file(threshold_report_path) != threshold_report_sha:
            raise ValueError("TRAIN-only threshold-selection report is missing or changed")
        threshold_report = json.loads(threshold_report_path.read_text(encoding="utf-8"))
        selected = threshold_report.get("thresholds", {}).get(f"{expected_threshold:.2f}", {})
        if (threshold_report.get("images") != 125 or threshold_report.get("gt_words") != 983
            or selected.get("aggregate", {}).get("exact_casefold_iou_ge_50") is None):
            raise ValueError("TRAIN-only threshold-selection report is incompatible")
    run_path = args.training_runs.resolve() / "letterbox/run_manifest.json"
    trained = json.loads(run_path.read_text(encoding="utf-8"))
    checkpoint = args.training_runs.resolve() / "letterbox/best.pt"
    if (trained.get("status") != "SUCCESS" or not trained.get("training_complete")
        or trained.get("checkpoint", {}).get("sha256") != config["checkpoint_sha256"]
        or trained.get("run_contract", {}).get("variant") != "letterbox"
        or trained.get("run_contract", {}).get("config_sha256") != sha256_file(TRAIN_CONFIG_PATH)
        or trained.get("run_contract", {}).get("validation_ids_sha256") != val_sha
        or sha256_file(checkpoint) != config["checkpoint_sha256"]):
        raise ValueError("Frozen selected letterbox checkpoint/run changed")
    if (sha256_file(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256
        or sha256_file(args.parseq_source_archive.resolve()) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Qualified PARSeq checkpoint/runtime archive SHA256 mismatch")
    if not args.parseq_source.resolve().is_dir():
        raise FileNotFoundError("Qualified PARSeq runtime source directory is missing")
    if args.preflight_only:
        summary = {"status": "PREFLIGHT_PASS", "split": "official_TEST_300",
                   "images": len(manifest.samples), "selected_variant": "letterbox",
                   "checkpoint_sha256": config["checkpoint_sha256"],
                   "test_manifest_sha256": config["test_manifest_sha256"],
                   "selection_report_sha256": config["selection_report_sha256"]}
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on requested device")
    device = torch.device(args.device)
    output = args.output_dir.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("TEST outputs must be outside the repository")
    protocol = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
                "result_class": config["result_class"], "dataset": "total_text",
                "split": "official_TEST_300", "config_sha256": sha256_file(config_path),
                "test_manifest_sha256": config["test_manifest_sha256"],
                "training_run_manifest_sha256": sha256_file(run_path),
                "training_config_sha256": sha256_file(TRAIN_CONFIG_PATH),
                "train_selection_report_sha256": config["selection_report_sha256"],
                "checkpoint_sha256": config["checkpoint_sha256"],
                "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                "segmentation": f"letterbox 512 input; sigmoid>={expected_threshold:g} native256; batch1",
                "localization": "native RETR_EXTERNAL contours; area OFF; top10 OFF; pad0",
                "crop": "inverse letterbox bbox to original RGB; no warping",
                "matching": config["matching"], "ground_truth_use": config["ground_truth_use"],
                "device": args.device, "parseq_batch_size": args.parseq_batch_size}
    if threshold_report_sha is not None:
        protocol["train_threshold_selection_report_sha256"] = threshold_report_sha
        protocol["evaluator_source_sha256"] = sha256_file(Path(__file__))
        protocol["threshold_entrypoint_sha256"] = sha256_file(
            REPO / "scripts/evaluate_totaltext_letterbox_threshold065_test.py")
    protocol_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        previous_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous_protocol != protocol:
            # A failed pre-inference run may have written protocol.json before
            # source-compatibility validation. Permit only that isolated hash
            # refresh; never relabel existing masks, crops, OCR or results.
            if threshold_report_sha is None:
                raise ValueError("Existing TEST output belongs to a different protocol")
            _refresh_failed_threshold_protocol(protocol_path, protocol)
    else:
        _atomic_json(protocol_path, protocol)

    shards = {}
    pending = []
    for sample in manifest.samples:
        key = sample.sample_id.replace(":", "__")
        shard_path = output / "samples" / f"{key}.json"
        mask_path = output / "masks" / f"{key}.png"
        if shard_path.exists():
            shards[sample.sample_id] = _validate_cached_sample(
                shard_path, mask_path, protocol_sha=protocol_sha, sample=sample,
                annotation_path=_annotation_path(sample, test_root),
                checkpoint_sha=config["checkpoint_sha256"])
        else:
            pending.append(sample)
    if pending:
        if device.type == "cuda":
            _prepare_cuda_memory_tracking(device)
        model, transform, report = _model_and_run("letterbox", args, training, val_sha)
        if report["checkpoint"]["sha256"] != config["checkpoint_sha256"]:
            raise ValueError("Loaded letterbox checkpoint differs from selection")
        model.eval().to(device)
        try:
            for position, sample in enumerate(pending, 1):
                rgb, _ = _rgb(sample, test_root)
                with torch.inference_mode():
                    logits = model(_tensor(rgb, transform).to(device))
                if tuple(logits.shape) != (1, 1, 256, 256):
                    raise RuntimeError(f"Unexpected native letterbox logits: {tuple(logits.shape)}")
                native = np.ascontiguousarray((torch.sigmoid(logits[0, 0]) >= expected_threshold).byte().cpu().numpy())
                del logits
                native *= letterbox_canvas_mask((rgb.shape[1], rgb.shape[0])).byte().numpy()
                key = sample.sample_id.replace(":", "__")
                mask_path = output / "masks" / f"{key}.png"
                mask_sha = _save_mask(mask_path, native)
                raw = contour_proposals(native, rgb.shape[:2], [0])["0"]
                for proposal in raw["proposals"]:
                    proposal["bbox_original"] = remap_letterbox_native_box(
                        proposal["bbox_native"], (rgb.shape[1], rgb.shape[0]))
                    proposal["bbox"] = remap_letterbox_native_box(
                        proposal["padded_bbox_native"], (rgb.shape[1], rgb.shape[0]))
                    x1, y1, x2, y2 = proposal["bbox"]
                    proposal["crop_rgb_sha256"] = crop_array_hash(
                        np.ascontiguousarray(rgb[y1:y2, x1:x2]))
                gt, annotation_sha = _ground_truth(sample, test_root, (rgb.shape[1], rgb.shape[0]))
                shard = {"schema_version": 1, "sample_id": sample.sample_id,
                         "protocol_sha256": protocol_sha, "image_sha256": sample.image_sha256,
                         "annotation_sha256": annotation_sha,
                         "checkpoint_sha256": config["checkpoint_sha256"],
                         "native_mask_file_sha256": mask_sha,
                         "native_mask_array_sha256": sha256(native.tobytes()).hexdigest(),
                         "ground_truth": gt, "raw_contours": raw["raw_contours"],
                         "proposals": raw["proposals"]}
                _atomic_json(output / "samples" / f"{key}.json", shard)
                shards[sample.sample_id] = shard
                if position % 10 == 0 or position == len(pending):
                    print(f"[letterbox mask] {position}/{len(pending)} new; {len(shards)}/300 total", flush=True)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        print("[letterbox mask] verified cached 300/300", flush=True)

    by_id = {sample.sample_id: sample for sample in manifest.samples}
    unique = set()
    pending_ocr = []
    for sample in manifest.samples:
        for proposal in shards[sample.sample_id]["proposals"]:
            crop_sha = proposal["crop_rgb_sha256"]
            if crop_sha in unique:
                continue
            unique.add(crop_sha)
            cached_path = output / "ocr" / f"{crop_sha}.json"
            if cached_path.exists():
                cached = json.loads(cached_path.read_text(encoding="utf-8"))
                if (cached.get("crop_rgb_sha256") != crop_sha
                    or cached.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
                    or cached.get("parseq_source_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
                    or not isinstance(cached.get("raw_text"), str)):
                    raise ValueError(f"Cached PARSeq result changed: {cached_path}")
            else:
                pending_ocr.append((sample.sample_id, proposal["bbox"], crop_sha))
    print(f"[letterbox PARSeq] unique_crops={len(unique)} pending={len(pending_ocr)}", flush=True)
    if pending_ocr:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=args.parseq_source, checkpoint=args.parseq_checkpoint,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive, device=args.device))
        try:
            for start in range(0, len(pending_ocr), args.parseq_batch_size):
                batch = pending_ocr[start:start + args.parseq_batch_size]
                rgb_by_sample = {}
                crops = []
                for sample_id, bbox, expected_hash in batch:
                    if sample_id not in rgb_by_sample:
                        rgb_by_sample[sample_id], _ = _rgb(by_id[sample_id], test_root)
                    x1, y1, x2, y2 = bbox
                    crop = np.ascontiguousarray(rgb_by_sample[sample_id][y1:y2, x1:x2])
                    if crop_array_hash(crop) != expected_hash:
                        raise ValueError(f"TEST crop changed after proposal freeze: {sample_id}")
                    crops.append(crop)
                predictions = recognizer.recognize_candidate_batch(crops)
                if len(predictions) != len(batch):
                    raise RuntimeError("PARSeq returned the wrong batch length")
                for (_, _, crop_sha), prediction in zip(batch, predictions):
                    _atomic_json(output / "ocr" / f"{crop_sha}.json", {
                        "schema_version": 1, "crop_rgb_sha256": crop_sha,
                        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                        **prediction.to_dict()})
                print(f"[letterbox PARSeq] {min(start + len(batch), len(pending_ocr))}/{len(pending_ocr)}", flush=True)
        finally:
            recognizer.release()

    keys = ("gt_words", "proposals", "geometry_iou_ge_25", "geometry_iou_ge_50",
            "geometry_iou_ge_75", "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")
    totals = {key: 0 for key in keys}
    rows = []
    for sample in manifest.samples:
        shard = shards[sample.sample_id]
        predictions = []
        for proposal in shard["proposals"]:
            ocr = json.loads((output / "ocr" / f"{proposal['crop_rgb_sha256']}.json").read_text(encoding="utf-8"))
            predictions.append({"rank": proposal["rank"], "bbox": proposal["bbox"],
                                "bbox_native": proposal["bbox_native"],
                                "text": ocr["raw_text"],
                                "crop_rgb_sha256": proposal["crop_rgb_sha256"]})
        metrics = _sample_metrics(predictions, shard["ground_truth"])
        for key in keys:
            totals[key] += metrics[key]
        rows.append({"sample_id": sample.sample_id, "ground_truth": shard["ground_truth"],
                     "predictions": predictions, "raw_contours": shard["raw_contours"],
                     "metrics": metrics})
    if totals["gt_words"] != config["test_care_words"]:
        raise RuntimeError(f"Official Total-Text TEST denominator changed: {totals['gt_words']}")
    for key in ("geometry_iou_ge_50", "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50"):
        totals[key + "_rate_all_gt"] = totals[key] / totals["gt_words"]
    totals["proposal_precision_iou_ge_50"] = (totals["geometry_iou_ge_50"] / totals["proposals"]
                                                 if totals["proposals"] else 0.0)
    result = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "result_class": config["result_class"], "protocol": protocol,
              "images": len(rows), "gt_words": totals["gt_words"],
              "checkpoint": trained["checkpoint"], "aggregate": totals,
              "per_image": rows}
    result_path = output / "results.json"
    if result_path.exists():
        if json.loads(result_path.read_text(encoding="utf-8")) != result:
            raise ValueError("Existing TEST result differs; refusing overwrite")
    else:
        _atomic_json(result_path, result)
    print(json.dumps({"result": str(result_path), "sha256": sha256_file(result_path),
                      "summary": totals}, indent=2, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--training-runs", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--parseq-batch-size", type=int, default=32)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.parseq_batch_size < 1:
        parser.error("--parseq-batch-size must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
