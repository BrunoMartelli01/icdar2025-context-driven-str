"""Restartable TRAIN-holdout pad-0 PARSeq check for the 100-epoch word-kernel model.

The 0.65 threshold comes from the pre-existing TRAIN-only letterbox diagnostic;
it is not selected here or from TEST. This is PRACTICAL_APPROXIMATION only.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from models.totaltext_word_kernel import PracticalTotalTextWordKernelAGUNet
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from scripts.evaluate_totaltext_highres_contour_padding import (
    _save_mask, crop_array_hash,
)
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import (
    _correct, _geometry_match, _image, _tensor, _train_gt,
)
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_word_kernel_100ep import (
    CONFIG_PATH, _config, _contract, native_instances, training_config,
)
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from segmentation.totaltext_word_kernel import instance_proposals
from scripts.train_totaltext_v4_image_frequency_ablation import (
    _config as source_config, _transform,
)
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_image_frequency_augment import (
    letterbox_canvas_mask, remap_letterbox_native_box,
)
from segmentation.training import load_checkpoint


REPO = Path(__file__).resolve().parents[1]


def native_prediction(logits: torch.Tensor, canvas: np.ndarray) -> np.ndarray:
    """Native instance IDs, not a binary union; GT never enters this function."""
    if tuple(logits.shape) != (1, 2, 256, 256):
        raise ValueError("Expected two native256 logit channels")
    return native_instances(logits[0], canvas, _config())[0]


def _save_labels(path: Path, labels: np.ndarray) -> str:
    from PIL import Image
    if labels.shape != (256, 256) or labels.min() < 0 or labels.max() > 65535:
        raise ValueError("Instance map cannot be encoded losslessly as uint16")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with Image.open(path) as opened:
            if not np.array_equal(np.asarray(opened), labels):
                raise ValueError(f"Cached instance map differs: {path}")
    else:
        temporary = path.with_suffix(".png.tmp")
        Image.fromarray(labels.astype(np.uint16)).save(temporary, format="PNG")
        temporary.replace(path)
    return sha256_file(path)


def _ocr_text(record: dict, crop_sha: str) -> str:
    if (record.get("crop_rgb_sha256") != crop_sha
        or record.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
        or record.get("parseq_source_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or not isinstance(record.get("raw_text"), str)):
        raise ValueError(f"PARSeq OCR cache mismatch: {crop_sha}")
    return record["raw_text"]


def run(args) -> dict:
    config = _config()
    source = source_config()
    root = args.target_root.resolve()
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["train_manifest_sha256"]:
        raise ValueError("Qualified TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("OCR diagnostic accepts TRAIN only")
    _, validation_ids, validation_sha = _split_indices(manifest, 202503, 125)
    run_dir = args.training_run.resolve()
    run_path = run_dir / "run_manifest.json"
    trained = json.loads(run_path.read_text(encoding="utf-8"))
    model = PracticalTotalTextWordKernelAGUNet(skip_floor=0.25,
                                                       encoder_pretrained=False)
    contract = _contract(config, source, validation_sha, model)
    contract["target_qualification_sha256"] = sha256_file(run_dir.parent / "target-qualification.json")
    checkpoint = run_dir / "best.pt"
    if (trained.get("status") != "SUCCESS" or trained.get("run_contract") != contract
        or trained.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint)
        or trained.get("validation_images") != 125):
        raise ValueError("Frozen word-kernel checkpoint provenance mismatch")
    reference = json.loads(args.baseline_report.resolve().read_text(encoding="utf-8"))
    baseline = reference.get("thresholds", {}).get("0.65", {}).get("aggregate", {})
    if (reference.get("images") != 125 or reference.get("gt_words") != 983
        or reference.get("protocol", {}).get("validation_ids_sha256") != validation_sha
        or baseline.get("gt_words") != 983
        or baseline.get("exact_casefold_iou_ge_50") != 437):
        raise ValueError("Frozen letterbox 0.65 TRAIN reference changed")
    if (sha256_file(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256
        or sha256_file(args.parseq_source_archive.resolve()) != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or not args.parseq_source.resolve().is_dir()):
        raise ValueError("Qualified PARSeq artifact changed")
    if args.preflight_only:
        result = {"status": "PREFLIGHT_PASS", "images": 125, "gt_words": 983,
                  "checkpoint_epoch": trained["checkpoint"]["epoch"],
                  "checkpoint_sha256": trained["checkpoint"]["sha256"],
                  "baseline_exact_casefold": 437}
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return result
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on requested device")
    device = torch.device(args.device)
    output = args.output_dir.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("OCR output must be outside repository")
    protocol = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
                "split": "TRAIN_heldout_125", "images": 125, "gt_words": 983,
                "config_sha256": sha256_file(CONFIG_PATH),
                "evaluator_source_sha256": sha256_file(Path(__file__)),
                "run_manifest_sha256": sha256_file(run_path),
                "checkpoint_sha256": trained["checkpoint"]["sha256"],
                "baseline_report_sha256": sha256_file(args.baseline_report.resolve()),
                "validation_ids_sha256": validation_sha,
                "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                "segmentation": "letterbox512; native256 full>=0.65; kernel>=0.5; batch1",
                "localization": config["postprocessing"],
                "crop": "inverse letterbox bbox to original RGB",
                "matching": "one-to-one bbox IoU>=0.5 plus exact casefold/Unicode P-S-stripped",
                "ground_truth_use": "post-hoc scoring only",
                "execution_classification": "HARDWARE_DEVIATION: sequential models and configurable PARSeq batch",
                "device": args.device, "parseq_batch_size": args.parseq_batch_size}
    protocol_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing output protocol differs")
    else:
        _atomic_json(protocol_path, protocol)
    if (output / "results.json").exists():
        result = json.loads((output / "results.json").read_text(encoding="utf-8"))
        if result.get("protocol") != protocol:
            raise ValueError("Completed OCR result protocol differs")
        print(f"RESULT {output / 'results.json'} SHA256 {sha256_file(output / 'results.json')}",
              flush=True)
        return result
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    shards = {}
    pending_masks = []
    for index in validation_ids:
        sample = manifest.samples[index]
        key = sample.sample_id.replace(":", "__")
        shard_path = output / "samples" / f"{key}.json"
        mask_path = output / "masks" / f"{key}.png"
        annotation_path = root / "datasets/total_text/train/annotations" / (
            sample.sample_id.split(":", 1)[1] + ".txt")
        if shard_path.exists():
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            if (shard.get("sample_id") != sample.sample_id
                or shard.get("protocol_sha256") != protocol_sha
                or shard.get("image_sha256") != sample.image_sha256
                or shard.get("annotation_sha256") != sha256_file(annotation_path)
                or shard.get("checkpoint_sha256") != trained["checkpoint"]["sha256"]
                or sha256_file(mask_path) != shard.get("native_mask_file_sha256")
                or sha256_file(output / "kernels" / f"{key}.png") != shard.get("kernel_file_sha256")):
                raise ValueError(f"Cached TRAIN shard changed: {shard_path}")
            shards[sample.sample_id] = shard
        else:
            pending_masks.append(sample)
    if pending_masks:
        if device.type == "cuda":
            _prepare_cuda_memory_tracking(device)
        tc = training_config(source)
        load_checkpoint(checkpoint, model=model,
            expected_architecture=model.architecture_identity,
            expected_run_contract=contract, expected_training_config=tc)
        model.to(device).eval()
        transform = _transform(source, "letterbox", train=False)
        try:
            for position, sample in enumerate(pending_masks, 1):
                rgb = _image(sample, root)
                with torch.inference_mode():
                    model_logits = model(_tensor(rgb, transform).to(device))
                canvas = letterbox_canvas_mask((rgb.shape[1], rgb.shape[0])).numpy()
                native, expansion_audit, probability = native_instances(
                    model_logits[0], canvas, config)
                key = sample.sample_id.replace(":", "__")
                mask_path = output / "masks" / f"{key}.png"
                mask_sha = _save_labels(mask_path, native)
                kernel_path = output / "kernels" / f"{key}.png"
                kernel_sha = _save_mask(kernel_path,
                    ((probability[1] >= config["kernel_threshold"]) & canvas).astype(np.uint8))
                proposals = instance_proposals(native)
                for proposal in proposals:
                    proposal["padded_bbox_native"] = list(proposal["bbox_native"])
                raw = {"raw_contours": len(proposals), "proposals": proposals}
                for proposal in raw["proposals"]:
                    proposal["bbox_original"] = remap_letterbox_native_box(
                        proposal["bbox_native"], (rgb.shape[1], rgb.shape[0]))
                    proposal["bbox"] = remap_letterbox_native_box(
                        proposal["padded_bbox_native"], (rgb.shape[1], rgb.shape[0]))
                    x1, y1, x2, y2 = proposal["bbox"]
                    proposal["crop_rgb_sha256"] = crop_array_hash(
                        np.ascontiguousarray(rgb[y1:y2, x1:x2]))
                gt, annotation_sha = _train_gt(sample, root, (rgb.shape[1], rgb.shape[0]))
                shard = {"schema_version": 1, "sample_id": sample.sample_id,
                         "protocol_sha256": protocol_sha,
                         "image_sha256": sample.image_sha256,
                         "annotation_sha256": annotation_sha,
                         "checkpoint_sha256": trained["checkpoint"]["sha256"],
                         "native_mask_file_sha256": mask_sha,
                         "native_mask_array_sha256": sha256(native.astype("<i4").tobytes()).hexdigest(),
                         "kernel_file_sha256": kernel_sha,
                         "expansion_audit": expansion_audit,
                         "ground_truth": gt, "instance_count": len(raw["proposals"]),
                         "proposals": raw["proposals"]}
                _atomic_json(output / "samples" / f"{key}.json", shard)
                shards[sample.sample_id] = shard
                del model_logits
                if position % 10 == 0 or position == len(pending_masks):
                    print(f"[mask] {position}/{len(pending_masks)}", flush=True)
        finally:
            model = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    del model
    gc.collect()
    if len(shards) != 125:
        raise RuntimeError("Missing TRAIN validation shards")
    crop_text = {}
    pending_ocr = []
    seen = set()
    for index in validation_ids:
        sample_id = manifest.samples[index].sample_id
        for proposal in shards[sample_id]["proposals"]:
            crop_sha = proposal["crop_rgb_sha256"]
            if crop_sha in seen:
                continue
            seen.add(crop_sha)
            cache_paths = [output / "ocr" / f"{crop_sha}.json"]
            cache_paths += [directory / f"{crop_sha}.json" for directory in args.reuse_ocr]
            found = next((path for path in cache_paths if path.exists()), None)
            if found is None:
                pending_ocr.append((sample_id, proposal["bbox"], crop_sha))
            else:
                crop_text[crop_sha] = _ocr_text(json.loads(found.read_text()), crop_sha)
    print(f"[PARSeq] reused={len(crop_text)} pending={len(pending_ocr)}", flush=True)
    if pending_ocr:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=args.parseq_source, checkpoint=args.parseq_checkpoint,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive, device=args.device))
        try:
            for start in range(0, len(pending_ocr), args.parseq_batch_size):
                batch = pending_ocr[start:start + args.parseq_batch_size]
                images = {}
                crops = []
                for sample_id, bbox, crop_sha in batch:
                    if sample_id not in images:
                        images[sample_id] = _image(by_id[sample_id], root)
                    x1, y1, x2, y2 = bbox
                    crop = np.ascontiguousarray(images[sample_id][y1:y2, x1:x2])
                    if crop_array_hash(crop) != crop_sha:
                        raise ValueError(f"Cached RGB crop changed: {sample_id}")
                    crops.append(crop)
                outputs = recognizer.recognize_candidate_batch(crops)
                if len(outputs) != len(batch):
                    raise RuntimeError("PARSeq batch output length changed")
                for (_, _, crop_sha), value in zip(batch, outputs):
                    record = {"schema_version": 1, "crop_rgb_sha256": crop_sha,
                              "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                              **value.to_dict()}
                    crop_text[crop_sha] = _ocr_text(record, crop_sha)
                    _atomic_json(output / "ocr" / f"{crop_sha}.json", record)
                print(f"[PARSeq] {min(start + len(batch), len(pending_ocr))}/{len(pending_ocr)}",
                      flush=True)
        finally:
            recognizer.release()
    totals = {key: 0 for key in ("gt_words", "proposals", "geometry_iou_ge_25",
              "geometry_iou_ge_50", "geometry_iou_ge_75",
              "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")}
    rows = []
    for index in validation_ids:
        sample_id = manifest.samples[index].sample_id
        shard = shards[sample_id]
        gt = shard["ground_truth"]
        predictions = [{"bbox": proposal["bbox"],
                        "text": crop_text[proposal["crop_rgb_sha256"]],
                        "crop_rgb_sha256": proposal["crop_rgb_sha256"]}
                       for proposal in shard["proposals"]]
        matched = {"geometry_iou_ge_25": _geometry_match(predictions, gt, 0.25),
                   "geometry_iou_ge_50": _geometry_match(predictions, gt, 0.5),
                   "geometry_iou_ge_75": _geometry_match(predictions, gt, 0.75),
                   "exact_casefold_iou_ge_50": _correct(predictions, gt, normalize=False),
                   "exact_no_symbols_iou_ge_50": _correct(predictions, gt, normalize=True)}
        totals["gt_words"] += len(gt)
        totals["proposals"] += len(predictions)
        for key, ids in matched.items():
            totals[key] += len(ids)
        rows.append({"sample_id": sample_id, "ground_truth": gt,
                     "predictions": predictions,
                     "matched_gt_indices": {key: sorted(ids) for key, ids in matched.items()}})
    if totals["gt_words"] != 983:
        raise RuntimeError("TRAIN GT denominator changed")
    totals["exact_casefold_rate_all_gt"] = totals["exact_casefold_iou_ge_50"] / 983
    totals["exact_no_symbols_rate_all_gt"] = totals["exact_no_symbols_iou_ge_50"] / 983
    tp, proposals = totals["geometry_iou_ge_50"], totals["proposals"]
    totals["proposal_precision_iou_ge_50"] = tp / proposals if proposals else 0.0
    totals["proposal_recall_iou_ge_50"] = tp / 983
    totals["proposal_f1_iou_ge_50"] = 2 * tp / (proposals + 983)
    baseline_rows = {r["sample_id"]: r for r in reference["thresholds"]["0.65"]["per_image"]}
    changes = {}
    for metric in ("exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50", "geometry_iou_ge_50"):
        gained = lost = 0
        for row in rows:
            before = set(baseline_rows[row["sample_id"]]["matched_gt_indices"][metric])
            after = set(row["matched_gt_indices"][metric])
            gained += len(after - before)
            lost += len(before - after)
        changes[metric] = {"gained": gained, "lost": lost, "net": gained - lost}
    result = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "protocol": protocol, "aggregate": totals,
              "frozen_letterbox_065": baseline, "changes_vs_letterbox_065": changes,
              "per_image": rows,
              "runtime": {"device": str(device), "segmentation_batch_size": 1,
                          "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                          "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                          "segmentation_dtype": "float32",
                          "segmentation_input_shape": [1, 3, 512, 512],
                          "parseq_batch_size": args.parseq_batch_size,
                          "model_lifecycle": "sequential_release_before_parseq",
                          "gpu_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                                       if device.type == "cuda" else None),
                          "gpu_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device)
                                                      if device.type == "cuda" else None)}}
    _atomic_json(output / "results.json", result)
    print(json.dumps({"result": str(output / "results.json"),
                      "sha256": sha256_file(output / "results.json"),
                      "word_kernel": totals,
                      "frozen_letterbox_065": baseline,
                      "changes_vs_letterbox_065": changes}, indent=2, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--training-run", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--reuse-ocr", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--parseq-batch-size", type=int, default=32)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.parseq_batch_size < 1:
        parser.error("--parseq-batch-size must be positive")
    try:
        run(args)
    except Exception as error:
        output = args.output_dir.resolve()
        if output != REPO and REPO not in output.parents:
            from scripts.train_totaltext_v4_highres_ablation import _json
            _json(output / "failure.json", {"status": "FAILED", "device": args.device,
                "classification": "PRACTICAL_APPROXIMATION",
                "type": type(error).__name__, "message": str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
