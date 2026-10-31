"""TEST500 word F1 for the branch frozen by TRAIN validation100."""

import argparse
import gc
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from segmentation.icdar2015_improvement_runtime import model as build_improvement_model
from recognition.icdar2015_word_f1 import aggregate_counts, sample_counts
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer, verify_parseq_runtime_source,
)
from recognition.scene_text_datasets import parse_icdar2015_annotation
from scripts.train_icdar2015_totaltext_transfer import write_json
from scripts.select_icdar2015_improvement import verify_selection
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.icdar2015_improvement_runtime import load_config as load_training_config, training_config
from segmentation.icdar2015_improvements import binary_proposals, make_transform, valid_canvas
from segmentation.training import load_checkpoint, runtime_environment, seed_everything


REPO = Path(__file__).resolve().parents[1]


def load_config(path):
    if path.resolve().parent.name != "approximate":
        raise ValueError("Explicit configs/approximate evaluation configuration required")
    value = json.loads(path.read_text())
    for key, expected in {"reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "dataset": "icdar2015", "split_role": "test", "expected_images": 500,
            "segmentation_batch_size": 1, "text_normalization": "unicode_casefold_then_isalnum",
            "prediction_ignore_policy": "all_outputs_count_including_ignored_regions"}.items():
        if value.get(key) != expected:
            raise ValueError(f"Incompatible word F1 evaluation configuration: {key}")
    if value["parseq_batch_size"] < 1 or not 0 <= value["probability_threshold"] <= 1:
        raise ValueError("Invalid OCR batch size or probability threshold")
    if (value["connectivity"] != 8 or value["min_component_area"] != 0
            or value["padding_native_pixels"] != 0 or value["maximum_proposals"] is not None):
        raise ValueError("This frozen diagnostic uses all 8-connected components, pad0")
    return value


def crop_hash(rgb):
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or not rgb.size:
        raise ValueError("Crop must contain nonempty uint8 RGB")
    return sha256(np.asarray(rgb.shape, dtype="<i8").tobytes() + np.ascontiguousarray(rgb).tobytes()).hexdigest()


def proposals(native, image_wh, config):
    expected=tuple(d//2 for d in config["image_hw"])
    if native.shape!=expected or native.dtype!=np.bool_:
        raise ValueError("Declared selected native shape required")
    return binary_proposals(native,image_wh,config["image_hw"])[0]


def original_rgb(sample, root):
    path = root / sample.image_path
    if file_sha256(path) != sample.image_sha256:
        raise ValueError(f"RGB checksum mismatch: {sample.sample_id}")
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))


def ground_truth(sample, root, image_wh):
    annotation = Path(sample.image_path).parent.parent / "annotations" / (sample.sample_id.split(":", 1)[1] + ".txt")
    instances = parse_icdar2015_annotation(root / annotation, image_size_wh=image_wh)
    width, height = image_wh
    words = []
    for item in instances:
        if item.ignore:
            continue
        x1, y1, x2, y2 = item.bbox_xyxy
        words.append({"text": item.transcription, "polygon": [list(point) for point in item.polygon_xy],
                      "bbox": [x1, y1, min(width, x2 + 1), min(height, y2 + 1)]})
    return words, sum(item.ignore for item in instances), file_sha256(root / annotation)


def run(args):
    started = time.monotonic()
    config, training = load_config(args.config), load_training_config(args.training_config)
    if file_sha256(args.training_config) != config["training_config_sha256"]:
        raise ValueError("Frozen training configuration SHA256 differs")
    if config["probability_threshold"] != training["probability_threshold"]:
        raise ValueError("TEST threshold must remain frozen from the TRAIN protocol")
    selection=verify_selection(args.selection_report,args.training_run,training)
    trained_path = args.training_run / "run_manifest.json"
    trained = json.loads(trained_path.read_text())
    if (trained.get("status") != "SUCCESS" or not trained.get("training_complete")
            or trained.get("result_class") != "approximate reproduction"
            or trained["completed_epochs"] != training["training"]["epochs"]
            or trained["run_contract"]["config_sha256"] != config["training_config_sha256"]):
        raise ValueError("Only the completed full training run can enter this TEST evaluation")
    config={**config,"image_hw":trained["run_contract"]["image_hw"]}
    for name, expected in trained["run_contract"]["source_sha256"].items():
        if file_sha256(REPO / name) != expected:
            raise ValueError(f"Training scientific source changed: {name}")
    checkpoint = args.training_run / trained["checkpoint"]["path"]
    if file_sha256(checkpoint) != trained["checkpoint"]["sha256"]:
        raise ValueError("Selected U-Net checkpoint checksum mismatch")
    if file_sha256(args.test_manifest) != config["test_manifest_sha256"]:
        raise ValueError("Official ICDAR2015 TEST manifest checksum mismatch")
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if (manifest.dataset != config["dataset"] or manifest.split_role != "test"
            or len(manifest.samples) != config["expected_images"]):
        raise ValueError("Expected complete official ICDAR2015 TEST500")
    if (file_sha256(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256
            or file_sha256(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Qualified PARSeq weights/archive checksum mismatch")
    runtime_source = verify_parseq_runtime_source(args.parseq_source,
        source_type=PARSEQ_ARCHIVE_SOURCE, source_archive=args.parseq_source_archive)
    output, root = args.output_dir.resolve(), args.data_root.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("Scientific outputs must be stored outside Git")
    scientific_sources = ("scripts/evaluate_icdar2015_improvement_test.py",
        "scripts/select_icdar2015_improvement.py", "segmentation/icdar2015_improvements.py",
        "recognition/icdar2015_word_f1.py", "recognition/deepsolo_word_f1.py", "recognition/parseq.py")
    protocol = {"schema_version": 1, "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction", "result_class": "approximate reproduction",
        "evaluation_config": config, "config_sha256": file_sha256(args.config),
        "selection_report_sha256":file_sha256(args.selection_report),"selected_branch":selection["selected_branch"],
        "training_run_manifest_sha256": file_sha256(trained_path),
        "selected_checkpoint": trained["checkpoint"], "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_source": runtime_source,
        "source_sha256": {p: file_sha256(REPO / p) for p in scientific_sources},
        "training_source_sha256": trained["run_contract"]["source_sha256"],
        "paths": {"data_root": str(root), "test_manifest": str(args.test_manifest.resolve()),
                  "training_run": str(args.training_run.resolve()),
                  "selection_report":str(args.selection_report.resolve())}, "device": args.device}
    protocol_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing evaluation protocol differs; preserve it and use a new directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError("Nonempty evaluation directory without a protocol")
        write_json(protocol_path, protocol)
    if args.preflight_only:
        return {"status": "PREFLIGHT_PASS", "images": len(manifest.samples), "protocol_sha256": protocol_sha}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no CPU substitution")
    seed_everything(training["training"]["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    paired = make_transform(training,config["image_hw"],augment=False)
    shards, pending = {}, []
    for sample in manifest.samples:
        path = output / "samples" / (sample.sample_id.replace(":", "_") + ".json")
        if not path.exists():
            pending.append(sample)
            continue
        shard = json.loads(path.read_text())
        if (shard["protocol_sha256"] != protocol_sha or shard["image_sha256"] != sample.image_sha256
                or file_sha256(output / shard["mask_path"]) != shard["mask_sha256"]):
            raise ValueError(f"Cached segmentation provenance differs: {sample.sample_id}")
        shards[sample.sample_id] = shard
    if pending:
        model = build_improvement_model(training,trained["run_contract"]["branch_config"]["normalization"])
        load_checkpoint(checkpoint, model=model, expected_architecture=model.architecture_identity,
            expected_run_contract=trained["run_contract"], expected_training_config=training_config(training,trained["run_contract"]["branch_config"]["loss"]))
        model.to(device).eval()
        with torch.inference_mode():
            for position, sample in enumerate(pending, 1):
                rgb = original_rgb(sample, root)
                source = Image.fromarray(rgb)
                dummy = Image.new("L", source.size, 0)
                image, _, _ = paired.transform_with_validity(source, dummy, Image.new("L", source.size, 255), 0)
                probability = model(image[None].to(device)).sigmoid()[0, 0].cpu().numpy()
                if not np.isfinite(probability).all():
                    raise FloatingPointError("Nonfinite segmentation probability")
                native = (probability >= config["probability_threshold"]) & valid_canvas(source.size,config["image_hw"])
                crops = proposals(native, source.size, config)
                for rank, crop in enumerate(crops):
                    x1, y1, x2, y2 = crop["bbox"]
                    crop.update(rank=rank, crop_rgb_sha256=crop_hash(rgb[y1:y2, x1:x2]))
                key = sample.sample_id.replace(":", "_")
                mask_path = output / "masks" / (key + ".npz")
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                if mask_path.exists():
                    # A crash between mask and shard writes leaves an orphan.
                    # Reuse it only after exact deterministic array verification.
                    with np.load(mask_path) as cached_mask:
                        if (not np.array_equal(cached_mask["probability"], probability)
                                or not np.array_equal(cached_mask["prediction"], native)):
                            raise ValueError("Orphaned segmentation intermediate differs")
                else:
                    temporary = mask_path.with_suffix(".tmp")
                    with temporary.open("wb") as stream:
                        np.savez_compressed(stream, probability=probability, prediction=native)
                    temporary.replace(mask_path)
                shard = {"sample_id": sample.sample_id, "protocol_sha256": protocol_sha,
                    "image_sha256": sample.image_sha256, "image_wh": list(source.size),
                    "mask_path": str(mask_path.relative_to(output)), "mask_sha256": file_sha256(mask_path),
                    "proposals": crops}
                write_json(output / "samples" / (key + ".json"), shard)
                shards[sample.sample_id] = shard
                if position % 25 == 0 or position == len(pending):
                    print(f"[ICDAR2015 TEST masks] {len(shards)}/{len(manifest.samples)}", flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    segmentation_peak = ({"allocated": torch.cuda.max_memory_allocated(device),
                          "reserved": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else None)
    unique, pending_ocr = set(), []
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    for sample in manifest.samples:
        for crop in shards[sample.sample_id]["proposals"]:
            key = crop["crop_rgb_sha256"]
            if key in unique:
                continue
            unique.add(key)
            path = output / "ocr" / (key + ".json")
            if path.exists():
                cached = json.loads(path.read_text())
                if cached["protocol_sha256"] != protocol_sha or cached["crop_rgb_sha256"] != key:
                    raise ValueError("Cached OCR provenance differs")
            else:
                pending_ocr.append((sample.sample_id, crop))
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    if pending_ocr:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=args.parseq_source, checkpoint=args.parseq_checkpoint, device=args.device,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE, runtime_source_archive=args.parseq_source_archive))
        for start in range(0, len(pending_ocr), config["parseq_batch_size"]):
            batch = pending_ocr[start:start + config["parseq_batch_size"]]
            images, crops = {}, []
            for sample_id, crop in batch:
                if sample_id not in images:
                    images[sample_id] = original_rgb(by_id[sample_id], root)
                x1, y1, x2, y2 = crop["bbox"]
                rgb = np.ascontiguousarray(images[sample_id][y1:y2, x1:x2])
                if crop_hash(rgb) != crop["crop_rgb_sha256"]:
                    raise ValueError("RGB crop changed after segmentation freeze")
                crops.append(rgb)
            predictions = recognizer.recognize_candidate_batch(crops)
            if len(predictions) != len(batch):
                raise RuntimeError("PARSeq batch output count differs")
            for (_, crop), prediction in zip(batch, predictions):
                key = crop["crop_rgb_sha256"]
                write_json(output / "ocr" / (key + ".json"), {"protocol_sha256": protocol_sha,
                    "crop_rgb_sha256": key, **prediction.to_dict()})
            if (start // config["parseq_batch_size"]) % 10 == 0 or start + len(batch) == len(pending_ocr):
                print(f"[ICDAR2015 PARSeq] {min(start + len(batch), len(pending_ocr))}/{len(pending_ocr)}", flush=True)
        recognizer.release()
    ocr_peak = ({"allocated": torch.cuda.max_memory_allocated(device),
                "reserved": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else None)
    # Only now read transcriptions/ignore GT, after all crops and OCR are frozen.
    rows, ignored_total = [], 0
    for sample in manifest.samples:
        shard = shards[sample.sample_id]
        words, ignored, annotation_sha = ground_truth(sample, root, tuple(shard["image_wh"]))
        ignored_total += ignored
        predictions = []
        for crop in shard["proposals"]:
            path = output / "ocr" / (crop["crop_rgb_sha256"] + ".json")
            cached = json.loads(path.read_text())
            predictions.append({**crop, "text": cached["raw_text"], "ocr_file_sha256": file_sha256(path)})
        counts = sample_counts(predictions, words, iou_threshold=config["matching_iou_threshold"])
        rows.append({"sample_id": sample.sample_id, "ground_truth": words,
            "ignored_gt_count": ignored, "annotation_sha256": annotation_sha,
            "predictions": predictions, "counts": counts})
    summary = aggregate_counts([row["counts"] for row in rows])
    if summary["images"] != config["expected_images"] or summary["gt_words"] != config["expected_care_words"]:
        raise RuntimeError(f"Official ICDAR2015 TEST denominator changed: {summary['images']}/{summary['gt_words']}")
    result = {"schema_version": 1, "status": "SUCCESS", "reproduction_mode": "approximate",
        "claim_label": "approximate reproduction", "result_class": "approximate reproduction",
        "protocol": protocol, "aggregate": summary, "ignored_gt": ignored_total, "failed_samples": 0,
        "per_image": rows, "runtime": runtime_environment(), "device": args.device,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "segmentation_peak_bytes": segmentation_peak, "ocr_peak_bytes": ocr_peak,
        "oom": False, "precision": "float32", "segmentation_batch_size": 1,
        "parseq_batch_size": config["parseq_batch_size"],
        "packages": {name: version(name) for name in ("torchvision", "numpy", "Pillow", "scipy")},
        "elapsed_seconds_this_invocation": time.monotonic() - started}
    result_path = output / "results.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if any(previous[key] != result[key] for key in ("protocol", "aggregate", "per_image")):
            raise ValueError("Completed evaluation result changed")
        return previous
    write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("config", "training-config", "training-run", "selection-report", "test-manifest", "data-root",
                 "parseq-source", "parseq-source-archive", "parseq-checkpoint", "output-dir"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception as error:
        write_json(args.output_dir / "failure.json", {"status": "FAILED", "oom": isinstance(error,torch.cuda.OutOfMemoryError),
            "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
            "error": str(error), "device": args.device, "precision": "float32",
            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None})
        raise
    print(json.dumps(result.get("aggregate", result), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
