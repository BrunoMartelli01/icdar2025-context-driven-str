"""APPROXIMATE TEST padding OCR and mask-size diagnostic; numeric outputs only."""
from __future__ import annotations

import argparse
import gc
import json
from hashlib import sha256
import os
from pathlib import Path
import platform
import resource
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch

from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_cached_totaltext_padding import (
    crop_hash, correct_word_ids, evaluation_rows, proposals_for_padding)
from scripts.evaluate_totaltext_v1_epoch447_ocr_padding import read_crop, seal, write
from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt, _sample_key
from segmentation.config import load_segmentation_config
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.evaluate import _model_from_config, _preprocess, validate_checkpoint
from segmentation.training import seed_everything

REPO = Path(__file__).resolve().parents[1]


def validate_target_variant(plan, contract):
    """Keep the historical V2 and explicit V3 evaluation namespaces distinct."""
    variant=plan['variant']
    target=contract.get('target_variant',contract.get('config',{}).get('target_variant'))
    if variant=='guarded-v3-selected-ocr-padding-v1':
        if target!='guarded-word-targets-v3' or not contract.get('guard_recipe_sha256'):
            raise ValueError('V3 OCR requires the recorded V3 target recipe')
    elif variant=='compact-v2-selected-ocr-padding-v1':
        if target=='guarded-word-targets-v3':
            raise ValueError('V3 training cannot be evaluated under a V2 plan')
    else:
        raise ValueError('Unsupported selected-checkpoint OCR variant')


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    return {"count": int(a.size), **{name: float(fn(a)) if a.size else None for name, fn in
        [("mean", np.mean), ("median", np.median), ("p05", lambda x: np.quantile(x, .05)),
         ("p95", lambda x: np.quantile(x, .95)), ("min", np.min), ("max", np.max)]}}


def measure_mask(mask, variants):
    """Count actual foreground pixels, including holes and rejected fragments."""
    if mask.shape != (128, 128) or not np.isin(mask, (0, 1)).all():
        raise ValueError("Expected binary native128 mask")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = [{"label": i, "bbox_native": [int(stats[i, 0]), int(stats[i, 1]),
        int(stats[i, 0] + stats[i, 2]), int(stats[i, 1] + stats[i, 3])],
        "foreground_area_native": int(stats[i, cv2.CC_STAT_AREA])} for i in range(1, n)]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for variant in variants.values():
        for p in variant["proposals"]:
            contour = contours[p["contour_index"]]
            x, y = contour[0, 0]
            label = int(labels[y, x])
            if label < 1:
                raise ValueError("External contour does not map to foreground")
            component = components[label - 1]
            if p["bbox_native"] != component["bbox_native"]:
                raise ValueError("Contour/component bounding boxes differ")
            p["component_label"] = label
            p["foreground_area_native"] = component["foreground_area_native"]
    pixels = int(np.count_nonzero(mask))
    if sum(c["foreground_area_native"] for c in components) != pixels:
        raise ValueError("Connected component areas do not partition foreground")
    return {"native_mask_hw": [128, 128], "foreground_pixels": pixels,
        "foreground_fraction": pixels / mask.size, "components_8_connected": components,
        "external_contours": len(contours)}


def mask_summary(shards, pads):
    components = [c for s in shards for c in s["mask_measurements"]["components_8_connected"]]
    result = {"native_mask_hw": [128, 128], "segmentation_input_hw": [256, 256],
        "images": len(shards), "foreground_pixels_per_image": distribution([
            s["mask_measurements"]["foreground_pixels"] for s in shards]),
        "foreground_fraction_per_image": distribution([
            s["mask_measurements"]["foreground_fraction"] for s in shards]),
        "all_component_foreground_area_native": distribution([c["foreground_area_native"] for c in components]),
        "components_per_image": distribution([len(s["mask_measurements"]["components_8_connected"]) for s in shards]),
        "retained_by_padding": {}}
    for pad in pads:
        proposals = [p for s in shards for p in s["variants"][str(pad)]["proposals"]]
        dimensions = lambda key, axis: distribution([p[key][axis + 2] - p[key][axis] for p in proposals])
        result["retained_by_padding"][str(pad)] = {
            "proposals": len(proposals),
            "component_foreground_area_native": distribution([p["foreground_area_native"] for p in proposals]),
            "bbox_area_native": distribution([p["bbox_area_native"] for p in proposals]),
            "bbox_width_native": dimensions("bbox_native", 0), "bbox_height_native": dimensions("bbox_native", 1),
            "bbox_width_original": dimensions("bbox", 0), "bbox_height_original": dimensions("bbox", 1),
            "padded_crop_width_native": dimensions("padded_bbox_native", 0),
            "padded_crop_height_native": dimensions("padded_bbox_native", 1),
            "padded_crop_width_original": dimensions("padded_bbox_original", 0),
            "padded_crop_height_original": dimensions("padded_bbox_original", 1),
            "foreground_to_bbox_area_ratio": distribution([
                p["foreground_area_native"] / p["bbox_area_native"] for p in proposals])}
    return result


def run(args):
    started = time.monotonic()
    plan = json.loads(args.config.read_text())
    if (args.config.resolve().parent != REPO / "configs/approximate"
            or plan["variant"] not in ("compact-v2-selected-ocr-padding-v1", "guarded-v3-selected-ocr-padding-v1")
            or plan["reproduction_mode"] != "approximate"):
        raise ValueError("Explicit isolated APPROXIMATE plan required")
    pads = plan["padding_native_pixels"]
    if pads != sorted(set(pads)) or any(type(p) is not int or p < 0 for p in pads) or 0 not in pads:
        raise ValueError("Invalid native padding grid")
    runtime = plan["runtime"]
    if runtime["precision"] != "float32" or runtime["device"] != "cuda:0" or not torch.cuda.is_available():
        raise ValueError("Qualified CUDA FP32 runtime required; no substitute")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != runtime["cublas_workspace_config"]:
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG to the explicit runtime plan before launching Python")
    device = torch.device(runtime["device"])
    torch.set_num_threads(runtime["cpu_threads"])
    torch.set_num_interop_threads(1)
    seed_everything(runtime["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    root, output, data = args.run_directory.resolve(), args.output_dir.resolve(), args.data_root.resolve()
    if output.is_relative_to(root) or output.is_relative_to(data) or output.exists():
        raise ValueError("Use a fresh output directory outside source run and dataset")
    for name, digest in plan["source_sha256"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Frozen source metadata changed: {name}")
    contract = json.loads((root / "run_contract.json").read_text())
    validate_target_variant(plan,contract)
    source = json.loads((root / "run_manifest.json").read_text())
    selection = json.loads((root / "checkpoint_selection.json").read_text())
    if not source["training_complete"] or source["completed_epochs"] != 500 or source["test_loaded"]:
        raise ValueError("Expected complete TRAIN-only checkpoint selection")
    expected = {r["epoch"]: r for r in selection.values()}
    if sorted(expected) != plan["checkpoint_epochs"]:
        raise ValueError("Selected epoch population differs")
    for name, digest in contract["code_sha256"].items():
        if file_sha256(REPO / name) != digest:
            raise ValueError(f"Training implementation changed: {name}")
    config_path = REPO / contract["config"]["source_config"]
    if file_sha256(config_path) != contract["source_config_sha256"]:
        raise ValueError("Training source configuration changed")
    config = load_segmentation_config(config_path)
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if (file_sha256(args.test_manifest) != plan["test_manifest_sha256"] or manifest.dataset != "total_text"
            or manifest.split_role != "test" or len(manifest.samples) != plan["expected_images"]):
        raise ValueError("Qualified complete official TEST manifest required")
    samples = list(manifest.samples)
    if args.limit_images is not None:
        if not 1 <= args.limit_images <= len(samples):
            raise ValueError("Invalid smoke limit")
        samples = samples[:args.limit_images]
    if (file_sha256(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256
            or file_sha256(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Pinned qualified PARSeq artifacts required")
    for sample in samples:
        if file_sha256(data / sample.image_path) != sample.image_sha256:
            raise ValueError(f"TEST image checksum differs: {sample.sample_id}")
    code = [Path(__file__), REPO / "scripts/evaluate_totaltext_v1_epoch447_ocr_padding.py",
        REPO / "scripts/evaluate_cached_totaltext_padding.py", REPO / "recognition/parseq.py",
        REPO / "recognition/scene_text_v1_evaluation.py", REPO / "recognition/scene_text_datasets.py",
        REPO / "scripts/evaluate_totaltext_v1_findcontours_parseq.py", REPO / "segmentation/evaluate.py",
        REPO / "localize/repository_contours.py", REPO / "localize/native_repository_contours.py"]
    protocol = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
        "result_class": "smoke-test; approximate reproduction" if args.limit_images else "approximate reproduction; complete TEST300 exploratory diagnostic",
        "plan": plan, "config_sha256": file_sha256(args.config), "source_run": str(root),
        "source_run_contract": contract, "sample_ids": [s.sample_id for s in samples],
        "data_root": str(data), "test_manifest": str(args.test_manifest.resolve()),
        "parseq_source": str(args.parseq_source.resolve()), "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "parseq_decode_ar": True, "parseq_refine_iters": 1,
        "code_sha256": {str(p.resolve()): file_sha256(p) for p in code},
        "runtime": {**runtime, "gpu": torch.cuda.get_device_name(device), "python": platform.python_version(),
            "torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__,
            "deterministic_algorithms": True, "tf32": False},
        "ground_truth_use": "All masks/proposals/OCR frozen before annotation loading; GT for evaluation only",
        "primary_metric": "micro maximum one-to-one unpadded bbox IoU>=0.5 and Unicode casefold exact; punctuation retained; all predictions count; all care GT in denominator",
        "size_metric": "Foreground pixel counts on native128, 8-connected components; pooled component distributions and per-image coverage; no mask dilation",
        "table_5_claim": False, "saved_raster_or_weight_files": 0}
    seal(output / "protocol.json", protocol)
    protocol_sha = file_sha256(output / "protocol.json")
    all_shards, image_paths = {}, {s.sample_id: data / s.image_path for s in samples}
    for epoch in plan["checkpoint_epochs"]:
        chosen = expected[epoch]
        checkpoint = root / "checkpoints" / chosen["checkpoint"]
        if checkpoint.is_symlink() or checkpoint.parent != root / "checkpoints":
            raise ValueError("Unsafe selected checkpoint path")
        model = _model_from_config(config).eval()
        meta = validate_checkpoint(checkpoint, model, chosen["checkpoint_sha256"], config.architecture.identity)
        if meta["epoch"] != epoch:
            raise ValueError("Loaded checkpoint epoch differs")
        model.to(device)
        shards = []
        for index, sample in enumerate(samples, 1):
            with Image.open(image_paths[sample.sample_id]) as im:
                rgb = im.convert("RGB")
            image = np.ascontiguousarray(rgb)
            with torch.inference_mode():
                logits = model(_preprocess(rgb).unsqueeze(0).to(device))
            if tuple(logits.shape) != (1, 1, 128, 128) or not torch.isfinite(logits).all():
                raise ValueError("Invalid native segmentation logits")
            native = (logits.sigmoid()[0, 0] >= plan["probability_threshold"]).cpu().numpy().astype(np.uint8)
            variants = proposals_for_padding(native, image.shape[:2], plan)
            measurements = measure_mask(native, variants)
            for variant in variants.values():
                for p in variant["proposals"]:
                    x1, y1, x2, y2 = p["padded_bbox_original"]
                    p["crop_array_sha256"] = crop_hash(image[y1:y2, x1:x2])
                    p["crop_hw"] = [y2-y1, x2-x1]
            shard = {"sample_id": sample.sample_id, "epoch": epoch, "protocol_sha256": protocol_sha,
                "image_sha256": sample.image_sha256, "image_hw": list(image.shape[:2]),
                "native_mask_array_sha256": sha256(native.tobytes()).hexdigest(),
                "mask_measurements": measurements, "variants": variants}
            seal(output / f"epoch-{epoch:04d}/proposals/{_sample_key(sample.sample_id)}.json", shard)
            shards.append(shard)
            if index % 50 == 0 or index == len(samples):
                print(f"[segmentation] epoch={epoch} images={index}/{len(samples)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
        all_shards[epoch] = shards
        del model, logits, image, rgb, native
        gc.collect()
        torch.cuda.empty_cache()
    unique = {}
    for shards in all_shards.values():
        for shard in shards:
            for variant in shard["variants"].values():
                for p in variant["proposals"]:
                    unique.setdefault(p["crop_array_sha256"], (shard["sample_id"], p))
    print(f"[PARSeq] unique_rgb_crops={len(unique)}; sequential CUDA FP32; RAM crops only", flush=True)
    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source.resolve(),
        checkpoint=args.parseq_checkpoint.resolve(), device=str(device), runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
        runtime_source_archive=args.parseq_source_archive.resolve()))
    ocr, pending = {}, list(unique.items())
    previous_id, cached_image = None, None
    try:
        for start in range(0, len(pending), runtime["parseq_batch_size"]):
            batch = pending[start:start+runtime["parseq_batch_size"]]
            crops = []
            for digest, (sid, proposal) in batch:
                if sid != previous_id:
                    if file_sha256(image_paths[sid]) != next(s.image_sha256 for s in samples if s.sample_id == sid):
                        raise ValueError("Source image changed before OCR")
                    with Image.open(image_paths[sid]) as im:
                        cached_image = np.ascontiguousarray(im.convert("RGB"))
                    previous_id = sid
                crops.append(read_crop(cached_image, proposal))
            decoded = recognizer.recognize_candidate_batch(crops)
            if len(decoded) != len(batch):
                raise RuntimeError("OCR output cardinality differs")
            with (output / "ocr.jsonl").open("a", encoding="utf-8") as stream:
                for (digest, _), value in zip(batch, decoded):
                    row = {"protocol_sha256": protocol_sha, "crop_array_sha256": digest, **value.to_dict()}
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    ocr[digest] = row
            if start % (runtime["parseq_batch_size"] * 16) == 0 or start + len(batch) == len(pending):
                print(f"[PARSeq] completed={start+len(batch)}/{len(pending)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    finally:
        recognizer.release()
        del recognizer
        gc.collect()
        torch.cuda.empty_cache()
    gt, annotations = {}, []
    for sample in samples:
        path = image_paths[sample.sample_id]
        annotation = path.parent.parent / "annotations" / (sample.sample_id.partition(":")[2] + ".txt")
        with Image.open(path) as im:
            gt[sample.sample_id] = _gt(annotation, im.size)
        annotations.append({"sample_id": sample.sample_id, "path": str(annotation), "sha256": file_sha256(annotation)})
    care_words = sum(len(words) for words in gt.values())
    if args.limit_images is None and care_words != plan["expected_care_words"]:
        raise ValueError("Complete TEST care denominator differs")
    seal(output / "ground_truth.json", {"ground_truth": gt, "annotation_provenance": annotations})
    results = {}
    for epoch, shards in all_shards.items():
        summary, ids = {}, {}
        for pad in pads:
            summary[str(pad)], ids[str(pad)] = {}, {}
            for name, padded in [("unpadded_detection_box", False), ("padded_crop_box", True)]:
                rows = evaluation_rows(shards, gt, ocr, pad, padded=padded)
                summary[str(pad)][name] = evaluate_images(rows, text_available=True)
                ids[str(pad)][name] = sorted(correct_word_ids(rows))
            base = {s["sample_id"]: {p["contour_index"] for p in s["variants"]["0"]["proposals"]} for s in shards}
            summary[str(pad)]["selection_changed_images_vs_pad0"] = [s["sample_id"] for s in shards
                if {p["contour_index"] for p in s["variants"][str(pad)]["proposals"]} != base[s["sample_id"]]]
        result = {"epoch": epoch, "checkpoint_sha256": expected[epoch]["checkpoint_sha256"],
            "summary": summary, "correct_gt_ids": ids, "mask_summary": mask_summary(shards, pads)}
        seal(output / f"epoch-{epoch:04d}/results.json", result)
        results[str(epoch)] = result
        print(json.dumps({"epoch": epoch, "correct_words": {p: r["unpadded_detection_box"]["recognition"]["exact_casefold_spatial_one_to_one"] for p, r in summary.items()}}), flush=True)
    torch.cuda.synchronize(device)
    report = {"status": "SUCCESS", "claim_label": "approximate reproduction", "result_class": protocol["result_class"],
        "protocol_sha256": protocol_sha, "images_per_checkpoint": len(samples), "care_words_per_checkpoint": care_words,
        "unique_rgb_crops": len(unique), "checkpoints": results, "elapsed_seconds": time.monotonic()-started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "saved_raster_or_weight_files": 0, "out_of_memory": False}
    write(output / "results.json", report)
    size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    if size > plan["output_budget_bytes"]:
        raise RuntimeError("Numeric output budget exceeded")
    print(json.dumps({"status": "SUCCESS", "numeric_output_bytes": size,
        "elapsed_seconds": report["elapsed_seconds"], "peak_cuda_allocated_bytes": report["peak_cuda_allocated_bytes"],
        "peak_cuda_reserved_bytes": report["peak_cuda_reserved_bytes"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["config", "run-directory", "test-manifest", "data-root", "parseq-source",
                 "parseq-source-archive", "parseq-checkpoint", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--limit-images", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        initialized = torch.cuda.is_initialized()
        write(args.output_dir / "failure.json", {"status": "FAILED", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
            "out_of_memory": isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)),
            "precision": "float32", "model_set": "sequential AuthorProtocolAGUNet then qualified PARSeq",
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if initialized else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if initialized else None})
        raise


if __name__ == "__main__":
    main()
