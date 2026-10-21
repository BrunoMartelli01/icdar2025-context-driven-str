"""APPROXIMATE matched TEST OCR across all retained native128 model families."""
from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_cached_totaltext_padding import crop_hash, evaluation_rows, proposals_for_padding
from scripts.evaluate_totaltext_compact_v2_selected_ocr import measure_mask, mask_summary
from scripts.evaluate_totaltext_v1_epoch447_ocr_padding import read_crop, write, seal
from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt, _sample_key
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.evaluate import _model_from_config, _preprocess, validate_checkpoint
from segmentation.training import seed_everything

REPO = Path(__file__).resolve().parents[1]


def inference_config(path):
    """Read pinned resolved snapshots; training-worker policy is not inference."""
    config = OmegaConf.load(path)
    required = {"reproduction_mode": "approximate", "transforms.image_size": [256, 256],
        "transforms.mask_size": [128, 128], "transforms.normalization": "imagenet",
        "transforms.resize_geometry": "stretch", "transforms.image_interpolation": "bilinear",
        "architecture.implementation": "author_clarified_ag_unet"}
    for dotted, expected in required.items():
        actual = OmegaConf.select(config, dotted)
        if actual != expected:
            raise ValueError(f"Unsupported inference snapshot {dotted}: {actual!r}")
    return config


def original_padding_proposals(proposals, image_hw, padding):
    """Freeze pad0 selected components; change only the crop coordinate frame."""
    h, w = image_hw
    output = []
    for p in proposals:
        x0, y0, x1, y1 = p["bbox"]
        output.append({**p, "padded_bbox_original": [max(0, x0-padding), max(0, y0-padding),
            min(w, x1+padding), min(h, y1+padding)], "padded_bbox_native": None,
            "padding_coordinate_frame": "original_image", "padding_per_side": padding})
    return {"proposals": output, "selection": "frozen_native_pad0_components"}


def run(args):
    started = time.monotonic()
    c = json.loads(args.config.read_text())
    if (args.config.resolve().parent != REPO / "configs/approximate"
        or c["variant"] != "all-native128-gt-review-v1" or c["reproduction_mode"] != "approximate"):
        raise ValueError("Isolated explicit APPROXIMATE plan required")
    rt = c["runtime"]
    if (not torch.cuda.is_available() or rt["device"] != "cuda:0" or rt["precision"] != "float32"
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != rt["cublas_workspace_config"]):
        raise ValueError("Explicit deterministic CUDA FP32 runtime required; no fallback")
    torch.set_num_threads(rt["cpu_threads"])
    torch.set_num_interop_threads(1)
    seed_everything(rt["seed"], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    device = torch.device(rt["device"])
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    if args.output.exists():
        raise FileExistsError("Use a fresh diagnostic output directory")
    manifest = load_segmentation_manifest(Path(c["test_manifest"]), require_checksums=True)
    if (file_sha256(Path(c["test_manifest"])) != c["test_manifest_sha256"]
        or len(manifest.samples) != c["test_images"] or manifest.split_role != "test"):
        raise ValueError("Qualified complete TEST manifest required")
    samples = list(manifest.samples)
    if args.limit_images:
        if not 1 <= args.limit_images <= len(samples):
            raise ValueError("Invalid smoke image limit")
        samples = samples[:args.limit_images]
    if (file_sha256(Path(c["parseq_checkpoint"])) != PARSEQ_CHECKPOINT_SHA256
        or file_sha256(Path(c["parseq_archive"])) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Qualified official PARSeq bytes required")
    data = Path(c["test_data_root"])
    image_paths = {s.sample_id: data / s.image_path for s in samples}
    expected_image_hashes = {s.sample_id: s.image_sha256 for s in samples}
    for sid, path in image_paths.items():
        if file_sha256(path) != expected_image_hashes[sid]:
            raise ValueError(f"Image changed: {sid}")
    for model in c["models"]:
        for field, hash_field in [("source_manifest", "source_manifest_sha256"), ("config", "config_sha256")]:
            if file_sha256(Path(model[field])) != model[hash_field]:
                raise ValueError(f"Frozen model source changed: {model['id']}/{field}")
    code = [Path(__file__), REPO / "scripts/evaluate_totaltext_compact_v2_selected_ocr.py",
        REPO / "scripts/evaluate_totaltext_v1_epoch447_ocr_padding.py", REPO / "scripts/evaluate_cached_totaltext_padding.py",
        REPO / "scripts/evaluate_totaltext_v1_findcontours_parseq.py", REPO / "recognition/parseq.py",
        REPO / "recognition/scene_text_v1_evaluation.py", REPO / "segmentation/evaluate.py",
        REPO / "segmentation/data.py", REPO / "segmentation/training.py", REPO / "models/unet_adv.py",
        REPO / "localize/repository_contours.py", REPO / "localize/native_repository_contours.py"]
    protocol = {"schema_version": 1, "claim_label": "approximate reproduction", "reproduction_mode": "approximate",
        "result_class": "smoke-test; approximate reproduction" if args.limit_images else "approximate reproduction; complete TEST300 post-hoc comparison",
        "config": c, "config_sha256": file_sha256(args.config), "sample_ids": [s.sample_id for s in samples],
        "runtime": {**rt, "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__, "numpy": np.__version__},
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256, "parseq_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "decode_ar": True, "refine_iters": 1, "code_sha256": {str(p.resolve()): file_sha256(p) for p in code},
        "metric": "maximum one-to-one unpadded original bbox IoU>=.5 plus Unicode-casefold exact; punctuation retained; care GT; all predictions count",
        "original6": "same pad0 selected components and native area policy; only crop padding changes to6 original RGB pixels per side; not a full original-pipeline replay",
        "ground_truth_use": "all TEST predictions and OCR frozen before annotations are loaded", "table5_claim": False}
    seal(args.output / "protocol.json", protocol)
    protocol_sha = file_sha256(args.output / "protocol.json")
    pipelines, states = {}, {}
    for record in c["models"]:
        config = inference_config(Path(record["config"]))
        if list(config.transforms.mask_size) != [128, 128] or config.transforms.normalization != "imagenet":
            raise ValueError("All candidates must really use native128 ImageNet preprocessing")
        model = _model_from_config(config).eval()
        metadata = validate_checkpoint(Path(record["checkpoint"]), model, record["checkpoint_sha256"], config.architecture.identity)
        if metadata["epoch"] != record["epoch"]:
            raise ValueError("Loaded epoch differs")
        state_hash = sha256()
        for name, tensor in model.state_dict().items():
            state_hash.update(name.encode()); state_hash.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        states[record["id"]] = state_hash.hexdigest()
        model.to(device)
        for area in record["areas"]:
            pipelines[f"{record['id']}-a{area}"] = {"model": record, "min_bbox_area": area, "shards": []}
        for i, sample in enumerate(samples, 1):
            with Image.open(image_paths[sample.sample_id]) as im:
                rgb = im.convert("RGB")
            image = np.ascontiguousarray(rgb)
            with torch.inference_mode():
                logits = model(_preprocess(rgb).unsqueeze(0).to(device))
            if tuple(logits.shape) != (1, 1, 128, 128) or not torch.isfinite(logits).all():
                raise ValueError("Invalid native128 logits")
            native = (logits.sigmoid()[0, 0] >= c["probability_threshold"]).cpu().numpy().astype(np.uint8)
            for area in record["areas"]:
                key = f"{record['id']}-a{area}"
                plan = {"padding_native_pixels": c["native_paddings"], "min_unpadded_bbox_area_native": area,
                    "max_proposals_per_image": c["max_proposals"]}
                variants = proposals_for_padding(native, image.shape[:2], plan)
                sizes = measure_mask(native, variants)
                variants["original6"] = original_padding_proposals(variants["0"]["proposals"], image.shape[:2], c["original_padding"])
                for v in variants.values():
                    for p in v["proposals"]:
                        x0, y0, x1, y1 = p["padded_bbox_original"]
                        p["crop_array_sha256"] = crop_hash(image[y0:y1, x0:x1])
                        p["crop_hw"] = [y1-y0, x1-x0]
                shard = {"sample_id": sample.sample_id, "protocol_sha256": protocol_sha, "model": record["id"],
                    "image_sha256": sample.image_sha256, "image_hw": list(image.shape[:2]),
                    "native_mask_array_sha256": sha256(native.tobytes()).hexdigest(), "mask_measurements": sizes, "variants": variants}
                seal(args.output / key / "proposals" / (_sample_key(sample.sample_id)+".json"), shard)
                pipelines[key]["shards"].append(shard)
            if i % 100 == 0 or i == len(samples):
                print(f"[native128] model={record['id']} images={i}/{len(samples)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        del model, logits, rgb, image, native
        gc.collect(); torch.cuda.empty_cache()
    unique = {}
    for pipeline in pipelines.values():
        for shard in pipeline["shards"]:
            for v in shard["variants"].values():
                for p in v["proposals"]:
                    unique.setdefault(p["crop_array_sha256"], (shard["sample_id"], p))
    print(f"[PARSeq] unique={len(unique)}; RAM crops only", flush=True)
    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(c["parseq_source"]),
        checkpoint=Path(c["parseq_checkpoint"]), device=str(device), runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
        runtime_source_archive=Path(c["parseq_archive"])))
    ocr, pending, previous_id, cached = {}, list(unique.items()), None, None
    try:
        for start in range(0, len(pending), rt["parseq_batch_size"]):
            batch = pending[start:start+rt["parseq_batch_size"]]
            crops = []
            for digest, (sid, p) in batch:
                if sid != previous_id:
                    if file_sha256(image_paths[sid]) != expected_image_hashes[sid]:
                        raise ValueError("RGB changed before recognition")
                    with Image.open(image_paths[sid]) as im:
                        cached = np.ascontiguousarray(im.convert("RGB"))
                    previous_id = sid
                crops.append(read_crop(cached, p))
            decoded = recognizer.recognize_candidate_batch(crops)
            if len(decoded) != len(batch):
                raise RuntimeError("OCR cardinality mismatch")
            with (args.output / "ocr.jsonl").open("a", encoding="utf-8") as stream:
                for (digest, _), value in zip(batch, decoded):
                    row = {"protocol_sha256": protocol_sha, "crop_array_sha256": digest, "raw_text": value.raw_text,
                        "sequence_confidence_native": value.sequence_confidence_native, "token_ids": value.token_ids}
                    ocr[digest] = row
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            if start % (rt["parseq_batch_size"]*64) == 0 or start+len(batch) == len(pending):
                print(f"[PARSeq] {start+len(batch)}/{len(pending)} elapsed={time.monotonic()-started:.1f}s", flush=True)
    finally:
        recognizer.release(); del recognizer
        gc.collect(); torch.cuda.empty_cache()
    gt, annotations = {}, []
    for sample in samples:
        path = image_paths[sample.sample_id]
        annotation = path.parent.parent / "annotations" / (sample.sample_id.partition(":")[2]+".txt")
        with Image.open(path) as im:
            gt[sample.sample_id] = _gt(annotation, im.size)
        annotations.append({"sample_id": sample.sample_id, "path": str(annotation), "sha256": file_sha256(annotation)})
    care = sum(map(len, gt.values()))
    if not args.limit_images and care != c["test_care_words"]:
        raise ValueError("Complete care denominator differs")
    seal(args.output / "ground_truth.json", {"ground_truth": gt, "annotation_provenance": annotations})
    reports = {}
    for key, pipeline in pipelines.items():
        summary = {}
        for pad in [*c["native_paddings"], "original6"]:
            rows = evaluation_rows(pipeline["shards"], gt, ocr, pad, padded=False)
            summary[str(pad)] = evaluate_images(rows, text_available=True)
        report = {"model": pipeline["model"], "min_bbox_area": pipeline["min_bbox_area"], "summary": summary,
            "mask_summary": mask_summary(pipeline["shards"], c["native_paddings"])}
        seal(args.output / key / "results.json", report)
        reports[key] = report
        print(json.dumps({"pipeline": key, "correct": {p:v["recognition"]["exact_casefold_spatial_one_to_one"] for p,v in summary.items()}}), flush=True)
    result = {"status": "SUCCESS", "claim_label": "approximate reproduction", "result_class": protocol["result_class"],
        "protocol_sha256": protocol_sha, "images_per_pipeline": len(samples), "care_words": care,
        "model_state_sha256": states, "pipelines": reports, "unique_rgb_crops": len(unique),
        "elapsed_seconds": time.monotonic()-started, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "saved_crops_or_weights": 0, "out_of_memory": False}
    write(args.output / "results.json", result)
    size = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
    if size > c["output_budget_bytes"]:
        raise RuntimeError("Numeric output budget exceeded")
    print(json.dumps({"status": "SUCCESS", "output_bytes": size, "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-images", type=int)
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        ready = torch.cuda.is_initialized()
        write(args.output / "failure.json", {"status": "FAILED", "error_type": type(error).__name__,
            "error": str(error), "traceback": traceback.format_exc(),
            "out_of_memory": isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError)),
            "precision": "float32", "device": "cuda:0", "models": "sequential AG-U-Net then PARSeq",
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if ready else None,
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved() if ready else None})
        raise
