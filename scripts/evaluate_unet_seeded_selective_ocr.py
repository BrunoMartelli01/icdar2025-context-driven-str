"""Fixed PARSeq evaluation of the six qualified seeded/selective replay arms."""

import argparse
import fcntl
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, PARSEQ_RUNTIME_REVISION, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.evaluate_unet_crop_links_ocr import cache_record, cached_text, score_predictions
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_seeded_selective import ARMS, DEFAULT_CONFIG, FILES, REPO, load_export, read_config
from scripts.summarize_unet_crop_links import paired_changes
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.training import runtime_environment


def run(args):
    # The replay contract must exist before acquiring a writable output lock.
    if not (args.run_root / "run_contract.json").is_file():
        raise ValueError("Run the seeded/selective replay before OCR")
    with (args.run_root / "ocr.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(args)


def _run(args):
    config, source = read_config(args.config, args.mode)
    torch.set_num_threads(config["cpu_threads"])
    if args.batch_size != config["ocr_batch_size"]:
        raise ValueError("OCR batch must match the explicit config; use a separate config/output for overrides")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable; no fallback")
    root, output = args.word_root.resolve(), args.run_root.resolve()
    contract = json.loads((output / "run_contract.json").read_text())
    if contract["resolved_config"] != config or contract["source_config"] != source:
        raise ValueError("Replay/OCR resolved configurations differ")
    for n, digest in contract["source_sha256"].items():
        if sha256_file(REPO / n) != digest: raise ValueError("Qualified replay source changed: " + n)
    if contract["runtime_inputs"]["word_root"] != str(root): raise ValueError("Word dataset root changed")
    if contract["result_class"] != "approximate reproduction" and not args.smoke:
        raise ValueError("Partial smoke exports cannot enter final OCR evaluation")
    result_class = "smoke-test result" if args.smoke else "approximate reproduction"
    manifest_path = root / source["train_manifest"]
    if sha256_file(manifest_path) != source["word_manifest_sha256"]: raise ValueError("Qualified word manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train": raise ValueError("Official TRAIN only")
    _, full_indices, split = _split_indices(manifest, source["seed"], source["validation_count"])
    if split != contract["validation_ids_sha256"]: raise ValueError("Frozen validation split changed")
    indices = contract["indices"]
    if (not args.smoke and indices != full_indices) or not set(indices) <= set(full_indices):
        raise ValueError("Incomplete frozen evaluation population")
    by_id = {manifest.samples[i].sample_id: manifest.samples[i] for i in indices}
    if (sha256_file(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256
            or sha256_file(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Qualified PARSeq artifact checksum changed")
    execution = {"device": args.device, "batch_size": args.batch_size, "dtype": "float32",
        "torch_version": torch.__version__, "runtime_revision": PARSEQ_RUNTIME_REVISION,
        "decode_ar": True, "refine_iters": 1, "checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "recognizer_source_sha256": sha256_file(REPO / "recognition/parseq.py"),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_threads": torch.get_num_threads()}
    cache = output / "corrective_ocr_cache"
    path = cache / "execution.json"
    if path.exists() and json.loads(path.read_text()) != execution: raise ValueError("OCR cache backend changed")
    external = args.cache_source.resolve() if args.cache_source else None
    if external is not None and json.loads((external / "execution.json").read_text()) != execution:
        raise ValueError("Source OCR cache backend differs; omit --cache-source for an isolated new cache")
    _json(path, execution)
    protocol = {"reproduction_mode": "approximate", "result_class": result_class,
                "resolved_config": config, "branches": list(ARMS), "cache_execution": execution,
                "replay_contract_sha256": sha256_file(output / "run_contract.json"),
                "validation_ids_sha256": split, "word_manifest_sha256": source["word_manifest_sha256"],
                "source_sha256": {n: sha256_file(REPO / n) for n in sorted(set(FILES))},
                "matching": "original RGB maximum one-to-one bbox IoU>=.5, exact casefold; secondary Unicode P/S stripping",
                "ignore": "care GT; wholly invalid supports excluded after crop recognition; partial crops retained",
                "crop": "original RGB, pad0, no area filter or instance cap",
                "cache_source": str(external) if external else None,
                "comparison_baseline": "control"}
    inputs, pending, texts, report_hashes = {}, {}, {}, {}
    reused_external = set()
    for arm, source_arm in ARMS.items():
        directory = output / arm; report_path = directory / "validation.json"
        report = json.loads(report_path.read_text()); receipt = json.loads((directory / "run_manifest.json").read_text())
        if (receipt.get("status") != "SUCCESS" or receipt.get("execution_kind") != "frozen_postprocess"
                or receipt["run_contract"] != contract or report["run_contract"] != contract
                or receipt["result_class"] != contract["result_class"] or report["result_class"] != contract["result_class"]
                or receipt["source_checkpoint"] != config["source_artifacts"][source_arm]
                or report["checkpoint_sha256"] != receipt["source_checkpoint"]["checkpoint_sha256"]
                or sha256_file(report_path) != receipt["validation_sha256"]
                or len(report["per_image"]) != len(by_id)
                or {r["sample_id"] for r in report["per_image"]} != set(by_id)
                or (not args.smoke and report["aggregate"]["gt_words"] != source["expected_validation_words"])):
            raise ValueError("Incomplete or incompatible replay exports: " + arm)
        report_hashes[arm] = sha256_file(report_path); inputs[arm] = []
        for row in report["per_image"]:
            sid = row["sample_id"]; sample = by_id[sid]
            _, exported, _ = load_export(output, arm, sid, row)
            image_path = root / sample.image_path
            if exported["image_sha256"] != sample.image_sha256 or sha256_file(image_path) != sample.image_sha256:
                raise ValueError("Exported source RGB changed")
            with Image.open(image_path) as image: rgb = np.asarray(image.convert("RGB"))
            proposals = []
            for p in exported["proposals"]:
                x, y, r, b = p["bbox"]
                if not (0 <= x < r <= rgb.shape[1] and 0 <= y < b <= rgb.shape[0]): raise ValueError("Crop outside RGB")
                digest = crop_array_hash(np.ascontiguousarray(rgb[y:b, x:r]))
                proposals.append({**p, "crop_rgb_sha256": digest})
                if digest in texts: continue
                cached = cache / (digest + ".json")
                if cached.exists(): texts[digest] = cached_text(json.loads(cached.read_text()), digest)
                elif external is not None and (external / cached.name).exists():
                    rec = json.loads((external / cached.name).read_text())
                    texts[digest] = cached_text(rec, digest); _json(cached, rec); reused_external.add(digest)
                else: pending[digest] = (image_path, p["bbox"])
            # Proposal selection and crop hashing precede annotation access.
            gt, ann_sha = _train_gt(sample, root, (rgb.shape[1], rgb.shape[0]))
            inputs[arm].append({"sample_id": sid, "ground_truth": gt, "annotation_sha256": ann_sha,
                                "proposals": proposals, "ignored_ids": row["ignored_instance_ids"]})
    protocol["validation_report_sha256"] = report_hashes
    protocol_path = output / "ocr_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError("OCR output protocol changed; use a new output root")
    _json(protocol_path, protocol)
    if device.type == "cuda": _prepare_cuda_memory_tracking(device)
    recognizer = None
    try:
        jobs = sorted((k, v) for k, v in pending.items() if k not in texts)
        if jobs:
            recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
                checkpoint=args.parseq_checkpoint, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                runtime_source_archive=args.parseq_source_archive, device=args.device))
        for start in range(0, len(jobs), args.batch_size):
            batch = jobs[start:start + args.batch_size]; crops = []
            for digest, (path, box) in batch:
                with Image.open(path) as image: rgb = np.asarray(image.convert("RGB"))
                x, y, r, b = box; crop = np.ascontiguousarray(rgb[y:b, x:r])
                if crop_array_hash(crop) != digest: raise ValueError("RGB crop changed before recognition")
                crops.append(crop)
            predictions = recognizer.recognize_candidate_batch(crops)
            if len(predictions) != len(batch): raise RuntimeError("PARSeq output count mismatch")
            for (digest, _), pred in zip(batch, predictions):
                rec = cache_record(pred, digest, recognizer.config)
                texts[digest] = cached_text(rec, digest); _json(cache / (digest + ".json"), rec)
            print(f"[PARSeq seeded/selective] {min(start+len(batch), len(jobs))}/{len(jobs)} unique new crops", flush=True)
        results = {}
        for arm, samples in inputs.items():
            rows = []
            for sample in samples:
                predictions = [{**p, "text": texts[p["crop_rgb_sha256"]]} for p in sample["proposals"]
                               if p["instance_id"] not in sample["ignored_ids"]]
                metrics, matched = score_predictions(predictions, sample["ground_truth"], source["ocr_iou_threshold"])
                rows.append({"sample_id": sample["sample_id"], **metrics, "ground_truth": sample["ground_truth"],
                             "annotation_sha256": sample["annotation_sha256"], "predictions": predictions,
                             "matched_gt_indices": {k: sorted(v) for k, v in matched.items()}})
            aggregate = {k: sum(r[k] for r in rows) for k in ("gt_words", "proposals", "geometry_iou_ge_50",
                         "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")}
            if not args.smoke and aggregate["gt_words"] != source["expected_validation_words"]:
                raise ValueError("OCR denominator changed")
            for k in ("exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50"):
                aggregate[k + "_recall_all_gt"] = aggregate[k]/aggregate["gt_words"] if aggregate["gt_words"] else 1.
            results[arm] = {"aggregate": aggregate, "per_image": rows}
        for arm, value in results.items():
            value["exact_changes_vs_control"] = paired_changes(value["per_image"], results["control"]["per_image"])
        result = {"status": "SUCCESS", "protocol": protocol, "arms": results, "environment": runtime_environment(),
                  "cache_reused_external": len(reused_external), "new_unique_crops_recognized": len(jobs),
                  "memory": {"device": str(device), "model_set": "PARSeq only; segmentation maps replayed separately",
                    "dtype": "float32", "batch_size": args.batch_size, "input_shape": [args.batch_size, 3, 32, 128], "oom": False,
                    "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}}
        _json(output / "ocr_comparison.json", result)
        return result
    except Exception as error:
        _json(output / "ocr_failure.json", {"status": "FAILED", "error_type": type(error).__name__, "message": str(error),
            "oom": isinstance(error, torch.cuda.OutOfMemoryError), "protocol": protocol,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
        raise
    finally:
        if recognizer is not None: recognizer.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("approximate",), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--word-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--parseq-source", type=Path, required=True)
    p.add_argument("--parseq-source-archive", type=Path, required=True)
    p.add_argument("--parseq-checkpoint", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--cache-source", type=Path, help="Read-only reuse; identical execution backend required")
    p.add_argument("--smoke", action="store_true", help="Separate, explicitly labeled partial OCR qualification")
    run(p.parse_args())
