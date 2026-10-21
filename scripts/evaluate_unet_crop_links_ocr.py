"""PARSeq on learned-crop exports; original RGB and immutable per-backend cache."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, PARSEQ_RUNTIME_REVISION, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct, _geometry_match, _train_gt
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_unet_crop_links import DEFAULT_CONFIG, read_config
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.training import runtime_environment


def cached_text(record, crop_sha):
    provenance = record.get("provenance", {})
    if (record.get("crop_rgb_sha256") != crop_sha or record.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
            or record.get("parseq_source_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
            or provenance.get("decode_ar") is not True or provenance.get("refine_iters") != 1
            or not isinstance(record.get("raw_text"), str)):
        raise ValueError("Qualified OCR cache provenance mismatch")
    return record["raw_text"]


def cache_record(prediction, crop_sha, recognizer_config):
    """Batch diagnostics omit provenance; attach the verified runtime settings.

    Called only after the real recognizer constructor has verified its archive,
    checkpoint, strict state loading and AR/refine defaults.
    """
    record = {**prediction.to_dict(), "crop_rgb_sha256": crop_sha,
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "provenance": {"decode_ar": recognizer_config.decode_ar, "refine_iters": recognizer_config.refine_iters,
                       "runtime_revision": PARSEQ_RUNTIME_REVISION,
                       "runtime_source_type": recognizer_config.runtime_source_type,
                       "runtime_source": str(recognizer_config.runtime_source.resolve()),
                       "strict_state_dict_load": True}}
    cached_text(record, crop_sha)
    return record


def score_predictions(predictions, gt, threshold):
    matched = {"geometry_iou_ge_50": _geometry_match(predictions, gt, threshold),
               "exact_casefold_iou_ge_50": _correct(predictions, gt, normalize=False),
               "exact_no_symbols_iou_ge_50": _correct(predictions, gt, normalize=True)}
    return {"gt_words": len(gt), "proposals": len(predictions), **{k: len(v) for k, v in matched.items()}}, matched


def checkpoint_path(directory, record):
    path = Path(record["checkpoint"]["path"])
    return path if path.is_absolute() else directory / path


def run(args):
    config, _ = read_config(args.config, args.mode)
    torch.set_num_threads(config["cpu_threads"])
    branches = config["embedding_arms"] if args.subset == "postprocess" else config["branches"] if args.subset == "training" else config["branches"] + config["embedding_arms"]
    root, output = args.word_root.resolve(), args.run_root.resolve()
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["word_manifest_sha256"]:
        raise ValueError("Qualified word dataset manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("This OCR study accepts held-out TRAIN only")
    _, validation_ids, split_sha = _split_indices(manifest, config["seed"], config["validation_count"])
    if split_sha != config["validation_ids_sha256"]:
        raise ValueError("Frozen validation split changed")
    if (sha256_file(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256
            or sha256_file(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Qualified local PARSeq artifacts changed")
    if args.batch_size < 1 or (str(args.device).startswith("cuda") and not torch.cuda.is_available()):
        raise ValueError("Invalid OCR batch or unavailable CUDA; no automatic fallback")
    by_id = {s.sample_id: s for s in manifest.samples}
    expected_ids = {manifest.samples[i].sample_id for i in validation_ids}
    protocol = {"reproduction_mode": "approximate", "result_class": "approximate reproduction",
        "validation_ids_sha256": split_sha, "word_manifest_sha256": config["word_manifest_sha256"],
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256, "parseq_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "decode_ar": True, "refine_iters": 1, "device": args.device, "batch_size": args.batch_size,
        "matching": "original RGB bbox maximum one-to-one IoU>=.50 with exact casefold or Unicode P/S stripping",
        "ignore": "GT care words only; wholly invalid native instances excluded AFTER crop recognition; partial boxes retained",
        "crop": "original RGB boxes from frozen first pass or learned crop refinement, pad0, no area filtering or instance cap",
        "source_sha256": sha256_file(Path(__file__)),
        "helper_source_sha256": {name: sha256_file(Path(__file__).resolve().parents[1] / name) for name in
            ("recognition/parseq.py", "recognition/scene_text_v1_evaluation.py", "recognition/scene_text_datasets.py",
             "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py", "scripts/evaluate_totaltext_highres_contour_padding.py",
             "segmentation/evaluate.py")}}
    inputs, pending, texts, source_hashes = {}, {}, {}, {}
    cache = output / "corrective_ocr_cache"
    execution = {"device": args.device, "batch_size": args.batch_size, "dtype": "float32",
        "torch_version": torch.__version__, "runtime_revision": PARSEQ_RUNTIME_REVISION,
        "decode_ar": True, "refine_iters": 1,
        "checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256, "archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "recognizer_source_sha256": sha256_file(Path(__file__).resolve().parents[1] / "recognition/parseq.py"),
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cpu_threads": torch.get_num_threads()}
    cache_protocol = cache / "execution.json"
    if cache_protocol.exists() and json.loads(cache_protocol.read_text()) != execution:
        raise ValueError("OCR cache execution backend changed; use a new output directory")
    _json(cache_protocol, execution)
    protocol.update(subset=args.subset, branches=branches, cache_execution=execution,
                    comparison_baseline="control" if "control" in branches else "embedding50_hard")
    for branch in branches:
        directory = output / branch
        trained = json.loads((directory / "run_manifest.json").read_text())
        report_path = directory / "validation.json"
        report = json.loads(report_path.read_text())
        if (trained.get("status") != "SUCCESS" or not trained.get("training_complete")
                or trained.get("completed_epochs") != (config["epochs"] if branch.startswith("crop_") else 0)
                or report.get("run_contract") != trained.get("run_contract")
                or report.get("checkpoint_sha256") != trained["checkpoint"]["sha256"]
                or sha256_file(checkpoint_path(directory, trained)) != trained["checkpoint"]["sha256"]
                or {r["sample_id"] for r in report["per_image"]} != expected_ids
                or len(report["per_image"]) != len(expected_ids)):
            raise ValueError("Incomplete or incompatible frozen validation exports: " + branch)
        source_hashes[branch] = sha256_file(report_path)
        inputs[branch] = []
        for row in report["per_image"]:
            sample_id = row["sample_id"]
            path = directory / "validation_exports" / sample_id.replace(":", "__") / "proposals.json"
            if sha256_file(path) != row["proposals_sha256"]:
                raise ValueError("Exported proposal receipt changed")
            record = json.loads(path.read_text())
            maps_path = path.parent / "maps.npz"
            if (sha256_file(maps_path) != row["maps_sha256"]
                    or record["maps_sha256"] != row["maps_sha256"]):
                raise ValueError("Exported original-coordinate supports changed")
            sample = by_id[sample_id]
            image_path = root / sample.image_path
            if record["sample_id"] != sample_id or record["image_sha256"] != sample.image_sha256 or sha256_file(image_path) != sample.image_sha256:
                raise ValueError("Exported source RGB changed")
            with Image.open(image_path) as opened: rgb = np.asarray(opened.convert("RGB"))
            proposals = []
            for proposal in record["proposals"]:
                x1, y1, x2, y2 = proposal["bbox"]
                if not (0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0]):
                    raise ValueError("Exported crop outside original RGB")
                digest = crop_array_hash(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
                proposals.append({**proposal, "crop_rgb_sha256": digest})
                cached = cache / (digest + ".json")
                if cached.exists():
                    texts[digest] = cached_text(json.loads(cached.read_text()), digest)
                else:
                    pending[digest] = (image_path, proposal["bbox"])
            # GT is consulted only after all crop geometry/hashes have been frozen.
            gt, annotation_sha = _train_gt(sample, root, (rgb.shape[1], rgb.shape[0]))
            inputs[branch].append({"sample_id": sample_id, "ground_truth": gt, "annotation_sha256": annotation_sha,
                "proposals": proposals, "ignored_ids": row["ignored_instance_ids"],
                "quality_scores": {int(k): v for k, v in record["quality_scores"].items()}})
    protocol["validation_report_sha256"] = source_hashes
    destination = output / ("ocr_protocol.json" if args.subset == "all" else args.subset + "_ocr_protocol.json")
    if destination.exists() and json.loads(destination.read_text()) != protocol:
        raise ValueError("OCR output protocol changed; preserve this run and use a new directory")
    _json(destination, protocol)
    device = torch.device(args.device)
    if device.type == "cuda": _prepare_cuda_memory_tracking(device)
    if pending:
        recognizer = None
        try:
            recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
                checkpoint=args.parseq_checkpoint, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                runtime_source_archive=args.parseq_source_archive, device=args.device))
            jobs = sorted(pending.items())
            for start in range(0, len(jobs), args.batch_size):
                batch = jobs[start:start + args.batch_size]
                crops = []
                for digest, (path, box) in batch:
                    with Image.open(path) as image: rgb = np.asarray(image.convert("RGB"))
                    x1, y1, x2, y2 = box
                    crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
                    if crop_array_hash(crop) != digest: raise ValueError("RGB crop hash changed")
                    crops.append(crop)
                recognized = recognizer.recognize_candidate_batch(crops)
                if len(recognized) != len(batch): raise RuntimeError("Recognizer output count mismatch")
                for (digest, _), prediction in zip(batch, recognized):
                    record = cache_record(prediction, digest, recognizer.config)
                    texts[digest] = cached_text(record, digest)
                    _json(cache / (digest + ".json"), record)
                print(f"[PARSeq] {min(start + len(batch), len(jobs))}/{len(jobs)} unique crops", flush=True)
        except Exception as error:
            _json(output / "ocr_failure.json", {"error_type": type(error).__name__, "message": str(error), "protocol": protocol,
                "oom": isinstance(error, torch.cuda.OutOfMemoryError),
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
            raise
        finally:
            if recognizer is not None: recognizer.release()
    results = {}
    for branch, samples in inputs.items():
        arms = {branch: []}
        for sample in samples:
            proposals = sample["proposals"]
            subsets = {branch: proposals}
            for arm, candidates in subsets.items():
                predictions = [{**p, "text": texts[p["crop_rgb_sha256"]]} for p in candidates
                               if p["instance_id"] not in sample["ignored_ids"]]
                metrics, matches = score_predictions(predictions, sample["ground_truth"], config["ocr_iou_threshold"])
                arms[arm].append({"sample_id": sample["sample_id"], **metrics, "ground_truth": sample["ground_truth"],
                    "annotation_sha256": sample["annotation_sha256"], "predictions": predictions,
                    "matched_gt_indices": {k: sorted(v) for k, v in matches.items()}})
        for arm, rows in arms.items():
            aggregate = {k: sum(r[k] for r in rows) for k in ("gt_words", "proposals", "geometry_iou_ge_50",
                         "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")}
            if aggregate["gt_words"] != config["expected_validation_words"]: raise ValueError("OCR GT denominator changed")
            for key in ("exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50"):
                aggregate[key + "_recall_all_gt"] = aggregate[key] / aggregate["gt_words"]
            results[arm] = {"aggregate": aggregate, "per_image": rows}
    baseline = "control" if "control" in results else "embedding50_hard"
    control = {r["sample_id"]: r for r in results[baseline]["per_image"]}
    for arm, result in results.items():
        gains = losses = 0
        for row in result["per_image"]:
            old = set(control[row["sample_id"]]["matched_gt_indices"]["exact_casefold_iou_ge_50"])
            new = set(row["matched_gt_indices"]["exact_casefold_iou_ge_50"])
            gains += len(new - old); losses += len(old - new)
        result["exact_casefold_changes_vs_control"] = {"gained": gains, "lost": losses, "net": gains - losses}
    for label, reference in (("reference", "reference_control"), ("embedding50_hard", "embedding50_hard")):
        if reference not in results:
            continue
        base = {r["sample_id"]: r for r in results[reference]["per_image"]}
        for arm, result in results.items():
            gains = losses = 0
            for row in result["per_image"]:
                old = set(base[row["sample_id"]]["matched_gt_indices"]["exact_casefold_iou_ge_50"])
                new = set(row["matched_gt_indices"]["exact_casefold_iou_ge_50"])
                gains += len(new - old); losses += len(old - new)
            result["exact_casefold_changes_vs_" + label] = {"gained": gains, "lost": losses, "net": gains - losses}
    protocol["comparison_baseline"] = baseline
    _json(destination, protocol)
    result = {"status": "SUCCESS", "protocol": protocol, "arms": results, "environment": runtime_environment(),
        "memory": {"device": str(device), "model_set": "PARSeq only", "dtype": "float32", "batch_size": args.batch_size,
            "input_shape": [args.batch_size, 3, 32, 128], "oom": False,
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}}
    _json(output / ("ocr_comparison.json" if args.subset == "all" else args.subset + "_ocr_comparison.json"), result)
    print(json.dumps({k: v["aggregate"] for k, v in results.items()}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("approximate",), required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--subset", choices=("all",), required=True)
    p.add_argument("--word-root", type=Path, required=True)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--parseq-source", type=Path, required=True)
    p.add_argument("--parseq-source-archive", type=Path, required=True)
    p.add_argument("--parseq-checkpoint", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    run(p.parse_args())
