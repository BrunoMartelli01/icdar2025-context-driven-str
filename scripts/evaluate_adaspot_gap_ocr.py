"""Unchanged qualified PARSeq on frozen Gap/control and gap-disabled word crops."""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np
from PIL import Image
import torch

from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _train_gt
from scripts.evaluate_unet_word_research_ocr import cache_record, cached_text, score_predictions
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.run_adaspot_gap import DEFAULT_CONFIG, REPO, read_config
from scripts.train_totaltext_v4_highres_ablation import _json, _split_indices
from segmentation.data import load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking


def compare_matched_contracts(left, right):
    fields = ("resolved_config", "resolved_preprocessing", "validation_ids_sha256", "runtime_inputs",
              "execution_device", "backend", "gpu_name", "training_config", "source_sha256")
    if any(key not in left or key not in right for key in fields):
        raise ValueError("Matched training provenance is incomplete")
    different = [key for key in fields if left[key] != right[key]]
    if different:
        raise ValueError("Matched training contracts differ: " + ", ".join(different))


def aggregate_rows(rows, expected_words):
    keys = ("gt_words", "proposals", "geometry_iou_ge_50", "exact_casefold_iou_ge_50",
            "exact_no_symbols_iou_ge_50")
    totals = {key: sum(row[key] for row in rows) for key in keys}
    if totals["gt_words"] != expected_words:
        raise ValueError("All expected OCR words must remain in the denominator")
    for name, key in (("exact_casefold", "exact_casefold_iou_ge_50"),
                      ("exact_no_symbols", "exact_no_symbols_iou_ge_50")):
        tp, n, gt = totals[key], totals["proposals"], totals["gt_words"]
        totals[name + "_precision"] = tp / n if n else 0.
        totals[name + "_recall"] = tp / gt if gt else 0.
        totals[name + "_hmean"] = 2 * tp / (n + gt) if n + gt else 0.
    return totals


def run(args):
    config, _ = read_config(args.config, args.mode)
    root, output = args.word_root.resolve(), args.run_root.resolve()
    if output == REPO or REPO in output.parents or root == output or root in output.parents:
        raise ValueError("OCR outputs must remain outside repository and dataset")
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["word_manifest_sha256"]:
        raise ValueError("Qualified word manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("This experiment accepts only held-out TRAIN images")
    _, val_ids, split_sha = _split_indices(manifest, config["seed"], config["validation_count"])
    if split_sha != config["validation_ids_sha256"]:
        raise ValueError("Validation split changed")
    if (sha256_file(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256
            or sha256_file(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Qualified PARSeq checkpoint/archive changed")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for PARSeq; no automatic CPU fallback")
    if type(args.batch_size) is not int or args.batch_size < 1:
        raise ValueError("OCR batch size must be positive")
    protocol = {"schema_version": 1, "reproduction_mode": "approximate", "result_class": "approximate reproduction",
        "resolved_config": config, "validation_ids_sha256": split_sha,
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256, "parseq_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "decode_ar": True, "refine_iters": 1, "dtype": "float32", "device": str(device), "batch_size": args.batch_size,
        "matching": "maximum_one_to_one_original_RGB_bbox_IoU>=0.50_and_exact_Unicode_casefold;_punctuation_retained",
        "ignore": "exclude_only_wholly_invalid_native_instances_after_crop_generation_and_OCR",
        "source_sha256": {name: sha256_file(REPO / name) for name in (
            "scripts/evaluate_adaspot_gap_ocr.py", "scripts/evaluate_unet_word_research_ocr.py",
            "scripts/run_adaspot_gap.py", "recognition/parseq.py",
            "scripts/evaluate_totaltext_word_kernel_parseq_val.py", "scripts/evaluate_totaltext_v4_image_frequency_val_parseq.py",
            "scripts/evaluate_totaltext_highres_contour_padding.py", "recognition/scene_text_v1_evaluation.py",
            "recognition/scene_text_datasets.py", "segmentation/evaluate.py")}}
    by_id = {s.sample_id: s for s in manifest.samples}
    expected_ids = {manifest.samples[i].sample_id for i in val_ids}
    cache = output / "parseq_ocr_cache"
    inputs, pending, texts, hashes = {}, {}, {}, {}
    common_contract = None
    for branch in config["branches"]:
        directory = output / branch
        receipt = json.loads((directory / "run_manifest.json").read_text())
        report_path = directory / "validation.json"
        report = json.loads(report_path.read_text())
        if (receipt.get("status") != "SUCCESS" or not receipt.get("training_complete")
                or receipt.get("completed_epochs") != config["epochs"]
                or report.get("run_contract") != receipt.get("run_contract")
                or receipt["run_contract"]["resolved_config"] != config
                or report.get("checkpoint_sha256") != receipt["checkpoint"]["sha256"]
                or sha256_file(directory / "best.pt") != receipt["checkpoint"]["sha256"]
                or len(report["per_image"]) != len(expected_ids)
                or {r["sample_id"] for r in report["per_image"]} != expected_ids):
            raise ValueError("OCR requires complete checksum-bound training/exports: " + branch)
        current_contract = receipt["run_contract"]
        compare_matched_contracts(current_contract, current_contract if common_contract is None else common_contract)
        common_contract = current_contract
        hashes[branch] = sha256_file(report_path)
        inputs[branch] = []
        if branch == "gap":
            inputs["gap_disabled"] = []
        for row in report["per_image"]:
            sample = by_id[row["sample_id"]]
            base = directory / "validation_exports" / sample.sample_id.replace(":", "__")
            path = base / "proposals.json"
            if sha256_file(path) != row["proposals_sha256"]:
                raise ValueError("Exported proposal receipt changed")
            record = json.loads(path.read_text())
            if sha256_file(base / "maps.npz") != record["maps_sha256"] or record["maps_sha256"] != row["maps_sha256"]:
                raise ValueError("Exported prediction maps changed")
            image_path = root / sample.image_path
            if (record["sample_id"] != sample.sample_id or record["image_sha256"] != sample.image_sha256
                    or sha256_file(image_path) != sample.image_sha256):
                raise ValueError("Frozen original RGB checksum changed")
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
            arms = {branch: (record["proposals"], row["ignored_instance_ids"])}
            if branch == "gap":
                arms["gap_disabled"] = (record["gap_disabled_proposals"], row["gap_disabled_ignored_instance_ids"])
            frozen = {}
            for arm, (proposals, ignored) in arms.items():
                values = []
                for proposal in proposals:
                    x1, y1, x2, y2 = proposal["bbox"]
                    if not (0 <= x1 < x2 <= rgb.shape[1] and 0 <= y1 < y2 <= rgb.shape[0]):
                        raise ValueError("Frozen crop exceeds RGB coordinates")
                    digest = crop_array_hash(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
                    values.append({**proposal, "crop_rgb_sha256": digest})
                    cached = cache / (digest + ".json")
                    if cached.exists():
                        texts[digest] = cached_text(json.loads(cached.read_text()), digest)
                    else:
                        pending[digest] = (image_path, proposal["bbox"])
                frozen[arm] = (values, ignored)
            # Annotation access follows freezing of EVERY arm's crop geometry.
            gt, annotation_sha = _train_gt(sample, root, (rgb.shape[1], rgb.shape[0]))
            for arm, (values, ignored) in frozen.items():
                inputs[arm].append({"sample_id": sample.sample_id, "proposals": values, "ignored_ids": ignored,
                    "ground_truth": gt, "annotation_sha256": annotation_sha})
    protocol["validation_report_sha256"] = hashes
    destination = output / "ocr_protocol.json"
    if destination.exists() and json.loads(destination.read_text()) != protocol:
        raise ValueError("OCR protocol changed; preserve outputs and use a new directory")
    _json(destination, protocol)
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    recognizer = None
    try:
        if pending:
            recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=args.parseq_source,
                checkpoint=args.parseq_checkpoint, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                runtime_source_archive=args.parseq_source_archive, device=str(device)))
            jobs = sorted(pending.items())
            for start in range(0, len(jobs), args.batch_size):
                items = jobs[start:start + args.batch_size]
                crops = []
                for digest, (path, box) in items:
                    with Image.open(path) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    x1, y1, x2, y2 = box
                    crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
                    if crop_array_hash(crop) != digest:
                        raise ValueError("RGB crop changed before recognition")
                    crops.append(crop)
                predictions = recognizer.recognize_candidate_batch(crops)
                if len(predictions) != len(items):
                    raise RuntimeError("PARSeq returned an incomplete batch")
                for (digest, _), prediction in zip(items, predictions):
                    value = cache_record(prediction, digest, recognizer.config)
                    texts[digest] = cached_text(value, digest)
                    _json(cache / (digest + ".json"), value)
                print("[PARSeq] {}/{} unique crops".format(min(start + len(items), len(jobs)), len(jobs)), flush=True)
    except Exception as error:
        _json(output / "ocr_failure.json", {"status": "FAILED", "error_type": type(error).__name__,
            "message": str(error), "traceback": traceback.format_exc(), "protocol": protocol,
            "oom": isinstance(error, torch.cuda.OutOfMemoryError),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
        raise
    finally:
        if recognizer is not None:
            recognizer.release()
    results = {}
    for arm, samples in inputs.items():
        rows = []
        for sample in samples:
            predictions = [{**p, "text": texts[p["crop_rgb_sha256"]]} for p in sample["proposals"]
                           if p["instance_id"] not in sample["ignored_ids"]]
            metrics, matches = score_predictions(predictions, sample["ground_truth"])
            rows.append({**metrics, "sample_id": sample["sample_id"], "ground_truth": sample["ground_truth"],
                "annotation_sha256": sample["annotation_sha256"], "predictions": predictions,
                "matched_gt_indices": {k: sorted(v) for k, v in matches.items()}})
        results[arm] = {"aggregate": aggregate_rows(rows, config["expected_validation_words"]), "per_image": rows}
    control = {r["sample_id"]: set(r["matched_gt_indices"]["exact_casefold_iou_ge_50"])
               for r in results["control"]["per_image"]}
    for result in results.values():
        gained = lost = 0
        for row in result["per_image"]:
            current = set(row["matched_gt_indices"]["exact_casefold_iou_ge_50"])
            gained += len(current - control[row["sample_id"]]); lost += len(control[row["sample_id"]] - current)
        result["changes_vs_control"] = {"gained": gained, "lost": lost, "net": gained - lost}
    result = {"status": "SUCCESS", "result_class": "approximate reproduction", "protocol": protocol,
        "arms": results, "memory": {"model_set": "PARSeq only", "dtype": "float32", "device": str(device),
            "batch_size": args.batch_size, "input_shape": [args.batch_size, 3, 32, 128], "oom": False,
            "new_crops_recognized": len(pending), "cache_reused_crops": len(texts) - len(pending),
            "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}}
    _json(output / "ocr_comparison.json", result)
    lines = ["# AdaSpot Gap — approximate reproduction", "", "Exploratory reused TRAIN holdout; local bbox metric.", "",
             "| Arm | Correct / GT | Proposals | Precision | Recall | Hmean | Gained / lost vs control |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, values in results.items():
        a, c = values["aggregate"], values["changes_vs_control"]
        lines.append("| {} | {} / {} | {} | {:.2%} | {:.2%} | {:.2%} | {} / {} |".format(arm,
            a["exact_casefold_iou_ge_50"], a["gt_words"], a["proposals"], a["exact_casefold_precision"],
            a["exact_casefold_recall"], a["exact_casefold_hmean"], c["gained"], c["lost"]))
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({name: v["aggregate"] for name, v in results.items()}, indent=2), flush=True)
    return result


def parser():
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
    return p


if __name__ == "__main__":
    run(parser().parse_args())
