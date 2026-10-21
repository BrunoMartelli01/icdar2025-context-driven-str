"""Frozen letterbox probability-threshold diagnostic on TRAIN validation only.

One GPU pass stores float32 native probabilities; subsequent runs reuse them.
PARSeq is run only for previously unseen RGB crops. No TEST data is loaded.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer,
)
from scripts.evaluate_totaltext_highres_contour_padding import (
    contour_proposals, crop_array_hash,
)
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import (
    _correct, _geometry_match, _image, _model_and_run, _tensor, _train_gt,
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
CONFIG_PATH = REPO / "configs/approximate/totaltext_letterbox_threshold_sweep_train_v1.json"
EXPECTED_THRESHOLDS = (0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65)
METRICS = ("geometry_iou_ge_25", "geometry_iou_ge_50", "geometry_iou_ge_75",
           "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")


def threshold_key(value: float) -> str:
    if value not in EXPECTED_THRESHOLDS:
        raise ValueError("Threshold is outside the predeclared diagnostic")
    return f"{value:.2f}"


def _config() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (config.get("classification") != "PRACTICAL_APPROXIMATION"
        or config.get("reproduction_mode") != "approximate"
        or config.get("split") != "TRAIN_heldout_125"
        or config.get("variant") != "letterbox"
        or tuple(config.get("thresholds", ())) != EXPECTED_THRESHOLDS
        or config.get("input_size") != [512, 512]
        or config.get("native_mask_size") != [256, 256]
        or config.get("padding_native_pixels") != 0
        or config.get("area_filter") != "disabled"
        or config.get("maximum_proposals") != "disabled"
        or config.get("matching") != "one_to_one_bbox_iou_ge_0_5"):
        raise ValueError("Frozen TRAIN threshold diagnostic changed")
    return config


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temp = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _load_probability(path: Path, metadata: dict, *, sample, protocol_sha: str) -> np.ndarray:
    if (metadata.get("sample_id") != sample.sample_id
        or metadata.get("image_sha256") != sample.image_sha256
        or metadata.get("protocol_sha256") != protocol_sha
        or sha256_file(path) != metadata.get("file_sha256")):
        raise ValueError(f"Probability cache provenance mismatch: {path}")
    probability = np.load(path, allow_pickle=False)
    if (probability.shape != (256, 256) or probability.dtype != np.float32
        or not np.isfinite(probability).all()
        or probability.min() < 0 or probability.max() > 1
        or sha256(probability.tobytes()).hexdigest() != metadata.get("array_sha256")):
        raise ValueError(f"Invalid cached native probabilities: {path}")
    return probability


def _freeze_probability(path: Path, metadata_path: Path, probability: np.ndarray,
                        *, sample, protocol_sha: str) -> None:
    if probability.shape != (256, 256) or probability.dtype != np.float32:
        raise ValueError("Expected float32 native 256x256 probabilities")
    encoded = BytesIO()
    np.save(encoded, probability, allow_pickle=False)
    _atomic_bytes(path, encoded.getvalue())
    _atomic_json(metadata_path, {
        "schema_version": 1, "sample_id": sample.sample_id,
        "image_sha256": sample.image_sha256, "protocol_sha256": protocol_sha,
        "file_sha256": sha256_file(path),
        "array_sha256": sha256(probability.tobytes()).hexdigest(),
    })


def _read_ocr(path: Path, crop_sha: str) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("crop_rgb_sha256") != crop_sha
        or value.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
        or value.get("parseq_source_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or not isinstance(value.get("raw_text"), str)):
        raise ValueError(f"PARSeq crop cache provenance mismatch: {path}")
    return value["raw_text"]


def _matched(predictions: list[dict], gt: list[dict]) -> dict[str, set[int]]:
    return {
        "geometry_iou_ge_25": _geometry_match(predictions, gt, 0.25),
        "geometry_iou_ge_50": _geometry_match(predictions, gt, 0.5),
        "geometry_iou_ge_75": _geometry_match(predictions, gt, 0.75),
        "exact_casefold_iou_ge_50": _correct(predictions, gt, normalize=False),
        "exact_no_symbols_iou_ge_50": _correct(predictions, gt, normalize=True),
    }


def _overlap(correct: dict[str, set[str]]) -> dict:
    """Auditable GT-word identity intersections, not just aggregate counts."""
    keys = tuple(sorted(correct, key=float))
    baseline = correct["0.50"]
    union = set().union(*correct.values())
    return {
        "correct_any_threshold": len(union),
        "gain_any_threshold_vs_0_50": len(union - baseline),
        "by_threshold": {
            key: {"correct": len(correct[key]),
                  "new_vs_0_50": len(correct[key] - baseline),
                  "lost_vs_0_50": len(baseline - correct[key]),
                  "unique_to_this_threshold": len(correct[key] - set().union(
                      *(correct[other] for other in keys if other != key)))}
            for key in keys},
        "intersection_matrix": {
            key: {other: len(correct[key] & correct[other]) for other in keys}
            for key in keys},
        "new_word_ids_vs_0_50": {
            key: sorted(correct[key] - baseline) for key in keys},
        "lost_word_ids_vs_0_50": {
            key: sorted(baseline - correct[key]) for key in keys},
    }


def run(args) -> dict:
    config = _config()
    training = train_config()
    root = args.target_root.resolve()
    manifest_path = root / training["train_manifest"]
    if sha256_file(manifest_path) != training["train_manifest_sha256"]:
        raise ValueError("Qualified Total-Text TRAIN manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Threshold sweep permits TRAIN only")
    _, val_ids, val_sha = _split_indices(manifest, training["training"]["seed"], 125)
    if len(val_ids) != 125 or len(set(val_ids)) != 125:
        raise ValueError("Expected frozen 125-image TRAIN validation")
    reference_dir = args.training_runs.resolve() / "validation-pad0-parseq-v1"
    reference_path = reference_dir / "results.json"
    if sha256_file(reference_path) != config["frozen_half_threshold_report_sha256"]:
        raise ValueError("Frozen 0.5 validation report changed")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference["protocol"]["validation_ids_sha256"] != val_sha
        or reference["variants"]["letterbox"]["aggregate"]["gt_words"] != 983):
        raise ValueError("Frozen 0.5 reference is not this TRAIN split")
    run_path = args.training_runs.resolve() / "letterbox/run_manifest.json"
    trained = json.loads(run_path.read_text(encoding="utf-8"))
    checkpoint = args.training_runs.resolve() / "letterbox/best.pt"
    if (trained.get("status") != "SUCCESS" or not trained.get("training_complete")
        or trained.get("checkpoint", {}).get("sha256") != config["checkpoint_sha256"]
        or sha256_file(checkpoint) != config["checkpoint_sha256"]):
        raise ValueError("Frozen letterbox checkpoint changed")
    if (sha256_file(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256
        or sha256_file(args.parseq_source_archive.resolve()) != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or not args.parseq_source.resolve().is_dir()):
        raise ValueError("Qualified PARSeq artifact changed or missing")
    output = args.output_dir.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("Diagnostic outputs must be outside the repository")
    protocol = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
                "result_class": config["result_class"], "split": config["split"],
                "validation_ids_sha256": val_sha, "train_manifest_sha256": sha256_file(manifest_path),
                "config_sha256": sha256_file(CONFIG_PATH),
                "training_config_sha256": sha256_file(TRAIN_CONFIG_PATH),
                "training_run_manifest_sha256": sha256_file(run_path),
                "checkpoint_sha256": config["checkpoint_sha256"],
                "half_threshold_reference_sha256": config["frozen_half_threshold_report_sha256"],
                "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                "thresholds": list(EXPECTED_THRESHOLDS),
                "mask": "native256 sigmoid probability >= threshold; letterbox canvas excluded",
                "crop": "RETR_EXTERNAL contours; no area/top10; pad0; inverse-letterbox RGB bbox",
                "ground_truth_use": config["ground_truth_use"], "device": args.device,
                "parseq_batch_size": args.parseq_batch_size}
    protocol_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing output belongs to another threshold protocol")
    elif not args.preflight_only:
        _atomic_json(protocol_path, protocol)
    reference_shards = {}
    for index in val_ids:
        sample = manifest.samples[index]
        shard_path = reference_dir / "letterbox/samples" / f"{sample.sample_id.replace(':', '__')}.json"
        shard = json.loads(shard_path.read_text(encoding="utf-8"))
        if (shard["sample_id"] != sample.sample_id
            or shard["image_sha256"] != sample.image_sha256
            or shard["checkpoint_sha256"] != config["checkpoint_sha256"]):
            raise ValueError(f"Frozen 0.5 sample provenance mismatch: {sample.sample_id}")
        reference_shards[sample.sample_id] = shard
    if args.preflight_only:
        summary = {"status": "PREFLIGHT_PASS", "split": "TRAIN_heldout_125",
                   "images": len(val_ids), "gt_words": 983,
                   "thresholds": list(EXPECTED_THRESHOLDS),
                   "checkpoint_sha256": config["checkpoint_sha256"]}
        print(json.dumps(summary, sort_keys=True), flush=True)
        return summary
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on requested device")
    device = torch.device(args.device)
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    by_id = {manifest.samples[index].sample_id: manifest.samples[index] for index in val_ids}
    probability_maps = {}
    pending = []
    for index in val_ids:
        sample = manifest.samples[index]
        key = sample.sample_id.replace(":", "__")
        path = output / "probabilities" / f"{key}.npy"
        metadata_path = output / "probabilities" / f"{key}.json"
        if path.exists() and metadata_path.exists():
            probability_maps[sample.sample_id] = _load_probability(
                path, json.loads(metadata_path.read_text(encoding="utf-8")),
                sample=sample, protocol_sha=protocol_sha)
        elif path.exists() or metadata_path.exists():
            # An interrupted write is not trusted. Recompute this image from
            # the pinned checkpoint, then replace only this diagnostic shard.
            print(f"[probabilities] incomplete shard will be regenerated: {sample.sample_id}",
                  flush=True)
            pending.append(sample)
        else:
            pending.append(sample)
    if pending:
        model, transform, model_report = _model_and_run("letterbox", args, training, val_sha)
        if model_report["checkpoint"]["sha256"] != config["checkpoint_sha256"]:
            raise ValueError("Loaded checkpoint differs from frozen selection")
        model.eval().to(device)
        try:
            for pos, sample in enumerate(pending, 1):
                rgb = _image(sample, root)
                with torch.inference_mode():
                    logits = model(_tensor(rgb, transform).to(device))
                if tuple(logits.shape) != (1, 1, 256, 256):
                    raise RuntimeError("Unexpected native letterbox logits")
                probability = np.ascontiguousarray(torch.sigmoid(logits[0, 0]).float().cpu().numpy())
                del logits
                probability *= letterbox_canvas_mask((rgb.shape[1], rgb.shape[0])).numpy()
                key = sample.sample_id.replace(":", "__")
                _freeze_probability(output / "probabilities" / f"{key}.npy",
                                    output / "probabilities" / f"{key}.json",
                                    probability, sample=sample, protocol_sha=protocol_sha)
                probability_maps[sample.sample_id] = probability
                if pos % 10 == 0 or pos == len(pending):
                    print(f"[probabilities] {pos}/{len(pending)} new; {len(probability_maps)}/125 total", flush=True)
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        print("[probabilities] verified cache 125/125", flush=True)
    shards = {threshold_key(t): {} for t in EXPECTED_THRESHOLDS}
    for position, index in enumerate(val_ids, 1):
        sample = manifest.samples[index]
        rgb = _image(sample, root)
        gt, annotation_sha = _train_gt(sample, root, (rgb.shape[1], rgb.shape[0]))
        if annotation_sha != reference_shards[sample.sample_id]["annotation_sha256"]:
            raise ValueError(f"TRAIN GT annotation changed: {sample.sample_id}")
        probability = probability_maps[sample.sample_id]
        for threshold in EXPECTED_THRESHOLDS:
            key = threshold_key(threshold)
            native = np.ascontiguousarray((probability >= threshold).astype(np.uint8))
            if key == "0.50" and sha256(native.tobytes()).hexdigest() != reference_shards[sample.sample_id]["native_mask_array_sha256"]:
                raise ValueError(f"0.5 native mask differs from frozen validation: {sample.sample_id}")
            contours = contour_proposals(native, rgb.shape[:2], [0])["0"]
            for proposal in contours["proposals"]:
                proposal["bbox"] = remap_letterbox_native_box(proposal["padded_bbox_native"],
                                                                (rgb.shape[1], rgb.shape[0]))
                x1, y1, x2, y2 = proposal["bbox"]
                proposal["crop_rgb_sha256"] = crop_array_hash(
                    np.ascontiguousarray(rgb[y1:y2, x1:x2]))
            shards[key][sample.sample_id] = {"ground_truth": gt,
                                               "proposals": contours["proposals"],
                                               "raw_contours": contours["raw_contours"]}
        if position % 25 == 0 or position == len(val_ids):
            print(f"[contours] {position}/125", flush=True)
    pending_ocr = []
    seen = set()
    for key in shards:
        for index in val_ids:
            sample_id = manifest.samples[index].sample_id
            for proposal in shards[key][sample_id]["proposals"]:
                crop_sha = proposal["crop_rgb_sha256"]
                if crop_sha in seen:
                    continue
                seen.add(crop_sha)
                own = output / "ocr" / f"{crop_sha}.json"
                prior = reference_dir / "ocr" / f"{crop_sha}.json"
                if own.exists():
                    _read_ocr(own, crop_sha)
                elif prior.exists():
                    _read_ocr(prior, crop_sha)
                else:
                    pending_ocr.append((sample_id, proposal["bbox"], crop_sha))
    print(f"[PARSeq] unique_crops={len(seen)} pending={len(pending_ocr)}", flush=True)
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
                        raise ValueError("Crop pixels changed after contour freeze")
                    crops.append(crop)
                values = recognizer.recognize_candidate_batch(crops)
                if len(values) != len(batch):
                    raise RuntimeError("PARSeq returned wrong batch length")
                for (_, _, crop_sha), value in zip(batch, values):
                    _atomic_json(output / "ocr" / f"{crop_sha}.json", {
                        "schema_version": 1, "crop_rgb_sha256": crop_sha,
                        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                        **value.to_dict()})
                print(f"[PARSeq] {min(start + len(batch), len(pending_ocr))}/{len(pending_ocr)}", flush=True)
        finally:
            recognizer.release()
    report = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "result_class": config["result_class"], "protocol": protocol,
              "images": len(val_ids), "gt_words": 983, "thresholds": {}, "overlap": {}}
    correct_by_metric = {metric: {} for metric in
                         ("exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")}
    reference_rows = {row["sample_id"]: row for row in reference["variants"]["letterbox"]["per_image"]}
    for key in shards:
        totals = {metric: 0 for metric in METRICS}
        totals.update({"gt_words": 0, "proposals": 0})
        rows = []
        correct = {metric: set() for metric in correct_by_metric}
        for index in val_ids:
            sample_id = manifest.samples[index].sample_id
            shard = shards[key][sample_id]
            predictions = []
            for proposal in shard["proposals"]:
                crop_sha = proposal["crop_rgb_sha256"]
                own = output / "ocr" / f"{crop_sha}.json"
                prior = reference_dir / "ocr" / f"{crop_sha}.json"
                text = _read_ocr(own if own.exists() else prior, crop_sha)
                predictions.append({"bbox": proposal["bbox"], "text": text,
                                    "crop_rgb_sha256": crop_sha})
            gt = shard["ground_truth"]
            matched = _matched(predictions, gt)
            totals["gt_words"] += len(gt)
            totals["proposals"] += len(predictions)
            for metric, ids in matched.items():
                totals[metric] += len(ids)
                if metric in correct:
                    correct[metric].update(f"{sample_id}#{i}" for i in ids)
            row = {"sample_id": sample_id, "ground_truth": gt,
                   "predictions": predictions,
                   "matched_gt_indices": {metric: sorted(ids) for metric, ids in matched.items()}}
            if key == "0.50":
                frozen = reference_rows[sample_id]
                for metric in METRICS:
                    if row["matched_gt_indices"][metric] != frozen["matched_gt_indices"][metric]:
                        raise ValueError(f"0.5 GT matching differs from frozen validation: {sample_id} {metric}")
            rows.append(row)
        if totals["gt_words"] != 983:
            raise ValueError("TRAIN validation GT denominator changed")
        if key == "0.50":
            frozen = reference["variants"]["letterbox"]["aggregate"]
            for metric in METRICS:
                if totals[metric] != frozen[metric]:
                    raise ValueError(f"0.5 aggregate differs from frozen validation: {metric}")
            if totals["proposals"] != frozen["proposals"]:
                raise ValueError("0.5 proposal count differs from frozen validation")
        totals["geometry_recall_iou_ge_50"] = totals["geometry_iou_ge_50"] / 983
        totals["exact_casefold_rate_all_gt"] = totals["exact_casefold_iou_ge_50"] / 983
        totals["exact_no_symbols_rate_all_gt"] = totals["exact_no_symbols_iou_ge_50"] / 983
        report["thresholds"][key] = {"aggregate": totals, "per_image": rows}
        for metric in correct_by_metric:
            correct_by_metric[metric][key] = correct[metric]
        print(f"[threshold {key}] geometry={totals['geometry_iou_ge_50']} "
              f"exact={totals['exact_casefold_iou_ge_50']} "
              f"no_symbols={totals['exact_no_symbols_iou_ge_50']}", flush=True)
    report["overlap"] = {metric: _overlap(by_threshold)
                         for metric, by_threshold in correct_by_metric.items()}
    result_path = output / "results.json"
    if result_path.exists():
        if json.loads(result_path.read_text(encoding="utf-8")) != report:
            raise ValueError("Existing threshold result differs; refusing overwrite")
    else:
        _atomic_json(result_path, report)
    print(json.dumps({"result": str(result_path), "sha256": sha256_file(result_path),
                      "summary": {key: value["aggregate"] for key, value in report["thresholds"].items()},
                      "overlap": report["overlap"]}, indent=2, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
