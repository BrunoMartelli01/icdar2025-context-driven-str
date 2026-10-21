#!/usr/bin/env python3
"""One-command Total-Text TRAIN crop/PARSeq/CLIP battery (approximate).

Frozen probability/OCR caches are read-only inputs. GT annotation files are
opened only for post-hoc evaluation and the explicitly marked oracle. The
frozen aggregate report is read earlier solely to verify baseline crop identity.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

import numpy as np
import torch

from recognition.crop_clip_battery import (
    generate_totaltext_candidates, select_per_anchor, totaltext_oracle_one_per_anchor,
)
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer,
)
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import _load_probability
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct, _image, _train_gt
from scripts.crop_clip_runtime import (
    _cuda_peak, array_sha, cached, crop_from_row, run_clip, sha, write_json,
)
from scripts.train_totaltext_v4_highres_ablation import _split_indices
from scripts.train_totaltext_v4_image_frequency_ablation import _config as train_config
from segmentation.data import load_segmentation_manifest


REPO = Path(__file__).resolve().parents[1]


def validate_config(config: dict) -> None:
    if (config.get("reproduction_mode") != "approximate"
        or config.get("classification") != "PRACTICAL_APPROXIMATION"
        or config.get("dataset") != "total_text"
        or config.get("split") != "TRAIN_heldout_125"
        or config.get("baseline_threshold") != 0.5
        or config.get("baseline_padding_native_pixels") != 0
        or config.get("thresholds") != [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        or config.get("padding_native_pixels") != [0, 2, 4, 8]
        or config.get("min_anchor_tight_native_iou") != 0.3
        or config.get("clip_model") != "openai/clip-vit-base-patch32"
        or config.get("clip_prompts") != {
            "written": 'a close-up photo of the written word "{text}"', "raw": "{text}"}
        or config.get("clip_margins") != [0.0, 0.02]
        or config.get("area_filter") != "disabled"
        or config.get("maximum_proposals") != "disabled"):
        raise ValueError("Total-Text crop/CLIP protocol differs from predeclared config")


def _key(sample_id: str) -> str:
    return sample_id.replace(":", "__")


def _ocr_record(path: Path, crop_sha: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("crop_rgb_sha256") != crop_sha
        or value.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
        or value.get("parseq_source_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or not isinstance(value.get("raw_text"), str)):
        raise RuntimeError(f"PARSeq crop cache provenance mismatch: {path}")
    return value


def _cached_crop_reading(crop_sha: str, primary: Path,
                         baseline: Path | None) -> dict | None:
    for root in (primary, baseline):
        if root is not None:
            path = root / f"{crop_sha}.json"
            if path.exists():
                return _ocr_record(path, crop_sha)
    return None


def _cached_crop_path(crop_sha: str, primary: Path,
                      baseline: Path | None) -> Path | None:
    for root in (primary, baseline):
        if root is not None:
            path = root / f"{crop_sha}.json"
            if path.exists():
                return path
    return None


def serializable_selections(selections: dict[str, dict[int, str]]) -> dict[str, dict[str, str]]:
    """Preserve anchor IDs through JSON write/read and resumable comparisons."""
    return {arm: {str(anchor): candidate_id for anchor, candidate_id in choices.items()}
            for arm, choices in selections.items()}


def prepare(sample: Any, *, data_root: Path, cache_root: Path,
            output_dir: Path, config: dict, plan_sha: str,
            cache_protocol_sha: str) -> dict:
    path = output_dir / "prepared" / f"{_key(sample.sample_id)}.json"
    image = _image(sample, data_root)
    sidecar = cache_root / "probabilities" / f"{_key(sample.sample_id)}.json"
    probability = _load_probability(
        cache_root / "probabilities" / f"{_key(sample.sample_id)}.npy",
        json.loads(sidecar.read_text(encoding="utf-8")),
        sample=sample, protocol_sha=cache_protocol_sha)
    contract = {"plan_sha256": plan_sha, "sample_id": sample.sample_id,
                "image_sha256": sample.image_sha256,
                "probability_sidecar_sha256": sha(sidecar)}
    previous = cached(path, contract)
    if previous is not None:
        if len({r["candidate_id"] for r in previous["candidates"]}) != len(previous["candidates"]):
            raise RuntimeError("duplicate prepared candidate IDs")
        return previous
    rows, statistics = generate_totaltext_candidates(
        probability, image, thresholds=config["thresholds"],
        pads=config["padding_native_pixels"],
        min_anchor_iou=config["min_anchor_tight_native_iou"])
    for row in rows:
        row["candidate_id"] = f"{_key(sample.sample_id)}:{row['candidate_id']}"
        x1, y1, x2, y2 = row["bbox"]
        crop = np.ascontiguousarray(image[y1:y2, x1:x2])
        row["crop_shape"] = list(crop.shape)
        row["crop_array_sha256"] = array_sha(crop)
        row["crop_rgb_sha256"] = crop_array_hash(crop)
    output = {"contract": contract, "reproduction_mode": "approximate",
              "claim_label": "approximate reproduction; Total-Text TRAIN diagnostic intermediate",
              "candidates": rows, "statistics": statistics}
    write_json(path, output)
    return output


def verify_prepared_baseline(samples: list[Any], prepared: dict[str, dict],
                             frozen_report: dict) -> None:
    """Check every pad-0 threshold-0.5 box and crop hash before model scoring."""
    frozen = {row["sample_id"]: row for row in
              frozen_report["thresholds"]["0.50"]["per_image"]}
    if len(frozen) != 125:
        raise RuntimeError("frozen Total-Text TRAIN reference is incomplete")
    for sample in samples:
        actual = [{"bbox": row["bbox"], "crop_rgb_sha256": row["crop_rgb_sha256"]}
                  for row in prepared[sample.sample_id]["candidates"] if row["baseline"]]
        expected = [{"bbox": row["bbox"], "crop_rgb_sha256": row["crop_rgb_sha256"]}
                    for row in frozen[sample.sample_id]["predictions"]]
        if actual != expected:
            raise RuntimeError(f"frozen Total-Text baseline geometry changed: {sample.sample_id}")


def run_parseq(samples: list[Any], prepared: dict[str, dict], *,
               data_root: Path, cache_root: Path, baseline_ocr_cache: Path,
               output_dir: Path,
               source: Path, archive: Path, checkpoint: Path,
               device: str, batch_size: int) -> tuple[dict, dict]:
    if (sha(checkpoint) != PARSEQ_CHECKPOINT_SHA256
        or sha(archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256 or not source.is_dir()):
        raise RuntimeError("qualified PARSeq artifacts missing or changed")
    output = {}
    recognizer = None
    peaks = []
    try:
        for position, sample in enumerate(samples, 1):
            rows = prepared[sample.sample_id]["candidates"]
            contract = {"prepared_sha256": sha(output_dir / "prepared" / f"{_key(sample.sample_id)}.json"),
                        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                        "parseq_adapter_sha256": sha(REPO / "recognition/parseq.py"),
                        "device": device, "batch_size": batch_size, "precision": "float32"}
            shard_path = output_dir / "ocr" / f"{_key(sample.sample_id)}.json"
            shard = cached(shard_path, contract)
            computed_now = 0
            if shard is None:
                image = _image(sample, data_root)
                pending = []
                readings = {}
                for row in rows:
                    crop_sha = row["crop_rgb_sha256"]
                    own = output_dir / "ocr_by_crop" / f"{crop_sha}.json"
                    prior = _cached_crop_reading(crop_sha, cache_root / "ocr", baseline_ocr_cache)
                    if own.exists() or prior is not None:
                        readings[crop_sha] = _ocr_record(own, crop_sha) if own.exists() else prior
                    elif crop_sha not in {item["crop_rgb_sha256"] for item in pending}:
                        pending.append(row)
                if pending and recognizer is None:
                    if device.startswith("cuda"):
                        torch.cuda.reset_peak_memory_stats(device)
                    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                        runtime_source=source, runtime_source_archive=archive,
                        runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                        checkpoint=checkpoint, device=device))
                for start in range(0, len(pending), batch_size):
                    batch = pending[start:start + batch_size]
                    values = recognizer.recognize_candidate_batch(
                        [crop_from_row(image, row) for row in batch])
                    if len(values) != len(batch):
                        raise RuntimeError("PARSeq returned incomplete batch")
                    for row, value in zip(batch, values):
                        crop_sha = row["crop_rgb_sha256"]
                        record = {"schema_version": 1, "crop_rgb_sha256": crop_sha,
                                  "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                                  "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                                  **value.to_dict()}
                        write_json(output_dir / "ocr_by_crop" / f"{crop_sha}.json", record)
                        readings[crop_sha] = record
                        computed_now += 1
                shard = {"contract": contract, "reproduction_mode": "approximate",
                         "claim_label": "approximate reproduction; TRAIN diagnostic intermediate",
                         "rows": [{"candidate_id": row["candidate_id"],
                                   "raw_text": readings[row["crop_rgb_sha256"]]["raw_text"],
                                   "mean_token_log_probability": readings[row["crop_rgb_sha256"]].get(
                                       "mean_token_log_probability"),
                                   "sequence_confidence_native": readings[row["crop_rgb_sha256"]].get(
                                       "sequence_confidence_native")}
                                  for row in rows], "cuda_peak": _cuda_peak(device),
                         "reused_ocr_crops": len(rows) - len(pending),
                         "new_ocr_crops": len(pending)}
                write_json(shard_path, shard)
            if [row["candidate_id"] for row in shard["rows"]] != [row["candidate_id"] for row in rows]:
                raise RuntimeError("PARSeq shard candidate ordering mismatch")
            if device.startswith("cuda") and shard.get("cuda_peak", {}).get("peak_reserved_bytes") is None:
                raise RuntimeError("PARSeq CUDA shard lacks memory peak")
            output[sample.sample_id] = {row["candidate_id"]: row for row in shard["rows"]}
            peaks.append(shard.get("cuda_peak", {}))
            print(f"[PARSeq] {position}/{len(samples)} {sample.sample_id} "
                  f"computed_now={computed_now} cached_in_shard={shard['new_ocr_crops']}", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
    return output, {key: max((peak[key] for peak in peaks if peak.get(key) is not None),
                             default=None) for key in ("peak_allocated_bytes", "peak_reserved_bytes")}


def cached_threshold_audit(samples: list[Any], prepared: dict[str, dict], *,
                           data_root: Path, probability_cache: Path,
                           baseline_ocr_cache: Path, frozen_report: dict) -> dict:
    """Real Total-Text threshold-only control using existing PARSeq crop caches."""
    frozen = {row["sample_id"]: row for row in
              frozen_report["thresholds"]["0.50"]["per_image"]}
    summary = {arm: {"exact": 0, "gt_words": 0, "recovered_vs_baseline": 0,
                     "damaged_vs_baseline": 0, "changed_anchors": 0}
               for arm in ("baseline_pad0_threshold050", "parseq_threshold",
                           "oracle_one_per_anchor_threshold")}
    per_image = []
    for sample in samples:
        candidates = []
        candidate_ocr = []
        for row in prepared[sample.sample_id]["candidates"]:
            if not any(origin["padding_native_pixels"] == 0 for origin in row["origins"]):
                continue
            record_path = _cached_crop_path(row["crop_rgb_sha256"],
                                            probability_cache / "ocr", baseline_ocr_cache)
            if record_path is None:
                raise RuntimeError(f"missing frozen threshold OCR crop {row['crop_rgb_sha256']}")
            record = _ocr_record(record_path, row["crop_rgb_sha256"])
            candidate_ocr.append({"candidate_id": row["candidate_id"],
                                  "crop_rgb_sha256": row["crop_rgb_sha256"],
                                  "raw_text": record["raw_text"],
                                  "mean_token_log_probability": record.get("mean_token_log_probability"),
                                  "ocr_record_sha256": sha(record_path)})
            candidates.append({**row, "ocr": {"raw_text": record["raw_text"]},
                               "scores": {"parseq": record["mean_token_log_probability"]
                                          if record.get("mean_token_log_probability") is not None else -1e9}})
        baseline = {row["anchor_rank"]: row["candidate_id"]
                    for row in candidates if row["baseline"]}
        selections = {"baseline_pad0_threshold050": baseline,
                      "parseq_threshold": select_per_anchor(candidates, score="parseq",
                                                            scope="threshold")}
        image = _image(sample, data_root)
        gt, annotation_sha = _train_gt(sample, data_root, (image.shape[1], image.shape[0]))
        selections["oracle_one_per_anchor_threshold"] = totaltext_oracle_one_per_anchor(
            candidates, gt)
        by_id = {row["candidate_id"]: row for row in candidates}
        matched = {}
        selected_predictions = {}
        for arm, choices in selections.items():
            predictions = [{"bbox": by_id[choices[anchor]]["bbox"],
                            "text": by_id[choices[anchor]]["ocr"]["raw_text"],
                            "candidate_id": choices[anchor]}
                           for anchor in sorted(choices)]
            selected_predictions[arm] = predictions
            matched[arm] = _correct(predictions, gt, normalize=False)
        if sorted(matched["baseline_pad0_threshold050"]) != \
                frozen[sample.sample_id]["matched_gt_indices"]["exact_casefold_iou_ge_50"]:
            raise RuntimeError(f"frozen Total-Text baseline OCR changed: {sample.sample_id}")
        base_ids = matched["baseline_pad0_threshold050"]
        for arm, ids in matched.items():
            entry = summary[arm]
            entry["exact"] += len(ids)
            entry["gt_words"] += len(gt)
            entry["recovered_vs_baseline"] += len(ids - base_ids)
            entry["damaged_vs_baseline"] += len(base_ids - ids)
            entry["changed_anchors"] += sum(value != baseline[anchor]
                                             for anchor, value in selections[arm].items())
        per_image.append({"sample_id": sample.sample_id, "ground_truth": gt,
                          "annotation_sha256": annotation_sha,
                          "candidate_ocr": candidate_ocr,
                          "selection": serializable_selections(selections),
                          "selected_predictions": selected_predictions,
                          "matched_gt_indices": {arm: sorted(ids) for arm, ids in matched.items()}})
    if len(samples) == 125 and (summary["baseline_pad0_threshold050"]["exact"] != 368
                                or summary["baseline_pad0_threshold050"]["gt_words"] != 983):
        raise RuntimeError("frozen Total-Text TRAIN aggregate changed")
    for entry in summary.values():
        entry["exact_rate"] = entry["exact"] / entry["gt_words"]
        entry["net_vs_baseline"] = entry["recovered_vs_baseline"] - entry["damaged_vs_baseline"]
    return {"schema_version": 1, "dataset": "total_text", "split": "TRAIN_heldout_125",
            "result_class": "SMOKE_TEST" if len(samples) < 125 else "APPROXIMATE_TRAIN_DIAGNOSTIC",
            "claim_label": "approximate reproduction; cached threshold-only Total-Text TRAIN diagnostic",
            "summary": summary, "per_image": per_image, "gt_used_in_selection": False,
            "oracle_uses_gt": True, "table_5_claim": False}


def evaluate(samples: list[Any], prepared: dict, ocr: dict, clip: dict, *,
             data_root: Path, config: dict, frozen_report: dict) -> dict:
    frozen_rows = {row["sample_id"]: row for row in frozen_report["thresholds"]["0.50"]["per_image"]}
    if len(frozen_rows) != 125:
        raise RuntimeError("frozen Total-Text TRAIN reference is incomplete")
    per_image = []
    aggregate: dict[str, dict[str, int]] = {}
    for sample in samples:
        rows = []
        for row in prepared[sample.sample_id]["candidates"]:
            reading = ocr[sample.sample_id][row["candidate_id"]]
            score = {**clip[sample.sample_id][row["candidate_id"]],
                     "parseq": reading["mean_token_log_probability"]
                     if reading["mean_token_log_probability"] is not None else -1e9}
            rows.append({**row, "ocr": reading, "scores": score})
        baseline = {row["anchor_rank"]: row["candidate_id"] for row in rows if row["baseline"]}
        selections = {"baseline_pad0_threshold050": baseline}
        for scope in ("padding", "threshold", "all"):
            selections[f"parseq_{scope}"] = select_per_anchor(rows, score="parseq", scope=scope)
            for prompt in ("written", "raw"):
                for margin in config["clip_margins"]:
                    selections[f"clip_{prompt}_{scope}_m{margin:g}"] = select_per_anchor(
                        rows, score=prompt, scope=scope, margin=margin)
        selections["clip_shuffled_control"] = select_per_anchor(
            rows, score="shuffled_written")
        image = _image(sample, data_root)
        gt, annotation_sha = _train_gt(sample, data_root, (image.shape[1], image.shape[0]))
        selections["oracle_one_per_anchor"] = totaltext_oracle_one_per_anchor(rows, gt)
        by_id = {row["candidate_id"]: row for row in rows}
        matched = {}
        for arm, choices in selections.items():
            predictions = [{"bbox": by_id[choices[anchor]]["bbox"],
                            "text": by_id[choices[anchor]]["ocr"]["raw_text"]}
                           for anchor in sorted(choices)]
            matched[arm] = _correct(predictions, gt, normalize=False)
        reference = frozen_rows[sample.sample_id]
        frozen_predictions = reference["predictions"]
        actual_baseline = [{"bbox": row["bbox"], "text": row["ocr"]["raw_text"],
                            "crop_rgb_sha256": row["crop_rgb_sha256"]}
                           for row in rows if row["baseline"]]
        if actual_baseline != frozen_predictions or sorted(matched["baseline_pad0_threshold050"]) != \
                reference["matched_gt_indices"]["exact_casefold_iou_ge_50"]:
            raise RuntimeError(f"frozen Total-Text baseline changed: {sample.sample_id}")
        per_image.append({"sample_id": sample.sample_id, "ground_truth": gt,
                          "annotation_sha256": annotation_sha,
                          "candidate_statistics": prepared[sample.sample_id]["statistics"],
                          "candidates": rows,
                          "selection": serializable_selections(selections),
                          "matched_gt_indices": {name: sorted(ids) for name, ids in matched.items()}})
        for arm, ids in matched.items():
            entry = aggregate.setdefault(arm, {"exact": 0, "gt_words": 0,
                                                "recovered_vs_baseline": 0,
                                                "damaged_vs_baseline": 0,
                                                "changed_anchors": 0})
            base_ids = matched["baseline_pad0_threshold050"]
            entry["exact"] += len(ids)
            entry["gt_words"] += len(gt)
            entry["recovered_vs_baseline"] += len(ids - base_ids)
            entry["damaged_vs_baseline"] += len(base_ids - ids)
            entry["changed_anchors"] += sum(value != baseline[anchor]
                                             for anchor, value in selections[arm].items())
        print(f"[evaluation] {len(per_image)}/{len(samples)} {sample.sample_id}", flush=True)
    if len(samples) == 125 and (aggregate["baseline_pad0_threshold050"]["exact"] != 368
                                or aggregate["baseline_pad0_threshold050"]["gt_words"] != 983):
        raise RuntimeError("frozen Total-Text TRAIN aggregate changed")
    for entry in aggregate.values():
        entry["exact_rate"] = entry["exact"] / entry["gt_words"]
        entry["net_vs_baseline"] = entry["recovered_vs_baseline"] - entry["damaged_vs_baseline"]
    return {"summary": aggregate, "per_image": per_image,
            "result_class": "SMOKE_TEST" if len(samples) < 125 else "APPROXIMATE_TRAIN_DIAGNOSTIC",
            "claim_label": "approximate reproduction; Total-Text TRAIN diagnostic",
            "table_5_claim": False, "gt_used_in_selection": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data_root", "probability_cache", "output_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    for name in ("parseq_source", "parseq_source_archive", "parseq_checkpoint", "clip_model_dir"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    parser.add_argument("--baseline-ocr-cache", type=Path)
    parser.add_argument("--clip-revision")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cached-threshold-audit", action="store_true")
    parser.add_argument("--parseq-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or (args.limit is not None and not 1 <= args.limit <= 124):
        parser.error("batch size must be positive and limit between 1 and 124")
    if sum((args.prepare_only, args.cached_threshold_audit, args.parseq_only)) > 1:
        parser.error("choose only one partial execution mode")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    data_root = args.data_root.resolve()
    threshold_root = args.probability_cache.resolve()
    training = train_config()
    manifest_path = data_root / training["train_manifest"]
    if sha(manifest_path) != training["train_manifest_sha256"]:
        raise RuntimeError("frozen Total-Text TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise RuntimeError("this suite permits only Total-Text TRAIN")
    _, val_ids, val_sha = _split_indices(manifest, training["training"]["seed"], 125)
    protocol_path = threshold_root / "protocol.json"
    cache_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (cache_protocol.get("validation_ids_sha256") != val_sha
        or cache_protocol.get("train_manifest_sha256") != sha(manifest_path)
        or cache_protocol.get("checkpoint_sha256") != config["segmentation_checkpoint_sha256"]
        or cache_protocol.get("thresholds") != config["thresholds"]
        or cache_protocol.get("config_sha256") != config["threshold_config_sha256"]
        or cache_protocol.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256):
        raise RuntimeError("Total-Text threshold cache protocol differs")
    cache_protocol_sha = sha256(json.dumps(cache_protocol, sort_keys=True).encode()).hexdigest()
    reference_path = threshold_root / "results.json"
    if sha(reference_path) != config["threshold_report_sha256"]:
        raise RuntimeError("frozen Total-Text threshold report changed")
    if not args.prepare_only and not args.cached_threshold_audit:
        required = ["parseq_source", "parseq_source_archive", "parseq_checkpoint",
                    "baseline_ocr_cache"]
        if not args.parseq_only:
            required.extend(("clip_model_dir", "clip_revision"))
        for name in required:
            if getattr(args, name) is None:
                parser.error(f"full battery requires --{name.replace('_', '-')}")
        if not args.parseq_only and (not re.fullmatch(r"[0-9a-f]{40}", args.clip_revision)
                                    or args.clip_model_dir.resolve().name != args.clip_revision):
            raise ValueError("CLIP snapshot must be named by a full commit SHA")
        paths = [args.parseq_source, args.parseq_source_archive, args.parseq_checkpoint]
        if not args.parseq_only:
            paths.extend((args.clip_model_dir / "config.json",
                          args.clip_model_dir / "preprocessor_config.json",
                          args.clip_model_dir / "tokenizer.json"))
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"model artifact missing: {path}")
        if not args.parseq_only and not ((args.clip_model_dir / "model.safetensors").is_file()
                                         or (args.clip_model_dir / "pytorch_model.bin").is_file()):
            raise FileNotFoundError("pinned CLIP weights missing")
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable on requested device")
    samples = [manifest.samples[index] for index in val_ids[:args.limit]]
    output_dir = args.output_dir.resolve()
    if output_dir == REPO or REPO in output_dir.parents:
        raise ValueError("diagnostic outputs must be outside the repository")
    plan = {"schema_version": 1, "reproduction_mode": "approximate", "dataset": "total_text",
            "split": "TRAIN_heldout_125", "result_class": "SMOKE_TEST" if args.limit else "APPROXIMATE_TRAIN_DIAGNOSTIC",
            "sample_ids": [s.sample_id for s in samples], "config": config,
            "config_sha256": sha(args.config), "manifest_sha256": sha(manifest_path),
            "threshold_protocol_sha256": sha(protocol_path),
            "threshold_report_sha256": sha(reference_path),
            "baseline_ocr_cache": str(args.baseline_ocr_cache.resolve())
            if args.baseline_ocr_cache is not None else None,
            "execution": {"device": args.device, "batch_size": args.batch_size,
                          "precision": "float32", "seed": 0,
                          "gpu_name": torch.cuda.get_device_name(args.device)
                          if args.device.startswith("cuda") else None,
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                          "cudnn_deterministic": torch.backends.cudnn.deterministic,
                          "cudnn_benchmark": torch.backends.cudnn.benchmark},
            "software": {"python": sys.version, "torch": torch.__version__,
                         "torch_cuda": torch.version.cuda, "numpy": np.__version__,
                         "pillow": importlib.metadata.version("Pillow"),
                         "opencv_python": importlib.metadata.version("opencv-python"),
                         "transformers": importlib.metadata.version("transformers")},
            "code_sha256": {name: sha(REPO / name) for name in (
                "recognition/crop_clip_battery.py", "scripts/run_totaltext_crop_clip_battery.py",
                "scripts/crop_clip_runtime.py", "recognition/parseq.py",
                "scripts/evaluate_totaltext_highres_contour_padding.py")},
            "gt_used_in_selection": False, "table_5_claim": False}
    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise RuntimeError("output directory belongs to a different plan")
    else:
        write_json(plan_path, plan)
    prepared = {}
    for position, sample in enumerate(samples, 1):
        prepared[sample.sample_id] = prepare(
            sample, data_root=data_root, cache_root=threshold_root,
            output_dir=output_dir, config=config, plan_sha=sha(plan_path),
            cache_protocol_sha=cache_protocol_sha)
        print(f"[prepare] {position}/{len(samples)} {sample.sample_id}", flush=True)
    frozen_report = json.loads(reference_path.read_text(encoding="utf-8"))
    verify_prepared_baseline(samples, prepared, frozen_report)
    if args.prepare_only:
        counts = {"images": len(samples),
                  "anchors": sum(v["statistics"]["baseline_anchors"] for v in prepared.values()),
                  "candidates": sum(v["statistics"]["candidates"] for v in prepared.values())}
        print(json.dumps(counts, sort_keys=True), flush=True)
        return 0
    if args.cached_threshold_audit:
        if args.baseline_ocr_cache is None or not args.baseline_ocr_cache.is_dir():
            parser.error("cached threshold audit requires --baseline-ocr-cache")
        audit = cached_threshold_audit(
            samples, prepared, data_root=data_root, probability_cache=threshold_root,
            baseline_ocr_cache=args.baseline_ocr_cache.resolve(), frozen_report=frozen_report)
        audit["plan_sha256"] = sha(plan_path)
        audit_path = output_dir / "cached_threshold_audit.json"
        if audit_path.exists():
            if json.loads(audit_path.read_text(encoding="utf-8")) != audit:
                raise RuntimeError("existing cached threshold audit differs")
        else:
            write_json(audit_path, audit)
        print(json.dumps({"report": str(audit_path), "summary": audit["summary"]}, sort_keys=True), flush=True)
        return 0
    stage = "PARSeq"
    try:
        ocr, parseq_peak = run_parseq(
            samples, prepared, data_root=data_root, cache_root=threshold_root,
            baseline_ocr_cache=args.baseline_ocr_cache.resolve(),
            output_dir=output_dir, source=args.parseq_source.resolve(),
            archive=args.parseq_source_archive.resolve(),
            checkpoint=args.parseq_checkpoint.resolve(), device=args.device,
            batch_size=args.batch_size)
        if args.parseq_only:
            counts = {"result_class": plan["result_class"], "images": len(samples),
                      "candidates": sum(len(value["candidates"]) for value in prepared.values()),
                      "new_ocr_crops_at_shard_creation": sum(
                          json.loads((output_dir / "ocr" / f"{_key(s.sample_id)}.json")
                                     .read_text())["new_ocr_crops"] for s in samples),
                      "parseq_peak": parseq_peak, "clip_scored": False}
            summary_path = output_dir / "parseq_only_summary.json"
            if summary_path.exists():
                if json.loads(summary_path.read_text(encoding="utf-8")) != counts:
                    raise RuntimeError("existing PARSeq-only summary differs")
            else:
                write_json(summary_path, counts)
            print(json.dumps(counts, sort_keys=True), flush=True)
            return 0
        stage = "CLIP"
        clip, clip_metadata = run_clip(
            samples, prepared, ocr, data_root=data_root, output_dir=output_dir,
            model_dir=args.clip_model_dir.resolve(), revision=args.clip_revision,
            device=args.device, batch_size=args.batch_size, templates=config["clip_prompts"])
    except torch.cuda.OutOfMemoryError as error:
        write_json(output_dir / "oom.json", {"result_class": "FAILED_RUN", "stage": stage,
                   "error": str(error), "device": args.device, "precision": "float32",
                   "batch_size": args.batch_size, "model_set": [stage],
                   "peak": _cuda_peak(args.device),
                   "candidate_crop_shapes_hwc": sorted({tuple(row["crop_shape"])
                       for value in prepared.values() for row in value["candidates"]})})
        raise
    result = evaluate(samples, prepared, ocr, clip, data_root=data_root, config=config,
                      frozen_report=frozen_report)
    result["plan_sha256"] = sha(plan_path)
    result["hardware"] = {"device": args.device, "precision": "float32",
                          "batch_size": args.batch_size, "model_loading": "sequential",
                          "gpu_name": torch.cuda.get_device_name(args.device)
                          if args.device.startswith("cuda") else None,
                          "parseq": parseq_peak, "clip": clip_metadata,
                          "upstream_probability_cache_device": cache_protocol.get("device")}
    result_path = output_dir / "results.json"
    if result_path.exists():
        if json.loads(result_path.read_text(encoding="utf-8")) != result:
            raise RuntimeError("existing results differ; refusing overwrite")
    else:
        write_json(result_path, result)
    print(json.dumps({"result": str(result_path), "summary": result["summary"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
