"""PARSeq on frozen Total-Text TRAIN-holdout crops from two word-F1 runs.

PRACTICAL_APPROXIMATION only. This does not change Algorithm 1 or evaluate TEST.
Native masks and crop identities are cached before GT is read for scoring.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset

from models.unet_adv import PracticalTotalTextHighResAGUNet, PracticalTotalTextSeparatorHeadAGUNet
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from scripts.evaluate_totaltext_highres_contour_padding import (
    _save_mask, contour_proposals, crop_array_hash,
)
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import (
    _correct, _geometry_match, _image, _train_gt,
)
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_letterbox_word_f1_separator_head import (
    BRANCHES, CONFIG_PATH, _config, _validation_reference,
)
from scripts.train_totaltext_v4_highres_ablation import _split_indices, _training_config
from scripts.train_totaltext_v4_image_frequency_ablation import (
    _config as source_config, _transform, proposal_metrics,
)
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_image_frequency_augment import remap_letterbox_native_box
from segmentation.training import load_checkpoint


REPO = Path(__file__).resolve().parents[1]


def native_word_mask(logits: torch.Tensor, canvas: np.ndarray, *, separator_head: bool) -> np.ndarray:
    """Use exactly the frozen training-time foreground/separator decision."""
    channels = 2 if separator_head else 1
    if tuple(logits.shape) != (channels, 256, 256) or canvas.shape != (256, 256):
        raise ValueError("Expected native 256x256 logits and letterbox canvas")
    foreground = (torch.sigmoid(logits[0]) >= 0.5).cpu().numpy()
    if separator_head:
        foreground &= ~((torch.sigmoid(logits[1]) >= 0.5).cpu().numpy())
    return np.ascontiguousarray((foreground & canvas.astype(bool)).astype(np.uint8))


def _check_ocr(record: dict, crop_sha: str) -> str:
    if (record.get("crop_rgb_sha256") != crop_sha
        or record.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
        or record.get("parseq_source_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or not isinstance(record.get("raw_text"), str)):
        raise ValueError(f"PARSeq cache provenance mismatch: {crop_sha}")
    return record["raw_text"]


def _runs(training_runs: Path, validation_sha: str, manifest_sha: str) -> dict:
    config = _config()
    expected_sources = {
        "model_source_sha256": REPO / "models/unet_adv.py",
        "trainer_source_sha256": REPO / "scripts/train_totaltext_letterbox_word_f1_separator_head.py",
        "training_primitives_sha256": REPO / "segmentation/training.py",
        "transform_source_sha256": REPO / "segmentation/totaltext_image_frequency_augment.py",
    }
    runs = {}
    for branch in BRANCHES:
        directory = training_runs / branch
        path = directory / "run_manifest.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        contract = report["run_contract"]
        if (report.get("status") != "SUCCESS" or contract.get("branch") != branch
            or contract.get("config_sha256") != sha256_file(CONFIG_PATH)
            or contract.get("source_training_config_sha256") != sha256_file(
                REPO / config["source_training_config"])
            or contract.get("manifest_sha256") != manifest_sha
            or contract.get("validation_ids_sha256") != validation_sha
            or contract.get("foreground_threshold") != 0.5
            or contract.get("separator_threshold") != 0.5
            or contract.get("postprocessing") != config["postprocessing"]
            or any(contract.get(key) != sha256_file(source)
                   for key, source in expected_sources.items())):
            raise ValueError(f"Frozen run contract changed: {branch}")
        checkpoint = directory / "best.pt"
        if sha256_file(checkpoint) != report["checkpoint"]["sha256"]:
            raise ValueError(f"Checkpoint SHA256 mismatch: {branch}")
        frozen = json.loads((directory / "validation_word_geometry.json").read_text())
        if (frozen.get("aggregate") != report["validation_word_geometry"]
            or len(frozen.get("per_image", [])) != 125):
            raise ValueError(f"Frozen validation geometry mismatch: {branch}")
        runs[branch] = {"directory": directory, "manifest_path": path,
                        "report": report, "frozen": frozen}
    return runs


def _cached_shard(path: Path, mask_path: Path, sample, annotation: Path,
                  protocol_sha: str, checkpoint_sha: str, frozen_row: dict) -> dict:
    shard = json.loads(path.read_text(encoding="utf-8"))
    if (shard.get("sample_id") != sample.sample_id
        or shard.get("protocol_sha256") != protocol_sha
        or shard.get("checkpoint_sha256") != checkpoint_sha
        or shard.get("image_sha256") != sample.image_sha256
        or shard.get("annotation_sha256") != sha256_file(annotation)
        or sha256_file(mask_path) != shard.get("native_mask_file_sha256")
        or shard.get("native_word_geometry") != frozen_row):
        raise ValueError(f"Cached validation shard changed: {path}")
    return shard


def run(args) -> dict:
    config = _config()
    source = source_config()
    target = args.target_root.resolve()
    training_runs = args.training_runs.resolve()
    manifest_path = target / source["train_manifest"]
    if sha256_file(manifest_path) != source["train_manifest_sha256"]:
        raise ValueError("Qualified Total-Text TRAIN manifest changed")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("This diagnostic accepts TRAIN only")
    _, validation_ids, validation_sha = _split_indices(manifest, 202503, 125)
    runs = _runs(training_runs, validation_sha, source["train_manifest_sha256"])
    if (sha256_file(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256
        or sha256_file(args.parseq_source_archive.resolve()) != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or not args.parseq_source.resolve().is_dir()):
        raise ValueError("Qualified PARSeq checkpoint/source provenance mismatch")
    if args.preflight_only:
        summary = {"status": "PREFLIGHT_PASS", "split": "TRAIN_heldout_125",
                   "images": 125, "variants": {name: runs[name]["report"]["checkpoint"]
                                             for name in BRANCHES}}
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary
    output = args.output_dir.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("Output directory must be outside repository")
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on requested device")
    device = torch.device(args.device)
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    protocol = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
                "result_class": "TRAIN-holdout OCR diagnostic; not paper reproduction",
                "dataset": "total_text", "split": "frozen_TRAIN_validation_125",
                "train_manifest_sha256": source["train_manifest_sha256"],
                "validation_ids_sha256": validation_sha,
                "config_sha256": sha256_file(CONFIG_PATH),
                "evaluator_source_sha256": sha256_file(Path(__file__)),
                "run_manifests_sha256": {name: sha256_file(runs[name]["manifest_path"])
                                         for name in BRANCHES},
                "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                "segmentation": "letterbox 512; native256 sigmoid>=0.5; separator>=0.5 subtracted only for two-head model",
                "localization": "native RETR_EXTERNAL contours; area OFF; top10 OFF; pad0",
                "crop": "inverse letterbox bbox to original RGB; no warping",
                "matching": "one-to-one bbox IoU>=0.5 plus exact casefold/Unicode P-S-stripped",
                "ground_truth_use": "post-hoc evaluation only",
                "device": args.device, "parseq_batch_size": args.parseq_batch_size}
    protocol_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing output belongs to a different protocol")
    else:
        _atomic_json(protocol_path, protocol)
    if (output / "results.json").exists():
        result = json.loads((output / "results.json").read_text(encoding="utf-8"))
        if result.get("protocol") != protocol:
            raise ValueError("Completed result protocol mismatch")
        print(f"RESULT {output / 'results.json'} SHA256 {sha256_file(output / 'results.json')}",
              flush=True)
        return result

    by_id = {sample.sample_id: sample for sample in manifest.samples}
    frozen_rows = {name: {row["sample_id"]: row for row in runs[name]["frozen"]["per_image"]}
                   for name in BRANCHES}
    shards = {name: {} for name in BRANCHES}
    transform = _transform(source, "letterbox", train=False)
    for name in BRANCHES:
        checkpoint_sha = runs[name]["report"]["checkpoint"]["sha256"]
        missing = []
        for index in validation_ids:
            sample = manifest.samples[index]
            key = sample.sample_id.replace(":", "__")
            shard_path = output / name / "samples" / f"{key}.json"
            mask_path = output / name / "masks" / f"{key}.png"
            annotation = target / "datasets/total_text/train/annotations" / (
                sample.sample_id.split(":", 1)[1] + ".txt")
            if shard_path.exists():
                shards[name][sample.sample_id] = _cached_shard(
                    shard_path, mask_path, sample, annotation, protocol_sha,
                    checkpoint_sha, frozen_rows[name][sample.sample_id])
            else:
                missing.append(sample.sample_id)
        if not missing:
            print(f"[mask {name}] verified cached 125/125", flush=True)
            continue
        model_cls = (PracticalTotalTextSeparatorHeadAGUNet if name == "word_f1_separator_head"
                     else PracticalTotalTextHighResAGUNet)
        model = model_cls(skip_floor=0.25, encoder_pretrained=False)
        training_config = replace(_training_config(source, {"separation_weight": 3.0}),
                                  epochs=100, checkpoint_selection="validation_word_f1")
        load_checkpoint(runs[name]["directory"] / "best.pt", model=model,
                        expected_architecture=model.architecture_identity,
                        expected_run_contract=runs[name]["report"]["run_contract"],
                        expected_training_config=training_config)
        model.to(device).eval()
        data = Subset(SegmentationDataset(manifest, target, transform=transform), validation_ids)
        loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0,
                            pin_memory=device.type == "cuda")
        references = _validation_reference(manifest, target, validation_ids, transform)
        try:
            with torch.inference_mode():
                position = 0
                for batch in loader:
                    logits = model(batch["image"].to(device))
                    if tuple(logits.shape[1:]) != ((2 if name == "word_f1_separator_head" else 1), 256, 256):
                        raise ValueError(f"Unexpected native logits shape: {name}")
                    for offset, sample_id in enumerate(batch["sample_id"]):
                        position += 1
                        if sample_id in shards[name]:
                            continue
                        sample = by_id[sample_id]
                        rgb = _image(sample, target)
                        owners_base, expected, canvas = references[sample_id]
                        native = native_word_mask(logits[offset], canvas,
                            separator_head=name == "word_f1_separator_head")
                        key = sample_id.replace(":", "__")
                        mask_path = output / name / "masks" / f"{key}.png"
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
                        target_mask = batch["mask"][offset, 0].numpy().astype(bool)
                        valid = batch["validity"][offset, 0].numpy().astype(bool)
                        owners = owners_base.copy()
                        owners[~target_mask | ~valid] = 0
                        geometry = proposal_metrics(native.astype(bool), owners,
                                                    expected_owner_ids=expected)
                        row = frozen_rows[name][sample_id]
                        if any(geometry[key] != value for key, value in row.items()
                               if key not in ("sample_id", "foreground_iou", "final_mask_iou")):
                            raise RuntimeError(f"Frozen batch-8 word geometry drift: {name} {sample_id}")
                        gt, annotation_sha = _train_gt(sample, target,
                                                       (rgb.shape[1], rgb.shape[0]))
                        shard = {"schema_version": 1, "sample_id": sample_id,
                                 "protocol_sha256": protocol_sha,
                                 "checkpoint_sha256": checkpoint_sha,
                                 "image_sha256": sample.image_sha256,
                                 "annotation_sha256": annotation_sha,
                                 "native_mask_file_sha256": mask_sha,
                                 "native_mask_array_sha256": sha256(native.tobytes()).hexdigest(),
                                 "native_word_geometry": row, "ground_truth": gt,
                                 "raw_contours": raw["raw_contours"], "proposals": raw["proposals"]}
                        _atomic_json(output / name / "samples" / f"{key}.json", shard)
                        shards[name][sample_id] = shard
                    if position % 8 == 0 or position == 125:
                        print(f"[mask {name}] {position}/125", flush=True)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if len(shards[name]) != 125:
            raise RuntimeError(f"Missing validation masks: {name}")

    crop_records = {}
    pending = []
    seen_crops = set()
    for index in validation_ids:
        sample_id = manifest.samples[index].sample_id
        for name in BRANCHES:
            for proposal in shards[name][sample_id]["proposals"]:
                crop_sha = proposal["crop_rgb_sha256"]
                if crop_sha in seen_crops:
                    continue
                seen_crops.add(crop_sha)
                local = output / "ocr" / f"{crop_sha}.json"
                reused = args.reuse_ocr / f"{crop_sha}.json" if args.reuse_ocr else None
                if local.exists():
                    crop_records[crop_sha] = _check_ocr(json.loads(local.read_text()), crop_sha)
                elif reused is not None and reused.exists():
                    crop_records[crop_sha] = _check_ocr(json.loads(reused.read_text()), crop_sha)
                else:
                    pending.append((sample_id, proposal["bbox"], crop_sha))
    print(f"[PARSeq] reusable={len(crop_records)} pending={len(pending)}", flush=True)
    if pending:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=args.parseq_source, checkpoint=args.parseq_checkpoint,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive, device=args.device))
        try:
            for start in range(0, len(pending), args.parseq_batch_size):
                batch = pending[start:start + args.parseq_batch_size]
                images = {}
                crops = []
                for sample_id, bbox, crop_sha in batch:
                    if sample_id not in images:
                        images[sample_id] = _image(by_id[sample_id], target)
                    x1, y1, x2, y2 = bbox
                    crop = np.ascontiguousarray(images[sample_id][y1:y2, x1:x2])
                    if crop_array_hash(crop) != crop_sha:
                        raise ValueError(f"Crop hash mismatch: {sample_id}")
                    crops.append(crop)
                values = recognizer.recognize_candidate_batch(crops)
                if len(values) != len(batch):
                    raise RuntimeError("PARSeq batch length mismatch")
                for (_, _, crop_sha), value in zip(batch, values):
                    record = {"schema_version": 1, "crop_rgb_sha256": crop_sha,
                              "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                              **value.to_dict()}
                    crop_records[crop_sha] = _check_ocr(record, crop_sha)
                    _atomic_json(output / "ocr" / f"{crop_sha}.json", record)
                print(f"[PARSeq] {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
        finally:
            recognizer.release()

    result = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "protocol": protocol, "images": 125, "variants": {},
              "runtime": {"device": str(device), "segmentation_dtype": "float32",
                          "segmentation_batch_size": 8,
                          "segmentation_input_shape": [8, 3, 512, 512],
                          "parseq_batch_size": args.parseq_batch_size,
                          "model_lifecycle": "sequential_release_before_parseq",
                          "gpu_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                                       if device.type == "cuda" else None),
                          "gpu_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device)
                                                      if device.type == "cuda" else None)}}
    for name in BRANCHES:
        totals = {key: 0 for key in ("gt_words", "proposals", "geometry_iou_ge_25",
                  "geometry_iou_ge_50", "geometry_iou_ge_75",
                  "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")}
        rows = []
        for index in validation_ids:
            sample_id = manifest.samples[index].sample_id
            shard = shards[name][sample_id]
            gt = shard["ground_truth"]
            predictions = [{"bbox": proposal["bbox"],
                            "text": crop_records[proposal["crop_rgb_sha256"]],
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
            raise RuntimeError("TRAIN-holdout GT word count changed")
        totals["exact_casefold_rate_all_gt"] = totals["exact_casefold_iou_ge_50"] / 983
        totals["exact_no_symbols_rate_all_gt"] = totals["exact_no_symbols_iou_ge_50"] / 983
        result["variants"][name] = {"checkpoint_sha256": runs[name]["report"]["checkpoint"]["sha256"],
                                    "aggregate": totals, "per_image": rows}
    _atomic_json(output / "results.json", result)
    print(json.dumps({"result": str(output / "results.json"),
                      "sha256": sha256_file(output / "results.json"),
                      "summary": {name: result["variants"][name]["aggregate"]
                                  for name in BRANCHES}}, indent=2, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--training-runs", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--reuse-ocr", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--parseq-batch-size", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.parseq_batch_size < 1:
        parser.error("--parseq-batch-size must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
