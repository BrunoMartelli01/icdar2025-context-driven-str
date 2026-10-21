"""TRAIN-only word/PARSeq audit of the fixed 500-epoch letterbox branches.

The weight-5 branch selected the original epoch-71 weights. Verify tensor
identity and reuse its frozen OCR; only the weight-3 epoch-263 model needs
new mask/crop/PARSeq inference. No TEST data is loaded.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import gc

import numpy as np
import torch

from models.unet_adv import PracticalTotalTextHighResAGUNet
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer,
)
from recognition.scene_text_v1_evaluation import box_iou
from scripts.evaluate_totaltext_highres_contour_padding import (
    _save_mask, contour_proposals, crop_array_hash,
)
from scripts.evaluate_totaltext_letterbox_pad0_test import _sample_metrics
from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import _read_ocr
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import (
    _image, _tensor, _train_gt,
)
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.train_totaltext_letterbox_500_separator_pair import (
    _source as qualified_pair_source,
)
from scripts.train_totaltext_v4_highres_ablation import _training_config
from scripts.train_totaltext_v4_image_frequency_ablation import (
    CONFIG_PATH as TRAIN_CONFIG_PATH, _config as train_config, _transform,
)
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.totaltext_image_frequency_augment import (
    letterbox_canvas_mask, remap_letterbox_native_box,
)
from segmentation.training import load_checkpoint


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_letterbox_500_word_check_train_v1.json"
METRICS = ("geometry_iou_ge_25", "geometry_iou_ge_50", "geometry_iou_ge_75",
           "exact_casefold_iou_ge_50", "exact_no_symbols_iou_ge_50")


def _config() -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (value.get("reproduction_mode") != "approximate"
        or value.get("classification") != "PRACTICAL_APPROXIMATION"
        or value.get("split") != "TRAIN_heldout_125"
        or value.get("gt_words") != 983
        or value.get("probability_threshold") != 0.5
        or value.get("padding_native_pixels") != 0
        or value.get("area_filter") != "disabled"
        or value.get("maximum_proposals") != "disabled"):
        raise ValueError("Frozen TRAIN word-audit protocol changed")
    return value


def _verify_identical_weights(source: Path, sep5: Path) -> int:
    """Do not equate reports merely because selected epochs are equal."""
    first = torch.load(source, map_location="cpu", weights_only=False)
    second = torch.load(sep5, map_location="cpu", weights_only=False)
    left, right = first["model_state_dict"], second["model_state_dict"]
    if (first["epoch"] != 71 or second["epoch"] != 71 or left.keys() != right.keys()
        or not all(torch.equal(left[key], right[key]) for key in left)):
        raise ValueError("Sep5 selected checkpoint is not identical to source epoch 71")
    return len(left)


def _word_ids(rows: list[dict], metric: str) -> set[str]:
    return {f"{row['sample_id']}#{index}" for row in rows
            for index in row["matched_gt_indices"][metric]}


def _best_spatial(predictions: list[dict], gt: dict) -> dict | None:
    if not predictions:
        return None
    best = max(predictions, key=lambda item: box_iou(item["bbox"], gt["bbox"]))
    return {"bbox": best["bbox"], "text": best["text"],
            "iou": box_iou(best["bbox"], gt["bbox"])}


def _changes(source_rows: list[dict], control_rows: list[dict], metric: str) -> dict:
    original, trained = _word_ids(source_rows, metric), _word_ids(control_rows, metric)
    source_by_id = {row["sample_id"]: row for row in source_rows}
    trained_by_id = {row["sample_id"]: row for row in control_rows}
    if source_by_id.keys() != trained_by_id.keys():
        raise ValueError("Compared branches have different image identities")

    def detail(identifier: str) -> dict:
        sample_id, index_text = identifier.rsplit("#", 1)
        index = int(index_text)
        source, control = source_by_id[sample_id], trained_by_id[sample_id]
        gt = source["ground_truth"][index]
        if gt != control["ground_truth"][index]:
            raise ValueError(f"GT changed between branches: {identifier}")
        return {"word_id": identifier, "sample_id": sample_id,
                "gt_index": index, "gt": gt,
                "source_best_spatial": _best_spatial(source["predictions"], gt),
                "control_best_spatial": _best_spatial(control["predictions"], gt)}

    gained, lost = trained - original, original - trained
    return {"source_correct": len(original), "control_correct": len(trained),
            "unchanged_correct": len(original & trained),
            "gained": len(gained), "lost": len(lost),
            "net": len(gained) - len(lost),
            "gained_words": [detail(identifier) for identifier in sorted(gained)],
            "lost_words": [detail(identifier) for identifier in sorted(lost)]}


def run(args) -> dict:
    config, training = _config(), train_config()
    root = args.target_root.resolve()
    pair_args = argparse.Namespace(target_root=root, source_run=args.source_run)
    manifest, _, val_ids, val_sha, source_checkpoint, source_report = qualified_pair_source(
        pair_args, {"fork_run_manifest_sha256":
                        "d4bba53d7913dd214f928c326c5984aab6ecbbfe8d444bb6fcb6c3bd47880383",
                    "fork_checkpoint_sha256": config["source_checkpoint_sha256"]}, training)
    reference_dir = args.source_run.resolve().parent / "validation-pad0-parseq-v1"
    reference_path = reference_dir / "results.json"
    if sha256_file(reference_path) != config["source_ocr_report_sha256"]:
        raise ValueError("Frozen epoch-71 TRAIN OCR report changed")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    if (reference["protocol"]["validation_ids_sha256"] != val_sha
        or reference["variants"]["letterbox"]["aggregate"]["gt_words"] != 983):
        raise ValueError("Source OCR report is not the same TRAIN validation split")
    source_rows = reference["variants"]["letterbox"]["per_image"]
    run_root = args.pair_run_root.resolve()
    reports = {}
    for name, label in (("letterbox_500_control", "control"),
                        ("letterbox_500_sep5", "sep5")):
        report_path = run_root / name / "run_manifest.json"
        checkpoint_path = run_root / name / "best.pt"
        if (sha256_file(report_path) != config[f"{label}_run_manifest_sha256"]
            or sha256_file(checkpoint_path) != config[f"{label}_checkpoint_sha256"]):
            raise ValueError(f"Frozen 500-epoch branch changed: {name}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report["status"] != "SUCCESS" or report["total_epochs"] != 500
            or report["run_contract"]["validation_ids_sha256"] != val_sha
            or report["checkpoint"]["sha256"] != config[f"{label}_checkpoint_sha256"]):
            raise ValueError(f"500-epoch branch provenance mismatch: {name}")
        reports[name] = report
    if reports["letterbox_500_control"]["checkpoint"]["epoch"] != 263:
        raise ValueError("Control selected checkpoint epoch changed")
    if reports["letterbox_500_sep5"]["checkpoint"]["epoch"] != 71:
        raise ValueError("Sep5 selected checkpoint epoch changed")
    equal_tensors = _verify_identical_weights(
        source_checkpoint, run_root / "letterbox_500_sep5/best.pt")
    if (sha256_file(args.parseq_checkpoint.resolve()) != PARSEQ_CHECKPOINT_SHA256
        or sha256_file(args.parseq_source_archive.resolve()) != PARSEQ_RUNTIME_ARCHIVE_SHA256
        or not args.parseq_source.resolve().is_dir()):
        raise ValueError("Qualified PARSeq artifact changed or missing")
    if args.preflight_only:
        status = {"status": "PREFLIGHT_PASS", "split": "TRAIN_heldout_125",
                  "images": len(val_ids), "gt_words": 983,
                  "sep5_equals_source_model_tensors": True,
                  "equal_model_tensors": equal_tensors,
                  "control_checkpoint_sha256": config["control_checkpoint_sha256"]}
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)
        return status
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on requested device")
    device = torch.device(args.device)
    output = args.output_dir.resolve()
    if output == REPO or REPO in output.parents:
        raise ValueError("Word-audit outputs must be outside the repository")
    protocol = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
                "split": "TRAIN_heldout_125", "validation_ids_sha256": val_sha,
                "config_sha256": sha256_file(CONFIG_PATH),
                "source_ocr_report_sha256": config["source_ocr_report_sha256"],
                "source_checkpoint_sha256": config["source_checkpoint_sha256"],
                "control_checkpoint_sha256": config["control_checkpoint_sha256"],
                "sep5_checkpoint_sha256": config["sep5_checkpoint_sha256"],
                "sep5_identical_model_tensors": equal_tensors,
                "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                "segmentation": "letterbox 512 input; sigmoid>=0.5 native256; batch1",
                "localization": "native RETR_EXTERNAL; area OFF; top10 OFF; pad0",
                "matching": "one-to-one bbox IoU>=0.5; casefold and Unicode symbols stripped",
                "ground_truth_use": "post-hoc TRAIN validation only", "device": args.device,
                "parseq_batch_size": args.parseq_batch_size}
    protocol_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing word-audit output belongs to another protocol")
    else:
        _atomic_json(protocol_path, protocol)
    by_id = {manifest.samples[index].sample_id: manifest.samples[index] for index in val_ids}
    shards = {}
    pending = []
    for index in val_ids:
        sample = manifest.samples[index]
        key = sample.sample_id.replace(":", "__")
        path = output / "samples" / f"{key}.json"
        mask_path = output / "masks" / f"{key}.png"
        if path.exists():
            shard = json.loads(path.read_text(encoding="utf-8"))
            if (shard.get("protocol_sha256") != protocol_sha
                or shard.get("sample_id") != sample.sample_id
                or shard.get("image_sha256") != sample.image_sha256
                or shard.get("checkpoint_sha256") != config["control_checkpoint_sha256"]
                or sha256_file(mask_path) != shard.get("native_mask_file_sha256")):
                raise ValueError(f"Cached control shard changed: {sample.sample_id}")
            shards[sample.sample_id] = shard
        else:
            pending.append(sample)
    if pending:
        if device.type == "cuda":
            _prepare_cuda_memory_tracking(device)
        model = PracticalTotalTextHighResAGUNet(skip_floor=0.25, encoder_pretrained=False)
        control = reports["letterbox_500_control"]
        load_checkpoint(run_root / "letterbox_500_control/best.pt", model=model,
            expected_architecture=model.architecture_identity,
            expected_run_contract=control["run_contract"],
            expected_training_config=_training_config(training, {"separation_weight": 3.0}))
        model.eval().to(device)
        transform = _transform(training, "letterbox", train=False)
        try:
            for position, sample in enumerate(pending, 1):
                rgb = _image(sample, root)
                with torch.inference_mode():
                    logits = model(_tensor(rgb, transform).to(device))
                if tuple(logits.shape) != (1, 1, 256, 256):
                    raise RuntimeError("Wrong native letterbox output shape")
                native = np.ascontiguousarray((torch.sigmoid(logits[0, 0]) >= 0.5).byte().cpu().numpy())
                del logits
                native *= letterbox_canvas_mask((rgb.shape[1], rgb.shape[0])).byte().numpy()
                key = sample.sample_id.replace(":", "__")
                mask_sha = _save_mask(output / "masks" / f"{key}.png", native)
                raw = contour_proposals(native, rgb.shape[:2], [0])["0"]
                for proposal in raw["proposals"]:
                    proposal["bbox"] = remap_letterbox_native_box(
                        proposal["padded_bbox_native"], (rgb.shape[1], rgb.shape[0]))
                    x1, y1, x2, y2 = proposal["bbox"]
                    proposal["crop_rgb_sha256"] = crop_array_hash(
                        np.ascontiguousarray(rgb[y1:y2, x1:x2]))
                gt, annotation_sha = _train_gt(sample, root, (rgb.shape[1], rgb.shape[0]))
                shard = {"schema_version": 1, "protocol_sha256": protocol_sha,
                         "sample_id": sample.sample_id,
                         "checkpoint_sha256": config["control_checkpoint_sha256"],
                         "image_sha256": sample.image_sha256,
                         "annotation_sha256": annotation_sha,
                         "native_mask_file_sha256": mask_sha,
                         "native_mask_array_sha256": sha256(native.tobytes()).hexdigest(),
                         "raw_contours": raw["raw_contours"],
                         "ground_truth": gt, "proposals": raw["proposals"]}
                _atomic_json(output / "samples" / f"{key}.json", shard)
                shards[sample.sample_id] = shard
                if position % 10 == 0 or position == len(pending):
                    print(f"[control masks] {position}/{len(pending)} new; {len(shards)}/125 total", flush=True)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    else:
        print("[control masks] verified cache 125/125", flush=True)
    pending_ocr = []
    seen = set()
    for index in val_ids:
        sample_id = manifest.samples[index].sample_id
        for proposal in shards[sample_id]["proposals"]:
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
    print(f"[control PARSeq] unique_crops={len(seen)} pending={len(pending_ocr)}", flush=True)
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
                        raise ValueError("Control crop pixels changed after freeze")
                    crops.append(crop)
                predictions = recognizer.recognize_candidate_batch(crops)
                if len(predictions) != len(batch):
                    raise RuntimeError("PARSeq returned wrong batch length")
                for (_, _, crop_sha), value in zip(batch, predictions):
                    _atomic_json(output / "ocr" / f"{crop_sha}.json", {
                        "schema_version": 1, "crop_rgb_sha256": crop_sha,
                        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                        **value.to_dict()})
                print(f"[control PARSeq] {min(start + len(batch), len(pending_ocr))}/{len(pending_ocr)}", flush=True)
        finally:
            recognizer.release()
    totals = {metric: 0 for metric in METRICS}
    totals.update({"gt_words": 0, "proposals": 0})
    rows = []
    source_by_id = {row["sample_id"]: row for row in source_rows}
    for index in val_ids:
        sample_id = manifest.samples[index].sample_id
        shard = shards[sample_id]
        source = source_by_id[sample_id]
        if shard["ground_truth"] != source["ground_truth"]:
            raise ValueError(f"TRAIN word GT changed: {sample_id}")
        predictions = []
        for proposal in shard["proposals"]:
            crop_sha = proposal["crop_rgb_sha256"]
            own = output / "ocr" / f"{crop_sha}.json"
            prior = reference_dir / "ocr" / f"{crop_sha}.json"
            predictions.append({"bbox": proposal["bbox"],
                                "text": _read_ocr(own if own.exists() else prior, crop_sha),
                                "crop_rgb_sha256": crop_sha})
        metrics = _sample_metrics(predictions, shard["ground_truth"])
        for metric in totals:
            totals[metric] += metrics[metric]
        rows.append({"sample_id": sample_id, "ground_truth": shard["ground_truth"],
                     "predictions": predictions,
                     "matched_gt_indices": metrics["matched_gt_indices"]})
    if totals["gt_words"] != 983:
        raise ValueError("Frozen TRAIN word denominator changed")
    source_totals = reference["variants"]["letterbox"]["aggregate"]
    for metric in METRICS:
        if source_totals[metric] != sum(len(row["matched_gt_indices"][metric]) for row in source_rows):
            raise ValueError(f"Frozen source per-word scores disagree: {metric}")
    report = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "result_class": "TRAIN-only word/PARSeq diagnostic; not TEST or Table 5",
              "protocol": protocol, "images": 125, "gt_words": 983,
              "runtime": {"device": str(device), "segmentation_dtype": "float32",
                          "segmentation_batch_size": 1,
                          "segmentation_input_shape": [1, 3, 512, 512],
                          "parseq_batch_size": args.parseq_batch_size,
                          "model_lifecycle": "sequential_release_before_parseq",
                          "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                              if device.type == "cuda" else None,
                          "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)
                              if device.type == "cuda" else None},
              "source_epoch71": {"aggregate": source_totals, "per_image": source_rows},
              "control_epoch263": {"aggregate": totals, "per_image": rows},
              "sep5_selected_epoch71": {"aggregate": source_totals,
                                       "identical_model_tensors": True,
                                       "reused_source_ocr": True},
              "changes": {metric: _changes(source_rows, rows, metric)
                          for metric in METRICS}}
    result_path = output / "results.json"
    if result_path.exists():
        if json.loads(result_path.read_text(encoding="utf-8")) != report:
            raise ValueError("Existing word-audit report differs; refusing overwrite")
    else:
        _atomic_json(result_path, report)
    print(json.dumps({"result": str(result_path), "sha256": sha256_file(result_path),
                      "source_epoch71": {metric: source_totals[metric] for metric in METRICS},
                      "control_epoch263": totals,
                      "sep5_selected_epoch71": "identical to source epoch71",
                      "changes": {metric: {key: value[key] for key in
                          ("source_correct", "control_correct", "gained", "lost", "net")}
                          for metric, value in report["changes"].items()}},
                     indent=2, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--pair-run-root", type=Path, required=True)
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
