"""Frozen TRAIN-holdout pad-0 contours and PARSeq for image/frequency ablations.

This is a PRACTICAL_APPROXIMATION diagnostic, never a TEST or paper Table 5 run.
Masks and crop identities are persisted before GT matching; exact per-crop OCR
is shared by RGB crop SHA256 across all five variants and is restartable.
"""

from __future__ import annotations

import argparse
import gc
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from models.totaltext_wavelet import PracticalTotalTextWaveletHighResAGUNet
from models.unet_adv import PracticalTotalTextHighResAGUNet
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from recognition.scene_text_datasets import parse_totaltext_annotation
from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.evaluate_totaltext_highres_contour_padding import (
    _save_mask, contour_proposals, crop_array_hash,
)
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.prune_segmentation_checkpoints import sha256_file
from scripts.rescore_totaltext_highres_padding_symbols import normalize_word
from scripts.train_totaltext_v4_highres_ablation import (
    CONFIG_PATH as OLD_CONFIG_PATH, _load_config as old_config,
    _split_indices, _training_config,
)
from scripts.train_totaltext_v4_image_frequency_ablation import (
    BASE_MODEL_SOURCE, CONFIG_PATH, TRANSFORM_SOURCE, VARIANTS, WAVELET_SOURCE,
    _config, _owner_map, _transform, _validation_geometry, proposal_metrics,
)
from segmentation.data import SegmentationDataset, load_segmentation_manifest
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
from segmentation.totaltext_image_frequency_augment import (
    letterbox_canvas_mask, remap_letterbox_native_box,
)
from segmentation.training import load_checkpoint, seed_everything


def _geometry_differences(cached_rows: list[dict], frozen_rows: list[dict]) -> list[dict]:
    """Compare native per-image metrics without tolerances or GT-based selection."""
    frozen = {row["sample_id"]: row for row in frozen_rows}
    cached = {row["sample_id"]: row for row in cached_rows}
    if (len(frozen) != len(frozen_rows) or len(cached) != len(cached_rows)
        or frozen.keys() != cached.keys()):
        raise RuntimeError("Validation geometry sample IDs are not identical")
    return [{"sample_id": sample_id,
             "frozen_batch_8": frozen[sample_id], "cached_batch_1": cached[sample_id]}
            for sample_id in sorted(frozen) if frozen[sample_id] != cached[sample_id]]


def _image(sample, root: Path) -> np.ndarray:
    path = root / sample.image_path
    if sha256_file(path) != sample.image_sha256:
        raise ValueError(f"TRAIN RGB changed: {sample.sample_id}")
    with Image.open(path) as opened:
        return np.ascontiguousarray(np.asarray(opened.convert("RGB")))


def _train_gt(sample, root: Path, image_wh: tuple[int, int]):
    number = sample.sample_id.split(":", 1)[1]
    path = root / "datasets/total_text/train/annotations" / f"{number}.txt"
    instances, _ = clip_instances_to_pixel_extent(parse_totaltext_annotation(path), image_wh)
    gt = [{"bbox": [item.bbox_xyxy[0], item.bbox_xyxy[1], item.bbox_xyxy[2] + 1,
                    item.bbox_xyxy[3] + 1], "text": item.transcription}
          for item in instances if not item.ignore]
    return gt, sha256_file(path)


def _tensor(rgb: np.ndarray, transform):
    source = Image.fromarray(rgb)
    # Validation never flips or rotates; the only possible geometric variant is letterbox.
    if transform.mode == "letterbox":
        from segmentation.totaltext_image_frequency_augment import letterbox_geometry
        source = letterbox_geometry(source, fill=tuple(transform.parameters["canvas_fill_rgb"]))
    source = TF.resize(source, [512, 512], interpolation=InterpolationMode.BILINEAR,
                       antialias=True)
    return TF.normalize(TF.to_tensor(source), (0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)).unsqueeze(0)


def _base_model_source_matches(recorded_sha256: str | None) -> bool:
    """Verify a historical single-head source despite an additive new class.

    This is not a relaxed hash check: after removing exactly the new, isolated
    two-head class, the remaining bytes must hash to the checkpoint's original
    source hash. Any other model-source drift still fails closed.
    """
    if not isinstance(recorded_sha256, str):
        return False
    if sha256_file(BASE_MODEL_SOURCE) == recorded_sha256:
        return True
    source = BASE_MODEL_SOURCE.read_bytes()
    start = b"\n\nclass PracticalTotalTextSeparatorHeadAGUNet"
    end = b"\n\n# ------------------------------- Quick Self-Test"
    if source.count(start) != 1 or source.count(end) != 1:
        return False
    before, remainder = source.split(start, 1)
    if end not in remainder:
        return False
    _, after = remainder.split(end, 1)
    historical = before + end + after
    return sha256(historical).hexdigest() == recorded_sha256


def _model_and_run(name, args, config, validation_sha):
    if name == "frozen_baseline":
        previous = old_config()
        run_dir = args.baseline_run.resolve()
        report = json.loads((run_dir / "run_manifest.json").read_text())
        contract = report["run_contract"]
        if (contract.get("variant") != "highres_separator_residual"
            or contract.get("config_sha256") != sha256_file(OLD_CONFIG_PATH)
            or contract.get("validation_ids_sha256") != validation_sha):
            raise ValueError("Frozen baseline does not match this validation population")
        model = PracticalTotalTextHighResAGUNet(skip_floor=0.25, encoder_pretrained=False)
        training = _training_config(previous, previous["variants"]["highres_separator_residual"])
    else:
        run_dir = args.training_runs.resolve() / name
        report = json.loads((run_dir / "run_manifest.json").read_text())
        contract = report["run_contract"]
        if (contract.get("variant") != name or contract.get("config_sha256") != sha256_file(CONFIG_PATH)
            or contract.get("validation_ids_sha256") != validation_sha
            or contract.get("transform_source_sha256") != sha256_file(TRANSFORM_SOURCE)
            or contract.get("wavelet_source_sha256") != sha256_file(WAVELET_SOURCE)
            or not _base_model_source_matches(contract.get("base_model_source_sha256"))):
            raise ValueError(f"Ablation run does not match this validation population: {name}")
        kind = config["variants"][name]["model"]
        cls = (PracticalTotalTextWaveletHighResAGUNet if kind == "haar_early_skip_adapter"
               else PracticalTotalTextHighResAGUNet)
        model = cls(skip_floor=0.25, encoder_pretrained=False)
        training = _training_config(config, {"separation_weight": 3.0})
    if report.get("status") != "SUCCESS" or not report.get("training_complete"):
        raise ValueError(f"Incomplete segmentation run: {name}")
    checkpoint = run_dir / "best.pt"
    if sha256_file(checkpoint) != report["checkpoint"]["sha256"]:
        raise ValueError(f"Checkpoint SHA256 mismatch: {name}")
    load_checkpoint(checkpoint, model=model, expected_architecture=model.architecture_identity,
        expected_run_contract=contract, expected_training_config=training)
    transform = _transform(config, name if name != "frozen_baseline" else "wavelet_haar",
                           train=False)
    return model, transform, report


def _correct(predictions, gt, *, normalize):
    edges = []
    for pi, prediction in enumerate(predictions):
        text = normalize_word(prediction["text"]) if normalize else prediction["text"].casefold()
        if not text:
            continue
        for gi, word in enumerate(gt):
            target = normalize_word(word["text"]) if normalize else word["text"].casefold()
            if target == text and box_iou(prediction["bbox"], word["bbox"]) >= 0.5:
                edges.append((pi, gi))
    return maximum_matching(len(predictions), len(gt), edges)


def _geometry_match(predictions, gt, threshold):
    return maximum_matching(len(predictions), len(gt), [
        (pi, gi) for pi, proposal in enumerate(predictions) for gi, word in enumerate(gt)
        if box_iou(proposal["bbox"], word["bbox"]) >= threshold])


def run(args):
    config = _config()
    root = args.target_root.resolve()
    manifest_path = root / config["train_manifest"]
    if sha256_file(manifest_path) != config["train_manifest_sha256"]:
        raise ValueError("Qualified Total-Text TRAIN manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "train":
        raise ValueError("Validation OCR permits TRAIN only")
    _, val_ids, val_sha = _split_indices(manifest, 202503, 125)
    if args.device == "cuda:0" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    if device.type == "cuda":
        _prepare_cuda_memory_tracking(device)
    out = args.output_dir.resolve()
    if out == Path(__file__).resolve().parents[1] or Path(__file__).resolve().parents[1] in out.parents:
        raise ValueError("OCR outputs must be outside repository")
    names = ("frozen_baseline", *VARIANTS)
    protocol = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
        "split": "TRAIN_heldout_125", "validation_ids_sha256": val_sha,
        "train_manifest_sha256": config["train_manifest_sha256"],
        "ablation_config_sha256": sha256_file(CONFIG_PATH),
        "baseline_config_sha256": sha256_file(OLD_CONFIG_PATH),
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_runtime_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "segmentation": "sigmoid>=0.5 native256; no area, no top10, pad0",
        "evaluation": "IoU>=0.5 one-to-one spatial exact, casefold and Unicode P/S-stripped",
        "ground_truth_use": "post-hoc only"}
    protocol_sha = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = out / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Existing validation OCR protocol differs")
    else:
        _atomic_json(protocol_path, protocol)
    shards = {}
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    for name in names:
        model, transform, run_report = _model_and_run(name, args, config, val_sha)
        model.eval().to(device)
        part = {}
        shard_dir = out / name / "samples"
        try:
            for pos, index in enumerate(val_ids, 1):
                sample = manifest.samples[index]
                sample_key = sample.sample_id.replace(":", "__")
                shard_path = shard_dir / f"{sample_key}.json"
                mask_path = out / name / "masks" / f"{sample_key}.png"
                annotation_path = root / "datasets/total_text/train/annotations" / (
                    sample.sample_id.split(":", 1)[1] + ".txt")
                if shard_path.exists():
                    shard = json.loads(shard_path.read_text())
                    if (shard.get("sample_id") != sample.sample_id
                        or shard.get("protocol_sha256") != protocol_sha
                        or shard.get("checkpoint_sha256") != run_report["checkpoint"]["sha256"]
                        or shard.get("image_sha256") != sample.image_sha256
                        or shard.get("annotation_sha256") != sha256_file(annotation_path)
                        or sha256_file(mask_path) != shard.get("native_mask_file_sha256")):
                        raise ValueError(f"Cached validation shard provenance mismatch: {shard_path}")
                else:
                    rgb = _image(sample, root)
                    with torch.inference_mode():
                        logits = model(_tensor(rgb, transform).to(device))
                    if logits.shape != (1, 1, 256, 256):
                        raise RuntimeError(f"Wrong native logits shape: {name}")
                    native = (torch.sigmoid(logits[0, 0]) >= 0.5).byte().cpu().numpy()
                    del logits
                    if transform.mode == "letterbox":
                        native = np.ascontiguousarray(native *
                            letterbox_canvas_mask((rgb.shape[1], rgb.shape[0])).byte().numpy())
                    mask_sha = _save_mask(mask_path, native)
                    raw = contour_proposals(native, rgb.shape[:2], [0])["0"]
                    if transform.mode == "letterbox":
                        for proposal in raw["proposals"]:
                            proposal["bbox_original"] = remap_letterbox_native_box(
                                proposal["bbox_native"], (rgb.shape[1], rgb.shape[0]))
                            proposal["bbox"] = remap_letterbox_native_box(
                                proposal["padded_bbox_native"], (rgb.shape[1], rgb.shape[0]))
                    for proposal in raw["proposals"]:
                        x1, y1, x2, y2 = proposal["bbox"]
                        proposal["crop_rgb_sha256"] = crop_array_hash(
                            np.ascontiguousarray(rgb[y1:y2, x1:x2]))
                    gt, annotation_sha = _train_gt(sample, root, (rgb.shape[1], rgb.shape[0]))
                    shard = {"schema_version": 1, "sample_id": sample.sample_id,
                        "protocol_sha256": protocol_sha,
                        "image_sha256": sample.image_sha256,
                        "annotation_sha256": annotation_sha,
                        "checkpoint_sha256": run_report["checkpoint"]["sha256"],
                        "native_mask_file_sha256": mask_sha,
                        "native_mask_array_sha256": sha256(native.tobytes()).hexdigest(),
                        "ground_truth": gt, "raw_contours": raw["raw_contours"],
                        "proposals": raw["proposals"]}
                    _atomic_json(shard_path, shard)
                part[sample.sample_id] = shard
                if pos % 25 == 0:
                    print(f"[mask {name}] {pos}/{len(val_ids)}", flush=True)
        finally:
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        shards[name] = part
    pending = []
    seen = set()
    for index in val_ids:
        sample_id = manifest.samples[index].sample_id
        for name in names:
            for proposal in shards[name][sample_id]["proposals"]:
                crop_sha = proposal["crop_rgb_sha256"]
                if crop_sha in seen:
                    continue
                seen.add(crop_sha)
                cache = out / "ocr" / f"{crop_sha}.json"
                if cache.exists():
                    record = json.loads(cache.read_text())
                    if (record.get("crop_rgb_sha256") != crop_sha
                        or record.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
                        or record.get("parseq_source_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
                        or not isinstance(record.get("raw_text"), str)):
                        raise ValueError(f"OCR cache provenance mismatch: {cache}")
                else:
                    pending.append((sample_id, proposal["bbox"], crop_sha))
    print(f"[PARSeq] unique_crops={len(seen)} pending={len(pending)}", flush=True)
    if pending:
        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
            runtime_source=args.parseq_source, checkpoint=args.parseq_checkpoint,
            runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=args.parseq_source_archive, device=args.device))
        try:
            for start in range(0, len(pending), args.parseq_batch_size):
                batch = pending[start:start + args.parseq_batch_size]
                crops = []
                rgb_by_sample = {}
                for sample_id, bbox, expected_hash in batch:
                    if sample_id not in rgb_by_sample:
                        rgb_by_sample[sample_id] = _image(by_id[sample_id], root)
                    rgb = rgb_by_sample[sample_id]
                    x1, y1, x2, y2 = bbox
                    crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
                    if crop_array_hash(crop) != expected_hash:
                        raise ValueError(f"Validation crop changed: {sample_id}")
                    crops.append(crop)
                predictions = recognizer.recognize_candidate_batch(crops)
                if len(predictions) != len(batch):
                    raise RuntimeError("PARSeq returned the wrong batch length")
                for (_, _, crop_sha), value in zip(batch, predictions):
                    _atomic_json(out / "ocr" / f"{crop_sha}.json", {
                        "schema_version": 1, "crop_rgb_sha256": crop_sha,
                        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                        **value.to_dict()})
                print(f"[PARSeq] {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
        finally:
            recognizer.release()
    report = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "protocol": protocol, "variants": {}, "images": len(val_ids),
              "geometry_batching_audits": {},
              "runtime": {"device": str(device), "segmentation_dtype": "float32",
                          "segmentation_batch_size": 1,
                          "parseq_batch_size": args.parseq_batch_size,
                          "model_lifecycle": "sequential_release_before_parseq",
                          "gpu_peak_allocated_bytes": (torch.cuda.max_memory_allocated(device)
                                                       if device.type == "cuda" else None),
                          "gpu_peak_reserved_bytes": (torch.cuda.max_memory_reserved(device)
                                                      if device.type == "cuda" else None)}}
    denominators = set()
    geometry_dataset = SegmentationDataset(manifest, root, transform=None)
    for name in names:
        totals = {"gt_words": 0, "proposals": 0, "geometry_iou_ge_25": 0,
                  "geometry_iou_ge_50": 0, "geometry_iou_ge_75": 0,
                  "exact_casefold_iou_ge_50": 0, "exact_no_symbols_iou_ge_50": 0}
        native_geometry = None
        cached_geometry_rows = []
        rows = []
        transform = _transform(config, name if name != "frozen_baseline" else "wavelet_haar",
                               train=False)
        geometry_dataset.transform = transform
        for index in val_ids:
            sample_id = manifest.samples[index].sample_id
            shard = shards[name][sample_id]
            gt = shard["ground_truth"]
            sample_key = sample_id.replace(":", "__")
            native_path = out / name / "masks" / f"{sample_key}.png"
            native = np.asarray(Image.open(native_path).convert("L")) > 0
            dataset_item = geometry_dataset[index]
            owners, expected_owner_ids = _owner_map(manifest.samples[index], root, transform)
            owners[(dataset_item["mask"][0].numpy() == 0)
                   | (dataset_item["validity"][0].numpy() == 0)] = 0
            word_geometry = proposal_metrics(native, owners, expected_owner_ids=expected_owner_ids)
            if native_geometry is None:
                native_geometry = {key: 0 for key in word_geometry}
            for key, value in word_geometry.items():
                native_geometry[key] += value
            cached_geometry_rows.append({"sample_id": sample_id, **word_geometry})
            predictions = []
            for proposal in shard["proposals"]:
                crop_sha = proposal["crop_rgb_sha256"]
                text = json.loads((out / "ocr" / f"{crop_sha}.json").read_text())["raw_text"]
                predictions.append({"bbox": proposal["bbox"], "text": text,
                                    "crop_rgb_sha256": crop_sha})
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
                         "native_word_geometry": word_geometry,
                         "matched_gt_indices": {key: sorted(ids) for key, ids in matched.items()}})
        denominators.add(totals["gt_words"])
        totals["exact_no_symbols_rate_all_gt"] = (
            totals["exact_no_symbols_iou_ge_50"] / totals["gt_words"] if totals["gt_words"] else 0.0)
        if native_geometry["gt_words"] != totals["gt_words"]:
            raise RuntimeError(f"Native-owner and annotation GT denominators differ: {name}")
        run_manifest = (args.baseline_run if name == "frozen_baseline"
                        else args.training_runs / name) / "run_manifest.json"
        trained = json.loads(run_manifest.read_text())
        recorded = (trained["validation_word_topology"] if name == "frozen_baseline"
                    else trained["validation_word_geometry"])
        frozen_geometry_path = (args.training_runs / "frozen_baseline_validation_geometry.json"
                                if name == "frozen_baseline" else
                                args.training_runs / name / "validation_word_geometry.json")
        frozen_geometry_file = json.loads(frozen_geometry_path.read_text())
        frozen_geometry = (frozen_geometry_file["geometry"] if name == "frozen_baseline"
                           else frozen_geometry_file)
        reference_aggregate = frozen_geometry["aggregate"]
        if any(reference_aggregate.get(key) != value for key, value in recorded.items()):
            raise RuntimeError(f"Frozen training geometry report disagrees with run manifest: {name}")
        differences = _geometry_differences(cached_geometry_rows, frozen_geometry["per_image"])
        if differences or native_geometry != reference_aggregate:
            # Training validated with batch_size=8; crop inference uses batch_size=1.
            # Reproduce the frozen batch-8 geometry exactly before accepting any
            # batch-size-dependent difference. Cached masks and OCR are untouched.
            by_position = {sample.sample_id: position for position, sample in
                           enumerate(manifest.samples)}
            for difference in differences:
                sample_id = difference["sample_id"]
                sample = by_id[sample_id]
                rgb = _image(sample, root)
                if not torch.equal(geometry_dataset[by_position[sample_id]]["image"],
                                   _tensor(rgb, transform)[0]):
                    raise RuntimeError(f"Validation preprocessing drift: {name} {sample_id}")
            seed_everything(int(config["training"]["seed"]),
                            deterministic_algorithms=bool(config["training"]["deterministic_algorithms"]))
            verification_model, _, _ = _model_and_run(name, args, config, val_sha)
            verification_model.to(device).eval()
            try:
                verification_loader = DataLoader(
                    Subset(SegmentationDataset(manifest, root, transform=transform), val_ids),
                    batch_size=int(config["training"]["batch_size"]), shuffle=False,
                    num_workers=0, pin_memory=device.type == "cuda")
                replayed = _validation_geometry(verification_model, verification_loader,
                                                 manifest, root, transform, device=device)
            finally:
                del verification_model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            replay_differences = _geometry_differences(replayed["per_image"],
                                                       frozen_geometry["per_image"])
            audit_path = out / "geometry_batching_audits" / f"{name}.json"
            audit = {"schema_version": 1, "variant": name,
                     "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
                     "checkpoint_sha256": trained["checkpoint"]["sha256"],
                     "frozen_geometry_sha256": sha256_file(frozen_geometry_path),
                     "cached_inference_batch_size": 1,
                     "training_validation_batch_size": config["training"]["batch_size"],
                     "cached_aggregate": native_geometry,
                     "frozen_aggregate": reference_aggregate,
                     "replayed_batch_8_aggregate": replayed["aggregate"],
                     "cached_vs_frozen_per_image": differences,
                     "replay_vs_frozen_per_image": replay_differences,
                     "status": "VERIFIED_BATCH_SIZE_DIFFERENCE" if not replay_differences
                         and replayed["aggregate"] == reference_aggregate else "UNRESOLVED_GEOMETRY_DRIFT"}
            if audit_path.exists():
                if json.loads(audit_path.read_text()) != audit:
                    raise RuntimeError(f"Existing batch-size audit differs; refusing overwrite: {audit_path}")
            else:
                _atomic_json(audit_path, audit)
            if audit["status"] != "VERIFIED_BATCH_SIZE_DIFFERENCE":
                raise RuntimeError(f"Frozen validation geometry not reproduced by batch-8 replay: {audit_path}")
            report["geometry_batching_audits"][name] = {
                "path": str(audit_path), "sha256": sha256_file(audit_path),
                "affected_images": len(differences), "status": audit["status"]}
        report["variants"][name] = {"checkpoint_sha256": shards[name][manifest.samples[val_ids[0]].sample_id]["checkpoint_sha256"],
                                    "aggregate": totals, "native_word_geometry": native_geometry,
                                    "per_image": rows}
    if len(denominators) != 1:
        raise RuntimeError("GT denominator differs across validation variants")
    _atomic_json(out / "results.json", report)
    print(json.dumps({"result": str(out / "results.json"),
        "sha256": sha256_file(out / "results.json"),
        "summary": {name: value["aggregate"] for name, value in report["variants"].items()}},
        indent=2, sort_keys=True), flush=True)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--training-runs", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--parseq-batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.parseq_batch_size < 1:
        parser.error("--parseq-batch-size must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
