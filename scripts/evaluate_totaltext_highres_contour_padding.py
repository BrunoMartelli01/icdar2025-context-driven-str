"""Restartable TEST-300 contour/padding/PARSeq diagnostic for three native-256 nets.

This is an exploratory PRACTICAL_APPROXIMATION, not Algorithm 1 or Table 5.
Predictions and OCR are cached before GT is used for post-hoc matching. No
area filter or top-10 cap is applied, as requested for Total-Text diagnostics.
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
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from recognition.scene_text_datasets import parse_totaltext_annotation
from recognition.scene_text_v1_evaluation import box_iou, evaluate_images, maximum_matching
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from scripts.train_totaltext_v4_highres_ablation import _load_config as _train_config
from scripts.train_totaltext_v4_highres_ablation import _training_config
from models.unet_adv import PracticalTotalTextHighResAGUNet
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
from segmentation.training import load_checkpoint


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs/approximate/totaltext_v4_highres_contour_padding_test_v1.json"
TRAIN_CONFIG_PATH = REPO / "configs/approximate/totaltext_v4_highres_ablation_v1.json"


def _config() -> dict:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if (value.get("classification") != "PRACTICAL_APPROXIMATION"
        or value.get("reproduction_mode") != "approximate"
        or value.get("variants") != ["highres_baseline", "highres_separator", "highres_separator_residual"]
        or value.get("mask_native_shape") != [256, 256]
        or value.get("probability_threshold") != 0.5
        or value.get("padding_native_pixels") != [0, 2, 4, 6, 8, 12]
        or value.get("area_filter") != "disabled"
        or value.get("maximum_proposals") != "disabled"):
        raise ValueError("The predeclared Total-Text contour/padding protocol changed")
    return value


def _sample_key(sample_id: str) -> str:
    return sample_id.replace(":", "__").replace("/", "_")


def remap_native256_box(box: list[int], image_hw: tuple[int, int]) -> list[int]:
    """Map exclusive native-256 xyxy to original RGB, floor/ceil at edges."""
    height, width = image_hw
    x1, y1, x2, y2 = (int(value) for value in box)
    if min(height, width) < 1 or not (0 <= x1 < x2 <= 256 and 0 <= y1 < y2 <= 256):
        raise ValueError("Invalid native-256 box or original image dimensions")
    return [x1 * width // 256, y1 * height // 256,
            (x2 * width + 255) // 256, (y2 * height + 255) // 256]


def crop_array_hash(rgb: np.ndarray) -> str:
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 or not rgb.size:
        raise ValueError("Expected nonempty uint8 RGB crop")
    value = np.ascontiguousarray(rgb)
    return sha256(np.asarray(value.shape, dtype="<i8").tobytes() + value.tobytes()).hexdigest()


def contour_proposals(native: np.ndarray, image_hw: tuple[int, int], pads: list[int]) -> dict[str, dict]:
    """No GT, area filter, top-10, grouping, or mask resampling."""
    if native.shape != (256, 256) or native.dtype != np.uint8 or not np.isin(native, (0, 1)).all():
        raise ValueError("Expected binary native-256 prediction")
    if sorted(set(pads)) != pads or any(pad < 0 for pad in pads):
        raise ValueError("Paddings must be unique, sorted, nonnegative native pixels")
    output = {}
    raw_count = None
    for pad in pads:
        audit = audit_repository_contours(native, RepositoryContourParams(0, pad, native.size))
        if raw_count is None:
            raw_count = audit["raw_external_contours"]
        elif raw_count != audit["raw_external_contours"]:
            raise RuntimeError("Padding unexpectedly changed raw contour count")
        rows = [{"rank": item["rank"], "contour_index": item["contour_index"],
                 "bbox_native": item["bbox"], "padded_bbox_native": item["padded_bbox"],
                 "bbox_original": remap_native256_box(item["bbox"], image_hw),
                 "bbox": remap_native256_box(item["padded_bbox"], image_hw)}
                for item in audit["retained"]]
        if len(rows) != raw_count or audit["rejected"]:
            raise RuntimeError("An unexpected contour filter/cap was applied")
        output[str(pad)] = {"raw_contours": raw_count, "proposals": rows}
    return output


def correct_gt_indices(predictions: list[dict], gt: list[dict], *, casefold: bool = True) -> set[int]:
    edges = [(prediction_index, gt_index)
             for prediction_index, prediction in enumerate(predictions)
             for gt_index, word in enumerate(gt)
             if box_iou(prediction["bbox"], word["bbox"]) >= 0.5
             and (prediction["text"].casefold() == word["text"].casefold()
                  if casefold else prediction["text"] == word["text"])]
    return maximum_matching(len(predictions), len(gt), edges)


def padding_overlap(correct: dict[str, set[str]], pads: list[int]) -> dict:
    keys = [str(pad) for pad in pads]
    if set(correct) != set(keys):
        raise ValueError("Missing padding correctness set")
    union = set().union(*(correct[key] for key in keys))
    frequency = {str(count): 0 for count in range(1, len(keys) + 1)}
    for word in union:
        frequency[str(sum(word in correct[key] for key in keys))] += 1
    return {
        "correct_per_padding": {key: len(correct[key]) for key in keys},
        "correct_with_any_padding": len(union),
        "gain_any_padding_vs_best_single": len(union) - max(map(lambda key: len(correct[key]), keys)),
        "frequency_across_paddings": frequency,
        "unique_to_one_padding": {key: len(correct[key] - set().union(
            *(correct[other] for other in keys if other != key))) for key in keys},
        "recovered_vs_padding_0": {key: len(correct[key] - correct["0"]) for key in keys},
        "lost_vs_padding_0": {key: len(correct["0"] - correct[key]) for key in keys},
        "intersection_matrix": {left: {right: len(correct[left] & correct[right])
                                      for right in keys} for left in keys},
    }


def _protocol(args: argparse.Namespace, config: dict, runs: dict, test_sha: str) -> dict:
    return {
        "schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
        "result_class": config["result_class"], "dataset": "total_text", "split": "official_TEST_300",
        "config_sha256": file_sha256(CONFIG_PATH),
        "training_config_sha256": file_sha256(TRAIN_CONFIG_PATH),
        "test_manifest_sha256": test_sha,
        "checkpoints": {name: {"sha256": run["checkpoint"]["sha256"],
                               "epoch": run["checkpoint"]["epoch"]}
                        for name, run in runs.items()},
        "native_mask": "sigmoid(logits) >= 0.5 on true 256x256 output; no resize",
        "contours": "cv2.findContours(RETR_EXTERNAL,CHAIN_APPROX_SIMPLE); area filter OFF; top10 OFF",
        "padding_native_pixels": config["padding_native_pixels"],
        "crop": "native padded bbox remapped with floor/ceil to original RGB, no warping",
        "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
        "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
        "evaluation": config["evaluation"],
        "ground_truth_use": "post-hoc only; never used for segmentation, contour localization or PARSeq",
        "test_padding_selection": "prohibited; exploratory TEST comparison, not a tuned deployment rule",
        "device": args.device, "parseq_batch_size": args.parseq_batch_size,
    }


def _runs(root: Path, config: dict) -> dict:
    train_config = _train_config()
    train_sha = file_sha256(TRAIN_CONFIG_PATH)
    output = {}
    for name in config["variants"]:
        manifest_path = root / name / "run_manifest.json"
        run = json.loads(manifest_path.read_text(encoding="utf-8"))
        contract = run.get("run_contract", {})
        if (run.get("status") != "SUCCESS" or not run.get("training_complete")
            or run.get("train_images") != 1130 or run.get("validation_images") != 125
            or len(run.get("epochs", [])) != 500
            or contract.get("variant") != name or contract.get("config_sha256") != train_sha
            or contract.get("input_shape") != [3, 512, 512]
            or contract.get("logits_shape") != [1, 256, 256]
            or contract.get("separation_weight") != train_config["variants"][name]["separation_weight"]
            or contract.get("skip_floor") != train_config["variants"][name]["skip_floor"]):
            raise ValueError(f"Incomplete or incompatible high-resolution training run: {name}")
        checkpoint = root / name / "best.pt"
        if file_sha256(checkpoint) != run["checkpoint"]["sha256"]:
            raise ValueError(f"Selected checkpoint SHA256 mismatch: {name}")
        output[name] = run
    if len({run["run_contract"]["validation_ids_sha256"] for run in output.values()}) != 1:
        raise ValueError("The three training variants did not use the same validation split")
    return output


def _rgb(sample, data_root: Path) -> tuple[np.ndarray, Path]:
    path = data_root / sample.image_path
    if file_sha256(path) != sample.image_sha256:
        raise ValueError(f"TEST image SHA256 mismatch: {sample.sample_id}")
    with Image.open(path) as opened:
        return np.ascontiguousarray(np.asarray(opened.convert("RGB"))), path


def _input_tensor(rgb: np.ndarray) -> torch.Tensor:
    image = Image.fromarray(rgb)
    image = TF.resize(image, [512, 512], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return TF.normalize(TF.to_tensor(image), mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)).unsqueeze(0)


def _ground_truth(sample, data_root: Path, image_wh: tuple[int, int]) -> tuple[list[dict], str]:
    path = _annotation_path(sample, data_root)
    annotation_sha = file_sha256(path)
    instances, _ = clip_instances_to_pixel_extent(parse_totaltext_annotation(path), image_wh)
    gt = [{"bbox": [item.bbox_xyxy[0], item.bbox_xyxy[1], item.bbox_xyxy[2] + 1,
                    item.bbox_xyxy[3] + 1], "text": item.transcription}
          for item in instances if not item.ignore]
    return gt, annotation_sha


def _annotation_path(sample, data_root: Path) -> Path:
    number = sample.sample_id.split(":", 1)[1]
    return data_root / "datasets/total_text/test/annotations" / f"{number}.txt"


def _save_mask(path: Path, native: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".png.tmp")
    Image.fromarray(native * 255, mode="L").save(temporary, format="PNG")
    if path.exists():
        if not np.array_equal(np.asarray(Image.open(path)) > 0, native > 0):
            raise ValueError(f"Existing native mask differs; refusing overwrite: {path}")
        temporary.unlink()
    else:
        temporary.replace(path)
    return file_sha256(path)


def _verify_shard(path: Path, mask_path: Path, protocol_sha: str, sample_id: str, image_sha: str,
                  annotation_path: Path, pads: list[int]) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (value.get("protocol_sha256") != protocol_sha or value.get("sample_id") != sample_id
        or value.get("image_sha256") != image_sha
        or value.get("annotation_sha256") != file_sha256(annotation_path)
        or set(value.get("paddings", {})) != {str(pad) for pad in pads}
        or file_sha256(mask_path) != value.get("native_mask_file_sha256")):
        raise ValueError(f"Cached TEST sample differs from run protocol: {path}")
    return value


def _prepare(args, manifest, runs: dict, protocol_sha: str, pads: list[int]) -> dict:
    output = {}
    device = torch.device(args.device)
    config = _train_config()
    for variant, run in runs.items():
        variant_dir = args.output_dir / variant
        cached = {}
        pending = []
        for sample in manifest.samples:
            key = _sample_key(sample.sample_id)
            shard_path = variant_dir / "samples" / f"{key}.json"
            mask_path = variant_dir / "masks" / f"{key}.png"
            if shard_path.exists():
                cached[sample.sample_id] = _verify_shard(
                    shard_path, mask_path, protocol_sha, sample.sample_id, sample.image_sha256,
                    _annotation_path(sample, args.data_root), pads)
            else:
                pending.append(sample)
        if pending:
            model = PracticalTotalTextHighResAGUNet(
                skip_floor=float(config["variants"][variant]["skip_floor"]), encoder_pretrained=False)
            load_checkpoint(args.training_runs / variant / "best.pt", model=model,
                            expected_architecture=PracticalTotalTextHighResAGUNet.PRACTICAL_IDENTITY,
                            expected_run_contract=run["run_contract"],
                            expected_training_config=_training_config(config, config["variants"][variant]))
            model.eval().to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            try:
                for index, sample in enumerate(pending, 1):
                    rgb, _ = _rgb(sample, args.data_root)
                    with torch.inference_mode():
                        logits = model(_input_tensor(rgb).to(device))
                    if tuple(logits.shape) != (1, 1, 256, 256):
                        raise RuntimeError(f"Unexpected native logits for {variant}: {tuple(logits.shape)}")
                    native = np.ascontiguousarray((torch.sigmoid(logits[0, 0]) >= 0.5).byte().cpu().numpy())
                    key = _sample_key(sample.sample_id)
                    mask_path = variant_dir / "masks" / f"{key}.png"
                    mask_sha = _save_mask(mask_path, native)
                    proposals_by_pad = contour_proposals(native, rgb.shape[:2], pads)
                    for pad_data in proposals_by_pad.values():
                        for proposal in pad_data["proposals"]:
                            x1, y1, x2, y2 = proposal["bbox"]
                            proposal["crop_rgb_sha256"] = crop_array_hash(rgb[y1:y2, x1:x2])
                    # Annotation entry occurs only after predictions and crop identities are frozen.
                    gt, annotation_sha = _ground_truth(sample, args.data_root, (rgb.shape[1], rgb.shape[0]))
                    shard = {"schema_version": 1, "protocol_sha256": protocol_sha,
                             "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                             "annotation_sha256": annotation_sha,
                             "native_mask_file_sha256": mask_sha,
                             "native_mask_array_sha256": sha256(native.tobytes()).hexdigest(),
                             "ground_truth": gt, "paddings": proposals_by_pad}
                    _atomic_json(variant_dir / "samples" / f"{key}.json", shard)
                    cached[sample.sample_id] = shard
                    if index % 10 == 0 or index == len(pending):
                        print(f"[mask {variant}] {index}/{len(pending)} new; {len(cached)}/300 total", flush=True)
            finally:
                del model
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            runtime = {"variant": variant, "device": str(device), "dtype": "float32",
                       "batch_size": 1, "input_shape": [1, 3, 512, 512],
                       "logits_shape": [1, 1, 256, 256],
                       "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
            runtime_path = variant_dir / "inference_runtime.json"
            if runtime_path.exists():
                previous = json.loads(runtime_path.read_text(encoding="utf-8"))
                if any(previous.get(key) != runtime[key] for key in ("variant", "device", "dtype", "batch_size", "input_shape", "logits_shape")):
                    raise ValueError("Existing inference runtime differs")
            else:
                _atomic_json(runtime_path, runtime)
        else:
            print(f"[mask {variant}] verified cached 300/300", flush=True)
        output[variant] = cached
    return output


def _recognize(args, manifest, shards: dict, protocol_sha: str, pads: list[int]) -> None:
    by_id = {sample.sample_id: sample for sample in manifest.samples}
    pending = []
    seen = set()
    total = 0
    for sample in manifest.samples:
        for variant in shards:
            shard = shards[variant][sample.sample_id]
            for pad in pads:
                for proposal in shard["paddings"][str(pad)]["proposals"]:
                    total += 1
                    crop_sha = proposal["crop_rgb_sha256"]
                    if crop_sha in seen:
                        continue
                    seen.add(crop_sha)
                    path = args.output_dir / "ocr" / f"{crop_sha}.json"
                    if path.exists():
                        cached = json.loads(path.read_text(encoding="utf-8"))
                        if (cached.get("crop_rgb_sha256") != crop_sha
                            or cached.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256
                            or cached.get("parseq_source_archive_sha256") != PARSEQ_RUNTIME_ARCHIVE_SHA256
                            or "raw_text" not in cached):
                            raise ValueError(f"OCR cache provenance mismatch: {path}")
                    else:
                        pending.append((sample.sample_id, proposal["bbox"], crop_sha))
    print(f"[PARSeq] proposals={total} unique_crops={len(seen)} pending={len(pending)}", flush=True)
    if not pending:
        return
    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
        runtime_source=args.parseq_source, checkpoint=args.parseq_checkpoint,
        runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
        runtime_source_archive=args.parseq_source_archive, device=args.device))
    last_id = None
    rgb = None
    try:
        for start in range(0, len(pending), args.parseq_batch_size):
            batch = pending[start:start + args.parseq_batch_size]
            crops = []
            for sample_id, bbox, crop_sha in batch:
                if sample_id != last_id:
                    rgb, _ = _rgb(by_id[sample_id], args.data_root)
                    last_id = sample_id
                x1, y1, x2, y2 = bbox
                crop = np.ascontiguousarray(rgb[y1:y2, x1:x2])
                if crop_array_hash(crop) != crop_sha:
                    raise ValueError(f"Crop changed after proposal freeze: {sample_id}")
                crops.append(crop)
            predictions = recognizer.recognize_candidate_batch(crops)
            if len(predictions) != len(batch):
                raise RuntimeError("PARSeq batch size mismatch")
            for (_, _, crop_sha), prediction in zip(batch, predictions):
                _atomic_json(args.output_dir / "ocr" / f"{crop_sha}.json", {
                    "schema_version": 1, "crop_rgb_sha256": crop_sha,
                    "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
                    "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
                    **prediction.to_dict(),
                })
            print(f"[PARSeq] {min(start + len(batch), len(pending))}/{len(pending)}", flush=True)
    finally:
        recognizer.release()


def _summarize(args, manifest, shards: dict, protocol: dict, pads: list[int]) -> dict:
    variants = {}
    gt_denominator = None
    for variant, per_sample in shards.items():
        padding_results = {}
        correct_by_pad = {}
        for pad in pads:
            rows = []
            correct = set()
            raw_contours = 0
            for sample in manifest.samples:
                shard = per_sample[sample.sample_id]
                gt = shard["ground_truth"]
                pad_data = shard["paddings"][str(pad)]
                predictions = []
                for proposal in pad_data["proposals"]:
                    crop_sha = proposal["crop_rgb_sha256"]
                    ocr = json.loads((args.output_dir / "ocr" / f"{crop_sha}.json").read_text(encoding="utf-8"))
                    predictions.append({"rank": proposal["rank"], "bbox": proposal["bbox"],
                                        "text": ocr["raw_text"],
                                        "confidence": ocr["sequence_confidence_native"]})
                rows.append({"sample_id": sample.sample_id, "ground_truth": gt,
                             "predictions": predictions})
                raw_contours += pad_data["raw_contours"]
                correct.update(f"{sample.sample_id}:{index}" for index in correct_gt_indices(predictions, gt))
            metrics = evaluate_images(rows, text_available=True)
            count = metrics["geometry"]["gt_instances"]
            if gt_denominator is None:
                gt_denominator = count
            if count != gt_denominator or metrics["recognition"]["exact_casefold_spatial_one_to_one"] != len(correct):
                raise RuntimeError("Padding metrics disagree with exact one-to-one GT identity sets")
            padding_results[str(pad)] = {"padding_native_pixels": pad, "raw_contours": raw_contours,
                                         **metrics}
            correct_by_pad[str(pad)] = correct
        variants[variant] = {"paddings": padding_results,
                             "exact_word_intersections": padding_overlap(correct_by_pad, pads),
                             "correct_word_ids_by_padding": {key: sorted(values)
                                                             for key, values in correct_by_pad.items()}}
    if gt_denominator != 2204:
        raise ValueError(f"Official Total-Text care-word denominator changed: {gt_denominator}")
    return {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
            "protocol": protocol, "images": len(manifest.samples), "gt_words": gt_denominator,
            "variants": variants}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--training-runs", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path, required=True)
    parser.add_argument("--parseq-source-archive", type=Path, required=True)
    parser.add_argument("--parseq-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--parseq-batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.parseq_batch_size < 1 or (args.device == "cuda:0" and not torch.cuda.is_available()):
        raise ValueError("Invalid PARSeq batch size or CUDA unavailable")
    config = _config()
    pads = config["padding_native_pixels"]
    if file_sha256(args.test_manifest) != config["test_manifest_sha256"]:
        raise ValueError("Official Total-Text TEST manifest SHA256 mismatch")
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "test" or len(manifest.samples) != 300:
        raise ValueError("Expected official Total-Text TEST-300 manifest")
    runs = _runs(args.training_runs, config)
    if (file_sha256(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256
        or file_sha256(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError("Qualified PARSeq checkpoint/runtime archive SHA256 mismatch")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == REPO or REPO in args.output_dir.parents:
        raise ValueError("Large diagnostic output must be outside the repository")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _protocol(args, config, runs, file_sha256(args.test_manifest))
    protocol_path = args.output_dir / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Existing output belongs to a different protocol")
    else:
        _atomic_json(protocol_path, protocol)
    protocol_sha = file_sha256(protocol_path)
    result_path = args.output_dir / "results.json"
    shards = _prepare(args, manifest, runs, protocol_sha, pads)
    _recognize(args, manifest, shards, protocol_sha, pads)
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("protocol") != protocol or result.get("images") != 300:
            raise ValueError("Existing completed result has incompatible provenance")
        print("[reuse]", result_path, "SHA256", file_sha256(result_path), flush=True)
        return 0
    result = _summarize(args, manifest, shards, protocol, pads)
    _atomic_json(result_path, result)
    print("[summary]", json.dumps({variant: {pad: {
        "iou_0_5": item["geometry"]["iou_ge_0_50_one_to_one"],
        "exact_casefold": item["recognition"]["exact_casefold_spatial_one_to_one"]}
        for pad, item in data["paddings"].items()}
        for variant, data in result["variants"].items()}, sort_keys=True), flush=True)
    print("JSON", result_path, "SHA256", file_sha256(result_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
