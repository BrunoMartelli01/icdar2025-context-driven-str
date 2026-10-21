"""TEST-300 PARSeq diagnostic from frozen v4 predicted binary masks.

The U-Net prediction exists only at native 128x128. Resolutions above 128 are
bilinear upscales of that *binary* mask, not new high-resolution U-Net outputs.
All branches remove the area filter and top-10 limit and retain zero padding.
This is a PRACTICAL_APPROXIMATION diagnostic, not a paper Table 5 result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_totaltext_v4_curved_crops import _atomic_json, _mask_from_cache
from scripts.sweep_totaltext_v4_gt_mask_resolution import (
    RESOLUTIONS, EXPECTED_TEST_MANIFEST,
    _geometry, _proposals, _sha_array,
)
from segmentation.data import file_sha256, load_segmentation_manifest


EXPECTED_PREDICTED_RESULTS = "f762aa83f276e0282babbed48dc5ad407e3edfe7da4c74fb9bf6849ee640c740"
EXPECTED_FROZEN_V4_RESULTS = "272613b094777b9836a1ffde2fd9265fcec17c84e7ccb60661f9a87ac0e29989"
EXPECTED_SEGMENTATION_CHECKPOINT = "e31a493d2752734d9d75f35a8cec077ff779655145be7735c0e3267ceea64985"
EXPECTED_PROTOCOL_CONFIG = "1706c1eca870067979819b6413b61f889c2e02c220a7433b7792b0576386a5a0"


def resize_predicted_binary(native: np.ndarray, resolution: int) -> np.ndarray:
    """Interpolate the frozen binary prediction; never imply new model detail."""
    if (native.shape != (128, 128) or native.dtype != np.uint8
        or not np.isin(native, (0, 1)).all() or resolution not in RESOLUTIONS):
        raise ValueError("Expected canonical native128 binary prediction and scheduled resolution")
    if resolution == 128:
        return np.ascontiguousarray(native.copy())
    resized = Image.fromarray(native * 255, mode="L").resize(
        (resolution, resolution), Image.Resampling.BILINEAR
    )
    return np.ascontiguousarray((np.asarray(resized) >= 128).astype(np.uint8))


def _protocol(args) -> dict:
    return {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "Total-Text TEST frozen-prediction resolution diagnostic; not paper Table 5",
        "split": "official TEST-300; 2204 care GT words",
        "test_manifest_sha256": EXPECTED_TEST_MANIFEST,
        "segmentation_checkpoint_sha256": EXPECTED_SEGMENTATION_CHECKPOINT,
        "predicted_results_sha256": EXPECTED_PREDICTED_RESULTS,
        "frozen_v4_results_sha256": EXPECTED_FROZEN_V4_RESULTS,
        "config_path": "configs/approximate/totaltext_v4_predicted_binary_resolution_no_filters_v1.json",
        "config_sha256": EXPECTED_PROTOCOL_CONFIG,
        "source": "frozen v4 U-Net sigmoid>=0.5 binary native128 TEST mask cache",
        "resolutions": list(RESOLUTIONS),
        "resize": "native128 binary mask -> PIL bilinear upscale -> >=0.5; 128 is byte-identical control",
        "limitation": "upscaling a binary prediction adds no new U-Net information or native high-resolution logits",
        "localization": "RETR_EXTERNAL repository contours; area filter OFF; top-10 cap OFF; padding=0",
        "geometry": "bbox remapped to original RGB with floor lower/ceil upper coordinates",
        "ground_truth_use": "post-hoc evaluation only; never used for candidate generation",
    }


def _inputs(args):
    config = Path(__file__).resolve().parents[1] / "configs/approximate/totaltext_v4_predicted_binary_resolution_no_filters_v1.json"
    if file_sha256(config) != EXPECTED_PROTOCOL_CONFIG:
        raise ValueError("Predicted no-filter diagnostic config SHA256 mismatch")
    if file_sha256(args.test_manifest) != EXPECTED_TEST_MANIFEST:
        raise ValueError("Expected official Total-Text TEST-300 manifest")
    if file_sha256(args.predicted_cache / "results.json") != EXPECTED_PREDICTED_RESULTS:
        raise ValueError("Frozen predicted mask report SHA256 mismatch")
    if file_sha256(args.frozen_v4_root / "results.json") != EXPECTED_FROZEN_V4_RESULTS:
        raise ValueError("Frozen v4 zero-pad OCR report SHA256 mismatch")
    predicted_protocol = json.loads((args.predicted_cache / "protocol.json").read_text(encoding="utf-8"))
    if (predicted_protocol.get("segmentation_checkpoint_sha256") != EXPECTED_SEGMENTATION_CHECKPOINT
        or predicted_protocol.get("test_manifest_sha256") != EXPECTED_TEST_MANIFEST
        or predicted_protocol.get("images") != 300):
        raise ValueError("Predicted mask cache checkpoint/split provenance mismatch")
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "test" or len(manifest.samples) != 300:
        raise ValueError("Expected official Total-Text TEST-300")
    return manifest


def _sample_inputs(args, sample):
    image_path = args.data_root / sample.image_path
    if file_sha256(image_path) != sample.image_sha256:
        raise ValueError(f"Source RGB checksum mismatch: {sample.sample_id}")
    with Image.open(image_path) as opened:
        image = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
    native, _ = _mask_from_cache(args.predicted_cache, sample.sample_id, sample.image_sha256)
    key = sample.sample_id.replace(":", "__")
    frozen = json.loads((args.frozen_v4_root / "samples" / f"{key}.json").read_text(encoding="utf-8"))
    if (frozen["sample_id"] != sample.sample_id or frozen["image_sha256"] != sample.image_sha256
        or frozen["branches"]["predicted_v4"]["native_mask_array_sha256"] != _sha_array(native)):
        raise ValueError(f"Frozen v4 predicted branch differs from mask cache: {sample.sample_id}")
    return image, native, frozen


def prepare(args) -> int:
    manifest = _inputs(args)
    _atomic_json(args.output_dir / "protocol.json", _protocol(args))
    protocol_sha = file_sha256(args.output_dir / "protocol.json")
    totals = {str(r): {"raw_contours": 0, "retained_proposals": 0,
                       "images_over_10_candidates": 0,
                       "one_to_one_iou": {str(t): 0 for t in (0.25, 0.5, 0.75)}}
              for r in RESOLUTIONS}
    gt_total = 0
    for index, sample in enumerate(manifest.samples, 1):
        image, native, frozen = _sample_inputs(args, sample)
        key = sample.sample_id.replace(":", "__")
        rows = {}
        for resolution in RESOLUTIONS:
            mask = resize_predicted_binary(native, resolution)
            audit, proposals = _proposals(mask, resolution, image, filter_mode="none")
            if resolution == 128:
                old = frozen["branches"]["predicted_v4"]["proposals"]
                available = {(tuple(item["bbox"]), item["crop_rgb_sha256"]) for item in proposals}
                if any((tuple(item["bbox"]), item["tight_rgb_sha256"]) not in available for item in old):
                    raise ValueError(f"Frozen zero-pad 128 crops absent: {sample.sample_id}")
                old_ocr = json.loads((args.frozen_v4_root / "ocr/predicted_v4/tight_rectangular" / f"{key}.json").read_text(encoding="utf-8"))["predictions"]
                if (len(old_ocr) != len(old)
                    or any(prior["rank"] != item["rank"] or prior["bbox"] != item["bbox"]
                           or prior["crop_rgb_sha256"] != item["tight_rgb_sha256"]
                           for item, prior in zip(old, old_ocr))):
                    raise ValueError(f"Frozen zero-pad predicted OCR cache differs: {sample.sample_id}")
            geometry = _geometry(frozen["ground_truth"], proposals)
            rows[str(resolution)] = {"mask_sha256": _sha_array(mask),
                                     "foreground_pixels": int(mask.sum()),
                                     "raw_contours": audit["raw_external_contours"],
                                     "geometry_one_to_one": geometry,
                                     "proposals": proposals}
            total = totals[str(resolution)]
            total["raw_contours"] += audit["raw_external_contours"]
            total["retained_proposals"] += len(proposals)
            total["images_over_10_candidates"] += len(proposals) > 10
            for threshold in total["one_to_one_iou"]:
                total["one_to_one_iou"][threshold] += geometry[threshold]
        _atomic_json(args.output_dir / "samples" / f"{key}.json",
                     {"schema_version": 1, "protocol_sha256": protocol_sha,
                      "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                      "ground_truth": frozen["ground_truth"], "resolutions": rows})
        gt_total += len(frozen["ground_truth"])
        if index % 10 == 0 or index == 300:
            print(f"[predicted resolution prepare] {index}/300", flush=True)
    if gt_total != 2204:
        raise ValueError("Total-Text care-word denominator changed")
    report = {"schema_version": 1, "protocol": _protocol(args), "images": 300,
              "gt_words": gt_total, "summary": totals}
    _atomic_json(args.output_dir / "prepared.json", report)
    print(json.dumps(totals, indent=2), flush=True)
    print("PREPARED", args.output_dir / "prepared.json", file_sha256(args.output_dir / "prepared.json"), flush=True)
    return 0


def recognize(args) -> int:
    import torch

    manifest = _inputs(args)
    prepared_path = args.output_dir / "prepared.json"
    prepared_sha = file_sha256(prepared_path)
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    if prepared["protocol"] != _protocol(args) or prepared["images"] != 300 or prepared["gt_words"] != 2204:
        raise ValueError("Prepared predicted resolution protocol differs")
    if file_sha256(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256:
        raise ValueError("PARSeq checkpoint SHA256 mismatch")
    if file_sha256(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError("PARSeq source archive SHA256 mismatch")
    final = args.output_dir / "results.json"
    if final.exists():
        report = json.loads(final.read_text(encoding="utf-8"))
        if report.get("prepared_sha256") != prepared_sha:
            raise ValueError("Completed report provenance differs")
        print(json.dumps(report["summary"], indent=2), flush=True)
        print("[reuse] JSON", final, "SHA256", file_sha256(final), flush=True)
        return 0
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("PARSeq recognition requires manual PAPER CUDA:0 execution")
    recognizer = None
    rows = {str(r): [] for r in RESOLUTIONS}
    hardware = {"device": args.device, "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "parseq_batch_size_max": 10}
    try:
        for index, sample in enumerate(manifest.samples, 1):
            key = sample.sample_id.replace(":", "__")
            shard = json.loads((args.output_dir / "samples" / f"{key}.json").read_text(encoding="utf-8"))
            if (shard["sample_id"] != sample.sample_id
                or shard["protocol_sha256"] != file_sha256(args.output_dir / "protocol.json")):
                raise ValueError("Predicted resolution sample provenance differs")
            image, native, frozen = _sample_inputs(args, sample)
            old = json.loads((args.frozen_v4_root / "ocr/predicted_v4/tight_rectangular" / f"{key}.json").read_text(encoding="utf-8"))["predictions"]
            old_proposals = frozen["branches"]["predicted_v4"]["proposals"]
            if (len(old) != len(old_proposals)
                or any(prior["rank"] != item["rank"] or prior["bbox"] != item["bbox"]
                       or prior["crop_rgb_sha256"] != item["tight_rgb_sha256"]
                       for item, prior in zip(old_proposals, old))):
                raise ValueError(f"Frozen predicted PARSeq OCR cache differs: {sample.sample_id}")
            known = {(tuple(item["bbox"]), item["crop_rgb_sha256"]):
                     (item["raw_text"], "frozen_v4_128") for item in old}
            for resolution in RESOLUTIONS:
                mask = resize_predicted_binary(native, resolution)
                audit, proposals = _proposals(mask, resolution, image, filter_mode="none")
                reference = shard["resolutions"][str(resolution)]
                if (_sha_array(mask) != reference["mask_sha256"]
                    or audit["raw_external_contours"] != reference["raw_contours"]
                    or proposals != reference["proposals"]):
                    raise ValueError(f"Predicted mask/proposals changed: {sample.sample_id}/{resolution}")
                ocr_path = args.output_dir / "ocr" / str(resolution) / f"{key}.json"
                if ocr_path.exists():
                    cached = json.loads(ocr_path.read_text(encoding="utf-8"))
                    if cached["sample_id"] != sample.sample_id or cached["prepared_sha256"] != prepared_sha:
                        raise ValueError("OCR shard provenance differs")
                    predictions = cached["predictions"]
                    if (len(predictions) != len(proposals)
                        or any(item["rank"] != prior["rank"] or item["bbox"] != prior["bbox"]
                               or item["crop_rgb_sha256"] != prior["crop_rgb_sha256"]
                               for item, prior in zip(proposals, predictions))):
                        raise ValueError("Predicted OCR crop cache mismatch")
                    for prediction in predictions:
                        cache_key = (tuple(prediction["bbox"]), prediction["crop_rgb_sha256"])
                        previous = known.get(cache_key)
                        if previous is not None and previous[0] != prediction["text"]:
                            raise ValueError("Same RGB crop has conflicting PARSeq text")
                        known[cache_key] = (prediction["text"], prediction["source"])
                else:
                    missing = {}
                    for item in proposals:
                        cache_key = (tuple(item["bbox"]), item["crop_rgb_sha256"])
                        if cache_key not in known:
                            missing[cache_key] = item
                    if missing and recognizer is None:
                        recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                            runtime_source=args.parseq_source.resolve(), checkpoint=args.parseq_checkpoint.resolve(),
                            device=args.device, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                            runtime_source_archive=args.parseq_source_archive.resolve()))
                        hardware["parseq_model_dtype"] = str(next(recognizer.model.parameters()).dtype)
                    pending = list(missing.items())
                    for start in range(0, len(pending), 10):
                        batch = pending[start:start + 10]
                        crops = [np.ascontiguousarray(image[y1:y2, x1:x2]) for _, item in batch
                                 for x1, y1, x2, y2 in [item["bbox"]]]
                        outputs = recognizer.recognize_candidate_batch(crops)
                        if len(outputs) != len(batch):
                            raise ValueError("PARSeq output count mismatch")
                        for (cache_key, _), output in zip(batch, outputs):
                            known[cache_key] = (output.raw_text, "new_parseq")
                    predictions = []
                    for item in proposals:
                        cache_key = (tuple(item["bbox"]), item["crop_rgb_sha256"])
                        text, origin = known[cache_key]
                        source = (origin if origin == "frozen_v4_128" or cache_key in missing
                                  else "reused_previous_resolution")
                        predictions.append({"rank": item["rank"], "bbox": item["bbox"],
                                            "crop_rgb_sha256": item["crop_rgb_sha256"],
                                            "text": text, "source": source})
                    _atomic_json(ocr_path, {"schema_version": 1, "sample_id": sample.sample_id,
                                            "resolution": resolution, "prepared_sha256": prepared_sha,
                                            "predictions": predictions})
                rows[str(resolution)].append({"sample_id": sample.sample_id,
                                              "ground_truth": frozen["ground_truth"],
                                              "predictions": predictions})
            if index % 10 == 0 or index == 300:
                print(f"[predicted resolution PARSeq] {index}/300", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
    hardware["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
    hardware["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
    summary = {resolution: evaluate_images(values, text_available=True)
               for resolution, values in rows.items()}
    sources = {str(r): {"frozen_v4_128": 0, "reused_previous_resolution": 0,
                        "new_parseq": 0} for r in RESOLUTIONS}
    for resolution in RESOLUTIONS:
        for sample in manifest.samples:
            key = sample.sample_id.replace(":", "__")
            ocr_path = args.output_dir / "ocr" / str(resolution) / f"{key}.json"
            for row in json.loads(ocr_path.read_text(encoding="utf-8"))["predictions"]:
                sources[str(resolution)][row["source"]] += 1
    report = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "reproduction_mode": "approximate",
              "result_class": "Total-Text TEST predicted-binary-upscale diagnostic; not paper Table 5",
              "prepared_sha256": prepared_sha,
              "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
              "hardware": hardware, "ocr_source_counts": sources, "summary": summary}
    _atomic_json(final, report)
    compact = {r: {"proposals": value["geometry"]["retained_proposals"],
                   "iou_0_5": value["geometry"]["iou_ge_0_50_one_to_one"],
                   "exact_case_sensitive": value["recognition"]["exact_case_sensitive_spatial_one_to_one"],
                   "exact_casefold": value["recognition"]["exact_casefold_spatial_one_to_one"]}
               for r, value in summary.items()}
    print(json.dumps(compact, indent=2), flush=True)
    print("JSON", final, "SHA256", file_sha256(final), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "recognize"))
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predicted-cache", type=Path, required=True)
    parser.add_argument("--frozen-v4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path)
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.action == "recognize" and not all((args.parseq_source, args.parseq_source_archive, args.parseq_checkpoint)):
        parser.error("recognize requires PARSeq source, archive, and checkpoint")
    return prepare(args) if args.action == "prepare" else recognize(args)


if __name__ == "__main__":
    raise SystemExit(main())
