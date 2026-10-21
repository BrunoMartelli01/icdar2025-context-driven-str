"""GT-v4 target-resolution sensitivity, with frozen native-128 control.

PRACTICAL_APPROXIMATION / TEST diagnostic only. The same full-resolution v4
oracle target is independently resized to 128, 256, 512, and 1024. This does
not emulate a U-Net trained or inferred at a higher native output resolution.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image

from localize.repository_contours import RepositoryContourParams, audit_repository_contours
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from recognition.scene_text_datasets import parse_totaltext_annotation
from recognition.scene_text_v1_evaluation import box_iou, evaluate_images, maximum_matching
from scripts.evaluate_totaltext_v4_curved_crops import _atomic_json
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
from segmentation.totaltext_curved_box_v4 import make_curved_word_boxes
from segmentation.totaltext_word_regions import make_word_regions


RESOLUTIONS = (128, 256, 512, 1024)
EXPECTED_TEST_MANIFEST = "da2f87e7cae765f426af39c9641535138d6657d2e9332e6fdf9e80f2ba6b69bd"
EXPECTED_V4_RESULTS = "272613b094777b9836a1ffde2fd9265fcec17c84e7ccb60661f9a87ac0e29989"
EXPECTED_NO_FILTER_POLICY = "f10aaf6dd32b6bdd1c6367d4473630bbd9c78b32b01327048cb00414e5e34d54"


def resize_v4_target(
    target: Image.Image, validity: Image.Image, separator: Image.Image,
    owners: Image.Image, resolution: int,
) -> np.ndarray:
    """Mirror qualified target resize/contact at a declared output resolution."""
    if resolution not in RESOLUTIONS or not (target.size == validity.size == separator.size == owners.size):
        raise ValueError("Unexpected resolution or v4 target geometry")
    size = (resolution, resolution)
    binary = np.asarray(target.resize(size, Image.Resampling.BILINEAR)) >= 128
    valid = np.asarray(validity.resize(size, Image.Resampling.NEAREST)) > 0
    sep = np.asarray(separator.resize(size, Image.Resampling.NEAREST)) > 0
    owner = np.asarray(owners.convert("I").resize(size, Image.Resampling.NEAREST), dtype=np.int32)
    contact = np.zeros(size, dtype=bool)
    for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
        y0a, y1a = max(0, -dy), resolution - max(0, dy)
        x0a, x1a = max(0, -dx), resolution - max(0, dx)
        y0b, y1b = max(0, dy), resolution - max(0, -dy)
        x0b, x1b = max(0, dx), resolution - max(0, -dx)
        left = owner[y0a:y1a, x0a:x1a]
        right = owner[y0b:y1b, x0b:x1b]
        different = (left > 0) & (right > 0) & (left != right)
        contact[y0a:y1a, x0a:x1a] |= different
        contact[y0b:y1b, x0b:x1b] |= different
    return np.ascontiguousarray((binary & valid & ~sep & ~contact & (owner > 0)).astype(np.uint8))


def remap_box(box: list[int], resolution: int, image_hw: tuple[int, int]) -> list[int]:
    height, width = image_hw
    x1, y1, x2, y2 = box
    if resolution not in RESOLUTIONS or min(height, width) < 1 or not (0 <= x1 < x2 <= resolution and 0 <= y1 < y2 <= resolution):
        raise ValueError("Invalid resolution, image geometry, or native box")
    result = [x1 * width // resolution, y1 * height // resolution,
              (x2 * width + resolution - 1) // resolution,
              (y2 * height + resolution - 1) // resolution]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError("Remapped crop is empty")
    return result


def scaled_area(resolution: int) -> int:
    if resolution not in RESOLUTIONS:
        raise ValueError("Unscheduled resolution")
    return 150 * (resolution // 128) ** 2


def _sha_array(array: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _inputs(args):
    if args.filter_mode == "none":
        policy = Path(__file__).resolve().parents[1] / "configs/approximate/scene_text_gt_oracle_no_filters_v1.json"
        if file_sha256(policy) != EXPECTED_NO_FILTER_POLICY:
            raise ValueError("Frozen no-filter GT-oracle policy SHA256 mismatch")
    if file_sha256(args.test_manifest) != EXPECTED_TEST_MANIFEST:
        raise ValueError("Expected official Total-Text TEST-300 manifest")
    if file_sha256(args.frozen_v4_root / "results.json") != EXPECTED_V4_RESULTS:
        raise ValueError("Frozen GT-v4 OCR report SHA256 mismatch")
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "test" or len(manifest.samples) != 300:
        raise ValueError("Expected official Total-Text TEST-300")
    return manifest


def _regenerate(args, sample):
    image_path = args.data_root / sample.image_path
    pixel_path = args.data_root / sample.mask_path
    validity_path = args.data_root / sample.validity_annotation_path
    for path, digest in ((image_path, sample.image_sha256), (pixel_path, sample.mask_sha256),
                         (validity_path, sample.validity_annotation_sha256)):
        if file_sha256(path) != digest:
            raise ValueError(f"Official TEST source changed: {path}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    with Image.open(pixel_path) as opened:
        pixel = opened.convert("L")
    with Image.open(validity_path) as opened:
        qualified_validity = np.asarray(opened.convert("L"))
    number = sample.sample_id.split(":", 1)[1]
    annotation = args.data_root / "datasets/total_text/test/annotations" / f"{number}.txt"
    instances, _ = clip_instances_to_pixel_extent(parse_totaltext_annotation(annotation), image.size)
    _, validity, separator, owners, _ = make_word_regions(image.size, instances)
    if not np.array_equal(np.asarray(validity), qualified_validity):
        raise ValueError(f"Qualified Don't Care validity changed: {sample.sample_id}")
    target, validity, separator, owners, _ = make_curved_word_boxes(
        image.size, instances, owners, pixel, separator, validity
    )
    frozen = json.loads((args.frozen_v4_root / "samples" / f"{sample.sample_id.replace(':', '__')}.json").read_text(encoding="utf-8"))
    if frozen["sample_id"] != sample.sample_id or frozen["annotation_sha256"] != file_sha256(annotation):
        raise ValueError("Frozen GT-v4 sample/annotation mismatch")
    return np.ascontiguousarray(np.asarray(image)), target, validity, separator, owners, frozen


def _proposals(mask: np.ndarray, resolution: int, image: np.ndarray,
               *, filter_mode: str = "frozen") -> tuple[dict, list[dict]]:
    if filter_mode not in ("frozen", "none"):
        raise ValueError("Unknown filter mode")
    params = (RepositoryContourParams(scaled_area(resolution), 0, 10)
              if filter_mode == "frozen"
              else RepositoryContourParams(0, 0, int(mask.size)))
    audit = audit_repository_contours(mask, params)
    rows = []
    for item in audit["retained"]:
        box = remap_box(item["bbox"], resolution, image.shape[:2])
        x1, y1, x2, y2 = box
        rows.append({"rank": item["rank"], "contour_index": item["contour_index"],
                     "bbox_mask": item["bbox"], "bbox": box,
                     "bbox_area_mask": item["bbox_area"],
                     "crop_rgb_sha256": _sha_array(image[y1:y2, x1:x2])})
    return audit, rows


def _geometry(gt: list[dict], proposals: list[dict]) -> dict[str, int]:
    return {str(threshold): len(maximum_matching(
        len(proposals), len(gt),
        [(i, j) for i, item in enumerate(proposals) for j, word in enumerate(gt)
         if box_iou(item["bbox"], word["bbox"]) >= threshold]
    )) for threshold in (0.25, 0.5, 0.75)}


def _protocol(filter_mode: str = "frozen") -> dict:
    if filter_mode not in ("frozen", "none"):
        raise ValueError("Unknown filter mode")
    protocol = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
            "result_class": "Total-Text TEST GT-v4 target-resolution diagnostic; not paper Table 5",
            "test_manifest_sha256": EXPECTED_TEST_MANIFEST,
            "frozen_v4_results_sha256": EXPECTED_V4_RESULTS,
            "source": "official-annotation-derived full-resolution GT-v4 target",
            "resolutions": list(RESOLUTIONS),
            "resize": "full-resolution target bilinear then >=0.5; owner/validity/separator nearest and owner-contact suppression",
            "localization": "frozen repository-style RETR_EXTERNAL bbox-area filter, area scaled from native128 150; pad=0; top10",
            "area_by_resolution": {str(r): scaled_area(r) for r in RESOLUTIONS},
            "crop": "tight RGB box remapped by floor lower/ceil upper; no curved cutout",
            "limitation": "higher-resolution GT targets; no claim about higher-resolution trained U-Net predictions"}
    if filter_mode == "none":
        protocol["localization"] = ("USER-REQUESTED DIAGNOSTIC: repository-style RETR_EXTERNAL; "
                                    "no bbox-area filter, no top-10 cap, pad=0")
        protocol["area_by_resolution"] = {str(r): 0 for r in RESOLUTIONS}
        protocol["filter_mode"] = "none"
        protocol["scientific_choice"] = "ASSUMPTION; GT-oracle practical diagnostic, not paper Algorithm 1"
        protocol["cap"] = "none; nmax set to mask pixel count, above every possible contour count"
        protocol["policy"] = "configs/approximate/scene_text_gt_oracle_no_filters_v1.json"
        protocol["policy_sha256"] = EXPECTED_NO_FILTER_POLICY
    return protocol


def prepare(args) -> int:
    manifest = _inputs(args)
    _atomic_json(args.output_dir / "protocol.json", _protocol(args.filter_mode))
    protocol_sha = file_sha256(args.output_dir / "protocol.json")
    totals = {str(r): {"raw_contours": 0, "retained_proposals": 0, "images_over_top10_before_cap": 0,
                       "one_to_one_iou": {str(t): 0 for t in (0.25, 0.5, 0.75)}} for r in RESOLUTIONS}
    gt_total = 0
    for index, sample in enumerate(manifest.samples, 1):
        image, target, validity, separator, owners, frozen = _regenerate(args, sample)
        key = sample.sample_id.replace(":", "__")
        rows = {}
        for resolution in RESOLUTIONS:
            mask = resize_v4_target(target, validity, separator, owners, resolution)
            if resolution == 128:
                native_file = args.frozen_v4_root / frozen["gt_native_mask"]["file"]
                if file_sha256(native_file) != frozen["gt_native_mask"]["sha256"]:
                    raise ValueError("Frozen GT-v4 native mask checksum mismatch")
                with Image.open(native_file) as opened:
                    native = np.asarray(opened.convert("L")) > 0
                if not np.array_equal(mask, native):
                    raise ValueError(f"GT-v4 target resolution128 differs: {sample.sample_id}")
            audit, proposals = _proposals(mask, resolution, image, filter_mode=args.filter_mode)
            if resolution == 128:
                old = frozen["branches"]["gt_v4_oracle"]["proposals"]
                if args.filter_mode == "frozen":
                    if (len(proposals) != len(old)
                        or any(item["bbox"] != ref["bbox"] or item["bbox_mask"] != ref["bbox_native"]
                               or item["crop_rgb_sha256"] != ref["tight_rgb_sha256"]
                               for item, ref in zip(proposals, old))):
                        raise ValueError(f"GT-v4 resolution128 proposals differ: {sample.sample_id}")
                else:
                    available = {(tuple(item["bbox"]), item["crop_rgb_sha256"]) for item in proposals}
                    if any((tuple(ref["bbox"]), ref["tight_rgb_sha256"]) not in available for ref in old):
                        raise ValueError(f"Filtered 128 baseline not contained in no-filter proposals: {sample.sample_id}")
            geometry = _geometry(frozen["ground_truth"], proposals)
            rows[str(resolution)] = {"mask_sha256": _sha_array(mask), "foreground_pixels": int(mask.sum()),
                                     "raw_contours": audit["raw_external_contours"],
                                     "candidates_before_cap": audit["candidates_before_cap"],
                                     "geometry_one_to_one": geometry, "proposals": proposals}
            summary = totals[str(resolution)]
            summary["raw_contours"] += audit["raw_external_contours"]
            summary["retained_proposals"] += len(proposals)
            summary["images_over_top10_before_cap"] += audit["candidates_before_cap"] > 10
            for threshold in summary["one_to_one_iou"]:
                summary["one_to_one_iou"][threshold] += geometry[threshold]
        _atomic_json(args.output_dir / "samples" / f"{key}.json",
                     {"schema_version": 1, "protocol_sha256": protocol_sha,
                      "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                      "ground_truth": frozen["ground_truth"], "resolutions": rows})
        gt_total += len(frozen["ground_truth"])
        if index % 10 == 0 or index == 300:
            print(f"[GT-v4 resolution prepare] {index}/300", flush=True)
    expected_128 = 1030 if args.filter_mode == "frozen" else 2204
    if gt_total != 2204 or totals["128"]["retained_proposals"] != expected_128:
        raise ValueError("Frozen TEST-300 GT/proposal denominator mismatch")
    report = {"schema_version": 1, "protocol": _protocol(args.filter_mode), "images": 300,
              "gt_words": gt_total, "summary": totals}
    _atomic_json(args.output_dir / "prepared.json", report)
    print(json.dumps(totals, indent=2), flush=True)
    print("PREPARED", args.output_dir / "prepared.json", file_sha256(args.output_dir / "prepared.json"), flush=True)
    return 0


def recognize(args) -> int:
    import torch
    manifest = _inputs(args)
    prepared = json.loads((args.output_dir / "prepared.json").read_text(encoding="utf-8"))
    if prepared["protocol"] != _protocol(args.filter_mode) or prepared["images"] != 300 or prepared["gt_words"] != 2204:
        raise ValueError("Prepared GT-v4 resolution protocol differs")
    if file_sha256(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256:
        raise ValueError("PARSeq checkpoint SHA256 mismatch")
    if file_sha256(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError("PARSeq source archive SHA256 mismatch")
    final = args.output_dir / "results.json"
    if final.exists():
        report = json.loads(final.read_text(encoding="utf-8"))
        if report.get("prepared_sha256") != file_sha256(args.output_dir / "prepared.json"):
            raise ValueError("Completed report provenance differs")
        print(json.dumps(report["summary"], indent=2), flush=True)
        print("[reuse] JSON", final, "SHA256", file_sha256(final), flush=True)
        return 0
    if args.device != "cuda:0" or not torch.cuda.is_available():
        raise RuntimeError("PARSeq must be run manually on PAPER CUDA:0")
    recognizer = None
    rows = {str(r): [] for r in RESOLUTIONS}
    hardware = {"device": args.device, "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                "parseq_batch_size_max": 10}
    try:
        for index, sample in enumerate(manifest.samples, 1):
            key = sample.sample_id.replace(":", "__")
            shard = json.loads((args.output_dir / "samples" / f"{key}.json").read_text(encoding="utf-8"))
            if shard["sample_id"] != sample.sample_id or shard["protocol_sha256"] != file_sha256(args.output_dir / "protocol.json"):
                raise ValueError("Resolution sample provenance differs")
            image, target, validity, separator, owners, frozen = _regenerate(args, sample)
            old_path = args.frozen_v4_root / "ocr/gt_v4_oracle/tight_rectangular" / f"{key}.json"
            old = json.loads(old_path.read_text(encoding="utf-8"))["predictions"]
            old_proposals = frozen["branches"]["gt_v4_oracle"]["proposals"]
            if (len(old) != len(old_proposals)
                or any(prior["rank"] != item["rank"] or prior["bbox"] != item["bbox"]
                       or prior["crop_rgb_sha256"] != item["tight_rgb_sha256"]
                       for item, prior in zip(old_proposals, old))):
                raise ValueError("Frozen 128 PARSeq OCR cache differs from verified crop metadata")
            known = {(tuple(item["bbox"]), item["crop_rgb_sha256"]):
                     (item["raw_text"], "frozen_v4_128") for item in old}
            for resolution in RESOLUTIONS:
                mask = resize_v4_target(target, validity, separator, owners, resolution)
                audit, proposals = _proposals(mask, resolution, image, filter_mode=args.filter_mode)
                reference = shard["resolutions"][str(resolution)]
                if (_sha_array(mask) != reference["mask_sha256"]
                    or audit["raw_external_contours"] != reference["raw_contours"]
                    or proposals != reference["proposals"]):
                    raise ValueError(f"GT-v4 mask/proposals changed: {sample.sample_id}/{resolution}")
                ocr_path = args.output_dir / "ocr" / str(resolution) / f"{key}.json"
                if ocr_path.exists():
                    cached = json.loads(ocr_path.read_text(encoding="utf-8"))
                    if cached["sample_id"] != sample.sample_id or cached["prepared_sha256"] != file_sha256(args.output_dir / "prepared.json"):
                        raise ValueError("OCR shard provenance differs")
                    predictions = cached["predictions"]
                    if (len(predictions) != len(proposals)
                        or any(item["rank"] != prior["rank"] or item["bbox"] != prior["bbox"]
                               or item["crop_rgb_sha256"] != prior["crop_rgb_sha256"]
                               for item, prior in zip(proposals, predictions))):
                        raise ValueError("OCR crop cache mismatch")
                    for prediction in predictions:
                        cache_key = (tuple(prediction["bbox"]), prediction["crop_rgb_sha256"])
                        previous = known.get(cache_key)
                        if previous is not None and previous[0] != prediction["text"]:
                            raise ValueError("Same RGB crop has conflicting PARSeq transcriptions")
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
                                            "resolution": resolution,
                                            "prepared_sha256": file_sha256(args.output_dir / "prepared.json"),
                                            "predictions": predictions})
                rows[str(resolution)].append({"sample_id": sample.sample_id,
                                              "ground_truth": frozen["ground_truth"],
                                              "predictions": predictions})
            if index % 10 == 0 or index == 300:
                print(f"[GT-v4 resolution PARSeq] {index}/300", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
    hardware["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
    hardware["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
    summary = {resolution: evaluate_images(values, text_available=True)
               for resolution, values in rows.items()}
    baseline = json.loads((args.frozen_v4_root / "results.json").read_text(encoding="utf-8"))["summary"]["gt_v4_oracle"]["tight_rectangular"]
    if args.filter_mode == "frozen" and summary["128"] != baseline:
        raise ValueError("GT-v4 resolution128 PARSeq evaluation differs from qualified baseline")
    reuse = {str(resolution): {"frozen_v4_128": 0, "reused_previous_resolution": 0,
                               "new_parseq": 0} for resolution in RESOLUTIONS}
    for resolution in RESOLUTIONS:
        for sample in manifest.samples:
            path = args.output_dir / "ocr" / str(resolution) / f"{sample.sample_id.replace(':', '__')}.json"
            for prediction in json.loads(path.read_text(encoding="utf-8"))["predictions"]:
                reuse[str(resolution)][prediction["source"]] += 1
    report = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "result_class": "TEST GT-v4 resolution sensitivity; not paper Table 5",
              "prepared_sha256": file_sha256(args.output_dir / "prepared.json"),
              "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
              "hardware": hardware, "ocr_source_counts": reuse, "summary": summary}
    _atomic_json(final, report)
    compact = {resolution: {"proposals": value["geometry"]["retained_proposals"],
                            "iou_0_5": value["geometry"]["iou_ge_0_50_one_to_one"],
                            "exact_case_sensitive": value["recognition"]["exact_case_sensitive_spatial_one_to_one"],
                            "exact_casefold": value["recognition"]["exact_casefold_spatial_one_to_one"]}
               for resolution, value in summary.items()}
    print(json.dumps(compact, indent=2), flush=True)
    print("JSON", final, "SHA256", file_sha256(final), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "recognize"))
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--frozen-v4-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parseq-source", type=Path)
    parser.add_argument("--parseq-source-archive", type=Path)
    parser.add_argument("--parseq-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--filter-mode", choices=("frozen", "none"), default="frozen")
    args = parser.parse_args()
    if args.action == "recognize" and not all((args.parseq_source, args.parseq_source_archive, args.parseq_checkpoint)):
        parser.error("recognize requires PARSeq source, archive, and checkpoint")
    return prepare(args) if args.action == "prepare" else recognize(args)


if __name__ == "__main__":
    raise SystemExit(main())
