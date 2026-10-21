"""Compare tight rectangular and curved-component crops on Total-Text TEST.

The GT-v4 branch is an oracle: TEST annotations construct its mask, never a
deployable prediction. The predicted branch reuses the frozen v4 TEST masks.
Neither branch changes the frozen paper/localization/recognition pipelines.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from localize.curved_component_crop import native_component_curved_crops
from localize.repository_contours import RepositoryContourParams
from recognition.parseq import (
    PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256,
    ParseqApproxConfig, ParseqApproxRecognizer,
)
from recognition.scene_text_datasets import parse_totaltext_annotation
from recognition.scene_text_v1_evaluation import box_iou, evaluate_images, maximum_matching
from segmentation.config import load_segmentation_config
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
from segmentation.totaltext_curved_box_v4 import make_curved_word_boxes
from segmentation.totaltext_word_regions import make_word_regions
from segmentation.train import _transform


EXPECTED_TEST_MANIFEST_SHA256 = "da2f87e7cae765f426af39c9641535138d6657d2e9332e6fdf9e80f2ba6b69bd"
EXPECTED_PREDICTED_RESULTS_SHA256 = "f762aa83f276e0282babbed48dc5ad407e3edfe7da4c74fb9bf6849ee640c740"
EXPECTED_CHECKPOINT_SHA256 = "e31a493d2752734d9d75f35a8cec077ff779655145be7735c0e3267ceea64985"
BRANCHES = ("gt_v4_oracle", "predicted_v4")
MODES = ("tight_rectangular", "curved_cutout")
THRESHOLDS = (0.25, 0.5, 0.75)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("prepare", "recognize"):
        command = sub.add_parser(action)
        command.add_argument("--test-manifest", type=Path, required=True)
        command.add_argument("--data-root", type=Path, required=True)
        command.add_argument("--segmentation-config", type=Path, required=True)
        command.add_argument("--localization-config", type=Path, required=True)
        command.add_argument("--predicted-cache", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        if action == "recognize":
            command.add_argument("--parseq-source", type=Path, required=True)
            command.add_argument("--parseq-source-archive", type=Path, required=True)
            command.add_argument("--parseq-checkpoint", type=Path, required=True)
            command.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Existing report differs: {path}")
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _key(sample_id: str) -> str:
    return sample_id.replace(":", "__")


def _array_sha(value: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _config(args: argparse.Namespace):
    if file_sha256(args.test_manifest) != EXPECTED_TEST_MANIFEST_SHA256:
        raise ValueError("Official Total-Text TEST manifest SHA256 mismatch")
    if file_sha256(args.predicted_cache / "results.json") != EXPECTED_PREDICTED_RESULTS_SHA256:
        raise ValueError("Frozen v4 predicted TEST results SHA256 mismatch")
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "test" or len(manifest.samples) != 300:
        raise ValueError("Expected official Total-Text TEST-300")
    config = load_segmentation_config(args.segmentation_config)
    if (config.run.variant_class != "PRACTICAL_APPROXIMATION"
        or int(config.training.epochs) != 500
        or tuple(config.transforms.mask_size) != (128, 128)
        or config.transforms.mask_interpolation != "bilinear"
        or float(config.transforms.target_binarization_threshold) != 0.5):
        raise ValueError("Expected frozen v4 TRAIN target transform")
    localizer = json.loads(args.localization_config.read_text(encoding="utf-8"))
    if (localizer.get("variant") != "curved-component-cutout-native128-no-padding-v1"
        or localizer.get("reproduction_mode") != "approximate"
        or localizer.get("classification") != "PRACTICAL_APPROXIMATION"):
        raise ValueError("Unexpected curved-cutout protocol")
    values = localizer["parameters"]
    params = RepositoryContourParams(int(values["min_area"]), int(values["pad"]), int(values["nmax"]))
    if (params.min_area, params.pad, params.nmax) != (150, 0, 10):
        raise ValueError("Expected frozen area150/top10 with user-requested zero padding")
    predicted_protocol = json.loads((args.predicted_cache / "protocol.json").read_text(encoding="utf-8"))
    if (predicted_protocol["segmentation_checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256
        or predicted_protocol["test_manifest_sha256"] != EXPECTED_TEST_MANIFEST_SHA256
        or predicted_protocol["images"] != 300):
        raise ValueError("Cached predicted masks do not belong to v4 TEST-300")
    protocol = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "Total-Text TEST crop-shape diagnostic; not paper Table 5",
        "gt_branch": "TEST-annotation-derived v4 oracle, not a deployable prediction",
        "gt_annotation_clipping": "qualified Total-Text TEST pixel-extent clipping before v4 target construction",
        "predicted_branch": "frozen v4 epoch-0458 TEST mask cache",
        "test_manifest_sha256": EXPECTED_TEST_MANIFEST_SHA256,
        "predicted_results_sha256": EXPECTED_PREDICTED_RESULTS_SHA256,
        "predicted_protocol_sha256": file_sha256(args.predicted_cache / "protocol.json"),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "segmentation_config_sha256": file_sha256(args.segmentation_config),
        "localization_config_sha256": file_sha256(args.localization_config),
        "localization_parameters": asdict(params),
        "component_support": "one native 8-connected label; bilinear resize to RGB dimensions, >=0.5",
        "crop": "tight original RGB bbox; curved version replaces off-component pixels with white; no spatial padding",
        "comparison": "same masks/proposals for tight rectangular and curved cutout within each branch",
        "matching": "one-to-one spatial IoU >=0.5; both case-sensitive and casefold exact; full 2204-word denominator",
    }
    return manifest, config, params, protocol


def _ground_truth(instances, image_wh: tuple[int, int]) -> list[dict]:
    width, height = image_wh
    words = []
    for item in instances:
        if item.ignore:
            continue
        x1, y1, x2, y2 = item.bbox_xyxy
        words.append({
            "bbox": [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)],
            "text": item.transcription,
            "granularity": "word",
        })
    return words


def _geometry(gt: list[dict], proposals) -> dict:
    boxes = [item.bbox_original for item in proposals]
    scores = [[box_iou(box, word["bbox"]) for word in gt] for box in boxes]
    matched = {}
    for threshold in THRESHOLDS:
        edges = [(proposal, word) for proposal, row in enumerate(scores)
                 for word, score in enumerate(row) if score >= threshold]
        matched[str(threshold)] = len(maximum_matching(len(boxes), len(gt), edges))
    return matched


def _mask_from_cache(cache: Path, sample_id: str, image_sha: str) -> tuple[np.ndarray, dict]:
    shard_path = cache / "samples" / f"{_key(sample_id)}.json"
    shard = json.loads(shard_path.read_text(encoding="utf-8"))
    if (shard["sample_id"] != sample_id or shard["image_sha256"] != image_sha
        or shard["protocol_sha256"] != file_sha256(cache / "protocol.json")):
        raise ValueError(f"Predicted mask shard provenance mismatch: {sample_id}")
    mask_file = cache / shard["native_mask"]["file"]
    if file_sha256(mask_file) != shard["native_mask"]["sha256"]:
        raise ValueError(f"Predicted mask PNG SHA256 mismatch: {sample_id}")
    with Image.open(mask_file) as opened:
        native = np.asarray(opened.convert("L")).copy()
    if native.shape != (128, 128) or not np.isin(native, (0, 255)).all():
        raise ValueError(f"Predicted mask is not canonical binary native128: {sample_id}")
    native = np.ascontiguousarray((native > 0).astype(np.uint8))
    if _array_sha(native) != shard["native_mask_array_sha256"]:
        raise ValueError(f"Predicted native mask array SHA256 mismatch: {sample_id}")
    return native, shard


def _proposal_rows(proposals) -> list[dict]:
    return [{
        "rank": item.rank,
        "contour_index": item.contour_index,
        "bbox_native": list(item.bbox_native),
        "bbox": list(item.bbox_original),
        "bbox_area_native": item.bbox_area_native,
        "component_pixels_native": item.component_pixels_native,
        "foreground_fraction_in_crop": item.foreground_fraction_in_crop,
        "tight_rgb_sha256": _array_sha(item.rectangular_rgb),
        "curved_rgb_sha256": _array_sha(item.curved_rgb),
    } for item in proposals]


def prepare(args: argparse.Namespace) -> int:
    manifest, config, params, protocol = _config(args)
    output = args.output_dir.resolve()
    _atomic_json(output / "protocol.json", protocol)
    protocol_sha = file_sha256(output / "protocol.json")
    transform = _transform(config.transforms, seed=int(config.training.seed), augmentation_enabled=False)
    rows = []
    for index, sample in enumerate(manifest.samples):
        key = _key(sample.sample_id)
        shard_path = output / "samples" / f"{key}.json"
        if shard_path.exists():
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            mask_path = output / shard["gt_native_mask"]["file"]
            if shard["protocol_sha256"] != protocol_sha or file_sha256(mask_path) != shard["gt_native_mask"]["sha256"]:
                raise ValueError(f"Existing GT-v4 TEST shard is corrupt: {sample.sample_id}")
            _mask_from_cache(args.predicted_cache, sample.sample_id, sample.image_sha256)
            rows.append(shard)
            continue
        image_path = args.data_root / sample.image_path
        pixel_path = args.data_root / sample.mask_path
        validity_path = args.data_root / sample.validity_annotation_path
        annotation = args.data_root / "datasets/total_text/test/annotations" / (sample.sample_id.split(":", 1)[1] + ".txt")
        for path, expected in ((image_path, sample.image_sha256), (pixel_path, sample.mask_sha256),
                               (validity_path, sample.validity_annotation_sha256)):
            if file_sha256(path) != expected:
                raise ValueError(f"Official TEST file checksum mismatch: {path}")
        with Image.open(image_path) as opened:
            rgb_image = opened.convert("RGB")
            image = np.ascontiguousarray(np.asarray(rgb_image))
        with Image.open(pixel_path) as opened:
            pixel_gt = opened.convert("L")
        with Image.open(validity_path) as opened:
            source_validity = np.asarray(opened.convert("L")).copy()
        # The qualified TEST pixel-mask dataset clips polygon vertices to the
        # image extent before producing Don't Care validity. Mirror that step.
        instances, _ = clip_instances_to_pixel_extent(
            parse_totaltext_annotation(annotation), rgb_image.size
        )
        base, validity, separator, owners, _ = make_word_regions(rgb_image.size, instances)
        if not np.array_equal(np.asarray(validity), source_validity):
            raise ValueError(f"TEST Don't Care validity differs from target builder: {sample.sample_id}")
        target, validity, separator, owners, target_audit = make_curved_word_boxes(
            rgb_image.size, instances, owners, pixel_gt, separator, validity
        )
        # SegmentationDataset converts the stored uint16 owner PNG to PIL "I"
        # before the qualified transform. Reproduce that loader contract here.
        owners = Image.fromarray(np.asarray(owners, dtype=np.int32), mode="I")
        _, tensor, _, _ = transform.transform_with_validity_and_separation(
            rgb_image, target, validity, separator, owners, index
        )
        gt_native = np.ascontiguousarray(tensor[0].numpy().astype(np.uint8))
        predicted_native, _ = _mask_from_cache(args.predicted_cache, sample.sample_id, sample.image_sha256)
        mask_file = output / "gt_native_masks" / f"{key}.png"
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        if mask_file.exists():
            with Image.open(mask_file) as opened:
                saved = np.asarray(opened.convert("L"))
            if not np.array_equal(saved, gt_native * 255):
                raise ValueError(f"Existing oracle mask differs: {sample.sample_id}")
        else:
            Image.fromarray(gt_native * 255, mode="L").save(mask_file, optimize=True)
        gt = _ground_truth(instances, rgb_image.size)
        branch_data = {}
        for branch, native in (("gt_v4_oracle", gt_native), ("predicted_v4", predicted_native)):
            audit, proposals = native_component_curved_crops(native, image, params)
            branch_data[branch] = {
                "native_mask_array_sha256": _array_sha(native),
                "raw_contours": audit["raw_external_contours"],
                "candidates_before_cap": audit["candidates_before_cap"],
                "geometry_one_to_one": _geometry(gt, proposals),
                "proposals": _proposal_rows(proposals),
            }
        shard = {
            "schema_version": 1, "protocol_sha256": protocol_sha,
            "sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
            "annotation_sha256": file_sha256(annotation),
            "gt_native_mask": {"file": mask_file.relative_to(output).as_posix(), "sha256": file_sha256(mask_file)},
            "target_audit": asdict(target_audit), "ground_truth": gt,
            "branches": branch_data,
        }
        _atomic_json(shard_path, shard)
        rows.append(shard)
        if (index + 1) % 10 == 0 or index + 1 == len(manifest.samples):
            print(f"[curved crop prepare] {index + 1}/{len(manifest.samples)}", flush=True)
    if len(rows) != 300 or sum(len(row["ground_truth"]) for row in rows) != 2204:
        raise ValueError("TEST-300/2204 care-word audit failed")
    summary = {"images": 300, "care_words": 2204, "branches": {}}
    for branch in BRANCHES:
        summary["branches"][branch] = {
            "raw_contours": sum(row["branches"][branch]["raw_contours"] for row in rows),
            "retained_proposals": sum(len(row["branches"][branch]["proposals"]) for row in rows),
            "images_over_top10_before_cap": sum(row["branches"][branch]["candidates_before_cap"] > 10 for row in rows),
            "one_to_one_iou": {str(t): sum(row["branches"][branch]["geometry_one_to_one"][str(t)] for row in rows) for t in THRESHOLDS},
        }
    report = {"schema_version": 1, "protocol": protocol, "summary": summary,
              "samples": [f"samples/{_key(row['sample_id'])}.json" for row in rows]}
    _atomic_json(output / "prepared.json", report)
    print(json.dumps(summary, indent=2), flush=True)
    print("PREPARED", output / "prepared.json", file_sha256(output / "prepared.json"), flush=True)
    return 0


def recognize(args: argparse.Namespace) -> int:
    import torch

    manifest, _, params, protocol = _config(args)
    output = args.output_dir.resolve()
    prepared = json.loads((output / "prepared.json").read_text(encoding="utf-8"))
    if prepared["protocol"] != protocol or prepared["summary"]["images"] != 300:
        raise ValueError("Prepared crop protocol does not match this invocation")
    if file_sha256(args.parseq_checkpoint) != PARSEQ_CHECKPOINT_SHA256:
        raise ValueError("PARSeq checkpoint SHA256 mismatch")
    if file_sha256(args.parseq_source_archive) != PARSEQ_RUNTIME_ARCHIVE_SHA256:
        raise ValueError("PARSeq runtime archive SHA256 mismatch")
    finished_path = output / "results.json"
    if finished_path.exists():
        finished = json.loads(finished_path.read_text(encoding="utf-8"))
        if (finished.get("protocol") != protocol
            or finished.get("prepared_sha256") != file_sha256(output / "prepared.json")
            or finished.get("parseq_checkpoint_sha256") != PARSEQ_CHECKPOINT_SHA256):
            raise ValueError("Existing completed recognition report has different provenance")
        for branch in BRANCHES:
            for mode in MODES:
                if any(not (output / path).is_file() for path in finished["ocr_shards"][branch][mode]):
                    raise ValueError("Completed report refers to a missing OCR shard")
        print(json.dumps(finished["summary"], indent=2), flush=True)
        print("[reuse] JSON", finished_path, "SHA256", file_sha256(finished_path), flush=True)
        return 0
    placement = torch.device(args.device)
    if placement.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This qualified PARSeq diagnostic requires an available CUDA device")
    device_index = placement.index if placement.index is not None else torch.cuda.current_device()
    hardware = {
        "device": args.device,
        "gpu": torch.cuda.get_device_name(device_index),
        "total_vram_bytes": torch.cuda.get_device_properties(device_index).total_memory,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "peak_scope": "whole recognition process; no silent peak reset",
        "parseq_batch_size_max": 10,
    }
    protocol_sha = file_sha256(output / "protocol.json")
    recognizer = None
    rows = {branch: {mode: [] for mode in MODES} for branch in BRANCHES}
    try:
        for index, sample in enumerate(manifest.samples, 1):
            shard = json.loads((output / "samples" / f"{_key(sample.sample_id)}.json").read_text(encoding="utf-8"))
            if shard["sample_id"] != sample.sample_id or shard["protocol_sha256"] != protocol_sha:
                raise ValueError(f"Prepared shard provenance mismatch: {sample.sample_id}")
            image_path = args.data_root / sample.image_path
            if file_sha256(image_path) != sample.image_sha256:
                raise ValueError(f"RGB image checksum mismatch: {sample.sample_id}")
            with Image.open(image_path) as opened:
                image = np.ascontiguousarray(np.asarray(opened.convert("RGB")))
            gt_file = output / shard["gt_native_mask"]["file"]
            if file_sha256(gt_file) != shard["gt_native_mask"]["sha256"]:
                raise ValueError(f"Oracle mask checksum mismatch: {sample.sample_id}")
            with Image.open(gt_file) as opened:
                gt_native = np.ascontiguousarray((np.asarray(opened.convert("L")) > 0).astype(np.uint8))
            predicted_native, _ = _mask_from_cache(args.predicted_cache, sample.sample_id, sample.image_sha256)
            for branch, native in (("gt_v4_oracle", gt_native), ("predicted_v4", predicted_native)):
                audit, proposals = native_component_curved_crops(native, image, params)
                if (audit["raw_external_contours"] != shard["branches"][branch]["raw_contours"]
                    or _array_sha(native) != shard["branches"][branch]["native_mask_array_sha256"]
                    or _proposal_rows(proposals) != shard["branches"][branch]["proposals"]):
                    raise ValueError(f"Crop/proposal regeneration differs: {sample.sample_id}/{branch}")
                for mode in MODES:
                    ocr_path = output / "ocr" / branch / mode / f"{_key(sample.sample_id)}.json"
                    if ocr_path.exists():
                        ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
                        if (ocr["protocol_sha256"] != protocol_sha
                            or ocr["sample_id"] != sample.sample_id
                            or len(ocr["predictions"]) != len(proposals)):
                            raise ValueError(f"OCR cache mismatch: {ocr_path}")
                        for value, proposal in zip(ocr["predictions"], proposals):
                            crop = proposal.curved_rgb if mode == "curved_cutout" else proposal.rectangular_rgb
                            if (value["rank"] != proposal.rank or value["crop_rgb_sha256"] != _array_sha(crop)):
                                raise ValueError(f"OCR crop hash mismatch: {ocr_path}")
                    else:
                        if recognizer is None:
                            recognizer = ParseqApproxRecognizer(ParseqApproxConfig(
                                runtime_source=args.parseq_source.resolve(), checkpoint=args.parseq_checkpoint.resolve(),
                                device=args.device, runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                                runtime_source_archive=args.parseq_source_archive.resolve(),
                            ))
                            hardware["parseq_model_dtype"] = str(next(recognizer.model.parameters()).dtype)
                        crops = [proposal.curved_rgb if mode == "curved_cutout" else proposal.rectangular_rgb for proposal in proposals]
                        predictions = recognizer.recognize_candidate_batch(crops) if crops else ()
                        ocr = {"schema_version": 1, "protocol_sha256": protocol_sha,
                               "sample_id": sample.sample_id, "branch": branch, "mode": mode,
                               "predictions": [{"rank": proposal.rank, "bbox": list(proposal.bbox_original),
                                                "crop_rgb_sha256": _array_sha(crop), **prediction.to_dict()}
                                               for proposal, crop, prediction in zip(proposals, crops, predictions)]}
                        _atomic_json(ocr_path, ocr)
                    rows[branch][mode].append({
                        "sample_id": sample.sample_id, "ground_truth": shard["ground_truth"],
                        "predictions": [{"rank": value["rank"], "bbox": value["bbox"],
                                         "text": value["raw_text"]} for value in ocr["predictions"]],
                    })
            if index % 10 == 0 or index == 300:
                print(f"[curved crop PARSeq] {index}/300", flush=True)
    finally:
        if recognizer is not None:
            recognizer.release()
    summary = {branch: {mode: evaluate_images(rows[branch][mode], text_available=True)
                        for mode in MODES} for branch in BRANCHES}
    hardware["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device_index)
    hardware["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device_index)
    previous = json.loads((args.predicted_cache / "results.json").read_text(encoding="utf-8"))
    report = {"schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
              "result_class": "Total-Text TEST curved-crop recognition diagnostic; not paper Table 5",
              "protocol": protocol, "prepared_sha256": file_sha256(output / "prepared.json"),
              "parseq_checkpoint_sha256": PARSEQ_CHECKPOINT_SHA256,
              "parseq_source_archive_sha256": PARSEQ_RUNTIME_ARCHIVE_SHA256,
              "hardware": hardware,
              "baseline_predicted_padded": previous["summary"],
              "summary": summary,
              "ocr_shards": {branch: {mode: [f"ocr/{branch}/{mode}/{_key(sample.sample_id)}.json" for sample in manifest.samples]
                                      for mode in MODES} for branch in BRANCHES}}
    _atomic_json(output / "results.json", report)
    compact = {branch: {mode: {
        "gt": result["geometry"]["gt_instances"],
        "proposals": result["geometry"]["retained_proposals"],
        "iou_0_5": result["geometry"]["iou_ge_0_50_one_to_one"],
        "exact_case_sensitive": result["recognition"]["exact_case_sensitive_spatial_one_to_one"],
        "exact_casefold": result["recognition"]["exact_casefold_spatial_one_to_one"],
        "lexical_casefold": result["recognition"]["image_level_lexical_exact_multiset_matches"],
    } for mode, result in modes.items()} for branch, modes in summary.items()}
    print(json.dumps(compact, indent=2), flush=True)
    print("JSON", output / "results.json", "SHA256", file_sha256(output / "results.json"), flush=True)
    return 0


if __name__ == "__main__":
    args = _args()
    raise SystemExit(prepare(args) if args.action == "prepare" else recognize(args))
