"""CPU-only TRAIN ceiling for the frozen v4 native-128 contour localizer.

The *training target* replaces model prediction. This is a geometric oracle,
not a recognition experiment and not an independent evaluation of the model.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image

from localize.native_repository_contours import remap_native_box
from localize.repository_contours import audit_repository_contours
from recognition.scene_text_datasets import parse_totaltext_annotation
from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.evaluate_totaltext_v1_findcontours_parseq import _localization_params
from segmentation.config import load_segmentation_config
from segmentation.data import SegmentationDataset, file_sha256, load_segmentation_manifest
from segmentation.train import _transform


EXPECTED_MANIFEST_SHA256 = "7ad8525d66294e07b2d63be0dd68d08a2e83dbecc04a4d55fb65cd6110c51bea"
EXPECTED_QUALIFICATION_SHA256 = "ea4edfc47954b1158d7299a1a37742d60bf92a8e1132b64149a32fe170806856"
EXPECTED_NATIVE_QUALIFICATION_SHA256 = "d4d8c94bc84d127d487f62d811e53bfec31e5126b3091d41c103406f4fa06a13"
THRESHOLDS = (0.25, 0.5, 0.75)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--segmentation-config", type=Path, required=True)
    parser.add_argument("--localization-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _gt_boxes(annotation: Path, image_wh: tuple[int, int]) -> list[list[int]]:
    width, height = image_wh
    boxes = []
    for item in parse_totaltext_annotation(annotation):
        if item.ignore:
            continue
        x1, y1, x2, y2 = item.bbox_xyxy
        box = [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)]
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"Degenerate care-word box in {annotation}: {box}")
        boxes.append(box)
    return boxes


def _stage_boxes(contours: dict, *, image_hw: tuple[int, int], min_area: int) -> dict[str, list[list[int]]]:
    """Recover all pre-filter boxes, checking the frozen localizer's retained output."""
    height, width = contours["mask_shape"]
    items = sorted(
        [*contours["retained"], *contours["rejected"]],
        key=lambda item: item["contour_index"],
    )
    if len(items) != contours["raw_external_contours"]:
        raise ValueError("A raw contour was lost from the audit")
    pad = int(contours["parameters"]["pad"])
    cap = int(contours["parameters"]["nmax"])
    accepted = []
    raw = []
    raw_tight = []
    for index, item in enumerate(items):
        if item["contour_index"] != index:
            raise ValueError("Non-contiguous contour IDs")
        x1, y1, x2, y2 = item["bbox"]
        padded = [max(0, x1 - pad), max(0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)]
        if "padded_bbox" in item and item["padded_bbox"] != padded:
            raise ValueError("Frozen contour padding mismatch")
        tight_box = remap_native_box(item["bbox"], image_hw)
        box = remap_native_box(padded, image_hw)
        raw.append(box)
        raw_tight.append(tight_box)
        if item["bbox_area"] >= min_area:
            accepted.append((item["contour_index"], box, tight_box, (padded[2] - padded[0]) * (padded[3] - padded[1])))
    accepted.sort(key=lambda entry: entry[3], reverse=True)
    retained = accepted[:cap]
    if [entry[0] for entry in retained] != [item["contour_index"] for item in contours["retained"]]:
        raise ValueError("Reconstructed top10 differs from frozen contour localizer")
    if len(accepted) != contours["candidates_before_cap"]:
        raise ValueError("Reconstructed area filter differs from frozen contour localizer")
    return {
        "raw": raw,
        "area_pass": [entry[1] for entry in accepted],
        "retained": [entry[1] for entry in retained],
        "raw_tight_diagnostic_only": raw_tight,
        "area_pass_tight_diagnostic_only": [entry[2] for entry in accepted],
        "retained_tight_diagnostic_only": [entry[2] for entry in retained],
    }


def _sample_geometry(gt: list[list[int]], stages: dict[str, list[list[int]]]) -> dict:
    best = {name: [] for name in stages}
    one_to_one = {name: {} for name in stages}
    for name, boxes in stages.items():
        overlaps = [[box_iou(box, word) for word in gt] for box in boxes]
        best[name] = [max((row[index] for row in overlaps), default=0.0) for index in range(len(gt))]
        for threshold in THRESHOLDS:
            edges = [
                (prediction, word)
                for prediction, row in enumerate(overlaps)
                for word, iou in enumerate(row)
                if iou >= threshold
            ]
            one_to_one[name][str(threshold)] = len(maximum_matching(len(boxes), len(gt), edges))
    losses = {key: 0 for key in ("no_raw_iou_0_5", "area_filter", "top10", "retained_but_competition", "retained_iou_0_5")}
    for index in range(len(gt)):
        if best["raw"][index] < 0.5:
            losses["no_raw_iou_0_5"] += 1
        elif best["area_pass"][index] < 0.5:
            losses["area_filter"] += 1
        elif best["retained"][index] < 0.5:
            losses["top10"] += 1
        else:
            losses["retained_iou_0_5"] += 1
    losses["retained_but_competition"] = losses["retained_iou_0_5"] - one_to_one["retained"]["0.5"]
    if sum(losses[key] for key in ("no_raw_iou_0_5", "area_filter", "top10", "retained_iou_0_5")) != len(gt):
        raise AssertionError("Loss attribution does not cover all words")
    result = {"best_iou": best, "one_to_one": one_to_one, "losses": losses}
    tight_names = ("raw_tight_diagnostic_only", "area_pass_tight_diagnostic_only", "retained_tight_diagnostic_only")
    if all(name in best for name in tight_names):
        first, second, third = (best[name] for name in tight_names)
        tight_losses = {
            "no_raw_tight_iou_0_5": sum(value < 0.5 for value in first),
            "area_filter_tight": sum(a >= 0.5 and b < 0.5 for a, b in zip(first, second)),
            "top10_tight": sum(a >= 0.5 and b < 0.5 for a, b in zip(second, third)),
            "retained_tight_iou_0_5": sum(value >= 0.5 for value in third),
        }
        if sum(tight_losses.values()) != len(gt):
            raise AssertionError("Tight-box loss attribution does not cover all words")
        result["tight_losses"] = tight_losses
    return result


def main() -> int:
    args = _arguments()
    root = args.target_root.resolve()
    manifest_path = root / "manifests/total_text_train.yaml"
    for path, expected in (
        (manifest_path, EXPECTED_MANIFEST_SHA256),
        (root / "qualification.json", EXPECTED_QUALIFICATION_SHA256),
        (root / "qualification_native128.json", EXPECTED_NATIVE_QUALIFICATION_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen v4 target provenance mismatch: {path}")
    manifest = load_segmentation_manifest(manifest_path, require_checksums=True)
    config = load_segmentation_config(args.segmentation_config.resolve())
    if manifest.dataset != "total_text" or manifest.split_role != "train" or len(manifest.samples) != 1255:
        raise ValueError("Expected official Total-Text TRAIN-1255 v4 target")
    if (config.run.variant_class != "PRACTICAL_APPROXIMATION"
        or config.data.train.manifest_sha256 != EXPECTED_MANIFEST_SHA256
        or tuple(config.transforms.mask_size) != (128, 128)
        or config.transforms.mask_interpolation != "bilinear"
        or float(config.transforms.target_binarization_threshold) != 0.5):
        raise ValueError("Segmentation config is not the frozen v4 training target protocol")
    params = _localization_params(args.localization_config.resolve())
    if (params.min_area, params.pad, params.nmax) != (150, 6, 10):
        raise ValueError("Expected frozen native-128 localizer 150/6/10")
    transform = _transform(config.transforms, seed=int(config.training.seed), augmentation_enabled=False)
    dataset = SegmentationDataset(manifest, root, transform=transform, verify_checksums=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic: {output}")

    stage_names = ("raw", "area_pass", "retained", "raw_tight_diagnostic_only", "area_pass_tight_diagnostic_only", "retained_tight_diagnostic_only")
    totals = {name: {str(t): 0 for t in THRESHOLDS} for name in stage_names}
    best_counts = {name: {str(t): 0 for t in THRESHOLDS} for name in totals}
    all_best = {name: [] for name in totals}
    counts = {name: 0 for name in totals}
    losses = {key: 0 for key in ("no_raw_iou_0_5", "area_filter", "top10", "retained_but_competition", "retained_iou_0_5")}
    tight_losses = {key: 0 for key in ("no_raw_tight_iou_0_5", "area_filter_tight", "top10_tight", "retained_tight_iou_0_5")}
    padding_area_ratios = []
    records = []
    for index, sample in enumerate(manifest.samples):
        transformed = dataset[index]
        native = np.ascontiguousarray(transformed["mask"].squeeze(0).numpy().astype(np.uint8))
        if native.shape != (128, 128) or not np.isin(native, (0, 1)).all():
            raise ValueError(f"Unexpected native target: {sample.sample_id}")
        image_path = root / sample.image_path
        with Image.open(image_path) as image:
            image_wh = image.size
        annotation = root / "datasets/total_text/train/annotations" / (sample.sample_id.split(":", 1)[1] + ".txt")
        gt = _gt_boxes(annotation, image_wh)
        contours = audit_repository_contours(native, params)
        stages = _stage_boxes(contours, image_hw=(image_wh[1], image_wh[0]), min_area=params.min_area)
        for padded, tight in zip(stages["retained"], stages["retained_tight_diagnostic_only"]):
            padded_area = (padded[2] - padded[0]) * (padded[3] - padded[1])
            tight_area = (tight[2] - tight[0]) * (tight[3] - tight[1])
            padding_area_ratios.append(padded_area / tight_area)
        geometry = _sample_geometry(gt, stages)
        for name, boxes in stages.items():
            counts[name] += len(boxes)
            all_best[name].extend(geometry["best_iou"][name])
            for threshold in THRESHOLDS:
                key = str(threshold)
                totals[name][key] += geometry["one_to_one"][name][key]
                best_counts[name][key] += sum(value >= threshold for value in geometry["best_iou"][name])
        for key, value in geometry["losses"].items():
            losses[key] += value
        for key, value in geometry["tight_losses"].items():
            tight_losses[key] += value
        records.append({
            "sample_id": sample.sample_id,
            "gt_words": len(gt),
            "native_mask_sha256": sha256(native.tobytes()).hexdigest(),
            "raw_contours": contours["raw_external_contours"],
            "candidates_before_cap": contours["candidates_before_cap"],
            "retained": len(stages["retained"]),
            "one_to_one": geometry["one_to_one"],
            "losses": geometry["losses"],
            "tight_losses": geometry["tight_losses"],
        })
        if (index + 1) % 50 == 0 or index + 1 == len(manifest.samples):
            print(f"[perfect-mask TRAIN] {index + 1}/{len(manifest.samples)}", flush=True)
    words = sum(record["gt_words"] for record in records)
    if words != 9285:
        raise ValueError(f"Unexpected official TRAIN care-word count: {words}")
    if totals["retained"]["0.5"] != losses["retained_iou_0_5"] - losses["retained_but_competition"]:
        raise AssertionError("One-to-one accounting mismatch")
    summary = {
        "images": len(records), "care_words": words,
        "stages": {
            name: {
                "proposals": counts[name], "proposals_per_image": counts[name] / len(records),
                "one_to_one_iou": totals[name],
                "best_per_gt_iou": best_counts[name],
                "median_best_iou": median(all_best[name]),
            } for name in totals
        },
        "images_over_top10_before_cap": sum(record["candidates_before_cap"] > 10 for record in records),
        "median_retained_padded_to_tight_area_ratio": median(padding_area_ratios),
        "loss_attribution_best_per_gt_iou_0_5": losses,
        "tight_control_loss_attribution_best_per_gt_iou_0_5": tight_losses,
    }
    report = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION",
        "result_class": "TRAIN perfect-target geometric oracle; not model evaluation, OCR accuracy, or paper Table 5",
        "provenance": {
            "target_manifest": str(manifest_path),
            "target_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "target_qualification_sha256": EXPECTED_QUALIFICATION_SHA256,
            "target_native_qualification_sha256": EXPECTED_NATIVE_QUALIFICATION_SHA256,
            "segmentation_config": str(args.segmentation_config.resolve()),
            "segmentation_config_sha256": file_sha256(args.segmentation_config.resolve()),
            "localization_config": str(args.localization_config.resolve()),
            "localization_config_sha256": file_sha256(args.localization_config.resolve()),
            "native_target": "v4 TRAIN mask through the exact no-augmentation training selection transform, including validity/owner/contact separation",
            "contour_filter": "frozen approximate unpadded native bbox area >=150; differs from paper component-area filter",
            "padding": "6 native pixels before original-image remap",
            "tight_controls": "zero-padding boxes only as a causal diagnostic; not a chosen inference variant",
            "sorting": "clipped padded bbox area, stable contour order",
            "cap": 10,
            "gt_use": "only for post-hoc geometry; no candidate selection or OCR",
        },
        "summary": summary,
        "samples": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Refusing to overwrite partial diagnostic: {temporary}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    temporary.replace(output)
    print(json.dumps(summary, indent=2), flush=True)
    print("JSON", output, "SHA256", file_sha256(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
