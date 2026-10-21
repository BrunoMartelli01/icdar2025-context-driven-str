#!/usr/bin/env python3
"""Export compact visual audits from frozen scene-text V1/PARSeq results.

The exporter performs no model inference.  It selects representative successes
and failures using declared diagnostic categories, then renders one composite
JPEG per selected image.  Ground truth is used only for post-hoc selection and
annotation of the diagnostic sheets.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import textwrap
import zipfile

from PIL import Image, ImageDraw, ImageFont
from rapidfuzz.fuzz import ratio
import yaml

from recognition.scene_text_manifest import load_scene_text_manifest
from recognition.scene_text_v1_evaluation import box_iou


DATASETS = ("ctw1500", "icdar2015", "total_text")
CATEGORIES = (
    "EXACT_STRONG",
    "NEAR_TEXT_SUCCESS",
    "GOOD_BOX_OCR_FAILURE",
    "PARTIAL_LOCALIZATION",
    "LOCALIZATION_MISS",
    "MULTI_GT_MERGE",
)
SEGMENTATION_SOURCES = {
    "ctw1500": (
        "ctw1500-author-hyperparameters-approximate",
        "segmentation_ctw1500_author_hyperparameters.yaml",
    ),
    "icdar2015": (
        "icdar2015-author-hyperparameters-approximate",
        "segmentation_icdar2015_author_hyperparameters.yaml",
    ),
    "total_text": (
        "total-text-author-hyperparameters-approximate",
        "segmentation_total_text_author_hyperparameters.yaml",
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default()


def _sample_order(sample_id: str) -> tuple[int, str]:
    raw = sample_id.partition(":")[2]
    try:
        return int(raw), sample_id
    except ValueError:
        return 10**12, sample_id


def _intersection_over_gt(proposal: list[int], gt: list[int]) -> float:
    px1, py1, px2, py2 = proposal
    gx1, gy1, gx2, gy2 = gt
    intersection = max(0, min(px2, gx2) - max(px1, gx1)) * max(
        0, min(py2, gy2) - max(py1, gy1)
    )
    area = max(0, gx2 - gx1) * max(0, gy2 - gy1)
    return intersection / area if area else 0.0


def _manifest_samples(data_root: Path, dataset: str) -> dict[str, dict]:
    manifest = data_root / "manifests" / f"{dataset}_test.yaml"
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload["samples"]}


def _ctw_ground_truth(data_root: Path) -> dict[str, list[dict]]:
    manifest = data_root / "manifests" / "ctw1500_recognition_test.json"
    samples, qualification = load_scene_text_manifest(manifest, data_root)
    if qualification["split"] != "test" or len(samples) != 500:
        raise ValueError("CTW1500 recognition manifest is not qualified TEST-500")
    result: dict[str, list[dict]] = {}
    for sample in samples:
        width, height = sample.image_size_wh
        words = []
        for instance in sample.instances:
            if instance.ignore:
                continue
            x1, y1, x2, y2 = instance.bbox_xyxy
            words.append({
                "bbox": [max(0, x1), max(0, y1), min(width, x2 + 1), min(height, y2 + 1)],
                "text": instance.transcription,
                "granularity": "text_line",
            })
        result[f"ctw1500:{int(sample.sample_id)}"] = words
    return result


def _load_frozen_rows(data_root: Path, run_root: Path, dataset: str) -> tuple[list[dict], dict]:
    run = run_root / f"{dataset}-specialized"
    result_path = run / "results.json"
    protocol_path = run / "protocol.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    expected = int(result["summary"]["images"])
    records = _manifest_samples(data_root, dataset)
    ctw_gt = _ctw_ground_truth(data_root) if dataset == "ctw1500" else None
    rows = []
    for proposal_path in sorted((run / "proposals").glob("*.json")):
        proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
        sample_id = str(proposal["sample_id"])
        if sample_id not in records:
            raise ValueError(f"sample absent from TEST manifest: {sample_id}")
        predictions = []
        key = sample_id.replace(":", "__")
        for block in proposal["proposals"]:
            rank = int(block["rank"])
            ocr_path = run / "ocr" / f"{key}__{rank:02d}.json"
            ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
            predictions.append({
                "rank": rank,
                "bbox": [int(value) for value in block["bbox"]],
                "text": str(ocr["raw_text"]),
                "confidence": float(ocr["sequence_confidence_native"]),
                "foreground_area": int(block["foreground_area"]),
            })
        ground_truth = ctw_gt[sample_id] if ctw_gt is not None else [
            value for value in proposal["ground_truth"] if not value.get("ignore", False)
        ]
        rows.append({
            "dataset": dataset,
            "sample_id": sample_id,
            "image": records[sample_id]["image"],
            "mask": records[sample_id]["mask"],
            "ground_truth": ground_truth,
            "predictions": predictions,
            "raw_component_count": int(proposal["raw_component_count"]),
            "groups_before_cap": int(proposal["groups_before_cap"]),
            "segmentation": proposal["segmentation"],
        })
    if len(rows) != expected:
        raise ValueError(f"{dataset}: expected {expected} frozen samples, found {len(rows)}")
    return rows, {
        "results": str(result_path.resolve()),
        "results_sha256": sha256_file(result_path),
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256_file(protocol_path),
        "summary": result["summary"],
    }


def _annotate(row: dict) -> dict:
    gt = row["ground_truth"]
    predictions = row["predictions"]
    gt_rows = []
    for index, word in enumerate(gt):
        ranked = sorted(
            predictions,
            key=lambda value: (-box_iou(value["bbox"], word["bbox"]), value["rank"]),
        )
        best = ranked[0] if ranked else None
        iou = box_iou(best["bbox"], word["bbox"]) if best else 0.0
        text = best["text"] if best else ""
        gt_rows.append({
            "gt_index": index,
            "bbox": word["bbox"],
            "text": str(word["text"]),
            "best_rank": best["rank"] if best else None,
            "best_text": text,
            "best_iou": iou,
            "strict_exact": bool(best and iou >= 0.5 and text == str(word["text"])),
            "casefold_exact": bool(best and iou >= 0.5 and text.casefold() == str(word["text"]).casefold()),
            "fuzzy": ratio(text.casefold(), str(word["text"]).casefold()) / 100.0 if best else 0.0,
        })
    prediction_rows = []
    for prediction in predictions:
        overlaps = [
            (index, box_iou(prediction["bbox"], word["bbox"]), _intersection_over_gt(prediction["bbox"], word["bbox"]))
            for index, word in enumerate(gt)
        ]
        best = max(overlaps, key=lambda value: (value[1], -value[0])) if overlaps else (None, 0.0, 0.0)
        prediction_rows.append({
            **prediction,
            "best_gt_index": best[0],
            "best_gt_iou": best[1],
            "significant_gt_indices": [index for index, _, coverage in overlaps if coverage >= 0.25],
        })
    return {**row, "gt_diagnostics": gt_rows, "prediction_diagnostics": prediction_rows}


def _category_scores(row: dict) -> dict[str, tuple]:
    gt = row["gt_diagnostics"]
    predictions = row["prediction_diagnostics"]
    exact = [value for value in gt if value["strict_exact"]]
    near = [
        value for value in gt
        if value["best_iou"] >= 0.5 and not value["strict_exact"] and value["fuzzy"] >= 0.8
    ]
    good_box_bad_ocr = [value for value in gt if value["best_iou"] >= 0.75 and value["fuzzy"] < 0.8]
    partial = [value for value in gt if 0.25 <= value["best_iou"] < 0.5]
    missed = [value for value in gt if value["best_iou"] < 0.10]
    multi = [value for value in predictions if len(value["significant_gt_indices"]) >= 2]
    return {
        "EXACT_STRONG": (len(exact), sum(value["best_iou"] for value in exact)) if exact else (),
        "NEAR_TEXT_SUCCESS": (len(near), max((value["fuzzy"] for value in near), default=0.0)) if near else (),
        "GOOD_BOX_OCR_FAILURE": (len(good_box_bad_ocr), max((value["best_iou"] for value in good_box_bad_ocr), default=0.0)) if good_box_bad_ocr else (),
        "PARTIAL_LOCALIZATION": (len(partial), max((value["best_iou"] for value in partial), default=0.0)) if partial else (),
        "LOCALIZATION_MISS": (len(missed), -max((value["best_iou"] for value in missed), default=0.0)) if missed else (),
        "MULTI_GT_MERGE": (max((len(value["significant_gt_indices"]) for value in multi), default=0), len(multi)) if multi else (),
    }


def _select(rows: list[dict]) -> list[tuple[str, dict]]:
    selected: list[tuple[str, dict]] = []
    used: set[str] = set()
    for category in CATEGORIES:
        candidates = []
        for row in rows:
            score = _category_scores(row)[category]
            if score and row["sample_id"] not in used:
                candidates.append((score, _sample_order(row["sample_id"]), row))
        if candidates:
            candidates.sort(key=lambda value: (value[0], tuple(-x if isinstance(x, int) else x for x in value[1])), reverse=True)
            chosen = candidates[0][2]
        else:
            fallbacks = [row for row in rows if row["sample_id"] not in used]
            if not fallbacks:
                break
            fallbacks.sort(key=lambda value: _sample_order(value["sample_id"]))
            chosen = fallbacks[0]
            category = f"FALLBACK_{category}"
        selected.append((category, chosen))
        used.add(chosen["sample_id"])
    return selected


def _attach_verified_prediction_masks(
    *,
    data_root: Path,
    dataset: str,
    selected: list[tuple[str, dict]],
    segmentation_runs_root: Path,
    configs_root: Path,
    device: str,
    mask_output: Path,
) -> None:
    """Rerun only selected masks and require identity with frozen GPU hashes."""

    import numpy as np
    import torch

    from pipeline.frozen_segmentation import backproject_binary_mask
    from scripts.evaluate_scene_text_v1_parseq import _Segmenter, _preprocess

    run_name, config_name = SEGMENTATION_SOURCES[dataset]
    run_manifest_path = segmentation_runs_root / run_name / "run_manifest.json"
    run = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run.get("status") != "SUCCESS" or run.get("success") is not True:
        raise ValueError(f"segmentation run is not successful: {run_manifest_path}")
    if len(run.get("epochs", [])) != 100:
        raise ValueError(f"segmentation run does not contain 100 epochs: {run_manifest_path}")
    checkpoint = Path(run["checkpoint_output_location"]).resolve()
    if sha256_file(checkpoint) != run["checkpoint"]["sha256"]:
        raise ValueError(f"segmentation checkpoint SHA mismatch: {checkpoint}")
    segmenter = _Segmenter(
        configs_root / config_name,
        checkpoint,
        run["checkpoint"]["sha256"],
        device,
    )
    mask_output.mkdir(parents=True, exist_ok=True)
    try:
        for index, (_, row) in enumerate(selected, start=1):
            with Image.open(data_root / row["image"]) as opened:
                rgb = opened.convert("RGB")
                image = np.asarray(rgb)
            inputs = _preprocess(rgb).unsqueeze(0).to(segmenter.device)
            with torch.inference_mode():
                logits = segmenter.model(inputs)
            if tuple(logits.shape) != (1, 1, 128, 128):
                raise RuntimeError(f"unexpected logits shape: {list(logits.shape)}")
            native = np.ascontiguousarray(
                (logits.sigmoid()[0, 0] >= 0.5).to(torch.uint8).cpu().numpy()
            )
            projected = np.ascontiguousarray(backproject_binary_mask(native, image.shape[:2]))
            actual = {
                "native_mask_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
                "projected_mask_sha256": hashlib.sha256(projected.tobytes()).hexdigest(),
                "native_foreground_pixels": int(native.sum()),
                "projected_foreground_pixels": int(projected.sum()),
            }
            for key, value in actual.items():
                if value != row["segmentation"][key]:
                    raise RuntimeError(
                        f"{dataset} {row['sample_id']}: regenerated {key} differs from frozen run "
                        f"({value!r} != {row['segmentation'][key]!r})"
                    )
            stem = row["sample_id"].replace(":", "__")
            native_name = f"{stem}__native128.png"
            projected_name = f"{stem}__projected.png"
            Image.fromarray(native * 255, mode="L").save(mask_output / native_name, optimize=True)
            Image.fromarray(projected * 255, mode="L").save(mask_output / projected_name, optimize=True)
            row["_predicted_native"] = native
            row["_predicted_projected"] = projected
            row["prediction_masks"] = {
                "native128_file": f"predicted_masks/{native_name}",
                "native128_file_sha256": sha256_file(mask_output / native_name),
                "projected_file": f"predicted_masks/{projected_name}",
                "projected_file_sha256": sha256_file(mask_output / projected_name),
                "frozen_array_hashes_verified": True,
                "device": device,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": run["checkpoint"]["sha256"],
            }
            print(
                f"[{dataset}] U-Net mask {index}/{len(selected)} verified: {row['sample_id']}",
                flush=True,
            )
    finally:
        segmenter.release()
        del segmenter
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _fit(image: Image.Image, size: tuple[int, int], nearest: bool = False) -> Image.Image:
    source = image.convert("RGB")
    method = Image.Resampling.NEAREST if nearest else Image.Resampling.LANCZOS
    source.thumbnail(size, method)
    canvas = Image.new("RGB", size, "#eeeeee")
    canvas.paste(source, ((size[0] - source.width) // 2, (size[1] - source.height) // 2))
    return canvas


def _boxes(image: Image.Image, rows: list[dict], *, gt: bool) -> Image.Image:
    shown = image.convert("RGB").copy()
    draw = ImageDraw.Draw(shown)
    face = _font(max(11, round(min(image.size) / 40)), True)
    for index, row in enumerate(rows):
        box = row["bbox"]
        color = "#00ff55" if gt else "#00d9ff"
        label = f"GT{index}: {row.get('text', '')}" if gt else f"P{row['rank']}: {row['text']}"
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=color, width=max(2, round(min(image.size) / 250)))
        draw.text((box[0] + 2, max(0, box[1] - face.size - 3)), label[:48], font=face, fill=color, stroke_width=2, stroke_fill="black")
    return shown


def _render_sheet(data_root: Path, category: str, row: dict, output: Path) -> dict:
    image_path = data_root / row["image"]
    mask_path = data_root / row["mask"]
    with Image.open(image_path) as opened:
        rgb = opened.convert("RGB")
    with Image.open(mask_path) as opened:
        gt_mask = opened.convert("L")
    gt_boxes = _boxes(rgb, row["ground_truth"], gt=True)
    pred_boxes = _boxes(rgb, row["predictions"], gt=False)

    width = 1800
    has_predictions = "_predicted_native" in row
    panel_columns = 3 if has_predictions else 4
    panel_w = 580 if has_predictions else 430
    panel_h = 330 if has_predictions else 350
    panel_rows = 2 if has_predictions else 1
    columns = 4
    crop_count = len(row["prediction_diagnostics"])
    crop_rows = max(1, math.ceil(crop_count / columns))
    panel_bottom = 112 + panel_rows * (panel_h + 52)
    height = panel_bottom + 90 + crop_rows * 230 + 100
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, width, 70), fill="#223148")
    draw.text((24, 14), f"{row['dataset']} TEST — {row['sample_id']} — {category}", font=_font(30, True), fill="white")
    panels = [
        ("Original RGB", rgb, False),
        ("GT binary mask", gt_mask, True),
    ]
    if has_predictions:
        panels.extend([
            ("U-Net prediction native 128x128 (SHA verified)", Image.fromarray(row["_predicted_native"] * 255, mode="L"), True),
            ("U-Net prediction remapped to original (SHA verified)", Image.fromarray(row["_predicted_projected"] * 255, mode="L"), True),
        ])
    panels.extend([
        ("Care GT boxes (green)", gt_boxes, False),
        ("Frozen V1 boxes + PARSeq (cyan)", pred_boxes, False),
    ])
    for index, (title, image, nearest) in enumerate(panels):
        col, line = index % panel_columns, index // panel_columns
        x = 20 + col * (panel_w + 15)
        panel_y = 82 + line * (panel_h + 52)
        draw.text((x, panel_y), title, font=_font(18, True), fill="#111111")
        canvas.paste(_fit(image, (panel_w, panel_h), nearest=nearest), (x, panel_y + 30))

    exact = sum(value["strict_exact"] for value in row["gt_diagnostics"])
    good = sum(value["best_iou"] >= 0.5 for value in row["gt_diagnostics"])
    draw.text(
        (28, panel_bottom + 6),
        f"GT={len(row['ground_truth'])} | proposals={len(row['predictions'])} | "
        f"GT with best IoU>=.50={good} | strict exact={exact} | "
        f"raw CC={row['raw_component_count']} | groups precap={row['groups_before_cap']}",
        font=_font(20, True), fill="#17243a",
    )
    draw.text((28, panel_bottom + 40), "Retained crops (actual frozen bbox and PARSeq output)", font=_font(22, True), fill="#111111")
    y0 = panel_bottom + 78
    card_w = 440
    for index, prediction in enumerate(row["prediction_diagnostics"]):
        col, line = index % columns, index // columns
        x, y = 20 + col * 445, y0 + line * 230
        draw.rectangle((x, y, x + card_w - 8, y + 215), outline="#9aa3ad", width=2, fill="#fafafa")
        x1, y1, x2, y2 = prediction["bbox"]
        crop = rgb.crop((x1, y1, x2, y2))
        canvas.paste(_fit(crop, (card_w - 28, 120)), (x + 10, y + 8))
        gt_index = prediction["best_gt_index"]
        gt_text = row["ground_truth"][gt_index]["text"] if gt_index is not None else "<none>"
        draw.text((x + 10, y + 134), f"P{prediction['rank']:02d} PARSeq: {prediction['text'][:40]}", font=_font(17, True), fill="#111111")
        draw.text((x + 10, y + 160), f"best GT: {str(gt_text)[:38]} | IoU={prediction['best_gt_iou']:.3f}", font=_font(15), fill="#333333")
        draw.text((x + 10, y + 185), f"bbox={prediction['bbox']} | conf={prediction['confidence']:.4f}", font=_font(13), fill="#555555")
    canvas.save(output, quality=90, optimize=True)
    return {"file": output.name, "sha256": sha256_file(output), "size_bytes": output.stat().st_size}


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _deterministic_zip(source: Path, output: Path) -> None:
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(value for value in source.rglob("*") if value.is_file()):
            info = zipfile.ZipInfo(str(path.relative_to(source)), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zip-output", type=Path, required=True)
    parser.add_argument("--include-predicted-masks", action="store_true")
    parser.add_argument("--segmentation-runs-root", type=Path)
    parser.add_argument("--configs-root", type=Path, default=Path("configs/approximate"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    run_root = args.run_root.resolve()
    output = args.output_dir.resolve()
    zip_output = args.zip_output.resolve()
    if output.exists() or zip_output.exists():
        raise FileExistsError("refusing to overwrite an existing visual bundle")
    output.mkdir(parents=True)
    sheets = output / "sheets"
    sheets.mkdir()
    if args.include_predicted_masks and args.segmentation_runs_root is None:
        raise ValueError("--segmentation-runs-root is required with --include-predicted-masks")

    sources = {}
    selections = []
    for dataset in DATASETS:
        frozen, source = _load_frozen_rows(data_root, run_root, dataset)
        annotated = [_annotate(row) for row in frozen]
        sources[dataset] = source
        selected = _select(annotated)
        if args.include_predicted_masks:
            _attach_verified_prediction_masks(
                data_root=data_root,
                dataset=dataset,
                selected=selected,
                segmentation_runs_root=args.segmentation_runs_root.resolve(),
                configs_root=args.configs_root.resolve(),
                device=args.device,
                mask_output=output / "predicted_masks",
            )
        for category, row in selected:
            name = f"{dataset}__{row['sample_id'].partition(':')[2]}__{category.lower()}.jpg"
            visual = _render_sheet(data_root, category, row, sheets / name)
            selections.append({
                "dataset": dataset,
                "sample_id": row["sample_id"],
                "category": category,
                "annotation_granularity": row["ground_truth"][0].get("granularity") if row["ground_truth"] else None,
                "gt_count": len(row["ground_truth"]),
                "proposal_count": len(row["predictions"]),
                "gt_iou_ge_0_50": sum(value["best_iou"] >= 0.5 for value in row["gt_diagnostics"]),
                "strict_exact": sum(value["strict_exact"] for value in row["gt_diagnostics"]),
                "ground_truth": row["gt_diagnostics"],
                "predictions": row["prediction_diagnostics"],
                "prediction_masks": row.get("prediction_masks"),
                "visual": visual,
            })
            print(f"[{dataset}] {category}: {row['sample_id']} -> {name}", flush=True)

    manifest = {
        "schema_version": 1,
        "classification": "PRACTICAL_APPROXIMATION_DIAGNOSTIC",
        "split": "official TEST splits; selected visual audit only",
        "table_5_claim": False,
        "inference_performed": bool(args.include_predicted_masks),
        "inference_scope": "selected 18 images only" if args.include_predicted_masks else None,
        "selection_uses_gt": True,
        "selection_use": "post-hoc visual diagnosis only; never model or parameter selection",
        "selection_categories": list(CATEGORIES),
        "important_scope_note": "CTW1500 annotations are text-line level; ICDAR2015 and Total-Text annotations are word level.",
        "predicted_mask_note": (
            "Selected native and projected U-Net masks were regenerated and accepted only after their array SHA256 and "
            "foreground counts matched the frozen GPU result shards."
            if args.include_predicted_masks else
            "Frozen result shards preserve mask hashes but not mask pixels. Predicted-mask panels are omitted."
        ),
        "sources": sources,
        "selections": selections,
    }
    _write_json(output / "manifest.json", manifest)
    with (output / "selection.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "dataset", "sample_id", "category", "annotation_granularity", "gt_count",
            "proposal_count", "gt_iou_ge_0_50", "strict_exact", "visual_file", "visual_sha256",
        ))
        writer.writeheader()
        for row in selections:
            writer.writerow({
                **{key: row[key] for key in writer.fieldnames if key in row},
                "visual_file": f"sheets/{row['visual']['file']}",
                "visual_sha256": row["visual"]["sha256"],
            })
    inference_sentence = (
        "U-Net inference was rerun only for these 18 selected images on the requested device; every native and "
        "projected mask was accepted only after exact comparison with the frozen array SHA256 and foreground count."
        if args.include_predicted_masks else
        "No inference was rerun; predicted-mask pixels were unavailable and are not shown."
    )
    (output / "README.md").write_text(
        "# Selected scene-text visual audit\n\n"
        "This compact bundle contains six deterministic diagnostic examples per dataset: two success-oriented "
        "categories and four important failure categories. Each JPEG contains the original image, prepared GT "
        "binary mask, U-Net masks when requested, care GT boxes/transcriptions, frozen practical-remap-v1 proposal boxes, actual crops, and "
        f"frozen PARSeq outputs. {inference_sentence}\n\n"
        "The examples are selected with GT only for post-hoc diagnosis and must not be used for tuning. CTW1500 "
        "is evaluated at text-line granularity, while ICDAR2015 and Total-Text are word-level. This is a "
        "PRACTICAL_APPROXIMATION diagnostic, not a paper Table 5 result.\n",
        encoding="utf-8",
    )
    _deterministic_zip(output, zip_output)
    print(f"BUNDLE {output}")
    print(f"ZIP {zip_output} SHA256 {sha256_file(zip_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
