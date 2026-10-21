"""One composite PNG per TEST image: RGB, both GT masks, prediction and all pad-0 crops.

This is a read-only visual audit of frozen outputs. No segmentation or PARSeq
inference is run, and the existing report is never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.sweep_totaltext_v4_gt_mask_resolution import _regenerate, resize_v4_target
from segmentation.data import file_sha256, load_segmentation_manifest


EXPECTED_RESULT_SHA256 = "6aebbfa409240c8f1b4dabc67a53eff37bf53fc6c9f1d542a22ec0a5cfda9df7"
EXPECTED_FROZEN_V4_SHA256 = "272613b094777b9836a1ffde2fd9265fcec17c84e7ccb60661f9a87ac0e29989"
EXPECTED_TEST_MANIFEST_SHA256 = "da2f87e7cae765f426af39c9641535138d6657d2e9332e6fdf9e80f2ba6b69bd"
VARIANT = "highres_separator_residual"
PADDING = "0"
CARD_W, CARD_H, COLUMNS = 340, 186, 4
CANVAS_W = CARD_W * COLUMNS + 40


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def matched_crop_to_word(predictions: list[dict], words: list[dict]) -> dict[int, int]:
    """Same deterministic maximum matching as the frozen recognition metric."""
    adjacency = [[word_index for word_index, word in enumerate(words)
                  if box_iou(prediction["bbox"], word["bbox"]) >= 0.5
                  and prediction["text"].casefold() == word["text"].casefold()]
                 for prediction in predictions]
    assigned: dict[int, int] = {}

    def visit(prediction_index: int, seen: set[int]) -> bool:
        for word_index in adjacency[prediction_index]:
            if word_index in seen:
                continue
            seen.add(word_index)
            previous = assigned.get(word_index)
            if previous is None or visit(previous, seen):
                assigned[word_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predictions)):
        visit(prediction_index, set())
    return {prediction_index: word_index for word_index, prediction_index in assigned.items()}


def classify_crop(prediction: dict, words: list[dict], *, matched_word_index: int | None) -> dict:
    best_index = max(range(len(words)), key=lambda index: box_iou(prediction["bbox"], words[index]["bbox"])) if words else None
    best_iou = box_iou(prediction["bbox"], words[best_index]["bbox"]) if best_index is not None else 0.0
    exact_spatial = [index for index, word in enumerate(words)
                     if box_iou(prediction["bbox"], word["bbox"]) >= 0.5
                     and prediction["text"].casefold() == word["text"].casefold()]
    if matched_word_index is not None:
        status = "CORRECT"
    elif exact_spatial:
        status = "DUPLICATE_MATCH"
    elif any(prediction["text"].casefold() == word["text"].casefold() for word in words):
        status = "TEXT_OK_BOX_BAD"
    elif best_iou >= 0.5:
        status = "OCR_WRONG"
    elif best_iou > 0:
        status = "BOX_PARTIAL"
    else:
        status = "NO_GT_OVERLAP"
    return {"status": status, "best_gt_index": best_index, "best_gt_iou": best_iou,
            "matched_gt_index": matched_word_index,
            "best_gt_text": words[best_index]["text"] if best_index is not None else None}


def _fit(canvas: Image.Image, source: Image.Image, box: tuple[int, int, int, int], *, mask: bool = False) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    copy = source.convert("RGB")
    copy.thumbnail((right - left, bottom - top), Image.Resampling.NEAREST if mask else Image.Resampling.LANCZOS)
    x = left + (right - left - copy.width) // 2
    y = top + (bottom - top - copy.height) // 2
    canvas.paste(copy, (x, y))
    return x, y, copy.width, copy.height


def _wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int,
             *, font: ImageFont.ImageFont, fill: tuple[int, int, int], max_lines: int = 2) -> None:
    # Full strings are also stored in the ZIP manifest; visual rows are width-limited.
    pending = text.replace("\n", " ")
    for line_number in range(max_lines):
        if not pending:
            break
        fit = len(pending)
        while fit > 1 and draw.textbbox((0, 0), pending[:fit], font=font)[2] > width:
            fit -= 1
        line = pending[:fit]
        if line_number == max_lines - 1 and fit < len(pending):
            while fit > 1 and draw.textbbox((0, 0), line.rstrip() + "…", font=font)[2] > width:
                fit -= 1
                line = pending[:fit]
            line = line.rstrip() + "…"
        draw.text((x, y + line_number * 21), line, font=font, fill=fill)
        pending = pending[fit:]


def render_image(sample_id: str, rgb: np.ndarray, official_gt: np.ndarray,
                 v4_gt_256: np.ndarray, predicted_256: np.ndarray,
                 proposals: list[dict], predictions: list[dict], words: list[dict],
                 matches: dict[int, int], assessments: list[dict]) -> bytes:
    rows = (len(proposals) + COLUMNS - 1) // COLUMNS
    canvas = Image.new("RGB", (CANVAS_W, 445 + max(rows, 1) * CARD_H), "white")
    draw = ImageDraw.Draw(canvas)
    title = _font(23)
    body = _font(17)
    small = _font(15)
    draw.text((20, 12), f"{sample_id} | residual U-Net | native 256 | FindContours pad 0", font=title, fill="black")
    draw.text((20, 43), f"GT words {len(words)} | crop {len(proposals)} | spatial exact {len(matches)}  "
              "(green=correct, red=incorrect, orange=text correct but box insufficient)", font=body, fill="black")
    image = Image.fromarray(rgb, mode="RGB")
    placement = _fit(canvas, image, (20, 90, 350, 385))
    for index, proposal in enumerate(proposals):
        px, py, pw, ph = placement
        x1, y1, x2, y2 = proposal["bbox"]
        color = (25, 150, 65) if index in matches else (220, 45, 35)
        draw.rectangle((px + x1 * pw / image.width, py + y1 * ph / image.height,
                        px + x2 * pw / image.width, py + y2 * ph / image.height), outline=color, width=2)
    draw.text((20, 392), "RGB + pad-0 crop boxes", font=small, fill="black")
    panels = [
        ("Official pixel GT", Image.fromarray(np.asarray(official_gt, dtype=np.uint8), mode="L")),
        ("GT v4 word target @256", Image.fromarray(v4_gt_256 * 255, mode="L")),
        ("Predicted mask @256", Image.fromarray(predicted_256 * 255, mode="L")),
    ]
    for column, (label, panel) in enumerate(panels, 1):
        left = 20 + column * CARD_W
        _fit(canvas, panel, (left, 90, left + 315, 385), mask=True)
        draw.text((left, 392), label, font=small, fill="black")
    draw.line((20, 430, CANVAS_W - 20, 430), fill=(140, 140, 140), width=2)
    for index, (proposal, prediction, assessment) in enumerate(zip(proposals, predictions, assessments)):
        left = 20 + (index % COLUMNS) * CARD_W
        top = 445 + (index // COLUMNS) * CARD_H
        status = assessment["status"]
        color = ((25, 150, 65) if status == "CORRECT" else
                 (220, 140, 20) if status == "TEXT_OK_BOX_BAD" else (220, 45, 35))
        draw.rectangle((left, top, left + CARD_W - 12, top + CARD_H - 8), outline=color, width=4)
        x1, y1, x2, y2 = proposal["bbox"]
        _fit(canvas, image.crop((x1, y1, x2, y2)), (left + 7, top + 7, left + CARD_W - 20, top + 95))
        draw.text((left + 8, top + 99), f"#{proposal['rank']:02d} {status} IoU={assessment['best_gt_iou']:.2f}",
                  font=small, fill=color)
        _wrapped(draw, f"PARSeq: {prediction['text']}", left + 8, top + 120, CARD_W - 30,
                 font=small, fill=(0, 0, 0))
        _wrapped(draw, f"GT: {assessment['best_gt_text'] or '<none>'}", left + 8, top + 156,
                 CARD_W - 30, font=small, fill=(50, 50, 50), max_lines=1)
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--frozen-v4-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.run_root / "results.json"
    if file_sha256(report_path) != EXPECTED_RESULT_SHA256:
        raise ValueError("Frozen high-resolution TEST result SHA256 mismatch")
    if file_sha256(args.frozen_v4_root / "results.json") != EXPECTED_FROZEN_V4_SHA256:
        raise ValueError("Frozen GT-v4 TEST result SHA256 mismatch")
    if file_sha256(args.test_manifest) != EXPECTED_TEST_MANIFEST_SHA256:
        raise ValueError("Official Total-Text TEST manifest SHA256 mismatch")
    if args.output_zip.exists():
        raise FileExistsError(f"Refusing to overwrite existing ZIP: {args.output_zip}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    correct_by_gt = set(report["variants"][VARIANT]["correct_word_ids_by_padding"][PADDING])
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "test" or len(manifest.samples) != 300:
        raise ValueError("Expected official Total-Text TEST-300")
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix="totaltext-review-", suffix=".zip.tmp", dir=args.output_zip.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    records = []
    totals = Counter()
    try:
        with ZipFile(temporary, "w", compression=ZIP_STORED, allowZip64=True) as bundle:
            for position, sample in enumerate(manifest.samples, 1):
                key = sample.sample_id.replace(":", "__")
                shard_path = args.run_root / VARIANT / "samples" / f"{key}.json"
                shard = json.loads(shard_path.read_text(encoding="utf-8"))
                if shard.get("sample_id") != sample.sample_id or shard.get("image_sha256") != sample.image_sha256:
                    raise ValueError(f"Sample provenance mismatch: {sample.sample_id}")
                rgb, target, validity, separator, owners, frozen = _regenerate(args, sample)
                gt_v4 = resize_v4_target(target, validity, separator, owners, 256)
                with Image.open(args.data_root / sample.mask_path) as opened:
                    official_gt = np.asarray(opened.convert("L")).copy()
                mask_path = args.run_root / VARIANT / "masks" / f"{key}.png"
                if file_sha256(mask_path) != shard["native_mask_file_sha256"]:
                    raise ValueError(f"Predicted native mask hash mismatch: {sample.sample_id}")
                with Image.open(mask_path) as opened:
                    predicted = (np.asarray(opened.convert("L")) > 0).astype(np.uint8)
                if predicted.shape != (256, 256):
                    raise ValueError(f"Predicted mask shape mismatch: {sample.sample_id}")
                proposals = shard["paddings"][PADDING]["proposals"]
                words = shard["ground_truth"]
                predictions = []
                for proposal in proposals:
                    x1, y1, x2, y2 = proposal["bbox"]
                    if crop_array_hash(rgb[y1:y2, x1:x2]) != proposal["crop_rgb_sha256"]:
                        raise ValueError(f"Frozen crop RGB mismatch: {sample.sample_id}")
                    ocr_path = args.run_root / "ocr" / f"{proposal['crop_rgb_sha256']}.json"
                    ocr = json.loads(ocr_path.read_text(encoding="utf-8"))
                    if ocr.get("crop_rgb_sha256") != proposal["crop_rgb_sha256"]:
                        raise ValueError(f"Frozen PARSeq cache mismatch: {ocr_path}")
                    predictions.append({"bbox": proposal["bbox"], "text": ocr["raw_text"]})
                matches = matched_crop_to_word(predictions, words)
                if set(matches.values()) != maximum_matching(
                    len(predictions), len(words),
                    [(i, j) for i, pred in enumerate(predictions) for j, word in enumerate(words)
                     if box_iou(pred["bbox"], word["bbox"]) >= 0.5
                     and pred["text"].casefold() == word["text"].casefold()]):
                    raise RuntimeError("Crop-level matching differs from frozen scoring")
                expected = {int(value.rsplit(":", 1)[1]) for value in correct_by_gt
                            if value.rsplit(":", 1)[0] == sample.sample_id}
                if set(matches.values()) != expected:
                    raise ValueError(f"GT word identities differ from frozen report: {sample.sample_id}")
                assessments = [classify_crop(prediction, words, matched_word_index=matches.get(index))
                               for index, prediction in enumerate(predictions)]
                png = render_image(sample.sample_id, rgb, official_gt, gt_v4, predicted,
                                   proposals, predictions, words, matches, assessments)
                name = f"images/{position:03d}_{key}.png"
                bundle.writestr(name, png)
                entries = [{"rank": proposal["rank"], "bbox": proposal["bbox"],
                            "parseq_text": prediction["text"], **assessment}
                           for proposal, prediction, assessment in zip(proposals, predictions, assessments)]
                records.append({"sample_id": sample.sample_id, "image": name,
                                "gt_words": len(words), "crops": entries})
                totals["images"] += 1
                totals["gt_words"] += len(words)
                totals["crops"] += len(entries)
                totals["correct_crops"] += len(matches)
                totals["incorrect_crops"] += len(entries) - len(matches)
                if position % 10 == 0 or position == len(manifest.samples):
                    print(f"[review ZIP] {position}/{len(manifest.samples)} images", flush=True)
            if (totals["images"], totals["gt_words"], totals["crops"], totals["correct_crops"]) != (300, 2204, 2432, 739):
                raise ValueError(f"Visual audit totals differ from frozen report: {dict(totals)}")
            bundle.writestr("manifest.json", json.dumps({
                "schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
                "source_results_sha256": EXPECTED_RESULT_SHA256,
                "variant": VARIANT, "padding_native_pixels": 0,
                "correctness": "one-to-one GT bbox IoU>=0.5 AND exact Unicode casefold, punctuation retained",
                "mask_panels": ["official Total-Text pixel GT", "qualified v4 word-region target native256", "residual model prediction native256"],
                "totals": dict(totals), "samples": records,
            }, ensure_ascii=False, indent=2) + "\n")
            bundle.writestr("README.txt", "One PNG per TEST image. Green crop = spatial exact word; red = not correct; orange = text matches a GT word but box is insufficient. All crops use native padding 0. Text comparison is Unicode casefold with punctuation retained. The official pixel GT and derived v4 native-256 word target are distinct panels. Full, untruncated PARSeq and GT text is in manifest.json.\n")
        temporary.replace(args.output_zip)
    finally:
        if temporary.exists():
            temporary.unlink()
    print("ZIP", args.output_zip, "SHA256", file_sha256(args.output_zip), flush=True)
    print("TOTALS", json.dumps(dict(totals), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
