"""Package strict or punctuation-normalized failures of GT-v4 256 OCR.

One composite PNG per GT word. All matching is post-hoc and leaves the
qualified recognition report unchanged. No model inference is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unicodedata
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.sweep_totaltext_v4_gt_mask_resolution import _regenerate, _sha_array, resize_v4_target
from segmentation.data import file_sha256, load_segmentation_manifest


EXPECTED_RESULTS_SHA256 = "4fe677ec8f13c5c9a5a5671d8240693ae00809535b5cba316317fa90fdfc09b7"
EXPECTED_TEST_MANIFEST_SHA256 = "da2f87e7cae765f426af39c9641535138d6657d2e9332e6fdf9e80f2ba6b69bd"


def without_punctuation_symbols(value: str) -> str:
    """Casefold and remove Unicode P/S characters from both comparison sides."""
    return "".join(char for char in value.casefold()
                   if unicodedata.category(char)[0] not in "PS")


def matched_gt_indices(predictions: list[dict], words: list[dict], mode: str) -> set[int]:
    if mode not in ("geometry", "strict", "casefold", "no_punctuation_symbols"):
        raise ValueError("Unknown diagnostic matching mode")
    edges = []
    for prediction_index, prediction in enumerate(predictions):
        for word_index, word in enumerate(words):
            if box_iou(prediction["bbox"], word["bbox"]) < 0.5:
                continue
            left, right = str(prediction["text"]), str(word["text"])
            if mode == "strict" and left != right:
                continue
            if mode == "casefold" and left.casefold() != right.casefold():
                continue
            if mode == "no_punctuation_symbols":
                normalized_right = without_punctuation_symbols(right)
                if not normalized_right or without_punctuation_symbols(left) != normalized_right:
                    continue
            edges.append((prediction_index, word_index))
    return maximum_matching(len(predictions), len(words), edges)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _write_lines(draw: ImageDraw.ImageDraw, label: str, value: str,
                 xy: tuple[int, int], *, width: int = 82) -> None:
    text = f"{label}: {value}"
    for index in range(0, min(len(text), width * 2), width):
        draw.text((xy[0], xy[1] + index // width * 27), text[index:index + width],
                  font=_font(18), fill="black")


def _paste_fit(canvas: Image.Image, source: Image.Image,
               box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    copy = source.copy()
    copy.thumbnail((right - left, bottom - top), Image.Resampling.LANCZOS)
    x = left + (right - left - copy.width) // 2
    y = top + (bottom - top - copy.height) // 2
    canvas.paste(copy, (x, y))
    return x, y, copy.width, copy.height


def _draw_box(draw: ImageDraw.ImageDraw, bbox: list[int], placement: tuple[int, int, int, int],
              source_wh: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y, displayed_w, displayed_h = placement
    source_w, source_h = source_wh
    x1, y1, x2, y2 = bbox
    draw.rectangle((x + x1 * displayed_w / source_w, y + y1 * displayed_h / source_h,
                    x + x2 * displayed_w / source_w, y + y2 * displayed_h / source_h),
                   outline=color, width=3)


def render_composite(row: dict, image: np.ndarray, mask: np.ndarray) -> bytes:
    canvas = Image.new("RGB", (1200, 900), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), f"GT-v4 256 | {row['sample_id']} | word {row['word_id']} | {row['category']}",
              font=_font(23), fill="black")
    _write_lines(draw, "GT", row["gt_text"], (20, 50))
    _write_lines(draw, "PARSeq (selected crop)", row["parseq_text"], (20, 105))
    draw.text((20, 160), f"IoU={row['best_iou']:.3f} | strict={row['strict_correct']} "
              f"casefold={row['casefold_correct']} no-punctuation/symbols={row['normalized_correct']}",
              font=_font(18), fill="black")
    draw.text((20, 188), "Red: GT word box    Blue: 256-mask component crop box", font=_font(18), fill="black")
    original = Image.fromarray(image, mode="RGB")
    placement = _paste_fit(canvas, original, (20, 235, 860, 610))
    _draw_box(draw, row["gt_bbox"], placement, original.size, (220, 20, 20))
    if row["crop_bbox"] is not None:
        _draw_box(draw, row["crop_bbox"], placement, original.size, (20, 50, 220))
    draw.text((875, 232), "GT-v4 binary mask 256", font=_font(18), fill="black")
    native = Image.fromarray(mask * 255, mode="L").convert("RGB")
    mask_placement = _paste_fit(canvas, native, (890, 270, 1170, 550))
    _draw_box(draw, row["gt_bbox"], mask_placement, original.size, (220, 20, 20))
    if row["crop_bbox"] is not None:
        _draw_box(draw, row["crop_bbox"], mask_placement, original.size, (20, 50, 220))
    draw.line((20, 635, 1180, 635), fill=(150, 150, 150), width=2)
    draw.text((20, 647), "RGB inside GT word box", font=_font(18), fill="black")
    draw.text((610, 647), "Exact RGB crop passed to PARSeq", font=_font(18), fill="black")
    x1, y1, x2, y2 = row["gt_bbox"]
    _paste_fit(canvas, original.crop((x1, y1, x2, y2)), (20, 690, 580, 875))
    if row["crop_bbox"] is not None:
        x1, y1, x2, y2 = row["crop_bbox"]
        _paste_fit(canvas, original.crop((x1, y1, x2, y2)), (610, 690, 1170, 875))
    else:
        draw.text((650, 755), "No predicted component", font=_font(22), fill="black")
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--frozen-v4-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--failure-mode", choices=("strict", "no_punctuation_symbols"),
                        default="strict")
    args = parser.parse_args()
    if file_sha256(args.report_root / "results.json") != EXPECTED_RESULTS_SHA256:
        raise ValueError("Frozen no-filter 256 recognition report SHA256 mismatch")
    if file_sha256(args.test_manifest) != EXPECTED_TEST_MANIFEST_SHA256:
        raise ValueError("Official TEST-300 manifest SHA256 mismatch")
    if args.output_zip.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output_zip}")
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    if manifest.dataset != "total_text" or manifest.split_role != "test" or len(manifest.samples) != 300:
        raise ValueError("Expected official Total-Text TEST-300")
    report = json.loads((args.report_root / "results.json").read_text(encoding="utf-8"))
    summary = Counter()
    selected = []
    for sample in manifest.samples:
        key = sample.sample_id.replace(":", "__")
        shard = json.loads((args.report_root / "samples" / f"{key}.json").read_text(encoding="utf-8"))
        ocr = json.loads((args.report_root / "ocr/256" / f"{key}.json").read_text(encoding="utf-8"))
        words = shard["ground_truth"]
        proposals = shard["resolutions"]["256"]["proposals"]
        predictions = ocr["predictions"]
        if (shard["sample_id"] != sample.sample_id or ocr["sample_id"] != sample.sample_id
            or len(proposals) != len(predictions)
            or any((a["rank"], a["bbox"], a["crop_rgb_sha256"])
                   != (b["rank"], b["bbox"], b["crop_rgb_sha256"])
                   for a, b in zip(proposals, predictions))):
            raise ValueError(f"Frozen proposal/OCR provenance mismatch: {sample.sample_id}")
        matches = {mode: matched_gt_indices(predictions, words, mode)
                   for mode in ("geometry", "strict", "casefold", "no_punctuation_symbols")}
        for mode, indices in matches.items():
            summary[f"matched_{mode}"] += len(indices)
        summary["gt_words"] += len(words)
        for word_index, word in enumerate(words):
            if word_index in matches[args.failure_mode]:
                continue
            best = (max(predictions, key=lambda prediction: (box_iou(prediction["bbox"], word["bbox"]),
                                                           -prediction["rank"])) if predictions else None)
            best_iou = box_iou(best["bbox"], word["bbox"]) if best else 0.0
            if word_index in matches["casefold"]:
                category = "CASE_ONLY"
            elif word_index in matches["no_punctuation_symbols"]:
                category = "PUNCTUATION_OR_SYMBOL_ONLY"
            elif word_index not in matches["geometry"]:
                category = "NO_VALID_IOU_0_50_CROP"
            else:
                category = "OCR_FAILURE_AFTER_NORMALIZATION"
            summary[category] += 1
            selected.append({"sample_id": sample.sample_id, "word_id": word_index + 1,
                             "gt_text": word["text"], "gt_bbox": word["bbox"],
                             "crop_bbox": best["bbox"] if best else None,
                             "crop_rank": best["rank"] if best else None,
                             "crop_rgb_sha256": best["crop_rgb_sha256"] if best else None,
                             "parseq_text": best["text"] if best else "",
                             "best_iou": best_iou, "category": category,
                             "strict_correct": False,
                             "casefold_correct": word_index in matches["casefold"],
                             "normalized_correct": word_index in matches["no_punctuation_symbols"]})
    expected = report["summary"]["256"]
    if (summary["gt_words"] != 2204 or summary["matched_geometry"] != expected["geometry"]["iou_ge_0_50_one_to_one"]
        or summary["matched_strict"] != expected["recognition"]["exact_case_sensitive_spatial_one_to_one"]
        or summary["matched_casefold"] != expected["recognition"]["exact_casefold_spatial_one_to_one"]
        or summary["matched_no_punctuation_symbols"] != 2060
        or len(selected) != (334 if args.failure_mode == "strict" else 144)):
        raise ValueError(f"Failure population differs from qualified summary: {dict(summary)}")
    if args.failure_mode == "no_punctuation_symbols" and any(
        row["category"] not in ("NO_VALID_IOU_0_50_CROP", "OCR_FAILURE_AFTER_NORMALIZATION")
        for row in selected
    ):
        raise ValueError("Normalized-failure archive contains a recoverable word")
    selected_by_sample: dict[str, list[dict]] = {}
    for row in selected:
        selected_by_sample.setdefault(row["sample_id"], []).append(row)
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=args.output_zip.name + ".", suffix=".partial",
                                         dir=args.output_zip.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
            files = []
            for sample_index, sample in enumerate(manifest.samples, 1):
                words = selected_by_sample.get(sample.sample_id)
                if not words:
                    continue
                image, target, validity, separator, owners, _ = _regenerate(args, sample)
                mask = resize_v4_target(target, validity, separator, owners, 256)
                key = sample.sample_id.replace(":", "__")
                shard = json.loads((args.report_root / "samples" / f"{key}.json").read_text(encoding="utf-8"))
                if _sha_array(mask) != shard["resolutions"]["256"]["mask_sha256"]:
                    raise ValueError(f"GT-v4 mask256 changed: {sample.sample_id}")
                for row in words:
                    if row["crop_bbox"] is not None:
                        x1, y1, x2, y2 = row["crop_bbox"]
                        if _sha_array(image[y1:y2, x1:x2]) != row["crop_rgb_sha256"]:
                            raise ValueError(f"PARSeq crop changed: {sample.sample_id}/{row['word_id']}")
                    payload = render_composite(row, image, mask)
                    name = f"{key}-word{row['word_id']:03d}.png"
                    bundle.writestr(name, payload)
                    files.append({**row, "file": name, "png_sha256": sha256(payload).hexdigest()})
                if sample_index % 20 == 0 or sample_index == 300:
                    print(f"[resolution256 failures] image {sample_index}/300; composites {len(files)}/{len(selected)}", flush=True)
            archive_manifest = {"schema_version": 1, "result_class": "TEST GT-oracle OCR visual diagnostic",
                                "recognition_report_sha256": EXPECTED_RESULTS_SHA256,
                                "resolution": 256,
                                "failure_mode": args.failure_mode,
                                "failure_definition": (
                                    "not matched by one-to-one spatial IoU>=0.5 and case-sensitive exact text"
                                    if args.failure_mode == "strict" else
                                    "not matched by one-to-one spatial IoU>=0.5 after symmetric Unicode "
                                    "casefold and punctuation/symbol removal"
                                ),
                                "normalizations": {"strict": "none", "casefold": "Unicode casefold only",
                                                   "no_punctuation_symbols": "symmetric Unicode casefold plus removal of P/S codepoints"},
                                "summary": dict(summary), "words": files}
            bundle.writestr("manifest.json", json.dumps(archive_manifest, indent=2, ensure_ascii=False) + "\n")
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None or len(archive.namelist()) != len(selected) + 1:
                raise ValueError("ZIP integrity or entry count failed")
        if args.output_zip.exists():
            raise FileExistsError(f"Refusing to overwrite: {args.output_zip}")
        os.replace(temporary, args.output_zip)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(dict(summary), indent=2, ensure_ascii=False), flush=True)
    print("ZIP", args.output_zip, "SHA256", file_sha256(args.output_zip), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
