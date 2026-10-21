"""CPU-only audit and composite examples of frozen Total-Text letterbox failures.

Diagnostic categories describe observed geometry, not proven causal attribution.
GT is used only post-hoc. No inference, crop change, or scientific tuning.
"""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZIP_STORED, ZipFile

import cv2
import numpy as np
from PIL import Image, ImageDraw

from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.evaluate_totaltext_highres_contour_padding import crop_array_hash
from scripts.package_totaltext_highres_residual_pad0_review import _fit, _font, _wrapped
from scripts.rescore_totaltext_highres_padding_symbols import normalize_word
from scripts.sweep_totaltext_v4_gt_mask_resolution import _regenerate, resize_v4_target
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.totaltext_image_frequency_augment import letterbox_geometry, remap_letterbox_native_box


CATEGORIES = {
    "PUNCTUATION_OR_SYMBOL_ONLY": "Correct after removing punctuation/symbols; not an alphanumeric transcription error.",
    "NO_PRED_FOREGROUND_IN_WORD_BOX": "No predicted foreground inside the native GT word box: no region detected there.",
    "MULTIWORD_FUSION": "The best crop covers at least half of two GT word boxes and is unsuitable for this word. This does not prove mask fusion.",
    "FRAGMENTATION_OR_TRUNCATION": "Incomplete crop: separate fragments yield a valid union, or the crop covers less than 80% of the GT word box.",
    "BOX_SHAPE_OR_EXTENT_MISMATCH": "Best box IoU <.50; evidence is insufficient to attribute the failure to merging or fragmentation.",
    "OCR_MISMATCH_WITH_ACCEPTABLE_BOX": "IoU>=.50 and coverage>=.80, but text differs. Legibility, orientation and completeness still require visual inspection.",
    "ONE_TO_ONE_MATCHING_CONFLICT": "A valid prediction exists but is assigned to another GT by one-to-one matching.",
}


def box_coverage(box: list[int], word: list[int]) -> float:
    overlap = max(0, min(box[2], word[2]) - max(box[0], word[0])) * max(0, min(box[3], word[3]) - max(box[1], word[1]))
    return overlap / ((word[2] - word[0]) * (word[3] - word[1]))


def native_word_box(box: list[int], image_wh: tuple[int, int]) -> list[int]:
    w, h = image_wh
    side = max(w, h)
    left, top = (side - w) // 2, (side - h) // 2
    return [max(0, (box[0] + left) * 256 // side), max(0, (box[1] + top) * 256 // side),
            min(256, math.ceil((box[2] + left) * 256 / side)), min(256, math.ceil((box[3] + top) * 256 / side))]


def classify(row: dict) -> str:
    if row["normalized_correct"]:
        return "PUNCTUATION_OR_SYMBOL_ONLY"
    if row["valid_exact_candidate_exists"]:
        return "ONE_TO_ONE_MATCHING_CONFLICT"
    if not row["predicted_pixels_in_word_box_native"]:
        return "NO_PRED_FOREGROUND_IN_WORD_BOX"
    if row["significantly_intersected_gt_words"] >= 2 and row["best_iou"] < 0.5:
        return "MULTIWORD_FUSION"
    if (row["word_box_coverage"] < 0.8 and row["best_iou"] >= 0.5) or (
        row["component_count_in_word_box"] >= 2 and row["component_union_iou"] >= 0.5 and row["best_iou"] < 0.5
    ):
        return "FRAGMENTATION_OR_TRUNCATION"
    if row["best_iou"] < 0.5:
        return "BOX_SHAPE_OR_EXTENT_MISMATCH"
    return "OCR_MISMATCH_WITH_ACCEPTABLE_BOX"


def select_examples(rows: list[dict], per_category: int) -> list[dict]:
    selected = []
    for category in CATEGORIES:
        candidates = sorted((r for r in rows if r["category"] == category),
                            key=lambda r: (len(r["gt_text"]) < 3,
                                           -r["significantly_intersected_gt_words"] if category == "MULTIWORD_FUSION" else 0,
                                           -r["component_count_in_word_box"] if category == "FRAGMENTATION_OR_TRUNCATION" else 0,
                                           -r["best_iou"] if category in {"OCR_MISMATCH_WITH_ACCEPTABLE_BOX", "PUNCTUATION_OR_SYMBOL_ONLY"} else r["best_iou"],
                                           r["sample_id"], r["word_index"] ))
        chosen, seen = [], set()
        for r in candidates:
            if r["sample_id"] not in seen:
                chosen.append(r); seen.add(r["sample_id"])
            if len(chosen) == per_category:
                break
        if len(chosen) < min(per_category, len(candidates)):
            chosen += [r for r in candidates if r not in chosen][:per_category - len(chosen)]
        selected.extend(chosen)
    return selected


def filter_and_reselect(report: dict, per_category: int) -> dict:
    """Offline view of an existing audit; ignore normalization-only failures."""
    if sum(report["category_counts"].values()) != len(report["per_word_failures"]):
        raise ValueError("Source failure population is inconsistent")
    rows = [dict(r, reason=CATEGORIES[r["category"]]) for r in report["per_word_failures"] if not r["normalized_correct"]]
    if any(r["category"] == "PUNCTUATION_OR_SYMBOL_ONLY" for r in rows):
        raise ValueError("Punctuation-only failure survived normalization")
    result = dict(report)
    result.update(failure_definition="casefold + Unicode punctuation/symbol removal; spatial IoU>=.50",
                  failures=len(rows), excluded_punctuation_symbol_only=len(report["per_word_failures"])-len(rows),
                  per_word_failures=rows, category_counts=dict(Counter(r["category"] for r in rows)),
                  categories={k:v for k,v in CATEGORIES.items() if k != "PUNCTUATION_OR_SYMBOL_ONLY"},
                  examples_per_category_requested=per_category,
                  selected_examples=select_examples(rows, per_category))
    return result


def lexical_audit(report: dict, failures: list[dict]) -> dict:
    """Same-image exact multiset overlap; no box constraint and no GT reuse."""
    normalizers = {"case_sensitive": str, "casefold": str.casefold, "no_symbols": normalize_word}
    totals = {name: 0 for name in normalizers}
    image_rows = []
    by_id = {im["sample_id"]: im for im in report["per_image"]}
    for sid, im in by_id.items():
        row = {"sample_id": sid, "gt_words": len(im["ground_truth"])}
        for name, norm in normalizers.items():
            predicted = Counter(value for p in im["predictions"] if (value := norm(p["text"])))
            gt = Counter(value for w in im["ground_truth"] if (value := norm(w["text"])))
            row[name] = sum((predicted & gt).values())
            totals[name] += row[name]
        image_rows.append(row)
    available = Counter()
    for row in failures:
        text = normalize_word(row["gt_text"])
        if text and any(normalize_word(p["text"]) == text for p in by_id[row["sample_id"]]["predictions"]):
            available[row["category"]] += 1
    gt_count = report["gt_words"]
    return {"classification": "PRACTICAL_APPROXIMATION", "reproduction_mode": "approximate",
            "protocol": "Exact whole-crop transcription multiset overlap WITHIN EACH IMAGE; no position/order; each GT and prediction consumed at most once; empty normalized strings do not match",
            "gt_words": gt_count, "images": report["images"], "lexical_counts": totals,
            "rates_all_gt": {name: value/gt_count for name,value in totals.items()},
            "spatial_counts": {"casefold":report["aggregate"]["exact_casefold_iou_ge_50"], "no_symbols":report["aggregate"]["exact_no_symbols_iou_ge_50"]},
            "gain_without_position": {name:totals[name]-report["aggregate"][key] for name,key in (("casefold","exact_casefold_iou_ge_50"),("no_symbols","exact_no_symbols_iou_ge_50"))},
            "normalized_failure_text_present_elsewhere_by_category":dict(available),
            "availability_warning":"Presence elsewhere is NOT a one-to-one recovery count; duplicate GT words may reuse already-counted predictions in this presence diagnostic.",
            "per_image":image_rows}


def write_new_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    if path.exists():
        raise FileExistsError(path)
    os.replace(temporary, path)


def analyze(args) -> dict:
    root = args.run_root.resolve()
    result_path = root / "results.json"
    if file_sha256(result_path) != args.expected_result_sha256:
        raise ValueError("Frozen result hash mismatch")
    report = json.loads(result_path.read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    protocol_hash = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (report["images"] != 300 or report["gt_words"] != 2204
        or report["checkpoint"]["sha256"] != protocol["checkpoint_sha256"]
        or protocol["segmentation"] != "letterbox 512 input; sigmoid>=0.5 native256; batch1"
        or protocol["localization"] != "native RETR_EXTERNAL contours; area OFF; top10 OFF; pad0"):
        raise ValueError("Unexpected frozen letterbox .50 protocol")
    if file_sha256(args.test_manifest) != protocol["test_manifest_sha256"]:
        raise ValueError("Test manifest mismatch")
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    by_id = {s.sample_id: s for s in manifest.samples}
    if len(by_id) != 300 or set(by_id) != {r["sample_id"] for r in report["per_image"]}:
        raise ValueError("Incomplete TEST-300 population")
    rows, category_counts = [], Counter()
    matched_totals = Counter()
    for position, image_row in enumerate(report["per_image"], 1):
        sid = image_row["sample_id"]; sample = by_id[sid]; key = sid.replace(":", "__")
        shard = json.loads((root / "samples" / f"{key}.json").read_text())
        image_path = args.data_root / sample.image_path
        mask_path = root / "masks" / f"{key}.png"
        if (shard["protocol_sha256"] != protocol_hash or shard["checkpoint_sha256"] != protocol["checkpoint_sha256"]
            or file_sha256(image_path) != sample.image_sha256 or shard["image_sha256"] != sample.image_sha256
            or file_sha256(mask_path) != shard["native_mask_file_sha256"]):
            raise ValueError(f"Frozen sample provenance mismatch: {sid}")
        native = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.uint8)
        if native.shape != (256, 256) or sha256(native.tobytes()).hexdigest() != shard["native_mask_array_sha256"]:
            raise ValueError(f"Frozen native mask mismatch: {sid}")
        with Image.open(image_path) as image:
            image_wh = image.size
        count, labels, stats, _ = cv2.connectedComponentsWithStats(native, connectivity=8)
        words, predictions = image_row["ground_truth"], image_row["predictions"]
        if words != shard["ground_truth"] or len(predictions) != len(shard["proposals"]):
            raise ValueError(f"Frozen words/proposals mismatch: {sid}")
        for p, frozen in zip(predictions, shard["proposals"]):
            ocr = json.loads((root / "ocr" / f"{p['crop_rgb_sha256']}.json").read_text())
            if (ocr["crop_rgb_sha256"] != p["crop_rgb_sha256"] or ocr["raw_text"] != p["text"]
                or ocr["parseq_checkpoint_sha256"] != protocol["parseq_checkpoint_sha256"]
                or ocr["parseq_source_archive_sha256"] != protocol["parseq_runtime_archive_sha256"]
                or p["bbox"] != frozen["bbox"]):
                raise ValueError(f"Frozen OCR mismatch: {sid}")
        assignments = {}
        for name, normalizer in (("exact_casefold_iou_ge_50", str.casefold), ("exact_no_symbols_iou_ge_50", normalize_word)):
            edges = [(pi, wi) for pi, p in enumerate(predictions) for wi, word in enumerate(words)
                     if box_iou(p["bbox"], word["bbox"]) >= 0.5 and normalizer(p["text"])
                     and normalizer(p["text"]) == normalizer(word["text"])]
            assignments[name] = maximum_matching(len(predictions), len(words), edges)
            if assignments[name] != set(image_row["metrics"]["matched_gt_indices"][name]):
                raise ValueError(f"Frozen recognition metric drift: {sid}, {name}")
            matched_totals[name] += len(assignments[name])
        for wi, word in enumerate(words):
            if wi in assignments["exact_casefold_iou_ge_50"]:
                continue
            best = max(predictions, key=lambda p: box_iou(p["bbox"], word["bbox"]), default=None)
            if best is not None and box_iou(best["bbox"], word["bbox"]) == 0:
                # Do not display an unrelated crop as this missing word's OCR.
                best = None
            # For normalization-only failures show the actual normalized-correct crop.
            if wi in assignments["exact_no_symbols_iou_ge_50"]:
                eligible = [p for p in predictions if box_iou(p["bbox"], word["bbox"]) >= .5
                            and normalize_word(p["text"]) == normalize_word(word["text"])]
                best = max(eligible, key=lambda p: box_iou(p["bbox"], word["bbox"]))
            x1, y1, x2, y2 = native_word_box(word["bbox"], image_wh)
            pixels = labels[y1:y2, x1:x2]
            ids = [int(x) for x in np.unique(pixels) if x > 0]
            component_boxes = []
            for cid in ids:
                x, y, w, h, _ = (int(v) for v in stats[cid])
                component_boxes.append(remap_letterbox_native_box([x, y, x+w, y+h], image_wh))
            union = ([min(b[0] for b in component_boxes), min(b[1] for b in component_boxes),
                      max(b[2] for b in component_boxes), max(b[3] for b in component_boxes)] if ids else None)
            significant = [j for j, other in enumerate(words) if best is not None and box_coverage(best["bbox"], other["bbox"]) >= .5]
            row = {"sample_id": sid, "word_index": wi, "gt_text": word["text"], "gt_bbox": word["bbox"],
                   "crop_bbox": best["bbox"] if best else None, "parseq_text": best["text"] if best else "",
                   "crop_rgb_sha256": best["crop_rgb_sha256"] if best else None,
                   "best_iou": box_iou(best["bbox"], word["bbox"]) if best else 0,
                   "word_box_coverage": box_coverage(best["bbox"], word["bbox"]) if best else 0,
                   "significantly_intersected_gt_words": len(significant), "intersected_gt_indices": significant,
                   "predicted_pixels_in_word_box_native": int(np.count_nonzero(pixels)),
                   "component_count_in_word_box": len(ids), "component_ids": ids,
                   "component_union_iou": box_iou(union, word["bbox"]) if union else 0,
                   "normalized_correct": wi in assignments["exact_no_symbols_iou_ge_50"],
                   "valid_exact_candidate_exists": any(box_iou(p["bbox"], word["bbox"]) >= .5 and p["text"].casefold() == word["text"].casefold() for p in predictions)}
            row["category"] = classify(row); row["reason"] = CATEGORIES[row["category"]]
            rows.append(row); category_counts[row["category"]] += 1
        if position % 50 == 0:
            print(f"[failure audit] {position}/300", flush=True)
    if dict(matched_totals) != {k: report["aggregate"][k] for k in matched_totals} or len(rows) != 2204 - 924:
        raise RuntimeError("Aggregate recognition control mismatch")
    examples = select_examples(rows, args.examples_per_category)
    result = {"schema_version": 1, "reproduction_mode": "approximate", "claim_label": "approximate reproduction",
              "classification": "PRACTICAL_APPROXIMATION", "result_class": "post-hoc TEST failure diagnostic; not Table 5",
              "images": 300, "gt_words": 2204, "exact_casefold": matched_totals["exact_casefold_iou_ge_50"],
              "exact_no_symbols": matched_totals["exact_no_symbols_iou_ge_50"], "failures": len(rows),
              "category_counts": dict(category_counts), "categories": CATEGORIES,
              "inference": "NONE: frozen segmentation/PARSeq cache only; CPU audit",
              "provenance": {"run_root": str(root), "results_sha256": file_sha256(result_path),
                             "protocol_sha256": file_sha256(root / "protocol.json"),
                             "test_manifest_sha256": file_sha256(args.test_manifest),
                             "checkpoint_sha256": protocol["checkpoint_sha256"],
                             "script_sha256": file_sha256(Path(__file__))},
              "diagnostic_policy": {"classification": "ASSUMPTION", "gt_use": "DIAGNOSTIC_ONLY post-hoc",
                  "word_region": "GT word bbox; CC incidence is box-based, not glyph identity",
                  "fusion_significance": "crop covers >=0.50 of each of >=2 GT word bboxes; this does not prove the native mask fuses words",
                  "crop_completeness_proxy": "word-bbox coverage >=0.80; not proven glyph completeness",
                  "ordering": list(CATEGORIES), "causality": "Observed evidence only; acceptable IoU does not prove OCR-alone failure",
                  "example_selection": "deterministic evidence ordering, prefer >=3 characters and distinct images; all failures remain in JSON"},
              "examples_per_category_requested": args.examples_per_category, "selected_examples": examples, "per_word_failures": rows}
    write_new_json(args.analysis_output, result)
    print(json.dumps({"counts": dict(category_counts), "examples": len(examples)}, sort_keys=True), flush=True)
    print("JSON", args.analysis_output, "SHA256", file_sha256(args.analysis_output), flush=True)
    return result


def draw_box(draw, box, placement, source_wh, color):
    x, y, w, h = placement; sw, sh = source_wh
    draw.rectangle((x + box[0]*w/sw, y + box[1]*h/sh, x + box[2]*w/sw, y + box[3]*h/sh), outline=color, width=3)


def fit_crop(canvas, source, box):
    """Enlarge small crops for display only; frozen RGB pixels are unchanged."""
    left, top, right, bottom = box
    scale = min((right-left)/source.width, (bottom-top)/source.height)
    display = source.resize((max(1, round(source.width*scale)), max(1, round(source.height*scale))),
                            Image.Resampling.LANCZOS)
    x, y = left+(right-left-display.width)//2, top+(bottom-top-display.height)//2
    canvas.paste(display, (x, y))
    return x, y, display.width, display.height


def render(row, rgb, official, target, predicted) -> bytes:
    canvas = Image.new("RGB", (1600, 1100), "white"); draw = ImageDraw.Draw(canvas)
    image = Image.fromarray(rgb)
    draw.text((20, 12), f"{row['sample_id']} | word #{row['word_index']} | {row['category']}", font=_font(23), fill="black")
    _wrapped(draw, CATEGORIES[row["category"]], 20, 47, 1560, font=_font(18), fill=(0,0,0), max_lines=3)
    panels = [("Original RGB image", image, False), ("Original GT (glyph/pixel-level)", official.convert("RGB"), True),
              ("GT V4 @256, letterboxed", Image.fromarray(target*255).convert("RGB"), True),
              ("Prediction @256, threshold .50", Image.fromarray(predicted*255).convert("RGB"), True)]
    for col, (label, panel, mask) in enumerate(panels):
        x = 20 + col*395; draw.text((x, 121), label, font=_font(17), fill="black")
        placement = _fit(canvas, panel, (x, 150, x+370, 520), mask=mask)
        is_native = col >= 2
        gtbox = native_word_box(row["gt_bbox"], image.size) if is_native else row["gt_bbox"]
        draw_box(draw, gtbox, placement, panel.size, (220,30,30))
        if row["crop_bbox"] is not None:
            cropbox = native_word_box(row["crop_bbox"], image.size) if is_native else row["crop_bbox"]
            draw_box(draw, cropbox, placement, panel.size, (30,80,220))
    draw.text((20, 540), "Red: GT word box | Blue: actual PARSeq crop | Native masks are NOT backprojected", font=_font(18), fill="black")
    draw.text((20, 580), "Actual RGB crop passed to PARSeq (padding 0)", font=_font(21), fill="black")
    if row["crop_bbox"]:
        fit_crop(canvas, image.crop(tuple(row["crop_bbox"])), (20, 620, 770, 865))
    else:
        draw.text((30, 715), "NO CROP OVERLAPS THIS GT WORD", font=_font(21), fill=(180,25,25))
    draw.text((815, 580), "RGB inside GT box (reference only; NO new OCR)", font=_font(18), fill="black")
    fit_crop(canvas, image.crop(tuple(row["gt_bbox"])), (815, 620, 1560, 865))
    _wrapped(draw, "GT WORD: " + row["gt_text"], 20, 890, 1540, font=_font(24), fill=(0,0,0), max_lines=2)
    text = row["parseq_text"] or ("<empty>" if row["crop_bbox"] else "<no associated crop>")
    _wrapped(draw, "PARSeq: " + text, 20, 945, 1540, font=_font(24), fill=(180,25,25), max_lines=2)
    draw.text((20, 1010), f"IoU={row['best_iou']:.3f} | GT box coverage={row['word_box_coverage']:.3f} | raw CC={row['component_count_in_word_box']} | intersected GT={row['significantly_intersected_gt_words']}", font=_font(18), fill="black")
    other_matches = row.get("same_text_other_crop_ranks", [])
    availability = "yes, crop rank(s) " + ", ".join(str(x) for x in other_matches) if other_matches else "no"
    draw.text((20, 1043), "GT text in another OCR crop of this image (ignoring symbols): " + availability, font=_font(17), fill="black")
    draw.text((20, 1072), "Diagnostic evidence, not proof of an exclusive cause. No new inference or tuning.", font=_font(16), fill="black")
    output = BytesIO(); canvas.save(output, format="PNG", optimize=True)
    return output.getvalue()


def package(args) -> None:
    report = json.loads(args.analysis_output.read_text())
    if report["provenance"]["results_sha256"] != file_sha256(args.run_root / "results.json"):
        raise ValueError("Source report changed after analysis")
    if file_sha256(args.frozen_v4_root / "results.json") != "272613b094777b9836a1ffde2fd9265fcec17c84e7ccb60661f9a87ac0e29989":
        raise ValueError("Qualified V4 GT source changed")
    if args.output_zip.exists():
        raise FileExistsError(args.output_zip)
    manifest = load_segmentation_manifest(args.test_manifest, require_checksums=True)
    by_id = {s.sample_id:s for s in manifest.samples}
    source = json.loads((args.run_root / "results.json").read_text())
    predictions_by_id = {im["sample_id"]:im["predictions"] for im in source["per_image"]}
    args.output_zip.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output_zip.with_suffix(".zip.partial")
    files = []; grouped = {}
    for row in report["selected_examples"]:
        grouped.setdefault(row["sample_id"], []).append(row)
    helper = SimpleNamespace(data_root=args.data_root, frozen_v4_root=args.frozen_v4_root)
    with partial.open("xb") as handle, ZipFile(handle, "w", compression=ZIP_STORED) as archive:
        for sid, examples in grouped.items():
            rgb, target, validity, sep, owners, frozen = _regenerate(helper, by_id[sid])
            frozen_gt_path = args.frozen_v4_root / frozen["gt_native_mask"]["file"]
            if file_sha256(frozen_gt_path) != frozen["gt_native_mask"]["sha256"]:
                raise ValueError("Qualified V4 target mask cache changed")
            frozen_native = (np.asarray(Image.open(frozen_gt_path).convert("L")) > 0).astype(np.uint8)
            if not np.array_equal(resize_v4_target(target, validity, sep, owners, 128), frozen_native):
                raise ValueError("Regenerated V4 target does not reproduce qualified native128")
            # Replay the qualified letterbox target transform, not a stretched GT.
            padded = [letterbox_geometry(im, fill=0) for im in (target, validity, sep, owners)]
            target256 = resize_v4_target(*padded, 256)
            official = Image.open(args.data_root / by_id[sid].mask_path).convert("L")
            predpath = args.run_root / "masks" / f"{sid.replace(':','__')}.png"
            shard = json.loads((args.run_root / "samples" / f"{sid.replace(':','__')}.json").read_text())
            if file_sha256(predpath) != shard["native_mask_file_sha256"]:
                raise ValueError("Native prediction changed after analysis")
            pred = (np.asarray(Image.open(predpath).convert("L")) > 0).astype(np.uint8)
            for row in examples:
                row = dict(row, reason=CATEGORIES[row["category"]])
                text = normalize_word(row["gt_text"])
                row["same_text_other_crop_ranks"] = [p["rank"] for p in predictions_by_id[sid]
                    if text and normalize_word(p["text"]) == text and p["crop_rgb_sha256"] != row["crop_rgb_sha256"]]
                if row["crop_bbox"]:
                    x1,y1,x2,y2=row["crop_bbox"]
                    if crop_array_hash(rgb[y1:y2,x1:x2]) != row["crop_rgb_sha256"]:
                        raise ValueError("Frozen crop RGB mismatch")
                name = f"{row['category']}/{sid.replace(':','__')}-word{row['word_index']:03d}.png"
                payload = render(row, rgb, official, target256, pred); archive.writestr(name, payload)
                files.append({"file":name, "sha256":sha256(payload).hexdigest(), **row})
                print(f"[failure panels] {len(files)}/{len(report['selected_examples'])} {name}", flush=True)
        archive.writestr("analysis.json", args.analysis_output.read_bytes())
        archive.writestr("manifest.json", json.dumps({"classification":"PRACTICAL_APPROXIMATION", "analysis_sha256":file_sha256(args.analysis_output), "packager_sha256":file_sha256(Path(__file__)), "gt_v4_construction_sha256":file_sha256(Path(__file__).resolve().parents[1] / "segmentation/totaltext_curved_box_v4.py"), "qualified_gt128_reproduced":True, "panels":files}, indent=2, ensure_ascii=False))
        archive.writestr("README.txt", "All panel labels/explanations are in English. OCR and GT strings are never translated. One PNG per example; all requested panels in ONE image. Red=GT box, blue=frozen pad0 crop. GT V4/pred native256 are letterboxed. Original GT is glyph-level. Small crops are enlarged for display only. Categories are post-hoc evidence, not proven causes. Each panel indicates whether the GT transcription exists in another crop of the same image. Failure definition and complete counts: analysis.json. No new inference or tuning.\n")
    with ZipFile(partial) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(files)+3:
            raise RuntimeError("ZIP validation failed")
    os.replace(partial, args.output_zip)
    print("ZIP", args.output_zip, "SHA256", file_sha256(args.output_zip), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("analyze", "reselect-normalized", "package"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--analysis-output", type=Path, required=True)
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--source-analysis", type=Path)
    parser.add_argument("--examples-per-category", type=int, default=3)
    parser.add_argument("--frozen-v4-root", type=Path)
    parser.add_argument("--output-zip", type=Path)
    args = parser.parse_args()
    if args.examples_per_category < 1:
        parser.error("examples-per-category must be positive")
    if args.action == "analyze":
        if not args.expected_result_sha256:
            parser.error("analyze requires expected-result-sha256")
        analyze(args)
    elif args.action == "reselect-normalized":
        if not args.source_analysis:
            parser.error("reselect-normalized requires source-analysis")
        source = json.loads(args.source_analysis.read_text())
        result = filter_and_reselect(source, args.examples_per_category)
        result["source_analysis_sha256"] = file_sha256(args.source_analysis)
        result["reselection_script_sha256"] = file_sha256(Path(__file__))
        frozen_results = json.loads((args.run_root / "results.json").read_text())
        if file_sha256(args.run_root / "results.json") != result["provenance"]["results_sha256"]:
            raise ValueError("Frozen report changed before lexical audit")
        result["lexical_audit"] = lexical_audit(frozen_results, result["per_word_failures"])
        if len(result["selected_examples"]) != len(result["category_counts"])*args.examples_per_category:
            raise ValueError("Not enough examples for every remaining category")
        write_new_json(args.analysis_output, result)
        print("JSON", args.analysis_output, "SHA256", file_sha256(args.analysis_output), flush=True)
    else:
        if not args.output_zip or not args.frozen_v4_root:
            parser.error("package requires output-zip and frozen-v4-root")
        package(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
