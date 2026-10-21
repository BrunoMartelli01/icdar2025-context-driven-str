"""CPU-only post-hoc OCR scoring: casefold plus Unicode P/S removal.

Uses frozen TEST-300 proposal boxes and PARSeq cache. No model inference,
threshold change, crop change, or padding selection is performed.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import unicodedata

from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.evaluate_totaltext_highres_contour_padding import padding_overlap
from scripts.evaluate_totaltext_v1_findcontours_parseq import _atomic_json
from segmentation.data import file_sha256


EXPECTED_RESULTS_SHA256 = "6aebbfa409240c8f1b4dabc67a53eff37bf53fc6c9f1d542a22ec0a5cfda9df7"


def normalize_word(value: str) -> str:
    """Unicode casefold, then remove punctuation and symbols; preserve spaces."""
    return "".join(character for character in value.casefold()
                   if unicodedata.category(character)[0] not in "PS")


def match_word_ids(predictions: list[dict], words: list[dict], *, normalize: bool) -> set[int]:
    edges = []
    for prediction_index, prediction in enumerate(predictions):
        left = normalize_word(prediction["text"]) if normalize else prediction["text"].casefold()
        if not left:
            continue
        for word_index, word in enumerate(words):
            right = normalize_word(word["text"]) if normalize else word["text"].casefold()
            if left == right and box_iou(prediction["bbox"], word["bbox"]) >= 0.5:
                edges.append((prediction_index, word_index))
    return maximum_matching(len(predictions), len(words), edges)


def lexical_multiset_matches(predictions: list[dict], words: list[dict]) -> int:
    predicted = Counter(value for prediction in predictions
                        if (value := normalize_word(prediction["text"])))
    gt = Counter(value for word in words if (value := normalize_word(word["text"])))
    return sum((predicted & gt).values())


def _load_ocr(root: Path, crop_hash: str, cache: dict[str, str]) -> str:
    if crop_hash not in cache:
        path = root / "ocr" / f"{crop_hash}.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("crop_rgb_sha256") != crop_hash or not isinstance(value.get("raw_text"), str):
            raise ValueError(f"OCR cache provenance mismatch: {path}")
        cache[crop_hash] = value["raw_text"]
    return cache[crop_hash]


def rescore(root: Path, report: dict) -> dict:
    variants = tuple(report["variants"])
    if set(variants) != {"highres_baseline", "highres_separator", "highres_separator_residual"}:
        raise ValueError("Expected the three frozen native-256 variants")
    pads = [0, 2, 4, 6, 8, 12]
    if report.get("images") != 300 or report.get("gt_words") != 2204:
        raise ValueError("Expected frozen official Total-Text TEST-300 result")
    totals = {(variant, str(pad)): {"normalized_spatial_exact": 0,
                                    "normalized_lexical_multiset": 0,
                                    "casefold_spatial_control": 0}
              for variant in variants for pad in pads}
    correct = {(variant, str(pad)): set() for variant in variants for pad in pads}
    casefold = {(variant, str(pad)): set() for variant in variants for pad in pads}
    gt_empty = 0
    gt_total = 0
    ocr_cache: dict[str, str] = {}
    baseline_shards = sorted((root / variants[0] / "samples").glob("*.json"))
    if len(baseline_shards) != 300:
        raise ValueError("Missing frozen baseline sample shards")
    for position, baseline_path in enumerate(baseline_shards, 1):
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        sample_id = baseline["sample_id"]
        words = baseline["ground_truth"]
        gt_total += len(words)
        gt_empty += sum(not normalize_word(word["text"]) for word in words)
        for variant in variants:
            path = root / variant / "samples" / baseline_path.name
            shard = baseline if variant == variants[0] else json.loads(path.read_text(encoding="utf-8"))
            if shard.get("sample_id") != sample_id or shard.get("ground_truth") != words:
                raise ValueError(f"GT differs across variants: {sample_id}")
            for pad in pads:
                key = (variant, str(pad))
                predictions = [{"bbox": proposal["bbox"],
                                "text": _load_ocr(root, proposal["crop_rgb_sha256"], ocr_cache)}
                               for proposal in shard["paddings"][str(pad)]["proposals"]]
                normalized_ids = match_word_ids(predictions, words, normalize=True)
                casefold_ids = match_word_ids(predictions, words, normalize=False)
                correct[key].update(f"{sample_id}:{index}" for index in normalized_ids)
                casefold[key].update(f"{sample_id}:{index}" for index in casefold_ids)
                totals[key]["normalized_spatial_exact"] += len(normalized_ids)
                totals[key]["casefold_spatial_control"] += len(casefold_ids)
                totals[key]["normalized_lexical_multiset"] += lexical_multiset_matches(predictions, words)
        if position % 50 == 0:
            print(f"[normalized rescore] {position}/300 images", flush=True)
    if gt_total != 2204:
        raise ValueError(f"Total-Text GT denominator changed: {gt_total}")
    output = {}
    for variant in variants:
        pad_rows = {}
        for pad in pads:
            key = (variant, str(pad))
            frozen_casefold = set(report["variants"][variant]["correct_word_ids_by_padding"][str(pad)])
            if casefold[key] != frozen_casefold:
                raise ValueError(f"Casefold control differs from frozen result: {variant}, pad={pad}")
            counts = totals[key]
            pad_rows[str(pad)] = {
                **counts,
                "normalized_spatial_exact_rate_all_gt": counts["normalized_spatial_exact"] / 2204,
                "normalized_lexical_rate_all_gt": counts["normalized_lexical_multiset"] / 2204,
                "newly_correct_vs_casefold": len(correct[key] - casefold[key]),
                "lost_vs_casefold_due_to_empty_guard": len(casefold[key] - correct[key]),
                "geometry_iou_ge_0_5_unchanged": report["variants"][variant]["paddings"][str(pad)]["geometry"]["iou_ge_0_50_one_to_one"],
            }
        output[variant] = {
            "paddings": pad_rows,
            "normalized_exact_word_intersections": padding_overlap(
                {str(pad): correct[(variant, str(pad))] for pad in pads}, pads),
            "normalized_correct_word_ids_by_padding": {
                str(pad): sorted(correct[(variant, str(pad))]) for pad in pads},
        }
    return {
        "schema_version": 1, "classification": "PRACTICAL_APPROXIMATION",
        "scope": "official Total-Text TEST-300 post-hoc diagnostic",
        "source_results_sha256": EXPECTED_RESULTS_SHA256,
        "normalization": "Unicode casefold, then remove codepoints whose Unicode general category begins P or S; preserve accents, digits and whitespace; empty normalized tokens cannot match",
        "spatial_matching": "maximum one-to-one bbox IoU>=0.5",
        "gt_words": 2204, "gt_words_empty_after_normalization": gt_empty,
        "inference_rerun": False, "variants": output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.run_root / "results.json"
    if file_sha256(report_path) != EXPECTED_RESULTS_SHA256:
        raise ValueError("Frozen source report SHA256 mismatch")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite normalized score: {args.output}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = rescore(args.run_root, report)
    _atomic_json(args.output, result)
    print("JSON", args.output, "SHA256", file_sha256(args.output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
