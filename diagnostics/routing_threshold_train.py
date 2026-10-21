"""Post-hoc TRAIN-only C separability audit, without changing routing or models.

Only whole-caption scores are used. A threshold yielding no observed errors is
an in-sample calibration result, not a correctness guarantee on unseen inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from diagnostics.caption_routing_train import checked
from diagnostics.repository_contours_train import _iou, _inter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values):
    if not values:
        return {"count": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}
    quantiles = np.quantile(values, [0, .1, .5, .9, 1])
    return dict(count=len(values), **dict(zip(("min", "p10", "median", "p90", "max"), map(float, quantiles))))


def token_labels(text, gt_words):
    """Casefold whitespace tokens, image-local multiplicity capped by GT.

    No punctuation stripping, partial matching or oracle choice of C is used.
    This tests returned tokens, not whether the output contains every GT word.
    """
    remaining = Counter(t for word in gt_words for t in word["text"].casefold().split())
    result = []
    for token in text.casefold().split():
        correct = remaining[token] > 0
        if correct:
            remaining[token] -= 1
        result.append({"token": token, "correct": correct})
    return result


def accepted_summary(records, threshold):
    accepted = [r for r in records if r["eligible"] and r["C"] >= threshold]
    correct = sum(r["correct"] for r in accepted)
    return {"threshold": threshold, "accepted_units": len(accepted),
        "correct_units": correct, "incorrect_units": len(accepted) - correct,
        "correct_words": sum(r["good_words"] for r in accepted),
        "incorrect_words": sum(r["bad_words"] for r in accepted),
        "precision_units": correct / len(accepted) if accepted else None,
        "rejected_units": len(records) - len(accepted)}


def separability(records):
    eligible = [r for r in records if r["eligible"]]
    if any(not math.isfinite(r["C"]) or not -.6 <= r["C"] <= 1 for r in eligible):
        raise ValueError("invalid confidence score")
    good = [r["C"] for r in eligible if r["correct"]]
    bad = [r["C"] for r in eligible if not r["correct"]]
    # C >= tau: exclude ties with an incorrect output, never round this bound.
    bound = max(bad) if bad else None
    threshold = max(0., math.nextafter(bound, math.inf)) if bad else 0.
    safe = accepted_summary(records, threshold)
    safe["correct_units_retained_fraction"] = safe["correct_units"] / len(good) if good else None
    safe["correct_units_lost"] = len(good) - safe["correct_units"]
    overlap = [v for v in good if bound is not None and v <= bound]
    return {
        "total_units": len(records), "eligible_units": len(eligible),
        "correct_C": distribution(good), "incorrect_C": distribution(bad),
        "correct_word_C": distribution([r["C"] for r in eligible for _ in range(r["good_words"])]),
        "incorrect_word_C": distribution([r["C"] for r in eligible for _ in range(r["bad_words"])]),
        "zero_observed_errors_train": safe,
        "strict_boundary_C_greater_than": bound,
        "useful_zero_error_threshold_exists": safe["correct_units"] > 0,
        "all_correct_separable_from_all_incorrect": bool(good) and (not bad or min(good) > max(bad)),
        "correct_units_at_or_below_max_incorrect_C": len(overlap),
        "at_frozen_tau_0_8": accepted_summary(records, .8),
        "highest_scoring_errors": sorted([r for r in eligible if not r["correct"]], key=lambda r: -r["C"])[:5],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routing-report", type=Path, required=True)
    parser.add_argument("--routing-sha256", required=True)
    parser.add_argument("--gt-report", type=Path, required=True)
    parser.add_argument("--gt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite an existing diagnostic")
    if sha(args.routing_report) != args.routing_sha256 or sha(args.gt_report) != args.gt_sha256:
        raise RuntimeError("frozen source hash mismatch")
    source = checked(args.routing_report)
    gt_source = json.loads(args.gt_report.read_text())
    scope = {"split": "TRAIN", "images": 229, "gt_words": 849}
    if source["scope"] != scope or gt_source["protocol"]["scope"] != scope:
        raise RuntimeError("requires frozen TRAIN229 / 849 words")
    gt = defaultdict(list)
    for word in gt_source["per_word"]:
        gt[word["sample_id"]].append({k: word[k] for k in ("gt_index", "bbox", "text")})
    if len(gt) != 229 or sum(map(len, gt.values())) != 849:
        raise RuntimeError("invalid GT population")
    summaries, all_rows, hashes = {}, {}, {}
    root = args.routing_report.parent
    for profile in source["caption_quality"]:
        rows = {"blocks_text_exact_positive_overlap": [], "blocks_spatial_exact_iou_ge_0_5": [],
                "paired_image_all_returned_tokens_correct": []}
        paths = sorted((root / "routes" / profile / "whole_caption").glob("[0-9]*.json"))
        if len(paths) != 229:
            raise RuntimeError("missing whole-caption samples")
        ids = set()
        for path in paths:
            sample = checked(path, source["contract_sha256"])
            sid = sample["sample_id"]
            if sid not in gt or sid in ids:
                raise RuntimeError("sample population mismatch")
            ids.add(sid)
            hashes[str(path.relative_to(root))] = sha(path)
            words = gt[sid]
            for block in sample["blocks"]:
                if block["window"] != sample["T2"]:
                    raise RuntimeError("only entire T2 is allowed")
                s, l, c = block["semantic_similarity"], block["lexical_similarity"], block["confidence"]
                if abs(c - (.6 * s + .4 * l)) > 1e-12:
                    raise RuntimeError("invalid stored block score")
                matching_words = [w for w in words if block["text"].casefold() == w["text"].casefold()]
                best = max(words, key=lambda w: _iou(block["bbox"], w["bbox"]))
                common = {"sample_id": sid, "candidate_id": block["candidate_id"],
                    "bbox": block["bbox"], "text": block["text"], "C": c, "S": s, "L": l,
                    "eligible": bool(block["text"].strip()), "best_gt_text": best["text"],
                    "best_gt_iou": _iou(block["bbox"], best["bbox"])}
                for mode, correct in (
                    ("blocks_text_exact_positive_overlap", any(_inter(block["bbox"], w["bbox"]) > 0 for w in matching_words)),
                    ("blocks_spatial_exact_iou_ge_0_5", any(_iou(block["bbox"], w["bbox"]) >= .5 for w in matching_words)),
                ):
                    rows[mode].append({**common, "correct": correct, "good_words": int(correct),
                        "bad_words": int(not correct and common["eligible"])})
            decision = sample["image_routes"]["selected_candidate_paired"]
            selected = "T1" if decision["S1"] >= decision["S3"] else "T3"
            expected = .6 * decision["S1" if selected == "T1" else "S3"] + .4 * decision["L1" if selected == "T1" else "L3"]
            if abs(expected - decision["C"]) > 1e-12 or decision["semantic_winner"] != selected:
                raise RuntimeError("stored paired score mismatch")
            if sample["T1_score"]["window"] != sample["T2"] or sample["T3_score"]["window"] != sample["T2"]:
                raise RuntimeError("paired routing must use whole caption")
            tokens = token_labels(decision["selected_output"], words)
            good = sum(t["correct"] for t in tokens)
            bad = len(tokens) - good
            rows["paired_image_all_returned_tokens_correct"].append({
                "sample_id": sid, "text": decision["selected_output"], "winner": selected,
                "C": decision["C"], "S": decision["S"], "L": decision["L"],
                "eligible": bool(tokens), "correct": bool(tokens) and bad == 0,
                "good_words": good, "bad_words": bad, "tokens": tokens,
                "gt_texts": [w["text"] for w in words]})
        if any(len(rows[m]) != 1203 for m in rows if m.startswith("blocks")):
            raise RuntimeError("requires frozen 1203 crop population")
        if sum(r["correct"] for r in rows["blocks_spatial_exact_iou_ge_0_5"]) != 488:
            raise RuntimeError("strict crop exact baseline mismatch")
        summaries[profile] = {mode: separability(records) for mode, records in rows.items()}
        all_rows[profile] = rows
        print(profile, json.dumps({mode: {"correct_range": value["correct_C"],
            "wrong_range": value["incorrect_C"], "zero_error": value["zero_observed_errors_train"]}
            for mode, value in summaries[profile].items()}), flush=True)
    result = {
        "classification": "PRACTICAL_APPROXIMATION_TRAIN_DIAGNOSTIC", "scope": scope,
        "caption_comparison": "whole_caption_only", "alpha": .6, "beta": .4,
        "threshold_deployed": False, "models_rerun": False, "test_used": False,
        "selection": "lowest representable nonnegative tau above maximum incorrect C; C>=tau, empty output always rejected",
        "limitations": ["TRAIN-only empirical thresholds, not a correctness guarantee on unseen data",
            "Per-crop C is not the image-level paired decision",
            "Paired all-returned-tokens-correct does not require complete image transcription; whitespace/casefold multiset matching has no spatial constraint",
            "Crop counts are not counts of unique GT words; repeated predictions can match the same GT word",
            "Existing semantic tie policy T1>=T3 preserved, no routing edits"],
        "label_definitions": {
            "blocks_text_exact_positive_overlap": "whole crop text equals a GT word casefold, with positive bbox intersection",
            "blocks_spatial_exact_iou_ge_0_5": "whole crop text equals a GT word casefold and bbox IoU>=.5",
            "paired_image_all_returned_tokens_correct": "nonempty semantic winner output has zero wrong tokens under per-image casefold whitespace multiset matching; omissions allowed"},
        "inputs": {"routing_report": str(args.routing_report.resolve()), "routing_sha256": args.routing_sha256,
            "gt_report": str(args.gt_report.resolve()), "gt_sha256": args.gt_sha256, "route_shards_sha256": hashes},
        "summary": summaries, "per_candidate": all_rows, "script_sha256": sha(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"JSON {args.output} SHA256 {sha(args.output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
