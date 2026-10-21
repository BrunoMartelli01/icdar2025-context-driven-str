"""Selective whole-image routing of frozen PARSeq readings (APPROXIMATE).

All words enter the minimum. No word selection, correction, or GT in inference.
"""
from __future__ import annotations

import hashlib
import math

from recognition.clip_crop_admission import FEATURES, JOINT, percentile


def partition(sample_ids: list[str], count: int, seed: int) -> dict:
    if len(set(sample_ids)) != len(sample_ids) or not 0 < count < len(sample_ids):
        raise ValueError("split requires unique IDs and nonempty calibration/validation")
    ordered = sorted(sample_ids, key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest())
    calibration = set(ordered[:count])
    return {"calibration": [sid for sid in sample_ids if sid in calibration],
            "validation": [sid for sid in sample_ids if sid not in calibration]}


def image_scores(rows: list[dict], reference: dict) -> dict:
    families = ("native_balanced", "native_mean", "native_min", "scene_ensemble",
                *(f"rank_{f}" for f in FEATURES), *JOINT, "native_scene")
    # Empty or whitespace-only OCR forces whole-image fallback; never vacuous acceptance.
    if not rows or any(not r["raw_text"].strip() for r in rows):
        return dict.fromkeys(families)
    values = []
    for row in rows:
        native = row["confidence"]
        if not math.isfinite(native) or not 0 <= native <= 1:
            raise ValueError("invalid PARSeq confidence")
        ranks = {"native_rank": percentile(reference["native_rank"], native)}
        ranks.update({f: percentile(reference[f], row["clip"][f]) for f in FEATURES})
        scene = row["clip"]["scene_ensemble"]
        if not math.isfinite(scene):
            raise ValueError("invalid scene evidence")
        scene_rank = percentile(reference["scene_ensemble"], scene)
        score = {"native_balanced": native, "native_mean": row["native"]["confidence_mean"],
                 "native_min": row["native"]["confidence_min"], "scene_ensemble": scene,
                 **{f"rank_{f}": ranks[f] for f in FEATURES},
                 **{name: math.prod(ranks[f] for f in features) ** (1 / len(features))
                    for name, features in JOINT.items()},
                 "native_scene": math.sqrt(ranks["native_rank"] * scene_rank)}
        if not all(math.isfinite(v) for v in score.values()):
            raise ValueError("nonfinite routing score")
        values.append(score)
    return {f: min(v[f] for v in values) for f in families}


def accepted(score: float | None, threshold: float | None) -> bool:
    if threshold is None or score is None:
        return False
    if not math.isfinite(score) or not math.isfinite(threshold):
        raise ValueError("nonfinite routing threshold/score")
    return score >= threshold


def summarize(rows: list[dict], scores: dict, family: str | None, threshold: float | None) -> dict:
    kept = rows if family is None else [r for r in rows if accepted(scores[r["sample_id"]][family], threshold)]
    routed = [r for r in rows if r not in kept]
    tp = sum(r["tp"] for r in kept)
    words = sum(r["words"] for r in kept)
    gt = sum(r["gt_words"] for r in kept)
    fp, fn = words - tp, gt - tp
    safe = lambda r: r["words"] > 0 and r["tp"] == r["words"]
    good = sum(safe(r) for r in kept)
    complete = sum(safe(r) and r["tp"] == r["gt_words"] for r in kept)
    # Exact one-sided 95% binomial lower bound only when no image errors observed.
    # Images are the unit, not correlated words within an image. No guarantee claim.
    lower = .05 ** (1 / len(kept)) if kept and good == len(kept) else None
    return {"images": len(rows), "accepted_images": len(kept), "fallback_images": len(routed),
            "coverage_images": len(kept)/len(rows) if rows else 0.,
            "detected_words": words, "correct_words": tp, "wrong_words": fp,
            "accepted_gt_words": gt, "missing_words_in_accepted_images": fn, "tn": 0,
            "accuracy_on_detected_words": tp/words if words else None,
            "accuracy_tn0_including_fn": tp/(tp+fp+fn) if tp+fp+fn else None,
            "safe_accepted_images": good, "unsafe_accepted_images": len(kept)-good,
            "accuracy_all_emitted_words_per_image": good/len(kept) if kept else None,
            "fully_correct_and_complete_images": complete,
            "already_correct_images_sent_to_deepsolo": sum(safe(r) for r in routed),
            "already_correct_words_sent_to_deepsolo": sum(r["tp"] for r in routed),
            "accepted_outputs_including_ignored": sum(r["outputs_all"] for r in kept),
            "zero_error_image_precision_lower95": lower}


def choose_threshold(rows: list[dict], scores: dict, family: str, target: float) -> dict:
    if not 0 < target <= 1:
        raise ValueError("invalid target accuracy")
    thresholds = sorted({scores[r["sample_id"]][family] for r in rows
                         if scores[r["sample_id"]][family] is not None})
    curve = [{"threshold": t, **summarize(rows, scores, family, t)} for t in thresholds]
    eligible = [p for p in curve if p["detected_words"] > 0
                and p["accuracy_on_detected_words"] >= target and p["unsafe_accepted_images"] == 0]
    best = min(eligible, key=lambda p: (-p["accepted_images"], -p["detected_words"], p["threshold"])) if eligible else None
    return {"family": family, "threshold": best["threshold"] if best else None,
            "status": "CALIBRATED" if best else "ABSTAIN_ALL_NO_SAFE_CALIBRATION_THRESHOLD",
            "calibration": best or summarize(rows, scores, family, None), "curve": curve}
