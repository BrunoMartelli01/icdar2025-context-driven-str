"""Post-hoc oracle word recall of an unrestricted crop hypothesis pool."""
from __future__ import annotations

from recognition.scene_text_v1_evaluation import box_iou, maximum_matching
from scripts.rescore_totaltext_highres_padding_symbols import normalize_word


def evaluate_pool(rows: list[dict], readings: dict[str, dict], ground_truth: list[dict],
                  iou_threshold: float) -> dict:
    normalized_gt = [normalize_word(word["text"]) for word in ground_truth]
    witnesses = {index: [] for index, value in enumerate(normalized_gt) if value}
    edges = []
    geometry = []
    for pi, row in enumerate(rows):
        prediction = normalize_word(readings[row["candidate_id"]]["raw_text"])
        for gi, word in enumerate(ground_truth):
            if not normalized_gt[gi] or box_iou(row["bbox"], word["bbox"]) < iou_threshold:
                continue
            geometry.append((pi, gi))
            if prediction and prediction == normalized_gt[gi]:
                edges.append((pi, gi))
                witnesses[gi].append(row["candidate_id"])
    # Every GT word is counted once even if hundreds of crops recognize it.
    matched = maximum_matching(len(rows), len(ground_truth), edges)
    return {"correct": len(matched), "correct_gt_indices": sorted(matched),
            "geometry_gt_indices": sorted(maximum_matching(len(rows), len(ground_truth), geometry)),
            "any_crop_correct_gt_indices": sorted(gi for gi, ids in witnesses.items() if ids),
            "witnesses": {str(gi): ids for gi, ids in witnesses.items() if ids},
            "candidates": len(rows), "distinct_crop_pixels": len({row["crop_rgb_sha256"] for row in rows})}
