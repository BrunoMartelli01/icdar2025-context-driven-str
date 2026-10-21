"""Training-free selection of coherent word proposals from frozen hypotheses.

This is a separate ``PRACTICAL_APPROXIMATION``.  It never changes the
paper-faithful Algorithm 1 or practical-remap-v1.  Ground truth is deliberately
absent from every API in this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import median
from typing import Any, Final, Iterable, Sequence


MODE: Final = "practical-candidate-selection-v1"
VARIANTS: Final = ("OCR_FIRST", "CONSENSUS_FIRST", "EQUAL_RANK_FUSION")
FAMILY_IOU_MIN: Final = 0.75
ANCHOR_OPTIONS: Final = 3
ANCHOR_MIN_IOU: Final = 0.50
MAX_SELECTED: Final = 10
UTILITY_FLOOR: Final = 0.0

WEIGHTS: Final = {
    "OCR_FIRST": {
        "ocr": 0.55, "consensus": 0.20, "boundary": 0.15,
        "density": 0.05, "v1": 0.05,
    },
    "CONSENSUS_FIRST": {
        "ocr": 0.20, "consensus": 0.35, "boundary": 0.25,
        "density": 0.10, "v1": 0.10,
    },
    "EQUAL_RANK_FUSION": {
        "ocr": 0.20, "consensus": 0.20, "boundary": 0.20,
        "density": 0.20, "v1": 0.20,
    },
}


def box_area(box: Sequence[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def intersection_area(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(int(left[2]), int(right[2])) - max(int(left[0]), int(right[0]))) * max(
        0, min(int(left[3]), int(right[3])) - max(int(left[1]), int(right[1]))
    )


def box_iou(left: Sequence[int], right: Sequence[int]) -> float:
    intersection = intersection_area(left, right)
    union = box_area(left) + box_area(right) - intersection
    return intersection / union if union else 0.0


def vertical_overlap(left: Sequence[int], right: Sequence[int]) -> float:
    overlap = max(0, min(int(left[3]), int(right[3])) - max(int(left[1]), int(right[1])))
    denominator = min(int(left[3]) - int(left[1]), int(right[3]) - int(right[1]))
    return overlap / denominator if denominator > 0 else 0.0


def percentile_ranks(values: Sequence[float]) -> tuple[float, ...]:
    """Stable mid-rank percentiles in [0,1], with non-finite values last."""
    if not values:
        return ()
    finite = [value for value in values if isfinite(value)]
    floor = min(finite) - 1.0 if finite else -1.0
    normalized = [value if isfinite(value) else floor for value in values]
    ordered = sorted((value, index) for index, value in enumerate(normalized))
    output = [0.0] * len(values)
    start = 0
    denominator = max(1, len(values) - 1)
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        rank = ((start + end - 1) / 2) / denominator
        for _, index in ordered[start:end]:
            output[index] = rank
        start = end
    return tuple(output)


def cluster_candidate_families(
    candidates: Sequence[dict[str, Any]], *, minimum_iou: float = FAMILY_IOU_MIN,
) -> tuple[tuple[int, ...], ...]:
    """Greedy non-transitive clustering of geometrically equivalent native boxes."""
    if not 0 < minimum_iou <= 1:
        raise ValueError("minimum_iou must be in (0,1]")
    families: list[list[int]] = []
    representatives: list[Sequence[int]] = []
    for index, candidate in enumerate(candidates):
        box = candidate["source_bbox_native"]
        matches = [
            (box_iou(box, representative), family_index)
            for family_index, representative in enumerate(representatives)
            if box_iou(box, representative) >= minimum_iou
        ]
        if matches:
            _, family_index = max(matches, key=lambda value: (value[0], -value[1]))
            families[family_index].append(index)
        else:
            representatives.append(box)
            families.append([index])
    return tuple(tuple(value) for value in families)


def attach_family_consensus(candidates: list[dict[str, Any]]) -> None:
    """Attach threshold/box persistence without consulting GT or OCR text.

    Support is the fraction of the four frozen probability thresholds present
    in a geometric family.  Stability is the mean IoU to the deterministic
    family medoid.  Their product deliberately rewards hypotheses that recur
    across thresholds while retaining a value in [0, 1].
    """
    families = cluster_candidate_families(candidates)
    for family_id, family in enumerate(families, start=1):
        medoid = max(
            family,
            key=lambda index: (
                sum(
                    box_iou(
                        candidates[index]["source_bbox_native"],
                        candidates[other]["source_bbox_native"],
                    )
                    for other in family
                ),
                -int(candidates[index]["candidate_index"]),
            ),
        )
        stability = sum(
            box_iou(
                candidates[medoid]["source_bbox_native"],
                candidates[index]["source_bbox_native"],
            )
            for index in family
        ) / len(family)
        threshold_support = len({
            float(candidates[index]["source_threshold"]) for index in family
        })
        consensus = (threshold_support / 4.0) * stability
        for index in family:
            candidates[index]["consensus_score"] = consensus
            candidates[index]["consensus_family_id"] = family_id
            candidates[index]["consensus_family_size"] = len(family)
            candidates[index]["consensus_threshold_support"] = threshold_support
            candidates[index]["consensus_stability"] = stability


def empirical_gap_percentile(value: float, population: Sequence[float]) -> float:
    if not population:
        return 1.0
    less = sum(item < value for item in population)
    equal = sum(item == value for item in population)
    return (less + 0.5 * equal) / len(population)


def span_boundary_score(
    *, component_boxes: Sequence[Sequence[int]], start: int, end: int,
) -> float:
    """Parameter-free, line-relative boundary/isolation score for [start,end)."""
    if not (0 <= start < end <= len(component_boxes)):
        raise ValueError("invalid contiguous component span")
    gaps = [
        float(max(0, int(component_boxes[index + 1][0]) - int(component_boxes[index][2])))
        for index in range(len(component_boxes) - 1)
    ]
    internal = gaps[start:end - 1]
    boundary_percentiles = []
    if start > 0:
        boundary_percentiles.append(empirical_gap_percentile(gaps[start - 1], gaps))
    else:
        boundary_percentiles.append(1.0)
    if end < len(component_boxes):
        boundary_percentiles.append(empirical_gap_percentile(gaps[end - 1], gaps))
    else:
        boundary_percentiles.append(1.0)
    external = min(boundary_percentiles)
    internal_penalty = max(
        (empirical_gap_percentile(value, gaps) for value in internal), default=0.0
    )
    return max(0.0, min(1.0, 0.5 * external + 0.5 * (1.0 - internal_penalty)))


def candidate_density(foreground_area: int, box: Sequence[int]) -> float:
    area = box_area(box)
    return min(1.0, foreground_area / area) if area else 0.0


def v1_support(box: Sequence[int], anchors: Sequence[Sequence[int]]) -> float:
    return max((box_iou(box, anchor) for anchor in anchors), default=0.0)


def attach_variant_scores(candidates: list[dict[str, Any]]) -> None:
    """Attach fixed, within-image rank-fusion scores in place."""
    raw = {
        "ocr": [float(row["ocr"]["mean_token_log_probability"] or float("-inf")) for row in candidates],
        "consensus": [float(row["consensus_score"]) for row in candidates],
        "boundary": [float(row["boundary_score"]) for row in candidates],
        "density": [float(row["foreground_density"]) for row in candidates],
        "v1": [float(row["v1_support"]) for row in candidates],
        "context": [float(row["context_score"]) for row in candidates],
    }
    ranks = {name: percentile_ranks(values) for name, values in raw.items()}
    for index, candidate in enumerate(candidates):
        candidate["feature_percentiles"] = {
            name: values[index] for name, values in ranks.items()
        }
        candidate["variant_scores"] = {
            variant: sum(
                weight * ranks[name][index] for name, weight in WEIGHTS[variant].items()
            )
            for variant in VARIANTS
        }


def _line_compatible(left: Sequence[int], right: Sequence[int]) -> bool:
    left_height = int(left[3]) - int(left[1])
    right_height = int(right[3]) - int(right[1])
    if min(left_height, right_height) <= 0:
        return False
    height_ratio = max(left_height, right_height) / min(left_height, right_height)
    center_distance = abs((left[1] + left[3]) / 2 - (right[1] + right[3]) / 2)
    reference = (left_height + right_height) / 2
    return (
        height_ratio <= 2.5
        and (vertical_overlap(left, right) >= 0.5 or center_distance / reference <= 0.5)
    )


def form_candidate_lines(candidates: Sequence[dict[str, Any]]) -> tuple[tuple[int, ...], ...]:
    lines: list[list[int]] = []
    for index in sorted(
        range(len(candidates)),
        key=lambda value: (
            (candidates[value]["source_bbox_native"][1] + candidates[value]["source_bbox_native"][3]) / 2,
            candidates[value]["source_bbox_native"][0], candidates[value]["candidate_id"],
        ),
    ):
        box = candidates[index]["source_bbox_native"]
        matches = []
        for line_index, members in enumerate(lines):
            member_boxes = [candidates[value]["source_bbox_native"] for value in members]
            reference = (
                min(value[0] for value in member_boxes), min(value[1] for value in member_boxes),
                max(value[2] for value in member_boxes), max(value[3] for value in member_boxes),
            )
            if _line_compatible(box, reference):
                matches.append((vertical_overlap(box, reference), line_index))
        if matches:
            lines[max(matches, key=lambda value: (value[0], -value[1]))[1]].append(index)
        else:
            lines.append([index])
    return tuple(tuple(value) for value in lines)


@dataclass(frozen=True)
class _Solution:
    utility: float
    indices: tuple[int, ...]


def weighted_interval_selection(
    candidates: Sequence[dict[str, Any]], indices: Sequence[int], *, variant: str,
    maximum: int = MAX_SELECTED,
) -> tuple[int, ...]:
    """Maximum-utility non-overlapping x intervals for one geometric line."""
    if variant not in VARIANTS:
        raise ValueError("unknown selector variant")
    ordered = sorted(
        indices,
        key=lambda value: (
            candidates[value]["source_bbox_native"][2],
            candidates[value]["source_bbox_native"][0], candidates[value]["candidate_id"],
        ),
    )
    previous = []
    for offset, index in enumerate(ordered):
        start = candidates[index]["source_bbox_native"][0]
        compatible = -1
        for prior in range(offset - 1, -1, -1):
            if candidates[ordered[prior]]["source_bbox_native"][2] <= start:
                compatible = prior
                break
        previous.append(compatible)
    dp: list[list[_Solution]] = [
        [_Solution(0.0, ()) for _ in range(maximum + 1)] for _ in range(len(ordered) + 1)
    ]
    for offset, index in enumerate(ordered, start=1):
        utility = float(candidates[index]["variant_scores"][variant]) - UTILITY_FLOOR
        for count in range(maximum + 1):
            best = dp[offset - 1][count]
            if count and utility > 0:
                base = dp[previous[offset - 1] + 1][count - 1]
                include = _Solution(base.utility + utility, base.indices + (index,))
                if (include.utility, -len(include.indices), tuple(
                    candidates[value]["candidate_id"] for value in include.indices
                )) > (best.utility, -len(best.indices), tuple(
                    candidates[value]["candidate_id"] for value in best.indices
                )):
                    best = include
            dp[offset][count] = best
    return max(
        dp[-1], key=lambda value: (
            value.utility, -len(value.indices),
            tuple(candidates[index]["candidate_id"] for index in value.indices),
        ),
    ).indices


def select_candidates(
    candidates: list[dict[str, Any]], *, anchors: Sequence[Sequence[int]],
    variant: str, maximum: int = MAX_SELECTED,
) -> tuple[dict[str, Any], ...]:
    if variant not in VARIANTS:
        raise ValueError("unknown selector variant")
    if maximum != MAX_SELECTED:
        raise ValueError("candidate-selection-v1 fixes maximum=10")
    if not anchors:
        return ()
    if any("consensus_score" not in row for row in candidates):
        attach_family_consensus(candidates)
    if any("variant_scores" not in row for row in candidates):
        attach_variant_scores(candidates)
    family_members: dict[int, list[int]] = {}
    for index, candidate in enumerate(candidates):
        family_members.setdefault(int(candidate["consensus_family_id"]), []).append(index)
    families = tuple(tuple(value) for _, value in sorted(family_members.items()))
    representatives = []
    for family_id, family in enumerate(families, start=1):
        thresholds = {float(candidates[index]["source_threshold"]) for index in family}
        representative_index = max(
            family,
            key=lambda index: (
                candidates[index]["variant_scores"][variant],
                candidates[index]["feature_percentiles"]["context"],
                candidates[index]["bbox_variant"] == "standard_pad6",
                -int(candidates[index]["candidate_index"]),
            ),
        )
        row = dict(candidates[representative_index])
        row["family_id"] = family_id
        row["family_size"] = len(family)
        row["family_threshold_support"] = len(thresholds)
        representatives.append(row)

    # v1 is the frozen recall anchor.  Candidate selection is deliberately
    # local: retain at most three geometrically intersecting alternatives per
    # anchor, then solve conflicts globally.  This prevents the 250+ proposal
    # pool from becoming a new unconstrained detector.
    option_indices: set[int] = set()
    for anchor_id, anchor in enumerate(anchors, start=1):
        matches = [
            index for index, candidate in enumerate(representatives)
            if box_iou(candidate["bbox"], anchor) >= ANCHOR_MIN_IOU
            and _line_compatible(candidate["bbox"], anchor)
        ]
        matches.sort(key=lambda index: (
            -representatives[index]["variant_scores"][variant],
            -representatives[index]["feature_percentiles"]["context"],
            representatives[index]["candidate_id"],
        ))
        if not matches:
            continue
        base_index = max(
            matches,
            key=lambda index: (
                box_iou(representatives[index]["bbox"], anchor),
                representatives[index]["bbox_variant"] == "standard_pad6",
                -int(representatives[index]["candidate_index"]),
            ),
        )
        choices = [base_index] + [index for index in matches if index != base_index]
        for index in choices[:ANCHOR_OPTIONS]:
            option_indices.add(index)
            representatives[index].setdefault("v1_anchor_ids", []).append(anchor_id)

    selected_indices = []
    for line in form_candidate_lines(representatives):
        line_options = tuple(index for index in line if index in option_indices)
        if not line_options:
            continue
        selected_indices.extend(weighted_interval_selection(
            representatives, line_options, variant=variant, maximum=maximum,
        ))
    # Resolve residual cross-line duplicates deterministically, then apply nmax.
    ordered = sorted(
        selected_indices,
        key=lambda index: (
            -representatives[index]["variant_scores"][variant],
            -representatives[index]["feature_percentiles"]["context"],
            representatives[index]["candidate_id"],
        ),
    )
    kept = []
    for index in ordered:
        candidate = representatives[index]
        if any(box_iou(candidate["bbox"], row["bbox"]) >= FAMILY_IOU_MIN for row in kept):
            continue
        kept.append(candidate)
        if len(kept) == maximum:
            break
    return tuple(sorted(
        kept,
        key=lambda row: (
            -int(row["foreground_area_native"]), row["bbox"][1], row["bbox"][0],
            row["candidate_id"],
        ),
    ))


def selector_contract() -> dict[str, Any]:
    return {
        "mode": MODE,
        "classification": "PRACTICAL_APPROXIMATION",
        "variants": list(VARIANTS),
        "family_iou_min": FAMILY_IOU_MIN,
        "utility_floor": UTILITY_FLOOR,
        "maximum_selected": MAX_SELECTED,
        "anchor_options": ANCHOR_OPTIONS,
        "anchor_min_iou": ANCHOR_MIN_IOU,
        "anchor_policy": "closest base plus two scored alternatives with IoU>=0.50 per frozen v1 proposal",
        "weights": WEIGHTS,
        "context_role": "exact-tie-break after the declared variant score",
        "gt_used_by_selector": False,
    }


__all__: Final = [
    "ANCHOR_MIN_IOU", "ANCHOR_OPTIONS", "FAMILY_IOU_MIN", "MAX_SELECTED", "MODE", "UTILITY_FLOOR",
    "VARIANTS", "WEIGHTS", "attach_family_consensus", "attach_variant_scores", "box_iou", "candidate_density",
    "cluster_candidate_families", "form_candidate_lines", "intersection_area",
    "percentile_ranks", "select_candidates", "selector_contract",
    "span_boundary_score", "v1_support", "weighted_interval_selection",
]
