"""Anchor-complete local word-proposal selection.

This module is a separate ``PRACTICAL_APPROXIMATION``.  Each frozen
practical-remap-v1 proposal is a real fallback candidate.  Alternatives cover
contiguous v1 anchors and are scored per covered anchor, preventing proposal
count from making fragments artificially preferable to a whole-word box.
Ground truth is absent from this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Final, Sequence

from .practical_candidate_selection import box_iou, percentile_ranks, vertical_overlap


MODE: Final = "practical-candidate-selection-v2"
CLASSIFICATION: Final = "PRACTICAL_APPROXIMATION"
VARIANTS: Final = (
    "OCR_ONLY_ANCHOR_MEAN",
    "OCR_CONSENSUS_ANCHOR_MEAN",
    "BALANCED_ANCHOR_MEAN",
)
VARIANT_WEIGHTS: Final = {
    "OCR_ONLY_ANCHOR_MEAN": {"ocr": 1.0},
    "OCR_CONSENSUS_ANCHOR_MEAN": {"ocr": 0.5, "consensus": 0.5},
    "BALANCED_ANCHOR_MEAN": {
        "ocr": 0.4, "consensus": 0.25, "boundary": 0.2, "density": 0.15,
    },
}
MAX_ALTERNATIVES_PER_ANCHOR_SPAN: Final = 3
MAX_SELECTED: Final = 10


def _contains(outer: Sequence[int], inner: Sequence[int]) -> bool:
    return (
        int(outer[0]) <= int(inner[0]) and int(outer[1]) <= int(inner[1])
        and int(outer[2]) >= int(inner[2]) and int(outer[3]) >= int(inner[3])
    )


def _line_compatible(left: Sequence[int], right: Sequence[int]) -> bool:
    left_height = int(left[3]) - int(left[1])
    right_height = int(right[3]) - int(right[1])
    if min(left_height, right_height) <= 0:
        return False
    reference_height = (left_height + right_height) / 2.0
    center_distance = abs((left[1] + left[3]) / 2.0 - (right[1] + right[3]) / 2.0)
    return (
        max(left_height, right_height) / min(left_height, right_height) <= 2.5
        and (vertical_overlap(left, right) >= 0.5 or center_distance / reference_height <= 0.5)
    )


def form_anchor_lines(anchors: Sequence[dict[str, Any]]) -> tuple[tuple[int, ...], ...]:
    """Deterministically form lines from exact v1 unpadded boxes."""
    lines: list[list[int]] = []
    for index in sorted(
        range(len(anchors)),
        key=lambda value: (
            (anchors[value]["tight_bbox"][1] + anchors[value]["tight_bbox"][3]) / 2.0,
            anchors[value]["tight_bbox"][0], anchors[value]["candidate_id"],
        ),
    ):
        box = anchors[index]["tight_bbox"]
        matches = []
        for line_index, members in enumerate(lines):
            member_boxes = [anchors[value]["tight_bbox"] for value in members]
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
    return tuple(
        tuple(sorted(line, key=lambda value: (
            anchors[value]["tight_bbox"][0], anchors[value]["tight_bbox"][1],
            anchors[value]["candidate_id"],
        )))
        for line in lines
    )


def attach_variant_scores(options: list[dict[str, Any]]) -> None:
    """Compute arithmetic means of within-image feature percentiles."""
    if not options:
        return
    raw = {
        "ocr": [
            float(row["ocr"]["mean_token_log_probability"])
            if row["ocr"].get("mean_token_log_probability") is not None else float("-inf")
            for row in options
        ],
        "consensus": [float(row["consensus_score"]) for row in options],
        "boundary": [float(row["boundary_score"]) for row in options],
        "density": [float(row["foreground_density"]) for row in options],
        "context": [float(row.get("context_score", 0.0)) for row in options],
    }
    ranks = {name: percentile_ranks(values) for name, values in raw.items()}
    for index, option in enumerate(options):
        option["feature_percentiles"] = {name: values[index] for name, values in ranks.items()}
        option["variant_scores"] = {
            variant: sum(weight * ranks[name][index] for name, weight in weights.items())
            for variant, weights in VARIANT_WEIGHTS.items()
        }


def assign_anchor_spans(
    anchors: Sequence[dict[str, Any]], alternatives: Sequence[dict[str, Any]],
) -> tuple[tuple[int, ...], list[dict[str, Any]]]:
    """Assign candidates to contiguous anchor spans using box containment only."""
    lines = form_anchor_lines(anchors)
    line_by_anchor = {anchor: line_id for line_id, line in enumerate(lines) for anchor in line}
    position_by_anchor = {anchor: position for line in lines for position, anchor in enumerate(line)}
    assigned: list[dict[str, Any]] = []
    for alternative in alternatives:
        covered = tuple(
            index for index, anchor in enumerate(anchors)
            if _contains(alternative["bbox"], anchor["tight_bbox"])
            and _line_compatible(alternative["bbox"], anchor["tight_bbox"])
        )
        if not covered:
            continue
        line_ids = {line_by_anchor[index] for index in covered}
        if len(line_ids) != 1:
            continue
        line_id = next(iter(line_ids))
        positions = sorted(position_by_anchor[index] for index in covered)
        if positions != list(range(positions[0], positions[-1] + 1)):
            continue
        line = lines[line_id]
        span = tuple(line[position] for position in positions)
        row = dict(alternative)
        row["covered_anchor_indices"] = span
        row["anchor_line_id"] = line_id
        row["anchor_span_start"] = positions[0]
        row["anchor_span_end"] = positions[-1] + 1
        assigned.append(row)
    return lines, assigned


@dataclass(frozen=True)
class _Solution:
    utility: float
    replacements: int
    rows: tuple[dict[str, Any], ...]


def _better(left: _Solution, right: _Solution, *, variant: str) -> _Solution:
    left_key = (
        left.utility, -left.replacements,
        tuple(row["candidate_id"] for row in left.rows),
    )
    right_key = (
        right.utility, -right.replacements,
        tuple(row["candidate_id"] for row in right.rows),
    )
    return left if left_key >= right_key else right


def select_anchor_complete(
    anchors: list[dict[str, Any]], alternatives: list[dict[str, Any]], *, variant: str,
) -> tuple[dict[str, Any], ...]:
    """Partition each anchor line and maximize mean score per covered anchor."""
    if variant not in VARIANTS:
        raise ValueError("unknown v2 selector variant")
    if len(anchors) > MAX_SELECTED:
        raise ValueError("v1 anchor input must already satisfy top10")
    all_options = anchors + alternatives
    attach_variant_scores(all_options)
    lines, assigned = assign_anchor_spans(anchors, alternatives)
    selected: list[dict[str, Any]] = []
    for line_id, line in enumerate(lines):
        options_by_start: dict[int, list[dict[str, Any]]] = {}
        for position, anchor_index in enumerate(line):
            base = dict(anchors[anchor_index])
            base["covered_anchor_indices"] = (anchor_index,)
            base["anchor_line_id"] = line_id
            base["anchor_span_start"] = position
            base["anchor_span_end"] = position + 1
            options_by_start.setdefault(position, []).append(base)
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for row in assigned:
            if row["anchor_line_id"] == line_id:
                grouped.setdefault((row["anchor_span_start"], row["anchor_span_end"]), []).append(row)
        for span, rows in grouped.items():
            rows.sort(key=lambda row: (
                -row["variant_scores"][variant],
                -row["feature_percentiles"]["context"], row["candidate_id"],
            ))
            options_by_start.setdefault(span[0], []).extend(rows[:MAX_ALTERNATIVES_PER_ANCHOR_SPAN])

        dp: list[_Solution | None] = [None] * (len(line) + 1)
        dp[0] = _Solution(0.0, 0, ())
        for position in range(len(line)):
            if dp[position] is None:
                raise RuntimeError("anchor partition became infeasible")
            for option in options_by_start.get(position, ()):
                end = int(option["anchor_span_end"])
                count = end - position
                if count < 1:
                    raise RuntimeError("invalid anchor span")
                candidate = _Solution(
                    utility=dp[position].utility + option["variant_scores"][variant] * count,
                    replacements=dp[position].replacements + int(option.get("source") != "v1_exact"),
                    rows=dp[position].rows + (option,),
                )
                dp[end] = candidate if dp[end] is None else _better(dp[end], candidate, variant=variant)
        if dp[-1] is None:
            raise RuntimeError("anchor line has no complete partition")
        selected.extend(dp[-1].rows)
    if len(selected) > MAX_SELECTED:
        raise RuntimeError("anchor-complete selection cannot increase v1 proposal count")
    return tuple(sorted(selected, key=lambda row: (
        min(int(value) for value in row["covered_anchor_indices"]), row["candidate_id"],
    )))


def selector_contract() -> dict[str, Any]:
    return {
        "mode": MODE, "classification": CLASSIFICATION,
        "variants": list(VARIANTS), "weights": VARIANT_WEIGHTS,
        "objective": "sum(candidate arithmetic-mean score * number of covered v1 anchors)",
        "interpretation": "fixed-denominator mean score per v1 anchor",
        "fallback": "every exact v1 crop is a feasible singleton option",
        "alternative_geometry": "same-line contiguous anchors whose unpadded v1 boxes are contained",
        "maximum_alternatives_per_anchor_span": MAX_ALTERNATIVES_PER_ANCHOR_SPAN,
        "maximum_selected": MAX_SELECTED, "gt_used": False,
    }


__all__: Final = [
    "CLASSIFICATION", "MAX_ALTERNATIVES_PER_ANCHOR_SPAN", "MAX_SELECTED", "MODE",
    "VARIANTS", "VARIANT_WEIGHTS", "assign_anchor_spans", "attach_variant_scores",
    "form_anchor_lines", "select_anchor_complete", "selector_contract",
]
