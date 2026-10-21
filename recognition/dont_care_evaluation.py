"""Neutralize final predictions for metrics only, retaining all original rows."""
from __future__ import annotations


def apply_neutralization(predictions: list[dict], geometry_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_id = {row["candidate_id"]: row for row in geometry_rows}
    if len(by_id) != len(geometry_rows):
        raise ValueError("duplicate geometry candidate IDs")
    if len({row["candidate_id"] for row in predictions}) != len(predictions):
        raise ValueError("duplicate prediction candidate IDs")
    evaluated, neutralized = [], []
    for prediction in predictions:
        record = by_id.get(prediction["candidate_id"])
        if record is None or type(record["neutralized"]) is not bool:
            raise ValueError("missing or invalid polygon neutralization decision")
        (neutralized if record["neutralized"] else evaluated).append(prediction)
    return evaluated, neutralized
