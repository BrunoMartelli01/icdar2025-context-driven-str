"""Explicit ASSUMPTION policies for DeepSolo's unresolved multi-output mapping."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple


REDUCTION_POLICIES = (
    "highest_score",
    "source_order_space_join",
    "reading_order_space_join",
)


def _text(detection: Mapping[str, Any]) -> str:
    value = detection.get("text", "")
    if not isinstance(value, str):
        raise ValueError("DeepSolo detection text must be a string")
    return value


def _reading_key(index_and_detection: Tuple[int, Mapping[str, Any]]) -> Tuple[float, float, int]:
    index, detection = index_and_detection
    geometry = detection.get("geometry")
    if not isinstance(geometry, Mapping):
        raise ValueError("reading_order_space_join requires detection geometry")
    points = geometry.get("boundary_points")
    if not isinstance(points, Sequence) or not points:
        raise ValueError("reading_order_space_join requires non-empty boundary_points")
    normalized = []
    for point in points:
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError("Each boundary point must contain x and y")
        normalized.append((float(point[0]), float(point[1])))
    return min(y for _, y in normalized), min(x for x, _ in normalized), index


def reduce_detections(detections: Iterable[Mapping[str, Any]], policy: str) -> str:
    """Reduce detections only through an explicitly named, non-paper-exact policy."""
    values = list(detections)
    if policy not in REDUCTION_POLICIES:
        raise ValueError("Unknown DeepSolo reduction policy: {}".format(policy))
    if not values:
        return ""
    if policy == "highest_score":
        for detection in values:
            if not isinstance(detection.get("score"), (int, float)):
                raise ValueError("highest_score requires numeric detection scores")
        selected = max(enumerate(values), key=lambda item: (float(item[1]["score"]), -item[0]))[1]
        return _text(selected)
    if policy == "reading_order_space_join":
        values = [item[1] for item in sorted(enumerate(values), key=_reading_key)]
    return " ".join(text for text in (_text(item) for item in values) if text)
