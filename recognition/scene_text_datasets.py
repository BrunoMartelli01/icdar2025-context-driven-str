"""Fail-closed annotation parsers for additional scene-text benchmarks.

This module only normalizes annotation geometry and provenance.  It does not
define a recognition metric or claim reproduction of a paper table.  In
particular, CTW1500 is line-level and is an out-of-paper practical extension.
"""

from __future__ import annotations

import ast
import csv
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
import xml.etree.ElementTree as ET


class SceneTextAnnotationError(ValueError):
    """Raised when an annotation cannot be interpreted without guessing."""


CTW1500_MISSING_TRANSCRIPTION = "<missing-transcription>"


@dataclass(frozen=True)
class SceneTextInstance:
    polygon_xy: tuple[tuple[int, int], ...]
    transcription: str
    ignore: bool
    granularity: str
    orientation: str | None = None

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        xs = [point[0] for point in self.polygon_xy]
        ys = [point[1] for point in self.polygon_xy]
        return min(xs), min(ys), max(xs), max(ys)


def _read_utf8_sig(path: Path) -> str:
    try:
        return path.read_bytes().decode("utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise SceneTextAnnotationError(f"Could not read UTF-8 annotation {path}: {error}") from error


def _validate_instance(
    instance: SceneTextInstance,
    *,
    path: Path,
    record: str,
    image_size_wh: tuple[int, int] | None,
) -> SceneTextInstance:
    if instance.ignore:
        if not instance.polygon_xy:
            raise SceneTextAnnotationError(f"Empty ignored geometry in {path} {record}")
    elif len(instance.polygon_xy) < 3 or len(set(instance.polygon_xy)) < 3:
        raise SceneTextAnnotationError(f"Degenerate care polygon in {path} {record}")
    if not instance.transcription:
        raise SceneTextAnnotationError(f"Empty transcription in {path} {record}")
    if image_size_wh is not None:
        width, height = image_size_wh
        # Polygon annotations describe geometric edges, so a point may equal
        # width/height even though a pixel index may not.
        for x, y in instance.polygon_xy:
            if not (0 <= x <= width and 0 <= y <= height):
                raise SceneTextAnnotationError(
                    f"Out-of-bounds polygon point {(x, y)} in {path} {record} "
                    f"for image {width}x{height}"
                )
    return instance


def _fixed_numeric_prefix(
    row: list[str], count: int, *, path: Path, record: str
) -> tuple[list[int], list[str]]:
    if len(row) < count:
        raise SceneTextAnnotationError(
            f"Malformed record in {path} {record}: expected at least {count} numeric fields"
        )
    try:
        return [int(raw.strip()) for raw in row[:count]], row[count:]
    except ValueError as error:
        raise SceneTextAnnotationError(
            f"Non-integer coordinate in {path} {record}"
        ) from error


def parse_icdar2015_annotation(
    path: Path, *, image_size_wh: tuple[int, int] | None = None
) -> tuple[SceneTextInstance, ...]:
    """Parse official ICDAR2015 Challenge 4 word quadrilaterals.

    The transcription is the complete CSV remainder after the eight numeric
    coordinates, which preserves unquoted commas found in real annotations.
    """

    output: list[SceneTextInstance] = []
    for line_number, row in enumerate(
        csv.reader(_read_utf8_sig(path).splitlines(), skipinitialspace=True), start=1
    ):
        if not row or all(not field.strip() for field in row):
            continue
        numbers, remainder = _fixed_numeric_prefix(
            row, 8, path=path, record=f"line {line_number}"
        )
        if not remainder:
            raise SceneTextAnnotationError(
                f"Malformed ICDAR2015 record in {path} line {line_number}: "
                "expected 8 integer coordinates followed by transcription"
            )
        transcription = ",".join(remainder).strip()
        polygon = tuple((numbers[index], numbers[index + 1]) for index in range(0, 8, 2))
        output.append(
            _validate_instance(
                SceneTextInstance(
                    polygon_xy=polygon,
                    transcription=transcription,
                    ignore=transcription == "###",
                    granularity="word",
                ),
                path=path,
                record=f"line {line_number}",
                image_size_wh=image_size_wh,
            )
        )
    if not output:
        raise SceneTextAnnotationError(f"No ICDAR2015 records in {path}")
    return tuple(output)


_TOTALTEXT_PATTERN = re.compile(
    r"^\s*x\s*:\s*\[\[(?P<x>.*?)\]\]\s*,\s*"
    r"y\s*:\s*\[\[(?P<y>.*?)\]\]\s*,\s*"
    r"ornt\s*:\s*(?P<orientation>\[.*?\])\s*,\s*"
    r"transcriptions\s*:\s*(?P<text>\[.*\])\s*$"
)


def _literal_singleton(raw: str, *, path: Path, line_number: int, field: str) -> str:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise SceneTextAnnotationError(
            f"Malformed Total-Text {field} in {path} line {line_number}"
        ) from error
    if not isinstance(value, (list, tuple)) or len(value) != 1 or not isinstance(value[0], str):
        raise SceneTextAnnotationError(
            f"Total-Text {field} must be a one-string list in {path} line {line_number}"
        )
    return value[0]


def _totaltext_orientation(raw: str, *, path: Path, line_number: int) -> str | None:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise SceneTextAnnotationError(
            f"Malformed Total-Text orientation in {path} line {line_number}"
        ) from error
    if value == []:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise SceneTextAnnotationError(
        f"Total-Text orientation must be empty or a one-string list in {path} line {line_number}"
    )


def _totaltext_transcription(
    raw: str, *, orientation: str | None, path: Path, line_number: int
) -> str:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as error:
        raise SceneTextAnnotationError(
            f"Malformed Total-Text transcription in {path} line {line_number}"
        ) from error
    if value == [] and orientation == "#":
        return "#"
    if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise SceneTextAnnotationError(
        f"Total-Text transcription must be a one-string list in {path} line {line_number}"
    )


def parse_totaltext_annotation(
    path: Path, *, image_size_wh: tuple[int, int] | None = None
) -> tuple[SceneTextInstance, ...]:
    """Parse official Total-Text polygon TXT annotations (legacy or refined)."""

    output: list[SceneTextInstance] = []
    logical_records: list[tuple[int, str]] = []
    record_start = 0
    record_parts: list[str] = []
    for physical_line, raw_line in enumerate(_read_utf8_sig(path).splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("x:"):
            if record_parts:
                logical_records.append((record_start, " ".join(record_parts)))
            record_start = physical_line
            record_parts = [line]
        elif record_parts:
            record_parts.append(line)
        else:
            raise SceneTextAnnotationError(
                f"Total-Text continuation without x record in {path} line {physical_line}"
            )
    if record_parts:
        logical_records.append((record_start, " ".join(record_parts)))

    for line_number, line in logical_records:
        match = _TOTALTEXT_PATTERN.match(line)
        if match is None:
            raise SceneTextAnnotationError(
                f"Malformed Total-Text record in {path} line {line_number}"
            )
        xs = [int(value) for value in re.findall(r"-?\d+", match.group("x"))]
        ys = [int(value) for value in re.findall(r"-?\d+", match.group("y"))]
        if len(xs) != len(ys) or not xs:
            raise SceneTextAnnotationError(
                f"Total-Text coordinate count mismatch in {path} line {line_number}"
            )
        orientation = _totaltext_orientation(
            match.group("orientation"), path=path, line_number=line_number
        )
        transcription = _totaltext_transcription(
            match.group("text"),
            orientation=orientation,
            path=path,
            line_number=line_number,
        )
        if orientation != "#" and len(xs) < 3:
            raise SceneTextAnnotationError(
                f"Total-Text care polygon is degenerate in {path} line {line_number}"
            )
        output.append(
            _validate_instance(
                SceneTextInstance(
                    polygon_xy=tuple(zip(xs, ys)),
                    transcription=transcription,
                    ignore=orientation == "#" or transcription == "###",
                    granularity="word",
                    orientation=orientation,
                ),
                path=path,
                record=f"line {line_number}",
                image_size_wh=image_size_wh,
            )
        )
    if not output:
        raise SceneTextAnnotationError(f"No Total-Text records in {path}")
    return tuple(output)


def parse_ctw1500_test_annotation(
    path: Path, *, image_size_wh: tuple[int, int] | None = None
) -> tuple[SceneTextInstance, ...]:
    """Parse CTW1500 TXT, including the official 2019 end-to-end revision.

    The end-to-end archive starts each file with its instance count and stores
    14 absolute polygon points plus a quoted transcription.  The historical
    detection archive stores bbox metadata plus 14 relative polygon points.
    """

    rows = [
        row
        for row in csv.reader(_read_utf8_sig(path).splitlines(), skipinitialspace=True)
        if row and any(field.strip() for field in row)
    ]
    if rows and len(rows[0]) == 1 and rows[0][0].strip().isascii() and rows[0][0].strip().isdecimal():
        declared_count = int(rows[0][0].strip())
        output: list[SceneTextInstance] = []
        for line_number, row in enumerate(rows[1:], start=2):
            numbers, remainder = _fixed_numeric_prefix(
                row, 28, path=path, record=f"line {line_number}"
            )
            transcription = ",".join(remainder).strip() if remainder else ""
            missing_transcription = not transcription
            if missing_transcription:
                # Six records in the checksum-frozen official E2E archive have
                # geometry but no text.  Preserve their geometry and exclude
                # them from recognition rather than inventing a label.
                transcription = CTW1500_MISSING_TRANSCRIPTION
            points = tuple(
                (numbers[index], numbers[index + 1])
                for index in range(0, 28, 2)
            )
            if image_size_wh is not None:
                width, height = image_size_wh
                # Match the already-qualified CTW1500 segmentation preparer:
                # the upstream annotations contain a small number of points
                # just outside the raster extent.
                points = tuple(
                    (min(max(x, 0), width - 1), min(max(y, 0), height - 1))
                    for x, y in points
                )
            output.append(
                _validate_instance(
                    SceneTextInstance(
                        polygon_xy=points,
                        transcription=transcription,
                        ignore=transcription == "###" or missing_transcription,
                        granularity="text_line",
                    ),
                    path=path,
                    record=f"line {line_number}",
                    image_size_wh=image_size_wh,
                )
            )
        # Eight TRAIN files in the official archive declare one more instance
        # than they physically contain.  Qualification records this upstream
        # anomaly; the parser never fabricates the missing geometry.
        if not output:
            raise SceneTextAnnotationError(f"No CTW1500 E2E records in {path}")
        return tuple(output)

    output: list[SceneTextInstance] = []
    for line_number, row in enumerate(rows, start=1):
        numbers, remainder = _fixed_numeric_prefix(
            row, 32, path=path, record=f"line {line_number}"
        )
        x0, y0 = numbers[:2]
        offsets = numbers[4:32]
        polygon = tuple(
            (x0 + offsets[index], y0 + offsets[index + 1])
            for index in range(0, 28, 2)
        )
        if not remainder or not ",".join(remainder).strip():
            raise SceneTextAnnotationError(
                f"CTW1500 test record in {path} line {line_number} has no transcription; "
                "a detection-only annotation archive cannot qualify recognition"
            )
        transcription = ",".join(remainder).strip()
        output.append(
            _validate_instance(
                SceneTextInstance(
                    polygon_xy=polygon,
                    transcription=transcription,
                    ignore=transcription == "###",
                    granularity="text_line",
                ),
                path=path,
                record=f"line {line_number}",
                image_size_wh=image_size_wh,
            )
        )
    if not output:
        raise SceneTextAnnotationError(f"No CTW1500 test records in {path}")
    return tuple(output)


def _first_child_text(element: ET.Element, names: Iterable[str]) -> str | None:
    wanted = {name.lower() for name in names}
    for child in element.iter():
        if child.tag.rsplit("}", 1)[-1].lower() in wanted and child.text and child.text.strip():
            return child.text.strip()
    return None


def parse_ctw1500_train_xml(
    path: Path, *, image_size_wh: tuple[int, int] | None = None
) -> tuple[SceneTextInstance, ...]:
    """Parse official CTW1500 train XML boxes without assuming child order."""

    try:
        root = ET.fromstring(_read_utf8_sig(path))
    except ET.ParseError as error:
        raise SceneTextAnnotationError(f"Malformed CTW1500 XML {path}: {error}") from error
    boxes = [element for element in root.iter() if element.tag.rsplit("}", 1)[-1].lower() == "box"]
    output: list[SceneTextInstance] = []
    for index, box in enumerate(boxes, start=1):
        raw_points = _first_child_text(box, ("segs", "polygon", "points"))
        if raw_points is None:
            raise SceneTextAnnotationError(f"CTW1500 box {index} in {path} has no polygon")
        coordinates = [int(value) for value in re.findall(r"-?\d+", raw_points)]
        if len(coordinates) < 6 or len(coordinates) % 2:
            raise SceneTextAnnotationError(f"Malformed CTW1500 polygon in {path} box {index}")
        transcription = _first_child_text(box, ("label", "transcription", "text"))
        if transcription is None:
            raise SceneTextAnnotationError(
                f"CTW1500 box {index} in {path} has no transcription; "
                "a detection-only annotation archive cannot qualify recognition"
            )
        output.append(
            _validate_instance(
                SceneTextInstance(
                    polygon_xy=tuple(
                        (coordinates[pos], coordinates[pos + 1])
                        for pos in range(0, len(coordinates), 2)
                    ),
                    transcription=transcription,
                    ignore=transcription == "###",
                    granularity="text_line",
                ),
                path=path,
                record=f"box {index}",
                image_size_wh=image_size_wh,
            )
        )
    if not output:
        raise SceneTextAnnotationError(f"No CTW1500 boxes in {path}")
    return tuple(output)


__all__ = [
    "CTW1500_MISSING_TRANSCRIPTION",
    "SceneTextAnnotationError",
    "SceneTextInstance",
    "parse_ctw1500_test_annotation",
    "parse_ctw1500_train_xml",
    "parse_icdar2015_annotation",
    "parse_totaltext_annotation",
]
