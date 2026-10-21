"""CPU-only Polygon3 ignore-region intersections; no images or model imports.

Runs in an existing isolated Python environment that provides Polygon3 and
Shapely. Native Polygon3 inputs mirror the benchmark evaluator. Shapely is an
independent intersection audit, never a replacement for the primary engine.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import traceback


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def neutralization_rows(image, threshold):
    import Polygon
    from shapely import make_valid
    from shapely.geometry import Polygon as ShapelyPolygon, Point, LineString, box
    from shapely.validation import explain_validity

    candidate_ids = [row["candidate_id"] for row in image["rows"]]
    if (any(not isinstance(cid, str) or not cid for cid in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)):
        raise ValueError("geometry candidate IDs must be unique nonempty strings")

    polygons, references, annotations = [], [], []
    for index, region in enumerate(image["ignored_regions"]):
        points = region["polygon_xy"]
        if (not points or any(len(p) != 2 or any(type(v) not in (int, float)
            or not math.isfinite(v) for v in p) for p in points)):
            raise ValueError("ignored polygon must contain finite coordinate pairs")
        polygons.append(Polygon.Polygon(points))
        if len(points) == 1:
            reference = Point(points[0])
        elif len(points) == 2:
            reference = LineString(points)
        else:
            reference = ShapelyPolygon(points)
        invalid = not reference.is_valid
        reason = explain_validity(reference)
        if invalid:
            reference = make_valid(reference)
        references.append(reference)
        if invalid or reference.area == 0:
            annotations.append({"ignored_region_index": index, "invalid_raw_polygon": invalid,
                "raw_validity": reason, "audit_geometry_type": reference.geom_type,
                "audit_zero_area": reference.area == 0,
                "primary_input_changed": False})
    results, comparisons = [], 0
    for row in image["rows"]:
        a = row["bbox"]
        if (len(a) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in a)
            or a[2] <= a[0] or a[3] <= a[1]):
            raise ValueError("predicted rectangle must have finite positive area")
        primary = Polygon.Polygon([[a[0], a[1]], [a[2], a[1]], [a[2], a[3]], [a[0], a[3]]])
        reference_box = box(*a)
        area = (a[2] - a[0]) * (a[3] - a[1])
        matching, overlaps = [], []
        for index, (polygon, reference) in enumerate(zip(polygons, references)):
            primary_area = (primary & polygon).area()
            independent_area = reference_box.intersection(reference).area
            if not math.isclose(primary_area, independent_area, rel_tol=1e-9, abs_tol=1e-7):
                raise RuntimeError("intersection audit differs for {} region {}: Polygon3={} Shapely={}".format(
                    row["candidate_id"], index, primary_area, independent_area))
            coverage = primary_area / area
            independent_coverage = independent_area / area
            if (coverage > threshold) != (independent_coverage > threshold):
                raise RuntimeError("ignore boundary audit differs for {} region {}".format(row["candidate_id"], index))
            if not 0 <= coverage <= 1 + 1e-9:
                raise RuntimeError("intersection coverage is outside [0,1]")
            overlaps.append(coverage)
            if coverage > threshold:
                matching.append(index)
            comparisons += 1
        results.append({"candidate_id": row["candidate_id"], "neutralized": bool(matching),
            "matching_ignored_region_indices": matching,
            "maximum_ignored_coverage": max(overlaps, default=0.)})
    return {"sample_id": image["sample_id"], "rows": results,
        "intersection_comparisons": comparisons, "geometry_annotations": annotations}


def evaluate_request(request):
    if (request.get("schema_version") != 1 or request.get("reproduction_mode") != "approximate"
        or request.get("phase") != "evaluation_only" or request.get("images_modified") is not False):
        raise ValueError("requires explicit approximate evaluation-only polygon request")
    threshold = request.get("ignored_prediction_coverage_threshold")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("ignore coverage threshold must be finite in (0,1)")
    versions = {name: importlib.metadata.version(name) for name in ("Polygon3", "Shapely")}
    if versions != request["expected_versions"]:
        raise RuntimeError("polygon runtime versions differ from configured identity")
    ids = [image["sample_id"] for image in request["images"]]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate geometry sample IDs")
    import Polygon.cPolygon
    import shapely.lib
    results = [neutralization_rows(image, threshold) for image in request["images"]]
    return {"schema_version": 1, "status": "PASS", "images": results,
        "runtime": {"python": sys.version, "platform": platform.platform(), "versions": versions,
            "Polygon3_extension_sha256": file_sha(Polygon.cPolygon.__file__),
            "Shapely_extension_sha256": file_sha(shapely.lib.__file__)},
        "intersection_comparisons": sum(image["intersection_comparisons"] for image in results),
        "models_loaded": False, "images_modified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = evaluate_request(json.loads(args.request.read_text(encoding="utf-8")))
        result["request_sha256"] = file_sha(args.request)
        args.response.write_text(json.dumps(result, allow_nan=False, sort_keys=True, indent=2) + "\n")
    except Exception as error:
        args.response.write_text(json.dumps({"status": "FAILED", "error": str(error),
            "error_type": type(error).__name__, "traceback": traceback.format_exc()}, indent=2) + "\n")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
