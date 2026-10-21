import copy
import json
from pathlib import Path
import subprocess

import pytest

from recognition.dont_care_evaluation import apply_neutralization
from scripts.evaluate_ctw1500_word_recovery import select_predictions
from scripts.evaluate_yolo_dont_care import (DEFAULT_CONFIG, REPO, run_geometry, score_words,
    validate_config)

PYTHON = REPO / ".venvs/context-str-deepsolo-host/bin/python"
WORKER = REPO / "scripts/dont_care_polygon_worker.py"


def prediction(cid="p", text="Word!", bbox=(0, 0, 10, 10), confidence=.9):
    return {"candidate_id": cid, "raw_text": text, "bbox": list(bbox),
        "detector_score": confidence, "token_probabilities": [1.] * len(text),
        "token_log_probabilities": [0.] * len(text), "crop_rgb_sha256": "frozen_pixels"}


def request(images):
    return {"schema_version": 1, "reproduction_mode": "approximate", "phase": "evaluation_only",
        "images_modified": False, "ignored_prediction_coverage_threshold": .5,
        "expected_versions": {"Polygon3": "3.0.9.1", "Shapely": "2.0.0"}, "images": images}


def image(sid, polygons, boxes=((0, 0, 10, 10),)):
    return {"sample_id": sid, "ignored_regions": [{"polygon_xy": points} for points in polygons],
        "rows": [{"candidate_id": f"p{i}", "bbox": list(b)} for i, b in enumerate(boxes)]}


@pytest.fixture(scope="module")
def polygon_examples(tmp_path_factory):
    output = tmp_path_factory.mktemp("polygon-examples")
    examples = [
        image("half", [[[0, 0], [5, 0], [5, 10], [0, 10]]]),
        image("over_half", [[[0, 0], [6, 0], [6, 10], [0, 10]]]),
        image("huge_region", [[[-100, -100], [100, -100], [100, 100], [-100, 100]]]),
        # A triangle's bbox covers the whole prediction, its polygon covers 50%.
        image("triangle", [[[0, 0], [10, 0], [0, 10]]]),
        # An L polygon's bbox is also the full box, but its area is only 36%.
        image("concave", [[[0, 0], [10, 0], [10, 2], [2, 2], [2, 10], [0, 10]]]),
        image("no_union", [[[0, 0], [4, 0], [4, 10], [0, 10]],
                           [[6, 0], [10, 0], [10, 10], [6, 10]]]),
        image("degenerate", [[[1, 1]], [[1, 1], [5, 5]], [[1, 1], [2, 2], [3, 3]]]),
        image("bowtie", [[[0, 0], [10, 10], [0, 10], [10, 0]]]),
        # Original out-of-frame vertices: clamp would make a larger in-box shape.
        image("unclamped", [[[-10, 0], [10, 0], [10, 10]]]),
        image("none", []), image("empty_predictions", [], boxes=[]),
    ]
    return run_geometry(request(examples), output, PYTHON, 30)


@pytest.mark.parametrize("sid,ignored,coverage", [
    ("half", False, .5), ("over_half", True, .6), ("huge_region", True, 1.),
    ("triangle", False, .5), ("concave", False, .36), ("no_union", False, .4),
    ("degenerate", False, 0.), ("bowtie", False, .5), ("unclamped", True, .75),
    ("none", False, 0.),
])
def test_native_polygon_fraction_strict_boundary_and_independent_audit(polygon_examples, sid, ignored, coverage):
    result = next(im for im in polygon_examples["images"] if im["sample_id"] == sid)["rows"][0]
    assert result["neutralized"] is ignored
    assert result["maximum_ignored_coverage"] == pytest.approx(coverage)
    assert polygon_examples["models_loaded"] is False
    assert polygon_examples["images_modified"] is False


def test_invalid_and_degenerate_polygons_are_logged_without_replacing_primary_inputs(polygon_examples):
    images = {im["sample_id"]: im for im in polygon_examples["images"]}
    assert len(images["degenerate"]["geometry_annotations"]) == 3
    note = images["bowtie"]["geometry_annotations"][0]
    assert note["invalid_raw_polygon"] is True
    assert note["audit_geometry_type"] == "MultiPolygon"
    assert note["primary_input_changed"] is False
    assert images["empty_predictions"]["rows"] == []
    assert polygon_examples["runtime"]["versions"] == {"Polygon3": "3.0.9.1", "Shapely": "2.0.0"}


@pytest.mark.parametrize("mutation", ["strict", "pixels_modified", "duplicate_ids", "zero_box", "wrong_versions"])
def test_worker_propagates_invalid_contract_failures(tmp_path, mutation):
    data = request([image("one", [])])
    if mutation == "strict":
        data["reproduction_mode"] = "strict"
    elif mutation == "pixels_modified":
        data["images_modified"] = True
    elif mutation == "duplicate_ids":
        data["images"][0]["rows"] *= 2
    elif mutation == "zero_box":
        data["images"][0]["rows"][0]["bbox"] = [0, 0, 0, 10]
    else:
        data["expected_versions"]["Polygon3"] = "unqualified"
    ip, op = tmp_path / "request.json", tmp_path / "response.json"
    ip.write_text(json.dumps(data))
    process = subprocess.run([str(PYTHON), str(WORKER), "--request", str(ip), "--response", str(op)],
        capture_output=True, text=True, timeout=30)
    assert process.returncode != 0
    result = json.loads(op.read_text())
    assert result["status"] == "FAILED" and result["traceback"]


def test_neutralizing_a_false_positive_changes_precision_without_changing_readings_or_recall():
    rows = [prediction("care"), prediction("ignore", text="unreadable", bbox=(20, 0, 30, 10))]
    original = copy.deepcopy(rows)
    geometry = [{"candidate_id": "care", "neutralized": False}, {"candidate_id": "ignore", "neutralized": True}]
    evaluated, ignored = apply_neutralization(rows, geometry)
    assert rows == original and evaluated[0] is rows[0] and ignored[0] is rows[1]
    gt = [{"text": "word", "bbox": [0, 0, 10, 10]}]
    config = json.loads(DEFAULT_CONFIG.read_text())["evaluation"]
    before = score_words("total_text", rows, gt, config)
    after = score_words("total_text", evaluated, gt, config)
    assert before["correct_words"] == after["correct_words"] == 1
    assert before["precision"] == .5 and after["precision"] == 1
    assert before["recall"] == after["recall"] == 1
    assert after["fp"] == after["fn"] == 0


def test_ignore_wins_over_care_match_and_no_prediction_is_resurrected_after_nms():
    rows = [prediction("high", confidence=.9), prediction("low", confidence=.8)]
    original = copy.deepcopy(rows)
    config = json.loads(DEFAULT_CONFIG.read_text())
    _, selected = select_predictions(rows, config["selection"], None)
    assert [r["candidate_id"] for r in selected] == ["high"]
    geometry = [{"candidate_id": "high", "neutralized": True}, {"candidate_id": "low", "neutralized": False}]
    evaluated, ignored = apply_neutralization(selected, geometry)
    assert evaluated == [] and len(ignored) == 1
    gt = [{"text": "word", "bbox": [0, 0, 10, 10]}]
    assert score_words("icdar13", selected, gt, config["evaluation"])["correct_words"] == 1
    assert score_words("icdar13", evaluated, gt, config["evaluation"])["correct_words"] == 0
    assert rows == original


def test_ctw_counts_words_and_neutralizes_all_tokens_of_ignored_prediction():
    rows = [prediction("care", "GO"), prediction("ignored", "GO GO")]
    gt = [{"text": "GO GO", "bbox": [0, 0, 10, 10]}]
    geometry = [{"candidate_id": "care", "neutralized": False}, {"candidate_id": "ignored", "neutralized": True}]
    evaluated, _ = apply_neutralization(rows, geometry)
    config = json.loads(DEFAULT_CONFIG.read_text())["evaluation"]
    before = score_words("ctw1500", rows, gt, config)
    after = score_words("ctw1500", evaluated, gt, config)
    assert (before["correct_words"], before["predicted_words"]) == (2, 3)
    assert (after["correct_words"], after["predicted_words"], after["gt_words"]) == (1, 1, 2)


@pytest.mark.parametrize("rows,geometry", [
    ([prediction()], []), ([prediction()], [{"candidate_id": "p", "neutralized": 1}]),
    ([prediction(), prediction()], [{"candidate_id": "p", "neutralized": False}]),
    ([prediction()], [{"candidate_id": "p", "neutralized": False}] * 2),
])
def test_missing_duplicate_or_nonbool_geometry_decisions_fail(rows, geometry):
    with pytest.raises(ValueError):
        apply_neutralization(rows, geometry)


@pytest.mark.parametrize("key,value", [("ignored_prediction_coverage_threshold", 0),
    ("ignored_prediction_coverage_threshold", True), ("ignored_prediction_coverage_threshold", .4),
    ("word_bbox_matching_iou", float("nan"))])
def test_config_rejects_changed_or_invalid_metric_thresholds(key, value):
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    validate_config(cfg)
    cfg["evaluation"][key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_strict_config_and_selection_changes_are_rejected():
    for key, value in [("reproduction_mode", "strict"), ("table_5_claim", True)]:
        cfg = json.loads(DEFAULT_CONFIG.read_text())
        cfg[key] = value
        with pytest.raises(ValueError):
            validate_config(cfg)
    cfg = json.loads(DEFAULT_CONFIG.read_text())
    cfg["selection"]["joint_threshold"] = .7
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_empty_population_metrics_are_explicit():
    config = json.loads(DEFAULT_CONFIG.read_text())["evaluation"]
    for name in ("ctw1500", "icdar13", "icdar2015", "total_text"):
        score = score_words(name, [], [], config)
        assert score["correct_words"] == score["predicted_words"] == score["gt_words"] == 0
        assert score["precision"] == score["recall"] == score["hmean"] == 0
