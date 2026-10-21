from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json

import pytest
import torch

from scripts import evaluate_word_centerline_ocr as module
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct
from scripts.evaluate_word_multitask_crops import ratios, score_predictions
from scripts.prune_segmentation_checkpoints import sha256_file


@pytest.fixture
def completed(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    source = repo / "scientific.py"
    source.write_text("# qualified fixture\n")
    monkeypatch.setattr(module, "REPO", repo)
    root = tmp_path / "run"
    root.mkdir()
    (root / "best.pt").write_bytes(b"selected fixture checkpoint")
    image = tmp_path / "image"
    image.write_bytes(b"qualified RGB fixture")
    annotation = tmp_path / "annotation"
    annotation.write_bytes(b"qualified annotation fixture")
    (root / "proposals.json").write_text("{}")
    sid = "total_text:1"
    split = sha256((sid + "\n").encode()).hexdigest()
    cfg = {"epochs": 20, "seed": 202503, "validation_count": 1,
           "expected_validation_words": 3, "validation_ids_sha256": split}
    contract = {"reproduction_mode": "approximate", "result_class": "approximate reproduction",
                "resolved_config": cfg, "source_sha256": {"scientific.py": sha256_file(source)},
                "validation_ids_sha256": split}
    checkpoint = {"path": "best.pt", "sha256": sha256_file(root / "best.pt")}
    report = {"status": "SUCCESS", "training_complete": True, "completed_epochs": 20,
              "result_class": "approximate reproduction", "run_contract": contract, "checkpoint": checkpoint}
    gt = [{"bbox": [0, 0, 10, 10], "text": "One"},
          {"bbox": [20, 0, 30, 10], "text": "Two!"},
          {"bbox": [40, 0, 50, 10], "text": "Three"}]
    proposals = [{"instance_id": i + 1, "bbox": g["bbox"], "crop_rgb_sha256": str(i) * 64}
                 for i, g in enumerate(gt)]
    row = {"sample_id": sid, "image_path": str(image), "image_sha256": sha256_file(image),
           "annotation_path": str(annotation), "annotation_sha256": sha256_file(annotation),
           "ground_truth": gt, "proposals": proposals, "ignored_prediction_ids": [3],
           "exports": {"proposals.json": sha256_file(root / "proposals.json")}}
    geometry = {"status": "SUCCESS", "reproduction_mode": "approximate", "result_class": "approximate reproduction",
                "protocol": {"run_contract": contract, "checkpoint_sha256": checkpoint["sha256"],
                             "source_seed": cfg["seed"], "expected_images": 1, "expected_care_words": 3},
                "branches": {module.BRANCH: {method: {"native": {}, "per_image": [deepcopy(row)]}
                                              for method in module.METHODS}}}
    (root / "run_manifest.json").write_text(json.dumps(report))
    (root / "geometry_results.json").write_text(json.dumps(geometry))
    return root, report, geometry


def recognized(geometry):
    readings = {"baseline": ["ONE", "bad", "Three"],
                "line_markers": ["bad", "two!", "Three"],
                "line_radius": ["one", "two", "Three"]}
    result = {"status": "SUCCESS", "result_class": "approximate reproduction", "branches": {module.BRANCH: {}}}
    for method in module.METHODS:
        rows = []
        total = Counter()
        for source in geometry["branches"][module.BRANCH][method]["per_image"]:
            predictions = [{**p, "text": text} for p, text in zip(source["proposals"], readings[method])]
            care = [p for p in predictions if p["instance_id"] not in source["ignored_prediction_ids"]]
            counts = score_predictions(care, source["ground_truth"], text_available=True)
            total.update(counts)
            rows.append({"sample_id": source["sample_id"], "ground_truth": source["ground_truth"],
                         "predictions_all": predictions, "ignored_prediction_ids": source["ignored_prediction_ids"],
                         "counts": counts, "correct_casefold_gt_indices": sorted(_correct(care, source["ground_truth"], normalize=False))})
        result["branches"][module.BRANCH][method] = {"aggregate": ratios(dict(total)), "per_image": rows}
    return result


def test_complete_training_checkpoint_split_and_exports_qualify(completed):
    root, report, geometry = completed
    assert module.qualify_run(root, "approximate") == (report, geometry)
    with pytest.raises(ValueError, match="APPROXIMATE"):
        module.qualify_run(root, "strict")


@pytest.mark.parametrize("change", [{"status": "IN_PROGRESS"}, {"training_complete": False},
                                    {"completed_epochs": 19}, {"completed_epochs": True}])
def test_incomplete_training_is_rejected_before_recognition(completed, change):
    root, report, _ = completed
    report.update(change)
    (root / "run_manifest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Incomplete"):
        module.qualify_run(root, "approximate")


@pytest.mark.parametrize("name,pattern", [("best.pt", "checkpoint changed"),
                                          ("proposals.json", "export checksum differs")])
def test_changed_checkpoint_and_export_fail(completed, name, pattern):
    root, _, _ = completed
    (root / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match=pattern):
        module.qualify_run(root, "approximate")


def test_changed_scientific_code_fails(completed):
    root, _, _ = completed
    (module.REPO / "scientific.py").write_text("# changed\n")
    with pytest.raises(ValueError, match="scientific source changed"):
        module.qualify_run(root, "approximate")


@pytest.mark.parametrize("case", ["checkpoint", "seed", "missing_image", "different_gt", "unknown_ignored"])
def test_geometry_population_and_provenance_are_not_guessed(completed, case):
    root, _, geometry = completed
    row = geometry["branches"][module.BRANCH]["line_radius"]["per_image"][0]
    if case == "checkpoint":
        geometry["protocol"]["checkpoint_sha256"] = "wrong"
    elif case == "seed":
        geometry["protocol"]["source_seed"] += 1
    elif case == "missing_image":
        geometry["branches"][module.BRANCH]["line_radius"]["per_image"] = []
    elif case == "different_gt":
        row["ground_truth"][0]["text"] = "modified"
    else:
        row["ignored_prediction_ids"] = [99]
    (root / "geometry_results.json").write_text(json.dumps(geometry))
    with pytest.raises(ValueError):
        module.qualify_run(root, "approximate")


def test_pairing_tracks_word_identity_even_when_net_gain_is_zero(completed):
    _, _, geometry = completed
    result = module.compare_recognition(geometry, recognized(geometry))
    paired = result["paired_vs_baseline"]["line_markers"]
    assert (paired["gained_count"], paired["lost_count"], paired["net_correct_words"]) == (1, 1, 0)
    assert paired["gained"][0]["gt_index"] == 1
    assert paired["gained"][0]["gt_text"] == "Two!"
    assert paired["lost"][0]["gt_index"] == 0
    # Wholly ignored predictions do not count, but every GT remains in recall.
    assert result["methods"]["baseline"]["gt_words"] == 3
    assert result["methods"]["baseline"]["proposals"] == 2
    assert result["methods"]["baseline"]["exact_casefold_recall_all_gt"] == pytest.approx(1 / 3)
    # Punctuation stripping is secondary and cannot create a primary gain.
    assert result["methods"]["line_radius"]["exact_casefold_iou_ge_50"] == 1
    assert result["methods"]["line_radius"]["exact_no_symbols_iou_ge_50"] == 2
    assert "+1 / -1" in module.markdown_report(result)


@pytest.mark.parametrize("case", ["crop", "count", "matching", "aggregate", "missing_image"])
def test_cached_recognition_is_recomputed_against_frozen_geometry(completed, case):
    _, _, geometry = completed
    result = recognized(geometry)
    part = result["branches"][module.BRANCH]["baseline"]
    row = part["per_image"][0]
    if case == "crop":
        row["predictions_all"][0]["bbox"] = [100, 100, 110, 110]
    elif case == "count":
        row["counts"]["exact_casefold_iou_ge_50"] += 1
    elif case == "matching":
        row["correct_casefold_gt_indices"] = [2]
    elif case == "aggregate":
        part["aggregate"]["exact_casefold_precision"] = 1.0
    else:
        part["per_image"] = []
    with pytest.raises(ValueError):
        module.compare_recognition(geometry, result)


def test_ocr_failure_and_oom_propagate_with_measurement_context(completed, monkeypatch):
    root, _, _ = completed
    monkeypatch.setattr(module, "OCR_SOURCES", ("scientific.py",))
    monkeypatch.setattr(module, "verify_parseq_runtime_source", lambda *a, **k: {"qualified_fixture": True})
    import recognition.parseq as parseq
    monkeypatch.setattr(parseq, "PARSEQ_CHECKPOINT_SHA256", sha256_file(root / "best.pt"))

    def fail(*args):
        raise torch.cuda.OutOfMemoryError("synthetic OOM propagation test")

    monkeypatch.setattr(module, "recognize", fail)
    argv = ["--mode", "approximate", "--run-root", str(root), "--device", "cpu",
            "--parseq-source", str(root), "--parseq-source-archive", str(root / "best.pt"),
            "--parseq-checkpoint", str(root / "best.pt")]
    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic OOM"):
        module.main(argv)
    failure = json.loads((root / "centerline_ocr_failure.json").read_text())
    assert failure["oom"] is True
    assert failure["input_shape"] == [8, 3, 32, 128]
    assert failure["gpu_peak_allocated_bytes"] is None
    assert not (root / "centerline_ocr_comparison.json").exists()
