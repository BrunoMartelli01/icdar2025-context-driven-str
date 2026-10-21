import json
from pathlib import Path

import numpy as np
import pytest
import torch

from recognition.crop_recovery_metric import evaluate_pool
from scripts.run_totaltext_crop_recovery_suite import (arm_definitions, freeze, measured_stage,
    read_tile, resolve_paths, score, validate_config)


REPO = Path(__file__).resolve().parents[2]


def rows_and_ocr(boxes, texts):
    rows = [{"candidate_id": str(i), "bbox": box, "crop_rgb_sha256": f"hash-{i}"}
            for i, box in enumerate(boxes)]
    readings = {str(i): {"raw_text": text} for i, text in enumerate(texts)}
    return rows, readings


def test_duplicate_predictions_count_gt_once_with_normalization():
    rows, readings = rows_and_ocr([[0, 0, 10, 10]] * 3, ["HELLO!", "hello", "HeLLo$"])
    result = evaluate_pool(rows, readings, [{"bbox": [0, 0, 10, 10], "text": "Hello"}], .5)
    assert result["correct"] == 1
    assert len(result["witnesses"]["0"]) == 3


def test_same_spelling_elsewhere_is_not_a_recovery():
    rows, readings = rows_and_ocr([[0, 0, 10, 10]], ["FOO"])
    words = [{"bbox": [0, 0, 10, 10], "text": "foo"},
             {"bbox": [20, 0, 30, 10], "text": "foo"}]
    assert evaluate_pool(rows, readings, words, .5)["correct_gt_indices"] == [0]


def test_one_crop_cannot_count_two_repeated_overlapping_annotations():
    rows, readings = rows_and_ocr([[0, 0, 10, 10]], ["x"])
    words = [{"bbox": [0, 0, 10, 10], "text": "x"}] * 2
    result = evaluate_pool(rows, readings, words, .5)
    assert result["correct"] == 1
    assert result["any_crop_correct_gt_indices"] == [0, 1]


def test_normalization_preserves_digits_accents_and_spaces_and_guards_empty():
    words = [{"bbox": [0, 0, 10, 10], "text": "Été-42"},
             {"bbox": [20, 0, 30, 10], "text": "#"},
             {"bbox": [40, 0, 50, 10], "text": "a b"}]
    rows, readings = rows_and_ocr([w["bbox"] for w in words], ["ÉTÉ42!", "$", "ab"])
    assert evaluate_pool(rows, readings, words, .5)["correct_gt_indices"] == [0]


def test_iou_boundary_and_incorrect_text_cannot_count():
    rows, readings = rows_and_ocr([[0, 0, 20, 10]], ["word"])
    words = [{"bbox": [0, 0, 10, 10], "text": "word"}]
    assert evaluate_pool(rows, readings, words, .5)["correct"] == 1
    assert evaluate_pool(rows, readings, words, .50001)["correct"] == 0
    readings["0"]["raw_text"] = "ward"
    assert evaluate_pool(rows, readings, words, .5)["correct"] == 0


def test_cumulative_arms_keep_all_baseline_hypotheses():
    arms = arm_definitions(True)
    assert set(arms["all_methods"]) == {"baseline", "threshold_tree", "erosion_reconstruction",
                                       "image_seams", "tiled_segmentation", "crop_variants"}
    for name, methods in arms.items():
        if name.startswith(("cumulative_", "baseline_plus_")):
            assert "baseline" in methods
    assert "tiled_segmentation" not in arm_definitions(False)["all_methods"]


def test_strict_mode_and_invalid_scientific_parameters_fail():
    config = json.loads((REPO / "configs/approximate/totaltext_crop_recovery_suite_train_v1.json").read_text())
    validate_config(config)
    config["reproduction_mode"] = "strict"
    with pytest.raises(ValueError, match="approximate"):
        validate_config(config)
    config["reproduction_mode"] = "approximate"
    config["threshold_tree"]["seed_pairs"] = [[.8, .4]]
    with pytest.raises(ValueError, match="seed"):
        validate_config(config)


def test_config_profile_resolves_paths_and_overrides(tmp_path):
    profile = json.loads((REPO / "configs/approximate/totaltext_crop_recovery_local_paths_v1.json").read_text())
    path = tmp_path / "paths.json"
    path.write_text(json.dumps(profile))
    actual = resolve_paths(path, {"data_root": tmp_path / "data"})
    assert actual["data_root"] == tmp_path / "data"
    assert actual["probability_cache"].is_absolute()


def test_resume_refuses_changed_contract(tmp_path):
    path = tmp_path / "resolved.json"
    freeze(path, {"device": "cpu"})
    freeze(path, {"device": "cpu"})
    with pytest.raises(RuntimeError, match="differs"):
        freeze(path, {"device": "cuda:0"})


def test_corrupt_tile_cache_fails(tmp_path):
    from scripts.crop_clip_runtime import sha
    path = tmp_path / "probability.npy"
    np.save(path, np.zeros((256, 256), np.float32))
    record = {"path": path.name, "shape": [256, 256], "file_sha256": sha(path)}
    assert read_tile(tmp_path, record).shape == (256, 256)
    np.save(path, np.ones((256, 256), np.float32))
    with pytest.raises(RuntimeError, match="hash"):
        read_tile(tmp_path, record)


def test_probability_sidecar_uses_canonical_protocol_hash(tmp_path):
    from hashlib import sha256
    from types import SimpleNamespace
    from scripts.crop_clip_runtime import sha
    from scripts.evaluate_totaltext_letterbox_threshold_sweep_train import _load_probability
    protocol = {"split": "TRAIN", "device": "cuda:0"}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    canonical = sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert canonical != sha(protocol_path)
    p = np.zeros((256, 256), np.float32)
    path = tmp_path / "probability.npy"
    np.save(path, p)
    sample = SimpleNamespace(sample_id="total_text:1", image_sha256="image")
    metadata = {"sample_id": sample.sample_id, "image_sha256": sample.image_sha256,
                "protocol_sha256": canonical, "file_sha256": sha(path),
                "array_sha256": sha256(p.tobytes()).hexdigest()}
    assert np.array_equal(_load_probability(path, metadata, sample=sample, protocol_sha=canonical), p)
    with pytest.raises(ValueError, match="provenance"):
        _load_probability(path, metadata, sample=sample, protocol_sha=sha(protocol_path))


def test_oom_failure_is_persisted_and_propagated(tmp_path):
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with measured_stage(tmp_path, "parseq", "cpu", {"batch_size": 16}):
            raise torch.cuda.OutOfMemoryError("injected boundary failure")
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["out_of_memory"] is True and failure["stage"] == "parseq"
    assert failure["details"]["batch_size"] == 16
    assert failure["result_class"] == "FAILED_RUN"


def test_score_accepts_frozen_aggregate_schema_and_refuses_gt_drift(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts import run_totaltext_crop_recovery_suite as runner
    sample = SimpleNamespace(sample_id="total_text:1")
    gt = [{"bbox": [0, 0, 10, 10], "text": "Foo!"}]
    rows, readings = rows_and_ocr([[0, 0, 10, 10]] * 2, ["FOO", "foo"])
    rows[0]["methods"] = ["baseline", "threshold_tree"]
    rows[1]["methods"] = ["threshold_tree"]
    for directory in ("prepared", "ocr"):
        (tmp_path / directory).mkdir()
        value = {"model_peak": {"peak_allocated_bytes": None, "peak_reserved_bytes": None}}
        (tmp_path / directory / "total_text__1.json").write_text(json.dumps(value))
    (tmp_path / "resolved_configuration.json").write_text("{}")
    monkeypatch.setattr(runner, "_image", lambda *_: np.zeros((10, 10, 3), np.uint8))
    monkeypatch.setattr(runner, "_train_gt", lambda *_: (gt, "annotation-file-hash"))
    reference = {"thresholds": {"0.50": {"per_image": [{"sample_id": sample.sample_id, "ground_truth": gt}]}}}
    config = json.loads((REPO / "configs/approximate/totaltext_crop_recovery_suite_train_v1.json").read_text())
    result = score([sample], {sample.sample_id: {"candidates": rows}}, {sample.sample_id: readings},
                   {"data_root": tmp_path}, config, tmp_path, {"result_class": "SMOKE_TEST"}, False, reference)
    assert result["summary"]["all_methods"]["correct"] == 1
    assert result["summary"]["all_methods"]["gt_words"] == 1
    reference["thresholds"]["0.50"]["per_image"][0]["ground_truth"] = [{"bbox": [0, 0, 10, 10], "text": "bar"}]
    with pytest.raises(RuntimeError, match="annotations changed"):
        score([sample], {sample.sample_id: {"candidates": rows}}, {sample.sample_id: readings},
              {"data_root": tmp_path}, config, tmp_path, {"result_class": "SMOKE_TEST"}, False, reference)
