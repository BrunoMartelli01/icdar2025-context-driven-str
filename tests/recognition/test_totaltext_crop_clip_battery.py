from types import SimpleNamespace

import numpy as np
import pytest

from recognition.crop_clip_battery import (
    candidate_scope, generate_totaltext_candidates, match_anchors,
    select_per_anchor, totaltext_oracle_one_per_anchor,
)
from scripts.run_totaltext_crop_clip_battery import (
    _ocr_record, serializable_selections, validate_config,
)
from scripts import crop_clip_runtime as clip_runtime


def test_native_padding_thresholds_and_exact_pad0_baseline() -> None:
    image = np.zeros((512, 512, 3), dtype=np.uint8)
    probability = np.zeros((256, 256), dtype=np.float32)
    probability[90:96, 70:74] = 0.9
    probability[90:96, 78:82] = 0.9
    probability[92:94, 74:78] = 0.4
    rows, stats = generate_totaltext_candidates(
        probability, image, thresholds=[0.35, 0.5, 0.65], pads=[0, 2, 4, 8])
    bases = [row for row in rows if row["baseline"]]
    assert len(bases) == 2
    assert stats["baseline_anchors"] == 2
    assert len({(row["anchor_rank"], tuple(row["bbox"])) for row in rows}) == len(rows)
    assert all(row["origins"][0] == {"threshold": 0.5,
                                      "padding_native_pixels": 0,
                                      "matched_tight_native_iou": 1.0} for row in bases)
    assert all(row["bbox"] == [2 * x for x in row["tight_bbox_native"]]
               for row in bases)
    assert any(any(origin["threshold"] == 0.35 for origin in row["origins"])
               for row in rows)


def test_threshold_matching_one_to_one_and_unmatched_count() -> None:
    baseline = [SimpleNamespace(rank=1, bbox=(0, 0, 10, 10)),
                SimpleNamespace(rank=2, bbox=(10, 0, 20, 10))]
    alternative = [SimpleNamespace(rank=1, bbox=(0, 0, 20, 10))]
    assert match_anchors(baseline, alternative, 0.3) == {1: (1, 0.5)}


def _rows():
    return [
        {"anchor_rank": 1, "candidate_id": "base", "baseline": True,
         "bbox": [0, 0, 10, 10], "origins": [{"threshold": 0.5, "padding_native_pixels": 0}],
         "ocr": {"raw_text": "wrong"}, "scores": {"written": .20}},
        {"anchor_rank": 1, "candidate_id": "pad", "baseline": False,
         "bbox": [0, 0, 12, 10], "origins": [{"threshold": 0.5, "padding_native_pixels": 2}],
         "ocr": {"raw_text": "right"}, "scores": {"written": .23}},
        {"anchor_rank": 1, "candidate_id": "threshold", "baseline": False,
         "bbox": [0, 0, 12, 9], "origins": [{"threshold": 0.65, "padding_native_pixels": 0}],
         "ocr": {"raw_text": "right"}, "scores": {"written": .21}},
    ]


def test_totaltext_scopes_margin_and_ties_keep_frozen_baseline() -> None:
    rows = _rows()
    assert select_per_anchor(rows, score="written", scope="all") == {1: "pad"}
    assert select_per_anchor(rows, score="written", scope="padding") == {1: "pad"}
    assert select_per_anchor(rows, score="written", scope="threshold") == {1: "threshold"}
    assert select_per_anchor(rows, score="written", scope="all", margin=.04) == {1: "base"}
    rows[1]["scores"]["written"] = .20
    assert select_per_anchor(rows, score="written", scope="padding") == {1: "base"}
    assert candidate_scope(rows[2], "threshold")
    assert not candidate_scope(rows[2], "padding")


def test_oracle_one_crop_per_anchor_and_iou_constraint() -> None:
    rows = _rows()
    words = [{"bbox": [0, 0, 10, 10], "text": "right"},
             {"bbox": [30, 30, 40, 40], "text": "missing"}]
    assert totaltext_oracle_one_per_anchor(rows, words) == {1: "pad"}
    rows[1]["bbox"] = [20, 20, 30, 30]
    assert totaltext_oracle_one_per_anchor(rows, words) == {1: "threshold"}


def test_oracle_uses_maximum_matching_across_overlapping_words() -> None:
    rows = [
        {"anchor_rank": 1, "candidate_id": "first", "baseline": True,
         "bbox": [5, 0, 10, 10], "ocr": {"raw_text": "WORD"}},
        {"anchor_rank": 2, "candidate_id": "second", "baseline": True,
         "bbox": [0, 0, 10, 10], "ocr": {"raw_text": "WORD"}},
    ]
    words = [{"bbox": [0, 0, 10, 10], "text": "WORD"},
             {"bbox": [5, 0, 15, 10], "text": "WORD"}]
    choices = totaltext_oracle_one_per_anchor(rows, words)
    assert choices == {1: "first", 2: "second"}
    from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _correct
    assert len(_correct(rows_to_predictions(rows, choices), words, normalize=False)) == 2


def rows_to_predictions(rows, choices):
    return [{"bbox": row["bbox"], "text": row["ocr"]["raw_text"]}
            for row in rows if choices[row["anchor_rank"]] == row["candidate_id"]]


def test_cached_parseq_rejects_mutated_provenance(tmp_path) -> None:
    path = tmp_path / "crop.json"
    path.write_text('{"crop_rgb_sha256":"x","parseq_checkpoint_sha256":"bad",'
                    '"parseq_source_archive_sha256":"bad","raw_text":"WORD"}')
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        _ocr_record(path, "x")


def test_configuration_cannot_switch_dataset_or_baseline() -> None:
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "configs/approximate/totaltext_crop_clip_battery_train_v1.json"
    config = json.loads(path.read_text())
    validate_config(config)
    config["dataset"] = "icdar13"
    with pytest.raises(ValueError, match="Total-Text"):
        validate_config(config)


def test_selected_anchor_ids_survive_json_round_trip() -> None:
    import json
    selections = serializable_selections({"baseline": {1: "a", 12: "b"}})
    assert json.loads(json.dumps(selections)) == selections
    assert selections == {"baseline": {"1": "a", "12": "b"}}


def test_clip_pairs_each_totaltext_crop_with_its_parseq_word(monkeypatch) -> None:
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, 2:, 0] = 1
    rows = [
        {"candidate_id": "total_text__1:a1:c0", "anchor_rank": 1,
         "bbox": [0, 0, 2, 2], "crop_shape": [2, 2, 3],
         "crop_array_sha256": clip_runtime.array_sha(image[:, :2]),
         "ocr": {"raw_text": "LEFT"}},
        {"candidate_id": "total_text__1:a1:c1", "anchor_rank": 1,
         "bbox": [2, 0, 4, 2], "crop_shape": [2, 2, 3],
         "crop_array_sha256": clip_runtime.array_sha(image[:, 2:]),
         "ocr": {"raw_text": "RIGHT"}},
    ]
    seen = []

    def fake_embeddings(_model, _processor, *, images, texts, device, batch_size):
        if images is not None:
            return np.array([[1, 0], [0, 1]], dtype=np.float32)
        seen.extend(texts)
        return np.array([[1, 0] if "LEFT" in text else [0, 1] for text in texts],
                        dtype=np.float32)

    monkeypatch.setattr(clip_runtime, "_embeddings", fake_embeddings)
    scores = clip_runtime.clip_scores(
        None, None, rows, image, device="cpu", batch_size=2,
        templates={"written": 'word "{text}"', "raw": "{text}"})
    assert scores[rows[0]["candidate_id"]] == {
        "written": 1.0, "raw": 1.0, "shuffled_written": 0.0}
    assert scores[rows[1]["candidate_id"]] == {
        "written": 1.0, "raw": 1.0, "shuffled_written": 0.0}
    assert 'word "LEFT"' in seen and 'word "RIGHT"' in seen


def test_clip_snapshot_identity_ignores_mutable_hub_metadata(tmp_path) -> None:
    (tmp_path / "config.json").write_text("model")
    metadata = tmp_path / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    (metadata / "config.json.metadata").write_text("first")
    first = clip_runtime.clip_snapshot_file_hashes(tmp_path)
    (metadata / "config.json.metadata").write_text("second")
    assert clip_runtime.clip_snapshot_file_hashes(tmp_path) == first
    assert set(first) == {"config.json"}
