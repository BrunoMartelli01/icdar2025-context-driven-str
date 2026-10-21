import numpy as np
from PIL import Image
import pytest

from scripts.evaluate_text_recall_refinements import diagnostic, save_instance_labels


def test_native384_export_preserves_ids_above_uint8_and_rejects_changed_cache(tmp_path):
    labels = np.zeros((384, 384), np.int32)
    labels[30:90, 20:70] = 1023
    path = tmp_path / "maps" / "instances.png"
    digest = save_instance_labels(path, labels)
    with Image.open(path) as opened:
        np.testing.assert_array_equal(np.asarray(opened), labels)
    assert save_instance_labels(path, labels) == digest
    labels[31, 21] = 2
    with pytest.raises(ValueError, match="differs"):
        save_instance_labels(path, labels)


def test_gt_crop_diagnostic_distinguishes_localization_crop_and_recognizer_failures():
    gt = [{"bbox": [0, 0, 10, 10], "text": "A"}, {"bbox": [20, 0, 30, 10], "text": "B"},
          {"bbox": [40, 0, 50, 10], "text": "C"}]
    oracle = {"sample_id": "x", "ground_truth": gt, "predictions_all": [
        {"gt_crop_index": i, "text": t} for i, t in enumerate(["A", "B", "X"])]}
    row = {"sample_id": "x", "ground_truth": gt, "correct_casefold_gt_indices": [], "ignored_prediction_ids": [],
           "predictions_all": [{"instance_id": 1, "bbox": [20, 0, 30, 10], "text": "X"},
                               {"instance_id": 2, "bbox": [40, 0, 50, 10], "text": "X"}]}
    result = diagnostic({"branches": {"oracle__diagnostic": {"gt_boxes": {"per_image": [oracle]}},
                                      "source": {"baseline": {"per_image": [row]}}}})
    counts = result["branches"]["source"]["baseline"]["counts"]
    assert counts["no_iou50_candidate_gt_crop_correct"] == 1
    assert counts["viable_crop_wrong_text_gt_crop_correct"] == 1
    assert counts["viable_crop_wrong_text_gt_crop_wrong"] == 1
    assert result["gt_box_fixed_pair_correct"] == 2
    assert result["gt_words"] == 3
