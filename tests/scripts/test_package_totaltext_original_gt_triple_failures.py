import json

from scripts.package_totaltext_original_gt_triple_failures import is_correct, select_failures


def test_symmetric_normalization_does_not_accept_wrong_letters():
    assert is_correct("FISHERMAN'S", "FISHERMANS")
    assert is_correct("TV-GAME", "TVGAME")
    assert not is_correct("FISHERMANS", "FISHERMEN")
    assert not is_correct("!", "?")


def test_selection_requires_actual_v4_crops_and_three_failures(tmp_path):
    (tmp_path / "samples").mkdir()
    for mode in ("tight_rectangular", "curved_cutout"):
        (tmp_path / "ocr/gt_v4_oracle" / mode).mkdir(parents=True)
    sample = {
        "sample_id": "total_text:1",
        "ground_truth": [{"bbox": [0, 0, 10, 10], "text": "BOOK"},
                         {"bbox": [20, 0, 30, 10], "text": "TREE"}],
        "branches": {"gt_v4_oracle": {"proposals": [{"bbox": [0, 0, 10, 10], "rank": 1}]}},
    }
    (tmp_path / "samples/total_text__1.json").write_text(json.dumps(sample))
    for mode, text in (("tight_rectangular", "B0OK"), ("curved_cutout", "BO0K")):
        (tmp_path / "ocr/gt_v4_oracle" / mode / "total_text__1.json").write_text(
            json.dumps({"predictions": [{"rank": 1, "raw_text": text}]})
        )
    component_report = {"words": [
        {"sample_id": "total_text:1", "word_id": 1, "reconstruction": "B-0OK",
         "component_texts": ["B", "-", "0", "O", "K"], "contour_indices": [1, 2, 3, 4, 5]},
        {"sample_id": "total_text:1", "word_id": 2, "reconstruction": "TR33",
         "component_texts": ["TR", "33"], "contour_indices": [6, 7]},
    ]}
    selected = select_failures(component_report, tmp_path)
    assert len(selected) == 1
    assert selected[0]["gt_text"] == "BOOK"
