from pipeline.e2e_comparison_diagnostic import (
    multiset_exact, repository_route, word_crop_diagnostic,
)
from scripts.diagnose_e2e_parseq_vs_deepsolo import (
    detection_envelope, exact_spatial_edges, maximum_word_matching,
)


def test_repository_route_uses_first_maximum_c3_and_semantic_winner() -> None:
    result = repository_route(
        t1="alpha", caption="scene text", s1=0.2,
        candidates=[
            {"candidate_id": "first", "text": "word", "semantic_similarity": 0.4},
            {"candidate_id": "second", "text": "word", "semantic_similarity": 0.4},
        ],
    )
    assert result["selected_crop"]["candidate_id"] == "first"
    assert result["semantic_winner"] == "T3"
    assert result["pre_fallback_output"] == "word"
    assert result["needs_fallback"] is True


def test_repository_route_can_avoid_fallback_at_exact_boundary() -> None:
    result = repository_route(
        t1="same", caption="same", s1=1.0, candidates=[],
    )
    assert result["C1"] == 1.0
    assert result["needs_fallback"] is False
    assert result["pre_fallback_output"] == "same"


def test_multiset_exact_is_one_to_one_and_casefolded() -> None:
    result = multiset_exact(["ONE", "one", "extra"], ["one", "two"])
    assert result["matches"] == 1
    assert result["predictions"] == 3
    assert result["references"] == 2
    assert result["precision"] == 1 / 3
    assert result["recall"] == 1 / 2


def test_word_crop_diagnostic_uses_best_geometry_before_text() -> None:
    result = word_crop_diagnostic(
        [[
            {"bbox": [0, 0, 10, 10], "text": "correct"},
            {"bbox": [0, 0, 5, 10], "text": "wrong"},
        ]],
        [[{"bbox": [0, 0, 10, 10], "text": "CORRECT"}]],
    )
    assert result["iou_ge_0_50"] == 1
    assert result["iou_ge_0_75"] == 1
    assert result["exact_casefold"] == 1
    assert result["retained_proposals"] == 2


def test_deepsolo_envelope_and_exact_spatial_matching() -> None:
    detection = {
        "text": "WORD",
        "geometry": {"boundary_points": [[1.0, 2.0], [9.0, 2.0], [9.0, 8.0], [1.0, 8.0]]},
    }
    detection["bbox"] = detection_envelope(detection)
    words = [
        {"text": "word", "bbox": [1, 2, 9, 8]},
        {"text": "other", "bbox": [20, 2, 30, 8]},
    ]
    edges = exact_spatial_edges([detection], words)
    assert edges == [(0, 0)]
    assert maximum_word_matching(1, 2, edges) == {0}


def test_maximum_word_matching_is_one_to_one_for_duplicate_words() -> None:
    edges = [(0, 0), (0, 1), (1, 0)]
    assert maximum_word_matching(2, 2, edges) == {0, 1}
