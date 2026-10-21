from scoring.context_candidate_ranking import aggregate, evaluate_word, flip_diagnostics


def _candidate(candidate_id, bbox, text, s, l, c, baseline):
    return {
        "candidate_id": candidate_id,
        "bbox": bbox,
        "ocr": {"raw_text": text, "mean_token_log_probability": baseline},
        "context_scores": {
            "semantic_similarity": s,
            "lexical_similarity": l,
            "fixed_context_score": c,
        },
    }


def test_context_ranking_uses_one_score_and_deterministic_ties():
    candidates = [
        _candidate("b", [0, 0, 10, 10], "word", .9, .1, .58, -2),
        _candidate("a", [0, 0, 20, 10], "wrong", .9, .2, .62, -1),
        _candidate("c", [30, 0, 40, 10], "elsewhere", 1, 1, 1, 0),
    ]
    row = evaluate_word(candidates, gt_bbox=[0, 0, 10, 10], gt_text="WORD")
    assert row is not None
    assert row["rankings"]["semantic_similarity"]["top1_candidate_id"] == "a"
    assert row["rankings"]["lexical_similarity"]["top1_candidate_id"] == "a"
    assert row["rankings"]["fixed_context_score"]["top1_candidate_id"] == "a"
    assert row["rankings"]["mean_token_log_probability"]["top1_candidate_id"] == "a"
    assert row["exists_iou_ge_0_50_and_parseq_exact_casefold"] is True


def test_aggregate_and_flip_counts_are_posthoc_only():
    first = evaluate_word(
        [
            _candidate("good", [0, 0, 10, 10], "x", .8, .8, .8, -2),
            _candidate("bad", [0, 0, 25, 10], "bad", .1, .1, .1, -1),
        ],
        gt_bbox=[0, 0, 10, 10], gt_text="x",
    )
    second = evaluate_word(
        [
            _candidate("good2", [0, 0, 10, 10], "y", .1, .1, .1, -1),
            _candidate("bad2", [0, 0, 25, 10], "bad", .8, .8, .8, -2),
        ],
        gt_bbox=[0, 0, 10, 10], gt_text="y",
    )
    rows = [first, second]
    result = aggregate(rows)
    assert result["fixed_context_score"]["top1_ge_0_50"] == 1
    assert result["mean_token_log_probability"]["top1_ge_0_50"] == 1
    assert result["fixed_context_score"]["exact_selectable_gt_words"] == 2
    assert result["fixed_context_score"]["exact_selectable_top1_parseq_exact"] == 1
    flips = flip_diagnostics(rows)
    assert flips["fixed_context_score"]["top1_ge_0_50"] == {
        "improved": 1, "worsened": 1, "net": 0, "unchanged": 0,
    }
