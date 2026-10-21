from __future__ import annotations

import numpy as np
import pytest

from scoring.feasibility import diagnose
from scoring.lexical import lexical_similarity
from scoring.semantic import MPNetApproxConfig, MPNetSemanticSimilarity
from scoring.strategies import STRATEGIES, decide


class _Encoder:
    max_seq_length = None
    def encode(self, texts, **kwargs):
        table = {"a": [1, 0], "b": [0, 1], "caption": [1, 0], "": [0, 0]}
        return np.asarray([table[text] for text in texts], dtype=float)


def test_mpnet_settings_and_semantic_similarity_are_deterministic():
    scorer = MPNetSemanticSimilarity(MPNetApproxConfig(), model=_Encoder())
    assert scorer.compare_candidates("a", "b", "caption") == (1.0, 0.0)
    assert scorer.compare_candidates("a", "b", "caption") == (1.0, 0.0)
    assert scorer.metadata["revision"] == "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    assert scorer.model.max_seq_length == 384


@pytest.mark.parametrize("mode,expected", [
    ("raw", False), ("lowercase", True), ("normalized_whitespace", False),
    ("unicode_normalized", False),
])
def test_named_lexical_modes(mode, expected):
    score = lexical_similarity("HELLO", "hello", mode).score
    assert (score == 1.0) is expected


def test_whitespace_and_unicode_modes_are_separate():
    assert lexical_similarity("a  b", "a b", "normalized_whitespace").score == 1
    assert lexical_similarity("e\u0301", "é", "unicode_normalized").score == 1
    assert lexical_similarity("", "", "raw").score == 1


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("score,needs", [(0.79, True), (0.8, False), (0.81, False)])
def test_threshold_boundary_for_each_strategy(strategy, score, needs):
    result = decide(strategy, T1="one", T2="caption", T3="two",
                    S1=score, S3=score - 0.01, L1=score, L3=score - 0.01)
    assert result.C == pytest.approx(score)
    assert result.needs_fallback is needs


def test_strategy_conflict_ties_and_empty_inputs_are_explicit():
    figure = decide("figure2_independent_max", T1="", T2="", T3="",
                    S1=0.9, S3=0.8, L1=0.1, L3=1.0)
    paired = decide("selected_candidate_paired", T1="", T2="", T3="",
                    S1=0.9, S3=0.8, L1=0.1, L3=1.0)
    repo = decide("repository_paired_threshold", T1="", T2="", T3="",
                  S1=0.9, S3=0.8, L1=0.1, L3=1.0)
    assert figure.C == pytest.approx(0.94)
    assert paired.C == pytest.approx(0.58)
    assert repo.confidence_winner == "T3"
    tie = decide("selected_candidate_paired", T1="a", T2="c", T3="b",
                 S1=0.5, S3=0.5, L1=0.5, L3=0.5)
    assert tie.semantic_winner == tie.confidence_winner == "T1"


def test_literal_whole_string_feasibility_is_below_tau_even_at_perfect_semantics():
    result = diagnose("CAFE", "The image shows a cafe sign above a busy street with people and bicycles.")
    assert result.required_lexical_at_perfect_semantic == pytest.approx(0.5)
    assert result.reaches_tau_even_if_semantic_is_one is False
