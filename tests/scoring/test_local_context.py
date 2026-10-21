import numpy as np

from scoring.local_context import (
    candidate_window_lengths, caption_tokens, caption_windows,
    repetition_diagnostics, score_local_candidates,
)
from scripts.diagnose_balanced_local_context_train import _attach_scores


class KeywordEmbedder:
    def embeddings(self, texts):
        values = []
        for text in texts:
            tokens = {value.casefold() for value in caption_tokens(text)}
            vector = np.asarray([1.0, 0.0]) if "shop" in tokens else np.asarray([0.0, 1.0])
            values.append(vector)
        return np.asarray(values)


def test_windows_are_contiguous_deterministic_and_keep_punctuation():
    assert caption_tokens("A shop, open-now.") == ("A", "shop", ",", "open-now", ".")
    assert caption_windows("one two three", (2, 1, 2)) == (
        "one", "two", "three", "one two", "two three",
    )


def test_window_lengths_are_candidate_relative():
    assert candidate_window_lengths("book shop", variant="exact_length") == (2,)
    assert candidate_window_lengths("book shop", variant="neighbor_lengths") == (1, 2, 3)
    assert candidate_window_lengths("", variant="neighbor_lengths") == ()


def test_local_score_uses_best_paired_window_and_empty_fails_closed():
    result = score_local_candidates(
        ["shop", ""], "a shop near town", KeywordEmbedder(), variant="exact_length",
    )
    assert result[0]["window"] == "shop"
    assert result[0]["semantic_similarity"] == 1.0
    assert result[0]["lexical_similarity"] == 1.0
    assert result[0]["confidence"] == 1.0
    assert result[1]["confidence"] == 0.0


def test_repetition_diagnostic_detects_loop_without_rewriting_caption():
    value = repetition_diagnostics("one two three four one two three four")
    assert value["has_duplicate_ngram"] is True
    assert value["duplicate_ngram_count"] == 1
    assert repetition_diagnostics("one two three four")["has_duplicate_ngram"] is False


def test_balanced_score_attachment_propagates_across_candidate_copies():
    base = {"candidate_id": "same", "ocr": {"raw_text": "shop"}}
    selections = {
        "v1": {"1": (dict(base),)},
        "balanced": {"1": (dict(base),)},
    }
    _attach_scores(selections, {"1": "a shop"}, KeywordEmbedder())
    assert selections["v1"]["1"][0]["practical_context"] == selections["balanced"]["1"][0]["practical_context"]
