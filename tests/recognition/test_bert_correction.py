"""Scientific boundaries and genuine tiny-BERT CPU integration (random fixture).

Random fixture outputs test execution only and are not OCR experiment results.
"""

from dataclasses import replace
import math

import pytest
import torch
from transformers import BertConfig, BertForMaskedLM, BertTokenizerFast

from recognition.bert_correction import (
    BertWordScorer, CorrectionPolicy, candidate_words, choose_candidate,
    correct_blocks, eligible_word, local_context,
)


@pytest.fixture
def policy():
    return CorrectionPolicy(min_word_length=3, long_word_length=6,
            short_max_edits=1, long_max_edits=2, max_relative_edit=.3,
            max_candidates=24, protect_min_token_probability=.95,
            edit_penalty=1., minimum_gain=1.5, neighbours_per_side=3,
            min_vertical_overlap=.5, max_horizontal_gap_heights=8., max_caption_words=80)


@pytest.fixture
def scorer(tmp_path):
    torch.manual_seed(123)
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
             "the", "cat", "car", "park", "##ing", "is", "open", "shop", "."]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab) + "\n")
    tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=True)
    model = BertForMaskedLM(BertConfig(vocab_size=len(vocab), hidden_size=16,
                           num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=64))
    return BertWordScorer(tokenizer, model, device="cpu", batch_size=3, max_tokens=48)


@pytest.mark.parametrize("word", ["1234", "NURI'S", "petro-sains", "BAR&GRILL", "café",
                                 "two words", "iPhone", "AB", "", "[MASK]"])
def test_protected_word_shapes(word, policy):
    assert not eligible_word(word, policy)
    assert candidate_words(word, ["car", "bar"], ["car"], policy)[0]["text"] == word
    assert len(candidate_words(word, ["car", "bar"], ["car"], policy)) == 1


def test_candidates_are_bounded_deduplicated_and_restore_case(policy):
    candidates = candidate_words("PARKIN", ["parking", "parks", "far", "market"],
                                 ["Parking Parking PARKIN"], policy)
    assert candidates[0]["text"] == "PARKIN"
    assert candidates[1] == {"text": "PARKING", "source": "OBSERVED_CONTEXT", "edit_distance": 1}
    assert len({v["text"] for v in candidates}) == len(candidates)
    assert all(v["edit_distance"] <= 2 for v in candidates)
    assert candidate_words("car", ["cat"], [], policy) == [
        {"text": "car", "source": "PARSeq_KEEP", "edit_distance": 0}]


def test_candidate_cap_and_order_are_deterministic(policy):
    cfg = replace(policy, max_relative_edit=.4, max_candidates=2)
    a = candidate_words("bar", ["cat", "car", "bat"], ["car"], cfg)
    b = candidate_words("bar", ["bat", "car", "cat"], ["car"], cfg)
    assert a == b
    assert a[1]["text"] == "car"


def test_strict_threshold_and_keep_ties(policy):
    candidates = [{"text": "parkin", "edit_distance": 0}, {"text": "parking", "edit_distance": 1}]
    scores = [{"mean_log_probability": -5}, {"mean_log_probability": -2.5}]
    assert choose_candidate(candidates, scores, policy)["selected_index"] == 0
    scores[1]["mean_log_probability"] = -2.49
    assert choose_candidate(candidates, scores, policy)["selected_index"] == 1


def test_missing_reference_abstains_and_nonfinite_fails(policy):
    candidates = [{"text": "xxx", "edit_distance": 0}, {"text": "cat", "edit_distance": 1}]
    scores = [{"mean_log_probability": None, "status": "unknown_or_special_target"},
              {"mean_log_probability": -1}]
    assert choose_candidate(candidates, scores, policy)["selected_index"] == 0
    with pytest.raises(ValueError, match="non-finite"):
        choose_candidate(candidates, [{"mean_log_probability": math.nan}, scores[1]], policy)


def test_line_context_excludes_other_lines_overlaps_and_far_words(policy):
    blocks = [{"bbox": [30, 0, 40, 10], "text": "target"},
              {"bbox": [10, 0, 20, 10], "text": "left"},
              {"bbox": [50, 0, 60, 10], "text": "right"},
              {"bbox": [10, 20, 20, 30], "text": "other"},
              {"bbox": [200, 0, 210, 10], "text": "far"},
              {"bbox": [35, 0, 45, 10], "text": "overlap"}]
    assert local_context(blocks, 0, policy) == (["left"], ["right"])


def test_word_l2r_masks_future_subwords_without_self_copy(scorer):
    tasks, status = scorer.masked_tasks("parking", ["the"], ["is", "open"], None)
    assert status == "scored" and len(tasks) == 2
    first, second = tasks
    mask = scorer.tokenizer.mask_token_id
    assert first["input_ids"][first["position"]] == mask
    assert first["input_ids"][second["position"]] == mask
    assert second["input_ids"][first["position"]] == first["label"]
    assert second["input_ids"][second["position"]] == mask
    assert first["label"] != mask and second["label"] != mask


def test_caption_is_separate_segment_and_never_scored_as_target(scorer):
    tasks, status = scorer.masked_tasks("parking", [], [], "the parking shop")
    assert status == "scored" and len(tasks) == 2
    assert all(t["token_type_ids"][t["position"]] == 1 for t in tasks)
    assert tasks[0]["input_ids"][2] == scorer.tokenizer.convert_tokens_to_ids("park")


def test_unknown_and_token_limit_are_explicit(scorer):
    assert scorer.masked_tasks("xyz", [], [], None) == ([], "unknown_or_special_target")
    assert scorer.masked_tasks("cat", ["the"] * 60, [], None) == ([], "input_token_limit")


def test_genuine_tiny_bert_executes_and_scores_are_deterministic(scorer):
    first = scorer.score(["cat", "parking", "xyz"], left=["the"], right=["is"], caption=None)
    second = scorer.score(["cat", "parking", "xyz"], left=["the"], right=["is"], caption=None)
    assert first == second
    assert [r["token_count"] for r in first] == [1, 2, 0]
    assert first[2]["mean_log_probability"] is None
    assert first[0]["mean_log_probability"] < 0
    assert first[1]["sum_log_probability"] == sum(first[1]["token_log_probabilities"])


def test_runtime_errors_propagate_without_keep_substitution(scorer, monkeypatch):
    def failure(**kwargs):
        raise RuntimeError("CUDA out of memory fixture")
    monkeypatch.setattr(scorer.model, "forward", failure)
    with pytest.raises(RuntimeError, match="out of memory"):
        scorer.score(["cat"], left=[], right=[], caption=None)


def test_visual_gate_boundary_preserves_crop_identity(policy, scorer):
    cfg = replace(policy, max_relative_edit=.4)
    block = {"rank": 1, "bbox": [0, 0, 10, 10], "crop_rgb_sha256": "a" * 64,
             "text": "CAT", "minimum_token_probability": .95}
    result = correct_blocks([block], condition="word_only", caption=None, scorer=scorer, policy=cfg)[0]
    assert result["text"] == block["text"]
    assert result["decision"]["reason"] == "visual_confidence_protected"
    assert result["scores"] == []
    assert result["bbox"] == block["bbox"] and result["crop_rgb_sha256"] == block["crop_rgb_sha256"]
    with pytest.raises(ValueError, match="caption"):
        correct_blocks([block], condition="local_ocr_caption", caption=None, scorer=scorer, policy=cfg)


def test_all_decisions_use_frozen_original_neighbours(policy):
    class Scorer:
        vocabulary = ["parking", "parkin", "market", "markets"]
        calls = []
        def score(self, candidates, **context):
            self.calls.append(context)
            return [{"mean_log_probability": -10 if i == 0 else -1, "status": "scored"}
                    for i, _ in enumerate(candidates)]
    scorer = Scorer()
    blocks = [{"rank": 1, "bbox": [0, 0, 20, 10], "crop_rgb_sha256": "a" * 64,
               "text": "parkin", "minimum_token_probability": .5},
              {"rank": 2, "bbox": [25, 0, 45, 10], "crop_rgb_sha256": "b" * 64,
               "text": "markets", "minimum_token_probability": .5}]
    result = correct_blocks(blocks, condition="local_ocr", caption=None, scorer=scorer, policy=policy)
    assert result[0]["text"] == "parking"
    assert scorer.calls[1]["left"] == ["parkin"]
    assert blocks[0]["text"] == "parkin"


def test_candidate_pool_is_identical_across_context_conditions(policy, scorer):
    block = {"rank": 1, "bbox": [0, 0, 20, 10], "crop_rgb_sha256": "a" * 64,
             "text": "parkin", "minimum_token_probability": .5}
    outputs = [correct_blocks([block], condition=condition, caption="A parking shop",
                              scorer=scorer, policy=policy)[0]
               for condition in ("word_only", "local_ocr", "local_ocr_caption")]
    assert outputs[0]["candidates"] == outputs[1]["candidates"] == outputs[2]["candidates"]
    assert outputs[0]["caption"] is None and outputs[2]["caption"] == "A parking shop"
