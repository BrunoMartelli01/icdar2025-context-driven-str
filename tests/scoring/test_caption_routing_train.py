from copy import deepcopy

import numpy as np
import pytest

from diagnostics.caption_routing_train import checked, seal_write, routes, route_scores, word_summary
from diagnostics.joined_t3 import join_crop_transcriptions


class FixedEmbeddings:
    def embeddings(self, texts):
        return np.tile([1., 0.], (len(texts), 1))


def test_each_original_block_and_reading_order_phrase_have_independent_scores():
    blocks = [{"candidate_id": "b", "bbox": [20, 0, 30, 10], "text": "world"},
              {"candidate_id": "a", "bbox": [0, 0, 10, 10], "text": "hello"}]
    before = deepcopy(blocks)
    joined = join_crop_transcriptions(blocks)
    assert joined["text"] == "hello world"
    scores = route_scores("hello", joined, blocks, "hello world", FixedEmbeddings(), "whole_caption")
    result = routes("hello", joined, blocks, "hello world", scores, ["selected_candidate_paired"])
    assert result["T3_score"]["lexical_similarity"] == 1
    assert len(result["blocks"]) == 2
    assert result["blocks"][0]["lexical_similarity"] < 1
    assert blocks == before


def test_block_threshold_boundary_and_empty_text_guard():
    blocks = [{"candidate_id": str(i), "bbox": [i*10, 0, i*10+5, 5], "text": t}
              for i, t in enumerate(["accepted", "fallback", ""])]
    score = {"semantic_similarity": 1., "lexical_similarity": .5, "confidence": .8}
    scores = [score, score, score, {**score, "confidence": .799999}, score]
    result = routes("", {"text": ""}, blocks, "caption", scores, ["selected_candidate_paired"])
    assert [b["needs_fallback"] for b in result["blocks"]] == [False, True, True]
    assert result["image_routes"]["selected_candidate_paired"]["needs_fallback"]


def test_semantic_winner_and_repository_control_remain_distinct():
    scores = [{"semantic_similarity": .95, "lexical_similarity": .1},
              {"semantic_similarity": .9, "lexical_similarity": 1.}]
    result = routes("one", {"text": "two"}, [], "caption", scores,
                    ["selected_candidate_paired", "repository_paired_threshold"])
    primary = result["image_routes"]["selected_candidate_paired"]
    control = result["image_routes"]["repository_paired_threshold"]
    assert primary["selected_output"] == control["selected_output"] == "one"
    assert primary["needs_fallback"] and not control["needs_fallback"]


def test_integrity_and_resume_do_not_overwrite_valid_output(tmp_path):
    path = tmp_path / "row.json"
    row = {"contract_sha256": "contract", "text": "hello"}
    seal_write(path, row)
    original = path.read_bytes()
    seal_write(path, row)
    assert path.read_bytes() == original
    assert checked(path, "contract")["text"] == "hello"
    with pytest.raises(RuntimeError):
        seal_write(path, {**row, "text": "different"})
    with pytest.raises(RuntimeError):
        checked(path, "other-contract")
    path.write_bytes(original.replace(b"hello", b"corrupted"))
    with pytest.raises(RuntimeError):
        checked(path, "contract")


def test_word_evaluation_preserves_denominator_and_caps_repeated_outputs():
    rows = [{"sample_id": "100", "text": "hello hello junk", "accepted": True},
            {"sample_id": "101", "text": "world", "accepted": False}]
    gt = {"100": [{"text": "Hello"}], "101": [{"text": "world"}]}
    result = word_summary(rows, gt, lambda r: r["text"], lambda r: r["accepted"])
    assert result["pre_fallback_word_matches"] == 2
    assert result["accepted_word_matches"] == 1
    assert result["accepted_word_recall"] == .5
    assert result["accepted_word_precision"] == 1/3


def test_frozen_config_is_train_only_and_never_appends_florence_ocr():
    from diagnostics.caption_routing_train import read
    cfg = read("configs/approximate/caption_routing_five_models_train_v1.json")
    assert cfg["split"] == "TRAIN" and cfg["images"] == 229
    assert len(cfg["caption_profiles"]) == 7
    assert "instructblip" in cfg["caption_profiles"]
    assert cfg["instructblip"]["revision"] == "bf9bd58cb3bb06e88c03b80f9cc346b90094bc6d"
    assert cfg["florence_large"]["task"] == "<MORE_DETAILED_CAPTION>"
    assert (cfg["alpha"], cfg["beta"], cfg["tau"]) == (.6, .4, .8)
