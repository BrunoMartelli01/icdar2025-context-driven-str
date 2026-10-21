import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_icdar13_routing_suite import local_route, seal_stage, summarize


def row(sid, fallback, text, deep_selected, deep_raw):
    return {"sample_id": sid, "branch": {"route": {
        "needs_fallback": fallback, "pre_fallback_output": text, "C": .2 if fallback else .9,
    }}, "deepsolo": {"selected_output": deep_selected, "recognized_strings": deep_raw}}


def test_all_fallback_distinguishes_image_accuracy_from_word_recall():
    rows = [row("1", True, "wrong", "ONE", ["ONE"]),
            row("2", True, "wrong", "three", ["three"])]
    gt = {"1": [{"text": "one"}, {"text": "two"}], "2": [{"text": "three"}]}
    result = summarize("branch", rows, gt)
    assert result["fallback_images"] == 2
    assert result["singular_final_gt_membership_rate"] == 1
    assert result["one_to_one_casefold_multiset"]["recall"] == 2 / 3
    assert result["accepted_precision"] is None


def test_conditional_fallback_and_counterfactual_overlap():
    rows = [row("1", False, "ONE", "wrong", ["ONE", "wrong"]),
            row("2", False, "wrong", "two", ["two"]),
            row("3", True, "three", "wrong", ["wrong"])]
    gt = {str(i): [{"text": word}] for i, word in enumerate(("one", "two", "three"), 1)}
    result = summarize("branch", rows, gt)
    assert result["fallback_images"] == 1
    assert result["accepted_correct_images"] == 1
    assert result["accepted_incorrect_images"] == 1
    assert result["accepted_correct_also_in_deepsolo_raw"] == 1
    assert result["accepted_wrong_replacing_correct_deepsolo"] == 1
    assert result["improved_images_vs_deepsolo_only"] == 1
    assert result["worsened_images_vs_deepsolo_only"] == 1
    assert result["singular_final_gt_membership_images"] == 1


class Embedder:
    def embeddings(self, texts):
        return np.ones((len(texts), 1))


def test_local_gate_empty_and_deterministic_tie():
    assert local_route([], "word", Embedder())["needs_fallback"]
    proposals = [{"text": "word", "candidate_id": str(i)} for i in range(2)]
    route = local_route(proposals, "word", Embedder())
    assert not route["needs_fallback"]
    assert route["selected_crop"]["candidate_id"] == "0"
    assert route["T1_used"] is False


def test_stage_seal_detects_content_corruption(tmp_path):
    directory = tmp_path / "stage"
    directory.mkdir()
    sample = SimpleNamespace(sample_id="1", image_sha256="image")
    path = directory / "0001.json"
    value = {"sample_id": "1", "image_sha256": "image", "contract_sha256": "contract"}
    path.write_text(json.dumps(value))
    seal_stage(tmp_path, "stage", [sample], "contract")
    seal_stage(tmp_path, "stage", [sample], "contract")
    path.write_text(json.dumps({**value, "text": "tampered"}))
    with pytest.raises(RuntimeError, match="content changed"):
        seal_stage(tmp_path, "stage", [sample], "contract")


def test_stage_seal_refuses_missing_or_cross_image_shards(tmp_path):
    sample = SimpleNamespace(sample_id="1", image_sha256="image")
    with pytest.raises(RuntimeError, match="incomplete"):
        seal_stage(tmp_path, "stage", [sample], "contract")
    (tmp_path / "stage").mkdir()
    (tmp_path / "stage" / "0001.json").write_text(json.dumps({
        "sample_id": "1", "image_sha256": "different", "contract_sha256": "contract",
    }))
    with pytest.raises(RuntimeError, match="wrong-image"):
        seal_stage(tmp_path, "stage", [sample], "contract")


def test_extra_routing_shares_frozen_proposals_and_resumes_without_models(tmp_path, monkeypatch):
    from scripts import evaluate_icdar13_routing_suite as suite

    class CountingEmbedder(Embedder):
        def __init__(self, *args):
            pass

        def release(self):
            pass

    monkeypatch.setattr(suite, "MPNetSemanticSimilarity", CountingEmbedder)
    crops = [{"candidate_id": "balanced", "text": "word", "bbox": [0, 0, 10, 10]}]
    branch = {"proposals": crops, "route": {"needs_fallback": True, "C": .2}}
    shards = {
        "04-routing": {"T1": "word", "T2": "word", "faithful": branch, "balanced": branch},
        "01-proposals": {"v1_anchors": [{"candidate_id": "v1", "ocr_key": "box", "bbox": [0, 0, 10, 10]}]},
        "02-parseq": {"crops": {"box": {"raw_text": "word"}}},
        "06-practical-captions": {"caption": {"raw_caption": "word"}},
    }
    for stage, value in shards.items():
        (tmp_path / stage).mkdir()
        (tmp_path / stage / "0001.json").write_text(json.dumps(value))
    samples = [SimpleNamespace(sample_id="1", image_sha256="image")]
    suite.extra_routes(samples, tmp_path, "contract")
    path = tmp_path / "07-suite-routing" / "0001.json"
    frozen_bytes = path.read_bytes()
    value = json.loads(frozen_bytes)
    assert all(name in value for name in suite.BRANCHES)
    assert value["faithful"] == branch
    assert value["balanced_local_antirepetition"]["proposals"] == crops
    assert value["balanced_local_repository_caption"]["proposals"] == crops
    assert not value["v1_local_antirepetition"]["route"]["needs_fallback"]
    assert value["deepsolo_only"]["route"]["needs_fallback"]

    def forbidden(*args):
        raise AssertionError("resuming complete routing must not reload MPNet")

    monkeypatch.setattr(suite, "MPNetSemanticSimilarity", forbidden)
    suite.extra_routes(samples, tmp_path, "contract")
    assert path.read_bytes() == frozen_bytes
