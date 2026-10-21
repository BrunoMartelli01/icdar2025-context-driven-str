from copy import deepcopy
from dataclasses import replace
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recognition.bert_crop_selection import (CropPolicy, admission_value, apply_filter,
    bert_choices, observable_sample, physical_features, score_requests, selections)
from recognition.group_word_selection import ProbabilityMass
from scripts.evaluate_totaltext_bert_crop_selection import load_config, path_for


def policy():
    import json
    config = json.loads(Path("configs/approximate/totaltext_bert_crop_selection_v1.json").read_text())
    return CropPolicy(**config["policy"])


def candidate(cid, text="NAME", box=None, confidence=.9, support=1.):
    return {"candidate_id": cid, "raw_text": text, "word": text.casefold(),
            "bbox": box or [10, 10, 30, 20], "scores": {
                "confidence_balanced": confidence, "robust_mask_support": support}}


def fixture_sample():
    mass = ProbabilityMass(np.full((100, 100), .8), (100, 100))
    one, two = candidate("a"), candidate("b", "NAMES", confidence=.89)
    evidence = [{"group_id": "g", "candidates": {"a": one, "b": two}}]
    rows = [{"candidate_id": c["candidate_id"], "bbox": c["bbox"], "ocr": {"raw_text": c["raw_text"]},
             "crop_rgb_sha256": "a" * 64} for c in (one, two)]
    base = [{"group_id": "g", "score": .9, **{k: one[k] for k in
            ("candidate_id", "bbox", "word", "raw_text")}}]
    return observable_sample(base, evidence, rows, mass, "A store with signs.", policy())


def likelihood(value, status="scored"):
    return {"mean_log_probability": value, "status": status}


def test_absolute_mask_mean_is_not_relative_completeness():
    mass = ProbabilityMass(np.full((20, 20), .2), (100, 50))
    c = candidate("a", box=[10, 10, 30, 20])
    p = physical_features(c, mass, policy())
    assert p["mask_mean"] == pytest.approx(.2)
    assert p["mask_support"] == 1
    assert p["height_cells"] == 2


@pytest.mark.parametrize("box", [[-1, 0, 2, 2], [0, 0, 101, 5], [1, 1, 1, 2]])
def test_crop_mapping_rejects_out_of_image_or_empty(box):
    with pytest.raises(ValueError):
        physical_features(candidate("a", box=box), ProbabilityMass(np.ones((10, 10)), (100, 100)), policy())


def test_native_readings_and_keep_preserved():
    s = fixture_sample()
    assert [c["raw_text"] for c in s["groups"][0]["candidates"]] == ["NAME", "NAMES"]
    assert s["groups"][0]["context"] == {"left": [], "right": []}


def test_ties_and_unavailable_reference_keep():
    sample = fixture_sample()
    for base in (likelihood(-3), likelihood(None, "unknown_or_special_target")):
        scores = {"g": {"scene": [base, likelihood(-3)]}}
        rows, trace = bert_choices(sample, scores, policy(), "scene")
        assert rows[0]["candidate_id"] == "a"
        assert trace[0]["selected_index"] == 0


def test_selects_existing_crop_with_language_gain():
    s = fixture_sample()
    scores = {"g": {"scene": [likelihood(-8), likelihood(-2)]}}
    rows, _ = bert_choices(s, scores, policy(), "scene")
    assert rows[0]["candidate_id"] == "b"
    assert rows[0]["raw_text"] == "NAMES"
    assert rows[0]["bbox"] == s["groups"][0]["candidates"][1]["bbox"]


def test_crop_collisions_revert_without_duplicate_assignments():
    s = fixture_sample()
    other = deepcopy(s["groups"][0])
    other["group_id"] = "h"
    other["baseline"]["group_id"] = "h"
    other["baseline"]["candidate_id"] = "c"
    other["candidates"][0]["candidate_id"] = "c"
    s["groups"].append(other)
    scores = {g: {"local": [likelihood(-8), likelihood(-2)]} for g in ("g", "h")}
    selected, trace = bert_choices(s, scores, policy(), "local")
    assert [r["candidate_id"] for r in selected] == ["a", "c"]
    assert all(t["conflict_reverted"] for t in trace)


@pytest.mark.parametrize("value", [math.nan, math.inf, .1])
def test_invalid_likelihood_fails(value):
    with pytest.raises(ValueError):
        bert_choices(fixture_sample(), {"g": {"local": [likelihood(value), likelihood(-2)]}}, policy(), "local")


def test_scores_cannot_change_group_population():
    with pytest.raises(ValueError):
        bert_choices(fixture_sample(), {}, policy(), "local")


def test_unknown_language_abstains_and_numbers_are_not_deleted():
    c = fixture_sample()["groups"][0]["candidates"][0]
    c["raw_text"], c["word"] = "2025", "2025"
    assert admission_value(c, likelihood(None, "unknown_or_special_target"), policy()) == c["physical"]["admission_visual"]


def test_language_is_not_used_as_visual_text_confidence():
    c = fixture_sample()["groups"][0]["candidates"][0]
    c["physical"]["admission_visual"] = 0
    assert admission_value(c, likelihood(-.001), policy()) == 0


def test_filter_threshold_boundary_and_crops_invariant():
    sample = fixture_sample()
    scores = {"g": {c: [likelihood(-3), likelihood(-3)] for c in ("local", "scene")}}
    selection = selections(sample, scores, policy())
    cutoff = selection["admission"]["visual"]["g"]
    assert len(apply_filter(selection, "visual", cutoff)) == 1
    assert apply_filter(selection, "visual", math.nextafter(cutoff, 1.)) == []
    assert apply_filter(selection, "scene_joint", 0)[0]["raw_text"] == "NAME"


@pytest.mark.parametrize("change", [{"maximum_readings": 0}, {"bert_weight": -1},
    {"language_log_scale": 0}, {"minimum_visual_ratio": 2}, {"admission_language_weight": math.nan}])
def test_policy_rejects_invalid_parameters(change):
    with pytest.raises(ValueError):
        replace(policy(), **change)


def test_approximate_namespace_and_no_unsafe_sample_paths():
    cfg = load_config(Path("configs/approximate/totaltext_bert_crop_selection_v1.json"))
    assert cfg["reproduction_mode"] == "approximate"
    with pytest.raises(ValueError):
        load_config(Path("configs/strict/totaltext_bert_crop_selection_v1.json"))
    with pytest.raises(ValueError):
        path_for(Path("/tmp"), "inference", "total_text:../../etc/passwd")


def test_cross_request_batch_restores_token_order():
    import torch

    class Tokenizer:
        def pad(self, features, **kwargs):
            return {k: torch.tensor([v[k] for v in features]) for k in features[0]}

    class Model:
        def __call__(self, **features):
            n, size = features["input_ids"].shape
            return SimpleNamespace(logits=torch.arange(5).float().repeat(n, size, 1))

    class Scorer:
        batch_size, device, tokenizer, model = 2, "cpu", Tokenizer(), Model()
        def masked_tasks(self, text, left, right, caption):
            if text == "?":
                return [], "unknown_or_special_target"
            labels = [1, 3] if text == "ab" else [2]
            return [{"input_ids": [0, 0], "attention_mask": [1, 1], "token_type_ids": [0, 0],
                     "position": 0, "label": i} for i in labels], "scored"

    requests = [{"text": text, "left": [], "right": [], "caption": None} for text in ("ab", "?", "c")]
    result = score_requests(Scorer(), requests)
    expected = torch.arange(5).float().log_softmax(0)
    assert result[0]["token_log_probabilities"] == pytest.approx([expected[1].item(), expected[3].item()])
    assert result[0]["mean_log_probability"] == pytest.approx((expected[1].item() + expected[3].item()) / 2)
    assert result[1]["mean_log_probability"] is None
    assert result[2]["mean_log_probability"] == pytest.approx(expected[2].item())


def test_context_uses_other_reliable_groups_and_excludes_shared_parent():
    a, b = candidate("a", "LEFT", [0, 10, 8, 20]), candidate("b", "TARGET", [12, 10, 30, 20])
    shared = candidate("parent", "LEFTTARGET", [0, 10, 30, 20])
    evidence = [{"group_id": "left", "candidates": {"a": a}},
                {"group_id": "target", "candidates": {"b": b}}]
    rows = [{"candidate_id": c["candidate_id"], "bbox": c["bbox"],
             "ocr": {"raw_text": c["raw_text"]}, "crop_rgb_sha256": "a" * 64}
            for c in (a, b, shared)]
    baseline = [{"group_id": g["group_id"], "score": .9, **{k: next(iter(g["candidates"].values()))[k]
                for k in ("candidate_id", "bbox", "word", "raw_text")}} for g in evidence]
    mass = ProbabilityMass(np.full((100, 100), .8), (100, 100))
    sample = observable_sample(baseline, evidence, rows, mass, "context", policy())
    assert sample["groups"][1]["context"]["left"] == ["LEFT"]
    evidence[0]["candidates"]["parent"] = shared
    evidence[1]["candidates"]["parent"] = shared
    sample = observable_sample(baseline, evidence, rows, mass, "context", policy())
    assert sample["groups"][1]["context"]["left"] == []


def test_language_cannot_invent_an_output_or_change_geometry():
    s = fixture_sample()
    scores = {"g": {c: [likelihood(-99), likelihood(-.01)] for c in ("local", "scene")}}
    result = selections(s, scores, policy())
    original = {c["candidate_id"]: c for c in s["groups"][0]["candidates"]}
    for rows in result["choices"].values():
        for row in rows:
            assert all(row[k] == original[row["candidate_id"]][k] for k in ("raw_text", "word", "bbox"))


def test_payload_mutation_rejected(tmp_path):
    from scripts.evaluate_totaltext_bert_correction import payload_digest, seal_json
    from scripts.evaluate_totaltext_bert_crop_selection import check_payload
    value = {"inputs_sha256": "inputs", "runtime_sha256": "runtime", "sample_id": "total_text:1"}
    value["payload_sha256"] = payload_digest(value)
    value["sample_id"] = "total_text:2"
    path = tmp_path / "shard.json"
    seal_json(path, value)
    with pytest.raises(ValueError):
        check_payload(path, {"inputs_sha256": "inputs"}, "runtime", "total_text:2")


def test_oom_is_recorded_and_propagated(tmp_path, monkeypatch):
    import torch
    import scripts.evaluate_totaltext_bert_crop_selection as runner
    def fail(args):
        raise torch.cuda.OutOfMemoryError("simulated unit-test OOM")
    monkeypatch.setattr(runner, "_infer", fail)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 123)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 456)
    with pytest.raises(torch.cuda.OutOfMemoryError):
        runner.infer(SimpleNamespace(output_dir=tmp_path, device="cuda:0"))
    import json
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["automatic_fallback"] is False
    assert failure["gpu_peak_allocated_bytes"] == 123


def test_policy_requires_all_explicit_scientific_parameters():
    with pytest.raises(TypeError):
        CropPolicy()


def test_cache_cannot_reuse_changed_word_geometry_or_context():
    from scripts.replay_totaltext_bert_crop_selection import identical_observations
    original = {"samples": [fixture_sample()]}
    identical_observations(original, deepcopy(original))
    for field in ("raw_text", "bbox"):
        changed = deepcopy(original)
        changed["samples"][0]["groups"][0]["candidates"][0][field] = "different"
        with pytest.raises(ValueError):
            identical_observations(changed, original)
    changed = deepcopy(original)
    changed["samples"][0]["groups"][0]["context"]["left"] = ["leaked target"]
    with pytest.raises(ValueError):
        identical_observations(changed, original)


def test_cache_authenticates_scoring_algorithm():
    from scripts.replay_totaltext_bert_crop_selection import function_hash
    a = function_hash("def score_requests(x):\n    return x + 1\n", "score_requests")
    b = function_hash("# comment\ndef score_requests(x):\n    return x + 1\n", "score_requests")
    c = function_hash("def score_requests(x):\n    return x + 2\n", "score_requests")
    assert a == b and a != c
    with pytest.raises(ValueError):
        function_hash("", "score_requests")
