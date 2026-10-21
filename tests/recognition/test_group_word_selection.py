import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recognition.group_word_selection import (ProbabilityMass, add_clip_scores, add_cross_scores,
    aligned_probability, completeness, cross_requests, group_evidence, normalize,
    select_groups, threshold_stability)
from recognition.group_word_selection_runtime import (clip_group_scores, forced_likelihood_batch,
                                                       square_canvas)
from scripts.run_crop_group_word_selection import config, evaluate_predictions, measured, validate_config


REPO = Path(__file__).resolve().parents[2]


def row(cid, text="Word", box=None, view="whole", threshold=.5, probability=.8):
    return {"candidate_id": cid, "bbox": box or [0, 0, 10, 10], "crop_rgb_sha256": cid,
            "methods": ["generator"], "origins": [{"method": "generator", "tile_id": view, "threshold": threshold}],
            "ocr": {"raw_text": text, "token_probabilities": [probability] * len(text),
                    "token_log_probabilities": [float(np.log(probability))] * len(text)}}


def groups(rows):
    return [{"group_id": "group", "candidate_ids": [r["candidate_id"] for r in rows],
             "reference_candidate_id": rows[0]["candidate_id"], "reference_bbox": rows[0]["bbox"]}]


def evidence(rows):
    return group_evidence(rows, groups(rows), allowed_methods={"generator"})


def test_normalization_preserves_digits_accents_spaces_and_handles_casefold():
    assert normalize("Straße! €É-12") == "strasse é12"
    assert normalize("$+-") == ""


def test_normalized_word_aliases_and_per_character_length_normalization():
    rows = [row("a", "WORD!"), row("b", "word"), row("c", "muchlonger")]
    e = evidence(rows)[0]
    assert set(e["words"]) == {"word", "muchlonger"}
    assert e["candidates"]["a"]["scores"]["confidence_mean"] == pytest.approx(.8)
    assert e["candidates"]["c"]["scores"]["confidence_mean"] == pytest.approx(.8)


def test_many_correlated_crops_cannot_overwhelm_independent_views():
    rows = [row(f"dup{i}", "wrong", threshold=.4 + i * .001) for i in range(30)]
    rows += [row("second", "right", view="windowA"), row("third", "right", view="windowB")]
    e = evidence(rows)[0]
    assert e["words"]["wrong"]["consensus"] == pytest.approx(1 / 3)
    assert e["words"]["right"]["consensus"] == pytest.approx(2 / 3)
    assert select_groups([e], "consensus")[0]["word"] == "right"


def test_threshold_interval_weights_not_crop_count():
    assert threshold_stability([.4, .5, .6], {.4, .5, .6}) == pytest.approx(1)
    assert threshold_stability([.4, .5, .6], {.5}) == pytest.approx(.5)
    assert threshold_stability([.4, .49, .5, .51, .6], {.49, .5, .51}) == pytest.approx(.55)
    assert threshold_stability([], set()) == 1


def test_soft_support_rewards_coverage_and_penalizes_foreign_foreground():
    p = np.zeros((10, 10), np.float32)
    p[2:8, 1:5] = 1
    p[2:8, 7:9] = 1
    mass = ProbabilityMass(p, (10, 10))
    reference = [1, 2, 5, 8]
    assert completeness(reference, reference, mass) == 1
    assert completeness([1, 2, 3, 8], reference, mass) < 1
    assert completeness([1, 2, 9, 8], reference, mass) < 1
    # Padding over true background carries no arbitrary border penalty.
    assert completeness([0, 1, 6, 9], reference, mass) == 1


def test_letterbox_probability_mass_fractional_coordinates_and_tile_alignment():
    p = np.zeros((8, 8), np.float32)
    p[2:6] = 1
    mass = ProbabilityMass(p, (8, 4))
    assert mass([0, 0, 8, 4]) == pytest.approx(32)
    assert mass([.25, .25, 1.25, 1.25]) == pytest.approx(1)
    whole = np.zeros((8, 8), np.float32)
    tile = np.ones((8, 8), np.float32)
    avg = aligned_probability(whole, [(tile, [4, 0, 8, 4])], (8, 4))
    assert np.all(avg[2:6, :4] == 0)
    assert np.all(avg[2:6, 4:] == .5)
    assert np.all(avg[:2] == 0)


def test_shared_crop_assignment_maximizes_total_without_duplicate_use():
    rows = [row("shared", "same"), row("other", "same", probability=.7)]
    base = evidence(rows)[0]
    a = json.loads(json.dumps(base)); a["group_id"] = "a"
    b = json.loads(json.dumps(base)); b["group_id"] = "b"; del b["candidates"]["other"]
    result = select_groups([a, b], "confidence_mean")
    assert len(result) == 2
    assert {r["candidate_id"] for r in result} == {"shared", "other"}
    assert [r["word"] for r in result] == ["same", "same"]


def test_selection_is_invariant_to_annotated_extra_fields_and_input_order():
    rows = [row("a", "first", probability=.6), row("b", "second", probability=.9)]
    before = select_groups(evidence(rows), "combined")
    for r in rows:
        r["ground_truth"] = "first"; r["dataset"] = "made_up_dataset"
    assert select_groups(evidence(list(reversed(rows))), "combined") == before


def test_cross_shortlist_and_scores_use_observed_raw_texts_and_eos_scores():
    rows = [row("a", "WORD!", view="one"), row("b", "other", view="two", probability=.7)]
    e = evidence(rows)
    requests, plan = cross_requests(e, rows, word_budget=2, crop_budget=2)
    assert len(requests) == 4
    assert {r["raw_text"] for r in requests} == {"WORD!", "other"}
    scores = {r["request_id"]: -.1 if r["raw_text"] == "other" else -1 for r in requests}
    add_cross_scores(e, plan, scores)
    assert select_groups(e, "cross_parseq")[0]["word"] == "other"


def test_clip_full_and_group_use_same_competitor_image_and_preserve_entire_input(monkeypatch):
    import recognition.group_word_selection_runtime as runtime
    image = np.zeros((4, 12, 3), np.uint8); image[:, 0] = 255; image[:, -1] = 127
    square = np.array(square_canvas(image))
    assert square.shape == (12, 12, 3)
    assert np.array_equal(square[4:8], image)
    e = evidence([row("a", "alpha", box=[0, 0, 12, 4]), row("b", "beta", box=[0, 0, 12, 4])])
    seen = []
    def embeddings(model, processor, *, images, texts, device, batch_size):
        if images is not None:
            seen.append(len(images)); return np.array([[1., 0.], [0., 1.]])
        return np.array([[1., 0.] if text == "alpha" else [0., 1.] for text in texts])
    monkeypatch.setattr(runtime, "_embeddings", embeddings)
    scores = clip_group_scores(None, None, image, e, templates={"raw": "{text}"}, device="cpu", batch_size=2)
    add_clip_scores(e, scores)
    assert seen == [2]  # One whole scene, one fixed group crop.
    assert select_groups(e, "clip_full_raw")[0]["word"] == "alpha"
    assert select_groups(e, "clip_group_raw")[0]["word"] == "beta"


def test_clip_declares_rgb_channel_order_for_tiny_reference_crop(monkeypatch):
    from recognition import group_word_selection_runtime as runtime
    image = np.zeros((8, 8, 3), np.uint8)
    rows = [row("a", "word", box=[0, 0, 3, 3])]
    group = evidence(rows)
    calls = []

    def processor(*, images, return_tensors, input_data_format):
        calls.append((len(images), return_tensors, input_data_format,
                      [item.size for item in images]))
        return {"pixel_values": torch.zeros(len(images), 3, 224, 224)}

    def embeddings(model, supplied_processor, *, images, texts, device, batch_size):
        if images is not None:
            supplied_processor(images=images, return_tensors="pt")
            return np.ones((len(images), 2)) / np.sqrt(2)
        return np.ones((len(texts), 2)) / np.sqrt(2)

    monkeypatch.setattr(runtime, "_embeddings", embeddings)
    scores = clip_group_scores(None, processor, image, group,
        templates={"raw": "{text}"}, device="cpu", batch_size=16)
    assert calls == [(2, "pt", "channels_last", [(8, 8), (3, 3)])]
    assert scores[group[0]["group_id"]]["word"]["clip_group_raw"] == pytest.approx(1)


class TinyTokenizer:
    eos_id, pad_id = 0, 4
    def encode(self, texts, device):
        values = [[3] + [1 if c == "a" else 2 for c in text] + [0] for text in texts]
        length = max(map(len, values))
        return torch.tensor([v + [4] * (length - len(v)) for v in values], device=device)


def test_teacher_forcing_uses_candidate_prefix_includes_eos_and_excludes_padding():
    class Model:
        max_label_length = 5
        def encode(self, images):
            self.encoded_count = len(images)
            return torch.zeros(len(images), 1, 1)
        def decode(self, target, memory, **kwargs):
            self.prefix = target.clone()
            # Three-class output: EOS, a, b. Constant scores permit hand calculation.
            return torch.tensor([.2, .6, .2]).log().expand(len(target), target.shape[1], 3)
        def head(self, value):
            return value
    model = Model()
    system = SimpleNamespace(model=model, tokenizer=TinyTokenizer(),
                             generate_attn_masks=lambda order: (None, None))
    recognizer = SimpleNamespace(model=system, config=SimpleNamespace(device="cpu"),
                                 transform=lambda image: torch.zeros(3, 2, 2))
    image = np.zeros((2, 2, 3), np.uint8)
    result = forced_likelihood_batch(recognizer, [image, image], ["a", "bb"], ["same", "same"])
    assert model.encoded_count == 1
    assert model.prefix.tolist() == [[3, 1, 0], [3, 2, 2]]
    assert result[0]["mean_log_probability_with_eos"] == pytest.approx((np.log(.6) + np.log(.2)) / 2)
    assert result[1]["mean_log_probability_with_eos"] == pytest.approx(np.log(.2))
    assert len(result[0]["token_log_probabilities_with_eos"]) == 2

    # Greedy no-EOS output can exceed the teacher-forcing positional capacity.
    result = forced_likelihood_batch(recognizer, [image, image], ["aaaaaa", "a"], ["same", "same"])
    assert result[0]["status"] == "UNSUPPORTED_LENGTH"
    assert result[0]["raw_text"] == "aaaaaa" and result[0]["mean_log_probability_with_eos"] is None
    assert result[1]["status"] == "SCORED"
    e = evidence([row("too_long", "aaaaaa")])
    requests, plans = cross_requests(e, [row("too_long", "aaaaaa")], word_budget=1, crop_budget=1)
    add_cross_scores(e, plans, {r["request_id"]: None for r in requests})
    assert select_groups(e, "cross_parseq") == []
    assert select_groups(e, "confidence_mean")[0]["word"] == "aaaaaa"


def test_metric_counts_repeated_words_spatially_once_and_keeps_empty_gt_denominator():
    predictions = [{"raw_text": "WORD!", "bbox": [0, 0, 10, 10]},
                   {"raw_text": "word", "bbox": [0, 0, 10, 10]}]
    gt = [{"text": "word", "bbox": [0, 0, 10, 10]}, {"text": "$", "bbox": [20, 0, 30, 10]}]
    result = evaluate_predictions(predictions, gt, .5)
    assert result["correct"] == 1 and result["predictions"] == 2


def test_oom_failure_is_logged_and_propagated(tmp_path):
    with pytest.raises(torch.cuda.OutOfMemoryError):
        with measured(tmp_path, "model", "cpu", {"model_set": ["test"], "batch_size": 1}):
            raise torch.cuda.OutOfMemoryError("synthetic OOM boundary test")
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["out_of_memory"] is True and failure["stage"] == "model"


def test_configuration_has_fixed_general_rules_and_rejects_strict_and_tuned_weights(tmp_path):
    path = REPO / "configs/approximate/crop_group_word_selection_v1.json"
    c = config(path)
    assert "dataset" not in c and "weights" not in c
    strict = json.loads(json.dumps(c)); strict["reproduction_mode"] = "strict"
    tuned = json.loads(json.dumps(c)); tuned["weights"] = {"consensus": .78}
    with pytest.raises(ValueError):
        validate_config(strict)
    with pytest.raises(ValueError):
        validate_config(tuned)
    # A config outside the scientific namespace cannot silently become a default.
    other = tmp_path / "config.json"; other.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="configs/approximate"):
        config(other)


def test_invalid_scores_and_empty_predictions_are_explicit():
    assert select_groups(evidence([row("empty", "$!")]), "combined") == []
    broken = row("a"); broken["ocr"]["token_probabilities"] = [1.2] * 4
    with pytest.raises(ValueError, match="character scores"):
        evidence([broken])
