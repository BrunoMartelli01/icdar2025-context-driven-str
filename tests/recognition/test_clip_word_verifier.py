import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from recognition import clip_word_verifier as verifier
from scripts import run_clip_word_verification as runner
from scripts.crop_clip_runtime import array_sha


REPO = Path(__file__).resolve().parents[2]


def config():
    return runner.load_config(REPO / "configs/approximate/clip_word_verification_train_v1.json")


def fixture():
    image = np.zeros((12, 20, 3), np.uint8)
    image[4:8, 4:16] = 255
    rows = []
    for cid, box, baseline, pad in [("base", [4, 4, 16, 8], True, 0),
                                    ("alternative", [4, 4, 17, 8], False, 0),
                                    ("padded", [2, 2, 18, 10], False, 2)]:
        crop = image[box[1]:box[3], box[0]:box[2]]
        rows.append({"candidate_id": cid, "anchor_rank": 1, "baseline": baseline,
            "bbox": box, "crop_shape": list(crop.shape), "crop_array_sha256": array_sha(crop),
            "origins": [{"threshold": .5 if baseline else .4, "padding_native_pixels": pad}]})
    readings = [{"candidate_id": cid, "raw_text": text, "mean_token_log_probability": lp}
                for cid, text, lp in [("base", "LEFT", -.2), ("alternative", "RIGHT", -.1),
                                       ("padded", "PADDED", -.01)]]
    return image, {"candidates": rows}, {"rows": readings}


class Tokenizer:
    def __call__(self, values, **kwargs):
        assert kwargs == {"padding": False, "truncation": False}
        return {"input_ids": [list(range(len(text.split()) + 2)) for text in values]}


def test_word_pool_excludes_padded_readings_and_freezes_baseline_geometry():
    _, prepared, ocr = fixture()
    pool = verifier.word_pools(prepared, ocr)[0]
    assert list(pool["words"]) == ["LEFT", "RIGHT"]
    assert pool["reference"]["bbox"] == [4, 4, 16, 8]
    assert pool["baseline_text"] == "LEFT"
    ocr["rows"].pop()
    with pytest.raises(ValueError, match="coverage"):
        verifier.word_pools(prepared, ocr)


def test_invalid_parseq_confidence_is_not_silently_accepted():
    _, prepared, ocr = fixture()
    ocr["rows"][0]["mean_token_log_probability"] = float("nan")
    with pytest.raises(ValueError, match="probability"):
        verifier.word_pools(prepared, ocr)


def test_context_budget_preserves_words_and_applies_same_prefix_to_all_candidates():
    template = 'word "{text}". Scene: {caption}'
    prompts, budget = verifier.context_prompts(Tokenizer(), ["FIRST", "TWO WORDS"],
        "one two three four five six", template=template, max_tokens=9, max_caption_words=80)
    assert budget["used_words"] == 3
    assert '"TWO WORDS"' in prompts["TWO WORDS"]
    assert all(text.endswith("one two three") for text in prompts.values())
    assert max(budget["prompt_token_lengths"]) == 9
    with pytest.raises(ValueError, match="refusing word truncation"):
        verifier.context_prompts(Tokenizer(), ["a b c d e f g"], "context",
            template=template, max_tokens=4, max_caption_words=80)


def test_masked_surround_removes_letters_without_changing_field_of_view():
    image, prepared, ocr = fixture()
    pool = verifier.word_pools(prepared, ocr)[0]
    views, box = verifier.image_views(image, pool, 2)
    assert box == [0, 2, 20, 10]
    surround, masked = np.asarray(views["surround"]), np.asarray(views["masked_surround"])
    assert surround.shape == masked.shape == (20, 20, 3)
    assert np.count_nonzero(surround) > 0
    assert np.count_nonzero(masked) == 0
    bad = image.copy()
    bad[4, 4] = 0
    with pytest.raises(RuntimeError, match="pixels changed"):
        verifier.image_views(bad, pool, 2)


def test_shared_image_scores_candidates_instead_of_ranking_different_crops(monkeypatch):
    image, prepared, ocr = fixture()
    pools = verifier.word_pools(prepared, ocr)
    seen_images = []

    def embeddings(_model, _processor, *, images, texts, **kwargs):
        if texts is not None:
            return np.array([[1., 0.] if "LEFT" in text else [0., 1.] if "RIGHT" in text
                             else [2**-.5, 2**-.5] for text in texts], np.float32)
        seen_images.extend(images)
        return np.array([[0., 1.], [1., 0.], [.6, .8], [0., 1.]], np.float32)

    monkeypatch.setattr(verifier, "_embeddings", embeddings)
    model = SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(max_position_embeddings=77)))
    scored = verifier.score_words(model, SimpleNamespace(tokenizer=Tokenizer()), image, pools,
        caption="A street scene.", shuffled_caption="A beach.", config=config(), device="cpu")
    assert len(seen_images) == 4  # one scene + three fixed views, independent of candidate count
    assert scored["scores"]["1"]["LEFT"]["crop_raw"] == 1
    assert scored["scores"]["1"]["RIGHT"]["crop_raw"] == 0
    selected = verifier.selections(scored, config())
    assert selected["crop_raw"]["1"] == "LEFT"
    assert selected["full_raw"]["1"] == "RIGHT"
    assert selected["baseline"]["1"] == "LEFT"
    assert "ground_truth" not in scored


def test_normalized_context_residual_differs_from_subtracting_common_constant():
    image = np.array([1., 0.])
    neutral = np.array([0., 1.])
    first = np.array([.2, np.sqrt(.96)])
    second = np.array([.6, .8])
    # Subtracting a caption-only scalar preserves ranks; normalized feature
    # differences can change them. This is an experimental geometric operation.
    assert image @ second > image @ first
    assert image @ verifier.unit(first-neutral) > image @ verifier.unit(second-neutral)
    with pytest.raises(ValueError, match="degenerate"):
        verifier.unit(np.zeros(2))


def test_margin_equality_and_ties_retain_baseline_and_empty_cannot_win():
    pool = {"baseline_text": "base", "words": {"base": {}, "other": {}}}
    assert verifier.best_word(pool, {"base": .5, "other": .5}) == "base"
    assert verifier.best_word(pool, {"base": .5, "other": .75}, margin=.25) == "base"
    assert verifier.best_word(pool, {"base": .5, "other": .76}, margin=.25) == "other"
    with pytest.raises(ValueError, match="nonempty"):
        verifier.best_word(pool, {"": 1.})


def test_empty_predictions_remain_in_every_arm_and_do_not_change_denominator():
    pool = {"anchor_rank": 1, "baseline_text": "", "words": {"": {"parseq": None}}}
    selected = verifier.selections({"pools": [pool], "scores": {"1": {}}}, config())
    assert len(selected) == 28
    assert all(choices == {"1": ""} for choices in selected.values())


def test_evaluation_keeps_fixed_boxes_all_care_gt_and_native_baseline_confidence(monkeypatch):
    image, prepared, ocr = fixture()
    pools = verifier.word_pools(prepared, ocr)
    scores = {"1": {"LEFT": {"crop_ensemble": .1, "surround_ensemble": .1},
                      "RIGHT": {"crop_ensemble": .3, "surround_ensemble": .3}}}
    scored = {"pools": pools, "scores": scores}
    selected = verifier.selections(scored, config())
    monkeypatch.setattr(runner, "read_image", lambda *args: (image, "image"))
    monkeypatch.setattr(runner, "_train_gt", lambda *args: ([
        {"bbox": [4, 4, 16, 8], "text": "RIGHT"},
        {"bbox": [0, 0, 3, 3], "text": "unmatched"}], "annotations"))
    context = {"pools": pools, "image_sha256": "image", "caption": "A plain scene."}
    result = runner.evaluate([SimpleNamespace(sample_id="example")], [context],
        [{"scored": scored, "selections": selected}], paths={"data_root": Path("unused")}, cfg=config())
    assert result["summary"]["baseline"]["exact"] == 0
    assert result["summary"]["agreement_gate"]["exact"] == 1
    assert result["fixed_geometry_oracle_exact"] == 1
    assert all(v["care_words"] == 2 for v in result["summary"].values())
    assert all(rows[0]["bbox"] == [4, 4, 16, 8]
               for rows in result["per_image"][0]["predictions"].values())
    assert result["per_image"][0]["correctness_diagnostics"][0]["baseline_parseq"] == -.2


@pytest.mark.parametrize("labels,scores,expected", [
    ([True, False], [1., 0.], 1.), ([True, False], [0., 1.], 0.),
    ([True, False], [1., 1.], .5), ([True, True, False, False], [0., 2., 0., 1.], .625),
    ([True], [1.], None), ([], [], None)])
def test_auc_has_midrank_ties_and_no_invented_one_class_value(labels, scores, expected):
    assert verifier.auc(labels, scores) == expected


def test_config_namespace_and_model_revisions_fail_closed(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config()))
    with pytest.raises(ValueError, match="configs/approximate"):
        runner.load_config(path)
    spec = config()["models"]["vit_b16"]
    model = tmp_path / spec["revision"]
    model.mkdir()
    with pytest.raises(FileNotFoundError, match="pinned.*snapshot missing"):
        runner.model_identity(model, spec)


def test_oom_is_recorded_and_propagated_without_substitution(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "paths_profile", lambda _path: {
        "output_root": tmp_path, "models": {"vit_b32": tmp_path}})
    def failed_prepare(*args):
        raise torch.cuda.OutOfMemoryError("simulated allocation failure; no GPU measurement")
    monkeypatch.setattr(runner, "prepare", failed_prepare)
    args = argparse.Namespace(config=REPO/"configs/approximate/clip_word_verification_train_v1.json",
        paths=tmp_path/"paths.json", model="vit_b32", device="cpu", limit=1)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="simulated"):
        runner.run(args)
    report = json.loads((tmp_path/"vit_b32/smoke-1/failure.json").read_text())
    assert report["out_of_memory"] is True
    assert report["status"] == "FAILED"
    assert report["cuda_peak"]["peak_allocated_bytes"] is None
