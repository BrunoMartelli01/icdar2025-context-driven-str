"""Prediction-only lexicon qualification and strict identity/failure boundaries."""

import hashlib

import pytest

from recognition.deepsolo_lexicon_deployment import DeepSoloVocabularyPostprocessor


def setup(tmp_path):
    v, p = tmp_path/"v.txt", tmp_path/"p.txt"
    v.write_text("CAT\n")
    p.write_text("CAT CAT\n")
    profile = dict(mode="approximate", claim_label="approximate reproduction", lexicon_kind="Generic",
        checkpoint_sha256="a"*64, neural=dict(score_threshold=.3, recognition_threshold=0.),
        lexicon_paths=dict(vocabulary=str(v), pairs=str(p)),
        lexicon_sha256=dict(vocabulary=hashlib.sha256(v.read_bytes()).hexdigest(),
                            pairs=hashlib.sha256(p.read_bytes()).hexdigest()),
        distance_exclusive=1.5, unmatched_policy="keep_unmatched")
    probabilities = [[float(i == n) for i in range(38)] for n in [2, 14, 19]]
    detection = dict(source_index=0, text="cot", score=.8, geometry={"preserve":True},
                     raw=dict(recognition_class_probabilities=probabilities))
    response = dict(success=True, artifact=dict(checkpoint_sha256="a"*64), detections=[detection])
    return profile, response


def test_corrects_native_response_without_targets_and_preserves_geometry(tmp_path):
    profile, response = setup(tmp_path)
    out = DeepSoloVocabularyPostprocessor(profile).apply(response)
    assert out["recognized_strings"] == ["CAT"]
    assert out["detections"][0]["geometry"] == {"preserve":True}
    assert response["detections"][0]["text"] == "cot"


def test_failure_and_checkpoint_identity_are_enforced(tmp_path):
    profile, response = setup(tmp_path)
    model = DeepSoloVocabularyPostprocessor(profile)
    response["artifact"]["checkpoint_sha256"] = "b"*64
    with pytest.raises(ValueError):
        model.apply(response)
    response["success"] = False
    with pytest.raises(RuntimeError):
        model.apply(response)


def test_changed_lexicon_fails_before_prediction(tmp_path):
    profile, _ = setup(tmp_path)
    with open(profile["lexicon_paths"]["vocabulary"], "a") as f:
        f.write("DOG\n")
    with pytest.raises(ValueError):
        DeepSoloVocabularyPostprocessor(profile)
