from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from recognition.clip_crop_admission import (FEATURES, CropAdmissionScorer,
    admission_scores, choose_operating_point, filter_rows, fit_calibration,
    percentile, validate_rows)
from scripts.run_clip_high_recall_admission import (
    DEFAULT_CONFIG, checked, payload_digest, seal, validate_config,
)
from scripts.crop_clip_runtime import sha


def row(cid="a", confidence=.9, value=.2):
    return {"candidate_id":cid,"group_id":cid,"bbox":[0,0,3,3],"raw_text":"ABC","word":"abc",
            "confidence":confidence,"physical":.8,"clip_status":"SCORED",
            "clip":{f:value for f in FEATURES}}


def test_rank_calibration_is_label_free_tie_preserving_and_order_invariant():
    rows=[row("a",.8,.1),row("b",.9,.2),row("c",.9,.2)]
    a=fit_calibration(rows)
    for r in rows: r["ground_truth"]="arbitrary labels"
    assert a == fit_calibration(list(reversed(rows)))
    assert percentile(a["clip_glyph"],.1) == pytest.approx(1/6)
    assert percentile(a["clip_glyph"],.2) == pytest.approx(2/3)
    assert percentile(a["clip_glyph"],0) == 0
    assert percentile(a["clip_glyph"],1) == 1


@pytest.mark.parametrize("reference,value", [([],.1),([1,0],.1),([float("nan")],.1),([0],float("inf"))])
def test_invalid_calibration_fails(reference,value):
    with pytest.raises(ValueError): percentile(reference,value)


def test_rank_combination_uses_equal_geometric_means_and_preserves_native_controls():
    rows=[row("a",.8,.1),row("b",.9,.2)]
    calibration=fit_calibration(rows)
    scores=admission_scores(rows[1],calibration,.5)
    assert scores["confidence"] == .9
    assert scores["physical"] == .8
    assert scores["rank_clip_glyph"] == .75
    assert scores["native_glyph"] == pytest.approx(.75)
    assert scores["native_glyph_textness"] == pytest.approx(.75)


def test_prefiltered_rows_cannot_enter_clip_arms_even_at_threshold_zero():
    r=row(confidence=.2)
    r.update(clip_status="NATIVE_PREFILTER_REJECTED",clip=None)
    s=admission_scores(r,{},.5)
    assert filter_rows([r],[s],"native_glyph",0) == []
    assert filter_rows([r],[s],"confidence",0) == [r]
    r["clip_status"]="SCORED"
    with pytest.raises(ValueError,match="prefilter"): admission_scores(r,{},.5)


def test_filter_threshold_boundary_preserves_original_crop_reading_and_box():
    rows=[row("a",.5),row("b",.9)]
    before=deepcopy(rows)
    values=[{"confidence":.5},{"confidence":.9}]
    assert filter_rows(rows,values,"confidence",.5) == rows
    assert filter_rows(rows,values,"confidence",.50001) == [rows[1]]
    assert rows == before
    with pytest.raises(ValueError): filter_rows(rows,values[:1],"confidence",.5)


def test_operating_point_protects_baseline_identities_not_just_total_correct():
    summary={"family@0.0":{"precision_proxy":.2,"retained_baseline_correct":100,"predictions":500},
             "family@0.5":{"precision_proxy":.8,"retained_baseline_correct":95,"predictions":125},
             "family@0.9":{"precision_proxy":.9,"retained_baseline_correct":94,"predictions":110}}
    assert choose_operating_point(summary,"family",.95,100) == "family@0.5"
    assert choose_operating_point(summary,"missing",.95,100) is None


def test_row_validation_rejects_identity_coordinate_and_confidence_errors():
    r=row()
    validate_rows([r],(3,3))
    with pytest.raises(ValueError): validate_rows([r,r],(3,3))
    for changed in ({"bbox":[0,0,4,3]},{"confidence":float("nan")},{"physical":2},{"raw_text":""}):
        with pytest.raises(ValueError): validate_rows([{**r,**changed}],(3,3))


def test_isolated_config_controls_are_explicit():
    c=json.loads(DEFAULT_CONFIG.read_text())
    validate_config(c)
    for changed in ({"reproduction_mode":"strict"},{"table_5_claim":True},{"native_prefilter":2},
                    {"thresholds":[0,.5,.5]},{"word_templates":["no candidate"]}):
        with pytest.raises(ValueError): validate_config({**c,**changed})


def test_sealed_artifacts_and_payload_detect_corruption(tmp_path):
    p=tmp_path/"payload.json"
    value={"contract":{"id":"sample"},"score":.3}
    value["payload_sha256"]=payload_digest(value)
    seal(p,value)
    seal(p,value)
    assert checked(p,sha(p)) == value
    changed={**value,"score":.4}
    assert payload_digest(changed) != value["payload_sha256"]
    with pytest.raises(RuntimeError): seal(p,changed)
    with pytest.raises(RuntimeError): checked(p,"0"*64)


def test_rgb_encoder_declares_channels_last_for_tiny_crops(monkeypatch):
    import recognition.clip_crop_admission as module
    seen=[]
    def processor(**kwargs):
        seen.append(kwargs)
        return {}
    def embeddings(model,wrapped,**kwargs):
        wrapped(images=kwargs["images"],return_tensors="pt")
        return np.ones((1,3))
    monkeypatch.setattr(module,"_embeddings",embeddings)
    scorer=object.__new__(CropAdmissionScorer)
    scorer.model=None; scorer.processor=processor; scorer.device="cpu"; scorer.batch_size=16
    scorer.encode_images([Image.new("RGB",(3,3))])
    assert seen[0]["input_data_format"] == "channels_last"


def test_crop_scoring_uses_word_and_blank_contrasts_without_text_mutation():
    scorer=object.__new__(CropAdmissionScorer)
    scorer.prepare=lambda words:None
    scorer.encode_images=lambda images:np.array([[1.,0.]])
    scorer.texts={"ABC":{"written":np.array([1.,0.]),"ensemble":np.array([1.,0.])}}
    scorer.glyphs={"ABC":np.array([1.,0.])}
    scorer.probes={"positive":np.array([1.,0.]),"negative":np.array([0.,1.]),"blank":np.array([0.,1.])}
    r=row(); before=deepcopy(r)
    scored=scorer.score([r],[np.zeros((3,3,3),np.uint8)])[0]
    assert scored == {feature:1. for feature in FEATURES}
    assert r == before
    with pytest.raises(ValueError): scorer.score([r],[])


def test_metric_counts_duplicate_crops_once_and_keeps_full_gt_denominator():
    from scripts.run_crop_group_word_selection import evaluate_predictions
    gt=[{"bbox":[0,0,3,3],"text":"ABC"},{"bbox":[4,0,7,3],"text":"OTHER"}]
    rows=[row("a"),row("b")]
    result=evaluate_predictions(rows,gt,.5)
    assert result["correct"] == 1 and result["predictions"] == 2
    assert result["correct_gt_indices"] == [0]


def test_runtime_memory_failure_is_recorded_and_propagated_without_fallback(monkeypatch,tmp_path):
    import scripts.run_clip_high_recall_admission as runner
    r=row()
    observation={"sample_id":"fixture","image_sha256":"0"*64,"image_wh":[3,3],"rows":[r]}
    monkeypatch.setattr(runner,"inputs",lambda *args:([observation],{"fixture":None},tmp_path,{},{}))
    monkeypatch.setattr(runner.base,"model_identity",lambda *args:{"fixture":True})
    monkeypatch.setattr(runner,"font_identity",lambda *args:{})
    def fail(*args): raise MemoryError("simulated memory failure; no real GPU inference")
    monkeypatch.setattr(runner,"read_image",fail)
    args=SimpleNamespace(config=DEFAULT_CONFIG,paths=runner.DEFAULT_PATHS,limit=1,
                         model="vit_b32",device="cpu",output_root=tmp_path,check=False)
    with pytest.raises(MemoryError,match="simulated memory failure"):
        runner.run(args)
    failures=list(tmp_path.rglob('failure.json'))
    assert len(failures) == 1
    failure=json.loads(failures[0].read_text())
    assert failure["out_of_memory"] is True
    assert failure["device"] == "cpu" and failure["precision"] == "float32"
    assert failure["stage"] == "image/crop_validation"
