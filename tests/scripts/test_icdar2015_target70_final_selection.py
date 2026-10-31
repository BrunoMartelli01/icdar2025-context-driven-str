"""Completion gates and validation-only selection; fixtures are not model results."""

import copy
import json
from pathlib import Path

import pytest

from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO
from segmentation.icdar2015_target70_selection import completed_run,load_selection_profile,model_selection_key


def test_selection_profile_rejects_strict_namespace(tmp_path):
    original=REPO/"configs/approximate/icdar2015_target70_selection_v1.json"
    assert load_selection_profile(original)["include_qualified_source"]
    path=tmp_path/"strict/profile.json";path.parent.mkdir();path.write_bytes(original.read_bytes())
    with pytest.raises(ValueError,match="approximate"):load_selection_profile(path)


def fixture_report(tmp_path):
    profile={"training":{"epochs":100},"branches":{"long768":{}}}
    names=("models/icdar2015_target70.py","segmentation/icdar2015_target70.py","scripts/train_icdar2015_target70.py")
    checkpoint=tmp_path/"fixture.pt";checkpoint.write_bytes(b"engineering fixture, not model weights")
    report={"status":"SUCCESS","training_complete":True,"result_class":"approximate reproduction","completed_epochs":100,
        "checkpoint":{"path":"fixture.pt","sha256":file_sha256(checkpoint)},"run_contract":{"branch":"long768",
            "config_sha256":"profile fixture","config":profile,"branch_config":{},"smoke_steps":None,"precision":"float32",
            "train_ids":["fit"],"validation_ids":["val"],"evaluated_ids":["val"],
            "source_sha256":{name:file_sha256(REPO/name) for name in names}}}
    (tmp_path/"epochs.json").write_text(json.dumps([{"epoch":i,"train":{"samples":1}} for i in range(1,101)]))
    return report,profile


@pytest.mark.parametrize("changes",[{"training_complete":False},{"completed_epochs":99},{"result_class":"smoke-test result"}])
def test_partial_or_smoke_training_cannot_enter_final_selection(tmp_path,changes):
    report,profile=fixture_report(tmp_path);report.update(changes)
    (tmp_path/"run_manifest.json").write_text(json.dumps(report))
    with pytest.raises(ValueError,match="completed full training"):
        completed_run(tmp_path,"long768",profile,"profile fixture",["fit"],["val"])


def test_missing_epoch_or_changed_validation_population_is_rejected(tmp_path):
    report,profile=fixture_report(tmp_path)
    (tmp_path/"run_manifest.json").write_text(json.dumps(report))
    assert completed_run(tmp_path,"long768",profile,"profile fixture",["fit"],["val"])["completed_epochs"]==100
    with pytest.raises(ValueError,match="populations differ"):
        completed_run(tmp_path,"long768",profile,"profile fixture",["fit"],["different-val"])
    (tmp_path/"epochs.json").write_text(json.dumps([{"epoch":i,"train":{"samples":1}} for i in range(2,101)]))
    with pytest.raises(ValueError,match="history"):
        completed_run(tmp_path,"long768",profile,"profile fixture",["fit"],["val"])


def candidate(f1,raw=.4,shape=(768,768),epoch=56,kind="trained",name="long768"):
    return {"name":name,"kind":kind,"image_hw":list(shape),"effective_epoch":epoch,"policy":{
        "emitted_aggregate":{"alphanumeric_casefold":{"f1":f1,"recall":.5}},
        "raw_aggregate":{"alphanumeric_casefold":{"f1":raw}},"crop_method":"axis0","probability_threshold":.5}}


def test_correct_word_f1_precedes_resolution_and_source_preference():
    source=candidate(.6,kind="source",name="source768")
    improved=candidate(.7,raw=.3,shape=(736,1280),epoch=100,name="adapt1280")
    assert model_selection_key(improved)>model_selection_key(source)
    tied=copy.deepcopy(source);tied.update(kind="trained",name="long768")
    assert model_selection_key(source)>model_selection_key(tied)
    expensive=copy.deepcopy(tied);expensive["image_hw"]=[736,1280]
    assert model_selection_key(tied)>model_selection_key(expensive)
