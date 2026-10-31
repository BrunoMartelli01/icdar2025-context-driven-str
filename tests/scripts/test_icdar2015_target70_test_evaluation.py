"""Unchanged TEST populations, exact qualification binding and raw-cache integrity."""

import copy
import json

import numpy as np
import pytest

from scripts.evaluate_icdar2015_target70 import load_test_config,require_ocr_qualification,reconstruct
from scripts.evaluate_icdar2015_improvement_test import crop_hash
from segmentation.data import file_sha256
from segmentation.icdar2015_improvement_runtime import REPO
from segmentation.icdar2015_improvements import binary_proposals


def test_test_profile_rejects_ignore_censoring_and_reduced_denominators(tmp_path):
    path=REPO/"configs/approximate/icdar2015_target70_word_f1_v1.json"
    config=load_test_config(path)
    root=tmp_path/"approximate";root.mkdir();candidate=root/"profile.json"
    for key,value in (("expected_images",499),("expected_care_words",2000),
            ("prediction_ignore_policy","remove_predictions_in_ignored_regions")):
        candidate.write_text(json.dumps({**config,key:value}))
        with pytest.raises(ValueError,match="unchanged|Unchanged"):load_test_config(candidate)


def test_qualification_must_belong_to_the_exact_selected_policy_and_all_validation():
    policy={"raw_aggregate":{"images":100},"emitted_aggregate":{"images":100}}
    chosen={"policy_report_sha256":"selected report","policy":policy}
    qualification={"status":"PASS","images":100,"care_words":430,"failed_samples":0,"oom":False,
        "text_differences":0,"token_id_differences":0,"output_policy_differences":0,
        "protocol":{"operating_results_sha256":"selected report","selected":policy,"device":"cuda:0"},
        "raw_aggregate":policy["raw_aggregate"],"emitted_aggregate":policy["emitted_aggregate"]}
    require_ocr_qualification(qualification,chosen,"cuda:0")
    for update in ({"images":99},{"output_policy_differences":1},{"token_id_differences":1}):
        with pytest.raises(ValueError,match="exact|Exact"):
            require_ocr_qualification({**qualification,**update},chosen,"cuda:0")
    changed=copy.deepcopy(chosen);changed["policy_report_sha256"]="different report"
    with pytest.raises(ValueError,match="exact|Exact"):require_ocr_qualification(qualification,changed,"cuda:0")


def test_cache_reopening_must_retain_every_component_including_singletons(tmp_path):
    rgb=np.full((64,96,3),127,np.uint8)
    probability=np.zeros((32,48),np.float32);probability[10:14,15:30]=1;probability[0,0]=1
    prediction=probability>=.5;proposals,_=binary_proposals(prediction,(96,64),(64,96))
    for p in proposals:
        x1,y1,x2,y2=p["bbox"];p["crop_rgb_sha256"]=crop_hash(rgb[y1:y2,x1:x2])
    path=tmp_path/"mask.npz";np.savez_compressed(path,probability=probability,prediction=prediction)
    shard={"mask_path":"mask.npz","mask_sha256":file_sha256(path),"image_wh":[96,64],"proposals":proposals}
    policy={"probability_threshold":.5,"crop_method":"axis0"}
    assert len(reconstruct(shard,rgb,tmp_path,(64,96),policy))==2
    shortened={**shard,"proposals":proposals[:1]}
    with pytest.raises(ValueError,match="All raw components"):reconstruct(shortened,rgb,tmp_path,(64,96),policy)
    changed=rgb.copy();changed[0,0]=0
    with pytest.raises(ValueError,match="crop changed"):reconstruct(shard,changed,tmp_path,(64,96),policy)
