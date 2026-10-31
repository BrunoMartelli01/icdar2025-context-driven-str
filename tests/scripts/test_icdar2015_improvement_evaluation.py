"""Dynamic-scale component/crop geometry and approximate evaluation isolation."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import label, find_objects

from scripts.evaluate_icdar2015_improvement_test import load_config, proposals
from scripts.evaluate_icdar2015_improvement_validation import verify_geometry_counts


ROOT=Path(__file__).resolve().parents[2]
PROFILE=ROOT/"configs/approximate/icdar2015_improvement_word_f1_v1.json"


@pytest.mark.parametrize("shape",[(512,512),(768,768),(576,1024)])
def test_every_component_and_area_matches_independent_scipy_at_selected_shape(shape):
    mask=np.zeros((shape[0]//2,shape[1]//2),dtype=bool)
    mask[100:104,30:50]=True
    mask[110,90]=True # a single pixel must count, too
    mask[120,100]=mask[121,101]=True # 8-connectivity
    config={**load_config(PROFILE),"image_hw":list(shape)}
    rows=proposals(mask,(1280,720),config)
    labels,number=label(mask,structure=np.ones((3,3)))
    expected=[]
    for component,slices in enumerate(find_objects(labels),1):
        yy,xx=slices
        expected.append((int((labels[slices]==component).sum()),(xx.start,yy.start,xx.stop,yy.stop)))
    assert number==len(rows)==3
    assert sorted(expected)==sorted((p["component_area"],tuple(p["bbox_native"])) for p in rows)
    assert [r["component_area"] for r in rows]==[80,2,1]
    assert all(0<=r["bbox"][0]<r["bbox"][2]<=1280 and 0<=r["bbox"][1]<r["bbox"][3]<=720 for r in rows)


def test_improved_test_profile_keeps_original_word_protocol_and_rejects_strict(tmp_path):
    config=load_config(PROFILE)
    assert config["expected_images"]==500 and config["expected_care_words"]==2077
    assert config["text_normalization"]=="unicode_casefold_then_isalnum"
    assert config["prediction_ignore_policy"]=="all_outputs_count_including_ignored_regions"
    strict=tmp_path/"strict";strict.mkdir();path=strict/"profile.json";path.write_text(json.dumps(config))
    with pytest.raises(ValueError,match="approximate"):load_config(path)


def test_frozen_geometry_rows_store_counts_and_native_box_replay_checks_them():
    boxes=[{"bbox":[0,0,10,10]},{"bbox":[20,0,30,10]}]
    words=[{"bbox":[0,0,10,10]}];config={"matching_iou_threshold":.5}
    row={"proposals":2,"matched_gt":1,"gt_words":1}
    verify_geometry_counts(row,boxes,words,config)
    with pytest.raises(ValueError,match="replay differs"):
        verify_geometry_counts({**row,"proposals":1},boxes,words,config)
    with pytest.raises(ValueError,match="replay differs"):
        verify_geometry_counts({**row,"matched_gt":0},boxes,words,config)
