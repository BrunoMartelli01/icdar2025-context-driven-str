import copy

import numpy as np
import pytest

from recognition.clip_router_evaluation import (
    aggregate_split, deepsolo_predictions, evaluate_predictions, fixed_image_score,
)
from recognition.clip_image_router import accepted
from scripts.run_clip_router_e2e import cached, prepare, save_shard, validate_deepsolo


def prediction(text="word", bbox=None, ignored=False):
    return {"text":text,"bbox":bbox or [0,0,10,10],"ignored":ignored}


def test_matching_is_one_to_one_and_requires_geometry_and_exact_text():
    gt=[{"text":"Word!","bbox":[0,0,10,10]}]
    outputs=[prediction("WORD!"),prediction("word!"),prediction("word"),prediction("word!",[50,50,60,60])]
    metrics=evaluate_predictions(outputs,gt,iou_threshold=.5)
    assert (metrics["tp"],metrics["fp"],metrics["fn"])==(1,3,0)


def test_ignore_empty_and_invalid_predictions_do_not_hide_misses():
    gt=[{"text":"word","bbox":[0,0,10,10]}]
    assert evaluate_predictions([],gt,iou_threshold=.5)["fn"]==1
    assert evaluate_predictions([prediction(ignored=True)],gt,iou_threshold=.5)["detected_words"]==0
    assert evaluate_predictions([{"text":"word","bbox":None,"ignored":False}],gt,iou_threshold=.5)["fp"]==1


def test_router_uses_fixed_reference_and_all_words():
    reference={"native":[.1,.5,.9],"word_advantage":[.1,.5,.9]}
    rows=[{"raw_text":"word","confidence":.9,"clip_word_advantage":.9},
          {"raw_text":"bad","confidence":.1,"clip_word_advantage":.1}]
    before=copy.deepcopy(reference)
    assert fixed_image_score(rows,reference)==pytest.approx(1/6)
    assert not accepted(fixed_image_score(rows,reference),.2)
    assert accepted(fixed_image_score(rows,reference),1/6)
    assert reference==before
    assert fixed_image_score([],reference) is None
    assert fixed_image_score([{**rows[0],"raw_text":" "}],reference) is None
    with pytest.raises(ValueError,match="native"):
        fixed_image_score([{**rows[0],"confidence":float('nan')}],reference)


def response(points,score=.4):
    return {"detections":[{"source_index":0,"score":score,"text":"word",
                           "geometry":{"boundary_points":points}}]}


def test_deepsolo_inclusive_filter_and_empty_envelope_as_fp():
    validity=np.ones((256,256),bool)
    assert len(deepsolo_predictions(response([[1,1],[5,1],[5,5]]),[10,10],1,validity,.4))==1
    assert not deepsolo_predictions(response([[1,1],[5,1],[5,5]],.399),[10,10],1,validity,.4)
    preds=deepsolo_predictions(response([[20,20],[25,20],[25,25]]),[10,10],1,validity,.4)
    assert preds[0]["bbox"] is None and not preds[0]["ignored"]
    assert evaluate_predictions(preds,[],iou_threshold=.5)["fp"]==1
    with pytest.raises(ValueError,match="invalid detector"):
        deepsolo_predictions(response([[1,1],[float('nan'),1],[5,5]]),[10,10],1,validity,.4)


def test_polygon_ignore_does_not_use_bbox_support():
    # Care pixels occur inside the bbox but outside the triangular polygon.
    validity=np.zeros((256,256),bool);validity[200:210,200:210]=True
    triangle=[[0,0],[9,0],[0,9]]
    pred=deepsolo_predictions(response(triangle),[10,10],1,validity,.4)[0]
    assert pred["ignored"]
    validity[10:20,10:20]=True
    assert not deepsolo_predictions(response(triangle),[10,10],1,validity,.4)[0]["ignored"]


def test_inverse_exif_before_spatial_matching():
    # Orientation6: forward stored(x,y)->(height-1-y,x).
    pred=deepsolo_predictions(response([[8,2],[8,5],[5,5],[5,2]]),[10,20],6,
        np.ones((256,256),bool),.4)[0]
    assert pred["bbox"]==[2,11,5,14]


def test_split_metrics_use_all_images_and_distinct_denominators():
    rows=[{"sample_id":"a","status":"SUCCESS","action":"accept_parseq",
           "metrics":evaluate_predictions([prediction(),prediction("bad")],[{"text":"word","bbox":[0,0,10,10]}],iou_threshold=.5)},
          {"sample_id":"b","status":"SUCCESS","action":"deepsolo",
           "metrics":evaluate_predictions([],[{"text":"missed","bbox":[0,0,10,10]}],iou_threshold=.5)}]
    agg=aggregate_split(rows,["a","b"],2)
    assert agg["accuracy_on_detected_words"]==.5
    assert agg["accuracy_tn0"]==pytest.approx(1/3)
    assert agg["deepsolo_images"]==1 and agg["images"]==2
    with pytest.raises(ValueError,match="Missing"):
        aggregate_split(rows[:1],["a","b"],2)
    with pytest.raises(ValueError,match="denominator"):
        aggregate_split(rows,["a","b"],3)
    with pytest.raises(ValueError,match="Failed"):
        aggregate_split([{**rows[0],"status":"FAILED"},rows[1]],["a","b"],2)


def test_raw_fallback_identity_and_threshold_preserve_evidence():
    raw={"schema_version":1,"request_id":"unit","success":True,"detections":[],"recognized_strings":[],
         "selected_output":"","reduction_policy":"highest_score","artifact":{
         "checkpoint_sha256":"ckpt","candidate":{"artifact_id":"candidate","qualified_config_sha256":"config"},
         "config_sha256":"config","inference_options":{"device":"cuda:0","score_threshold":.05,"lexicon_type":None},
         "resolved_inference_threshold":.05,"load_validation":{"strict_compatibility_check":True,
         "missing_keys":[],"unexpected_keys":[],"incorrect_shapes":[]}},
         "latency":{},"placement":{"device":"cuda:0"},"memory":{"peak":42},"error":None}
    cfg={"checkpoint_sha256":"ckpt","candidate":"candidate","device":"cuda:0"}
    assert validate_deepsolo(raw,cfg,.05)==raw
    with pytest.raises(ValueError,match="incompatible"):
        validate_deepsolo(raw,cfg,.4)
    bad=copy.deepcopy(raw);bad['artifact']['load_validation']['missing_keys']=['layer']
    with pytest.raises(ValueError,match="incompatible"):
        validate_deepsolo(bad,cfg,.05)


def test_cached_shards_detect_payload_and_contract_tampering(tmp_path):
    import json
    path=tmp_path/'shard.json';contract={"sample_id":"a"}
    save_shard(path,contract,{"value":1})
    assert cached(path,contract)['value']==1
    with pytest.raises(ValueError,match="contract"):
        cached(path,{"sample_id":"b"})
    row=json.loads(path.read_text());row['value']=2;path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match="payload"):
        cached(path,contract)


def test_routing_never_uses_labels_or_evaluation_ignore(tmp_path):
    # Actual stage boundary: frozen observed evidence only, labels can be opaque.
    word={"instance_id":1,"raw_text":"word","confidence":.9,"clip_word_advantage":.9}
    observed={"sample_id":"a","image_path":"unused","image_sha256":"abc","rows":[word]}
    cfg={"threshold":.5};router_cfg={"execution":{"batch_size":16}}
    reference={"native":[.1,.5,.9],"word_advantage":[.1,.5,.9]}
    scores=[]
    for index,ignored in enumerate(([],[1])):
        samples=[{"observed":observed,"ignored_ids":ignored,"ground_truth":object()}]
        routes=prepare('train',samples,tmp_path/str(index),'plan',cfg,router_cfg,{}, {},
            {"model_id":"unit"},reference,'cpu')
        assert routes[0]['action']=='accept_parseq' and routes[0]['ground_truth_in_routing'] is False
        scores.append(routes[0]['score'])
    assert scores[0]==scores[1]
