import copy

import numpy as np
import pytest

from scripts import run_clip_image_router_word_advantage as fixed
from scripts.run_clip_image_router_fallback import validate_response


def test_word_background_contrast_checks_actual_word_and_preserves_empty_ocr(monkeypatch):
    # Basis vectors represent a visual word match and an unrelated background.
    # Reversing the word vector must reverse the score, with the same image.
    cfg={"execution":{"batch_size":16},"word_templates":["written {text}","sign {text}","printed {text}"],
         "textness_prompts":{"positive":["text"],"negative":["background"]}}
    def embeddings(model,processor,*,images,texts,device,batch_size):
        if images is not None:
            assert len(images)==2
            assert all(np.asarray(im).shape==(8,8,3) for im in images)
            return np.array([[1.,0.],[1.,0.]])
        if texts==["text","background"]:
            return np.eye(2)
        return np.array([[-1.,0.] if 'wrong' in t else [1.,0.] for t in texts])
    monkeypatch.setattr(fixed,"_embeddings",embeddings)
    scorer=fixed.WordAdvantageScorer(None,None,cfg,"cpu")
    rows=[{"raw_text":"word"},{"raw_text":"wrong"},{"raw_text":""}]
    before=copy.deepcopy(rows)
    crops=[np.zeros((4,8,3),np.uint8) for _ in rows]
    assert scorer.score(rows,crops)==[1.,-1.,None]
    assert rows==before


def response():
    # Contract-only fixture; not a substitute for inference or evidence.
    return {"schema_version":1,"request_id":"unit","success":True,"detections":[],"recognized_strings":[],
            "selected_output":"","reduction_policy":"reading_order_space_join","artifact":{
             "checkpoint_sha256":"abc","candidate":{"artifact_id":"candidate"},
             "inference_options":{"device":"cuda:0","score_threshold":.4,"lexicon_type":None}},
            "latency":{},"placement":{"device":"cuda:0"},"memory":{},"error":None}


def config():
    return {"device":"cuda:0","score_threshold":.4,"checkpoint_sha256":"abc","candidate":"candidate"}


def test_fallback_requires_exact_candidate_options_and_preserves_memory_records():
    value=response();value['memory']={'cuda_max_memory_allocated_bytes':42,'oom':False}
    assert validate_response(value,config())==value
    for field,replacement in [('device','cpu'),('score_threshold',.05),('checkpoint_sha256','different'),('candidate','other')]:
        cfg=config();cfg[field]=replacement
        with pytest.raises(RuntimeError):validate_response(value,cfg)


def test_fallback_never_accepts_unsuccessful_worker_response():
    value=response();value.update(success=False,error={"type":"RuntimeError","message":"failed"},selected_output=None)
    with pytest.raises((RuntimeError,ValueError)):validate_response(value,config())
