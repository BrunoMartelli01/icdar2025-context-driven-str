"""Grounded inference boundaries: spatial completeness, local controls and schemas."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from recognition.grounded_context_router import (
    feature_sets, inventory_features, local_views, normalized_text, similarity,
    verification_features, verification_jobs,
    generation_incomplete,
)


SETTINGS = json.loads((Path(__file__).resolve().parents[2]/'configs/approximate/grounded_context_router_v1.json').read_text())['geometry_settings']


def row(cid, box, text):
    return {'candidate_id':cid,'bbox':box,'raw_text':text}


def inventory(quads, texts):
    return {'quad_boxes':quads,'labels':texts}


EMPTY_OBJECTS={'bboxes':[],'labels':[]}


def test_missing_line_is_visible_even_when_existing_words_match():
    rows=[row('a',[0,0,20,9],'OPEN')]
    inv=inventory([[0,0,19,0,19,8,0,8],[0,20,19,20,19,28,0,28]],['OPEN','NOW'])
    f,e=inventory_features(inv,EMPTY_OBJECTS,rows,[40,40],SETTINGS)
    assert f['inventory_log_missing_regions']>0 and f['inventory_lexical_disagreement']>0
    assert e['ocr_regions'][1]['candidate_ids']==[]


def test_many_yolo_words_can_support_one_independent_line():
    rows=[row('b',[12,0,20,9],'NOW'),row('a',[0,0,10,9],'OPEN')]
    f,e=inventory_features(inventory([[0,0,20,0,20,8,0,8]],['OPEN NOW']),EMPTY_OBJECTS,rows,[30,30],SETTINGS)
    assert e['ocr_regions'][0]['candidate_ids']==['a','b']
    assert f['inventory_lexical_disagreement']==0
    assert f['inventory_unsupported_yolo_fraction']==0


def test_quad_coverage_is_not_its_bounding_box_coverage():
    # A diamond shares its bbox with large empty corners. Corner coverage cannot count as text coverage.
    rows=[row('corner',[0,0,2,2],'X')]
    f,e=inventory_features(inventory([[10,0,20,10,10,20,0,10]],['SIGN']),EMPTY_OBJECTS,rows,[21,21],SETTINGS)
    assert e['ocr_regions'][0]['coverage']==0
    assert f['inventory_unsupported_yolo_fraction']==1


def test_punctuation_repetition_and_unicode_are_not_erased():
    assert normalized_text('  Straße  OPEN OPEN! ' )=='strasse open open!'
    assert similarity('OPEN OPEN','OPEN')<1
    assert similarity('OPEN!','OPEN')<1


def test_outside_independent_prediction_is_reported_and_empty_inventory_is_missing():
    f,e=inventory_features(inventory([[30,30,35,30,35,35,30,35]],['X']),EMPTY_OBJECTS,[],[20,20],SETTINGS)
    assert f['inventory_invalid_regions']==1 and f['inventory_empty']==1
    assert not e['ocr_regions'][0]['valid']


def test_object_inventory_only_keeps_explicit_text_bearing_roles():
    objects={'bboxes':[[0,0,20,20],[0,0,20,20]],'labels':['a tree','a sign on a wall']}
    f,e=inventory_features(inventory([],[]),objects,[],[20,20],SETTINGS)
    assert len(e['text_objects'])==1 and f['inventory_log_empty_text_objects']>0


@pytest.mark.parametrize('bad',[[0,0,21,3],[4,3,2,5],[0.,0,5,5]])
def test_bad_original_geometry_fails(bad):
    with pytest.raises(ValueError): inventory_features(inventory([],[]),EMPTY_OBJECTS,[row('a',bad,'A')],[20,20],SETTINGS)


def test_probe_selects_unsupported_word_without_labels_and_retains_local_context():
    rows=[row('a',[0,10,10,20],'OPEN'),row('b',[12,10,24,20],'NOON')]
    _,e=inventory_features(inventory([[0,10,10,10,10,19,0,19]],['OPEN']),EMPTY_OBJECTS,rows,[30,30],SETTINGS)
    jobs=verification_jobs(rows,[{'candidate_id':'a','visual':.9},{'candidate_id':'b','visual':.8}],e,[30,30],SETTINGS)
    assert len(jobs)==1 and jobs[0]['candidate_id']=='b'
    assert jobs[0]['line_candidate_ids']==['a','b']
    assert jobs[0]['alternatives']==['NOON','NOOX']


def test_local_erasure_preserves_every_pixel_outside_target():
    rgb=np.arange(30*40*3,dtype=np.uint16).reshape(30,40,3).astype(np.uint8); original=rgb.copy()
    job={'context_box':[5,5,30,25],'target_box':[10,8,20,18]}
    views=local_views(rgb,job,128)
    mask=np.zeros((20,25),bool);mask[3:13,5:15]=True
    assert np.array_equal(views['local'][~mask],views['erased'][~mask])
    assert np.all(views['erased'][mask]==128) and np.all(views['blank']==128)
    assert np.array_equal(rgb,original)


def test_visual_control_exposes_language_prior_and_unchecked_words():
    same=[{'yes_probability':.9},{'yes_probability':.1}]
    probes=[{'answers':{k:copy.deepcopy(same) for k in ['local','erased','blank']}}]
    f=verification_features(probes,10)
    assert f['verification_support_mean']==.9 and f['verification_visual_margin_mean']==0
    assert f['verification_blank_corrected_mean']==0 and f['verification_fraction']==.1
    assert verification_features([],0)['verification_missing']==1


def test_ablation_schemas_keep_raw_visual_features_out_of_core():
    v,s=feature_sets({'quality':.5},{'core':['quality']},{'missing_region':1.},{'checked':1.},[.2,.3])
    assert s['core']==['quality'] and s['visual']==['quality','vision_0','vision_1']
    assert s['verified']==['quality','missing_region','checked']
    assert set(s['combined'])==set(v)
    with pytest.raises(ValueError): feature_sets({}, {'core':[]},{},{},[float('nan')])


def test_ground_truth_is_not_an_inference_argument():
    with pytest.raises(TypeError):
        inventory_features(inventory([],[]),EMPTY_OBJECTS,[],[1,1],SETTINGS,ground_truth=['OPEN'])


def test_florence_parser_excludes_only_structural_decoder_start():
    from recognition.grounded_context_runtime import florence_parser_input
    class Processor:
        def batch_decode(self,tokens,skip_special_tokens):
            assert not skip_special_tokens
            return [','.join(str(i) for i in tokens[0])]
    assert florence_parser_input(Processor(),np.array([[2,0,55,2,66,2]]),2)=='0,55,2,66,2'
    with pytest.raises(ValueError): florence_parser_input(Processor(),np.array([[0,55,2]]),2)


def test_new_visual_probe_propagates_oom():
    import torch
    from recognition.grounded_context_runtime import FrozenLocalVerifier
    from types import SimpleNamespace
    verifier=FrozenLocalVerifier.__new__(FrozenLocalVerifier)
    verifier.model=SimpleNamespace(training=False,parameters=lambda:[])
    def oom(_): raise torch.cuda.OutOfMemoryError('injected local probe failure')
    verifier.queries=oom
    with pytest.raises(torch.cuda.OutOfMemoryError,match='injected local probe failure'):
        verifier.verify({'local':np.zeros((2,2,3),np.uint8)},{'alternatives':['A','B']},'{word}')


def test_test_inventory_cannot_precede_frozen_choices(tmp_path,monkeypatch):
    from scripts.run_grounded_context_router import inventory_stage
    from recognition import grounded_context_runtime
    def forbidden_load(*args,**kwargs): raise AssertionError('Model must not load before freeze validation')
    monkeypatch.setattr(grounded_context_runtime,'FrozenInventory',forbidden_load)
    (tmp_path/'resolved_configuration.json').write_text('{}')
    with pytest.raises(ValueError,match='Freeze TRAIN choices'):
        inventory_stage({}, {}, tmp_path,'test',{})


@pytest.mark.parametrize('ended,length,expected',[(True,12,False),(False,12,True),(True,1024,True)])
def test_forced_eos_at_limit_is_not_natural_completion(ended,length,expected):
    assert generation_incomplete({'ended_with_eos':ended,'generated_tokens':length},1024)==expected


def test_explicit_inventory_reuse_preserves_raw_bytes_and_rejects_drift(tmp_path):
    from scripts.run_grounded_context_router import inventory_stage
    from scripts.run_yolo_f1_quality_study import sha
    source=tmp_path/'source';target=tmp_path/'target'
    rawpath=source/'train/inventory/a.json';rawpath.parent.mkdir(parents=True)
    raw={'plan_sha256':'original','sample_id':'a','image_sha256':'rgb','ground_truth_used':False,
         'ocr_predictions_given_to_model':False,'frozen_choices_sha256':None,'tasks':{'raw':'UNCHANGED'}}
    rawpath.write_text(json.dumps(raw));receiptpath=source/'train/inventory_receipt.json'
    receiptpath.write_text(json.dumps({'plan_sha256':'original','seconds':5,'model':{'optimizer_steps':0}}))
    target.mkdir();(target/'resolved_configuration.json').write_text('{}')
    reuse={'root':str(source),'plan_sha256':'original','receipt_sha256':sha(receiptpath),
           'files_sha256':{'train/inventory/a.json':sha(rawpath)}}
    observed={'a':{'sample':{'image_sha256':'rgb'}}}
    inventory_stage({}, {'reused_train_inventory':reuse},target,'train',observed)
    rebound=json.loads((target/'train/inventory/a.json').read_text())
    assert rebound['tasks']==raw['tasks'] and rebound['reused_source_sha256']==sha(rawpath)
    second=tmp_path/'second';second.mkdir();(second/'resolved_configuration.json').write_text('{}')
    rawpath.write_text(json.dumps({**raw,'tasks':{'raw':'DRIFT'}}))
    with pytest.raises(ValueError,match='Frozen input changed'):
        inventory_stage({}, {'reused_train_inventory':reuse},second,'train',observed)
