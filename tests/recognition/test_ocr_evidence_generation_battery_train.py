import copy
import json
from pathlib import Path

import pytest

from diagnostics.ocr_evidence_generation_battery_train import (
    caption_view, choices_for, make_request, neighbor_indices, observation, parse_answer,
    validate_confidence, variants,
)
from diagnostics.lavis_captioner_candidates import digest


class Tokenizer:
    def encode(self,s,add_special_tokens=False): return s.split()
    def decode(self,ids,skip_special_tokens=True):return ' '.join(ids)


@pytest.fixture
def cfg():return json.loads(Path('configs/approximate/ocr_evidence_generation_battery_train_v1.json').read_text())


def block(i,text='cat'):
    return {'id':str(i),'candidate_id':str(i),'bbox':[i*10,0,i*10+8,10],'text':text,
            'confidence':{'sequence_confidence_native':.8,'mean_token_log_probability':-.1,'minimum_token_probability':.7,
                          'token_probabilities':[.99,.7,.99]},
            'ar_candidates':[{'text':text,'logprob':-.2},{'text':text+'s','logprob':-2.},
                             {'text':'dog','logprob':-3.},{'text':'','logprob':-4.}]}


@pytest.fixture
def sample():return {'sample_id':'100','blocks':[block(i) for i in range(5)],
                    'captions':{'florence2-base-ft':'A cat sits. The sign says "OPEN".', 'instructblip':'A cat scene.'},
                    'gt_text':'SECRET_TARGET','gt_bbox':[999]*4}


def variant(cfg,name):return next(v for v in variants(cfg) if v['id']==name)


def test_frozen_forty(cfg):
    vv=variants(cfg); assert len(vv)==40
    assert [v['beam'] for v in vv if v['id'] in ('IMAGE_BEAM1','IMAGE_BEAM3','IMAGE_BEAM5','IMAGE_BEAM10','IMAGE_BEAM20')]==[1,3,5,10,20]


def test_neighbors_are_total_not_per_side(sample):
    assert neighbor_indices(sample['blocks'],2,2,'reading_order')==[1,3]
    assert neighbor_indices(sample['blocks'],0,2,'reading_order')==[1,2]
    assert neighbor_indices(sample['blocks'],2,0,'reading_order')==[]
    assert neighbor_indices(sample['blocks'],2,'all','reading_order')==[0,1,3,4]


def test_spatial_different_from_index(sample):
    sample['blocks'][4]['bbox']=[21,0,22,1]
    assert 4 in neighbor_indices(sample['blocks'],2,2,'bbox_center')


def test_no_gt_and_one_target_no_concatenation(sample,cfg):
    q=make_request(sample,2,variant(cfg,'IMAGE_BEAM3'),cfg,Tokenizer())
    assert 'SECRET_TARGET' not in str(q) and 'gt_bbox' not in str(q)
    payload=json.loads(q['messages'][-1]['content'])
    assert payload['target']['ocr']=='cat'
    assert len(payload['other_separate_crops_in_reading_order'])==4
    assert all(len(b['ar_hypotheses'])==3 for b in payload['other_separate_crops_in_reading_order'])


def test_no_invented_beams(sample,cfg):
    sample['blocks'][2]['ar_candidates']=[]
    q=make_request(sample,2,variant(cfg,'IMAGE_BEAM20'),cfg,Tokenizer())
    t=json.loads(q['messages'][-1]['content'])['target']
    assert t['ar_hypotheses']==[] and t['ocr']=='cat' and 'unavailable' in t['ar_cache_status']


def test_rank_cut_does_not_refill_empty_or_casefold_dedupe(cfg):
    b=block(0);v=variant(cfg,'AR3_SCORE');b['ar_candidates'][1]['text']=''
    o=observation(b,v,cfg,True)
    assert [x['rank'] for x in o['ar_hypotheses']]==[1,3]
    assert o['ar_hypotheses'][1]['delta_to_ar_best']==-2.8


def test_character_mask_never_replaces_primary(cfg):
    b=block(0);o=observation(b,variant(cfg,'RICH_MASK_HINT'),cfg,True)
    assert o['ocr']=='cat' and o['uncertain_character_hint']=='c?t'
    assert o['character_probabilities']==[['c',.99],['a',.7],['t',.99]]


def test_caption_prefix_and_distinct_views(cfg):
    v=variant(cfg,'CAP16');v['caption_tokens']=2
    assert caption_view('one two three',v,'',Tokenizer())==('one two',3,2)
    assert caption_view('a "RED" sign',variant(cfg,'RICH_QUOTED_TEXT'),'',Tokenizer())[0]=='RED'
    assert caption_view('A cat sleeps. Blue sky.',variant(cfg,'RICH_RELEVANT_SENTENCES'),'cat',Tokenizer())[0]=='A cat sleeps.'
    assert caption_view('Blue sky.',variant(cfg,'RICH_RELEVANT_SENTENCES'),'cat',Tokenizer())[0]==''


def test_duplicate_wrong_caption_controls(sample,cfg):
    q=make_request(sample,0,variant(cfg,'RICH_WRONG_CAPTION_CONTROL'),cfg,Tokenizer(),'Wrong caption.')
    assert json.loads(q['messages'][-1]['content'])['fallible_scene_context']=='Wrong caption.'
    view=caption_view('A cat.',variant(cfg,'RICH_DUPLICATED_CAPTION_CONTROL'),'',Tokenizer())[0]
    assert view=='A cat.\nA cat.'


def test_order_ablation_preserves_evidence(sample,cfg):
    a=make_request(sample,0,variant(cfg,'RICH'),cfg,Tokenizer())
    b=make_request(sample,0,variant(cfg,'RICH_CAPTION_FIRST'),cfg,Tokenizer())
    assert json.loads(a['messages'][-1]['content'])==json.loads(b['messages'][-1]['content'])
    assert a['messages']!=b['messages']
    rev=make_request(sample,0,variant(cfg,'RICH_REVERSED_AR_CONTROL'),cfg,Tokenizer())
    assert json.loads(rev['messages'][-1]['content'])['target']['ar_hypotheses']==list(reversed(json.loads(a['messages'][-1]['content'])['target']['ar_hypotheses']))


def test_identical_prompts_can_share_inference(sample,cfg):
    sample['blocks']=sample['blocks'][:2]
    a=make_request(sample,0,variant(cfg,'CAP32'),cfg,Tokenizer())
    b=make_request(sample,0,variant(cfg,'NEIGHBOR4'),cfg,Tokenizer())
    assert a['messages']==b['messages']


@pytest.mark.parametrize('raw,fmt,choices,expected',[
    ('Café!','plain',[],'Café!'),('<KEEP>','plain',[],'BASE'),
    ('{"text":"SHOP"}','json',[],'SHOP'),('{"operation":"KEEP"}','edit',[],'BASE'),
    ('{"operation":"REPLACE","text":"(A120)"}','edit',[],'(A120)'),
    ('{"index":1}','choice',['BASE','other'],'other'),
])
def test_output_schemas(raw,fmt,choices,expected):assert parse_answer(raw,'BASE',fmt,choices)['text']==expected


@pytest.mark.parametrize('raw,fmt,choices',[
    ('a description','plain',[]),('','plain',[]),('{"text":"TWO WORDS"}','json',[]),
    ('{"text":"WORD","extra":1}','json',[]),('{"index":true}','choice',['BASE','B']),
    ('{"index":2}','choice',['BASE','B']),('```json\n{}\n```','json',[]),
])
def test_invalid_preserves_exact_original(raw,fmt,choices):
    assert parse_answer(raw,'BASE',fmt,choices)['text']=='BASE'


def test_truncation_and_empty_preserve():
    assert parse_answer('RIGHT','wrong','plain',ended=False)['text']=='wrong'
    assert parse_answer('invented','','plain')['text']==''


def test_confidence_alignment_fails_closed():
    b=block(0);r={**b['confidence'],'raw_text':'cat','predicted_length':3,'token_log_probabilities':[-.1,-.1,-.1]}
    assert validate_confidence(r,'cat')['minimum_token_probability']==.7
    with pytest.raises(RuntimeError):validate_confidence(r,'CAT')
    with pytest.raises(RuntimeError):validate_confidence({**r,'token_probabilities':[.9]},'cat')
