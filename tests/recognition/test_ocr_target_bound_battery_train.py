import copy
import json
from pathlib import Path

import pytest

from diagnostics.ocr_target_bound_battery_train import (
    expanded, gate, job_hash, make_prompt, package, parse, reconstruct,
)
from diagnostics.caption_routing_train import checked, seal_write


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return ' '.join(tokens)


@pytest.fixture
def cfg():
    return json.loads(Path('configs/approximate/ocr_target_bound_battery_train_v2.json').read_text())


@pytest.fixture
def sample():
    blocks=[]
    for i in range(3):
        text='c4t'
        blocks.append({'id':f'v1:0100:00{i}', 'candidate_id':f'v1:0100:00{i}', 'text':text,
            'bbox':[i*10,0,i*10+8,10],
            'confidence':{'sequence_confidence_native':.6,'mean_token_log_probability':-.1,
                          'minimum_token_probability':.6,'token_probabilities':[.99,.6,.99]},
            'ar_candidates':[{'text':text,'logprob':-.2},{'text':'cat','logprob':-.4}]})
    return {'sample_id':'100','blocks':blocks,'captions':{'florence2-base-ft':'A shop says "CAT".',
            'instructblip':'A cat on a table.'}, 'gt_text':'DO_NOT_LEAK_THIS'}


def prompt(sample,cfg,mechanism='BOUND_JSON',pkg='LOCAL_BEAM10'):
    return make_prompt(sample,1,package(cfg,pkg),mechanism,cfg,Tokenizer())


def response(q, **fields):
    return json.dumps({'target_id':q['candidate_id'], **fields})


def test_eighty_unique_train_only(cfg):
    assert len(set(expanded(cfg)))==80
    bad=copy.deepcopy(cfg);bad['scope']['split']='TEST'
    with pytest.raises(ValueError):expanded(bad)


@pytest.mark.parametrize('mechanism', ['PLAIN_TARGET_LAST','BOUND_JSON','KEEP_REPLACE','CHAR_PATCH','COPY_REPAIR','UNCERTAIN_PATCH','AR_CHOICE'])
def test_prompts_target_last_no_gt(sample,cfg,mechanism):
    q=prompt(sample,cfg,mechanism)
    assert 'DO_NOT_LEAK_THIS' not in json.dumps(q)
    payload=json.loads(q['messages'][-1]['content'])
    assert list(payload)[-1]=='TARGET'
    assert payload['TARGET']['id']==sample['blocks'][1]['id']
    assert len(payload['context_not_the_answer']['other_crops'])==2


def test_synthetic_label_changes_do_not_change_prompt(sample,cfg):
    first=prompt(sample,cfg)
    sample['gt_text']='ANOTHER';sample['gt_bbox']=[999]*4
    assert prompt(sample,cfg)==first


def test_missing_beam_only_keep(sample,cfg):
    sample['blocks'][1]['ar_candidates']=[]
    q=prompt(sample,cfg,'AR_CHOICE','TARGET_ONLY')
    assert q['choices']==['c4t']
    assert parse(response(q,index=1),q,sample['blocks'][1],cfg)['text']=='c4t'


def test_bound_json_exact_id(sample,cfg):
    q=prompt(sample,cfg);b=sample['blocks'][1]
    assert parse(response(q,text='cat'),q,b,cfg)['text']=='cat'
    assert not parse(json.dumps({'target_id':'other','text':'cat'}),q,b,cfg)['valid']
    assert not parse(response(q,text='two words'),q,b,cfg)['valid']
    assert not parse(response(q,text='cat',explanation='why'),q,b,cfg)['valid']
    assert not parse(response(q,text='cat'),q,b,cfg,ended=False)['valid']


def test_copy_must_be_exact(sample,cfg):
    q=prompt(sample,cfg,'COPY_REPAIR');b=sample['blocks'][1]
    assert parse(response(q,observed='c4t',text='cat'),q,b,cfg)['text']=='cat'
    assert parse(response(q,observed='C4T',text='cat'),q,b,cfg)['reason']=='copied_original_mismatch'


def test_patch_original_indices_simultaneous(sample,cfg):
    q=prompt(sample,cfg,'CHAR_PATCH');b=sample['blocks'][1]
    edits=[{'start':0,'end':1,'text':''},{'start':2,'end':3,'text':'r'}]
    assert parse(response(q,edits=edits),q,b,cfg)['text']=='4r'
    assert parse(response(q,edits=[{'start':3,'end':3,'text':'s'}]),q,b,cfg)['text']=='c4ts'
    assert parse(response(q,edits=[]),q,b,cfg)['text']=='c4t'


@pytest.mark.parametrize('edits', [
    [{'start':-1,'end':1,'text':'x'}],
    [{'start':0,'end':5,'text':'x'}],
    [{'start':False,'end':1,'text':'x'}],
    [{'start':0,'end':2,'text':'a'},{'start':1,'end':3,'text':'b'}],
    [{'start':0,'end':0,'text':'a'},{'start':0,'end':0,'text':'b'}],
    [{'start':0,'end':3,'text':'floor'}],
    [{'start':0,'end':1,'text':'a b'}],
    [{'start':0,'end':1,'text':5}],
    [{'start':0,'end':1,'text':'x','extra':0}],
])
def test_bad_patch_falls_back(sample,cfg,edits):
    q=prompt(sample,cfg,'CHAR_PATCH')
    r=parse(response(q,edits=edits),q,sample['blocks'][1],cfg)
    assert not r['valid'] and r['text']=='c4t'


def test_uncertain_patch(sample,cfg):
    q=prompt(sample,cfg,'UNCERTAIN_PATCH');b=sample['blocks'][1]
    assert parse(response(q,edits=[{'start':1,'end':2,'text':'a'}]),q,b,cfg)['text']=='cat'
    assert not parse(response(q,edits=[{'start':0,'end':1,'text':'b'}]),q,b,cfg)['valid']
    assert not parse(response(q,edits=[{'start':1,'end':1,'text':'a'}]),q,b,cfg)['valid']


def test_bool_not_choice_index(sample,cfg):
    q=prompt(sample,cfg,'AR_CHOICE')
    assert not parse(response(q,index=True),q,sample['blocks'][1],cfg)['valid']


def test_keep_preserves_punctuation(sample,cfg):
    sample['blocks'][1]['text']='M.O.T.'
    q=prompt(sample,cfg,'KEEP_REPLACE')
    assert parse(response(q,operation='KEEP'),q,sample['blocks'][1],cfg)['text']=='M.O.T.'


def test_gates_and_judge_no_new_string(sample,cfg):
    primary={'text':'cat','valid':True,'reason':'valid'}
    assert gate('EDIT1_GATE',primary,'c4t',cfg)['text']=='cat'
    assert gate('EDIT1_GATE',primary,'dog',cfg)['text']=='dog'
    assert gate('CONTEXT_AGREEMENT',primary,'c4t',cfg,{'text':'CAT','valid':True})['text']=='cat'
    assert gate('CONTEXT_AGREEMENT',primary,'c4t',cfg,{'text':'bat','valid':True})['text']=='c4t'
    assert gate('VERIFY_PROPOSAL',primary,'c4t',cfg,{'valid':True,'reason':'ACCEPT'})['text']=='cat'
    assert gate('VERIFY_PROPOSAL',primary,'c4t',cfg,None)['text']=='c4t'
    q=make_prompt(sample,1,package(cfg,'TARGET_ONLY'),'JUDGE',cfg,Tokenizer(),proposal='cat')
    assert parse(response(q,decision='ACCEPT'),q,sample['blocks'][1],cfg)['reason']=='ACCEPT'
    assert not parse(response(q,decision='ACCEPT',text='dog'),q,sample['blocks'][1],cfg)['valid']


def test_no_context_deduplicates(sample,cfg):
    q1=make_prompt(sample,1,package(cfg,'LOCAL_BEAM10'),'BOUND_JSON',cfg,Tokenizer(),no_context=True)
    q2=make_prompt(sample,1,package(cfg,'TARGET_BEAM10'),'BOUND_JSON',cfg,Tokenizer())
    assert job_hash(q1)==job_hash(q2)
    desc={'sample_id':'100','index':1,'package':'LOCAL_BEAM10','mechanism':'BOUND_JSON','no_context':True}
    assert reconstruct(desc,{'100':sample},cfg,Tokenizer())==q1


def test_quoted_package_and_limited_caption(sample,cfg):
    q=prompt(sample,cfg,'BOUND_JSON','IMAGE_BEAM10')
    payload=json.loads(q['messages'][-1]['content'])
    assert payload['context_not_the_answer']['fallible_caption']=='CAT'
    assert len(payload['context_not_the_answer']['other_crops'])==2


def test_resume_integrity_and_no_overwrite(tmp_path):
    path=tmp_path/'saved.json';seal_write(path,{'test':1});seal_write(path,{'test':1})
    with pytest.raises(RuntimeError):seal_write(path,{'test':2})
    path.write_text('{"test":2,"payload_sha256":"incorrect"}')
    with pytest.raises(RuntimeError):checked(path)
