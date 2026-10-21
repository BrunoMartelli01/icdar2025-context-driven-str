import json

import pytest

from scripts.run_yolo_router_comparison import checked_shard, save_shard, score_images, select_rows


def row(name, text, bbox, detector=.9):
    return {'candidate_id':name,'bbox':bbox,'raw_text':text,'detector_score':detector,
            'token_log_probabilities':[-.01]*len(text),'token_probabilities':[.99]*len(text),
            'confidence':.9,'clip_word_advantage':.2}


def test_selector_retains_care_and_ignore_blindly_and_suppresses_only_same_reading():
    a=row('a','CAT',[0,0,10,10]);duplicate=row('b','cat!',[0,0,10,10],.8)
    other=row('c','DOG',[0,0,10,10]);empty=row('d','',[20,20,30,30])
    selected=select_rows([a,duplicate,other,empty],{'selection':{'joint_threshold':.6,'nms_iou':.5}})
    assert {r['candidate_id'] for r in selected} == {'a','c'}
    a['ignored']=True
    assert select_rows([a],{'selection':{'joint_threshold':.6,'nms_iou':.5}}) == [a]


def test_clip_minimum_never_certifies_empty_or_hides_weak_word():
    reference={'native':[.1,.9],'word_advantage':[.1,.2]}
    a=row('a','A',[0,0,2,2]);weak={**a,'clip_word_advantage':-.5}
    features={'a':{'rows':[a,weak],'paper_scores':{'paper_selected':.2,'joined_t3':.2}},
              'b':{'rows':[],'paper_scores':{'paper_selected':None,'joined_t3':None}}}
    scores=score_images(features,reference)
    assert scores['clip']['a'] == 0
    assert scores['clip']['b'] is None
    assert all(values['b'] is None for values in scores.values())


def test_sealed_shard_rejects_wrong_filename_and_changed_payload(tmp_path):
    path=tmp_path/'total_text__1.json'
    save_shard(path,'plan',{'sample_id':'total_text:1','value':3})
    assert checked_shard(path,'plan')['value']==3
    with pytest.raises(ValueError):checked_shard(path,'other plan')
    wrong=tmp_path/'total_text__2.json';wrong.write_bytes(path.read_bytes())
    with pytest.raises(ValueError):checked_shard(wrong,'plan')
    data=json.loads(path.read_text());data['value']=4;path.write_text(json.dumps(data))
    with pytest.raises(ValueError):checked_shard(path,'plan')
