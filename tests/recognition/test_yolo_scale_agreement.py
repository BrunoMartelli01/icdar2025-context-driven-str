import math
from recognition.yolo_scale_agreement import scale_pairs


def row(cid,box,text):
    return {'candidate_id':cid,'bbox':box,'raw_text':text,'detector_score':.8,
            'token_probabilities':[.9]*len(text),'token_log_probabilities':[math.log(.9)]*len(text)}


def test_agreement_requires_same_word_same_location_and_one_to_one():
    base=[row('a',[0,0,10,10],'Hello'),row('b',[0,0,10,10],'Hello'),row('c',[20,0,30,10],'Hello')]
    high=[row('x',[0,0,10,10],'HELLO'),row('y',[20,0,30,10],'World')]
    pairs=scale_pairs(base,high,.5)
    assert len(pairs)==1 and pairs[0]['paired_candidate_id']=='x'
    assert pairs[0]['bbox']==[0,0,10,10]
