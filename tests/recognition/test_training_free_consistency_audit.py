from scripts.audit_yolo_frozen_consistency import match_count,edit_distance,calibrate


def test_independent_sparse_matching_and_character_edit_distance():
    a=[{'raw_text':'X','bbox':[0,0,10,10]},{'raw_text':'x','bbox':[4,0,14,10]}]
    b=[{'raw_text':'x','bbox':[1,0,11,10]},{'raw_text':'x','bbox':[-3,0,7,10]}]
    assert match_count(a,b,.4)==2
    assert match_count(a,[],.4)==0
    assert edit_distance('books','book')==1
    assert edit_distance('abc','xyz')==3
    assert edit_distance('','abc')==3


def test_independent_threshold_does_not_split_score_ties():
    b={'a':{'light':dict(tp=2,fp=0,fn=0)},'b':{'light':dict(tp=0,fp=1,fn=2)}}
    assert calibrate({'a':.8,'b':.8},b,['a','b'],.9,1)==(None,None)
    cut,point=calibrate({'a':.9,'b':.8},b,['a','b'],.9,1)
    assert cut==.9 and point['accepted_images']==point['good_images']==1
