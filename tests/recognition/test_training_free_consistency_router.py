import pytest
from recognition.training_free_consistency_router import (
    scale_agreement,crop_stability,good_image,calibrate,evaluate,make_features,FAMILIES)


def row(cid,text,box=(0,0,10,10)):
    return {'candidate_id':cid,'raw_text':text,'bbox':list(box),'confidence':.8,'clip_word_advantage':.2}


def test_scale_pairs_casefold_duplicates_extra_words_and_geometry():
    a=[row('a','Word'),row('b','word',(20,0,30,10))]
    b=[row('c','WORD'),row('d','extra',(20,0,30,10))]
    result=scale_agreement(a,b,.5)
    assert len(result['pairs'])==1 and result['score']==.5
    assert scale_agreement(a,[row('c','WORD')],.5)['score']==pytest.approx(2/3)
    assert scale_agreement([row('a','word!')],[row('c','word')],.5)['score']==0
    assert scale_agreement([row('a','word')],[row('b','word',(100,100,110,110))],.5)['score']==0
    assert scale_agreement([],[],.5)['score']==0
    with pytest.raises(ValueError):scale_agreement([row('a','x'),row('a','x')],b,.5)


def test_matching_prefers_two_valid_pairs_over_one_larger_iou():
    a=[row('a','x',(0,0,10,10)),row('b','x',(4,0,14,10))]
    b=[row('c','x',(1,0,11,10)),row('d','x',(-3,0,7,10))]
    result=scale_agreement(a,b,.4)
    assert len(result['pairs'])==2
    assert {tuple((p['left'],p['right'])) for p in result['pairs']}=={('a','d'),('b','c')}


def test_crop_agreement_uses_all_views_and_never_vacuously_accepts_empty_strings():
    rows=[row('a','Word'),row('b','book')]
    views=[{'original_candidate_id':s,'fraction':f,'raw_text':t} for s,ts in
        [('a',['word','WORD']),('b',['book','books'])] for f,t in zip([-.05,.05],ts)]
    result=crop_stability(rows,views,[-.05,.05])
    assert result['exact_fraction']==.5
    assert result['edit_agreement']==pytest.approx(.9)
    with pytest.raises(ValueError):crop_stability(rows,views[:-1],[-.05,.05])
    with pytest.raises(ValueError):crop_stability(rows,views+views[:1],[-.05,.05])


def counts(tp,fp,fn):
    return dict(tp=tp,fp=fp,fn=fn,detected_words=tp+fp,gt_words=tp+fn,all_outputs=tp+fp,ignored_outputs=0)


def test_quality_is_per_image_f1_including_missing_words():
    assert good_image(counts(2,1,0),.8)
    assert good_image(counts(2,0,1),.8)
    assert not good_image(counts(2,0,2),.8)
    assert not good_image(counts(0,0,0),.8)


def test_calibration_uses_exact_declared_ids_and_abstains_without_a_threshold():
    b={'a':{'light':counts(2,0,0),'deepsolo':counts(2,0,0)},
       'b':{'light':counts(0,1,2),'deepsolo':counts(2,0,0)}}
    fit=calibrate({'a':.9,'b':.8},b,['a','b'],target=.9,quality_threshold=.8,minimum_accepted=1)
    assert fit['threshold']==.9 and fit['formal_risk_guarantee'] is False
    assert calibrate({'a':.9,'b':.8},b,['a','b'],target=.9,quality_threshold=.8,minimum_accepted=2)['threshold'] is None
    with pytest.raises(ValueError):calibrate({'a':.9,'b':.8},b,['a'],target=.9,quality_threshold=.8,minimum_accepted=1)
    result=evaluate(b,['a','b'],{'a'},.8)
    assert result['accepted_good_images']==1 and result['combined']['fn']==0
    assert evaluate(b,['a','b'],set(),.8)['combined']['tp']==4


def test_features_preserve_outputs_and_empty_readings_force_fallback():
    cov={'text_coverage':1.,'kernel_coverage':.9}
    base={'rows':[],'coverage':cov};ref={'native':[.2,.8],'word_advantage':[.1,.2]}
    old={'weight':0.,'reference':[1.],'coverage_field':'text_coverage'}
    f=make_features(base,{'rows':[]},[],reference=ref,previous_selector=old,fractions=[-.05,.05],min_iou=.5)
    assert set(f['scores'])==set(FAMILIES) and all(v is None for v in f['scores'].values())
    a=row('a','word');base['rows']=[a]
    views=[{'original_candidate_id':'a','fraction':v,'raw_text':'word'} for v in [-.05,.05]]
    f=make_features(base,{'rows':[row('b','WORD')]},views,reference=ref,previous_selector=old,fractions=[-.05,.05],min_iou=.5)
    assert f['scores']['scale_only']==1 and f['scores']['crop_only']==1
    assert not f['ground_truth_used'] and not f['models_trained'] and not f['original_word_outputs_changed']
    assert base['rows']==[a]
