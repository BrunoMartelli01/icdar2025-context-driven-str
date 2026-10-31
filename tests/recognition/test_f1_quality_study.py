from fractions import Fraction
import pytest
from recognition.f1_quality_study import (aggregate,good,image_f1,evaluate,threshold_curve,
    choose_cut,sorted_ids,proxy_scores,NEW_FAMILIES)


def c(tp,fp,fn):
    return {'tp':tp,'fp':fp,'fn':fn,'detected_words':tp+fp,'gt_words':tp+fn,
            'all_outputs':tp+fp,'ignored_outputs':0}


def test_quality_is_exact_and_distinct_from_word_aggregation():
    assert good(c(4,1,1),.8) and not good(c(4,1,1),.80001)
    assert not good(c(0,0,0),.5)
    branch={'small':{'light':c(1,0,0),'deepsolo':c(1,0,0)},
            'large':{'light':c(0,0,9),'deepsolo':c(9,0,0)}}
    v=evaluate(branch,list(branch),set(branch),.8)
    assert v['good_fraction']==v['accepted_macro_image_f1']==.5
    assert v['accepted_words']['f1']==pytest.approx(2/11)


def test_threshold_curve_preserves_ties_missing_scores_and_denominators():
    b={s:{'light':c(1,0,1),'deepsolo':c(2,0,0)}for s in ('a','b','empty')}
    scores={'a':.8,'b':.8,'empty':None}
    curve=threshold_curve(scores,b,list(b),[.5,.8])
    assert len(curve)==1 and curve[0]['accepted_images']==2
    assert curve[0]['good_images']=={'0.5':2,'0.8':0}
    p=evaluate(b,list(b),['a','b'],.8)
    assert curve[0]['combined_counts']['gt_words']==6
    assert curve[0]['combined_f1']==p['combined']['f1']
    assert sorted_ids(scores,list(b))==['a','b']


def test_no_qualifying_cut_abstains_instead_of_claiming_success():
    b={'a':{'light':c(0,1,1),'deepsolo':c(1,0,0)}}
    curve=threshold_curve({'a':1.},b,['a'],[.8])
    fit=choose_cut(curve,q=.8,purity=.9,minimum_accepted=1,objective='max_f1')
    assert fit['threshold']is None and fit['calibration']is None


def test_coverage_and_micro_f1_objectives_are_different():
    b={'a':{'light':c(1,0,0),'deepsolo':c(0,0,1)},
       'b':{'light':c(1,1,1),'deepsolo':c(2,0,0)}}
    curve=threshold_curve({'a':.9,'b':.8},b,list(b),[.5])
    kw=dict(q=.5,purity=1.,minimum_accepted=1)
    assert choose_cut(curve,objective='max_bypass',**kw)['threshold']==.8
    assert choose_cut(curve,objective='max_f1',**kw)['threshold']==.9


def test_calibration_rejects_undeclared_population_and_nonfinite_scores():
    with pytest.raises(ValueError):sorted_ids({'train':1.,'test':2.},['train'])
    with pytest.raises(ValueError):sorted_ids({'a':float('nan')},['a'])


def test_new_proxies_are_label_free_and_empty_outputs_abstain():
    rows=[{'candidate_id':'a','raw_text':'OPEN','confidence':1.,'clip_word_advantage':.5,'bbox':[0,0,10,10]}]
    con={'crop_stability':{'words':[{'candidate_id':'a','minimum_edit_agreement':1.}]},
         'coverage':{'text_coverage':.5,'kernel_coverage':.5},'scale_agreement':{'score':1.},
         'scores':{'all_checks_softcrop':.5}}
    context={'scores':{'base_mpnet':.5}}
    values=proxy_scores(rows,con,context,[0.,1.],{'lower_quantile':.2,'duplicate_iou':.5})
    assert set(values)==set(NEW_FAMILIES)
    assert values['proxy_f1_native']==pytest.approx(2/3)
    assert values['proxy_f1_duplicate_penalty']==values['proxy_f1_visual']
    assert set(proxy_scores([],con,context,[0.,1.],{'lower_quantile':.2,'duplicate_iou':.5}).values())=={None}


def test_measured_word_counts_reconstruct_combined_f1_exactly():
    a=c(285,26,52);r=c(1457,385,410)
    wa=2*a['tp']+a['fp']+a['fn'];wr=2*r['tp']+r['fp']+r['fn']
    combined=aggregate([a,r])
    assert float((wa*image_f1(a)+wr*image_f1(r))/(wa+wr))==combined['f1']


def test_failed_family_selection_cannot_be_rescued_by_calibration():
    from scripts.run_yolo_f1_quality_study import split_evaluate
    cfg={'quality_f1_thresholds':[.8],'minimum_bypass_fraction':.2,'bypass_fractions':[]}
    b={'a':{'light':c(1,0,0),'deepsolo':c(0,0,1)}}
    frozen={'policies':{'q':{'quality':.8,'selected_family':'family',
        'selection_qualified':False,'families':{'family':{'threshold':.5,'status':'EMPIRICAL_CUT'}}}},
        'budget_selected_families':{}}
    out=split_evaluate(cfg,{}, {'a':{'scores':{'family':1.}}},b,['a'],frozen)
    policy=out['policies']['q']
    assert policy['families']['family']['accepted_images']==1
    assert policy['selected']['accepted_images']==0
    assert policy['selected']['calibration_status']=='ABSTAIN_ALL_NO_QUALIFYING_SELECTION_FAMILY'
    assert policy['selected']['combined']==out['deepsolo_only']


def test_descriptive_auc_and_ap_preserve_score_ties_and_missing_values():
    from scripts.analyze_yolo_f1_quality_study import discrimination
    d=discrimination([1.,1.,0.,None],[True,False,True,False])
    assert d['auroc']==.625
    assert d['average_precision']==pytest.approx(Fraction(7,12))
    assert d['missing_scores']==1
    tied=discrimination([None]*4,[True,False,True,False])
    assert tied['auroc']==tied['average_precision']==.5


def test_macro_f1_and_serialized_point_are_identical_across_hash_seeds():
    import os
    import subprocess
    import sys
    program='''
import json
from recognition.f1_quality_study import evaluate
branches={}
for i in range(1,18):
    counts=dict(tp=i,fp=i%4,fn=i%7,detected_words=i+i%4,gt_words=i+i%7,
                all_outputs=i+i%4,ignored_outputs=0)
    branches[f"sample{i}"]=dict(light=counts,deepsolo=counts)
print(json.dumps(evaluate(branches,list(branches),set(branches),.8),sort_keys=True))
'''
    outputs=[subprocess.check_output([sys.executable,'-c',program],
        env={**os.environ,'PYTHONHASHSEED':str(seed)},text=True)for seed in (1,2,37)]
    assert len(set(outputs))==1
