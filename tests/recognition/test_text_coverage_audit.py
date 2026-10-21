import numpy as np
import pytest

from scripts.analyze_yolo_coverage_resolution import counts, midcdf, combine, bootstrap


def test_independent_counter_caps_duplicates_and_keeps_all_missing_words():
    rows=[{'raw_text':'WORD','bbox':[0,0,2,2]}, {'raw_text':'word','bbox':[0,0,2,2]},
          {'raw_text':'ignore','bbox':[2,0,4,2]}]
    validity=np.ones((4,4),dtype=bool);validity[:2,2:]=False
    gt=[{'text':'word'}, {'text':'missing'}]
    result=counts(rows,gt,validity,(4,4))
    assert result == dict(tp=1,fp=1,fn=1,detected_words=2,gt_words=2,all_outputs=3,ignored_outputs=1)
    assert counts([],gt,validity,(4,4))['fn']==2


def test_cdf_ties_and_paired_bootstrap_of_identical_gates():
    assert midcdf([0,1,1,2],1)==.5
    assert midcdf([0,1,1,2],-1)==0
    assert midcdf([0,1,1,2],3)==1
    word=dict(tp=1,fp=1,fn=1,detected_words=2,gt_words=2,all_outputs=2,ignored_outputs=0)
    branches={s:{'light':word,'deepsolo':word} for s in ['a','b']}
    metrics=combine(branches,['a','b'],{'a'})
    assert metrics['f1']==.5 and metrics['accuracy_tn0']==pytest.approx(1/3)
    ci=bootstrap(branches,['a','b'],{'a'},{'a'},100,7)
    assert ci['delta_f1']==ci['delta_tn0']==0
    assert ci['f1_ci95']==ci['tn0_ci95']==[0,0]
