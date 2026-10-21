import copy
import math

import pytest

from diagnostics.parseq_visual_margin_context_gate_train import (
    attach_visual, best_key, evaluate_choices, finite, margin_diagnostic, select,
)


def row(deltas=(0., -.1, -3.), context=(0., 1., 100.)):
    return {'sample_id':'100','candidate_id':'crop','v1_text':'bad',
            'candidates':[{'text':t,'ar_rank':i+1} for i,t in enumerate(('bad','good','far'))],
            'delta_visual':list(deltas),'force_keep_reason':None,'reference_rank':1,
            'scores':{'local':[{'qwen_mean_logprob':v} for v in context]}}


def test_visual_gate_limits_strong_context_and_is_inclusive():
    r = row()
    assert select(r,'local',.1,.5,0) == 1
    assert select(r,'local',.09,.5,0) == 0
    assert select(r,'local',4,.5,0) == 2


def test_strict_tau_boundary_and_keep_ties():
    r = row((0.,-.25,-3.),(0.,1.,100.))
    assert select(r,'local',.25,.5,.25) == 0
    assert select(r,'local',.25,.5,.249) == 1
    assert select(r,'local',.25,.25,0) == 0


def test_alternative_ties_use_frozen_order():
    assert select(row((0.,-.1,-.1),(0.,1.,1.)),'local',.1,.5,0) == 1


def test_controls_are_distinct_and_visual_only_can_restore_ar_top1():
    r = row((0.,-.25,-3.),(0.,.1,100.))
    assert select(r,'local',.25,.05,0) == 0
    assert select(r,'local',.25,None,0,'visual_margin_gate_only') == 1
    assert select(r,None,None,None,0,'visual_only') == 0
    assert select(row((0.,.1,-3.)),None,None,None,0,'visual_only') == 1


def test_no_gt_in_selection():
    a = row(); b = copy.deepcopy(a); b['gt_text'] = 'far'; b['correct_index']=2
    assert select(a,'local',.1,.5,0) == select(b,'local',.1,.5,0)


def raw_row(v1, alternatives):
    return {'sample_id':'100','candidate_id':'crop','v1_text':v1,
            'candidates':[{'text':v1,'source':'V1','ar_rank':None}]+alternatives,'scores':{}}


def test_real_visual_reference_not_rank_surrogate():
    ar = [{'text':'best','logprob':-1.},{'text':'v1','logprob':-2.}]
    r = raw_row('v1',[{'text':'best','ar_rank':1,'ar_logprob':-1.}])
    out = attach_visual(r,ar)
    assert out['reference_rank'] == 2 and out['delta_visual'] == [0.,1.]


@pytest.mark.parametrize('text,reason',[('other','reference_absent_from_saved_beam'),
                                      ('BEST','case_only_reference_mismatch')])
def test_missing_exact_reference_is_keep(text,reason):
    out = attach_visual(raw_row(text,[{'text':'best','ar_rank':1,'ar_logprob':-1.}]),
                        [{'text':'best','logprob':-1.}])
    assert out['force_keep_reason'] == reason and out['delta_visual'][1] is None
    assert select(out,'local',1,.5,0) == 0


def test_bad_score_identity_fails():
    with pytest.raises(ValueError):
        attach_visual(raw_row('v1',[{'text':'other','ar_rank':2,'ar_logprob':-3.}]),
                      [{'text':'v1','logprob':-1.},{'text':'other','logprob':-2.}])
    for x in [math.nan,math.inf,-math.inf]:
        with pytest.raises(ValueError): finite(x)


def test_shared_crop_has_one_decision_not_oracle_per_gt():
    r = row(); rows = {('100','crop'):r}
    labs = [{'sample_id':'100','candidate_id':'crop','gt_text':t,'v1_exact':t=='bad'} for t in ('bad','good')]
    m = evaluate_choices(labs,rows,{('100','crop'):1},{'baseline_exact':1,'total_gt_words':849})
    assert m['exact']==1 and m['recovered']==1 and m['damaged']==1 and m['changed_crops']==1
    assert m['net']==0


def test_margin_counts_missing_reference_and_gt_only_posthoc():
    r = row(); k=('100','crop');lab={'sample_id':'100','candidate_id':'crop','gt_text':'good','v1_exact':False}
    d = margin_diagnostic([lab],{k:r},[.09,.1])
    assert [x['recoverable_gt_rows_inside_margin'] for x in d['within_margin']] == [0,1]
    r['force_keep_reason']='missing';r['delta_visual'][1]=None
    assert margin_diagnostic([lab],{k:r},[.1])['within_margin'][0]['recoverable_gt_rows_inside_margin']==0


def test_train_best_tiebreak_prefers_fewer_damaged_then_changes():
    a={'exact':505,'damaged':1,'changed_crops':20,'m':.5,'beta':.1,'tau':.1}
    assert best_key(a)<best_key({**a,'damaged':2,'changed_crops':10})
    assert best_key(a)>best_key({**a,'changed_crops':19})
