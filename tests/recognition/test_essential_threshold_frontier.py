from itertools import combinations
import numpy as np

from recognition.essential_threshold_frontier import (threshold_curve,apply_cut,minimum_cost,maximum_f1,pareto,
    simultaneous_bootstrap,gain_identity,common_thresholds,observable_length_rescale)


def test_rescaling_needs_only_the_observed_word_count():
    assert observable_length_rescale(6.,[{},{}])==2.
    assert observable_length_rescale(None,[{},{}]) is None


def counts(tp,n,g):return dict(tp=tp,fp=n-tp,fn=g-tp,detected_words=n,gt_words=g,all_outputs=n,ignored_outputs=0)


def fixture():
    return {'a':dict(light=counts(2,2,2),deepsolo=counts(1,2,2)),
            'b':dict(light=counts(0,1,2),deepsolo=counts(2,2,2)),
            'c':dict(light=counts(1,1,2),deepsolo=counts(1,2,2))}


def test_tied_scores_form_one_attainable_cut_and_ineligible_abstains():
    branches=fixture();scores={'a':2.,'b':None,'c':2.};curve=threshold_curve(branches,scores)
    assert [p['accepted_images'] for p in curve]==[0,2]
    assert apply_cut(curve,3.)['deepsolo_images']==3
    assert apply_cut(curve,2.)['deepsolo_images']==1
    assert apply_cut(curve,-100.)['deepsolo_images']==1
    assert minimum_cost(curve,curve[0])['deepsolo_images']==1


def test_frontier_selection_and_exact_gain_identity_on_every_subset():
    branches=fixture();scores={'a':3.,'b':1.,'c':2.};curve=threshold_curve(branches,scores)
    assert maximum_f1(curve)['threshold']==2.
    # All-light is3/5 versus all-fallback4/6; keeping fallback on b is necessary.
    assert minimum_cost(curve,curve[0])['deepsolo_images']==1
    assert all(not any(q['deepsolo_images']<=p['deepsolo_images'] and q['f1']>=p['f1'] and
        (q['deepsolo_images']<p['deepsolo_images'] or q['f1']>p['f1']) for q in curve) for p in pareto(curve))
    for size in range(4):
        for accepted in combinations(branches,size):
            a=set(accepted);tp=sum(b['light' if s in a else 'deepsolo']['tp'] for s,b in branches.items())
            den=sum(b['light' if s in a else 'deepsolo']['detected_words']+b['light']['gt_words'] for s,b in branches.items())
            identity=gain_identity(branches,a)
            assert identity['positive']==(2*tp/den>identity['baseline_f1'])


def test_bootstrap_is_deterministic_and_all_fallback_gain_is_exactly_zero():
    branches=fixture();scores={'a':3.,'b':1.,'c':2.};curve=threshold_curve(branches,scores)
    first=simultaneous_bootstrap(curve,branches,scores,100,23)
    assert first==simultaneous_bootstrap(curve,branches,scores,100,23)
    assert first['lower_gain'][0]==0 and first['pointwise_gain95'][0]==[0.,0.]
    assert first['simultaneous_critical_value']>=0


def test_common_cut_preserves_counts_and_checks_each_dataset():
    a=threshold_curve(fixture(),{'a':3.,'b':1.,'c':2.});b=threshold_curve(fixture(),{'a':4.,'b':0.,'c':2.})
    common=common_thresholds({'one':a,'two':b})
    assert common[0]['deepsolo_images']==6 and common[0]['no_loss_all'] and not common[0]['strict_gain_all']
    assert any(p['strict_gain_all'] for p in common)
    for p in common:
        assert p['total_images']==6 and p['deepsolo_images']==sum(v['deepsolo_images'] for v in p['datasets'].values())
