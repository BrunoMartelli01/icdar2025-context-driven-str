"""Tests for independent route denominators and fixed-cardinality certificates."""
import pytest
from scripts.audit_icdar_budget_router import counts,verify_route,oracle_certificate


def branch(tp,pred,gt):
    return dict(tp=tp,fp=pred-tp,fn=gt-tp,detected_words=pred,gt_words=gt,all_outputs=pred+1,ignored_outputs=1)


def test_independent_route_retains_ignored_and_empty_output_denominators():
    b={'a':dict(light=branch(1,3,2),deepsolo=branch(2,2,2)),
       'b':dict(light=branch(2,2,2),deepsolo=branch(1,2,2))}
    c=counts(b,list(b),['b'])
    assert c['tp']==4 and c['f1']==1 and c['all_outputs']==6 and c['ignored_outputs']==2
    point=dict(accepted_ids=['b'],accepted_images=1,deepsolo_images=1,total_images=2,deepsolo_percentage=50.,combined=c)
    verify_route(b,list(b),['b'],point)
    point['deepsolo_images']=0
    with pytest.raises(AssertionError):verify_route(b,list(b),['b'],point)


def test_oracle_certificate_rejects_suboptimal_acceptance():
    b={'a':dict(light=branch(1,3,2),deepsolo=branch(2,2,2)),
       'b':dict(light=branch(2,2,2),deepsolo=branch(1,2,2))}
    oracle_certificate(b,list(b),list(b),['b'])
    with pytest.raises(AssertionError):oracle_certificate(b,list(b),list(b),['a'])
