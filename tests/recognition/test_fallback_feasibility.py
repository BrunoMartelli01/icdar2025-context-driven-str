from fractions import Fraction
from itertools import combinations
import numpy as np
import pytest
from recognition.fallback_feasibility import target_minimum,global_oracle,totals


def test_exact_minimum_and_global_oracle_match_exhaustive_subsets():
    rng=np.random.default_rng(8)
    for trial in range(20):
        b={}
        for i in range(6):
            gt=int(rng.integers(1,6));b[str(i)]={}
            for name in ['light','deepsolo']:
                pred=int(rng.integers(0,7));b[str(i)][name]=dict(tp=int(rng.integers(0,min(gt,pred)+1)),detected_words=pred,gt_words=gt)
        eligible=list(b) if trial%2 else list(b)[:-2]
        points=[totals(b,keep) for k in range(len(eligible)+1) for keep in combinations(eligible,k)]
        target=Fraction(3,5);possible=[v for v in points if v['f1']>=target]
        actual=target_minimum(b,target,eligible)
        assert actual['minimum_deepsolo_images']==min((v['deepsolo_images'] for v in possible),default=None)
        expected=max(points,key=lambda v:(v['f1'],-v['deepsolo_images']));upper=global_oracle(b,eligible)
        assert upper['f1']==expected['f1'] and upper['deepsolo_images']==expected['deepsolo_images']
        assert upper['certificate']=='0'


def test_mismatched_gt_denominator_fails():
    b={'a':dict(light=dict(tp=1,detected_words=1,gt_words=1),deepsolo=dict(tp=1,detected_words=1,gt_words=2))}
    with pytest.raises(ValueError):target_minimum(b,.8)
