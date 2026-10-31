from fractions import Fraction
from itertools import combinations
from recognition.budget_target_oracle import analyze_budget


def counts(tp,pred,gt):
    return dict(tp=tp,fp=pred-tp,fn=gt-tp,detected_words=pred,gt_words=gt,all_outputs=pred,ignored_outputs=0)


def test_exact_and_at_most_oracles_match_exhaustive_choices():
    b={'a':dict(light=counts(1,1,2),deepsolo=counts(2,2,2)),
       'b':dict(light=counts(1,1,1),deepsolo=counts(0,1,1)),
       'c':dict(light=counts(0,0,1),deepsolo=counts(1,1,1)),
       'd':dict(light=counts(1,3,1),deepsolo=counts(1,1,1))}
    for eligible in [list(b),['a','b','c']]:
        for budget in range(len(b)-len(eligible),len(b)+1):
            r=analyze_budget(b,eligible,budget,Fraction(4,5));best=Fraction(0);minimum=None
            for k in range(len(eligible)+1):
                for keep in combinations(eligible,k):
                    tp=sum(row['light' if s in keep else 'deepsolo']['tp'] for s,row in b.items())
                    den=sum(row['light' if s in keep else 'deepsolo']['detected_words']+row['light']['gt_words'] for s,row in b.items());f=Fraction(2*tp,den)
                    if len(b)-k<=budget:best=max(best,f)
                    if f>=Fraction(4,5):minimum=min(minimum,len(b)-k) if minimum is not None else len(b)-k
            p=r['best_within_budget'];assert Fraction(2*p['tp'],p['detected_words']+p['gt_words'])==best
            assert r['minimum_deepsolo_images']==minimum
