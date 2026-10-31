from itertools import combinations
from fractions import Fraction
from scripts.analyze_essential_oracle import fixed_acceptance_oracle,minimum_deepsolo_oracle,aggregate_choice

def c(tp,n,g):return dict(tp=tp,fp=n-tp,fn=g-tp,detected_words=n,gt_words=g,all_outputs=n,ignored_outputs=0)

def test_privileged_fractional_oracles_match_exhaustive_subsets():
    b={'a':dict(light=c(2,2,2),deepsolo=c(1,2,2)),'b':dict(light=c(0,1,2),deepsolo=c(2,2,2)),
       'c':dict(light=c(1,1,2),deepsolo=c(1,2,2)),'d':dict(light=c(0,0,1),deepsolo=c(0,1,1))}
    rate=lambda p:Fraction(2*p['tp'],p['detected_words']+p['gt_words'])
    all_points=[]
    for size in range(5):
        points=[aggregate_choice(b,subset) for subset in combinations(b,size)];all_points+=points
        oracle=fixed_acceptance_oracle(b,list(b),size)
        assert rate(oracle)==max(map(rate,points)) and oracle['optimality_residual']==0
    base=rate(aggregate_choice(b,[]));oracle=minimum_deepsolo_oracle(b,list(b))
    assert oracle['deepsolo_images']==min(p['deepsolo_images'] for p in all_points if rate(p)>=base)
    assert oracle['next_acceptance_best_utility'] is None or oracle['next_acceptance_best_utility']<0
