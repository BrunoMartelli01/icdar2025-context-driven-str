import numpy as np
import pytest

from recognition.uniform_pooled_gate import domain_weights,macro_lower_gain


def test_domains_have_equal_total_weight_despite_different_image_and_crop_counts():
    domains=np.array(['a','a','b']);w=domain_weights(domains,[1,3,100])
    assert w.sum()==pytest.approx(3)
    assert w[domains=='a'].sum()==pytest.approx(w[domains=='b'].sum())
    assert w[1]/w[0]==pytest.approx(3)
    with pytest.raises(ValueError):domain_weights(['a'],[0])


def test_paired_domain_bootstrap_retains_identity_and_detects_consistent_changes():
    base=dict(a=np.tile([1,1,1],(3,1)),b=np.tile([2,1,1],(7,1)))
    assert macro_lower_gain(base,base,1,100,.05)==0
    other={k:v+np.array([1,-1,-1]) for k,v in base.items()}
    assert macro_lower_gain(base,other,1,100,.05)>0
    assert macro_lower_gain(other,base,1,100,.05)<0
    with pytest.raises(ValueError):macro_lower_gain(base,{},1,100,.05)
