import numpy as np
import pytest
from sklearn.linear_model import Ridge

from recognition.common_budget_pipeline import (counts,micro,choose_light,grouped_folds,
    map_features,fit_count_head,predict_counts,word_features,risk_features,validation_roles)


def test_counts_keep_duplicate_empty_and_ignored_denominators():
    rows=[dict(raw_text=x) for x in ['ONE!','one','!!!','ignored']]
    c=counts(rows,[dict(text='one'),dict(text='!!!')],[False,False,False,True])
    assert (c['tp'],c['fp'],c['fn'],c['ignored_outputs'])==(1,2,1,1)
    assert micro([c])['f1']==.4


def test_budget_is_exact_deterministic_and_guard_failure_is_explicit():
    scores={'a':1.,'b':1.,'c':0.,'d':-1.};keys={'a':'2','b':'1','c':'3','d':'4'}
    assert choose_light(scores,keys,2)==['b','a']
    assert choose_light(dict(reversed(list(scores.items()))),keys,2)==['b','a']
    with pytest.raises(ValueError,match='Guard'):choose_light(scores,keys,1,['a','b'])
    with pytest.raises(ValueError,match='Nonfinite'):choose_light(dict(scores,a=float('nan')),keys,2)


def test_two_target_ridge_reproduces_direct_utility_ridge():
    rng=np.random.default_rng(42);x=rng.normal(size=(40,8));y=rng.normal(size=(40,2));weights=rng.uniform(.2,8,40)
    bundle=fit_count_head(x,y,weights,dict(kind='ridge',alpha=10.),42)
    prediction=predict_counts(bundle,x);reference=.79
    alternate=Ridge(alpha=10.).fit(bundle['scaler'].transform(x),2*y[:,0]-reference*y[:,1],sample_weight=weights)
    np.testing.assert_allclose(2*prediction[:,0]-reference*prediction[:,1],alternate.predict(bundle['scaler'].transform(x)),atol=1e-12)


def test_missing_kernel_is_marked_and_cannot_change_text_summary():
    p=np.zeros((1,16,16),np.float32);p[0,3:8,4:10]=.8
    settings=dict(probability_threshold=.5,minimum_component_area=1)
    single=map_features(p,[32,32],[],settings)
    dual=map_features(np.concatenate([p,np.ones_like(p)],axis=0),[32,32],[],settings)
    np.testing.assert_array_equal(single[:6],dual[:6])
    np.testing.assert_array_equal(single[6:12],np.zeros(6))
    assert single[-1]==0 and dual[-1]==1


def test_rgb_group_folds_do_not_split_pixel_duplicates():
    ids=[str(i) for i in range(12)];keys={s:str(int(s)//2) for s in ids}
    a=grouped_folds(ids,keys,3,7);b=grouped_folds(list(reversed(ids)),keys,3,7)
    assert a==b and sorted(s for f in a for s in f)==sorted(ids)
    owner={s:i for i,f in enumerate(a) for s in f}
    assert all(owner[str(i)]==owner[str(i+1)] for i in range(0,12,2))


def test_word_risk_covers_empty_and_rejected_observable_words():
    rows=[dict(raw_text='Hello!',confidence=.9,detector_score=.8,bbox=[0,0,10,5],token_probabilities=[.7,.9])]
    trace=[dict(potential_match=None,accepted=False,changed=False)]
    assert word_features(rows,trace,[20,20]).shape==(1,18)
    assert word_features([],[],[20,20]).shape==(0,18)
    assert risk_features(np.array([.8]),trace).shape==(16,)
    assert risk_features(np.empty(0),[])[-1]==1


def test_strict_is_rejected_before_any_artifact_read():
    from pathlib import Path
    from types import SimpleNamespace
    from scripts.run_common_budget_pipeline import prepare
    with pytest.raises(ValueError,match='APPROXIMATE'):
        prepare(SimpleNamespace(mode='strict',config=Path('configs/strict/missing.json')))


def test_independent_fit_audit_handles_floating_point_constant_features():
    from scripts.audit_common_budget_pipeline import audit_head
    rng=np.random.default_rng(19);x=rng.normal(size=(40,8));x[:,1]=np.log(16/9)
    y=rng.normal(size=(40,2));w=rng.uniform(1,10,size=40)
    bundle=fit_count_head(x,y,w,dict(kind='ridge',alpha=10.),19)
    audit_head(bundle,x,y,w,19)


def test_pooled_validation_preserves_fitting_and_test_boundaries():
    original=dict(fit=['f'],selection=['s'],calibration=['c'],test=['t'])
    pooled=validation_roles(original,'pooled')
    assert pooled==dict(fit=['f'],selection=['c','s'],calibration=[],test=['t'])
    assert original['selection']==['s'] and original['calibration']==['c']
    with pytest.raises(ValueError,match='overlap'):
        validation_roles(dict(original,calibration=['s']),'pooled')


def test_deployment_rejects_changed_independent_qualification(tmp_path):
    import json
    from recognition.common_budget_deployment import CommonBudgetPipeline
    (tmp_path/'audit.json').write_text('{"status":"PASS"}')
    (tmp_path/'deployment_qualification.json').write_text(json.dumps(dict(status='PASS',audit_path='audit.json',audit_sha256='0'*64,artifact_sha256={})))
    with pytest.raises(ValueError,match='qualification'):
        CommonBudgetPipeline(tmp_path,'irrelevant')
