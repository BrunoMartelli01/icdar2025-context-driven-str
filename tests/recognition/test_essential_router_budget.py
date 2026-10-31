from scripts.run_essential_router_budget import split_holdout,scalar_cut,threshold_ids


def test_deployment_ties_and_missing_observations(monkeypatch):
    import recognition.essential_budget_router as interface
    deployment=dict(bundle={},feature_settings={},policies=[dict(target_calibration_acceptance=.5,threshold=1.,tie_sample_id='b')])
    monkeypatch.setattr(interface,'predict_observation_gate',lambda *a,**k:dict(predicted_utility=1.))
    assert interface.predict_essential_budget(deployment,'a',[],[1,1])['use_unet']['0.5']
    assert not interface.predict_essential_budget(deployment,'c',[],[1,1])['use_unet']['0.5']
    monkeypatch.setattr(interface,'predict_observation_gate',lambda *a,**k:dict(predicted_utility=None))
    assert not interface.predict_essential_budget(deployment,'a',[],[1,1])['use_unet']['0.5']


def test_budget_roles_are_disjoint_deterministic_and_independent_of_input_order():
    ids=[str(i) for i in range(125)];a,b=split_holdout(ids,62,20261027)
    assert len(a)==62 and len(b)==63 and not set(a)&set(b) and set(a)|set(b)==set(ids)
    assert (a,b)==split_holdout(list(reversed(ids)),62,20261027)


def test_fixed_validation_cut_keeps_ties_deterministic_and_never_accepts_ineligible():
    scores={'a':1.,'b':1.,'c':1.,'d':None};cut=scalar_cut(scores,list(scores),.5)
    assert cut==dict(threshold=1.,tie_sample_id='b')
    assert threshold_ids(scores,cut)==['a','b']
    assert threshold_ids({'a':2.,'b':None,'c':1.,'d':0.},cut)==['a']
