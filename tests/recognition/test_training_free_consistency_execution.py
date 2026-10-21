"""Boundary fixtures; no substituted outputs qualify a scientific experiment."""
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import run_yolo_frozen_consistency as runner
from recognition.training_free_consistency_router import FAMILIES


@pytest.mark.parametrize('error',[RuntimeError('crop pixel mismatch fixture'),torch.cuda.OutOfMemoryError('OOM fixture')])
def test_probe_failure_is_preserved_and_propagated(tmp_path,monkeypatch,error):
    cfg={'execution':{'seed':7,'cpu_threads':1}}
    monkeypatch.setattr(runner,'prepare',lambda args:(cfg,{'code_sha256':{}},tmp_path,{'train':{}},{'train':{}}))
    for name in ['manual_seed','set_num_threads','use_deterministic_algorithms']:
        monkeypatch.setattr(torch,name,lambda *args,**kwargs:None)
    monkeypatch.setattr(runner.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='fixture'))
    def fail(*args):raise error
    monkeypatch.setattr(runner,'views_for_split',fail)
    with pytest.raises(type(error),match=str(error)):runner.run(SimpleNamespace(check=False))
    failures=list((tmp_path/'failures').glob('*.json'));assert len(failures)==1
    failure=json.loads(failures[0].read_text())
    assert failure['status']=='FAILED' and failure['partial_results_are_not_final']
    assert failure['out_of_memory']==isinstance(error,torch.cuda.OutOfMemoryError)
    assert not (tmp_path/'results.json').exists() and not (tmp_path/'frozen_thresholds.json').exists()


def test_state_hash_covers_parameters_and_buffers():
    model=torch.nn.Linear(2,2).eval().requires_grad_(False)
    model.register_buffer('counter',torch.tensor(0))
    before=runner.state_sha(model)
    assert before==runner.state_sha(model)
    model.counter.add_(1)
    assert before!=runner.state_sha(model)
    before=runner.state_sha(model)
    model.weight[0,0].add_(1)
    assert before!=runner.state_sha(model)


def test_new_thresholds_are_rejected_after_test_features_exist(tmp_path):
    p=tmp_path/'test/features';p.mkdir(parents=True);(p/'fixture.json').write_text('{}')
    cfg={'target_good_image_fractions':[.9],'quality_f1_threshold':.8,
        'minimum_calibration_accepted':1,'bypass_fractions':[.5]}
    plan={'selection_ids':['a','b'],'threshold_calibration_ids':['c','d']}
    f={s:{'scores':{k:.9-i*.1 for k in FAMILIES}} for i,s in enumerate(['a','b','c','d'])}
    count=dict(tp=1,fp=0,fn=0,detected_words=1,gt_words=1,all_outputs=1,ignored_outputs=0)
    b={s:{'light':count,'deepsolo':count} for s in f}
    with pytest.raises(ValueError,match='after TEST consistency features'):
        runner.freeze_thresholds(cfg,plan,tmp_path,f,b)
    assert not (tmp_path/'frozen_thresholds.json').exists()
