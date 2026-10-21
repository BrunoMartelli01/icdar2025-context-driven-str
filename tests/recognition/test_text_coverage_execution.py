"""Failure-boundary fixtures; these do not qualify GPU inference or scientific results."""
import json
from types import SimpleNamespace

import pytest
import torch

from scripts import run_yolo_coverage_resolution as runner


def test_out_of_memory_is_recorded_and_propagated_without_partial_success(tmp_path, monkeypatch):
    cfg={'execution':{'seed':7,'cpu_threads':1},'resolutions':[640]}
    plan={'code_sha256':{}}
    monkeypatch.setattr(runner,'prepare',lambda args:(cfg,plan,tmp_path,{}, {},{}))
    for name in ['manual_seed','set_num_threads','use_deterministic_algorithms']:
        monkeypatch.setattr(torch,name,lambda *args,**kwargs:None)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *args:None)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *args:'fixture')
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *args:SimpleNamespace(total_memory=0))
    monkeypatch.setattr(runner.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout='fixture'))
    monkeypatch.setattr(runner,'_cuda_peak',lambda device:dict(peak_allocated_bytes=0,peak_reserved_bytes=0))
    def fail(*args):raise torch.cuda.OutOfMemoryError('injected boundary fixture')
    monkeypatch.setattr(runner,'observations',fail)
    with pytest.raises(torch.cuda.OutOfMemoryError,match='injected boundary fixture'):
        runner.run(SimpleNamespace(check=False))
    failure_paths=list((tmp_path/'failures').glob('*.json'))
    assert len(failure_paths)==1
    failure=json.loads(failure_paths[0].read_text())
    assert failure['status']=='FAILED' and failure['out_of_memory']
    assert failure['partial_results_are_not_final']
    assert not (tmp_path/'results.json').exists()
