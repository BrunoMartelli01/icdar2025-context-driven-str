"""GPU failures produce explicit failed artifacts and propagate to the caller."""

import json
import sys

import pytest
import torch

from scripts import screen_icdar2015_unet_resolutions as screen
from scripts import train_icdar2015_unet_improvement as train


@pytest.mark.parametrize("entry",[screen,train])
def test_cuda_oom_is_recorded_and_never_retried_with_reduced_inputs(entry,monkeypatch,tmp_path):
    calls=[]
    def fail(args):
        calls.append(args)
        raise torch.cuda.OutOfMemoryError("deliberate simulated OOM")
    monkeypatch.setattr(entry,"run",fail)
    monkeypatch.setattr(torch.cuda,"is_initialized",lambda:False)
    flags=["command","--config",str(tmp_path/"profile.json"),"--data-root",str(tmp_path/"data"),
        "--source-checkpoint",str(tmp_path/"source.pt"),"--output-dir",str(tmp_path/"out"),"--device","cuda:0"]
    if entry is train:
        flags += ["--screening-report",str(tmp_path/"screen.json"),"--branch","larger_bce","--validation-batch-size","1"]
    else:flags += ["--batch-size","1"]
    monkeypatch.setattr(sys,"argv",flags)
    with pytest.raises(torch.cuda.OutOfMemoryError,match="deliberate simulated OOM"):entry.main()
    assert len(calls)==1
    report=json.loads((tmp_path/"out/failure.json").read_text())
    assert report["status"]=="FAILED" and report["oom"] is True
    assert report["precision"]=="float32" and report["device"]=="cuda:0"
    assert report["cuda_peak_allocated_bytes"] is None
