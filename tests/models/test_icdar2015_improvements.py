"""Norm migration retains every convolution/affine tensor and freezes encoder."""

import gc
import torch
from torch import nn
import pytest

from models.icdar2015_improvements import ImprovementAGUNet


def test_strict_source_load_and_explicit_groupnorm_migration():
    torch.set_num_threads(2)
    source=ImprovementAGUNet(normalization="batchnorm",max_groups=32,skip_floor=.25)
    state=source.state_dict()
    improved=ImprovementAGUNet(normalization="groupnorm",max_groups=32,skip_floor=.25,source_state=state)
    assert len(improved.normalization_migration)==21
    removed={row["name"]+"."+field for row in improved.normalization_migration for field in row["removed_buffers"]}
    assert set(state)-set(improved.state_dict())==removed
    assert not (set(improved.state_dict())-set(state))
    assert all(torch.equal(v,state[k]) for k,v in improved.state_dict().items())
    improved.train()
    assert all(not m.training for m in improved.encoder.modules() if isinstance(m,nn.BatchNorm2d))
    assert not any(p.requires_grad for p in improved.encoder.parameters())
    assert improved(torch.zeros(1,3,64,96)).shape==(1,1,32,48)
    # Reloading its own architecture is also strict; missing source keys fail.
    improved.load_state_dict(improved.state_dict(),strict=True)
    del improved;gc.collect()
    bad=dict(state);bad.pop("head.bias")
    with pytest.raises(RuntimeError,match="Missing key"):
        ImprovementAGUNet(normalization="batchnorm",max_groups=32,skip_floor=.25,source_state=bad)
