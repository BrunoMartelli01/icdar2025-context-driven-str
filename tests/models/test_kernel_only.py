import pytest
import torch

from models.adaspot_gap import AdaSpotGapAGUNet
from models.kernel_only import KernelOnlyAGUNet


def test_real_extraction_keeps_kernel_logits_and_one_output():
    torch.set_num_threads(1)
    source=AdaSpotGapAGUNet('control',skip_floor=.25,initial_gap_bias=-2).eval()
    model=KernelOnlyAGUNet(skip_floor=.25,encoder_pretrained=False).eval()
    state=source.state_dict()
    report=model.initialize_text_kernel_state(state)
    assert report['strict_load'] and report['source_channel']==1
    assert torch.equal(model.head.weight,state['head.weight'][1:2])
    assert torch.equal(model.head.bias,state['head.bias'][1:2])
    x=torch.randn(2,3,64,64)
    with torch.inference_mode():
        expected=source(x)[:,1:2];actual=model(x)
    assert actual.shape==(2,1,32,32)
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)
    model.train()
    assert not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())
    with pytest.raises(ValueError,match='every'):
        model.initialize_text_kernel_state({k:v for k,v in state.items() if k!='head.bias'})
    with pytest.raises(ValueError,match='two source'):
        model.initialize_text_kernel_state({**state,'head.bias':torch.zeros(3)})
    with pytest.raises(ValueError,match='dtype'):
        model.initialize_text_kernel_state({**state,'head.weight':state['head.weight'].double()})
