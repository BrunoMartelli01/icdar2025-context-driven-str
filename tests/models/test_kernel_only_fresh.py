import torch
import pytest
from models.kernel_only_fresh import FreshKernelAGUNet


def test_real_new_decoder_and_head_are_untouched_by_encoder_loading():
    torch.set_num_threads(1);torch.manual_seed(202503)
    model=FreshKernelAGUNet(skip_floor=.25,encoder_pretrained=False)
    head=model.head.weight.detach().clone();decoder=model.bottleneck[0].weight.detach().flatten()[:64].clone()
    state=model.encoder.state_dict();state={**state,'conv1.weight':torch.full_like(state['conv1.weight'],.25)}
    report=model.initialize_encoder(state)
    assert not report['segmentation_checkpoint_loaded'] and report['decoder_initialization']=='new_random'
    assert torch.equal(model.head.weight,head)
    assert torch.equal(model.bottleneck[0].weight.flatten()[:64],decoder)
    assert torch.all(model.encoder.conv1.weight==.25)
    assert model.architecture_identity=='experimental-fresh-kernel-only-agunet-v1'
    assert not hasattr(model,'initialize_text_kernel_state')
    model.eval()
    with torch.inference_mode():assert model(torch.randn(2,3,64,64)).shape==(2,1,32,32)
    model.train();assert not model.encoder.training and all(not p.requires_grad for p in model.encoder.parameters())
    with pytest.raises(ValueError,match='every'):model.initialize_encoder({k:v for k,v in state.items() if k!='fc.weight'})
    with pytest.raises(ValueError,match='dtype'):model.initialize_encoder({**state,'conv1.weight':state['conv1.weight'].double()})
