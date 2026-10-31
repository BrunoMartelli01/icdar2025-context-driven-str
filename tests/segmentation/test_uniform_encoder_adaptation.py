from copy import deepcopy

import numpy as np
import pytest
import torch

from segmentation.icdar13_word_refit import WordRefitUNet
from segmentation.uniform_encoder_adaptation import enable_adaptation, optimizer, validation_forward


def test_adaptation_reenables_only_upper_convolutions_and_captures_actual_context():
    torch.set_num_threads(1)
    net = WordRefitUNet(partial_encoder=False, skip_floor=.25, initial_gap_bias=-2.)
    net.encoder.requires_grad_(False)  # strict donor initialization freezes it
    image = torch.randn(1, 3, 64, 64)
    net.eval()
    with torch.inference_mode():
        before, context = validation_forward(net, image, {})
    assert context is None
    c = dict(encoder_adaptation=dict(train_stages=['layer3', 'layer4'], freeze_batchnorm=True),
        training=dict(learning_rate=1e-5, encoder_learning_rate=1e-6, weight_decay=1e-4))
    enable_adaptation(net, c)
    net.train()
    assert not net.encoder.training
    for name, p in net.encoder.named_parameters():
        assert p.requires_grad == (name.startswith(('layer3.', 'layer4.')) and '.bn' not in name and '.downsample.1.' not in name)
    net.eval()
    with torch.inference_mode():
        after, vector = validation_forward(net, image, c)
    assert torch.equal(before, after) and vector.shape == (1, 4096)
    assert not net.encoder.layer4._forward_hooks
    groups = optimizer(net, c).param_groups
    assert [g['lr'] for g in groups] == [1e-5, 1e-6]
    selected = [id(p) for g in groups for p in g['params']]
    assert len(selected) == len(set(selected)) == sum(p.requires_grad for p in net.parameters())
    buffers = {k:v.clone() for k,v in net.encoder.named_buffers()}
    net.train()
    net(image).square().mean().backward()
    assert net.encoder.layer3[0].conv1.weight.grad.abs().sum() > 0
    assert net.encoder.layer4[0].conv1.weight.grad.abs().sum() > 0
    assert net.encoder.layer2[0].conv1.weight.grad is None
    assert all(torch.equal(v, buffers[k]) for k,v in net.encoder.named_buffers())


def test_legacy_optimizer_is_numerically_identical_and_invalid_lr_fails():
    a = torch.nn.Linear(2, 1); b = deepcopy(a)
    c = dict(training=dict(learning_rate=1e-5, weight_decay=1e-4))
    oa = optimizer(a, c); ob = torch.optim.AdamW(b.parameters(), lr=1e-5, weight_decay=1e-4)
    x = torch.ones(2, 2)
    a(x).sum().backward(); b(x).sum().backward(); oa.step(); ob.step()
    assert all(torch.equal(u,v) for u,v in zip(a.parameters(), b.parameters()))
    c.update(encoder_adaptation={}); c['training']['encoder_learning_rate'] = float('nan')
    with pytest.raises(ValueError, match='encoder LR'): optimizer(a, c)


def test_completed_storage_preserves_nonoptimizer_fields_and_rejects_selected_last(tmp_path):
    from scripts.run_uniform_unet_training import write, read
    from scripts.compact_uniform_completed_storage import compact
    from segmentation.uniform_unet_fusion import file_hash
    root = tmp_path/'completed'; root.mkdir()
    torch.save(dict(model_state_dict={'x':torch.ones(2)}, optimizer_state_dict={'x':torch.zeros(4)},
        rng=np.array([1,2]), history=[{'loss':3}]), root/'last.pt')
    torch.save({'x':torch.ones(2)}, root/'best.pt')
    write(root/'resolved_configuration.json', {})
    write(root/'independent_training_audit.json',dict(status='PASS'))
    complete=dict(status='COMPLETE',selected_epoch=1,epochs=1,checkpoint_sha256=file_hash(root/'best.pt'))
    write(root/'complete.json',complete)
    receipt=tmp_path/'receipt.json'
    before=file_hash(root/'last.pt')
    with pytest.raises(ValueError, match='nonselected'):compact([root], receipt)
    assert not receipt.exists() and file_hash(root/'last.pt')==before
    complete['epochs']=2;write(root/'complete.json',complete);compact([root],receipt)
    saved=torch.load(root/'last.pt',weights_only=False)
    assert set(saved)=={'model_state_dict','rng','history'} and torch.equal(saved['model_state_dict']['x'],torch.ones(2))
    assert read(receipt)['cases'][0]['all_nonoptimizer_state_exact']
