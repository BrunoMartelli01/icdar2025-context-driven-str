"""Explicit upper-encoder adaptation; the legacy frozen path remains unchanged."""

import math

import torch


def enable_adaptation(net, config):
    policy = config.get('encoder_adaptation')
    if policy is None:
        return
    if policy != {'train_stages': ['layer3', 'layer4'], 'freeze_batchnorm': True}:
        raise ValueError('Only the declared layer3/layer4, frozen-BN intervention is qualified')
    # The strict text-encoder initializer freezes everything. Enable gradients
    # AFTER initialization, and retain the existing no-grad lower-stage path.
    net.partial_encoder = True
    net.encoder.requires_grad_(False)
    net.encoder.layer3.requires_grad_(True)
    net.encoder.layer4.requires_grad_(True)
    for module in net.encoder.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.requires_grad_(False)
    net.encoder.eval()


def optimizer(net, config):
    t = config['training']
    if 'encoder_adaptation' not in config:
        return torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
            lr=t['learning_rate'], weight_decay=t['weight_decay'])
    lr = t['encoder_learning_rate']
    if not math.isfinite(lr) or not 0 < lr <= t['learning_rate']:
        raise ValueError('Explicit positive encoder LR no larger than decoder LR required')
    groups = [dict(params=[p for name, p in net.named_parameters()
        if p.requires_grad and name.startswith('encoder.') == enc],
        lr=lr if enc else t['learning_rate']) for enc in (False, True)]
    if any(not group['params'] for group in groups):
        raise ValueError('Both encoder and decoder must actually train')
    return torch.optim.AdamW(groups, weight_decay=t['weight_decay'])


def validation_forward(net, image, config):
    if 'encoder_adaptation' not in config:
        return net(image), None
    captured = []
    def capture(_module, _inputs, output):
        captured.append(torch.cat((output.mean((2, 3)), output.amax((2, 3))), dim=1).detach())
    handle = net.encoder.layer4.register_forward_hook(capture)
    try:
        logits = net(image)
    finally:
        handle.remove()
    if len(captured) != 1 or captured[0].shape != (image.shape[0], 4096):
        raise ValueError('One actual4096-D encoder vector per image required')
    if not torch.isfinite(captured[0]).all():
        raise ValueError('Nonfinite adapted encoder context')
    return logits, captured[0]
