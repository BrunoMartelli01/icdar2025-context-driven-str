"""Controlled approximate encoder-decay ablation with fresh encoder moments."""

import torch


def restored_optimizer(network, payload, config, branch):
    t = config['training']
    decoder = [p for n, p in network.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
    encoder = [p for n, p in network.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    old = payload['optimizer_state_dict']
    if len(old['param_groups']) != 2 or [g['group_name'] for g in old['param_groups']] != ['decoder','encoder_layer3_layer4']:
        raise ValueError('Qualified decoder/deep-encoder Adam groups required')
    if [len(g['params']) for g in old['param_groups']] != [len(decoder),len(encoder)]:
        raise ValueError('Source parameter-group ordering/counts changed')
    for g in old['param_groups']:
        if g['weight_decay'] != t['weight_decay'] or tuple(g['betas']) != tuple(t['adam_betas']) or g['eps'] != t['adam_epsilon']:
            raise ValueError('Qualified source Adam settings changed')
    if branch['encoder_weight_decay'] not in (0.0,t['weight_decay']):
        raise ValueError('Explicit matched zero/source encoder decay required')
    optimizer = torch.optim.Adam([dict(params=decoder),dict(params=encoder)],
        lr=t['learning_rate'],weight_decay=t['weight_decay'],betas=tuple(t['adam_betas']),eps=t['adam_epsilon'])
    optimizer.load_state_dict(old)
    # Decoder moments/steps remain exactly inherited. Encoder moments include the
    # old coupled L2 term, so explicitly discard them in BOTH matched branches.
    for p in encoder:
        optimizer.state.pop(p,None)
    for g,name,rate in zip(optimizer.param_groups,('decoder','encoder_layer3_layer4'),
            (t['learning_rate'],branch['encoder_learning_rate'])):
        g.update(group_name=name,lr=rate,initial_lr=rate)
    optimizer.param_groups[1]['weight_decay'] = branch['encoder_weight_decay']
    return optimizer


def word_selection_key(raw,emitted,geometry,epoch):
    return (raw['alphanumeric_casefold']['f1'],emitted['alphanumeric_casefold']['f1'],
            geometry['proposal_f1_iou_50'],-epoch)
