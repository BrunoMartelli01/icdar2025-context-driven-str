"""Isolated owner-balanced text loss, preserving the qualified kernel objective."""

import torch

from segmentation.icdar13_matched_training import matched_word_loss
from segmentation.icdar2015_improvements import instance_balanced_loss
from segmentation.word_multitask import masked_bce_dice


def training_loss(logits,batch,config):
    if 'instance_balanced_text' not in config:
        return matched_word_loss(logits,batch,config,'totaltext_bce_dice')
    target,valid,separator,kernel=(batch[k] for k in ('mask','validity','separation','kernel'))
    if logits.shape!=(len(target),2,*target.shape[-2:]):raise ValueError('Aligned full-word/kernel logits required')
    if torch.any(kernel>target) or torch.any(kernel>valid):raise ValueError('Kernel outside valid foreground')
    if torch.any((separator!=0)&(separator!=1)) or torch.any((separator>0)&((target>0)|(valid==0))):
        raise ValueError('Separator must remain valid non-text background')
    options=config['instance_balanced_text']
    if set(options)!={'dice_weight','dice_margin','epsilon'}:raise ValueError('Explicit instance text loss options required')
    text=instance_balanced_loss(logits[:,:1],target,valid,separator,batch['owners'],
        separation_weight=config['separation_weight'],**options)
    core=masked_bce_dice(logits[:,1:2],kernel,valid,**config['source_loss'])
    return text+config['kernel_loss_weight']*core
