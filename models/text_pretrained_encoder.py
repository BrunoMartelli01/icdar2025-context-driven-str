"""Strict functional ResNet50 mapping from a pinned text-pretrained checkpoint.

Only the encoder initializes from the detector. The U-Net decoder, PARSeq and
fallback remain separate models. There is no DeepSolo recognition in forward.
"""

import re

import torch


def encoder_destination(name):
    if name == 'stem.conv1.weight':
        return 'conv1.weight'
    if name.startswith('stem.conv1.norm.'):
        return 'bn1.'+name[len('stem.conv1.norm.'):]
    match = re.fullmatch(r'res([2-5])\.(\d+)\.(.+)', name)
    if match is None:
        raise ValueError('Unrecognized pretrained ResNet50 tensor: '+name)
    stage, block, suffix = int(match[1])-1, match[2], match[3]
    if suffix == 'shortcut.weight':
        suffix = 'downsample.0.weight'
    elif suffix.startswith('shortcut.norm.'):
        suffix = 'downsample.1.'+suffix[len('shortcut.norm.'):]
    else:
        norm = re.fullmatch(r'conv([1-3])\.norm\.(.+)', suffix)
        if norm is not None:
            suffix = 'bn'+norm[1]+'.'+norm[2]
        elif re.fullmatch(r'conv[1-3]\.weight', suffix) is None:
            raise ValueError('Unsupported pretrained bottleneck tensor: '+name)
    return 'layer'+str(stage)+'.'+block+'.'+suffix


def initialize_text_encoder(encoder, source, *, prefix):
    """Require all functional tensors and exact shapes/dtypes, load strictly."""
    source_keys = [name for name in source if name.startswith(prefix)]
    if not source_keys:
        raise ValueError('Pinned checkpoint has no selected encoder prefix')
    original = encoder.state_dict()
    nonfunctional = {name for name in original if name.startswith('fc.') or
                     name.endswith('.num_batches_tracked')}
    expected = set(original)-nonfunctional
    transferred = {}
    mapping = {}
    for name in source_keys:
        destination = encoder_destination(name[len(prefix):])
        if destination in transferred:
            raise ValueError('Duplicate pretrained mapping: '+destination)
        if destination not in expected:
            raise ValueError('Unexpected pretrained functional key: '+destination)
        value = source[name]
        if value.shape != original[destination].shape or value.dtype != original[destination].dtype:
            raise ValueError('Pretrained tensor shape/dtype mismatch: '+name)
        transferred[destination] = value
        mapping[name] = destination
    if set(transferred) != expected:
        raise ValueError('Missing pretrained functional encoder keys: '+str(sorted(expected-set(transferred))))
    complete = dict(original)
    complete.update(transferred)
    generated = []
    for name in sorted(nonfunctional):
        if name.endswith('.num_batches_tracked'):
            complete[name] = torch.zeros_like(complete[name])
            generated.append(name)
    encoder.load_state_dict(complete, strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    return dict(strict_load=True, copied_tensors=len(mapping), mapping=mapping,
                generated_unused_bn_counters=generated,
                retained_unused_classifier=sorted(name for name in nonfunctional if name.startswith('fc.')),
                encoder_frozen=True, checkpoint_component='ResNet50 only')
