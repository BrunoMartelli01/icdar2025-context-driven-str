import pytest
import torch
from torchvision.models import resnet50

from models.text_pretrained_encoder import encoder_destination, initialize_text_encoder


def reverse_keys(encoder):
    source = {}
    for name, value in encoder.state_dict().items():
        if name.startswith('fc.') or name.endswith('.num_batches_tracked'):
            continue
        if name == 'conv1.weight':
            original = 'stem.conv1.weight'
        elif name.startswith('bn1.'):
            original = 'stem.conv1.norm.'+name[len('bn1.'):]
        else:
            stage, block, *suffix = name.split('.')
            prefix = 'res'+str(int(stage[len('layer'):])+1)+'.'+block+'.'
            text = '.'.join(suffix)
            if text.startswith('downsample.0.'):
                original = prefix+'shortcut.'+text[len('downsample.0.'):]
            elif text.startswith('downsample.1.'):
                original = prefix+'shortcut.norm.'+text[len('downsample.1.'):]
            elif text.startswith('bn'):
                original = prefix+'conv'+text[2]+'.norm.'+text[4:]
            else:
                original = prefix+text
        source['body.'+original] = value.clone()
    return source


def test_complete_functional_mapping_reproduces_encoder_and_fails_missing_key():
    encoder = resnet50(weights=None).eval()
    source = reverse_keys(encoder)
    assert len(source) == 265
    expected = {name: value.clone() for name, value in encoder.state_dict().items()}
    with torch.no_grad():
        encoder.conv1.weight.zero_()
    receipt = initialize_text_encoder(encoder, source, prefix='body.')
    assert receipt['copied_tensors'] == 265 and receipt['strict_load']
    for name, value in expected.items():
        assert torch.equal(encoder.state_dict()[name], value)
    assert all(not p.requires_grad for p in encoder.parameters())
    incomplete = dict(source)
    incomplete.pop(next(iter(incomplete)))
    with pytest.raises(ValueError, match='Missing'):
        initialize_text_encoder(encoder, incomplete, prefix='body.')


def test_mapping_rejects_unsupported_deformable_or_changed_tensor():
    assert encoder_destination('res4.2.shortcut.norm.running_var') == 'layer3.2.downsample.1.running_var'
    with pytest.raises(ValueError):
        encoder_destination('res4.2.conv2_offset.weight')
    encoder = resnet50(weights=None)
    source = reverse_keys(encoder)
    source['body.stem.conv1.weight'] = torch.zeros(1)
    with pytest.raises(ValueError, match='shape/dtype'):
        initialize_text_encoder(encoder, source, prefix='body.')
