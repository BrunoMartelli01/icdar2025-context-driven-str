"""Verify every converted text-pretrained encoder stage against legacy outputs."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.models import resnet50

from models.text_pretrained_encoder import initialize_text_encoder
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = read(args.config)
    root = Path(c['output_root'])
    for path, digest in c['pinned_inputs'].items():
        assert sha(path) == digest
    legacy = read(root/'legacy_receipt.json')
    assert legacy['status'] == 'PASS' and legacy['config_sha256'] == sha(args.config)
    torch.set_num_threads(c['cpu_threads'])
    torch.manual_seed(20261029)
    net = resnet50(weights=None)
    state = torch.load(c['checkpoint'], map_location='cpu', weights_only=False)['model']
    mapping = initialize_text_encoder(net, state, prefix=c['prefix'])
    assert mapping['copied_tensors'] == 265
    comparisons = []
    for sample in c['inputs']:
        path = root/(sample['id']+'_legacy_features.npz')
        assert legacy['files_sha256'][str(path)] == sha(path)
        assert sha(sample['path']) == sample['sha256']
        x = torch.from_numpy(np.load(sample['path'], allow_pickle=False))
        with torch.no_grad():
            stem = net.relu(net.bn1(net.conv1(x)))
            r2 = net.layer1(net.maxpool(stem))
            r3 = net.layer2(r2)
            r4 = net.layer3(r3)
            r5 = net.layer4(r4)
        actual = dict(stem=stem, res2=r2, res3=r3, res4=r4, res5=r5)
        with np.load(path, allow_pickle=False) as expected:
            for name, value in actual.items():
                a, b = value.numpy(), expected[name]
                assert a.shape == b.shape
                np.testing.assert_allclose(a, b, atol=c['atol'], rtol=c['rtol'])
                comparisons.append(dict(sample_id=sample['sample_id'], stage=name,
                    shape=list(a.shape), max_absolute_error=float(np.abs(a-b).max()),
                    rms_error=float(np.sqrt(np.mean((a-b)**2)))))
    # The unused classification layer is retained deterministically;
    # the functional pretrained encoder contains exactly the265 mapped tensors.
    output = root/'converted_encoder_state.pt'
    torch.save(net.state_dict(), output)
    seal(root/'main_receipt.json', dict(status='PASS', comparisons=comparisons,
        mapping=mapping, converted_state_sha256=sha(output), config_sha256=sha(args.config),
        source_checkpoint_sha256=c['pinned_inputs'][c['checkpoint']],
        full_deepsolo_recognizer_calls=0, component='ResNet50 encoder only',
        torch=torch.__version__, device='cpu', code_sha256=sha(__file__),
        limitation='Text-pretrained encoder saw official IC13 TRAIN, including heldout head validation; no fresh-validation or strict reproduction claim.'))
    print('PASS', len(comparisons), 'cross-environment feature maps',
          'max error', max(p['max_absolute_error'] for p in comparisons), flush=True)


if __name__ == '__main__':
    main()
