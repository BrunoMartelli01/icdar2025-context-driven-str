"""Export only a pinned legacy ResNet50 encoder for cross-environment checks."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from deepsolo.candidates import verify_source_identity


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assert config['mode'] == 'approximate'
    for path, digest in config['pinned_inputs'].items():
        assert sha(path) == digest, path
    identity = verify_source_identity(Path(config['legacy_source']),
        source_type=config['source_type'], declared_revision=config['source_revision'],
        source_commit_marker=Path(config['source_commit_marker']),
        source_archive=Path(config['source_archive']),
        source_archive_sha256=config['source_archive_sha256'])
    revision = identity['source_revision']
    sys.path.insert(0, config['legacy_source'])
    from adet.config import get_cfg
    from detectron2.modeling import build_backbone
    cfg = get_cfg()
    cfg.merge_from_file(config['legacy_config'])
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.RESNETS.OUT_FEATURES = ['res2', 'res3', 'res4', 'res5']
    assert cfg.MODEL.RESNETS.STRIDE_IN_1X1 is False
    assert cfg.MODEL.RESNETS.RES5_DILATION == 1
    assert cfg.MODEL.RESNETS.NORM == 'FrozenBN'
    assert list(cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE) == [False]*4
    assert list(cfg.MODEL.PIXEL_MEAN) == [123.675, 116.28, 103.53]
    assert list(cfg.MODEL.PIXEL_STD) == [58.395, 57.12, 57.375]
    assert cfg.INPUT.FORMAT == 'RGB'
    torch.set_num_threads(config['cpu_threads'])
    model = build_backbone(cfg).eval()
    source = torch.load(config['checkpoint'], map_location='cpu')['model']
    prefix = config['prefix']
    state = {name[len(prefix):]: value for name, value in source.items() if name.startswith(prefix)}
    model.load_state_dict(state, strict=True)
    files = {}
    root = Path(config['output_root'])
    for sample in config['inputs']:
        value = np.load(sample['path'], allow_pickle=False)
        assert sha(sample['path']) == sample['sha256'] and value.dtype == np.float32
        with torch.no_grad():
            x = torch.from_numpy(value)
            output = model(x)
            stem = torch.relu(model.stem.conv1(x))
        path = root/(sample['id']+'_legacy_features.npz')
        np.savez_compressed(path, stem=stem.numpy(), **{k: v.numpy() for k, v in output.items()})
        files[str(path)] = sha(path)
    receipt = dict(status='PASS', component='ResNet50 encoder only',
        full_deepsolo_recognizer_calls=0, device='cpu', revision=revision,
        state_tensors=len(state), torch=torch.__version__, files_sha256=files,
        config_sha256=sha(args.config), code_sha256=sha(__file__), source_identity=identity)
    (root/'legacy_receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True)+'\n')
    print('PASS legacy encoder', len(state), 'tensors', len(files), 'images')


if __name__ == '__main__':
    main()
