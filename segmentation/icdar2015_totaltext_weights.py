"""Explicit, checksum-pinned Total-Text weight transfer; approximate only."""

import json

import torch

from models.icdar2015_target70 import Target70AGUNet
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.icdar2015_improvement_runtime import REPO
from segmentation.icdar2015_target70 import load_profile


def load_transfer_profile(path):
    config = load_profile(path)
    if config['experiment_kind'] != 'icdar2015_explicit_totaltext_weight_transfer':
        raise ValueError('Explicit Total-Text weight-transfer profile required')
    if set(config['sources']) != {'tt263', 'tt_partial23'}:
        raise ValueError('Both predeclared Total-Text sources required')
    original = REPO / config['source_training_config']
    if file_sha256(original) != config['source_training_config_sha256']:
        raise ValueError('Fixed ICDAR training/split configuration changed')
    return config


def verify_source_metadata(payload, report, spec):
    if (report['status'] != 'SUCCESS' or not report['training_complete']
            or report['checkpoint']['sha256'] != spec['checkpoint_sha256']
            or report['checkpoint']['epoch'] != spec['epoch']
            or payload['architecture'] != spec['architecture']
            or payload['epoch'] != spec['epoch']
            or payload['run_contract'] != report['run_contract']
            or report['run_contract']['manifest_sha256'] != spec['manifest_sha256']
            or report['run_contract']['architecture'] != spec['architecture']):
        raise ValueError('Completed Total-Text source provenance mismatch')


def load_transfer_source(checkpoint, source_report, spec, config, branch):
    if (file_sha256(checkpoint) != spec['checkpoint_sha256']
            or file_sha256(source_report) != spec['run_manifest_sha256']):
        raise ValueError('Pinned Total-Text source bytes changed')
    report = json.loads(source_report.read_text())
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    verify_source_metadata(payload, report, spec)
    model = Target70AGUNet(encoder_stages=branch['encoder_trainable_stages'],
        skip_floor=config['skip_floor'], max_groups=config['group_norm_max_groups'],
        source_state=payload['model_state_dict'])  # Complete strict=True load.
    return model, report


def verify_disjoint_manifests(config, icdar_manifest, totaltext_manifest):
    expected = config['datasets']['total_text']['manifest_sha256']
    if file_sha256(totaltext_manifest) != expected:
        raise ValueError('Pinned Total-Text exposure manifest changed')
    tt = load_segmentation_manifest(totaltext_manifest, require_checksums=True)
    if len(tt.samples) != config['datasets']['total_text']['images']:
        raise ValueError('Incomplete Total-Text exposure population')
    overlap = {s.image_sha256 for s in tt.samples} & {s.image_sha256 for s in icdar_manifest.samples}
    if overlap:
        raise ValueError('Exact image-hash overlap between Total-Text and ICDAR TRAIN')
    return dict(totaltext_images=len(tt.samples), icdar_train_images=len(icdar_manifest.samples),
                exact_image_hash_overlap=0, scope='Complete recorded manifests; not perceptual duplicate detection')


def fresh_optimizer(model, config, branch):
    t = config['training']
    decoder = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('encoder.')]
    encoder = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith('encoder.')]
    if not decoder or not encoder:
        raise ValueError('Explicit decoder and partial-encoder parameter groups required')
    return torch.optim.Adam([
        dict(params=decoder, lr=t['learning_rate'], initial_lr=t['learning_rate'], group_name='decoder'),
        dict(params=encoder, lr=branch['encoder_learning_rate'], initial_lr=branch['encoder_learning_rate'],
             group_name='encoder_layer3_layer4')], weight_decay=t['weight_decay'],
        betas=tuple(t['adam_betas']), eps=t['adam_epsilon'])


def raw_word_selection_key(raw, emitted, geometry, epoch):
    return (raw['alphanumeric_casefold']['f1'], emitted['alphanumeric_casefold']['f1'],
            geometry['proposal_f1_iou_50'], -epoch)
