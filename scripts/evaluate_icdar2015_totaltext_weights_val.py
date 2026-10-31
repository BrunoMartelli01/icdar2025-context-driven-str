"""Compare pinned Total-Text weights on complete fixed ICDAR TRAIN VAL100."""

import argparse
from pathlib import Path
import socket

import torch
from torch.utils.data import DataLoader, Subset

from segmentation.data import SegmentationDataset, file_sha256
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.icdar2015_improvements import OwnerDataset, make_transform
from segmentation.icdar2015_improvement_runtime import load_training_population, references, evaluate_geometry, source_hashes
from segmentation.icdar2015_target70 import ValidationOCR
from segmentation.icdar2015_totaltext_weights import load_transfer_profile, load_transfer_source, verify_disjoint_manifests
from segmentation.training import seed_everything, runtime_environment
from scripts.train_icdar2015_totaltext_transfer import write_json


def run(args):
    config = load_transfer_profile(args.config); branch = config['branches']['native1280']
    manifest, fit, val = load_training_population(config, args.data_root)
    exposure = verify_disjoint_manifests(config, manifest, args.totaltext_manifest)
    if args.output_dir.exists():
        raise ValueError('Preserve existing validation output')
    seed_everything(config['training']['seed'], deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no substitution')
        _prepare_cuda_memory_tracking(device)
    model, source = load_transfer_source(args.source_checkpoint, args.source_report,
        config['sources'][args.source], config, branch)
    model.to(device); shape = branch['image_hw']; refs = references(manifest, args.data_root, val)
    data = OwnerDataset(SegmentationDataset(manifest, args.data_root,
        transform=make_transform(config, shape, augment=False), verify_checksums=True))
    loader = DataLoader(Subset(data, val), batch_size=1, shuffle=False, num_workers=2)
    contract = dict(config=config, config_sha256=file_sha256(args.config), source=args.source,
        source_report_sha256=file_sha256(args.source_report), source_checkpoint_sha256=file_sha256(args.source_checkpoint),
        source_run_contract=source['run_contract'], exposure=exposure,
        train_ids=[manifest.samples[i].sample_id for i in fit], validation_ids=list(refs),
        data_root=str(args.data_root.resolve()), device=args.device, image_hw=shape, precision='float32',
        source_sha256=source_hashes('segmentation/icdar2015_target70.py',
            'segmentation/icdar2015_totaltext_weights.py', 'models/icdar2015_target70.py',
            'scripts/evaluate_icdar2015_totaltext_weights_val.py'))
    write_json(args.output_dir/'run_contract.json', contract)
    masks = args.output_dir/'masks'
    details = evaluate_geometry(model, loader, refs, config, shape, device=device, save_dir=masks)
    write_json(args.output_dir/'geometry.json', details)
    ocr = ValidationOCR(args, config)
    words = ocr.evaluate(details, refs, manifest, args.data_root, shape, masks)
    write_json(args.output_dir/'words.json', words)
    result = dict(status='SUCCESS', reproduction_mode='approximate', claim_label='approximate reproduction',
        result_class='complete fixed TRAIN VAL100; not TEST', run_contract=contract,
        validation_geometry=details['aggregate'], validation_words=words['aggregate'],
        validation_emitted=words['emitted_aggregate'], emission_policy=words['emission_policy'],
        runtime=runtime_environment(), hostname=socket.gethostname(), failed_samples=0, oom=False,
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
    write_json(args.output_dir/'results.json', result)
    print(dict(source=args.source, geometry=details['aggregate']['proposal_f1_iou_50'],
        raw=words['aggregate']['alphanumeric_casefold'], emitted=words['emitted_aggregate']['alphanumeric_casefold']), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ['config','data-root','totaltext-manifest','source-checkpoint','source-report',
              'parseq-source','parseq-source-archive','parseq-checkpoint','output-dir']:
        parser.add_argument('--'+n, type=Path, required=True)
    parser.add_argument('--source', choices=['tt263','tt_partial23'], required=True)
    parser.add_argument('--device', required=True)
    args=parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Preserve existing validation output')
    try:
        run(args)
    except Exception as exc:
        write_json(args.output_dir/'failure.json', dict(status='FAILED', error_type=type(exc).__name__,
            error=str(exc), oom=isinstance(exc,torch.cuda.OutOfMemoryError)))
        raise


if __name__ == '__main__': main()
