"""Prepare/retrain identical Text+Kernel U-Nets on three explicit TRAIN splits."""

import argparse
import copy
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from segmentation.uniform_unet_fusion import (
    file_hash, checked_mode, validate_roles, annotation, word_counts, aggregate_counts,
)

REPO = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    temporary.replace(path)


def merge_targets(records, manifest_path, data_root):
    import yaml
    manifest = yaml.safe_load(Path(manifest_path).read_text())
    lookup = {row['id'].split(':')[-1]: row for row in manifest['samples']}
    result = []
    for record in records:
        row = lookup[record['sample_id'].split(':')[-1]]
        if row['image_sha256'] != record['image_sha256']:
            raise ValueError('Word-target RGB differs: '+record['sample_id'])
        value = dict(record)
        if 'validity_path' in value:
            value['evaluation_validity_path'] = value['validity_path']
            value['evaluation_validity_sha256'] = value['validity_sha256']
        for field, key in [('mask', 'mask'), ('validity', 'validity_annotation'),
                           ('separator', 'separation_annotation'), ('owners', 'word_owner_annotation')]:
            value[field+'_path'] = str(Path(data_root)/row[key])
            value[field+'_sha256'] = row[key+'_sha256']
        supplied_counts = manifest['acquisition'].get('care_words_by_id')
        value['care_words'] = (supplied_counts[row['id']] if supplied_counts is not None
                               else len(annotation(value)[0]))
        result.append(value)
    return result


def prepare(config_path, smoke):
    c = read(config_path)
    checked_mode(config_path, c)
    for path, digest in c['pinned_inputs'].items():
        if file_hash(path) != digest:
            raise ValueError('Changed pinned input: '+path)
    root = Path(c['output_root'])/('smoke' if smoke else 'full')
    if root.exists():
        raise ValueError('Preserve prepared run: '+str(root))
    icdar = read(c['icdar_population'])
    ic13 = read(c['icdar13_targets_plan'])
    tt = read(c['totaltext_population'])
    cases = {}
    for dataset in c['dataset_order']:
        if dataset == 'totaltext':
            train = [dict(r['token_sample'], split='train') for role in ('fit', 'calibration', 'selection') for r in tt['records'][role]]
            test = [dict(r['token_sample'], split='test') for r in tt['records']['test']]
            fit = sorted(r['sample_id'] for role in ('fit', 'calibration') for r in tt['records'][role])
            held = sorted(r['sample_id'] for r in tt['records']['selection'])
            train = merge_targets(train, c['target_manifests'][dataset], c['target_roots'][dataset])
        else:
            train = copy.deepcopy(icdar['records'][dataset]['train'])
            test = copy.deepcopy(icdar['records'][dataset]['test'])
            fit = icdar['partitions'][dataset]['fit']
            held = sorted(icdar['partitions'][dataset]['selection']+icdar['partitions'][dataset]['calibration'])
            if dataset == 'icdar13':
                train = [dict(r, **{k: v for k, v in ic13['records'][r['sample_id']].items()
                         if k.endswith('_path') or k.endswith('_sha256') or k == 'care_words'}) for r in train]
            else:
                train = merge_targets(train, c['target_manifests'][dataset], c['target_roots'][dataset])
        parts = dict(fit=fit, selection=held, calibration=[], test=[r['sample_id'] for r in test])
        lookup = {r['sample_id']: r for r in train+test}
        validate_roles(lookup, parts)
        if smoke:
            parts = {role: ids[:(8 if role == 'fit' else 4)] for role, ids in parts.items()}
        for record in train:
            if record['sample_id'] not in parts['fit']+parts['selection']:
                continue
            for field in ('image', 'annotation', 'mask', 'validity', 'separator', 'owners'):
                if file_hash(record[field+'_path']) != record[field+'_sha256']:
                    raise ValueError('Changed training bytes: '+record['sample_id']+':'+field)
        plan = dict(mode='approximate', claim_label='approximate reproduction', dataset=dataset,
            config=c['shared'], partitions=parts, train_records=train, test_records=test,
            paths=c['parseq'], initialization=c['initialization'], smoke=smoke,
            result_class='smoke-test result' if smoke else 'approximate reproduction',
            historical_test_exposure=True, upstream_text_encoder_train_exposure=True,
            test_used_for_selection=False, target=c['targets'][dataset], budget=c['budgets'][dataset],
            root_config_sha256=file_hash(config_path))
        cases[dataset] = plan
    root.mkdir(parents=True)
    sources = {}
    for package in c['snapshot_packages']:
        for source in (REPO/package).rglob('*.py'):
            relative = source.relative_to(REPO)
            dest = root/'code_snapshot'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
            sources[str(relative)] = file_hash(dest)
            if file_hash(source) != sources[str(relative)]:
                raise ValueError('Source changed during snapshot')
    write(root/'source_manifest.json', sources)
    write(root/'resolved_configuration.json', c)
    for dataset, plan in cases.items():
        plan['source_manifest_sha256'] = file_hash(root/'source_manifest.json')
        write(root/dataset/'resolved_configuration.json', plan)
    print('PREPARED', root, {k: {role: len(ids) for role, ids in p['partitions'].items()} for k, p in cases.items()}, flush=True)


def make_model(plan):
    import torch
    from segmentation.icdar13_word_refit import WordRefitUNet
    from models.text_pretrained_encoder import initialize_text_encoder
    c, initial = plan['config'], plan['initialization']
    net = WordRefitUNet(partial_encoder=False, skip_floor=c['skip_floor'], initial_gap_bias=c['initial_gap_bias'])
    state = torch.load(initial['source_checkpoint'], map_location='cpu', weights_only=False)
    net.load_state_dict(state['model_state_dict'], strict=True)
    del state
    donor = torch.load(initial['encoder_checkpoint'], map_location='cpu', weights_only=False)['model']
    receipt = initialize_text_encoder(net.encoder, donor, prefix=initial['encoder_prefix'])
    from segmentation.uniform_encoder_adaptation import enable_adaptation
    enable_adaptation(net, c)
    return net.to(c['device']), receipt


def validate(net, plan, out, epoch, ocr):
    import numpy as np
    import torch
    from segmentation.icdar13_word_refit import WordDataset
    from scripts.train_icdar13_word_refit import read_rows
    from scripts.icdar_budget_data import predictions, key
    from segmentation.icdar13_matched_training import emit_words
    c = plan['config']
    lookup = {r['sample_id']: r for r in plan['train_records']}
    records = [lookup[s] for s in plan['partitions']['selection']]
    dataset = WordDataset(records, c, augment=False)
    directory = out/'selection'/f'epoch_{epoch:03d}'
    if directory.exists():
        raise ValueError('Preserve previous validation: '+str(directory))
    counts = []
    net.eval()
    for index, r in enumerate(records):
        with torch.inference_mode():
            from segmentation.uniform_encoder_adaptation import validation_forward
            logits, context = validation_forward(net, dataset[index]['image'][None].to(c['device']), c)
            probability = logits[0].sigmoid().cpu().numpy()
        _, raw, expansion = read_rows(probability, r, c['inference'], ocr, c['parseq_batch_size'])
        rows = emit_words(raw, c['inference']['minimum_transcript_probability'])
        truth, validity = annotation(r)
        ignored = [p['ignored'] for p in predictions(rows, r, validity)]
        value = word_counts(rows, truth, ignored)
        counts.append(value)
        mp = directory/'maps'/(key(r['sample_id'])+'.npz')
        mp.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(probabilities=probability)
        if context is not None:
            arrays['encoder_context'] = context[0].cpu().numpy()
        np.savez_compressed(mp, **arrays)
        write(directory/'samples'/(key(r['sample_id'])+'.json'), dict(rows=rows, raw_rows=raw,
            counts=value, image_sha256=r['image_sha256'], maps_sha256=file_hash(mp),
            ground_truth_used_for_prediction=False, expansion=expansion))
    result = aggregate_counts(counts)
    write(directory/'metrics.json', dict(counts=result, deepsolo_images=0, total_images=len(records), deepsolo_percentage=0.))
    return result


def train(root):
    import hashlib
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from segmentation.icdar13_word_refit import WordDataset
    from segmentation.uniform_instance_objective import training_loss
    from scripts.run_icdar13_dense_instances import runtime
    from scripts.train_icdar13_word_refit import recognizer, save_checkpoint
    plan = read(root/'resolved_configuration.json')
    c, t = plan['config'], plan['config']['training']
    if (root/'complete.json').exists():
        print('ALREADY COMPLETE', root, flush=True)
        return
    if shutil.disk_usage(root).free < c['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient disk space for U-Net checkpoints and validation')
    runtime(c)
    net, initialization = make_model(plan)
    write(root/'initialization.json', initialization)
    from segmentation.uniform_encoder_adaptation import optimizer as make_optimizer
    optimizer = make_optimizer(net, c)
    lookup = {r['sample_id']: r for r in plan['train_records']}
    dataset_type = WordDataset
    if 'photometric_augmentation' in c:
        from segmentation.uniform_photometric_training import PhotometricWordDataset
        dataset_type = PhotometricWordDataset
    fit = dataset_type([lookup[s] for s in plan['partitions']['fit']], c, augment=True)
    history, best, best_epoch = [], None, None
    if (root/'last.pt').exists():
        state = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
        if state['plan_sha256'] != file_hash(root/'resolved_configuration.json'):
            raise ValueError('Resume plan changed')
        net.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        history, best, best_epoch = state['history'], state['best'], state['best_epoch']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(state['torch_rng']); torch.cuda.set_rng_state_all(state['cuda_rng'])
        del state
    ocr = recognizer(plan)
    try:
        if best is None:
            best = validate(net, plan, root, 0, ocr)
            if 'continuation_control' in plan and best != plan['continuation_control']['expected_initial_validation']:
                raise ValueError('Continuation epoch0 does not reproduce the original validation counts')
            best_epoch = 0
            save_checkpoint(root/'best.pt', dict(model_state_dict=net.state_dict(), epoch=0,
                validation=best, plan_sha256=file_hash(root/'resolved_configuration.json')))
        epochs = 1 if plan['smoke'] else t['epochs']
        for epoch in range(len(history)+1, epochs+1):
            started = time.monotonic()
            if shutil.disk_usage(root).free < c['minimum_free_disk_bytes']:
                raise RuntimeError('Disk reserve exhausted; last completed epoch remains resumable')
            fit.transform.set_epoch(epoch-1)
            loader = DataLoader(fit, batch_size=t['physical_batch_size'], shuffle=True,
                generator=torch.Generator().manual_seed(c['seed']+epoch),
                num_workers=t['num_workers'], pin_memory=True)
            net.train(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            seen, total, windows, window_images, steps = [], 0., 0, 0, 0
            def update():
                nonlocal windows, window_images, steps
                for parameter in net.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_images)
                torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],
                    t['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                windows = window_images = 0
                steps += 1
            for batch in loader:
                local = {k: v.to(c['device']) if torch.is_tensor(v) else v for k, v in batch.items()}
                logits = net(local['image'])
                # The inherited Dice loss aggregates within its input batch.
                # Compute per-image losses explicitly, identically for all domains.
                losses = [training_loss(logits[i:i+1],
                    {k: v[i:i+1] for k, v in local.items() if torch.is_tensor(v)}, c)
                    for i in range(len(batch['sample_id']))]
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite U-Net loss')
                count = len(batch['sample_id'])
                (loss*count).backward()
                windows += 1; window_images += count
                total += float(loss.detach())*count
                seen.extend(batch['sample_id'])
                if windows == t['gradient_accumulation']:
                    update()
            if windows:
                update()
            if sorted(seen) != sorted(plan['partitions']['fit']):
                raise ValueError('Incomplete epoch')
            row = dict(epoch=epoch, loss=total/len(seen), images=len(seen), steps=steps,
                order_sha256=hashlib.sha256(('\n'.join(seen)+'\n').encode()).hexdigest(),
                gpu_allocated=torch.cuda.max_memory_allocated(), gpu_reserved=torch.cuda.max_memory_reserved(),
                training_seconds=time.monotonic()-started)
            if epoch % c['validation_interval'] == 0 or epoch == epochs:
                row['selection'] = validate(net, plan, root, epoch, ocr)
                if row['selection']['f1'] > best['f1']:
                    best, best_epoch = row['selection'], epoch
                    save_checkpoint(root/'best.pt', dict(model_state_dict=net.state_dict(), epoch=epoch,
                        validation=best, plan_sha256=file_hash(root/'resolved_configuration.json')))
            history.append(row)
            save_checkpoint(root/'last.pt', dict(model_state_dict=net.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), history=history, best=best, best_epoch=best_epoch,
                plan_sha256=file_hash(root/'resolved_configuration.json'), python_rng=random.getstate(),
                numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()))
            write(root/'epochs.json', history)
            print(plan['dataset'], 'EPOCH', epoch, 'loss', row['loss'], 'best', best_epoch,
                  'selection', row.get('selection'), flush=True)
        write(root/'complete.json', dict(status='COMPLETE', epochs=epochs,
            selected_epoch=best_epoch, best_selection=best, checkpoint_sha256=file_hash(root/'best.pt'),
            gpu=torch.cuda.get_device_name(0), precision='float32', input_hw=c['image_hw'],
            native_hw=[v//2 for v in c['image_hw']], batch_size=t['physical_batch_size'],
            gradient_accumulation=t['gradient_accumulation'], out_of_memory=False,
            parameter_count=sum(p.numel() for p in net.parameters()),
            trainable_parameters=sum(p.numel() for p in net.parameters() if p.requires_grad),
            actual_new_deepsolo_calls=0))
    except BaseException as exc:
        write(root/f'failure_{time.time_ns()}.json', dict(error=repr(exc), traceback=traceback.format_exc(),
            out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError)))
        raise
    finally:
        ocr.release()
        del net
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage', choices=['prepare', 'train'], required=True)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.config, args.smoke)
    else:
        train(args.run_root)
