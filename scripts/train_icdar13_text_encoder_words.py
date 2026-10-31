"""Matched word-U-Net adaptation with ImageNet versus text-pretrained encoders."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.text_pretrained_encoder import initialize_text_encoder
from segmentation.icdar13_word_refit import WordRefitUNet, WordDataset
from segmentation.icdar13_matched_training import matched_word_loss
from scripts.icdar_budget_data import read, sha
from scripts.run_yolo_f1_quality_study import seal
from scripts.run_icdar13_dense_instances import runtime
from scripts.train_icdar13_word_refit import recognizer, save_checkpoint
from scripts.run_icdar13_matched_word_training import infer_maps, score_maps

REPO = Path(__file__).resolve().parents[1]


def progress(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)


def prepare(args):
    if args.config.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate configuration required')
    c = read(args.config)
    assert c['mode'] == 'approximate'
    for path, digest in c['pinned_inputs'].items():
        assert sha(path) == digest, path
    donor = read(c['encoder_qualification'])
    assert donor['status'] == 'PASS' and donor['mapping']['copied_tensors'] == 265
    parent = read(c['parent_plan'])
    partitions = dict(parent['partitions'])
    tt_ids = list(parent['tt_ids'])
    if args.smoke:
        partitions = {role: ids[:(8 if role == 'fit' else 4)] for role, ids in partitions.items()}
        tt_ids = tt_ids[:8]
    fitting = tt_ids+partitions['fit']
    records = parent['records']
    test_hashes = {r['rgb_sha256'] for r in parent['test_records']}
    for sid in fitting:
        record = records[sid]
        assert record['rgb_sha256'] not in test_hashes
        for field in ('image', 'mask', 'owners', 'validity', 'separator'):
            assert sha(record[field+'_path']) == record[field+'_sha256']
    label = args.run_name or ('smoke' if args.smoke else 'full')
    if Path(label).name != label:
        raise ValueError('Run name must be a single path component')
    root = Path(c['output_root'])/label
    root.mkdir(parents=True, exist_ok=False)
    plan = dict(mode='approximate', claim_label='approximate reproduction', config=c,
        config_sha256=sha(args.config), train_records=list(records.values()),
        test_records=parent['test_records'], partitions=partitions,
        fitting_ids=fitting, tt_ids=tt_ids, smoke=args.smoke,
        result_class='smoke-test result' if args.smoke else 'approximate reproduction',
        historical_test_exposure=True, test_used_for_selection=False,
        encoder_validation_exposure='The text-pretrained donor used IC13 TRAIN, including new-head validation; not a fresh encoder holdout',
        hardware=dict(model_set='One frozen-encoder Attention-Gated word U-Net and genuine PARSeq; no DeepSolo recognizer',
            precision='float32', image_hw=c['image_hw'], physical_batch=c['training']['physical_batch_size'],
            gradient_accumulation=c['training']['gradient_accumulation'], parseq_batch=c['parseq_batch_size']),
        paths={k: parent['paths'][k] for k in ('parseq_source', 'parseq_archive', 'parseq_checkpoint')})
    files = {}
    for folder in ('models', 'segmentation', 'scripts', 'recognition', 'localize',
                   'caption', 'scoring', 'pipeline', 'research'):
        for path in (REPO/folder).rglob('*.py'):
            relative = path.relative_to(REPO)
            dest = root/'code_snapshot'/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            data = path.read_bytes()
            dest.write_bytes(data)
            files[str(relative)] = hashlib.sha256(data).hexdigest()
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    plan['code_sha256'] = files
    seal(root/'resolved_configuration.json', plan)
    print('PREPARED', root, len(fitting), 'fitting images', flush=True)


def model(plan, arm):
    c = plan['config']
    net = WordRefitUNet(partial_encoder=False, skip_floor=c['skip_floor'],
                        initial_gap_bias=c['initial_gap_bias'])
    source = torch.load(c['source_checkpoint'], map_location='cpu', weights_only=False)
    net.load_state_dict(source['model_state_dict'], strict=True)
    receipt = dict(initialization='ImageNet encoder from qualified complete TT U-Net state')
    if arm == 'text_encoder':
        donor = torch.load(c['encoder_checkpoint'], map_location='cpu', weights_only=False)['model']
        receipt = initialize_text_encoder(net.encoder, donor, prefix=c['encoder_prefix'])
    elif arm != 'imagenet_control':
        raise ValueError('Unknown initialization arm')
    return net.to(c['device']), receipt


def validate(net, plan, ocr, out, epoch):
    directory = out/'selection'/('epoch_'+str(epoch).zfill(3))
    infer_maps(net, plan, 'selection', directory/'maps')
    return score_maps(plan, 'selection', directory/'maps', plan['config']['inference'],
                      ocr, directory)['counts']


def train(plan, root, arm):
    c, t = plan['config'], plan['config']['training']
    if shutil.disk_usage(root).free < c['minimum_free_disk_bytes']:
        raise RuntimeError('Insufficient disk space for complete joint checkpoints')
    torch.manual_seed(c['seed'])
    np.random.seed(c['seed'])
    net, initialization = model(plan, arm)
    out = root/arm
    if (out/'complete.json').exists():
        return
    seal(out/'initialization.json', initialization)
    lookup = {r['sample_id']: r for r in plan['train_records']}
    dataset = WordDataset([lookup[sid] for sid in plan['fitting_ids']], c, augment=True)
    optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],
                                  lr=t['learning_rate'], weight_decay=t['weight_decay'])
    history, best, best_epoch = [], None, None
    if (out/'last.pt').exists():
        state = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        assert state['plan_sha256'] == sha(root/'resolved_configuration.json')
        net.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer_state_dict'])
        history, best, best_epoch = state['history'], state['best'], state['best_epoch']
    ocr = recognizer(plan)
    try:
        if best is None:
            best = validate(net, plan, ocr, out, 0)
            best_epoch = 0
            save_checkpoint(out/'best.pt', dict(model_state_dict=net.state_dict(), epoch=0,
                selection=best, plan_sha256=sha(root/'resolved_configuration.json')))
        epochs = 1 if plan['smoke'] else t['epochs']
        start = time.monotonic()
        for epoch in range(len(history)+1, epochs+1):
            dataset.transform.set_epoch(epoch-1)
            loader = DataLoader(dataset, batch_size=t['physical_batch_size'], shuffle=True,
                generator=torch.Generator().manual_seed(c['seed']+epoch),
                num_workers=t['num_workers'], pin_memory=True)
            net.train()
            optimizer.zero_grad(set_to_none=True)
            seen, total, window, window_images, steps = [], 0., 0, 0, 0
            torch.cuda.reset_peak_memory_stats()
            def update():
                nonlocal window, window_images, steps
                for parameter in net.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(window_images)
                torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],
                                               t['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                window = window_images = 0
                steps += 1
            for batch in loader:
                local = {k: v.to(c['device']) if torch.is_tensor(v) else v for k, v in batch.items()}
                logits = net(local['image'])
                losses = []
                for i, sid in enumerate(batch['sample_id']):
                    item = {k: v[i:i+1] for k, v in local.items() if torch.is_tensor(v)}
                    loss = matched_word_loss(logits[i:i+1], item, c, 'totaltext_bce_dice')
                    losses.append(loss*(t['icdar_loss_weight'] if sid.startswith('icdar13:') else 1.))
                loss = torch.stack(losses).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite word segmentation loss')
                n = len(batch['sample_id'])
                (loss*n).backward()
                total += float(loss.detach())*n
                window += 1
                window_images += n
                seen.extend(batch['sample_id'])
                if window == t['gradient_accumulation']:
                    update()
            if window:
                update()
            assert sorted(seen) == sorted(plan['fitting_ids'])
            row = dict(epoch=epoch, loss=total/len(seen), images=len(seen), steps=steps,
                order_sha256=hashlib.sha256(('\n'.join(seen)+'\n').encode()).hexdigest(),
                gpu_allocated=torch.cuda.max_memory_allocated(), gpu_reserved=torch.cuda.max_memory_reserved())
            if epoch % c['validation_interval'] == 0 or epoch == epochs:
                value = validate(net, plan, ocr, out, epoch)
                row['selection'] = value
                if value['f1'] > best['f1']:
                    best, best_epoch = value, epoch
                    save_checkpoint(out/'best.pt', dict(model_state_dict=net.state_dict(),
                        epoch=epoch, selection=best, plan_sha256=sha(root/'resolved_configuration.json')))
            history.append(row)
            save_checkpoint(out/'last.pt', dict(model_state_dict=net.state_dict(),
                optimizer_state_dict=optimizer.state_dict(), history=history,
                best=best, best_epoch=best_epoch, plan_sha256=sha(root/'resolved_configuration.json')))
            progress(out/'epochs.json', history)
            print(arm, 'EPOCH', epoch, 'loss', row['loss'], 'selection', row.get('selection'),
                  'best_epoch', best_epoch, 'seconds', time.monotonic()-start, flush=True)
        seal(out/'complete.json', dict(status='COMPLETE', fit_ids=plan['fitting_ids'],
            epochs=epochs, selected_epoch=best_epoch, best_selection=best,
            checkpoint_sha256=sha(out/'best.pt'), actual_new_deepsolo_recognizer_calls=0))
    finally:
        ocr.release()
        del net
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--run-root', type=Path)
    parser.add_argument('--stage', choices=['prepare', 'train'], required=True)
    parser.add_argument('--arm', choices=['text_encoder', 'imagenet_control'])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--run-name')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args)
        return
    if args.run_root is None or args.arm is None:
        raise ValueError('Explicit prepared run and arm required')
    root = args.run_root
    plan = read(root/'resolved_configuration.json')
    for path, digest in plan['config']['pinned_inputs'].items():
        assert sha(path) == digest, path
    runtime(plan['config'])
    try:
        train(plan, root, args.arm)
    except BaseException:
        seal(root/('failure_'+args.arm+'_'+str(time.time_ns())+'.json'),
             dict(status='FAILED', traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
