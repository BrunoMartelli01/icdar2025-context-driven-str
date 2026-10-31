"""Export qualified full or decoder-heldout U-Net/PARSeq observations."""

import argparse
import gc
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from PIL import Image
import torch

from recognition.unet_expert_records import observation
from segmentation.icdar13_word_refit import WordRefitUNet
from segmentation.icdar2015_improvements import make_transform
from scripts.icdar_budget_data import read, sha, key
from scripts.run_icdar13_dense_instances import runtime
from scripts.run_yolo_f1_quality_study import seal
from scripts.train_icdar13_word_refit import read_rows, recognizer

REPO = Path(__file__).resolve().parents[1]


def checked(path):
    if path.resolve().parent != REPO / 'configs/approximate':
        raise ValueError('Explicit approximate U-Net export config required')
    c = read(path)
    if c['mode'] != 'approximate' or c['kind'] not in ('seen', 'fold', 'test'):
        raise ValueError('Explicit approximate export kind required')
    for file, digest in c['pinned_inputs'].items():
        if sha(file) != digest:
            raise ValueError('Changed pinned U-Net export input: ' + file)
    return c


def verify_full_expert(c):
    frozen = read(c['frozen_expert']); old = read(c['source_policy'])
    audit = read(c['source_expert_audit'])
    if (frozen['status'] != 'FROZEN' or frozen['test_used_for_selection'] or
            old['status'] != 'FROZEN' or old['test_used_for_selection'] or
            audit['status'] != 'PASS' or audit['selection_only'] or
            sha(old['choice']['checkpoint']) != old['choice']['checkpoint_sha256'] or
            frozen['winner']['checkpoint_sha256'] != old['choice']['checkpoint_sha256'] or
            frozen['source_policy_sha256'] != sha(c['source_policy'])):
        raise ValueError('Independently qualified full-fitting U-Net required')
    return frozen, old


def write_sample(root, sid, record, raw, rows, maps, map_path, checkpoint, frozen_path, *, excluded):
    target = root / 'samples' / (key(sid) + '.json')
    seal(target, dict(sample_id=sid, rows=rows, raw_rows=raw, map_summary=maps,
        map_path=str(map_path), map_sha256=sha(map_path), ground_truth_used=False,
        image_sha256=record['image_sha256'], checkpoint_sha256=checkpoint,
        frozen_expert_sha256=sha(frozen_path), image_excluded=False,
        decoder_fit_excluded=excluded, upstream_encoder_exposure=True))
    return dict(path=str(target), sha256=sha(target))


def run(path):
    c = checked(path); frozen, old = verify_full_expert(c)
    original = read(c['original_plan']); split = 'test' if c['kind'] == 'test' else 'train'
    records = {r['sample_id']: r for r in original['records']['icdar13'][split]}
    ids = c['prediction_ids']
    if len(ids) != len(set(ids)) or not set(ids) <= set(records):
        raise ValueError('Export must use the explicit distinct image population')
    policy = frozen['winner']['policy']; settings = dict(old['choice']['policy'])
    settings['minimum_transcript_probability'] = 0.0
    root = Path(c['output_root'])
    if root.exists():
        raise ValueError('Preserve existing U-Net export')
    if c['kind'] == 'test':
        gate = Path(c['frozen_router'])
        for name in ('selection_frozen.json', 'deployment_qualification.json'):
            if c['pinned_inputs'].get(str(gate / name)) != sha(gate / name):
                raise ValueError('TEST export requires an explicitly pinned trained gate')
        if read(gate / 'selection_frozen.json')['test_used_for_selection'] or read(gate / 'deployment_qualification.json')['status'] != 'PASS':
            raise ValueError('Qualified frozen TRAIN-only gate required before TEST export')
    root.mkdir(parents=True, exist_ok=False)
    code = {}
    for package in c['snapshot_packages']:
        for source in (REPO / package).rglob('*.py'):
            dest = root / 'code_snapshot' / source.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, dest)
            code[str(source.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO / 'configs', root / 'code_snapshot/configs')
    checkpoint = old['choice']['checkpoint_sha256']; training_plan = read(c['full_training_plan'])
    if c['kind'] == 'fold':
        train_root = Path(c['fold_training_root']); training_plan = read(train_root / 'resolved_configuration.json')
        complete = read(train_root / 'text_encoder/complete.json'); audit = read(train_root / 'independent_audit.json')
        if (complete['status'] != 'COMPLETE' or complete['epochs'] != c['fixed_epochs'] or
                audit['status'] != 'PASS' or set(ids) != set(training_plan['partitions']['decoder_heldout']) or
                set(ids) & set(training_plan['fitting_ids']) or set(ids) & set(training_plan['partitions']['selection'])):
            raise ValueError('Complete independently audited decoder-heldout training required')
        checkpoint_path = train_root / 'text_encoder/last.pt'; checkpoint = sha(checkpoint_path)
    else:
        checkpoint_path = Path(old['choice']['checkpoint'])
    seal(root / 'resolved_configuration.json', dict(mode='approximate', claim_label='approximate reproduction',
        config=c, config_sha256=sha(path), source_sha256=code, checkpoint_sha256=checkpoint,
        ground_truth_used=False, decoder_fit_excluded=c['kind'] == 'fold', upstream_encoder_exposure=True,
        test_used_for_selection=False, historical_test_exposure=True))
    manifest = {}; crops = 0; memory = None; started = time.monotonic(); reader = net = None
    try:
        if c['kind'] == 'seen':
            source_root = Path(c['source_expert_root']) / 'selected_expert'
            role_index = {}
            for role in ('fit', 'selection', 'calibration'):
                for sid in training_plan['partitions'][role]:
                    if sid in role_index:
                        raise ValueError('Duplicate qualified full-expert role')
                    role_index[sid] = role
            if set(ids) != set(role_index):
                raise ValueError('Complete qualified full-expert TRAIN export required')
            for sid in ids:
                record = records[sid]; role = role_index[sid]
                source = source_root / role / 'samples' / (key(sid) + '.json')
                mp = source_root / role / 'maps' / (key(sid) + '.npz'); sample = read(source)
                if (sample['ground_truth_used_for_prediction'] or sample['image_sha256'] != record['image_sha256'] or
                        sample['policy_sha256'] != sha(c['source_policy']) or sample['maps_sha256'] != sha(mp)):
                    raise ValueError('Changed authenticated full U-Net observation')
                with np.load(mp, allow_pickle=False) as z:
                    raw, emitted, maps = observation(sample['raw_rows'], z['probabilities'], record['image_wh'], policy, c['map_settings'])
                manifest[sid] = write_sample(root, sid, record, raw, emitted, maps, mp, checkpoint, c['frozen_expert'], excluded=False)
        else:
            runtime(training_plan['config']); torch.cuda.reset_peak_memory_stats()
            state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if c['kind'] == 'fold':
                if len(state['history']) != c['fixed_epochs'] or state['plan_sha256'] != sha(Path(c['fold_training_root']) / 'resolved_configuration.json'):
                    raise ValueError('Fold must export its fixed final epoch')
            elif state['epoch'] != c['fixed_epochs']:
                raise ValueError('Full expert epoch changed')
            tc = training_plan['config']
            net = WordRefitUNet(partial_encoder=False, skip_floor=tc['skip_floor'], initial_gap_bias=tc['initial_gap_bias'])
            net.load_state_dict(state['model_state_dict'], strict=True); del state
            net = net.to(tc['device']).eval(); reader = recognizer(training_plan)
            transform = make_transform(tc, tc['image_hw'], augment=False)
            for sid in ids:
                source = records[sid]
                record = {k: source[k] for k in ('sample_id', 'image_path', 'image_sha256', 'image_wh')}
                if sha(record['image_path']) != record['image_sha256']:
                    raise ValueError('Changed original image before U-Net inference')
                with Image.open(record['image_path']) as image:
                    rgb = image.convert('RGB'); x, _ = transform(rgb, Image.new('L', rgb.size, 0), 0)
                with torch.inference_mode():
                    probability = net(x[None].to(tc['device']))[0].sigmoid().cpu().numpy()
                _, candidates, _ = read_rows(probability, record, settings, reader, tc['parseq_batch_size'])
                raw, emitted, maps = observation(candidates, probability, record['image_wh'], policy, c['map_settings'])
                mp = root / 'maps' / (key(sid) + '.npz'); mp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(mp, probabilities=probability)
                manifest[sid] = write_sample(root, sid, record, raw, emitted, maps, mp, checkpoint, c['frozen_expert'], excluded=c['kind'] == 'fold')
                crops += len(raw)
            torch.cuda.synchronize()
            memory = dict(gpu=torch.cuda.get_device_name(0), precision='float32', batch_size=1,
                image_hw=tc['image_hw'], parseq_batch_size=tc['parseq_batch_size'],
                peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved())
    except BaseException as exc:
        seal(root / 'failure.json', dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc(),
            out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError)))
        raise
    finally:
        if reader is not None:
            reader.release()
        if net is not None:
            del net
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    seal(root / 'index.json', manifest)
    seal(root / 'completion.json', dict(status='COMPLETE', mode='approximate', images=len(ids), split=split,
        actual_unet_image_inferences=0 if c['kind'] == 'seen' else len(ids),
        actual_parseq_crop_inferences=crops, actual_new_deepsolo_inferences=0,
        decoder_fit_excluded=c['kind'] == 'fold', upstream_encoder_exposure=True,
        ground_truth_used=False, test_used=split == 'test', memory=memory, out_of_memory=False,
        seconds=time.monotonic() - started, scope='Offline export; not end-to-end online latency'))
    print('EXPORTED', c['kind'], len(ids), 'crops', crops, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--config', type=Path, required=True)
    run(parser.parse_args().config)
