"""Controlled head-only/full YOLO word-box adaptation, TRAIN/validation only."""

import argparse
import copy
import importlib.metadata
from pathlib import Path
import shutil
import traceback

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer

from localize.word_detector_training import normalized_box, mask_training_ignore, strict_initialization, training_roles
from segmentation.separated_word_gt import parse_icdar13_character_groups
from segmentation.scene_text_preparation import clip_instances_to_pixel_extent
from scripts.icdar_budget_data import read, sha, key, annotation
from scripts.run_yolo_f1_quality_study import seal

REPO = Path(__file__).resolve().parents[1]


class LocalDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Core training is in DetectionTrainer. Remove optional external service
        # integrations before any training callback runs; keep local callbacks.
        self.callbacks = {event: [f for f in functions if f.__module__ in
            ('ultralytics.utils.callbacks.base', __name__)] for event, functions in self.callbacks.items()}


def export_data(plan, root, smoke):
    original = read(plan['original_plan']); parts = original['partitions']['icdar13']
    lookup = {r['sample_id']: r for r in original['records']['icdar13']['train']}
    protocol = plan.get('fit_protocol', 'heldout_selection')
    roles = training_roles(parts, protocol)
    if protocol == 'all_train_fixed_epochs' and (
            plan['candidate_epochs'] != [plan['epochs']] or
            plan['train_args'].get('val') is not False):
        raise ValueError('Final refit requires one fixed last epoch and no validation-based selection')
    if smoke:
        validation = roles['validation'][:plan['smoke_validation_images']]
        if protocol == 'all_train_fixed_epochs':
            n = plan['smoke_fit_images'] - len(validation)
            if n < 1:
                raise ValueError('Final-refit smoke requires FIT and overlapping validation images')
            train = sorted(parts['fit'][:n] + validation)
        else:
            train = roles['train'][:plan['smoke_fit_images']]
        roles = dict(train=train, validation=validation)
    test_rgb = {r['rgb_sha256'] for r in original['records']['icdar13']['test']}
    if test_rgb & {lookup[s]['rgb_sha256'] for ids in roles.values() for s in ids}:
        raise ValueError('TRAIN/TEST RGB overlap')
    audit = {}; validation_audit = {}
    for role, ids in roles.items():
        for sid in ids:
            record = lookup[sid]
            if record['split'] != 'train' or sha(record['image_path']) != record['image_sha256']:
                raise ValueError('Only checksum-verified TRAIN RGB can be exported')
            gt, _ = annotation(record)
            with Image.open(record['image_path']) as im:
                rgb = np.ascontiguousarray(im.convert('RGB'))
            ignored_pixels = 0
            if role == 'train':
                words = parse_icdar13_character_groups(Path(record['annotation_path']))
                words, _ = clip_instances_to_pixel_extent(words, tuple(record['image_wh']))
                mask = Image.new('L', tuple(record['image_wh']), 255); draw = ImageDraw.Draw(mask)
                for ignored, fill in [(True, 0), (False, 255)]:
                    for word in words:
                        if word.ignore == ignored:
                            draw.polygon(list(word.polygon_xy), fill=fill)
                valid = np.asarray(mask) > 0; ignored_pixels = int((~valid).sum())
                rgb = mask_training_ignore(rgb, valid, plan['training_ignore_fill'])
            image = root/'dataset/images'/role/(key(sid)+'.png')
            label = root/'dataset/labels'/role/(key(sid)+'.txt')
            image.parent.mkdir(parents=True, exist_ok=True); label.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(image)
            label.write_text(''.join('0 '+' '.join(format(v, '.17g') for v in normalized_box(
                g['bbox'], record['image_wh']))+'\n' for g in gt))
            entry = dict(role=role, image_sha256=sha(image), label_sha256=sha(label),
                original_image_sha256=record['image_sha256'], annotation_sha256=record['annotation_sha256'],
                care_words=len(gt), training_masked_pixels=ignored_pixels)
            if role == 'validation':
                validation_audit[sid] = entry
            if role == 'train' or sid not in audit:
                audit[sid] = entry
    dataset = root/'dataset/data.yaml'
    dataset.write_text(yaml.safe_dump(dict(path=str(root/'dataset'), train='images/train',
        val='images/validation', names={0: 'text'}, nc=1), sort_keys=False))
    seal(root/'dataset_manifest.json', dict(records=audit, validation_records=validation_audit,
        roles=roles, fit_protocol=protocol, validation_overlaps_training=bool(
            set(roles['train']) & set(roles['validation'])), test_used=False))
    return dataset, roles


def run(config_path, smoke, run_name=None):
    if config_path.resolve().parent != REPO/'configs/approximate':
        raise ValueError('Explicit approximate namespace required')
    c = read(config_path)
    if c['mode'] != 'approximate' or importlib.metadata.version('ultralytics') != c['ultralytics_version']:
        raise ValueError('Qualified approximate YOLO package/config required')
    for path, digest in c['pinned_inputs'].items():
        if sha(path) != digest:
            raise ValueError('Changed YOLO training input: '+path)
    if torch.cuda.mem_get_info()[0] < c['minimum_free_gpu_bytes']:
        raise RuntimeError('Insufficient free CUDA memory')
    root = Path(c['output_root'])/(run_name or ('smoke' if smoke else 'full'))
    root.mkdir(parents=True, exist_ok=False)
    source = {}
    for package in ('scripts', 'recognition', 'localize', 'segmentation', 'models',
                    'pipeline', 'scoring', 'caption', 'research'):
        for file in (REPO/package).rglob('*.py'):
            dest = root/'code_snapshot'/file.relative_to(REPO)
            dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(file, dest)
            source[str(file.relative_to(REPO))] = sha(dest)
    shutil.copytree(REPO/'configs', root/'code_snapshot/configs')
    dataset, roles = export_data(c, root, smoke)
    seal(root/'resolved_configuration.json', dict(config=c, config_sha256=sha(config_path),
        mode='approximate', claim_label='approximate reproduction; detector adaptation', smoke=smoke,
        result_class='smoke-test result' if smoke else 'TRAIN detector fitting',
        source_sha256=source, roles=roles, test_used=False,
        hardware=dict(gpu=torch.cuda.get_device_name(0), precision='float32', batch_size=c['train_args']['batch'],
            input_shape=[c['train_args']['batch'], 3, c['train_args']['imgsz'], c['train_args']['imgsz']]),
        packages={p: importlib.metadata.version(p) for p in ['torch', 'torchvision', 'ultralytics']}))
    all_arms = {}
    for arm, freeze in c['arms'].items():
        model = YOLO(c['checkpoint']).model.float()
        # The serialized upstream module can retain a CPU-bound training loss
        # object. Recreate that non-network object after CUDA placement on its
        # first loss call; every network tensor is still loaded strictly below.
        if hasattr(model, 'criterion'):
            delattr(model, 'criterion')
        initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        audits = []; initialized = {}
        def initialize(trainer):
            initialized.update(strict_initialization(trainer.model, initial))
            torch.cuda.reset_peak_memory_stats()
        def frozen_batch(trainer):
            for module in list(trainer.model.model)[:freeze]:
                module.eval()
        def epoch_end(trainer):
            state = trainer.model.state_dict()
            prefixes = tuple(f'model.{i}.' for i in range(freeze))
            frozen = [k for k in state if prefixes and k.startswith(prefixes)]
            if any(not torch.equal(state[k].detach().cpu(), initial[k]) for k in frozen):
                raise ValueError('Head-only detector changed frozen backbone tensors')
            audits.append(dict(epoch=trainer.epoch+1, frozen_tensors=len(frozen),
                frozen_exact=True, allocated=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.max_memory_reserved()))
        def save_epoch(trainer):
            epoch = trainer.epoch+1
            points = [1] if smoke else c['candidate_epochs']
            if epoch in points:
                dest = root/arm/'candidates'/f'epoch_{epoch:03d}.pt'
                dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(trainer.last, dest)
        args = dict(c['train_args'], model=c['checkpoint'], data=str(dataset), project=str(root),
                    name=arm, exist_ok=False, freeze=freeze, epochs=1 if smoke else c['epochs'])
        if smoke:
            args['close_mosaic'] = 0
        trainer = LocalDetectionTrainer(overrides=args)
        # Supply the complete preloaded module so setup_model does not use an
        # intersection/non-strict checkpoint transfer. Assert a full strict load.
        trainer.model = model
        for event, function in [('on_pretrain_routine_end', initialize),
            ('on_train_batch_start', frozen_batch), ('on_train_epoch_end', epoch_end),
            ('on_model_save', save_epoch)]:
            trainer.add_callback(event, function)
        try:
            trainer.train()
        except BaseException as exc:
            seal(root/(arm+'_failure.json'), dict(error=repr(exc), traceback=traceback.format_exc(),
                out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError),
                allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved()))
            raise
        points = {str(p.relative_to(root)): sha(p) for p in (root/arm/'candidates').glob('*.pt')}
        if len(points) != (1 if smoke else len(c['candidate_epochs'])):
            raise ValueError('Missing predeclared detector checkpoint candidate')
        selection = ('Predetermined final epoch; validation overlaps fitting and cannot select checkpoint'
            if c.get('fit_protocol', 'heldout_selection') == 'all_train_fixed_epochs'
            else 'No OCR checkpoint selected yet; validation OCR evaluation required')
        all_arms[arm] = dict(status='COMPLETE', initialization=initialized, epochs=audits,
            checkpoint_sha256=points, selected_by=selection,
            total_parameters=sum(p.numel() for p in trainer.model.parameters()), out_of_memory=False)
        seal(root/arm/'training_audit.json', all_arms[arm])
        del trainer, model; torch.cuda.empty_cache()
    seal(root/'completion.json', dict(status='COMPLETE', mode='approximate', smoke=smoke,
        claim_label='approximate reproduction; detector adaptation', arms=all_arms,
        test_used=False, actual_new_deepsolo_inferences=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--run-name')
    a = p.parse_args(); run(a.config, a.smoke, a.run_name)
