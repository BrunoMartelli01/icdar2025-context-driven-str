"""Fold-only auxiliary support fitting and paired-benefit context-router training."""
from __future__ import annotations

import copy
import gc
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from sklearn.model_selection import GroupKFold

from recognition.token_context_router import BenefitRouter, DenseSupport, benefit_loss, dense_features
from recognition.trainable_image_router import oracle
from recognition.f1_quality_study import aggregate
from scripts.run_yolo_f1_quality_study import read, sha, seal
from scripts.crop_clip_runtime import _cuda_peak
from scripts.token_context_data import key, arrays, annotations


def point(branches, ids, accepted):
    accepted = set(accepted)
    if not accepted <= set(ids):
        raise ValueError('Routing population changed')
    return {'images': len(ids), 'accepted_images': len(accepted), 'fallback_images': len(ids)-len(accepted),
        'accepted_ids': sorted(accepted), 'combined': aggregate(branches[s]['light' if s in accepted else 'deepsolo'] for s in ids),
        'accepted_words': aggregate(branches[s]['light'] for s in accepted),
        'same_accepted_if_deepsolo': aggregate(branches[s]['deepsolo'] for s in accepted)}


def ranked(scores, ids, fraction):
    return sorted((s for s in ids if scores[s] is not None), key=lambda s: (-scores[s], s))[:round(len(ids)*fraction)]


def targets(branches, ids):
    return np.array([[branches[s]['light']['tp']-branches[s]['deepsolo']['tp'],
        branches[s]['light']['detected_words']-branches[s]['deepsolo']['detected_words']] for s in ids], np.float32)


def annotation_targets(plan, root, sample, features):
    """TRAIN-only weak patch text support and local literal word compatibility."""
    gt = annotations(sample); w, h = sample['image_wh']; geometry = features['patch_geometry']
    labels = np.zeros(len(geometry), np.float32)
    for g in gt:
        a, b, c, d = g['bbox']
        labels[(geometry[:, 0]*w >= a)&(geometry[:, 0]*w < c)&(geometry[:, 1]*h >= b)&(geometry[:, 1]*h < d)] = 1
    with Image.open(sample['validity_path']) as im:
        validity = np.asarray(im.convert('L')) > 0
    xx = np.clip((geometry[:, 0]*w).astype(int), 0, w-1); yy = np.clip((geometry[:, 1]*h).astype(int), 0, h-1)
    mask = features['patch_valid']&validity[yy, xx]
    rows = read(root/'train/light'/(key(sample['sample_id'])+'.json'))['rows']; word = []
    def overlap(a, b):
        n = max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))
        return n/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-n)
    for r in rows:
        word.append(float(any(r['raw_text'].casefold() == g['text'].casefold() and
            overlap(r['bbox'], g['bbox']) >= plan['config']['dense_training']['auxiliary_iou'] for g in gt)) if r['raw_text'].strip() else -1.)
    return {'patch_labels': labels, 'patch_mask': mask, 'word_labels': np.array(word, np.float32)}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_model(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('Refusing to replace an immutable trained model: '+str(path))
    temp = path.with_suffix('.tmp'); torch.save(value, temp); temp.replace(path)


def fit_support(plan, root, features, labels, ids, name, seed):
    cfg = plan['config']; settings = cfg['dense_training']; path = root/'models'/(name+'__support.pt')
    if path.exists():
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['training_ids'] != ids or saved['plan_sha256'] != sha(root/'resolved_configuration.json'):
            raise ValueError('Dense support fitting scope changed')
        model = DenseSupport(cfg['architecture']['clip_dim'], settings['hidden_dim']); model.load_state_dict(saved['state_dict'], strict=True)
        return model.to(cfg['execution']['device']).eval(), saved
    seed_all(seed); device = cfg['execution']['device']; torch.cuda.reset_peak_memory_stats(device); started = time.monotonic()
    model = DenseSupport(cfg['architecture']['clip_dim'], settings['hidden_dim']).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    positive = sum(float(labels[s]['patch_labels'][labels[s]['patch_mask']].sum()) for s in ids)
    total = sum(int(labels[s]['patch_mask'].sum()) for s in ids)
    if positive == 0 or positive == total:
        raise ValueError('Dense support fitting needs both annotated-support and background patches')
    weight = torch.tensor((total-positive)/positive, device=device); updates = 0; history = []
    for epoch in range(settings['epochs']):
        order = list(ids); random.Random(seed+epoch).shuffle(order); losses = []
        for sid in order:
            a, label = features[sid], labels[sid]; mask = label['patch_mask']
            pp = torch.tensor(a['patches'][mask], device=device); target = torch.tensor(label['patch_labels'][mask], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = (F.binary_cross_entropy_with_logits(model.textness(pp).flatten(), target, pos_weight=weight)
                    if mask.any() else model.textness.weight.sum()*0)
            word = label['word_labels']; valid = word >= 0
            if valid.any():
                pooled = []
                for box in a['word_geometry'][valid, :4]:
                    xy = a['patch_geometry']; selected = a['patch_valid']&(xy[:, 0] >= box[0])&(xy[:, 0] <= box[2])&(xy[:, 1] >= box[1])&(xy[:, 1] <= box[3])
                    pooled.append(a['patches'][selected].mean(0) if selected.any() else np.zeros(cfg['architecture']['clip_dim'], np.float32))
                logits = model.paired_logits(torch.tensor(np.asarray(pooled), device=device), torch.tensor(a['word_text'][valid], device=device))
                loss = loss+settings['word_weight']*F.binary_cross_entropy_with_logits(logits, torch.tensor(word[valid], device=device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite supervised dense-support loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); updates += 1; losses.append(float(loss.detach()))
        history.append({'epoch': epoch+1, 'loss': float(np.mean(losses))})
    model.eval()
    saved = {'plan_sha256': sha(root/'resolved_configuration.json'), 'training_ids': ids, 'seed': seed,
        'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}, 'optimizer_steps': updates,
        'positive_patch_targets': positive, 'patch_targets': total, 'history': history,
        'supervision': 'TRAIN-only care-annotation bbox support with original validity; local exact-word compatibility, not one-to-one output correctness',
        'seconds': time.monotonic()-started, 'cuda_peak': _cuda_peak(device)}
    save_model(path, saved); return model, saved


def derive_dense(plan, root, features, support, name):
    result = {}; device = plan['config']['execution']['device']
    for sid, values in features.items():
        path = root/'derived_dense'/name/(key(sid)+'.json')
        model_path = root/'models'/(name+'__support.pt')
        if path.exists():
            v = read(path)
            if v['model_sha256'] != sha(model_path):
                raise ValueError('Derived dense model changed')
            result[sid] = np.array(v['values'], np.float32); continue
        vector, diagnostic = dense_features(values, support, plan['config']['transport'], device)
        seal(path, {'model_sha256': sha(model_path), 'values': vector.tolist(), 'diagnostic': diagnostic})
        result[sid] = vector
    return result


def extra_features(features, dense, arm, ids):
    values = []
    for sid in ids:
        a = features[sid]; parts = []
        if arm in ('likelihood', 'combined'):
            parts.append(a['likelihood'])
        if arm in ('dense', 'combined'):
            parts.append(dense[sid])
        values.append(np.concatenate(parts) if parts else np.empty(0, np.float32))
    return np.asarray(values, np.float32)


def collate(features, ids, extra, mean, scale, device):
    max_words = max(1, max(len(features[s]['word_text']) for s in ids)); patch_count = len(features[ids[0]]['patches'])
    sample = features[ids[0]]; n = len(ids)
    fields = {'patches': np.stack([features[s]['patches'] for s in ids]),
        'patch_geometry': np.stack([features[s]['patch_geometry'] for s in ids]),
        'patch_padding': np.stack([~features[s]['patch_valid'] for s in ids]),
        'queries': np.stack([features[s]['queries'] for s in ids]),
        'word_padding': np.ones((n, max_words), bool)}
    for name in ('word_text', 'word_image', 'word_geometry'):
        fields[name] = np.zeros((n, max_words, sample[name].shape[1]), np.float32)
    for i, sid in enumerate(ids):
        words = len(features[sid]['word_text']); fields['word_padding'][i, :words] = False
        for name in ('word_text', 'word_image', 'word_geometry'):
            fields[name][i, :words] = features[sid][name]
    all_scalars = np.concatenate([np.stack([features[s]['scalars'] for s in ids]), extra], axis=1)
    normalized = ((all_scalars-mean)/scale).astype(np.float32); d = len(sample['scalars'])
    fields['scalars'], fields['extra'] = normalized[:, :d], normalized[:, d:]
    return {name: torch.as_tensor(value, device=device) for name, value in fields.items()}


def score_model(model, preprocessing, features, dense, ids, cfg):
    scores = {}; predictions = {}; device = cfg['execution']['device']; model.eval(); arm = preprocessing['arm']
    with torch.inference_mode():
        for start in range(0, len(ids), cfg['training']['batch_size']):
            subset = ids[start:start+cfg['training']['batch_size']]
            extra = extra_features(features, dense, arm, subset)
            batch = collate(features, subset, extra, np.array(preprocessing['mean'], np.float32), np.array(preprocessing['scale'], np.float32), device)
            p, _ = model(batch); p = p.cpu().numpy()*preprocessing['target_scale']+np.array(preprocessing['target_mean'])
            if not np.isfinite(p).all():
                raise RuntimeError('Nonfinite branch-benefit predictions')
            for sid, value in zip(subset, p):
                predictions[sid] = value.tolist()
                scores[sid] = float(2*value[0]-preprocessing['reference_f1']*value[1]) if features[sid]['eligible'] else None
    return scores, predictions


def fit_router(plan, root, features, dense, labels, branches, ids, arm, name, seed, epochs, predict_ids):
    cfg = plan['config']; device = cfg['execution']['device']; settings = cfg['training']; seed_all(seed)
    extra = extra_features(features, dense, arm, ids); scalar = np.stack([features[s]['scalars'] for s in ids])
    x = np.concatenate([scalar, extra], 1); mean, scale = x.mean(0), x.std(0); scale[scale == 0] = 1.
    y = targets(branches, ids); target_mean = y.mean(0); target_scale = max(float(np.sqrt(np.mean((y-target_mean)**2))), settings['minimum_target_scale'])
    reference = aggregate(branches[s]['deepsolo'] for s in ids)['f1']
    preprocessing = {'arm': arm, 'training_ids': ids, 'mean': mean.tolist(), 'scale': scale.tolist(),
        'target_mean': target_mean.tolist(), 'target_scale': target_scale, 'reference_f1': reference, 'seed': seed,
        'architecture': cfg['architecture'], 'extra_dim': extra.shape[1]}
    model = BenefitRouter(cfg['architecture'], scalar.shape[1], extra.shape[1], arm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    checkpoints = {}; history = []; updates = 0; start = time.monotonic(); torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, epochs+1):
        model.train(); order = list(ids); random.Random(seed+epoch).shuffle(order); losses = []
        for offset in range(0, len(order), settings['batch_size']):
            group = order[offset:offset+settings['batch_size']]
            batch = collate(features, group, extra_features(features, dense, arm, group), mean, scale, device)
            target = torch.tensor((targets(branches, group)-target_mean)/target_scale, device=device)
            optimizer.zero_grad(set_to_none=True); prediction, word_logits = model(batch)
            loss, details = benefit_loss(prediction, target, reference, settings)
            if word_logits is not None:
                word_target = torch.full_like(word_logits, -1)
                for i, sid in enumerate(group):
                    word_target[i, :len(labels[sid]['word_labels'])] = torch.tensor(labels[sid]['word_labels'], device=device)
                mask = word_target >= 0
                if mask.any():
                    loss = loss+settings['auxiliary_word_weight']*F.binary_cross_entropy_with_logits(word_logits[mask], word_target[mask])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite paired-benefit training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), settings['gradient_clip'], error_if_nonfinite=True)
            optimizer.step(); updates += 1; losses.append(float(loss.detach()))
        history.append({'epoch': epoch, 'loss': float(np.mean(losses))})
        if epoch in settings['checkpoints'] or epoch == epochs:
            path = root/'models'/(name+'__epoch'+str(epoch)+'.pt')
            saved = {'plan_sha256': sha(root/'resolved_configuration.json'), 'preprocessing': preprocessing,
                'state_dict': {n: v.detach().cpu() for n, v in model.state_dict().items()}, 'epoch': epoch,
                'optimizer_steps': updates, 'history': history.copy(), 'parameters': sum(p.numel() for p in model.parameters()),
                'precision': 'float32', 'device': device, 'batch_size': settings['batch_size'],
                'maximum_patch_tokens': int(batch['patches'].shape[1]), 'last_word_shape': list(batch['word_text'].shape),
                'cuda_peak': _cuda_peak(device), 'seconds': time.monotonic()-start}
            if not path.exists():
                save_model(path, saved)
            else:
                existing = torch.load(path, map_location='cpu', weights_only=True)
                if existing['preprocessing'] != preprocessing or any(not torch.equal(existing['state_dict'][k], v) for k, v in saved['state_dict'].items()):
                    raise ValueError('Retraining changed an immutable checkpoint')
            scores, predictions = score_model(model, preprocessing, features, dense, predict_ids, cfg)
            checkpoints[str(epoch)] = {'scores': scores, 'count_predictions': predictions, 'model_path': str(path.relative_to(root)), 'model_sha256': sha(path)}
    del model, optimizer; gc.collect(); torch.cuda.empty_cache()
    return checkpoints


def load_model(plan, root, path):
    saved = torch.load(root/path, map_location='cpu', weights_only=True); p = saved['preprocessing']
    if saved['plan_sha256'] != sha(root/'resolved_configuration.json'):
        raise ValueError('Model belongs to another resolved experiment')
    model = BenefitRouter(p['architecture'], plan['config']['architecture']['scalar_dim'], p['extra_dim'], p['arm'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return model.to(plan['config']['execution']['device']).eval(), p


def train_and_freeze(plan, root, branches):
    if (root/'frozen_choices.json').exists():
        value = read(root/'frozen_choices.json')
        if value['plan_sha256'] != sha(root/'resolved_configuration.json'):
            raise ValueError('Frozen choices identity changed')
        return
    if list((root/'test/features').glob('*.json')):
        raise ValueError('Cannot fit after fresh TEST context vectors')
    cfg = plan['config']; ids = sorted(branches); samples = {s['sample_id']: s for s in plan['samples']['train']}
    features = {}; labels = {}
    for sid in ids:
        features[sid], meta = arrays(root, 'train', sid); features[sid]['eligible'] = meta['eligible']
        labels[sid] = annotation_targets(plan, root, samples[sid], features[sid])
    cv = {}; model_index = {}; fold_trace = []
    for population, members, arms in [('full', ids, cfg['arms']), ('legacy80', sorted(plan['legacy_fit_ids']), ['scalar'])]:
        folds = list(GroupKFold(n_splits=cfg['cv_folds']).split(members, groups=[samples[s]['rgb_sha256'] for s in members]))
        cv[population] = {arm: {str(epoch): {} for epoch in cfg['training']['checkpoints']} for arm in arms}
        for fold, (a, b) in enumerate(folds):
            fit, held = [members[i] for i in a], [members[i] for i in b]; seed = cfg['execution']['seed']+fold
            if set(fit)&set(held) or {samples[s]['rgb_sha256'] for s in fit}&{samples[s]['rgb_sha256'] for s in held}:
                raise ValueError('Cross-validation pixel-group leakage')
            if population == 'full':
                support, receipt = fit_support(plan, root, features, labels, fit, population+str(fold), seed)
                dense = derive_dense(plan, root, {s: features[s] for s in members}, support, population+str(fold))
                support = None; gc.collect(); torch.cuda.empty_cache()
            else:
                dense = {}
            fold_trace.append({'population': population, 'fold': fold, 'training_ids': fit, 'held_ids': held,
                'training_rgb_sha256': [samples[s]['rgb_sha256'] for s in fit], 'held_rgb_sha256': [samples[s]['rgb_sha256'] for s in held]})
            for index, arm in enumerate(arms):
                name = population+str(fold)+'__'+arm
                found = fit_router(plan, root, features, dense, labels, branches, fit, arm, name,
                    seed+cfg['training']['seed_stride']*index, cfg['training']['epochs'], held)
                for epoch, values in found.items():
                    cv[population][arm][epoch].update(values['scores']); model_index[name+'__'+epoch] = {k: values[k] for k in ('model_path', 'model_sha256')}
                print(f'[CV {population} fold{fold}] {arm}', flush=True)
    frozen = {'plan_sha256': sha(root/'resolved_configuration.json'), 'test_used_for_selection': False,
        'test_features_present_at_freeze': 0, 'arms': {}, 'fold_trace': fold_trace, 'cv_models': model_index}
    candidates = []
    for population, arm_scores in cv.items():
        members = ids if population == 'full' else sorted(plan['legacy_fit_ids'])
        for index, (arm, epochs) in enumerate(arm_scores.items()):
            points = {epoch: point(branches, members, ranked(scores, members, cfg['primary_bypass_fraction'])) for epoch, scores in epochs.items()}
            selected = max(points, key=lambda e: (points[e]['combined']['f1'], -int(e)))
            name = arm if population == 'full' else 'scalar_legacy80'
            frozen['arms'][name] = {'arm': arm, 'population': population, 'epoch': int(selected), 'oof_points': points}
            if population == 'full':
                candidates.append((points[selected]['combined']['f1'], -index, name))
    frozen['selected_arm'] = max(candidates)[2]
    support, _ = fit_support(plan, root, features, labels, ids, 'final', cfg['execution']['seed'])
    dense = derive_dense(plan, root, features, support, 'final'); support = None; gc.collect(); torch.cuda.empty_cache()
    for index, (name, choice) in enumerate(frozen['arms'].items()):
        members = ids if choice['population'] == 'full' else sorted(plan['legacy_fit_ids'])
        found = fit_router(plan, root, features, dense, labels, branches, members, choice['arm'], 'final__'+name,
            cfg['execution']['seed']+cfg['training']['seed_stride']*index, choice['epoch'], ids)[str(choice['epoch'])]
        choice.update({k: found[k] for k in ('model_path', 'model_sha256')})
        seal(root/'predictions'/('train__'+name+'.json'), found)
    seal(root/'cv_predictions.json', cv); frozen['cv_predictions_sha256'] = sha(root/'cv_predictions.json')
    frozen['support_model_sha256'] = sha(root/'models/final__support.pt')
    seal(root/'frozen_choices.json', frozen)
    print('[frozen] full-TRAIN models/epochs/arm selected before TEST context extraction', flush=True)


def evaluate(plan, root):
    cfg = plan['config']; frozen = read(root/'frozen_choices.json'); output = {'status': 'SUCCESS',
        'reproduction_mode': 'approximate', 'claim_label': 'approximate reproduction', 'result_class': plan['result_class'],
        'table_5_claim': False, 'plan_sha256': sha(root/'resolved_configuration.json'), 'frozen_choices_sha256': sha(root/'frozen_choices.json'),
        'selected_arm': frozen['selected_arm'], 'primary_arm': cfg['primary_arm'], 'base_word_outputs_changed': False, 'splits': {}}
    for split in ('train', 'test'):
        branches = read(root/split/'branch_metrics.json')['branches']; ids = sorted(branches); features = {}
        for sid in ids:
            features[sid], meta = arrays(root, split, sid); features[sid]['eligible'] = meta['eligible']
        support = DenseSupport(cfg['architecture']['clip_dim'], cfg['dense_training']['hidden_dim'])
        saved = torch.load(root/'models/final__support.pt', map_location='cpu', weights_only=True)
        if saved['training_ids'] != sorted(s['sample_id'] for s in plan['samples']['train']):
            raise ValueError('Final support fitting scope changed')
        support.load_state_dict(saved['state_dict'], strict=True); support.to(cfg['execution']['device']).eval()
        dense = derive_dense(plan, root, features, support, 'final') if split == 'train' else derive_test_dense(plan, root, features, support)
        support = None; gc.collect(); torch.cuda.empty_cache(); arms = {}
        eligible = [s for s in ids if features[s]['eligible']]
        oracles = {str(q): oracle({s: branches[s] for s in ids}, ids, eligible, round(len(ids)*q)) for q in cfg['bypass_fractions']}
        for name, choice in frozen['arms'].items():
            if sha(root/choice['model_path']) != choice['model_sha256']:
                raise ValueError('Final router model changed')
            model, preprocessing = load_model(plan, root, choice['model_path'])
            scores, predictions = score_model(model, preprocessing, features, dense, ids, cfg)
            model = None; gc.collect(); torch.cuda.empty_cache()
            seal(root/'predictions'/(split+'__'+name+'__evaluated.json'), {'scores': scores, 'count_predictions': predictions})
            arms[name] = {}
            for fraction in cfg['bypass_fractions']:
                p = point(branches, ids, ranked(scores, ids, fraction)); f = p['combined']['f1']
                p.update(delta_vs_all_deepsolo_pp=100*(f-aggregate(branches[s]['deepsolo'] for s in ids)['f1']),
                    oracle_regret_pp=100*(oracles[str(fraction)]['f1']-f))
                arms[name][str(fraction)] = p
        output['splits'][split] = {'images': len(ids), 'light_only': aggregate(branches[s]['light'] for s in ids),
            'deepsolo_only': aggregate(branches[s]['deepsolo'] for s in ids), 'oracles': oracles, 'arms': arms,
            'selected': arms[frozen['selected_arm']], 'primary': arms[cfg['primary_arm']]}
    seal(root/'results.json', output)
    print(json.dumps({'status': 'SUCCESS', 'root': str(root), 'results_sha256': sha(root/'results.json')}), flush=True)


def derive_test_dense(plan, root, features, support):
    values = {}
    for sid, data in features.items():
        vector, diag = dense_features(data, support, plan['config']['transport'], plan['config']['execution']['device'])
        seal(root/'derived_dense/test'/(key(sid)+'.json'), {'model_sha256': sha(root/'models/final__support.pt'),
            'values': vector.tolist(), 'diagnostic': diag, 'ground_truth_used': False})
        values[sid] = vector
    return values
