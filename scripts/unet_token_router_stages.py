"""Qualified frozen expert stages and matched token-head fitting/evaluation."""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from sklearn.model_selection import GroupKFold

from scripts.run_yolo_f1_quality_study import read, sha, seal
from scripts.crop_clip_runtime import array_sha, _cuda_peak
from scripts.contextual_lm_verifier_runtime import state_digest
from scripts.token_context_data import rgb, key, annotations, write_arrays, arrays
from scripts.token_context_training import fit_router, load_model, score_model, point, ranked
from recognition.token_context_router import base_features, word_geometry
from recognition.unet_token_routing import unet_proposals, selected_rows, weak_word_labels, fit_thresholds, apply_threshold, fallback_counts
from recognition.clip_router_evaluation import native_mask, deepsolo_predictions
from recognition.deepsolo_word_f1 import recognition_confidence, apply_policy
from recognition.f1_quality_study import aggregate
from recognition.trainable_image_router import oracle
from scripts.replay_router_improved_deepsolo import metric_counts, independent_counts, independent_route, conditional_interval


def binding(root, sample):
    return dict(plan_sha256=sha(root/'resolved_configuration.json'), sample_id=sample['sample_id'], image_sha256=sample['image_sha256'])


def validate_receipt(root, path):
    if not path.exists():
        return False
    value = read(path)
    if value['plan_sha256'] != sha(root/'resolved_configuration.json'):
        raise ValueError('Stage receipt plan changed')
    for name, digest in value['files_sha256'].items():
        if sha(root/name) != digest:
            raise ValueError('Completed stage output changed')
    return True


def light(plan, root, split):
    receipt = root/split/'unet_receipt.json'
    if validate_receipt(root, receipt):
        return
    from models.adaspot_gap import AdaSpotGapAGUNet
    from recognition.parseq import ParseqApproxRecognizer, ParseqApproxConfig, PARSEQ_ARCHIVE_SOURCE
    from segmentation.word_slot_data import image_tensor
    cfg, paths = plan['config'], plan['paths']; settings = cfg['unet']; start = time.monotonic()
    # The exact hash-qualified local checkpoint includes optimizer/NumPy state.
    # Trust is restricted to the pinned local bytes, never an arbitrary pickle.
    saved = torch.load(paths['unet_checkpoint'], map_location='cpu', weights_only=False)
    if saved['run_contract'] != read(paths['unet_contract']):
        raise ValueError('Qualified U-Net contract changed')
    model = AdaSpotGapAGUNet('control', skip_floor=settings['skip_floor'], initial_gap_bias=settings['initial_gap_bias'])
    model.load_state_dict(saved['model_state_dict'], strict=True); del saved
    model.requires_grad_(False).eval().to(cfg['execution']['device'])
    recognizer = ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(paths['parseq_source']),
        checkpoint=Path(paths['parseq_checkpoint']), runtime_source_archive=Path(paths['parseq_archive']),
        runtime_source_type=PARSEQ_ARCHIVE_SOURCE, device='cpu'))
    recognizer.model.requires_grad_(False).eval(); before, rb = state_digest(model), state_digest(recognizer.model)
    torch.cuda.reset_peak_memory_stats(); files = {}; count = 0
    try:
        for i, sample in enumerate(plan['samples'][split], 1):
            p = root/split/'unet'/(key(sample['sample_id'])+'.json')
            if p.exists():
                cached = read(p)
                if cached['contract'] != binding(root, sample):
                    raise ValueError('U-Net cached raster identity changed')
                maps = root/split/'unet_maps'/(key(sample['sample_id'])+'.npz')
                if sha(maps) != cached['maps_sha256']:
                    raise ValueError('Cached U-Net probability/instance maps changed')
                files[str(p.relative_to(root))] = sha(p)
                files[str(maps.relative_to(root))] = sha(maps)
                continue
            image = rgb(sample)
            with torch.inference_mode():
                probability = model(image_tensor(Image.fromarray(image)).to(cfg['execution']['device']))[0].sigmoid().cpu().numpy()
            proposals, labels, audit = unet_proposals(probability, sample['image_wh'], settings)
            candidates = []
            for offset in range(0, len(proposals), cfg['execution']['parseq_batch_size']):
                group = proposals[offset:offset+cfg['execution']['parseq_batch_size']]
                crops = [np.ascontiguousarray(image[r['bbox'][1]:r['bbox'][3], r['bbox'][0]:r['bbox'][2]]) for r in group]
                decoded = recognizer.recognize_candidate_batch(crops)
                if len(decoded) != len(group):
                    raise RuntimeError('Incomplete real PARSeq response')
                for box, crop, value in zip(group, crops, decoded):
                    candidates.append(dict(box, **value.to_dict(), candidate_id=sample['sample_id']+':unet:'+str(box['instance_id']),
                                           crop_rgb_sha256=array_sha(crop), crop_shape=list(crop.shape)))
            selected = selected_rows(candidates, settings)
            maps = root/split/'unet_maps'/(key(sample['sample_id'])+'.npz')
            write_arrays(maps, dict(probabilities=probability, labels=labels))
            seal(p, dict(contract=binding(root, sample), candidates=candidates, rows=selected, geometry_audit=audit,
                maps_sha256=sha(maps), ground_truth_used=False, segmentation_input_shape=[1,3,512,512],
                recognition_device='cpu', recognition_shape=[cfg['execution']['parseq_batch_size'],3,32,128]))
            files[str(p.relative_to(root))] = sha(p); files[str(maps.relative_to(root))] = sha(maps); count += len(selected)
            if i % cfg['execution']['log_every'] == 0 or i == len(plan['samples'][split]):
                print(f'[U-Net/PARSeq {split}] {i}/{len(plan["samples"][split])}', flush=True)
        after, ra = state_digest(model), state_digest(recognizer.model)
        if after != before or ra != rb:
            raise RuntimeError('Frozen U-Net or PARSeq weights changed')
        seal(receipt, dict(plan_sha256=sha(root/'resolved_configuration.json'), files_sha256=files,
            images=len(plan['samples'][split]), seconds=time.monotonic()-start, model='qualified Text+Kernel ResNet50 Attention U-Net',
            unet_state_before=before, unet_state_after=after, parseq_state_before=rb, parseq_state_after=ra,
            strict_checkpoint_loading=True, precision='float32', device=cfg['execution']['device'], batch_size=1, cuda_peak=_cuda_peak('cuda:0')))
    finally:
        recognizer.release(); del model, recognizer; gc.collect(); torch.cuda.empty_cache()


def fallback(plan, root, split):
    receipt = root/split/'fallback_receipt.json'
    if validate_receipt(root, receipt):
        return
    cfg, paths = plan['config'], plan['paths']; source = Path(paths['improved_root'])
    processed_sha = sha(source/'processed_predictions.json')
    known = {r['sample_id']: r for r in read(source/'processed_predictions.json')['splits'][split]}
    missing = [s for s in plan['samples'][split] if s['sample_id'] not in known]
    if missing:
        if split != 'train':
            raise ValueError('Expected a complete qualified TEST fallback cache')
        population = root/'missing_fallback_population.json'
        seal(population, dict(reproduction_mode='approximate', images=dict(train=missing, test=[])))
        export_cfg = read(paths['deepsolo_profile']); export_cfg['source_plan_sha256'] = sha(population)
        export_cfg['resolutions'] = {'native': export_cfg['resolutions']['native']}
        config_path = root/'configs/approximate/missing_fallback_export.json'; seal(config_path, export_cfg)
        output = root/'fallback_export'
        command = [paths['deepsolo_python'], '-u', '-m', 'scripts.run_deepsolo_word_f1_study', '--mode', 'approximate',
            '--stage', 'export', '--config', str(config_path), '--source-plan', str(population), '--runtime', paths['deepsolo_runtime'],
            '--source-code-root', paths['deepsolo_source'], '--data-root', paths['data_root'], '--output-root', str(output), '--splits', 'train']
        subprocess.run(command, cwd=root/'code_snapshot', env={**os.environ, 'PYTHONPATH':str(root/'code_snapshot')}, check=True)
    files = {}; reused = generated = 0
    for sample in plan['samples'][split]:
        sid = sample['sample_id']
        if sid in known:
            row = known[sid]
            if row['image_sha256'] != sample['image_sha256'] or row['ground_truth'] != annotations(sample):
                raise ValueError('Qualified fallback sample changed')
            predictions = apply_policy(row['native'], cfg['fallback_policy']); provenance = dict(reused_processed_sha256=processed_sha)
            reused += 1
        else:
            p = root/'fallback_export/native'/split/(key(sid)+'.json'); raw = read(p)
            payload = {k:v for k,v in raw.items() if k != 'payload_sha256'}
            if (not raw['response']['success'] or raw['contract']['image_sha256'] != sample['image_sha256']
                    or raw['postprocess_resize'] != read(paths['deepsolo_profile'])['resolutions']['native']
                    or raw['payload_sha256'] != hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()):
                raise ValueError('New qualified fallback shard failed')
            with Image.open(sample['validity_path']) as im:
                validity = native_mask(im.convert('L'), cfg['metric']['native_ignore_size'])
            predictions = deepsolo_predictions(raw['response'], sample['image_wh'], sample['orientation'], validity, cfg['fallback_policy']['score_threshold'])
            detection = {d['source_index']:d for d in raw['response']['detections']}
            for pred in predictions:
                pred['recognition_confidence'] = recognition_confidence(detection[pred['instance_id']]['raw']['recognition_class_probabilities'])
            predictions = apply_policy(predictions, cfg['fallback_policy']); provenance = dict(raw_sha256=sha(p)); generated += 1
        p = root/split/'fallback'/(key(sid)+'.json')
        seal(p, dict(contract=binding(root, sample), predictions=predictions, provenance=provenance))
        files[str(p.relative_to(root))] = sha(p)
    seal(receipt, dict(plan_sha256=sha(root/'resolved_configuration.json'), files_sha256=files,
        images=len(plan['samples'][split]), reused=reused, new_model_inferences=generated,
        deployment_counts='Reported separately; this stage populates TRAIN supervision and cached evaluation'))


def light_rows(plan, root, split, model, sid):
    p = root/split/'unet'/(key(sid)+'.json') if model == 'unet' else Path(plan['paths']['token_root'])/split/'light'/(key(sid)+'.json')
    return read(p)['rows']


def light_predictions(rows, sample, size):
    with Image.open(sample['validity_path']) as im:
        validity = native_mask(im.convert('L'), size)
    predictions = []
    for row in rows:
        a,b,c,d = row['bbox']; mask = Image.new('L', tuple(sample['image_wh']), 0)
        ImageDraw.Draw(mask).rectangle([a,b,c-1,d-1], fill=255); footprint = native_mask(mask, size)
        predictions.append(dict(text=row['raw_text'], bbox=row['bbox'], ignored=bool(footprint.any() and not np.any(footprint&validity))))
    return predictions


def branches(plan, root, split):
    path = root/split/'branches.json'
    if path.exists():
        return
    result = {name:{} for name in plan['config']['backbones']}; records = {}
    for sample in plan['samples'][split]:
        sid = sample['sample_id']; gt = annotations(sample); heavy = read(root/split/'fallback'/(key(sid)+'.json'))['predictions']
        heavy_counts = metric_counts(heavy, gt, 'casefold_alnum'); record = dict(ground_truth=gt, deepsolo=heavy, light={})
        for name in result:
            rows = light_rows(plan, root, split, name, sid); predictions = light_predictions(rows, sample, plan['config']['metric']['native_ignore_size'])
            counts = metric_counts(predictions, gt, 'casefold_alnum')
            if counts != independent_counts(predictions, gt, 'casefold_alnum'):
                raise ValueError('Independent light word-count replay failed')
            result[name][sid] = dict(light=counts, deepsolo=heavy_counts); record['light'][name] = predictions
        records[sid] = record
    seal(path, dict(branches=result, records=records, metric=plan['config']['metric'], plan_sha256=sha(root/'resolved_configuration.json')))


def features(plan, root, split):
    receipt = root/split/'features_receipt.json'
    if validate_receipt(root, receipt):
        return
    if split == 'test' and not (root/'frozen_choices.json').exists():
        raise ValueError('Freeze TRAIN choices before fresh TEST word features')
    from recognition.token_context_runtime import FrozenDenseCLIP
    cfg, paths = plan['config'], plan['paths']; model = FrozenDenseCLIP(paths['clip_snapshot'], cfg['clip'], cfg['execution']['device'])
    files = {}; equivalence = None
    try:
        for index, sample in enumerate(plan['samples'][split], 1):
            sid = sample['sample_id']; p = root/split/'unet_features'/(key(sid)+'.npz')
            rows = light_rows(plan, root, split, 'unet', sid); image = rgb(sample)
            source = Path(paths['token_root'])/split/'features'/(key(sid)+'.npz')
            metadata = p.with_suffix('.json')
            expected = dict(contract=binding(root, sample),
                rows_sha256=sha(root/split/'unet'/(key(sid)+'.json')),
                source_scene_sha256=sha(source),
                frozen_choices_sha256=sha(root/'frozen_choices.json') if split=='test' else None)
            if p.exists():
                if not metadata.exists():
                    raise ValueError('Unbound partial U-Net feature archive: '+str(p))
                cached = read(metadata)
                if cached != dict(expected, arrays_sha256=sha(p)):
                    raise ValueError('Partial U-Net feature identity changed')
            crops = [np.ascontiguousarray(image[r['bbox'][1]:r['bbox'][3],r['bbox'][0]:r['bbox'][2]]) for r in rows]
            if any(array_sha(crop) != row['crop_rgb_sha256'] for crop,row in zip(crops,rows)):
                raise ValueError('Original U-Net RGB crop changed')
            if equivalence is None:
                actual,_ = model.infer(image, rows)
                with np.load(source, allow_pickle=False) as a:
                    errors = {k:float(np.max(np.abs(actual[k].astype(float)-a[k].astype(float)))) for k in ['patches','patch_geometry','patch_mass']}
                    if not np.array_equal(actual['patch_valid'], a['patch_valid']) or any(v > cfg['clip']['native_tolerance'] for v in errors.values()):
                        raise ValueError('Reused image-only scene tokens differ')
                equivalence = dict(status='PASS', sample_id=sid, maximum_errors=errors)
                text, images = actual['word_text'], actual['word_image']
            elif not p.exists():
                text, images = model.text([r['raw_text'] for r in rows]), model.images(crops)
            if not p.exists():
                write_arrays(p, dict(word_text=text, word_image=images, word_geometry=word_geometry(rows, sample['image_wh']),
                    scalars=base_features(rows, sample['image_wh'])))
                seal(metadata, dict(expected, arrays_sha256=sha(p)))
            files[str(p.relative_to(root))] = sha(p)
            files[str(metadata.relative_to(root))] = sha(metadata)
            if index % cfg['execution']['log_every'] == 0:
                print(f'[U-Net word CLIP {split}] {index}/{len(plan["samples"][split])}', flush=True)
        if equivalence is None:
            raise ValueError('No image-only scene equivalence qualification was performed')
        seal(receipt, dict(plan_sha256=sha(root/'resolved_configuration.json'), files_sha256=files, model=model.finish(),
            reused_global_scene_qualification=equivalence, fresh_TEST_bound_to=sha(root/'frozen_choices.json') if split=='test' else None))
    finally:
        model.close()


def load_features(plan, root, split):
    result = {name:{} for name in plan['config']['backbones']}; source = Path(plan['paths']['token_root'])
    for sample in plan['samples'][split]:
        sid = sample['sample_id']; original, metadata = arrays(source, split, sid)
        original['eligible'] = metadata['eligible']
        result['yolo'][sid] = original
        value = dict(original)
        with np.load(root/split/'unet_features'/(key(sid)+'.npz'), allow_pickle=False) as a:
            value.update({k:a[k] for k in a.files})
        rows = light_rows(plan, root, split, 'unet', sid)
        value['eligible'] = bool(rows) and all(r['raw_text'].strip() for r in rows)
        result['unet'][sid] = value
    return result


def train(plan, root):
    if (root/'frozen_choices.json').exists():
        if read(root/'frozen_choices.json')['plan_sha256'] != sha(root/'resolved_configuration.json'):
            raise ValueError('Frozen TRAIN choice identity changed')
        return
    test_files = list((root/'test/unet_features').glob('*.npz'))
    if test_files:
        raise ValueError('Cannot fit or select heads after fresh TEST word features')
    cfg = plan['config']; source = read(root/'train/branches.json')['branches']; features = load_features(plan, root, 'train')
    samples = {s['sample_id']:s for s in plan['samples']['train']}; ids = sorted(samples); groups = [samples[s]['rgb_sha256'] for s in ids]
    folds = list(GroupKFold(n_splits=cfg['cv_folds']).split(ids, groups=groups)); choices = {}; trace = []; index = {}
    for model in cfg['backbones']:
        labels = {s:dict(word_labels=weak_word_labels(light_rows(plan,root,'train',model,s), annotations(samples[s]), cfg['labels'])) for s in ids}
        oof = {str(e):{} for e in cfg['training']['checkpoints']}
        for fold,(fit_i, held_i) in enumerate(folds):
            fit, held = [ids[i] for i in fit_i], [ids[i] for i in held_i]
            if set(samples[s]['rgb_sha256'] for s in fit)&set(samples[s]['rgb_sha256'] for s in held):
                raise ValueError('Fold RGB leakage')
            receipt = root/'training'/(model+'__fold'+str(fold)+'.json')
            if receipt.exists():
                value = read(receipt)
                if value['fit_ids'] != fit or value['held_ids'] != held:
                    raise ValueError('Fitting scope changed on resume')
                found = value['checkpoints']
                for v in found.values():
                    if sha(root/v['model_path']) != v['model_sha256']:
                        raise ValueError('Completed model checkpoint changed')
            else:
                found = fit_router(plan, root, features[model], {}, labels, source[model], fit, 'token_context',
                    model+'__fold'+str(fold), cfg['execution']['seed']+fold, cfg['training']['epochs'], held)
                seal(receipt, dict(fit_ids=fit, held_ids=held, checkpoints=found))
            for epoch, v in found.items():
                oof[epoch].update(v['scores']); index[model+'__fold'+str(fold)+'__'+epoch] = dict(model_path=v['model_path'], model_sha256=v['model_sha256'])
            trace.append(dict(backbone=model, fold=fold, fitting_ids=fit, held_ids=held))
            print(f'[router fit {model}] fold {fold+1}/{cfg["cv_folds"]}', flush=True)
        if any(set(v) != set(ids) for v in oof.values()):
            raise ValueError('Incomplete OOF scores')
        points = {e:point(source[model], ids, ranked(v, ids, cfg['primary_bypass_fraction'])) for e,v in oof.items()}
        epoch = max(points, key=lambda e:(points[e]['combined']['f1'], -int(e)))
        selected = oof[epoch]
        final = fit_router(plan, root, features[model], {}, labels, source[model], ids, 'token_context',
            model+'__final', cfg['execution']['seed'], int(epoch), ids)[epoch]
        choices[model] = dict(epoch=int(epoch), oof_points=points, oof_scores=selected,
            model_path=final['model_path'], model_sha256=final['model_sha256'],
            scalar_thresholds=fit_thresholds(selected, cfg['train_acceptance_targets']), final_train=final)
    seal(root/'frozen_choices.json', dict(plan_sha256=sha(root/'resolved_configuration.json'), choices=choices,
        fold_trace=trace, cv_models=index, test_used_for_selection=False, test_new_feature_files_at_freeze=len(test_files)))
    print('[frozen] matched U-Net and YOLO heads and TRAIN thresholds', flush=True)


def evaluate(plan, root):
    if (root/'results.json').exists():
        return
    cfg = plan['config']; frozen = read(root/'frozen_choices.json'); output = {}
    for split in ('train', 'test'):
        data = read(root/split/'branches.json')['branches']; ids = sorted(data['unet']); output[split] = {}
        features = load_features(plan, root, split) if split == 'test' else None
        ss = {s['sample_id']:s for s in plan['samples'][split]}; groups = [ss[s]['rgb_sha256'] for s in ids]
        unique = sorted(set(groups)); mapping = np.array([unique.index(g) for g in groups]); rng = np.random.default_rng(cfg['execution']['seed'])
        weights = np.array([np.bincount(rng.integers(0,len(unique),len(unique)),minlength=len(unique))[mapping]
            for _ in range(cfg['bootstrap_replicates'])], float)
        for model in cfg['backbones']:
            choice = frozen['choices'][model]
            if split == 'train':
                scores = choice['oof_scores']
            else:
                if sha(root/choice['model_path']) != choice['model_sha256']:
                    raise ValueError('Frozen learned checkpoint changed')
                network, preprocessing = load_model(plan, root, choice['model_path'])
                scores, counts = score_model(network, preprocessing, features[model], {}, ids, cfg)
                del network; gc.collect(); torch.cuda.empty_cache()
                seal(root/'predictions'/(model+'__test.json'), dict(scores=scores, count_predictions=counts))
            eligible = [s for s in ids if scores[s] is not None]; cache = {}
            def summarize(accepted):
                value = point(data[model],ids,accepted); n = len(accepted)
                if n not in cache:
                    cache[n] = oracle(data[model],ids,eligible,n)
                    cost = dict(images=len(ids), accepted_images=n, fallback_images=len(ids)-n)
                    cache[n].update(cost, **fallback_counts(cost))
                value.update(fallback_counts(value), oracle=cache[n])
                if split == 'test':
                    value['paired_F1_interval_vs_all_deepsolo'] = conditional_interval(data[model],ids,accepted,weights)
                return value
            thresholds = [dict(kind='TRAIN_OOF_quantile', **p) for p in choice['scalar_thresholds']]
            thresholds += [dict(kind='fixed_utility',action='score_at_least',threshold=t) for t in cfg['fixed_utility_thresholds']]
            transferred = [dict(policy=p, **summarize(apply_threshold(scores,p))) for p in thresholds]
            quota = {str(q):summarize(ranked(scores,ids,q)) for q in cfg['bypass_fractions']}
            output[split][model] = dict(all_light=aggregate(data[model][s]['light'] for s in ids),
                all_deepsolo=aggregate(data[model][s]['deepsolo'] for s in ids), scores=scores,
                scalar_threshold_results=transferred, equal_quota_results=quota,
                base_deepsolo_images=0, base_total_images=len(ids), base_deepsolo_percentage=0.,
                all_deepsolo_images=len(ids), all_deepsolo_percentage=100.)
    seal(root/'results.json', dict(status='SUCCESS',reproduction_mode='approximate',claim_label='approximate reproduction',
        result_class=plan['result_class'], official_evaluation=False, table_5_claim=False,
        plan_sha256=sha(root/'resolved_configuration.json'),frozen_choices_sha256=sha(root/'frozen_choices.json'),
        splits=output, routing_cost_scope='Logical selective decisions over cached outputs; new TRAIN fallback inference logged separately',
        thresholds_fitted_on='TRAIN grouped OOF only', test_selection=False))


def audit(plan, root):
    frozen = read(root/'frozen_choices.json'); result = read(root/'results.json'); checks = 0
    ids_train = {s['sample_id'] for s in plan['samples']['train']}
    for name, choice in frozen['choices'].items():
        p = root/choice['model_path']
        if sha(p) != choice['model_sha256']:
            raise ValueError('Final learned head changed')
        saved = torch.load(p,map_location='cpu',weights_only=True)
        if set(saved['preprocessing']['training_ids']) != ids_train or saved['preprocessing']['seed'] != plan['config']['execution']['seed']:
            raise ValueError('Final model did not fit the complete TRAIN population')
    for trace in frozen['fold_trace']:
        if set(trace['fitting_ids'])&set(trace['held_ids']) or set(trace['fitting_ids'])|set(trace['held_ids']) != ids_train:
            raise ValueError('Invalid OOF fitting/held scope')
        for epoch in plan['config']['training']['checkpoints']:
            entry = frozen['cv_models'][trace['backbone']+'__fold'+str(trace['fold'])+'__'+str(epoch)]
            path = root/entry['model_path']
            if sha(path) != entry['model_sha256']:
                raise ValueError('Cross-validation model changed')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if (saved['plan_sha256'] != sha(root/'resolved_configuration.json')
                    or saved['preprocessing']['training_ids'] != trace['fitting_ids']
                    or saved['preprocessing']['seed'] != plan['config']['execution']['seed']+trace['fold']):
                raise ValueError('Saved fold model does not match its TRAIN-only fitting scope')
            checks += 1
    for split in ('train','test'):
        records = read(root/split/'branches.json'); ids = sorted(records['records'])
        for sid, raw in records['records'].items():
            for name in plan['config']['backbones']:
                for which,pred in [('light',raw['light'][name]),('deepsolo',raw['deepsolo'])]:
                    if independent_counts(pred,raw['ground_truth'],'casefold_alnum') != records['branches'][name][sid][which]:
                        raise ValueError('Independent per-image count audit failed')
                    checks += 1
        for name,value in result['splits'][split].items():
            for p in [*value['equal_quota_results'].values(),*value['scalar_threshold_results']]:
                independent = independent_route(records['branches'][name],ids,p['accepted_ids'])
                if any(p['combined'][k] != v for k,v in independent.items()) or fallback_counts(p)['deepsolo_percentage'] != p['deepsolo_percentage']:
                    raise ValueError('Independent route or cost audit failed')
                checks += 1
    if plan['result_class'] != 'SMOKE_TEST':
        endpoint = result['splits']['test']['unet']['all_deepsolo']
        expected = read(Path(plan['paths']['improved_root'])/'deployment_profiles.json')['profiles']['native_word']['splits']['test']['word']['metrics']
        if any(endpoint[k] != expected[k] for k in ['tp','fp','fn','detected_words','gt_words']):
            raise ValueError('Improved all-DeepSolo endpoint did not replay exactly')
    seal(root/'independent_audit.json',dict(status='PASS',checks=checks,results_sha256=sha(root/'results.json'),
        scope='Independent word/route/count checks, full-TRAIN fit identity, fold membership and qualified fallback endpoint; not independent neural refitting'))
