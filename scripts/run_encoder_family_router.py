"""Train matched CLIP/BLIP/combined approximate image-deferral heads."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch

from recognition.encoder_family_router import EncoderFamilyRouter, collate_observations, count_targets
from recognition.token_context_router import benefit_loss
from recognition.unet_corrected_token_interfaces import route_summary
from scripts.run_essential_router_budget import rank, scalar_cut, threshold_ids

read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def pin(path, pins):
    path = Path(path); digest = sha(path)
    if str(path) in pins and pins[str(path)] != digest:
        raise ValueError('Changed source: '+str(path))
    pins[str(path)] = digest


def load_population(config, roles, split, pins):
    """Authenticate caches; labels are held outside the observation dictionaries."""
    compact = Path(config['paths']['lightweight_run']); source = Path(config['paths']['token_source'])
    source_records = {r['sample_id']: r for r in read(source/'resolved_configuration.json')['records'][split]}
    readings_path = source/split/'corrected.json'; pin(readings_path, pins); readings = read(readings_path)
    data = {}; branches = {}; eligibility = {}; entries = []
    for role in roles:
        feature = compact/(role+'_features.npz'); schema_path = compact/(role+'_feature_schema.json')
        branch_path = compact/(role+'_branches.json')
        for path in (feature, schema_path, branch_path): pin(path, pins)
        schema = read(schema_path)
        if sha(feature) != schema['arrays_sha256']: raise ValueError('Changed scalar cache')
        with np.load(feature, allow_pickle=False) as z: scalars = np.concatenate([z['ocr'], z['mask']], 1)
        branches.update(read(branch_path)); eligibility.update(schema['eligible'])
        for sid, scalar in zip(schema['ids'], scalars):
            if sid in data: raise ValueError('Duplicate sample across roles')
            npz = source/split/'features/corrected'/(sid.replace(':', '__')+'.npz')
            meta_path = npz.with_suffix('.json'); pin(meta_path, pins); meta = read(meta_path)
            if meta['ground_truth_used'] or meta['transcript_likelihood_used']:
                raise ValueError('Forbidden feature source')
            if meta['image_sha256'] != source_records[sid]['image_sha256'] or sha(npz) != meta['arrays_sha256']:
                raise ValueError('Image/cache binding failure')
            pin(npz, pins)
            with np.load(npz, allow_pickle=False) as z:
                row = dict(texts=[r['raw_text'] for r in readings[sid]], geometry=z['word_geometry'],
                           scalars=scalar.astype(np.float32), patches=z['patches'][:196].copy(),
                           patch_geometry=z['patch_geometry'][:196].copy(), patch_valid=z['patch_valid'][:196].copy(),
                           queries=z['queries'], word_text=z['word_text'], word_image=z['word_image'])
            if row['geometry'].shape != (len(row['texts']), 12) or row['queries'].shape != (32, 768):
                raise ValueError('Unexpected context/word schema')
            if any(not np.isfinite(v).all() for v in row.values() if isinstance(v, np.ndarray)):
                raise ValueError('Nonfinite observation')
            data[sid] = row; entries.append(dict(sample_id=sid, image_sha256=meta['image_sha256'],
                arrays_sha256=meta['arrays_sha256'], words=len(row['texts']), role=role))
    return data, branches, eligibility, entries


def normalize(data, mean, scale):
    for row in data.values(): row['scalars'] = ((row['scalars']-mean)/scale).astype(np.float32)


@torch.inference_mode()
def predict(model, ids, data, eligible, settings, device):
    model.eval(); result = {}
    for start in range(0, len(ids), settings['batch_size']):
        selected = ids[start:start+settings['batch_size']]
        x = collate_observations([data[s] for s in selected], model.arm, device)
        predictions = model(x).cpu().numpy()*settings['target_scale']+np.asarray(settings['target_mean'])
        values = 2*predictions[:, 0]-settings['reference_f1']*predictions[:, 1]
        for sid, value in zip(selected, values): result[sid] = float(value) if eligible[sid] else None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['approximate'], required=True)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true'); args = parser.parse_args(); c = read(args.config)
    if c['mode'] != args.mode or args.config.resolve().parent.parts[-2:] != ('configs', 'approximate'):
        raise ValueError('Explicit isolated approximate configuration required')
    root = args.output.resolve(); root.mkdir(exist_ok=False); (root/'models').mkdir()
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    torch.set_num_threads(c['cpu_threads']); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if not torch.cuda.is_available(): raise RuntimeError('Explicit CUDA run required')
    free, total = torch.cuda.mem_get_info()
    if free < c['minimum_free_gpu_bytes']: raise RuntimeError('Insufficient free GPU memory; no automatic substitution')
    hardware = dict(host=subprocess.check_output(['hostname', '-f'], text=True).strip(), python=sys.version,
        torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        free_bytes_at_start=free, total_bytes=total, device='cuda:0', precision='float32', batch_size=c['batch_size'],
        frozen_encoders='authenticated cached outputs, not resident during head training', amp=False, tf32=False)
    pins = dict(c['input_sha256'])
    for path in list(pins): pin(path, pins)
    fit, fit_branches, eligible, fit_entries = load_population(c, ['fit', 'calibration'], 'train', pins)
    held, held_branches, held_eligible, held_entries = load_population(c, ['selection'], 'train', pins)
    original = read(Path(c['paths']['budget_reference'])/'frozen_choices.json')
    selection = original['selection_ids']; calibration = original['calibration_ids']; fit_ids = sorted(fit)
    if set(fit)&set(held) or set(selection)&set(calibration) or set(selection+calibration) != set(held):
        raise ValueError('Expert/selection/calibration overlap')
    if len(fit_ids) != c['expert_fit_images']: raise ValueError('Incomplete training population')
    if args.smoke:
        fit_ids = fit_ids[:32]; selection = selection[:6]; calibration = calibration[:6]
    scalars = np.stack([fit[s]['scalars'] for s in fit_ids]).astype(np.float64)
    mean = scalars.mean(0); scale = scalars.std(0); scale[scale == 0] = 1
    normalize(fit, mean, scale); normalize(held, mean, scale)
    targets = count_targets(fit_branches, fit_ids); target_mean = targets.mean(0)
    target_scale = max(float(targets.std()), c['minimum_target_scale'])
    ref = route_summary(fit_branches, fit_ids, [])['f1']
    settings = dict(batch_size=c['batch_size'], target_mean=target_mean.tolist(), target_scale=target_scale, reference_f1=ref)
    resolved = dict(mode='approximate', claim_label='smoke-test result' if args.smoke else 'approximate reproduction',
        source_config=c, config_sha256=sha(args.config), hardware=hardware, fit_ids=fit_ids, selection_ids=selection,
        calibration_ids=calibration, feature_mean=mean.tolist(), feature_scale=scale.tolist(), inference_settings=settings,
        feature_manifest=fit_entries+held_entries, input_sha256=pins, frozen_encoders=True,
        training_supervision='paired native DeepSolo/light counts; GT excluded from observation dictionaries',
        test_used=False, expert_gradient_holdout_verified=True)
    write(root/'resolved_configuration.json', resolved)
    epochs = 1 if args.smoke else c['epochs']; checks = [1] if args.smoke else c['checkpoints']
    arm_choices = {}; histories = {}; memory = {}; start_time = time.monotonic()
    for arm in c['arms']:
        torch.manual_seed(c['seed']); model = EncoderFamilyRouter(arm, c['architecture'], c['seed']).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
        target = torch.as_tensor((targets-target_mean)/target_scale, device='cuda')
        history = []; best = None; torch.cuda.reset_peak_memory_stats(); arm_start = time.monotonic()
        for epoch in range(1, epochs+1):
            model.train(); order = np.random.default_rng(c['seed']+epoch).permutation(len(fit_ids)); losses = []
            for start in range(0, len(order), c['batch_size']):
                indices = order[start:start+c['batch_size']]; ids = [fit_ids[i] for i in indices]
                batch = collate_observations([fit[s] for s in ids], arm, 'cuda')
                optimizer.zero_grad(set_to_none=True)
                loss, _ = benefit_loss(model(batch), target[indices], ref, c)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite training loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), c['gradient_clip'], error_if_nonfinite=True)
                optimizer.step(); losses.append(float(loss.detach()))
            record = dict(epoch=epoch, loss=float(np.mean(losses)))
            if epoch in checks:
                scores = predict(model, selection, held, held_eligible, settings, 'cuda')
                summary = route_summary(held_branches, selection, rank(scores, selection, c['selection_acceptance']))
                path = root/'models'/(arm+'__epoch'+str(epoch)+'.pt')
                torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}, arm=arm,
                    architecture=c['architecture'], seed=c['seed'], scalar_mean=mean.tolist(), scalar_scale=scale.tolist(),
                    inference_settings=settings, fit_ids=fit_ids, epoch=epoch, mode='approximate'), path)
                point = dict(arm=arm, epoch=epoch, checkpoint_path=str(path), checkpoint_sha256=sha(path),
                    head_parameters=sum(p.numel() for p in model.parameters()),
                    head_tensor_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()),
                    checkpoint_file_bytes=path.stat().st_size, selection_summary=summary,
                    frozen_encoder_bytes=c['expected_frozen_encoder_bytes'][arm])
                record['selection'] = point
                if best is None or summary['f1'] > best['selection_summary']['f1']: best = point
                print(json.dumps(dict(arm=arm, epoch=epoch, loss=record['loss'], selection_f1=summary['f1'],
                    D=summary['deepsolo_images'], N=summary['total_images'], percent=summary['deepsolo_percent'])), flush=True)
            history.append(record)
        memory[arm] = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            training_seconds=time.monotonic()-arm_start, oom=False, model_set='trainable router head only',
            precision='float32', device='cuda:0', batch_size=c['batch_size'], context_tokens=dict(clip=196, blip=32),
            variable_input_shapes='all UTF8 bytes and all corrected predicted words, dynamically padded')
        checkpoint = torch.load(best['checkpoint_path'], map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        scores = predict(model, calibration, held, held_eligible, settings, 'cuda'); policies = []
        for fraction in c['calibration_acceptances']:
            cut = scalar_cut(scores, calibration, fraction)
            policies.append(dict(target_calibration_acceptance=fraction, **cut,
                calibration_summary=route_summary(held_branches, calibration, threshold_ids(scores, cut))))
        best['policies'] = policies; arm_choices[arm] = best; histories[arm] = history
        del model, optimizer, target, checkpoint, batch, loss; gc.collect(); torch.cuda.empty_cache()
    best_f1 = max(v['selection_summary']['f1'] for v in arm_choices.values())
    candidates = [v for v in arm_choices.values() if v['selection_summary']['f1'] >= best_f1-c['selection_tolerance']]
    primary = min(candidates, key=lambda v:(v['frozen_encoder_bytes']+v['head_tensor_bytes'], -v['selection_summary']['f1'], v['arm']))['arm']
    frozen = dict(mode='approximate', claim_label=resolved['claim_label'], primary_arm=primary, arm_choices=arm_choices,
        selection_ids=selection, calibration_ids=calibration, no_refit=True, test_used=False,
        resolved_configuration_sha256=sha(root/'resolved_configuration.json'), selection_rule=c['selection_rule'])
    write(root/'training_history.json', histories); write(root/'training_memory.json', memory)
    write(root/'frozen_choices.json', frozen)
    # Do not load TEST observations or outcomes until the checkpoint and cuts for every arm are frozen.
    test, test_branches, test_eligible, test_entries = load_population(c, ['test'], 'test', pins)
    normalize(test, mean, scale); test_ids = sorted(test)
    if args.smoke: test_ids = test_ids[:8]
    results = {}; all_scores = {}
    for arm, chosen in arm_choices.items():
        model = EncoderFamilyRouter(arm, c['architecture'], c['seed']).cuda()
        if sha(chosen['checkpoint_path']) != chosen['checkpoint_sha256']: raise ValueError('Changed chosen checkpoint')
        ck = torch.load(chosen['checkpoint_path'], map_location='cpu', weights_only=True)
        model.load_state_dict(ck['state_dict'], strict=True)
        scores = predict(model, test_ids, test, test_eligible, settings, 'cuda'); all_scores[arm] = scores
        policies = [dict(target_calibration_acceptance=p['target_calibration_acceptance'], threshold=p['threshold'],
                         tie_sample_id=p['tie_sample_id'],
                         **route_summary(test_branches, test_ids, threshold_ids(scores, p))) for p in chosen['policies']]
        results[arm] = dict(chosen={k:v for k,v in chosen.items() if k!='policies'}, policies=policies,
            exact50=route_summary(test_branches, test_ids, rank(scores, test_ids, .5)))
        del model, ck; gc.collect(); torch.cuda.empty_cache()
    write(root/'test_scores.json', all_scores)
    write(root/'test_sources.json', dict(entries=test_entries, input_sha256=pins))
    result = dict(status='SUCCESS', mode='approximate', claim_label=resolved['claim_label'], primary_arm=primary,
        primary_acceptance=.5, arms=results, test_images=len(test_ids), new_expert_inference=0,
        elapsed_seconds=time.monotonic()-start_time, frozen_choices_sha256=sha(root/'frozen_choices.json'),
        paper_comparison='Nominal only: paper79.8 is undefined recognition accuracy, ours symbol-free lexical microF1; paper TotalText fallback count unavailable.',
        historical_test_exposure=True, fit_images=len(fit_ids), selection_images=len(selection), calibration_images=len(calibration))
    write(root/'results.json', result)
    write(root/'completion.json', dict(status='SUCCESS', results_sha256=sha(root/'results.json')))
    print(json.dumps(dict(primary=primary, test=[dict(arm=a, f1=v['policies'][1]['f1'], D=v['policies'][1]['deepsolo_images'],
        N=v['policies'][1]['total_images'], percent=v['policies'][1]['deepsolo_percent']) for a,v in results.items()])), flush=True)


if __name__ == '__main__': main()
