"""RGB-only common U-Net/PARSeq, CPU utility gate and isolated DeepSolo pipeline."""

import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

from recognition.uniform_utility_deployment import UniformUtilityPipeline
from recognition.uniform_fallback import UniformFallbackPostprocessor
from recognition.uniform_risk_fusion import route
from recognition.hard_fallback_budget import (
    validate_policy, batch_quota as hard_batch_quota, verify_route,
)
from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import key
from segmentation.uniform_unet_fusion import file_hash


def input_records(manifest):
    """Accept only RGB paths and IDs; never accept evaluation fields at this boundary."""
    if set(manifest) != {'samples'} or not manifest['samples']:
        raise ValueError('Nonempty RGB-only samples manifest required')
    result = []
    for row in manifest['samples']:
        if (set(row) != {'sample_id', 'image_path'} or not isinstance(row['sample_id'], str) or
                re.fullmatch(r'[A-Za-z0-9_.:-]+', row['sample_id']) is None or row['sample_id'] in ('.', '..')):
            raise ValueError('Only sample_id and image_path are allowed as inference input')
        path = Path(row['image_path']).resolve()
        with Image.open(path) as image:
            rgb = np.asarray(image.convert('RGB'))
            result.append(dict(sample_id=row['sample_id'], image_path=str(path), image_sha256=file_hash(path),
                image_wh=list(image.size), orientation=int(image.getexif().get(274, 1)),
                rgb_sha256=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()))
    if len({r['sample_id'] for r in result}) != len(result):
        raise ValueError('Unique inference sample IDs required')
    if len({key(r['sample_id']) for r in result}) != len(result):
        raise ValueError('Inference sample IDs collide in the versioned file interface')
    return result


def batch_quota(policy_quota, reference_images, images):
    if type(policy_quota) is not int or type(reference_images) is not int or type(images) is not int or not 0 <= policy_quota <= reference_images or min(reference_images, images) < 1:
        raise ValueError('Valid frozen reference quota and nonempty batch required')
    # Integer ceiling avoids floating-point off-by-one changes at full TEST N.
    return (policy_quota*images+reference_images-1)//reference_images


def frontend(training_root, records, output, *, capture_context=False):
    import torch
    from scripts.export_uniform_unet import checked_training, model, rows_with_sources
    from scripts.run_icdar13_dense_instances import runtime
    from scripts.train_icdar13_word_refit import recognizer, read_rows
    from segmentation.icdar2015_improvements import make_transform
    plan, complete = checked_training(training_root)
    c = plan['config']; policy = read(training_root/'emission_selection/frozen.json')
    if (policy['status'] != 'FROZEN' or policy['test_used_for_selection'] or
            policy['checkpoint_sha256'] != complete['checkpoint_sha256']):
        raise ValueError('Qualified validation-selected U-Net crop policy required')
    runtime(c); started = time.perf_counter(); net, ocr = model(plan, training_root), recognizer(plan)
    loading = time.perf_counter()-started
    encoded={}
    handle=net.encoder.layer4.register_forward_hook(lambda module,args,value:encoded.update(e4=value)) if capture_context else None
    transform = make_transform(c, c['image_hw'], augment=False)
    settings = dict(policy['selected']['policy'], minimum_transcript_probability=0.)
    index, crops = {}, 0; torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    try:
        for r in records:
            if file_hash(r['image_path']) != r['image_sha256']:
                raise ValueError('Input RGB changed before frontend inference')
            with Image.open(r['image_path']) as image:
                rgb = image.convert('RGB'); tensor, _ = transform(rgb, Image.new('L', rgb.size, 0), 0)
            with torch.inference_mode():
                probability = net(tensor[None].to(c['device']))[0].sigmoid().cpu().numpy()
                extra=dict(encoder_context=torch.cat((encoded['e4'].mean((2,3)),encoded['e4'].amax((2,3))),dim=1)[0].cpu().numpy()) if capture_context else {}
            _, raw, expansion = read_rows(probability, r, settings, ocr, c['parseq_batch_size'])
            raw, emitted = rows_with_sources(raw, policy['selected']['policy']['minimum_transcript_probability'], r['image_wh'])
            mp = output/'maps'/(key(r['sample_id'])+'.npz'); mp.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(mp, probabilities=probability, **extra)
            path = output/'samples'/(key(r['sample_id'])+'.json')
            write(path, dict(raw_rows=raw, rows=emitted, image_wh=r['image_wh'],
                rgb_sha256=r['rgb_sha256'], image_sha256=r['image_sha256'],
                maps_path=str(mp), maps_sha256=file_hash(mp), ground_truth_used=False, expansion=expansion))
            index[r['sample_id']] = dict(path=str(path), sha256=file_hash(path)); crops += len(raw)
        torch.cuda.synchronize()
        write(output/'index.json', index)
        receipt = dict(status='COMPLETE', model_loading_seconds=loading, seconds=time.perf_counter()-started,
            actual_unet_image_inferences=len(records), actual_parseq_crop_inferences=crops,
            peak_allocated=torch.cuda.max_memory_allocated(), peak_reserved=torch.cuda.max_memory_reserved(),
            gpu=torch.cuda.get_device_name(0), precision='float32', input_hw=c['image_hw'],
            unet_batch_size=1, parseq_batch_size=c['parseq_batch_size'], ground_truth_used=False,
            checkpoint_sha256=complete['checkpoint_sha256'], policy_sha256=file_hash(training_root/'emission_selection/frozen.json'))
        write(output/'complete.json', receipt)
        return index, receipt
    finally:
        if handle is not None:handle.remove()
        encoded.clear()
        ocr.release(); del ocr, net; gc.collect(); torch.cuda.empty_cache()


def run(config_path, manifest_path, dataset, policy_quota, output):
    started = time.perf_counter(); c = read(config_path)
    if c['mode'] != 'approximate' or config_path.resolve().parent.name != 'approximate':
        raise ValueError('Explicit approximate unified pipeline profile required')
    hard_budget = (validate_policy(c['hard_budgets'], dataset, policy_quota)
                   if 'hard_budgets' in c else None)
    for path, digest in c['pinned_inputs'].items():
        if file_hash(path) != digest:
            raise ValueError('Changed unified pipeline artifact: '+path)
    fallback_config = read(c['fallback_profiles'])
    qualification = read(Path(fallback_config['output_root'])/'postprocessor_qualification.json')
    if qualification['status'] != 'PASS' or qualification['config_sha256'] != file_hash(c['fallback_profiles']):
        raise ValueError('Qualified common fallback postprocessor required')
    if qualification['postprocessor_sha256'] != file_hash(Path(__file__).parents[1]/'recognition/uniform_fallback.py'):
        raise ValueError('Fallback postprocessor changed after parity qualification')
    # Deploy only neural/runtime/vocabulary resources. Historical annotation-
    # bearing cache files belong to the parity audit, never to RGB inference.
    resources = set()
    for profile in fallback_config['profiles'].values():
        resources.update([profile['neural_profile'], profile['runtime']])
        if profile['lexicon'] is not None:
            resources.update(profile['lexicon'][name] for name in ('vocabulary', 'pairs'))
    for path in resources:
        if file_hash(path) != fallback_config['pinned_inputs'][path]:
            raise ValueError('Changed qualified fallback resource: '+path)
    utility = UniformUtilityPipeline(Path(c['utility_config']), dataset, policy_quota)
    training_root = Path(utility.config['training_root'])/dataset
    plan = read(training_root/'resolved_configuration.json')
    records = input_records(read(manifest_path)); ids = [r['sample_id'] for r in records]
    k = (hard_batch_quota(hard_budget, policy_quota, len(plan['partitions']['test']), len(records))
         if hard_budget is not None else batch_quota(policy_quota, len(plan['partitions']['test']), len(records)))
    output.mkdir(parents=True, exist_ok=False)
    write(output/'contract.json', dict(mode='approximate', claim_label='approximate reproduction',
        config_sha256=file_hash(config_path), manifest_sha256=file_hash(manifest_path),
        pipeline_source_sha256=file_hash(__file__), dataset=dataset, policy_quota=policy_quota,
        records=records, hard_budget=hard_budget, ground_truth_used=False,
        execution='fresh neural inference, conditional heavy branch'))
    capture_context='encoder_context' in utility.config
    index, front = frontend(training_root, records, output/'frontend',capture_context=True) if capture_context else frontend(training_root, records, output/'frontend')
    observed = {}; started_route = time.perf_counter()
    with threadpool_limits(limits=c['router_cpu_threads']):
        for sid, entry in index.items():
            if file_hash(entry['path']) != entry['sha256']:
                raise ValueError('Changed fresh frontend observation')
            sample = read(entry['path'])
            if sample['ground_truth_used'] or file_hash(sample['maps_path']) != sample['maps_sha256']:
                raise ValueError('Annotation-free actual probability maps required')
            with np.load(sample['maps_path'], allow_pickle=False) as z:
                observed[sid] = utility.observe(sample['raw_rows'], sample['image_wh'], z['probabilities'],
                    z['encoder_context'] if capture_context else None)
    decision = route(observed, {r['sample_id']: r['rgb_sha256'] for r in records}, k, utility.policy['guard'])
    if hard_budget is not None:
        verify_route(decision, ids, hard_budget, policy_quota)
    route_seconds = time.perf_counter()-started_route
    write(output/'observations.json', observed)
    write(output/'decisions_before_fallback.json', dict(status='FROZEN', **decision,
        observations_sha256=file_hash(output/'observations.json'), reference_policy_quota=policy_quota,
        reference_images=len(plan['partitions']['test']), hard_budget=hard_budget,
        exact_batch_quota_rule=('floor(reference_fraction * batch_size)' if hard_budget is not None
                               else 'ceil(reference_fraction * batch_size)')))
    profile = fallback_config['profiles'][dataset]
    selected = set(decision['fallback_ids']); lookup = {r['sample_id']: r for r in records}
    export = dict(mode='approximate', claim_label='approximate reproduction',
        config=dict(mode='approximate', minimum_free_gpu_bytes=c['minimum_free_fallback_gpu_bytes'],
            paths=dict(deepsolo_profile=profile['neural_profile'], deepsolo_runtime=profile['runtime'], deepsolo_source=profile['source'])),
        population=dict(test={dataset: [{name: lookup[s][name] for name in ('sample_id', 'image_path', 'image_sha256')} for s in sorted(selected)]}),
        selected_before_inference=True, ground_truth_paths_included=False,
        decision_sha256=file_hash(output/'decisions_before_fallback.json'),
        deepsolo_images=k, total_images=len(records), deepsolo_percentage=100*k/len(records))
    write(output/'fallback_plan.json', export)
    fallback_started = time.perf_counter(); neural = dict(new_model_inferences=0, reused=0)
    if selected:
        with (output/'fallback.log').open('x') as log:
            subprocess.run([profile['python'], '-u', '-m', 'scripts.export_essential_transfer_deepsolo',
                '--plan', str(output/'fallback_plan.json'), '--output', str(output/'fallback')],
                cwd=Path(__file__).parents[1], stdout=log, stderr=subprocess.STDOUT, check=True)
        protocol = read(output/'fallback/protocol.json')
        if protocol['source_identity']['checkpoint_sha256'] != profile['checkpoint_sha256']:
            raise ValueError('Conditional DeepSolo returned a different checkpoint')
        neural = read(output/'fallback/receipt_limit_0.json')
        if neural['new_model_inferences']+neural['reused'] != k:
            raise ValueError('Incomplete conditional DeepSolo population')
    fallback_seconds = time.perf_counter()-fallback_started
    processor = UniformFallbackPostprocessor(profile); results = {}; started_combine = time.perf_counter()
    for sid in ids:
        if sid in selected:
            raw = read(output/'fallback'/(key(sid)+'.json'))
            content = {name: value for name, value in raw.items() if name != 'payload_sha256'}
            if (raw['payload_sha256'] != hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest() or
                    raw['contract']['plan_sha256'] != file_hash(output/'fallback_plan.json') or
                    raw['contract']['sample_id'] != sid or raw['contract']['image_sha256'] != lookup[sid]['image_sha256']):
                raise ValueError('Changed conditional DeepSolo output')
            r = lookup[sid]
            heavy = processor.apply(raw['response'], r['image_wh'], r['orientation'], profile['checkpoint_sha256'])
            rows = utility.combine_after_fallback(observed[sid], heavy['rows'])
            trace = heavy['lexical_trace']
        else:
            rows, trace = observed[sid]['rows'], []
        results[sid] = dict(rows=rows, fallback_lexical_trace=trace, action='deepsolo' if sid in selected else 'light', ground_truth_used=False)
    combine_seconds = time.perf_counter()-started_combine
    write(output/'predictions.json', results)
    write(output/'complete.json', dict(status='COMPLETE', mode='approximate', claim_label='approximate reproduction',
        images=len(records), deepsolo_images=k, total_images=len(records), deepsolo_percentage=100*k/len(records),
        hard_budget=hard_budget,
        actual_new_deepsolo_calls=neural['new_model_inferences'], reused_deepsolo_outputs=neural['reused'],
        actual_new_unet_calls=front['actual_unet_image_inferences'], actual_new_parseq_crop_calls=front['actual_parseq_crop_inferences'],
        frontend=front, route_seconds=route_seconds, fallback_subprocess_wall_seconds=fallback_seconds,
        router_cpu_threads=c['router_cpu_threads'],
        combine_seconds=combine_seconds, total_wall_seconds=time.perf_counter()-started,
        predictions_sha256=file_hash(output/'predictions.json'), ground_truth_used=False,
        scope='Fresh RGB-only pipeline. Timing includes initialization/I/O and selected legacy subprocess; repeated idle-GPU matched controls required for a general speed claim.'))
