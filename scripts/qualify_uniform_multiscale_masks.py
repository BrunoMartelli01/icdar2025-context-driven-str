"""Blend authenticated native/scaled VAL maps, read fresh crops, then replay."""

import argparse
import gc
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from scripts.run_uniform_unet_training import read, write
from scripts.qualify_uniform_scaled_frontend import inputs, evaluate
from scripts.export_uniform_unet import rows_with_sources
from scripts.train_icdar13_word_refit import recognizer, read_rows
from scripts.run_icdar13_dense_instances import runtime
from scripts.icdar_budget_data import key
from recognition.uniform_encoder_context import raw_context
from segmentation.uniform_unet_fusion import file_hash
from segmentation.uniform_multiscale_masks import fuse


def generate(path):
    c, u, source, plan, records, ids = inputs(path)
    root = Path(c['output_root']); root.mkdir(parents=True, exist_ok=False)
    write(root/'resolved_configuration.json', c)
    repo = Path(__file__).resolve().parents[1]; manifest = {}
    for package in c['snapshot_packages']:
        for p in (repo/package).rglob('*.py'):
            relative=p.relative_to(repo); dest=root/'code_snapshot'/relative
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
            manifest[str(relative)] = file_hash(dest)
    write(root/'source_manifest.json', manifest)
    small = source/'export_train'; large = Path(c['scaled_export'])
    ni, li = read(small/'index.json'), read(large/'index.json')
    nn = read(source/'complete.json'); lc = read(large/'complete.json')
    if lc['checkpoint_sha256'] != nn['checkpoint_sha256'] or lc['ground_truth_used']:
        raise ValueError('Both scales must use the same qualified checkpoint without GT')
    context = raw_context(u, {s:records[s] for s in ids})
    settings = dict(read(source/'emission_selection/frozen.json')['selected']['policy'],
        minimum_transcript_probability=0.)
    prepared = {}; evidence = {}
    for sid in ids:
        samples=[]
        for folder,index in ((small,ni),(large,li)):
            entry=index[sid];sample=read(entry['path']);mp=folder/'maps'/(key(sid)+'.npz')
            if (file_hash(entry['path'])!=entry['sha256'] or file_hash(mp)!=sample['maps_sha256'] or
                sample['ground_truth_used'] or sample['image_sha256']!=records[sid]['image_sha256']):
                raise ValueError('Changed or unauthenticated scale map')
            evidence[entry['path']]=entry['sha256'];evidence[str(mp)]=sample['maps_sha256']
            with np.load(mp,allow_pickle=False) as z:samples.append(z['probabilities'])
        prepared[sid] = fuse(*samples,c['larger_weight'])
    write(root/'source_map_evidence.json',evidence)
    runtime(plan['config']);ocr=recognizer(plan);index={};crops=0
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        for sid in ids:
            r=records[sid]
            if file_hash(r['image_path'])!=r['image_sha256']:raise ValueError('Changed crop RGB')
            probability=prepared[sid]
            _,raw,expansion=read_rows(probability,r,settings,ocr,plan['config']['parseq_batch_size'])
            raw,rows=rows_with_sources(raw,0.,r['image_wh'])
            mp=root/'scaled/maps'/(key(sid)+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(mp,probabilities=probability,encoder_context=context[sid])
            sample=root/'scaled/samples'/(key(sid)+'.json')
            write(sample,dict(raw_rows=raw,rows=rows,image_wh=r['image_wh'],image_sha256=r['image_sha256'],
                rgb_sha256=r['rgb_sha256'],maps_path=str(mp),maps_sha256=file_hash(mp),ground_truth_used=False,expansion=expansion))
            index[sid]=dict(path=str(sample),sha256=file_hash(sample));crops+=len(raw)
        torch.cuda.synchronize();write(root/'scaled/index.json',index)
        write(root/'scaled/complete.json',dict(status='COMPLETE',mode='approximate',claim_label='approximate reproduction',
            actual_unet_image_inferences=0,unet_inferences_reused=2*len(ids),actual_parseq_crop_inferences=crops,
            actual_deepsolo_calls=0,total_images=len(ids),precision='float32',parseq_batch_size=plan['config']['parseq_batch_size'],
            gpu=torch.cuda.get_device_name(0),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            out_of_memory=False,seconds=time.perf_counter()-started,checkpoint_sha256=nn['checkpoint_sha256'],ground_truth_used=False,
            deployment_requirement='Two U-Net forwards per image at768 and1024, one native encoder context and one crop-reading pass. Offline replay time is not online frontend latency.',
            context_scope='Actual native768 encoder context, intentionally retained; not a claim that blended maps come from one forward.',
            crop_settings=settings,source_sha256=file_hash(__file__)))
    finally:
        ocr.release();del ocr;gc.collect();torch.cuda.empty_cache()


def audit_maps(path):
    c,u,source,plan,records,ids=inputs(path);root=Path(c['output_root'])
    for name,digest in read(root/'source_map_evidence.json').items():
        if file_hash(name)!=digest:raise ValueError('Changed source probability evidence')
    context=raw_context(u,{s:records[s] for s in ids})
    output=read(root/'scaled/index.json');native=read(source/'export_train/index.json')
    large=Path(c['scaled_export']);larger=read(large/'index.json')
    for sid in ids:
        paths=[source/'export_train/maps'/(key(sid)+'.npz'),large/'maps'/(key(sid)+'.npz')]
        arrays=[]
        for mp in paths:
            with np.load(mp,allow_pickle=False) as z:arrays.append(z['probabilities'])
        resized=torch.nn.functional.interpolate(torch.from_numpy(arrays[1])[None],size=arrays[0].shape[1:],
            mode='bilinear',align_corners=False)[0].numpy()
        expected=arrays[0]*(1-c['larger_weight'])+resized*c['larger_weight']
        sample=read(output[sid]['path']);mp=root/'scaled/maps'/(key(sid)+'.npz')
        assert file_hash(output[sid]['path'])==output[sid]['sha256'] and file_hash(mp)==sample['maps_sha256']
        with np.load(mp,allow_pickle=False) as z:
            assert np.array_equal(z['probabilities'],expected) and np.array_equal(z['encoder_context'],context[sid])
    write(root/'independent_map_audit.json',dict(status='PASS',images=len(ids),
        scope='Independent probability arithmetic after shared qualified half-pixel interpolation; exact native-context and source-hash checks. OCR is not independently re-inferred.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=['export','score','audit'],required=True);a=p.parse_args()
    try:
        with threadpool_limits(limits=2):
            if a.stage=='export':generate(a.config)
            else:
                if a.stage=='audit':audit_maps(a.config)
                evaluate(a.config,a.stage=='audit')
    except BaseException as exc:
        root=Path(read(a.config)['output_root'])
        if root.exists():write(root/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),
            out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError)))
        raise
