"""Cached native/second-scale word augmentation and independent VAL replay."""

import argparse
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

from scripts.qualify_uniform_scaled_frontend import inputs, evaluate
from scripts.run_uniform_unet_training import read, write
from scripts.icdar_budget_data import key
from recognition.uniform_encoder_context import raw_context
from recognition.uniform_multiscale_words import augment, iou
from segmentation.uniform_unet_fusion import file_hash


def samples(c,source,records,ids):
    folders=[source/'export_train',Path(c['scaled_export'])];indices=[read(p/'index.json') for p in folders]
    result={};evidence={}
    for sid in ids:
        result[sid]=[]
        for folder,index in zip(folders,indices):
            item=index[sid];sample=read(item['path']);mp=folder/'maps'/(key(sid)+'.npz')
            if (file_hash(item['path'])!=item['sha256'] or sample['ground_truth_used'] or
                sample['image_sha256']!=records[sid]['image_sha256'] or file_hash(mp)!=sample['maps_sha256']):
                raise ValueError('Unauthenticated word-view input')
            result[sid].append(sample);evidence[item['path']]=item['sha256'];evidence[str(mp)]=sample['maps_sha256']
    return result,evidence


def generate(path):
    c,u,source,plan,records,ids=inputs(path);root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    if shutil.disk_usage(root).free<c['minimum_free_disk_bytes']:raise RuntimeError('Word-view storage reserve exhausted')
    write(root/'resolved_configuration.json',c);repo=Path(__file__).resolve().parents[1];manifest={}
    for package in c['snapshot_packages']:
        for p in (repo/package).rglob('*.py'):
            relative=p.relative_to(repo);dest=root/'code_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest);manifest[str(relative)]=file_hash(dest)
    write(root/'source_manifest.json',manifest)
    complete=read(source/'complete.json');large=read(Path(c['scaled_export'])/'complete.json')
    assert complete['checkpoint_sha256']==large['checkpoint_sha256'] and not large['ground_truth_used']
    entries,evidence=samples(c,source,records,ids);write(root/'source_map_evidence.json',evidence)
    context=raw_context(u,{s:records[s] for s in ids});index={};added=0
    for sid in ids:
        native,big=entries[sid];rows=augment(native['raw_rows'],big['raw_rows'],**c['word_augmentation'])
        with np.load(source/'export_train/maps'/(key(sid)+'.npz'),allow_pickle=False) as z:probability=z['probabilities']
        mp=root/'scaled/maps'/(key(sid)+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(mp,probabilities=probability,encoder_context=context[sid])
        p=root/'scaled/samples'/(key(sid)+'.json')
        write(p,dict(raw_rows=rows,rows=rows,image_wh=records[sid]['image_wh'],image_sha256=records[sid]['image_sha256'],
            rgb_sha256=records[sid]['rgb_sha256'],maps_path=str(mp),maps_sha256=file_hash(mp),ground_truth_used=False,
            source_native_rows=len(native['raw_rows']),added_scale_rows=len(rows)-len(native['raw_rows'])))
        index[sid]=dict(path=str(p),sha256=file_hash(p));added+=len(rows)-len(native['raw_rows'])
    write(root/'scaled/index.json',index)
    settings=dict(read(source/'emission_selection/frozen.json')['selected']['policy'],minimum_transcript_probability=0.)
    write(root/'scaled/complete.json',dict(status='COMPLETE',mode='approximate',claim_label='approximate reproduction',
        checkpoint_sha256=complete['checkpoint_sha256'],crop_settings=settings,ground_truth_used=False,
        total_images=len(ids),actual_unet_image_inferences=0,actual_parseq_crop_inferences=0,actual_deepsolo_calls=0,
        added_raw_candidates=added,source_sha256=file_hash(__file__),
        deployment_requirement='Two U-Net forwards and both scales PARSeq crop readings per image. Native masks/context retained; unchanged neural parameter count. Cached replay is not online latency.'))


def audit_views(path):
    c,u,source,plan,records,ids=inputs(path);root=Path(c['output_root'])
    for name,digest in read(root/'source_map_evidence.json').items():assert file_hash(name)==digest
    entries,_=samples(c,source,records,ids);context=raw_context(u,{s:records[s] for s in ids})
    index=read(root/'scaled/index.json');policy=c['word_augmentation']
    for sid in ids:
        a,b=entries[sid];expected=[dict(r) for r in a['raw_rows']]
        candidates=[(i,r) for i,r in enumerate(b['raw_rows']) if r['sequence_confidence_native']>=policy['minimum_sequence_probability'] and any(x.isalnum() for x in r['raw_text'])]
        candidates=sorted(candidates,key=lambda z:(-z[1]['sequence_confidence_native'],tuple(z[1]['bbox']),z[1]['raw_text'],z[0]))
        for i,r in candidates:
            if len(expected)>=policy['maximum_candidates']:break
            if not any(iou(r['bbox'],v['bbox'])>=policy['maximum_overlap'] for v in expected):
                expected.append(dict(r,multiscale_added=True,scale_source_row=i,source_row=len(expected)))
        item=index[sid];sample=read(item['path']);mp=root/'scaled/maps'/(key(sid)+'.npz')
        assert file_hash(item['path'])==item['sha256'] and sample['raw_rows']==expected and file_hash(mp)==sample['maps_sha256']
        with np.load(mp,allow_pickle=False) as x,np.load(source/'export_train/maps'/(key(sid)+'.npz'),allow_pickle=False) as y:
            assert np.array_equal(x['probabilities'],y['probabilities']) and np.array_equal(x['encoder_context'],context[sid])
    write(root/'independent_map_audit.json',dict(status='PASS',images=len(ids),
        scope='Independent candidate ordering/admission replay with shared IoU primitive; exact native mask/context, source bytes and native-prefix preservation. No new neural calls.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=['export','score','audit'],required=True);a=p.parse_args()
    try:
        with threadpool_limits(limits=2):
            if a.stage=='export':generate(a.config)
            else:
                if a.stage=='audit':audit_views(a.config)
                evaluate(a.config,a.stage=='audit')
    except BaseException as exc:
        root=Path(read(a.config)['output_root'])
        if root.exists():write(root/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
