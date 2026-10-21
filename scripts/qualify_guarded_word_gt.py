"""Qualify dynamic V3 targets on complete TRAIN, with no training/GPU/crop job."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import platform
import resource
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import scipy
import torch

from segmentation.data import file_sha256
from segmentation.guarded_word_gt import make_guarded_dataset


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')


def independent_check(owners, audit):
    # OpenCV independently checks the SciPy decisions; all source IDs count.
    for row in audit['per_word']:
        ys,xs=np.where(owners==row['owner'])
        count=0
        if len(ys):
            roi=(owners[ys.min():ys.max()+1,xs.min():xs.max()+1]==row['owner']).astype(np.uint8)
            count=int(cv2.connectedComponents(roi,connectivity=8)[0]-1)
        if count!=row['after_components'] or len(ys)!=row['after_pixels']:
            raise AssertionError('Independent owner topology disagrees')
    count,labels=cv2.connectedComponents((owners>0).astype(np.uint8),connectivity=8)
    for component in range(1,count):
        if len(np.unique(owners[labels==component]))!=1:
            raise AssertionError('Distinct owners share a connected component')


def run(args):
    started=time.monotonic();c=json.loads(args.config.read_text())
    if args.output.exists():raise FileExistsError('Use a fresh result directory')
    if args.config.resolve().parent.name!='approximate' or c['reproduction_mode']!='approximate':
        raise ValueError('Explicit isolated APPROXIMATE configuration required')
    torch.set_num_threads(c['cpu_threads']);torch.set_num_interop_threads(1);cv2.setNumThreads(c['cpu_threads'])
    dataset=make_guarded_dataset(args.config,augmentation_enabled=False,verify_checksums=True)
    if len(dataset)!=c['expected_images']:raise ValueError('Unexpected source TRAIN population')
    ids=list(range(len(dataset)))
    if args.limit_images:
        priority=['total_text:1006','total_text:114','total_text:1009']
        by_id={s.sample_id:i for i,s in enumerate(dataset.manifest.samples)}
        ids=[by_id[sid] for sid in priority]+[i for i in ids if dataset.manifest.samples[i].sample_id not in priority]
        ids=ids[:args.limit_images]
        if not ids:raise ValueError('Smoke must include at least one image')
    files=[Path(__file__).resolve(),Path('segmentation/guarded_word_gt.py').resolve(),
        Path('segmentation/transforms.py').resolve(),Path('segmentation/data.py').resolve(),Path('segmentation/train.py').resolve()]
    protocol={'reproduction_mode':'approximate','claim_label':'approximate reproduction','resolved_config':c,
        'result_class':'smoke-test result' if args.limit_images else 'approximate reproduction; full TRAIN target qualification',
        'config_sha256':file_sha256(args.config),'code_sha256':{str(p):file_sha256(p) for p in files},
        'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'opencv':cv2.__version__,
        'device':'cpu','gpu_inference':False,'sample_ids':[dataset.manifest.samples[i].sample_id for i in ids],
        'augmentation_index':'original full-manifest dataset index; never subset index',
        'preview_contract':'native128 owner/mask previews only; training must use make_guarded_dataset(config, augmentation_enabled=True)',
        'causal_ocr_claim':False,'new_training':False}
    write(args.output/'protocol.json',protocol)
    views=[('unaugmented',None)]+[(f'augmented_epoch_{e}',e) for e in c['augmentation_epochs']]
    summary={};previews=[];independent=0
    for name,epoch in views:
        if epoch is not None:
            # Rebuild the same paired policy, keeping dataset ordering exactly.
            dataset=make_guarded_dataset(args.config,augmentation_enabled=True,verify_checksums=False)
            dataset.transform.set_epoch(epoch)
        stats=Counter();reasons=Counter();records=[];hashes={}
        for j,index in enumerate(ids,1):
            item=dataset[index];adapter=dataset.transform;audit=adapter.last_audit;own=adapter.last_owners.copy()
            if not np.array_equal(own>0,item['mask'][0].numpy()>0):
                raise AssertionError('Returned training mask differs from guarded owner raster')
            if torch.any(item['mask'].bool()&item['separation'].bool()):
                raise AssertionError('Restored foreground remains in negative separator')
            if torch.any(item['mask'].bool()&~item['validity'].bool()):
                raise AssertionError('Foreground restored into invalid pixels')
            independent_check(own,audit);independent+=1
            stats.update(audit['stats'])
            reasons.update(reason for word in audit['per_word'] for reason in word['unresolved'])
            digest=sha256(own.astype('<i4').tobytes()).hexdigest();hashes[index]=digest
            records.append({**audit,'native_owner_array_sha256':digest,
                'rgb_tensor_sha256':sha256(item['image'].numpy().tobytes()).hexdigest()})
            if epoch is None:
                key=item['sample_id'].replace(':','__');entries={}
                for field,array in [('owners',own.astype(np.uint16)),('masks',(own>0).astype(np.uint8)*255)]:
                    path=args.output/'native_preview'/field/(key+'.png');path.parent.mkdir(parents=True,exist_ok=True)
                    Image.fromarray(array).save(path);entries[field]={'path':str(path.relative_to(args.output)),'sha256':file_sha256(path)}
                    with Image.open(path) as im:
                        if not np.array_equal(np.asarray(im),array):raise AssertionError('Native preview PNG round-trip changed pixels')
                previews.append({'sample_id':item['sample_id'],'source_image':item['image_path'],
                    'source_image_sha256':dataset.manifest.samples[index].image_sha256,'preview_only':True,**entries})
            if j%250==0 or j==len(ids):print(f'[{name}] {j}/{len(ids)} elapsed={time.monotonic()-started:.1f}s',flush=True)
        # Determinism checks actually re-enter the runtime adapter on real data.
        for index in ids[:3]:
            again=dataset[index];again_hash=sha256(dataset.transform.last_owners.astype('<i4').tobytes()).hexdigest()
            if again_hash!=hashes[index]:raise AssertionError('Same epoch/index target is not deterministic')
        if not args.limit_images and stats['words']!=c['expected_words']:
            raise AssertionError('Source words disappeared from qualification denominator')
        if stats['missing_after']>stats['missing_before'] or stats['fragmented_after']>stats['fragmented_before']:
            raise AssertionError('Guard introduces additional missing or fragmented words')
        if epoch is None and not args.limit_images:
            if stats['missing_before']!=8 or stats['fragmented_before']!=275:
                raise AssertionError('Unchanged V2 reference does not replay previous native128 audit')
        summary[name]={'images':len(ids),'stats':dict(stats),'unresolved_reason_counts':dict(reasons),
            'deterministic_replays':min(3,len(ids)),'target_array_hashes':{dataset.manifest.samples[i].sample_id:h for i,h in hashes.items()}}
        write(args.output/'views'/(name+'.json'),{'summary':summary[name],'per_image':records})
        print(json.dumps({'view':name,**summary[name]['stats'],'unresolved_reason_counts':dict(reasons)}),flush=True)
    write(args.output/'native_preview/manifest.json',{'claim_label':'approximate reproduction',
        'contract':protocol['preview_contract'],'native_size':[128,128],'samples':previews})
    report={'status':'SUCCESS','claim_label':'approximate reproduction','result_class':protocol['result_class'],
        'protocol_sha256':file_sha256(args.output/'protocol.json'),'views':summary,'independent_opencv_image_checks':independent,
        'elapsed_seconds':time.monotonic()-started,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'new_training':False,'gpu_inference':False,'saved_rgb_crops_or_weights':0,'unresolved_targets_retained_and_reported':True}
    write(args.output/'results.json',report)
    if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>c['output_budget_bytes']:
        raise RuntimeError('Target qualification output budget exceeded')
    print(json.dumps({k:v for k,v in report.items() if k!='views'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-images',type=int)
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        write(args.output/'failure.json',{'status':'FAILED','error':str(error),'traceback':traceback.format_exc()})
        raise
