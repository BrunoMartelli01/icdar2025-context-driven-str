"""APPROXIMATE cap-only OCR ablation from complete frozen native128 contours."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np
from PIL import Image
import torch

from localize.native_repository_contours import remap_native_box
from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, PARSEQ_CHECKPOINT_SHA256,
    PARSEQ_RUNTIME_ARCHIVE_SHA256, ParseqApproxConfig, ParseqApproxRecognizer)
from recognition.scene_text_v1_evaluation import evaluate_images
from scripts.evaluate_cached_totaltext_padding import crop_hash
from scripts.evaluate_totaltext_v1_epoch447_ocr_padding import read_crop, write, seal
from scripts.evaluate_totaltext_v1_findcontours_parseq import _sample_key
from segmentation.data import file_sha256, load_segmentation_manifest
from segmentation.training import seed_everything

REPO=Path(__file__).resolve().parents[1]


def reconstruct(shard):
    variant=shard['variants']['0'];items={}
    for p in variant['proposals']:
        items[p['contour_index']]={'contour_index':p['contour_index'],'bbox_native':p['bbox_native'],'area':p['bbox_area_native']}
    for p in variant['rejected']:
        if p['contour_index'] in items:raise ValueError('Duplicate frozen contour')
        items[p['contour_index']]={'contour_index':p['contour_index'],'bbox_native':p['bbox'],'area':p['bbox_area']}
    if len(items)!=variant['raw_contours'] or set(items)!=set(range(len(items))):
        raise ValueError('Frozen candidates do not cover every contour')
    for p in items.values():
        x0,y0,x1,y1=p['bbox_native']
        if not 0<=x0<x1<=128 or not 0<=y0<y1<=128 or (x1-x0)*(y1-y0)!=p['area']:
            raise ValueError('Invalid cached native geometry')
    return [items[i] for i in sorted(items)]


def select(candidates, image_hw, *, area, padding, limit):
    if limit is not None and (type(limit) is not int or limit<1):
        raise ValueError('Limit must be a positive integer or explicit None')
    output=[]
    for p in candidates:
        if p['area']<area:continue
        x0,y0,x1,y1=p['bbox_native']
        padded=[max(0,x0-padding),max(0,y0-padding),min(128,x1+padding),min(128,y1+padding)]
        output.append({'contour_index':p['contour_index'],'bbox_native':p['bbox_native'],
            'bbox_area_native':p['area'],'padded_bbox_native':padded,
            'bbox':remap_native_box(p['bbox_native'],image_hw),
            'padded_bbox_original':remap_native_box(padded,image_hw),
            'sort_area':(padded[2]-padded[0])*(padded[3]-padded[1])})
    output.sort(key=lambda p:p['sort_area'],reverse=True)
    return [dict(p,rank=i+1) for i,p in enumerate(output[:limit])]


def run(args):
    started=time.monotonic();c=json.loads(args.config.read_text());source=Path(c['source_directory'])
    if args.config.resolve().parent!=REPO/'configs/approximate' or c['reproduction_mode']!='approximate' or c['variant']!='native128-crop-limit-ablation-v1':
        raise ValueError('Explicit isolated APPROXIMATE cap ablation required')
    if args.output.exists():raise FileExistsError('Use a fresh result directory')
    for name,digest in c['source_sha256'].items():
        if file_sha256(source/name)!=digest:raise ValueError('Frozen source changed: '+name)
    publication_path=source.parent/'publication_manifest.json'
    if file_sha256(publication_path)!=c['source_publication_manifest_sha256']:
        raise ValueError('Frozen publication manifest changed')
    published={row['path']:row['sha256'] for row in json.loads(publication_path.read_text())['files']}
    old=json.loads((source/'protocol.json').read_text());old_sha=file_sha256(source/'protocol.json')
    source_result=json.loads((source/'results.json').read_text())
    if source_result['status']!='SUCCESS' or source_result['images_per_pipeline']!=c['expected_images'] or source_result['care_words']!=c['expected_care_words']:
        raise ValueError('Complete frozen source required')
    rt=c['runtime']
    if (rt!=old['runtime'] and any(rt[k]!=old['runtime'][k] for k in rt)):
        raise ValueError('Cached PARSeq runtime differs')
    if not torch.cuda.is_available() or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=rt['cublas_workspace_config']:
        raise ValueError('Qualified CUDA runtime required, no fallback')
    if (old['parseq_checkpoint_sha256']!=PARSEQ_CHECKPOINT_SHA256 or not old['decode_ar'] or old['refine_iters']!=1
        or file_sha256(Path(c['parseq_checkpoint']))!=PARSEQ_CHECKPOINT_SHA256
        or file_sha256(Path(c['parseq_archive']))!=PARSEQ_RUNTIME_ARCHIVE_SHA256):
        raise ValueError('Pinned qualified recognition artifacts required')
    if file_sha256(REPO/'recognition/parseq.py')!=old['code_sha256'][str(REPO/'recognition/parseq.py')]:
        raise ValueError('Recognizer adapter changed since cached OCR')
    torch.set_num_threads(rt['cpu_threads']);torch.set_num_interop_threads(1)
    seed_everything(rt['seed'],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(0);torch.cuda.reset_peak_memory_stats()
    ids=old['sample_ids']
    if args.limit_images:
        if not 1<=args.limit_images<=len(ids):raise ValueError('Invalid smoke limit')
        prioritized=c['smoke_sample_ids']+[sid for sid in ids if sid not in c['smoke_sample_ids']]
        if len(set(prioritized))!=len(ids) or set(prioritized)!=set(ids):raise ValueError('Invalid smoke selection')
        ids=prioritized[:args.limit_images]
    code=[Path(__file__),REPO/'recognition/parseq.py',REPO/'localize/native_repository_contours.py',
        REPO/'recognition/scene_text_v1_evaluation.py',REPO/'scripts/evaluate_totaltext_v1_epoch447_ocr_padding.py']
    protocol={'schema_version':1,'claim_label':c['claim_label'],'reproduction_mode':'approximate','plan':c,
        'result_class':'smoke-test; approximate reproduction' if args.limit_images else 'approximate reproduction; complete TEST300 cap ablation',
        'config_sha256':file_sha256(args.config),'sample_ids':ids,'runtime':rt,
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'numpy':np.__version__,
        'source_protocol_sha256':old_sha,'source_model_states':source_result['model_state_sha256'],
        'code_sha256':{str(p.resolve()):file_sha256(p) for p in code},
        'metric':'micro maximum one-to-one unpadded bboxIoU>=.5 and Unicode-casefold exact; punctuation retained; every prediction/care word counts',
        'selection':'only bbox-area, padded bbox stable rank and cap; no GT or OCR score affects proposals',
        'recognition':'fixed exact-RGB PARSeq readings across caps; qualified old GPU OCR retained, extra crops inferred in RAM',
        'table5_claim':False,'new_segmentation_inference':False}
    seal(args.output/'protocol.json',protocol);protocol_sha=file_sha256(args.output/'protocol.json')
    shards={};unique={};image_paths={};replays=0
    for model in c['models']:
        shards[model]=[]
        for sid in ids:
            key=_sample_key(sid);path=source/(model+'-a0')/'proposals'/(key+'.json')
            cached=json.loads(path.read_text())
            if cached['protocol_sha256']!=old_sha or cached['sample_id']!=sid:raise ValueError('Cached identity/protocol differs')
            if not image_paths:
                manifest=load_segmentation_manifest(Path(old['config']['test_manifest']),require_checksums=True)
                image_paths={s.sample_id:Path(old['config']['test_data_root'])/s.image_path for s in manifest.samples}
            image_path=image_paths[sid]
            if file_sha256(image_path)!=cached['image_sha256']:raise ValueError('Frozen RGB changed')
            with Image.open(image_path) as im:image=np.ascontiguousarray(im.convert('RGB'))
            candidates=reconstruct(cached);variants={}
            previous_by_area={}
            for area in c['area_filters']:
                previous_path=source/(model+f'-a{area}')/'proposals'/(key+'.json')
                if file_sha256(previous_path)!=published[str(previous_path.relative_to(source.parent))]:
                    raise ValueError('Published source shard changed')
                previous_by_area[area]=json.loads(previous_path.read_text())
            for pad in c['native_paddings']:
                proposals=select(candidates,cached['image_hw'],area=0,padding=pad,limit=None)
                for p in proposals:
                    x0,y0,x1,y1=p['padded_bbox_original'];digest=crop_hash(image[y0:y1,x0:x1])
                    p['crop_array_sha256']=digest;p['crop_hw']=[y1-y0,x1-x0]
                    unique.setdefault(digest,(sid,p,cached['image_sha256']))
                variants[str(pad)]={'proposals':proposals}
                for area in c['area_filters']:
                    previous=previous_by_area[area]
                    saved=previous['variants'][str(pad)]['proposals']
                    replay=[p for p in proposals if p['bbox_area_native']>=area][:10]
                    fields=['contour_index','bbox_native','bbox','padded_bbox_original','crop_array_sha256']
                    if [[p[f] for f in fields] for p in replay]!=[[p[f] for f in fields] for p in saved]:
                        raise ValueError('Reconstructed top10 does not match frozen source')
                    replays+=1
            shard={'sample_id':sid,'model':model,'protocol_sha256':protocol_sha,
                'image_hw':cached['image_hw'],'image_sha256':cached['image_sha256'],
                'native_mask_array_sha256':cached['native_mask_array_sha256'],
                'source_shard_sha256':file_sha256(path),'variants':variants}
            seal(args.output/model/'proposals'/(key+'.json'),shard);shards[model].append(shard)
        print(f'[candidates] {model} images={len(ids)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    ocr={};pending=[]
    with (args.output/'ocr.jsonl').open('w',encoding='utf-8') as target:
        for line in (source/'ocr.jsonl').open():
            row=json.loads(line);digest=row['crop_array_sha256']
            if digest not in unique:continue
            if row['protocol_sha256']!=old_sha or digest in ocr:raise ValueError('Invalid duplicate/source OCR')
            new={**row,'protocol_sha256':protocol_sha,'reuse_source_protocol_sha256':old_sha,
                'reuse_source_line_sha256':sha256(line.encode()).hexdigest()}
            ocr[digest]=new;target.write(json.dumps(new,ensure_ascii=False)+'\n')
    reused=len(ocr)
    pending=[(digest,item) for digest,item in unique.items() if digest not in ocr]
    print(f'[OCR] unique={len(unique)} reused={reused} new={len(pending)}',flush=True)
    if pending:
        recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(c['parseq_source']),
            checkpoint=Path(c['parseq_checkpoint']),device='cuda:0',runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
            runtime_source_archive=Path(c['parseq_archive'])))
        prev,cached_rgb=None,None
        try:
            for start in range(0,len(pending),rt['parseq_batch_size']):
                batch=pending[start:start+rt['parseq_batch_size']];crops=[]
                for digest,(sid,proposal,image_sha) in batch:
                    if sid!=prev:
                        if file_sha256(image_paths[sid])!=image_sha:raise ValueError('RGB changed before OCR')
                        with Image.open(image_paths[sid]) as im:cached_rgb=np.ascontiguousarray(im.convert('RGB'))
                        prev=sid
                    crops.append(read_crop(cached_rgb,proposal))
                values=recognizer.recognize_candidate_batch(crops)
                if len(values)!=len(batch):raise RuntimeError('OCR cardinality mismatch')
                with (args.output/'ocr.jsonl').open('a',encoding='utf-8') as target:
                    for (digest,_),value in zip(batch,values):
                        row={'protocol_sha256':protocol_sha,'crop_array_sha256':digest,'raw_text':value.raw_text,
                            'sequence_confidence_native':value.sequence_confidence_native,'token_ids':value.token_ids}
                        ocr[digest]=row;target.write(json.dumps(row,ensure_ascii=False)+'\n')
                if start%(rt['parseq_batch_size']*32)==0 or start+len(batch)==len(pending):
                    print(f'[OCR] new={start+len(batch)}/{len(pending)}',flush=True)
        finally:recognizer.release()
    gt_data=json.loads((source/'ground_truth.json').read_text());gt={sid:gt_data['ground_truth'][sid] for sid in ids}
    care=sum(map(len,gt.values()))
    if not args.limit_images and care!=c['expected_care_words']:raise ValueError('Care denominator differs')
    seal(args.output/'ground_truth.json',{'ground_truth':gt,'annotation_provenance':gt_data['annotation_provenance']})
    results={}
    for model,records in shards.items():
        results[model]={}
        for area in c['area_filters']:
            results[model][str(area)]={}
            for cap in c['limits']:
                label='all' if cap is None else str(cap);summary={}
                for pad in c['native_paddings']:
                    rows=[]
                    for record in records:
                        selected=[p for p in record['variants'][str(pad)]['proposals'] if p['bbox_area_native']>=area][:cap]
                        rows.append({'sample_id':record['sample_id'],'ground_truth':gt[record['sample_id']],
                            'predictions':[{'rank':i+1,'bbox':p['bbox'],'text':ocr[p['crop_array_sha256']]['raw_text'],
                                'confidence':ocr[p['crop_array_sha256']]['sequence_confidence_native']} for i,p in enumerate(selected)]})
                    measured=evaluate_images(rows,text_available=True);tp=measured['recognition']['exact_casefold_spatial_one_to_one'];n=measured['geometry']['retained_proposals']
                    measured['exact_end_to_end']={'tp':tp,'predictions':n,'gt':care,'precision':tp/n if n else 0,
                        'recall':tp/care if care else 0,'f1':2*tp/(n+care) if n+care else 0}
                    measured['images_above_cap']=sum(sum(p['bbox_area_native']>=area for p in s['variants'][str(pad)]['proposals'])>cap for s in records) if cap is not None else 0
                    summary[str(pad)]=measured
                    if cap==10 and not args.limit_images:
                        old_metrics=source_result['pipelines'][model+f'-a{area}']['summary'][str(pad)]
                        if measured['geometry']!=old_metrics['geometry'] or measured['recognition']!=old_metrics['recognition']:
                            raise ValueError('Top10 OCR baseline does not exactly replay')
                results[model][str(area)][label]=summary
            for pad in c['native_paddings']:
                counts=[results[model][str(area)]['all' if cap is None else str(cap)][str(pad)]['exact_end_to_end']['tp'] for cap in c['limits']]
                if counts!=sorted(counts):raise ValueError('Nested caps unexpectedly lose correct words with fixed OCR')
            print(json.dumps({'model':model,'area':area,'pad1':{cap:v['1']['exact_end_to_end'] for cap,v in results[model][str(area)].items()}}),flush=True)
    report={'status':'SUCCESS','claim_label':c['claim_label'],'result_class':protocol['result_class'],'protocol_sha256':protocol_sha,
        'images':len(ids),'care_words':care,'results':results,'unique_rgb_crops':len(unique),'reused_ocr_crops':reused,
        'new_ocr_crops':len(pending),'exact_top10_proposal_replays':replays,'elapsed_seconds':time.monotonic()-started,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),
        'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'saved_crops_or_weights':0,'out_of_memory':False}
    write(args.output/'results.json',report)
    if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file())>c['output_budget_bytes']:raise RuntimeError('Numeric output budget exceeded')
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-images',type=int)
    args=p.parse_args()
    try:run(args)
    except Exception as error:
        ready=torch.cuda.is_initialized()
        write(args.output/'failure.json',{'status':'FAILED','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
            'out_of_memory':isinstance(error,(MemoryError,torch.cuda.OutOfMemoryError)),'device':'cuda:0','precision':'float32',
            'model_set':['qualified_PARSeq'],'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if ready else None,
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved() if ready else None})
        raise
