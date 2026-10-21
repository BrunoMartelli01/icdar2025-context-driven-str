"""Exploratory TEST battery: OCR-supported filtering, crop views and multiscale YOLO."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
from pathlib import Path
import random
import sys
import time
import traceback
import numpy as np
from PIL import Image
import torch

from recognition.parseq import (PARSEQ_ARCHIVE_SOURCE, ParseqApproxConfig, ParseqApproxRecognizer,
    PARSEQ_CHECKPOINT_SHA256, PARSEQ_RUNTIME_ARCHIVE_SHA256)
from recognition.yolo_word_refinement import (arm_predictions, crop_box, nms, ocr_quality,
                                              tile_windows, translated_slots)
from scripts.analyze_totaltext_crop_word_groups import freeze
from scripts.crop_clip_runtime import _cuda_peak, array_sha, cached, read_image, sha, write_json
from scripts.evaluate_totaltext_yolo_direct_parseq import direct_crop, metrics
from scripts.evaluate_totaltext_yolo_word_slots import boxes_to_slots, read_json, validate_checkpoint
from scripts.run_crop_group_word_selection import evaluate_predictions
from segmentation.data import load_segmentation_manifest

REPO=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=REPO/'configs/approximate/totaltext_yolo_improvements_v1.json'
DEFAULT_PATHS=REPO/'configs/approximate/totaltext_yolo_improvements_local_paths_v1.json'
MANIFEST_PATHS=REPO/'configs/approximate/totaltext_independent_detector_slots_local_paths_v1.json'
CODE=('scripts/evaluate_totaltext_yolo_improvements.py','recognition/yolo_word_refinement.py',
      'recognition/parseq.py','recognition/group_word_selection.py',
      'scripts/run_crop_group_word_selection.py','recognition/scene_text_v1_evaluation.py',
      'localize/independent_word_slots.py','scripts/evaluate_totaltext_yolo_word_slots.py',
      'scripts/evaluate_totaltext_yolo_direct_parseq.py','scripts/crop_clip_runtime.py')


def validate_config(c):
    if (c.get('reproduction_mode')!='approximate' or c.get('dataset')!='total_text'
        or c.get('split')!='TEST' or c.get('exploratory_test_analysis') is not True
        or c.get('table_5_claim') is not False or c.get('schema_version')!=1
        or c.get('reference_score') not in c.get('score_controls',[])
        or any(not 0<x<1 for x in c['score_controls']+c['joint_controls'])
        or not 0<c['merge_iou']<1 or not 0<c['matching_iou']<=1
        or not 0<=c['tile_overlap']<1 or not 0<c['tile_confidence_floor']<1
        or not 0<c['variant_detector_floor']<1
        or any(not -.5<x<.5 or x==0 for x in c['crop_fractions'])
        or any(type(c[k]) is not int or c[k]<=0 for k in ('batch_size','cpu_threads','highres_size','tile_size'))
        or type(c['seed']) is not int or not 0<=c['seed']<2**32):
        raise ValueError('invalid isolated approximate YOLO improvement config')


def peak_from_shards(shards):
    return {k:max((s.get('model_peak',{}).get(k) or 0 for s in shards),default=0) or None
            for k in ('peak_allocated_bytes','peak_reserved_bytes')}


def evaluate(output, base, source, ids, cfg, include_new):
    totals={}; gt_count=0; gt_hashes={}
    for sid in ids:
        key=sid.replace(':','__')
        baseline=read_json(base/'ocr'/f'{key}.json')['rows']
        families={'base640':baseline}
        if include_new:
            for family in ('high960','tiles640'):
                families[family]=read_json(output/'ocr'/family/f'{key}.json')['rows']
        arms={f'{family}/{name}':rows for family,rows in families.items()
              for name,rows in arm_predictions(rows,cfg).items()}
        if include_new:
            variants=read_json(output/'crop_views'/f'{key}.json')['rows']
            by_id={r['candidate_id']:r for r in baseline}
            views={cid:[row] for cid,row in by_id.items()}
            for row in variants:
                views[row['original_candidate_id']].append(row)
            best=[]
            for cid,rows in views.items():
                # Baseline wins equal-confidence ties; geometry remains the original box.
                selected=max(rows,key=ocr_quality)
                best.append({**selected,'candidate_id':cid,'bbox':by_id[cid]['bbox']})
            for name,rows in arm_predictions(best,cfg).items():
                arms[f'crop_best/{name}']=rows
            for fraction in cfg['crop_fractions']:
                changed={r['original_candidate_id']:r for r in variants if r['fraction']==fraction}
                fixed=[{**changed.get(r['candidate_id'],r),'bbox':r['bbox']} for r in baseline]
                arms[f'crop_fixed_{fraction}/raw_{cfg["reference_score"]}']=[r for r in fixed if r['detector_score']>=cfg['reference_score']]
        gtpath=source/'evaluation'/f'{key}.json'
        gt=read_json(gtpath)['ground_truth'];gt_hashes[sid]=sha(gtpath);gt_count+=len(gt)
        scored={}
        for name,rows in arms.items():
            result=evaluate_predictions(rows,gt,cfg['matching_iou'])
            totals.setdefault(name,Counter()).update(outputs=len(rows),correct=result['correct'])
            scored[name]={**result,'selected_candidate_ids':[r['candidate_id'] for r in rows]}
        freeze(output/('evaluation' if include_new else 'cached_evaluation')/f'{key}.json',
               {'sample_id':sid,'gt_words':len(gt),'arms':scored})
    result={'schema_version':1,'reproduction_mode':'approximate','exploratory_test_analysis':True,
            'result_class':'APPROXIMATE_TEST_DIAGNOSTIC' if len(ids)==300 else 'SMOKE_TEST' if len(ids)==1 else 'PARTIAL_DATASET',
            'images':len(ids),'gt_words':gt_count,'plan_sha256':sha(output/'resolved_configuration.json'),
            'summary':{k:metrics(r['correct'],r['outputs'],gt_count) for k,r in totals.items()},
            'gt_shards_sha256':gt_hashes,
            'metric':'micro one-to-one exact casefold/P-S-stripped word and original detection bbox IoU>=0.5; empty predictions retained except explicitly named nonempty/joint arms',
            'limitation':'local bbox proxy; ignore-aware polygon protocol is separate; TEST analyzed to choose method families, no untouched validation claim'}
    if include_new:
        detector_shards=[read_json(output/'detector'/f'{sid.replace(":","__")}.json') for sid in ids]
        ocr_shards=[read_json(output/'ocr'/family/f'{sid.replace(":","__")}.json')
                    for sid in ids for family in ('high960','tiles640')]
        view_shards=[read_json(output/'crop_views'/f'{sid.replace(":","__")}.json') for sid in ids]
        result['execution']={'detector_recorded_seconds':sum(s['seconds'] for s in detector_shards),
            'ocr_recorded_seconds':sum(s['seconds'] for s in ocr_shards+view_shards),
            'new_ocr':sum(s['new_ocr'] for s in ocr_shards+view_shards),
            'detector_peak':peak_from_shards(detector_shards),
            'parseq_peak':peak_from_shards(ocr_shards+view_shards)}
    target=output/('results.json' if include_new else 'cached_results.json')
    freeze(target,result)
    print(json.dumps({'report':str(target),'summary':result['summary']}),flush=True)
    return result


def run(args):
    for path in (args.config,args.paths):
        if path.resolve().parent!=REPO/'configs/approximate':raise ValueError('isolated approximate config required')
    cfg=read_json(args.config);validate_config(cfg);paths=read_json(args.paths)
    if paths['reproduction_mode']!='approximate':raise ValueError('approximate paths required')
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; choose --device cpu explicitly')
    if args.limit is not None and args.limit<1:raise ValueError('--limit must be positive')
    base=Path(paths['baseline_run']);bp=read_json(base/'resolved_configuration.json');br=read_json(base/'results.json')
    if br['plan_sha256']!=sha(base/'resolved_configuration.json') or br['images']!=300:raise RuntimeError('baseline contract differs')
    detector=Path(bp['detector_run']);dp=read_json(detector/'resolved_configuration.json')
    if sha(detector/'resolved_configuration.json')!=bp['detector_plan_sha256']:raise RuntimeError('detector plan differs')
    source=Path(bp['gt_source']);sp=read_json(source/'resolved_configuration.json');p=sp['paths']
    if sha(source/'resolved_configuration.json')!=bp['source_plan_sha256']:raise RuntimeError('source changed')
    mp=read_json(MANIFEST_PATHS);manifest=Path(mp['test_manifest'])
    if sha(manifest)!=bp['manifest_sha256']:raise RuntimeError('manifest changed')
    samples={s.sample_id:s for s in load_segmentation_manifest(manifest,require_checksums=True).samples}
    ids=bp['sample_ids'][:args.limit] if args.limit else bp['sample_ids']
    for sid in ids:
        key=sid.replace(':','__');old=read_json(base/'ocr'/f'{key}.json')
        receipt=read_json(base/'evaluation'/f'{key}.json')
        if (old['contract']['plan_sha256']!=sha(base/'resolved_configuration.json')
            or receipt['ocr_sha256']!=sha(base/'ocr'/f'{key}.json')
            or sha(detector/'detector'/f'{key}.json')!=bp['detector_shards_sha256'][sid]):
            raise RuntimeError(f'baseline OCR/detector receipt differs: {sid}')
    output=args.output.resolve() if args.output else Path(paths['output_root'])/args.device.replace(':','_')/('full' if args.limit is None else f'limit{args.limit}')
    if output==REPO or REPO in output.parents:raise ValueError('results must stay outside repo')
    checkpoint=Path(dp['checkpoint']['path']);identity=validate_checkpoint(checkpoint,dp['config']['checkpoint_source'])
    if sha(Path(p['parseq_checkpoint']))!=PARSEQ_CHECKPOINT_SHA256 or sha(Path(p['parseq_archive']))!=PARSEQ_RUNTIME_ARCHIVE_SHA256:raise RuntimeError('PARSeq artifacts differ')
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.set_num_threads(cfg['cpu_threads'])
    plan={'schema_version':1,'reproduction_mode':'approximate','config':cfg,'config_sha256':sha(args.config),
          'paths_sha256':sha(args.paths),'baseline_run':str(base),'baseline_plan_sha256':sha(base/'resolved_configuration.json'),
          'baseline_ocr_sha256':{sid:sha(base/'ocr'/f'{sid.replace(":","__")}.json') for sid in ids},
          'baseline_detector_sha256':{sid:sha(detector/'detector'/f'{sid.replace(":","__")}.json') for sid in ids},
          'sample_ids':ids,'manifest_sha256':sha(manifest),'yolo_checkpoint':identity,
          'parseq_checkpoint_sha256':PARSEQ_CHECKPOINT_SHA256,'parseq_archive_sha256':PARSEQ_RUNTIME_ARCHIVE_SHA256,
          'device':args.device,'precision':'float32','batch_size':cfg['batch_size'],'cpu_threads':cfg['cpu_threads'],
          'hardware_deviation':'CPU FP32' if args.device=='cpu' else 'sequential GPU FP32, reduced batch',
          'gpu':torch.cuda.get_device_name(0) if args.device.startswith('cuda') else None,
          'python':sys.version,'torch':torch.__version__,'ultralytics':importlib.metadata.version('ultralytics'),
          'code_sha256':{name:sha(REPO/name) for name in CODE},
          'ground_truth_use':'baseline diagnostic motivated method families; no GT enters detection, crop generation, OCR or filtering'}
    freeze(output/'resolved_configuration.json',plan);ph=sha(output/'resolved_configuration.json')
    if args.stage=='cached':return evaluate(output,base,source,ids,cfg,False)
    yolo=None;parseq=None;stage='detector';sid=None
    try:
        for position,sid in enumerate(ids,1):
            key=sid.replace(':','__');contract={'plan_sha256':ph,'sample_id':sid}
            existing=cached(output/'detector'/f'{key}.json',contract)
            if existing is None:
                if yolo is None:
                    from ultralytics import YOLO
                    if plan['ultralytics']!=dp['ultralytics']:raise RuntimeError('YOLO runtime version changed')
                    if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats(args.device)
                    yolo=YOLO(str(checkpoint))
                image,imhash=read_image(Path(p['data_root']),samples[sid],samples[sid].image_sha256)
                wh=(image.shape[1],image.shape[0]);bgr=np.ascontiguousarray(image[:,:,::-1]);start=time.perf_counter()
                def infer(pixels,size,floor,prefix,window):
                    r=yolo.predict(pixels,imgsz=size,conf=floor,iou=dp['config']['inference']['nms_iou'],
                                   max_det=dp['config']['inference']['max_detections'],device=args.device,half=False,verbose=False)[0]
                    slots,rejected=boxes_to_slots(r.boxes.xyxy.cpu().numpy(),r.boxes.conf.cpu().numpy(),r.boxes.cls.cpu().numpy(),
                                                  (pixels.shape[1],pixels.shape[0]),floor)
                    if rejected:raise RuntimeError('new detector produced out-of-image boxes')
                    return translated_slots(slots,window,wh,prefix)
                high=infer(bgr,cfg['highres_size'],dp['config']['inference']['confidence_floor'],f'{sid}:high',[0,0,*wh])
                original=read_json(detector/'detector'/f'{key}.json')['slots']
                tiled=translated_slots(original,[0,0,*wh],wh,f'{sid}:base')
                windows=tile_windows(wh,cfg['tile_overlap'])
                for index,window in enumerate(windows):
                    x1,y1,x2,y2=window
                    tiled.extend(infer(np.ascontiguousarray(bgr[y1:y2,x1:x2]),cfg['tile_size'],cfg['tile_confidence_floor'],f'{sid}:tile{index}',window))
                tiled=nms(tiled,cfg['merge_iou'])
                if args.device.startswith('cuda'):torch.cuda.synchronize(args.device)
                existing={'contract':contract,'sample_id':sid,'image_sha256':imhash,'image_wh':wh,
                          'families':{'high960':high,'tiles640':tiled},'tile_windows':windows,
                          'seconds':time.perf_counter()-start,'model_peak':_cuda_peak(args.device),'ground_truth_use':'none'}
                write_json(output/'detector'/f'{key}.json',existing)
            if position%25==0 or position==len(ids):print(f'[YOLO improvements] detector {position}/{len(ids)}',flush=True)
        del yolo;yolo=None;gc.collect()
        if args.device.startswith('cuda'):torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(args.device)
        stage='parseq'
        for position,sid in enumerate(ids,1):
            key=sid.replace(':','__');detpath=output/'detector'/f'{key}.json';d=read_json(detpath)
            image=None
            baseline=read_json(base/'ocr'/f'{key}.json')['rows']
            by_pixels={r['crop_rgb_sha256']:{**r,'ocr_execution':{'device':bp['device'],
                'precision':bp['precision'],'batch_size':bp['batch_size'],'source':'frozen baseline'}} for r in baseline}
            targets=[(family,d['families'][family],output/'ocr'/family/f'{key}.json') for family in ('high960','tiles640')]
            variants=[]
            for row in baseline:
                if row['detector_score']<cfg['variant_detector_floor']:continue
                for fraction in cfg['crop_fractions']:
                    variants.append({'candidate_id':f"{row['candidate_id']}:view{fraction}",
                        'original_candidate_id':row['candidate_id'],'bbox':row['bbox'],
                        'crop_bbox':crop_box(row['bbox'],tuple(d['image_wh']),fraction),
                        'fraction':fraction,'detector_score':row['detector_score']})
            targets.append(('views',variants,output/'crop_views'/f'{key}.json'))
            for family,proposals,target in targets:
                contract={'plan_sha256':ph,'detector_sha256':sha(detpath),'family':family}
                if cached(target,contract) is not None:continue
                if image is None:image,_=read_image(Path(p['data_root']),samples[sid],d['image_sha256'])
                rows=[];pending=[];start=time.perf_counter()
                for row in proposals:
                    crop=direct_crop(image,{'bbox':row.get('crop_bbox',row['bbox'])});h=array_sha(crop)
                    metadata={**row,'crop_rgb_sha256':h,'crop_shape':list(crop.shape)}
                    if h in by_pixels:
                        old=by_pixels[h];rows.append({**old,**metadata})
                    else:pending.append((metadata,crop))
                for offset in range(0,len(pending),cfg['batch_size']):
                    group=pending[offset:offset+cfg['batch_size']]
                    if parseq is None:
                        parseq=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(p['parseq_source']),
                            runtime_source_archive=Path(p['parseq_archive']),runtime_source_type=PARSEQ_ARCHIVE_SOURCE,
                            checkpoint=Path(p['parseq_checkpoint']),device=args.device))
                    values=parseq.recognize_candidate_batch([crop for _,crop in group])
                    if len(values)!=len(group):raise RuntimeError('PARSeq cardinality differs')
                    for (metadata,_),value in zip(group,values):
                        record={**metadata,**value.to_dict(),'ocr_execution':{'device':args.device,
                            'precision':'float32','batch_size':cfg['batch_size'],'source':'new inference'}}
                        rows.append(record);by_pixels[metadata['crop_rgb_sha256']]=record
                byid={r['candidate_id']:r for r in rows}
                if len(byid)!=len(proposals):raise RuntimeError('OCR proposal IDs differ')
                ordered=[byid[r['candidate_id']] for r in proposals]
                write_json(target,{'contract':contract,'sample_id':sid,'rows':ordered,
                                   'new_ocr':len(pending),'seconds':time.perf_counter()-start,
                                   'model_peak':_cuda_peak(args.device),'ground_truth_use':'none'})
            if position%25==0 or position==len(ids):print(f'[YOLO improvements] OCR {position}/{len(ids)}',flush=True)
    except Exception as exc:
        write_json(output/'failure.json',{'status':'FAILED','stage':stage,'sample_id':sid,
            'error':str(exc),'traceback':traceback.format_exc(),'out_of_memory':isinstance(exc,torch.cuda.OutOfMemoryError),
            'device':args.device,'precision':'float32','batch_size':cfg['batch_size'],'model_peak':_cuda_peak(args.device)})
        raise
    finally:
        if parseq is not None:parseq.release()
    return evaluate(output,base,source,ids,cfg,True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--paths',type=Path,default=DEFAULT_PATHS)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--stage',choices=('cached','all'),default='all')
    run(parser.parse_args())


if __name__=='__main__':main()
