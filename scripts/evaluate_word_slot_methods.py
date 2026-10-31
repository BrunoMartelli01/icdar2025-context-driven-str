"""Held-out-TRAIN threshold selection and harmonized paired word-slot evaluation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from models.word_slot_unet import sampling_grid,deterministic_sample
from scripts.run_word_slot_methods import setup,make_model,load_prior,load_delta,reader,memory,digest,checked_cache
from segmentation.word_slot_data import read,write,sha,sample_key,load_rgb,canvas_to_original
from recognition.word_slot_evaluation import measure,aggregate,nms,select

REPO=Path(__file__).resolve().parents[1]


def raw_path(root,arm,phase,sid):return root/'predictions'/arm/phase/(sample_key(sid)+'.json')


def infer(root,config,arms,phases):
    protocol=read(root/'protocol.json');device=config['runtime']['device'];budget=config['evaluation']['ocr_budget']
    recognizer=reader(config);hashes={};all_runtime={}
    for arm in arms:
        model=None;options=config['arms'].get(arm,{})
        if arm.startswith('pixel_'):
            checkpoint_sha=config['input_sha256']['source_checkpoint']
        else:
            checkpoint=root/'branches'/arm/'last.pt'
            if read(checkpoint.parent/'complete.json')['status']!='SUCCESS':raise ValueError('Incomplete branch: '+arm)
            checkpoint_sha=sha(checkpoint);model=make_model(config,arm);load_prior(model,config);load_delta(model,checkpoint,config);model.eval().to(device)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter();new_images=0;ocr_calls=0
        for phase in phases:
            for number,record in enumerate(protocol['index'][phase],1):
                path=raw_path(root,arm,phase,record['sample_id'])
                contract={'checkpoint_sha256':checkpoint_sha,'config_digest':digest(config),'image_sha256':record['image_sha256'],
                          'decoder_source_sha256':sha(REPO/'models/word_slot_unet.py'),'evaluator_sha256':sha(Path(__file__))}
                if path.exists():
                    if read(path)['contract']!=contract:raise ValueError('Prediction cache contract changed')
                    hashes[str(path.relative_to(root))]=sha(path);continue
                cached=checked_cache(root,record)
                rgb=load_rgb(record);tick=time.perf_counter();offsets=None
                with torch.no_grad():
                    if model is None:
                        boxes=cached['target']['prior_boxes'].to(device)[:budget];presence=torch.ones(len(boxes),device=device)
                        ids=list(range(len(boxes)));masks=None
                    else:
                        pyramid=tuple(x.to(device) for x in cached['pyramid'])
                        output=model.decode_pyramid(pyramid,prior_boxes=cached['target']['prior_boxes'].to(device))
                        boxes=output['boxes'];presence=output['presence'].sigmoid()
                        ids=nms(boxes.cpu().tolist(),presence.cpu().tolist(),config['evaluation']['nms_iou'],budget)
                        boxes=boxes[ids];presence=presence[ids]
                        masks=F.interpolate(output['masks'][None,ids],(config['training']['mask_size'],)*2,mode='nearest')[0]
                        if options.get('strip'):offsets=output['strip_offsets'][ids]
                    original_boxes=canvas_to_original(boxes,record['image_wh'])
                    normalized=original_boxes/original_boxes.new_tensor(record['image_wh']*2)
                    grid=sampling_grid(normalized,offsets,height=32,width=128,strength=config['training']['strip_strength'])
                    if arm=='pixel_pil':
                        crops=[]
                        for box in original_boxes.cpu().tolist():
                            x1,y1,x2,y2=box
                            crop_box=(max(0,int(np.floor(x1))),max(0,int(np.floor(y1))),
                                      min(rgb.width,max(int(np.ceil(x2)),int(np.floor(x1))+1)),
                                      min(rgb.height,max(int(np.ceil(y2)),int(np.floor(y1))+1)))
                            crops.append(recognizer.transform(rgb.crop(crop_box)))
                        tensors=torch.stack(crops).to(device) if crops else torch.empty(0,3,32,128,device=device)
                    else:tensors=deterministic_sample(TF.to_tensor(rgb),grid,device)*2-1
                    if len(boxes):
                        probabilities=recognizer.model(tensors).softmax(-1)
                        text,token_probabilities=recognizer.model.tokenizer.decode(probabilities)
                    else:text=[];token_probabilities=[]
                    predictions=[]
                    for i,(hypothesis,prob) in enumerate(zip(text,token_probabilities)):
                        predictions.append({'bbox':original_boxes[i].cpu().tolist(),'text':hypothesis,
                          'confidence':float(prob.prod()),'mean_token_confidence':float(prob.mean()) if len(prob) else 0.,
                          'presence':float(presence[i]),'slot_index':ids[i],
                          'crop_tensor_sha256':hashlib.sha256(tensors[i].cpu().numpy().tobytes()).hexdigest()})
                    torch.cuda.synchronize();seconds=time.perf_counter()-tick
                    if masks is not None:
                        artifact=root/'intermediates'/arm/phase/(sample_key(record['sample_id'])+'.npz')
                        artifact.parent.mkdir(parents=True,exist_ok=True)
                        arrays={'selected_mask_logits':masks.cpu().numpy(),'boxes_canvas':boxes.cpu().numpy()}
                        if offsets is not None:arrays['rgb_grid']=grid.cpu().numpy()
                        np.savez_compressed(artifact,**arrays)
                    write(path,{'contract':contract,'predictions':predictions,'sample_id':record['sample_id'],
                          'new_ocr_crops':len(boxes),'unique_ocr_crops':len({p['crop_tensor_sha256'] for p in predictions}),
                          'seconds':seconds,'ground_truth_used_for_inference':False})
                    hashes[str(path.relative_to(root))]=sha(path);new_images+=1;ocr_calls+=len(boxes)
                if number%50==0 or number==len(protocol['index'][phase]):print('INFER',arm,phase,number,flush=True)
        all_runtime[arm]={'new_images':new_images,'new_ocr_calls':ocr_calls,'seconds':time.perf_counter()-start,
                          'models':['qualified PARSeq full RGB']+(['frozen U-Net prior and trained four-scale word-slot decoder'] if model else []),
                          'precision':'float32','device':device,'segmentation_batch_size':1,'maximum_ocr_batch_size':budget,
                          'ocr_input_shape':[budget,3,32,128],'rgb_sampler_device':'cpu',**memory()}
        del model;gc.collect();torch.cuda.empty_cache()
    write(root/'inference_runtime.json',all_runtime);write(root/'prediction_sha256.json',hashes)
    del recognizer;gc.collect();torch.cuda.empty_cache()


def bootstrap_delta(student,deep,draws,seed):
    rng=np.random.default_rng(seed)
    s=np.array([[r[k] for k in ('tp','predictions','gt_words')] for r in student])
    d=np.array([[r[k] for k in ('tp','predictions','gt_words')] for r in deep])
    values=[]
    for _ in range(draws):
        ids=rng.integers(0,len(s),len(s));a,b=s[ids].sum(0),d[ids].sum(0)
        values.append(2*a[0]/max(a[1]+a[2],1)-2*b[0]/max(b[1]+b[2],1))
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]


def summarize(root,config,arms,phases):
    index=read(root/'protocol.json')['index'];iou=config['evaluation']['iou_threshold'];selections={};tables={};per_image={}
    normalization=config['evaluation'].get('text_normalization','casefold')
    for arm in arms:
        calibration=index['calibration'];sweep=[]
        for presence,confidence in itertools.product(config['evaluation']['presence_thresholds'],config['evaluation']['confidence_thresholds']):
            rows=[]
            for record in calibration:
                predictions=read(raw_path(root,arm,'calibration',record['sample_id']))['predictions']
                rows.append(measure(select(predictions,presence,confidence),record['ground_truth'],iou,normalization))
            metric=aggregate(rows);sweep.append({'presence':presence,'confidence':confidence,**metric})
        chosen=max(sweep,key=lambda r:(r['f1'],-r['fp'],r['tp'],-r['presence'],-r['confidence']))
        selections[arm]={'chosen':chosen,'sweep':sweep,'selection_split':'calibration TRAIN100 only','test_used':False}
        for phase in phases:
            rows=[];deep_rows=[];raw_crops=0
            for record in index[phase]:
                sid=record['sample_id'];raw=read(raw_path(root,arm,phase,sid));raw_crops+=raw['unique_ocr_crops']
                predictions=select(raw['predictions'],chosen['presence'],chosen['confidence'])
                metric=measure(predictions,record['ground_truth'],iou,normalization)
                deep=read(root/'deepsolo'/phase/(sample_key(sid)+'.json'))['metric'];deep_rows.append(deep)
                a,b=set(metric['matched_gt_indices']),set(deep['matched_gt_indices'])
                rows.append({'sample_id':sid,**metric,'both_correct':len(a&b),'unet_only':len(a-b),
                             'deepsolo_only':len(b-a),'neither':len(record['ground_truth'])-len(a|b)})
            metric=aggregate(rows)
            metric.update({k:sum(r[k] for r in rows) for k in ('both_correct','unet_only','deepsolo_only','neither')})
            metric['unique_ocr_crops']=raw_crops;metric['maximum_unique_ocr_per_image']=config['evaluation']['ocr_budget']
            metric['delta_f1_vs_deepsolo']=metric['f1']-aggregate(deep_rows)['f1']
            if phase=='test':
                metric['paired_bootstrap_delta_f1_95_interval']=bootstrap_delta(rows,deep_rows,config['evaluation']['bootstrap_draws'],config['seed'])
                retained=[r for r in rows if r['sample_id'] not in config['evaluation']['known_duplicate_test_ids']]
                metric['known_duplicate_exclusion_sensitivity']=aggregate(retained)
            tables.setdefault(phase,{})[arm]=metric;per_image.setdefault(phase,{})[arm]=rows
    for phase in phases:
        tables[phase]['deepsolo']=aggregate([read(root/'deepsolo'/phase/(sample_key(r['sample_id'])+'.json'))['metric'] for r in index[phase]])
    write(root/'threshold_selection.json',selections);write(root/'word_results.json',tables);write(root/'word_per_image.json',per_image)
    return tables


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate']);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',choices=['infer','summarize'],required=True)
    parser.add_argument('--arms',nargs='+');parser.add_argument('--phases',nargs='+',default=['calibration','evaluation','test'],choices=['calibration','evaluation','test'])
    args=parser.parse_args();config=read(args.config);root=args.output.resolve()
    if REPO/'configs/approximate' not in args.config.resolve().parents or config['reproduction_mode']!=args.mode:raise ValueError('Explicit approximate namespace required')
    if read(root/'protocol.json')['config_digest']!=digest(config):raise ValueError('Changed experiment configuration')
    arms=args.arms or ['pixel_control','pixel_pil',*config['arms']]
    if any(a not in ['pixel_control','pixel_pil',*config['arms']] for a in arms):raise ValueError('Unknown method')
    try:
        if args.stage=='infer':
            hardware=setup(config);write(root/'evaluation_hardware.json',hardware);infer(root,config,arms,args.phases)
        else:summarize(root,config,arms,args.phases)
    except Exception as error:
        write(root/('failure_evaluation_'+str(time.time_ns())+'.json'),{'status':'FAILED','type':type(error).__name__,
              'message':str(error),'traceback':traceback.format_exc(),'oom':isinstance(error,torch.cuda.OutOfMemoryError)})
        raise


if __name__=='__main__':main()
