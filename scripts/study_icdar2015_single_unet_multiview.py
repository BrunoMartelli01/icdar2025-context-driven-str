"""Fixed-VAL multiscale/flip study of one frozen U-Net; never reads TEST."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,Subset

from recognition.icdar2015_word_f1 import sample_counts,aggregate_counts
from segmentation.data import SegmentationDataset,file_sha256
from segmentation.evaluate import _prepare_cuda_memory_tracking
from segmentation.icdar2015_improvement_runtime import REPO,load_training_population,references,source_hashes
from segmentation.icdar2015_improvements import make_transform,binary_proposals,valid_canvas
from segmentation.icdar2015_operating_policy import component_crop
from segmentation.icdar2015_seed_partition import separated_mask
from segmentation.icdar2015_multiview import emitted
from segmentation.icdar2015_target70 import load_profile,ValidationOCR
from segmentation.icdar2015_target70_selection import verify_frozen_selection,load_selected_model
from segmentation.training import seed_everything,runtime_environment
from scripts.calibrate_icdar2015_target70_operating_policy import recognize
from scripts.evaluate_icdar2015_totaltext_transfer_word_f1 import crop_hash
from scripts.train_icdar2015_totaltext_transfer import write_json


def run(a):
    c=json.loads(a.config.read_text())
    if a.config.resolve().parent.name!='approximate' or c['reproduction_mode']!='approximate' or c['claim_label']!='approximate reproduction':
        raise ValueError('Explicit approximate multiview study required')
    for field,path in [('source_selection_sha256',a.selection),('source_selection_profile_sha256',a.selection_profile),
            ('native_baseline_sha256',a.native_baseline),('seed_baseline_sha256',a.seed_baseline)]:
        if file_sha256(path)!=c[field]:raise ValueError('Pinned source artifact changed: '+field)
    selection=verify_frozen_selection(a.selection,a.selection_profile)
    if selection['selected']['checkpoint_sha256']!=c['source_checkpoint_sha256']:raise ValueError('One qualified c467 model required')
    config=load_profile(REPO/selection['config']['training_profile']);manifest,_,val=load_training_population(config,a.data_root)
    refs=references(manifest,a.data_root,val)
    if len(refs)!=100 or sum(len(r['ground_truth']) for r in refs.values())!=430:raise ValueError('Fixed VAL100/430care required')
    device=torch.device(a.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Explicit CUDA required; no substitution')
    seed_everything(config['training']['seed'],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_selected_model(selection).to(device).eval();_prepare_cuda_memory_tracking(device)
    images={}
    for i in val:
        sample=manifest.samples[i];path=a.data_root/sample.image_path
        if file_sha256(path)!=sample.image_sha256:raise ValueError('Validation RGB changed')
        with Image.open(path) as im:images[sample.sample_id]=np.array(im.convert('RGB'))
    contract=dict(config=c,config_sha256=file_sha256(a.config),source_checkpoint_sha256=c['source_checkpoint_sha256'],
        data_root=str(a.data_root.resolve()),source_selection=str(a.selection.resolve()),source_selection_profile=str(a.selection_profile.resolve()),
        device=a.device,precision='float32',model_count=1,segmentation_batch_size=1,ocr_device='cpu',parseq_batch_size=config['parseq_batch_size'],
        source_sha256=source_hashes('scripts/study_icdar2015_single_unet_multiview.py','segmentation/icdar2015_multiview.py',
            'segmentation/icdar2015_seed_partition.py','segmentation/icdar2015_operating_policy.py','segmentation/icdar2015_target70_selection.py',
            'localize/word_multitask.py','localize/crop_recovery.py'))
    write_json(a.output_dir/'run_contract.json',contract);ocr=ValidationOCR(a,config);streams={};view_records=[]
    for view in c['views']:
        name=view['name']+('_flip' if view['horizontal_flip'] else '');shape=view['hw'];torch.cuda.reset_peak_memory_stats(device);start=time.monotonic()
        dataset=SegmentationDataset(manifest,a.data_root,transform=make_transform(config,shape,augment=False),verify_checksums=False)
        loader=DataLoader(Subset(dataset,val),batch_size=1,shuffle=False,num_workers=2);probabilities={};masks={}
        with torch.inference_mode():
            for batch in loader:
                key=batch['sample_id'][0];x=batch['image'].to(device)
                if view['horizontal_flip']:x=x.flip(3)
                logits=model(x)
                if view['horizontal_flip']:logits=logits.flip(3)
                p=logits.sigmoid().cpu().numpy()[0,0]
                if p.shape!=tuple(n//2 for n in shape) or not np.isfinite(p).all():raise RuntimeError('Invalid native probability map')
                canvas=valid_canvas(refs[key]['image_wh'],shape);cc=(p>=c['probability_threshold'])&canvas
                seed,_=separated_mask(p,canvas,support_threshold=c['seed_support'],seed_threshold=c['seed_threshold'])
                path=a.output_dir/'views'/name/'masks'/(key.replace(':','_')+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,probability=p,prediction_cc=cc,prediction_seed=seed)
                probabilities[key]=str(path);masks[key]=(cc,seed)
        if set(masks)!=set(refs):raise RuntimeError('Incomplete view population')
        for post in c['postprocessing']:
            stream=name+'/'+post['name'];rows=[];crops=[]
            for key,ref in refs.items():
                prediction=masks[key][1 if post['name']=='seed' else 0]
                proposals,labels=binary_proposals(prediction,ref['image_wh'],shape)
                for p in proposals:
                    crop=component_crop(images[key],labels,p,shape,post['crop'])
                    if not crop.size:raise ValueError('Empty crop; no hypothesis omissions')
                    p.update(crop_rgb_sha256=crop_hash(crop),stream_id=stream,view=name,crop_method=post['crop'])
                    crops.append(crop)
                rows.append(dict(sample_id=key,ground_truth=ref['ground_truth'],predictions=proposals,
                    mask_file=probabilities[key],mask_sha256=file_sha256(Path(probabilities[key]))))
            recognize(crops,ocr)
            for row in rows:
                for p in row['predictions']:
                    path=ocr.cache/(p['crop_rgb_sha256']+'.json');out=json.loads(path.read_text())
                    p.update(text=out['raw_text'],sequence_confidence=out['sequence_confidence_native'],
                        mean_token_log_probability=out['mean_token_log_probability'],ocr_sha256=file_sha256(path))
                row['counts']=sample_counts(row['predictions'],row['ground_truth'],iou_threshold=c['matching_iou_threshold'])
            raw=aggregate_counts([r['counts'] for r in rows])
            if stream=='native/cc' and raw!=json.loads(a.native_baseline.read_text())['selected']['raw_aggregate']:
                raise RuntimeError('Qualified native axis2 raw replay differs')
            if stream=='native/seed' and raw!=json.loads(a.seed_baseline.read_text())['selected']['raw']:
                raise RuntimeError('Qualified native confidence-seed raw replay differs')
            path=a.output_dir/'streams'/(stream.replace('/','_')+'.json');write_json(path,rows)
            streams[stream]=dict(per_image=rows,raw=raw,per_image_file=str(path.relative_to(a.output_dir)),per_image_sha256=file_sha256(path))
            print(f'{stream}: rawWordF1={raw["alphanumeric_casefold"]["f1"]:.6f}, proposals={raw["predictions"]}',flush=True)
        view_records.append(dict(view=name,shape=shape,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),elapsed_seconds=time.monotonic()-start))
        write_json(a.output_dir/'views.json',view_records);write_json(a.output_dir/'live_status.json',dict(status='COLLECTING',completed_views=len(view_records),total_views=len(c['views'])))
    del model;torch.cuda.empty_cache();candidates=[]
    by_stream={s:{r['sample_id']:r['predictions'] for r in v['per_image']} for s,v in streams.items()}
    for pool,ids in c['pools'].items():
        combined={key:[p for stream in ids for p in by_stream[stream][key]] for key in refs}
        for field,thresholds in c['confidence_grids'].items():
            for threshold in thresholds:
                for length in c['minimum_lengths']:
                    for overlap in c['nms_iou_thresholds']:
                        policy=dict(field=field,threshold=threshold,minimum_length=length,nms_iou=overlap)
                        rows=[]
                        for key,ref in refs.items():
                            keep=emitted(combined[key],**policy)
                            rows.append(dict(sample_id=key,ground_truth=ref['ground_truth'],predictions=keep,
                                counts=sample_counts(keep,ref['ground_truth'],iou_threshold=c['matching_iou_threshold'])))
                        aggregate=aggregate_counts([r['counts'] for r in rows]);index=len(candidates)
                        path=a.output_dir/'candidates'/f'{index:03d}.json';write_json(path,rows)
                        candidates.append(dict(pool=pool,streams=ids,policy=policy,aggregate=aggregate,index=index,
                            per_image_file=str(path.relative_to(a.output_dir)),per_image_sha256=file_sha256(path)))
        write_json(a.output_dir/'candidates.json',candidates)
        best=max([x for x in candidates if x['pool']==pool],key=lambda x:x['aggregate']['alphanumeric_casefold']['f1'])
        print(f'pool={pool}: best emitted F1={best["aggregate"]["alphanumeric_casefold"]["f1"]:.6f}',flush=True)
    chosen=max(candidates,key=lambda x:(x['aggregate']['alphanumeric_casefold']['f1'],x['aggregate']['alphanumeric_casefold']['recall'],-len(x['streams']),-x['index']))
    source_checks={k:{x:y for x,y in v.items() if x!='per_image'} for k,v in streams.items()}
    write_json(a.output_dir/'results.json',dict(status='SUCCESS',reproduction_mode='approximate',claim_label='approximate reproduction',
        result_class='complete fixed VAL100 single-U-Net multiview calibration; not TEST',run_contract=contract,candidates=candidates,selected=chosen,
        streams=source_checks,view_records=view_records,failed_samples=0,oom=False,test_data_used=False,
        deepsolo_images=0,total_images=100,deepsolo_percentage=0.,runtime=runtime_environment()))
    write_json(a.output_dir/'live_status.json',dict(status='SUCCESS',completed_views=len(view_records),total_views=len(c['views'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['config','selection','selection-profile','native-baseline','seed-baseline','data-root','parseq-source','parseq-source-archive','parseq-checkpoint','output-dir']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--device',required=True);a=p.parse_args()
    if a.output_dir.exists():raise ValueError('Preserve existing study output')
    try:run(a)
    except Exception as e:
        write_json(a.output_dir/'failure.json',dict(status='FAILED',error_type=type(e).__name__,error=str(e),oom=isinstance(e,torch.cuda.OutOfMemoryError),
            peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            peak_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None))
        raise


if __name__=='__main__':main()
