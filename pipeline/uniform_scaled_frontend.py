"""Isolated resolution experiment; original qualified frontend remains unchanged."""

import gc
from pathlib import Path
import time

import numpy as np
from PIL import Image

from scripts.run_uniform_unet_training import read,write
from segmentation.uniform_unet_fusion import file_hash
from scripts.icdar_budget_data import key


def scaled_policy(policy,canvas):
    if type(canvas) is not int or canvas<32 or canvas%32:
        raise ValueError('Explicit square canvas divisible by32 required')
    return dict(policy,native_size=canvas//2,minimum_transcript_probability=0.)


def export(training_root,records,output,canvas):
    import torch
    from scripts.export_uniform_unet import checked_training,model,rows_with_sources
    from scripts.train_icdar13_word_refit import recognizer,read_rows
    from scripts.run_icdar13_dense_instances import runtime
    from segmentation.icdar2015_improvements import make_transform
    plan,complete=checked_training(training_root);c=plan['config'];frozen=read(training_root/'emission_selection/frozen.json')
    if frozen['checkpoint_sha256']!=complete['checkpoint_sha256'] or frozen['test_used_for_selection']:raise ValueError('Qualified original crop policy required')
    settings=scaled_policy(frozen['selected']['policy'],canvas);runtime(c);started=time.perf_counter()
    net,ocr=model(plan,training_root),recognizer(plan);loading=time.perf_counter()-started;encoded={}
    hook=net.encoder.layer4.register_forward_hook(lambda module,args,value:encoded.update(e4=value))
    transform=make_transform(c,[canvas,canvas],augment=False);index={};crops=0
    torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
    try:
        for record in records:
            if file_hash(record['image_path'])!=record['image_sha256']:raise ValueError('Changed inference RGB')
            with Image.open(record['image_path']) as image:
                rgb=image.convert('RGB');tensor,_=transform(rgb,Image.new('L',rgb.size,0),0)
            with torch.inference_mode():
                probability=net(tensor[None].to(c['device']))[0].sigmoid().cpu().numpy()
                context=torch.cat((encoded['e4'].mean((2,3)),encoded['e4'].amax((2,3))),dim=1)[0].cpu().numpy()
            if probability.shape!=(2,canvas//2,canvas//2) or context.shape!=(4096,):raise ValueError('Unexpected scaled neural outputs')
            _,raw,expansion=read_rows(probability,record,settings,ocr,c['parseq_batch_size'])
            raw,rows=rows_with_sources(raw,frozen['selected']['policy']['minimum_transcript_probability'],record['image_wh'])
            mp=output/'maps'/(key(record['sample_id'])+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(mp,probabilities=probability,encoder_context=context)
            path=output/'samples'/(key(record['sample_id'])+'.json')
            write(path,dict(raw_rows=raw,rows=rows,image_wh=record['image_wh'],image_sha256=record['image_sha256'],
                rgb_sha256=record['rgb_sha256'],maps_path=str(mp),maps_sha256=file_hash(mp),ground_truth_used=False,expansion=expansion))
            index[record['sample_id']]=dict(path=str(path),sha256=file_hash(path));crops+=len(raw)
        torch.cuda.synchronize();write(output/'index.json',index)
        write(output/'complete.json',dict(status='COMPLETE',mode='approximate',actual_unet_image_inferences=len(records),
            actual_parseq_crop_inferences=crops,actual_deepsolo_calls=0,input_hw=[canvas,canvas],native_hw=[canvas//2,canvas//2],
            precision='float32',unet_batch_size=1,parseq_batch_size=c['parseq_batch_size'],gpu=torch.cuda.get_device_name(0),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),out_of_memory=False,
            seconds=time.perf_counter()-started,model_loading_seconds=loading,checkpoint_sha256=complete['checkpoint_sha256'],
            crop_settings=settings,ground_truth_used=False,source_sha256=file_hash(__file__)))
    finally:
        hook.remove();encoded.clear();ocr.release();del ocr,net;gc.collect();torch.cuda.empty_cache()
