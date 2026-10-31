"""Prepare real PARSeq fallback-box readings on native TRAIN for all3 domains."""

import argparse
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image
import torch

from scripts.run_uniform_unet_training import read,write
from scripts.train_icdar13_word_refit import recognizer
from scripts.run_icdar13_dense_instances import runtime
from scripts.icdar_budget_data import key
from recognition.fallback_reread import padded_box
from segmentation.uniform_unet_fusion import file_hash


def run(path):
    c=read(path)
    if c['mode']!='approximate' or path.resolve().parent.name!='approximate':raise ValueError('Explicit approximate reader profile required')
    for p,h in c['pinned_inputs'].items():
        if file_hash(p)!=h:raise ValueError('Changed reader input: '+p)
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False);write(root/'resolved_configuration.json',c)
    plan=read(Path(c['training_root'])/c['datasets'][0]/'resolved_configuration.json');runtime(plan['config']);ocr=recognizer(plan)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();totals={}
    try:
        for ds in c['datasets']:
            plan=read(Path(c['training_root'])/ds/'resolved_configuration.json');lookup={r['sample_id']:r for r in plan['train_records']}
            ids=plan['partitions']['fit']+plan['partitions']['selection'];teacher=read(c['teacher_sources'][ds])['native_benchmark']
            if set(ids)!=set(teacher):raise ValueError('Exact native TRAIN/selection teacher coverage required')
            index={};count=invalid=0
            for sid in ids:
                r=lookup[sid]
                if file_hash(r['image_path'])!=r['image_sha256']:raise ValueError('Changed reread RGB')
                with Image.open(r['image_path']) as im:rgb=np.ascontiguousarray(im.convert('RGB'))
                rows=[];items=[]
                for original in teacher[sid]['rows']:
                    row=dict(instance_id=original['instance_id'],bbox=original['bbox'],rereadings={});rows.append(row)
                    if row['bbox'] is None:invalid+=1;continue
                    for padding in c['paddings']:
                        box=padded_box(row['bbox'],r['image_wh'],padding);a,b,cc,d=box;crop=np.ascontiguousarray(rgb[b:d,a:cc]);items.append((row,padding,box,crop))
                for offset in range(0,len(items),c['batch_size']):
                    group=items[offset:offset+c['batch_size']];outputs=ocr.recognize_candidate_batch([v[3] for v in group])
                    if len(outputs)!=len(group):raise RuntimeError('Incomplete real PARSeq reread')
                    for (row,padding,box,crop),output in zip(group,outputs):
                        row['rereadings'][str(padding)]=dict(output.to_dict(),crop_bbox=box,crop_rgb_sha256=__import__('hashlib').sha256(crop.tobytes()).hexdigest())
                    count+=len(group)
                target=root/ds/'samples'/(key(sid)+'.json');write(target,dict(sample_id=sid,image_sha256=r['image_sha256'],image_wh=r['image_wh'],rows=rows,ground_truth_used=False))
                index[sid]=dict(path=str(target),sha256=file_hash(target))
            write(root/ds/'index.json',index);totals[ds]=dict(images=len(index),actual_parseq_crop_inferences=count,invalid_envelopes=invalid,
                logical_source_deepsolo_images=len(index),total_images=len(index),deepsolo_percentage=100.,actual_new_deepsolo_calls=0)
            print('TRAIN FALLBACK REREAD',ds,totals[ds],flush=True)
        write(root/'complete.json',dict(status='COMPLETE',mode='approximate',result_class='approximate reproduction',scope='Native TRAIN inference preparation only; no TEST or recognition-target claim',
            totals=totals,ground_truth_used=False,actual_new_deepsolo_calls=0,seconds=time.monotonic()-start,
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),precision='float32',batch_size=c['batch_size'],
            model_set='Pinned genuine PARSeq only; DeepSolo outputs cached',gpu=torch.cuda.get_device_name(0),out_of_memory=False,source_sha256=file_hash(__file__)))
    except BaseException as exc:
        write(root/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),traceback=traceback.format_exc(),out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError),
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()));raise
    finally:
        ocr.release();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();run(a.config)
