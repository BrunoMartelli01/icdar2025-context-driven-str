"""Isolated, already-installed Ultralytics 8.1.0 student training/inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np
from PIL import Image
import torch
import ultralytics
from ultralytics import YOLO


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root):
    launch=read(root/'launch.json');c=launch['config']
    if ultralytics.__version__!='8.1.0':raise RuntimeError('Qualified Ultralytics8.1.0 required; no installation')
    if sha(c['checkpoint'])!=c['checkpoint_sha256']:raise ValueError('YOLO checkpoint changed')
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    index=read(root/'inference_index.json');hardware=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,ultralytics=ultralytics.__version__,python=platform.python_version(),precision='float32',amp=False,batch_size=c['batch_size'],input_shape=[c['batch_size'],3,c['image_size'],c['image_size']],seed=launch['seed'],deterministic_requested=True,deterministic_limit='Ultralytics8.1.0 configures its own deterministic/warn-only backend settings; same requests for both arms')
    write(root/'runtime.json',hardware)
    for arm in ('initial','gt_control','distilled'):
        directory=root/arm;directory.mkdir(exist_ok=True)
        if (directory/'complete.json').exists():continue
        model=YOLO(c['checkpoint']);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
        if arm!='initial':
            model.train(data=str(directory/'data/dataset.yaml'),epochs=1 if launch['smoke'] else c['epochs'],batch=c['batch_size'],imgsz=c['image_size'],
                device=0,workers=0,amp=False,seed=launch['seed'],deterministic=True,optimizer='AdamW',lr0=c['learning_rate'],lrf=1.,weight_decay=c['weight_decay'],
                warmup_epochs=0.,patience=0,val=False,plots=False,save=True,save_period=-1,project=str(directory),name='training',exist_ok=False,
                pretrained=True,verbose=False,**c['augmentation'])
            checkpoint=directory/'training/weights/last.pt'
            if not checkpoint.is_file():raise RuntimeError('Final YOLO checkpoint missing')
            model=YOLO(checkpoint)
        else:checkpoint=Path(c['checkpoint'])
        memory=dict(**hardware,models='YOLOv8s student; final epoch fixed before scientific evaluation',peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),oom=False,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),training_seconds=time.perf_counter()-tick)
        write(directory/'training_memory.json',memory)
        outputs={}
        for phase,entries in index.items():
            outputs[phase]={}
            for number,entry in enumerate(entries,1):
                if sha(entry['image_path'])!=entry['image_sha256']:raise ValueError('YOLO RGB changed')
                with Image.open(entry['image_path']) as image:rgb=np.ascontiguousarray(image.convert('RGB'))
                result=model.predict(source=np.ascontiguousarray(rgb[:,:,::-1]),device=0,imgsz=c['image_size'],conf=min(c['score_thresholds']),iou=c['nms_iou'],max_det=c['max_detections'],half=False,verbose=False)[0]
                outputs[phase][entry['sample_id']]=[dict(bbox=box,confidence=float(conf)) for box,conf in zip(result.boxes.xyxy.detach().cpu().tolist(),result.boxes.conf.detach().cpu().tolist())]
                if number%100==0 or number==len(entries):print(f'STUDENT {arm} {phase} {number}/{len(entries)}',flush=True)
        write(directory/'inference.json',outputs);write(directory/'complete.json',dict(status='SUCCESS',checkpoint_sha256=sha(checkpoint),seconds=time.perf_counter()-tick))
        del model;torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    try:run(args.root.resolve())
    except BaseException as error:
        write(args.root/'failure.json',dict(status='FAILED',type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),oom=isinstance(error,torch.cuda.OutOfMemoryError)))
        raise
