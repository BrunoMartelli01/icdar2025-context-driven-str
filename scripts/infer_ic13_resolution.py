"""Separate detector and PARSeq stages for validation-selected YOLO resolution."""

import argparse
from pathlib import Path

import numpy as np
import torch

from scripts.icdar_budget_data import read,sha,key
from scripts.icdar_budget_stages import pixels,recognizer,peak
from scripts.run_yolo_f1_quality_study import seal


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--stage',choices=['detect_selection','read_selection','detect_remaining','read_remaining'],required=True);a=p.parse_args();c=read(a.config);root=Path(c['study_root']);plan=read(Path(c['old_router'])/'resolved_configuration.json');records={r['sample_id']:r for rows in plan['records']['icdar13'].values() for r in rows};selected=set(plan['partitions']['icdar13']['selection'])
    assert c['mode']=='approximate' and sha(c['checkpoint'])==c['checkpoint_sha256'];torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed']);np.random.seed(c['seed']);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    if torch.cuda.mem_get_info()[0]<c['minimum_free_gpu_bytes']:raise RuntimeError('Insufficient free GPU memory')
    remaining=a.stage.endswith('remaining');sizes=c.get('remaining_sizes',[read(root/'resolution_frozen.json')['selected']['size']]) if remaining else c['sizes'];ids=sorted(set(records)-selected) if remaining else sorted(selected);model=None;done=0
    try:
        for size in sizes:
            for sid in ids:
                r=records[sid];rgb=pixels(r);dest=root/str(size)/('detector' if a.stage.startswith('detect') else 'ocr')/(key(sid)+'.json')
                if dest.exists():
                    old=read(dest);assert old['image_sha256']==r['image_sha256'];continue
                torch.cuda.reset_peak_memory_stats()
                if a.stage.startswith('detect'):
                    from ultralytics import YOLO
                    from scripts.evaluate_totaltext_yolo_word_slots import boxes_to_slots
                    if model is None:model=YOLO(c['checkpoint'])
                    s=c['detector_settings'];shapes=[];hook=model.model.register_forward_pre_hook(lambda m,x:shapes.append(list(x[0].shape)))
                    try:out=model.predict(np.ascontiguousarray(rgb[:,:,::-1]),imgsz=size,conf=s['confidence_floor'],iou=s['nms_iou'],max_det=s['max_detections'],device='cuda:0',half=False,verbose=False)[0]
                    finally:hook.remove()
                    slots,rejected=boxes_to_slots(out.boxes.xyxy.cpu().numpy(),out.boxes.conf.cpu().numpy(),out.boxes.cls.cpu().numpy(),tuple(r['image_wh']),s['confidence_floor']);data=dict(slots=slots,rejected=rejected,model_input_shapes=shapes)
                else:
                    from recognition.group_word_selection import reading_confidence
                    if model is None:model=recognizer(plan)
                    source=root/str(size)/'detector'/(key(sid)+'.json');slots=read(source)['slots'];rows=[]
                    for start in range(0,len(slots),c['parseq_batch_size']):
                        group=slots[start:start+c['parseq_batch_size']];crops=[np.ascontiguousarray(rgb[s['bbox'][1]:s['bbox'][3],s['bbox'][0]:s['bbox'][2]]) for s in group];values=model.recognize_candidate_batch(crops)
                        if len(group)!=len(values):raise RuntimeError('Incomplete PARSeq response')
                        for s,v in zip(group,values):
                            row=dict(candidate_id=sid+':'+str(size)+':'+str(s['source_index']),bbox=s['bbox'],detector_score=s['score'],**v.to_dict());row['confidence']=reading_confidence(row)['confidence_balanced'];rows.append(row)
                    data=dict(rows=rows,detector_path=str(source),detector_sha256=sha(source),parseq_input_shapes=[[min(c['parseq_batch_size'],len(slots)-i),3,32,128] for i in range(0,len(slots),c['parseq_batch_size'])])
                seal(dest,dict(data,image_sha256=r['image_sha256'],ground_truth_used=False,size=size,precision='float32',gpu=peak(),batch_size=1 if a.stage.startswith('detect') else c['parseq_batch_size'],gpu_name=torch.cuda.get_device_name(0)));done+=1
                if done%50==0:print(a.stage,size,done,flush=True)
    finally:
        if model is not None and a.stage.startswith('read'):model.release()
    seal(root/(a.stage+'_receipt.json'),dict(status='PASS',new_images=done,sizes=sizes,ids=ids,config_sha256=sha(a.config),code_sha256=sha(__file__),actual_new_deepsolo_inferences=0))
    print(a.stage,'PASS',done,flush=True)


if __name__=='__main__':main()
