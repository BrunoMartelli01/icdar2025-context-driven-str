"""Recover discarded low-joint-confidence PARSeq hypotheses on frozen YOLO boxes."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from recognition.group_word_selection import reading_confidence
from scripts.icdar_budget_data import read,sha,key
from scripts.icdar_budget_stages import recognizer,pixels,peak
from scripts.crop_clip_runtime import array_sha
from scripts.run_yolo_f1_quality_study import seal


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=read(a.config);assert c['mode']=='approximate'
    root=Path(c['study_root']);old=Path(c['old_router']);plan=read(old/'resolved_configuration.json');torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    if torch.cuda.mem_get_info()[0]<c['minimum_free_gpu_bytes']:raise RuntimeError('Insufficient free GPU memory')
    model=None;generated=0;reused=0;manifest={}
    try:
        for split,records in plan['records']['icdar13'].items():
            for i,r in enumerate(records,1):
                sid=r['sample_id'];path=root/'raw_ocr'/(key(sid)+'.json');source=old/'yolo_detector'/(key(sid)+'.json');slots=read(source);assert slots['image_sha256']==r['image_sha256'];rgb=pixels(r)
                if split=='test':
                    prior=Path(c['test_raw_ocr'])/(key(sid)+'.json');data=read(prior);rows=data['rows']
                    for row in rows:
                        x1,y1,x2,y2=row['bbox'];assert array_sha(np.ascontiguousarray(rgb[y1:y2,x1:x2]))==row['crop_rgb_sha256']
                    meta=dict(source_path=str(prior),source_sha256=sha(prior),reused=True);reused+=1
                else:
                    if model is None:model=recognizer(plan)
                    torch.cuda.reset_peak_memory_stats();rows=[]
                    for start in range(0,len(slots['slots']),c['parseq_batch_size']):
                        group=slots['slots'][start:start+c['parseq_batch_size']];crops=[np.ascontiguousarray(rgb[s['bbox'][1]:s['bbox'][3],s['bbox'][0]:s['bbox'][2]]) for s in group];values=model.recognize_candidate_batch(crops)
                        if len(values)!=len(group):raise RuntimeError('Incomplete PARSeq output')
                        rows.extend(dict(candidate_id=sid+':raw:'+str(s['source_index']),bbox=s['bbox'],detector_score=s['score'],**value.to_dict()) for s,value in zip(group,values))
                    meta=dict(gpu=peak(),precision='float32',batch_size=c['parseq_batch_size'],model_set='PARSeq only; YOLO boxes reused',reused=False);generated+=1
                rows=[dict(row,confidence=reading_confidence(row)['confidence_balanced']) for row in rows]
                seal(path,dict(image_sha256=r['image_sha256'],rows=rows,ground_truth_used=False,detector_path=str(source),detector_sha256=sha(source),metadata=meta));manifest[str(path)]=sha(path)
                if i%50==0 or i==len(records):print(split,i,'/',len(records),flush=True)
    finally:
        if model is not None:model.release()
    seal(root/'raw_ocr_receipt.json',dict(status='PASS',new_light_images=generated,reused_light_images=reused,actual_new_deepsolo_inferences=0,files_sha256=manifest,code_sha256=sha(__file__),gpu=torch.cuda.get_device_name(0),scope='Offline counterfactual light readings; final61-image DeepSolo routing reported separately'))


if __name__=='__main__':main()
