"""GT-blind, restartable native DeepSolo export on explicitly supplied ICDAR images."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

from scripts.infer_deepsolo_word_f1 import validate_selected_config
from scripts.run_deepsolo_word_f1_study import configure_model_resize

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')

def run(args):
    plan=read(args.plan);c=plan['config'];paths=c['paths'];cfg=read(paths['deepsolo_profile']);validate_selected_config(cfg)
    if c['mode']!='approximate' or plan['claim_label']!='approximate reproduction':raise ValueError('Explicit approximate plan required')
    from deepsolo.candidates import resolve_runtime_candidate,verify_host_source_identity
    runtime=verify_host_source_identity(Path(paths['deepsolo_runtime']),Path(paths['deepsolo_source']))
    resolved=resolve_runtime_candidate(runtime,cfg['candidate'])
    for a,b in [('checkpoint_sha256','checkpoint_sha256'),('config_sha256','upstream_config_sha256'),('base_config_sha256','upstream_base_config_sha256')]:
        if resolved[a]!=cfg[b]:raise ValueError('Native selected DeepSolo asset changed')
    sys.path.insert(0,paths['deepsolo_source'])
    import cv2
    import numpy as np
    import torch
    from adet.config import get_cfg
    from deepsolo.worker import _load_model_and_validate,_predict,_serialize_instances
    random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed']);torch.set_num_threads(cfg['cpu_threads'])
    torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    # The qualified legacy torch1.9 runtime has no cuda.mem_get_info API.
    memory_lines=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    if len(memory_lines)!=1:raise RuntimeError('This qualified worker requires the declared single-GPU host')
    if int(memory_lines[0])*1024*1024<c['minimum_free_gpu_bytes']:raise RuntimeError('Insufficient free GPU memory; other jobs preserved')
    base=get_cfg();base.merge_from_file(resolved['config_path']);base.MODEL.WEIGHTS=resolved['checkpoint_path'];base.MODEL.DEVICE=cfg['device']
    base.MODEL.TRANSFORMER.INFERENCE_TH_TEST=cfg['inference_floor']
    base.INPUT.MIN_SIZE_TEST,base.INPUT.MAX_SIZE_TEST=cfg['resolutions']['native'];base.freeze()
    model,load=_load_model_and_validate(base,resolved['checkpoint_path']);resize=configure_model_resize(model,base)
    binding=dict(plan_sha256=sha(args.plan),profile_sha256=sha(paths['deepsolo_profile']),code_sha256=sha(__file__))
    protocol=args.output/'protocol.json'
    if protocol.exists():
        if read(protocol)['binding']!=binding:raise ValueError('Changed immutable DeepSolo export')
    else:save(protocol,dict(binding=binding,source_identity=resolved,load_validation=load,runtime=runtime,
        precision='float32',batch_size=1,gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,
        resize=resize,ground_truth_used=False,model_set='Native ViTAEv2S/TextOCR DeepSolo only',inference_floor=cfg['inference_floor']))
    generated=0;reused=0;files={};start=time.monotonic()
    for dataset,samples in plan['population']['test'].items():
        samples=samples[:args.limit_per_dataset] if args.limit_per_dataset else samples
        for number,sample in enumerate(samples,1):
            sid=sample['sample_id'];path=args.output/(sid.replace(':','__')+'.json')
            contract=dict(binding,sample_id=sid,image_sha256=sample['image_sha256'])
            if path.exists():
                old=read(path);payload={k:v for k,v in old.items() if k!='payload_sha256'}
                if old['contract']!=contract or old['payload_sha256']!=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest():raise ValueError('Changed completed DeepSolo shard')
                reused+=1
            else:
                if sha(sample['image_path'])!=sample['image_sha256']:raise ValueError('Changed RGB before DeepSolo')
                image=cv2.imread(sample['image_path'],cv2.IMREAD_COLOR)
                if image is None:raise ValueError('Cannot decode image')
                torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();shapes=[]
                hook=model.register_forward_pre_hook(lambda m,x:shapes.append(list(x[0][0]['image'].shape)))
                try:
                    predicted=_predict(model,base,image,True);torch.cuda.synchronize()
                    detections=_serialize_instances(predicted['instances'],base)
                finally:hook.remove()
                row=dict(status='SUCCESS',contract=contract,response=dict(success=True,detections=detections),
                    input_chw=shapes[0],precision='float32',batch_size=1,seconds=time.perf_counter()-tick,
                    gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    ground_truth_used=False)
                row['payload_sha256']=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest();save(path,row);generated+=1
            files[path.name]=sha(path)
            if number%25==0 or number==len(samples):print(dataset,number,len(samples),'new',generated,'reused',reused,flush=True)
    save(args.output/('receipt_limit_'+str(args.limit_per_dataset)+'.json'),dict(status='SUCCESS',binding=binding,
        new_model_inferences=generated,reused=reused,files_sha256=files,seconds=time.monotonic()-start,
        result_class='partial-dataset result' if args.limit_per_dataset else 'approximate reproduction',
        deployment_scope='Offline counterfactual export; logical router counts are reported separately.'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit-per-dataset',type=int,default=0);args=p.parse_args()
    try:run(args)
    except Exception as error:
        save(args.output/('failure_'+str(time.time_ns())+'.json'),dict(status='FAILED',message=str(error),
            traceback=traceback.format_exc(),oom='out of memory' in str(error).lower()));raise

if __name__=='__main__':main()
