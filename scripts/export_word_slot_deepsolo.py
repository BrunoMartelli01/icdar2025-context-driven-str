"""Legacy Python3.8 batch export for frozen DeepSolo specialist training labels.

Uses the qualified loader, preprocessing and serialization unchanged. One
checkpoint load per export is an explicit process-lifecycle hardware deviation.
The main environment never imports Detectron2. A failed image aborts the export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--phases',nargs='+',default=['fit','calibration'],choices=['fit','calibration'])
    args=parser.parse_args();root=args.run;protocol=read(root/'protocol.json');config=protocol['config'];paths=config['paths']
    # Verify frozen official source/config/checkpoint identities before importing
    # legacy code; failures are written with traceback and never skipped.
    from deepsolo.candidates import verify_host_source_identity,resolve_runtime_candidate
    plan=read(paths['deepsolo_plan']);candidate=plan['candidate_resolution']['candidate']['artifact_id']
    runtime=read(paths['deepsolo_runtime'])
    source=plan['candidate_resolution']['source_revision_evidence']['commit_marker_path']
    source_root=plan['candidate_resolution']['runtime_metadata']['source_identity_verification']['host_source_root']
    runtime=verify_host_source_identity(Path(paths['deepsolo_runtime']),Path(source_root))
    resolved=resolve_runtime_candidate(runtime,candidate)
    if resolved['checkpoint_sha256']!=plan['candidate_resolution']['checkpoint_sha256']:raise ValueError('Wrong DeepSolo checkpoint')
    sys.path.insert(0,str(Path(source_root)))
    import cv2
    import torch
    from adet.config import get_cfg
    from deepsolo.worker import _load_model_and_validate,_predict,_serialize_instances
    torch.set_num_threads(config['runtime']['cpu_threads'])
    torch.backends.cudnn.enabled=True;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
    cfg=get_cfg();cfg.merge_from_file(resolved['config_path']);cfg.MODEL.WEIGHTS=resolved['checkpoint_path'];cfg.MODEL.DEVICE='cuda:0'
    threshold=plan['config']['score_threshold'];cfg.MODEL.TRANSFORMER.INFERENCE_TH_TEST=threshold;cfg.freeze()
    model,load=_load_model_and_validate(cfg,resolved['checkpoint_path']);torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();count=0
    try:
        for phase in args.phases:
            for i,record in enumerate(protocol['index'][phase],1):
                path=root/'deep_training_raw'/phase/(record['sample_id'].replace(':','__')+'.json')
                if path.exists():
                    old=read(path)
                    if old['image_sha256']!=record['image_sha256'] or old['checkpoint_sha256']!=resolved['checkpoint_sha256']:
                        raise ValueError('Changed DeepSolo cache')
                    continue
                if sha(record['image_path'])!=record['image_sha256']:raise ValueError('Changed training RGB')
                image=cv2.imread(record['image_path'],cv2.IMREAD_COLOR)
                if image is None:raise ValueError('Could not decode training RGB')
                tick=time.perf_counter();prediction=_predict(model,cfg,image,True);torch.cuda.synchronize()
                detections=_serialize_instances(prediction['instances'],cfg)
                write(path,{'schema_version':1,'status':'SUCCESS','sample_id':record['sample_id'],
                      'image_sha256':record['image_sha256'],'checkpoint_sha256':resolved['checkpoint_sha256'],
                      'config_sha256':resolved['config_sha256'],'score_threshold':threshold,'detections':detections,
                      'inference_seconds':time.perf_counter()-tick,'precision':'float32','batch_size':1})
                count+=1
                if i%50==0 or i==len(protocol['index'][phase]):print('DEEPSOLO_LABELS',phase,i,flush=True)
        write(root/'deep_training_export.json',{'status':'SUCCESS','new_images':count,'phases':args.phases,
              'checkpoint_sha256':resolved['checkpoint_sha256'],'config_sha256':resolved['config_sha256'],
              'strict_load':load,'seconds':time.perf_counter()-started,'gpu':torch.cuda.get_device_name(),
              'precision':'float32','device':'cuda:0','batch_size':1,'cudnn_deterministic':False,
              'classification':'HARDWARE_DEVIATION','process_policy':'one legacy checkpoint load, sequential batch1 image export',
              'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    except Exception as error:
        write(root/'deep_training_failure.json',{'status':'FAILED','type':type(error).__name__,'message':str(error),
              'traceback':traceback.format_exc(),'oom':'out of memory' in str(error).lower()})
        raise


if __name__=='__main__':main()
