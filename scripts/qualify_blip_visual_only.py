"""Compare explicitly pruned native BLIP visual queries against authenticated TRAIN caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
from PIL import Image
import torch

from recognition.blip_visual_only import load_visual_only

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--device',choices=['cpu','cuda'],required=True)
    args=parser.parse_args();c=read(args.config)
    if c['mode']!='approximate' or args.output.exists():raise ValueError('New explicit approximate qualification required')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(c['cpu_threads'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    free,total=torch.cuda.mem_get_info() if args.device=='cuda' else (None,None)
    if args.device=='cuda' and free<c['thin_minimum_free_gpu_bytes']:raise RuntimeError('Insufficient free GPU memory for visual-only BLIP qualification')
    started=time.monotonic();model,processor=load_visual_only(c['paths']['blip_snapshot'])
    count=sum(t.numel()*t.element_size() for t in model.state_dict().values())
    if count!=c['expected_frozen_encoder_bytes']['blip']:raise ValueError('Unexpected visual-only weight size')
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    model=model.to(args.device);source=Path(c['paths']['token_source'])
    plan=read(source/'resolved_configuration.json');records=sorted((r for r in plan['records']['train'] if r['phase']=='fit'),key=lambda r:r['sample_id'])[:c['thin_qualification_images']]
    comparisons=[]
    for record in records:
        path=Path(record['image_path']);sid=record['sample_id']
        if sha(path)!=record['image_sha256']:raise ValueError('Changed qualification RGB')
        npz=source/'train/features/corrected'/(sid.replace(':','__')+'.npz');meta=read(npz.with_suffix('.json'))
        if sha(npz)!=meta['arrays_sha256'] or meta['image_sha256']!=record['image_sha256']:raise ValueError('Changed reference cache')
        rgb=Image.open(path).convert('RGB');pixels=processor(images=rgb,return_tensors='pt')['pixel_values'].to(args.device)
        queries=model(pixels)[0].cpu().numpy()
        with np.load(npz,allow_pickle=False) as z:reference=z['queries']
        error=float(np.max(np.abs(queries-reference)))
        comparisons.append(dict(sample_id=sid,image_sha256=record['image_sha256'],arrays_sha256=meta['arrays_sha256'],
            query_shape=list(queries.shape),input_shape=list(pixels.shape),max_absolute_error=error,
            matches=bool(np.allclose(queries,reference,atol=c['thin_tolerance'],rtol=0))))
    result=dict(status='PASS' if all(r['matches'] for r in comparisons) else 'FAIL',mode='approximate',
        claim_label='approximate reproduction',comparisons=comparisons,tolerance=c['thin_tolerance'],
        strict_checkpoint_load=True,missing_keys=[],unexpected_keys=[],language_model_constructed=False,
        language_projection_constructed=False,weight_tensor_bytes=count,precision='float32',batch_size=1,
        device=torch.cuda.get_device_name() if args.device=='cuda' else 'cpu',free_bytes_at_start=free,total_device_bytes=total,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else None,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        hardware_deviation='Explicit '+args.device+' FP32 equivalence qualification; source queries were generated on GPU.',
        oom=False,elapsed_seconds=time.monotonic()-started,config_sha256=sha(args.config),
        checkpoint_index_sha256=sha(Path(c['paths']['blip_snapshot'])/'model.safetensors.index.json'),
        source_plan_sha256=sha(source/'resolved_configuration.json'),new_deepsolo_inference_images=0,
        operation='One original RGB BLIP vision/QFormer pass; no captions, language decoder, erased views or likelihoods')
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    if result['status']!='PASS':raise RuntimeError('Visual-only query equivalence failed')


if __name__=='__main__':main()
