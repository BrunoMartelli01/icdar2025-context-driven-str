"""GT-blind development and qualification of longer native scene captions."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import platform
from importlib.metadata import version
import time
import traceback

import numpy as np

from caption.blip2_captioner import BLIP2ApproxConfig
from caption.paper_style_blip2 import LocalPaperBLIP2, InstructionCaptionConfig
from caption.detailed_scene import detailed_profile
from scripts.run_paper_caption_comparison import sha, read, seal, image_rgb, digest


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--profiles',nargs='+',required=True)
    p.add_argument('--role',choices=['development','qualification','paper_example'],required=True)
    args=p.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    c=read(args.config);protocol=read(c['sample_protocol']);model=None
    if c['mode']!='approximate':raise ValueError('Explicit approximate mode required')
    if c.get('runtime_dependencies'):
        sys.path.insert(0,c['runtime_dependencies'])
    import torch
    torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ids=(protocol['roles']['paper_example'] if args.role=='paper_example' else
         [sid for d,v in protocol['roles'].items() if isinstance(v,dict) for sid in v[args.role]])
    sources={f:sha(f) for f in c['scientific_source_files']}
    plan=dict(config=c,config_sha256=sha(args.config),sample_protocol_sha256=sha(c['sample_protocol']),
              source_sha256=sources,runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                  python=platform.python_version(),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
                  packages={p:version(p) for p in ('transformers','safetensors','numpy','Pillow')},
                  deterministic_algorithms=True,tf32=False),
              scope='Caption quality only; no OCR or annotation input; no router evaluation or TEST-based profile selection')
    plan_path=root/('plan_'+args.role+'.json')
    if plan_path.exists():
        if read(plan_path)!=plan:raise ValueError('Changed caption probe plan')
    else:seal(plan_path,plan)
    try:
        for profile_name in args.profiles:
            profile=c['profiles'][profile_name];family=profile['caption_model'];asset=c['assets'][family]
            snapshot=Path(c['hf_cache'])/('models--'+asset['model_id'].replace('/','--'))/'snapshots'/asset['revision']
            for f,h in asset['weight_sha256'].items():
                if sha(snapshot/f)!=h:raise ValueError('Changed model weights: '+f)
            asset_files={str(f.relative_to(snapshot)):sha(f) for f in snapshot.rglob('*') if f.is_file() and f.suffix in ('.json','.model','.py')}
            if family=='qwen':
                from caption.qwen_scene import LocalQwenScene
                model=LocalQwenScene(snapshot,c['device'],profile['min_pixels'],profile['max_pixels'],
                    profile.get('placement','single_device'),profile.get('gpu_weight_budget'),profile.get('cpu_weight_budget'))
            elif family=='florence':
                from caption.florence_scene import LocalFlorenceScene
                model=LocalFlorenceScene(snapshot,c['device'])
            else:
                cls=InstructionCaptionConfig if family=='instructblip' else BLIP2ApproxConfig
                model=LocalPaperBLIP2(cls(model_id=asset['model_id'],revision=asset['revision'],device=c['device'],
                    dtype='float16',placement='single_device',quantization='none'),snapshot,c['caption_max_memory'])
            torch.cuda.init();started=time.monotonic()
            for offset in range(0,len(ids),c['batch_size']):
                batch=ids[offset:offset+c['batch_size']]
                paths=[root/profile_name/args.role/(hashlib.sha256(s.encode()).hexdigest()+'.json') for s in batch]
                if all(path.exists() for path in paths):
                    for path,sid in zip(paths,batch):
                        previous=read(path)
                        if previous['plan_sha256']!=digest(plan) or previous['profile']!=profile or previous['sample']!=protocol['samples'][sid] or previous['payload_sha256']!=digest(previous['result']):
                            raise ValueError('Invalid existing caption probe shard')
                    continue
                if any(path.exists() for path in paths):raise ValueError('Partial native batch; preserve and restart in a fresh root')
                rgbs=[image_rgb(protocol['samples'][sid]) for sid in batch]
                results=(model.generate(rgbs,profile,c['caption_target']) if family in ('florence','qwen') else
                         detailed_profile(model,rgbs,profile,c['caption_target']))
                for path,sid,result,rgb in zip(paths,batch,results,rgbs):
                    seal(path,dict(plan_sha256=digest(plan),profile=profile,sample=protocol['samples'][sid],
                        model_asset=dict(asset,snapshot=str(snapshot),configuration_file_sha256=asset_files),
                        result=result,payload_sha256=digest(result),source_shape=list(rgb.shape),batch_ids=batch,
                        peak_allocated_bytes=max(a['result']['peak_allocated_bytes'] for a in result['answers']),
                        peak_reserved_bytes=max(a['result']['peak_reserved_bytes'] for a in result['answers'])))
                print(json.dumps(dict(profile=profile_name,role=args.role,done=offset+len(batch),total=len(ids),
                    seconds=time.monotonic()-started,captions=[dict(sample_id=s,words=r['quality']['words'],text=r['raw_caption']) for s,r in zip(batch,results)])),flush=True)
            model.release();model=None;gc.collect();torch.cuda.empty_cache()
    except Exception as e:
        seal(root/('failure_'+str(time.time_ns())+'.json'),dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
        raise
    finally:
        if model is not None:model.release()
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
