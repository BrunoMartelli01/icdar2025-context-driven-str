"""Qualify native caption batches under the frozen development-selected prompt."""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
import time
import traceback
import platform
from importlib.metadata import version

from scripts.run_paper_caption_comparison import read,sha,seal,digest,image_rgb
from caption.qwen_batch_scene import LocalQwenBatchScene


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--role',choices=['qualification','paper_example'],required=True)
    args=p.parse_args();c=read(args.config);protocol=read(c['sample_protocol']);root=args.root;root.mkdir(parents=True,exist_ok=True);model=None
    if c['mode']!='approximate':raise ValueError('Explicit approximate caption mode required')
    import torch
    torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    name=c['selected_profile'];profile=c['profiles'][name];asset=c['assets']['qwen']
    ids=(protocol['roles']['paper_example'] if args.role=='paper_example' else [s for v in protocol['roles'].values() if isinstance(v,dict) for s in v['qualification']])
    snapshot=Path(c['hf_cache'])/('models--'+asset['model_id'].replace('/','--'))/'snapshots'/asset['revision']
    plan=dict(config=c,config_sha256=sha(args.config),sample_protocol_sha256=sha(c['sample_protocol']),
        source_sha256={f:sha(f) for f in c['scientific_source_files']},runtime=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
            python=platform.python_version(),cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),
            packages={p:version(p) for p in ('transformers','safetensors','numpy','Pillow')},deterministic_algorithms=True,tf32=False),
        scope='Native batch hardware qualification; frozen prompt; no OCR/GT/F1 input')
    path=root/('plan_'+args.role+'.json')
    if path.exists():
        if read(path)!=plan:raise ValueError('Changed native batch plan')
    else:seal(path,plan)
    try:
        for f,h in asset['weight_sha256'].items():
            if sha(snapshot/f)!=h:raise ValueError('Model shard SHA256 mismatch: '+f)
        files={str(p.relative_to(snapshot)):sha(p) for p in snapshot.glob('*.json')}
        model=LocalQwenBatchScene(snapshot,c['device'],profile['min_pixels'],profile['max_pixels'],profile['placement'],profile['gpu_weight_budget'],profile['cpu_weight_budget'])
        for offset in range(0,len(ids),c['batch_size']):
            batch=ids[offset:offset+c['batch_size']];paths=[root/name/args.role/(hashlib.sha256(s.encode()).hexdigest()+'.json') for s in batch]
            if any(p.exists() for p in paths):raise ValueError('Existing batch outputs; use a fresh root')
            rgbs=[image_rgb(protocol['samples'][s]) for s in batch];results=model.generate(rgbs,profile,c['caption_target'])
            for sid,path,rgb,result in zip(batch,paths,rgbs,results):
                native=result['answers'][0]['result']
                seal(path,dict(plan_sha256=digest(plan),profile=profile,sample=protocol['samples'][sid],
                    model_asset=dict(asset,snapshot=str(snapshot),configuration_file_sha256=files),result=result,payload_sha256=digest(result),
                    source_shape=list(rgb.shape),batch_ids=batch,peak_allocated_bytes=native['peak_allocated_bytes'],peak_reserved_bytes=native['peak_reserved_bytes']))
            print(dict(role=args.role,done=offset+len(batch),total=len(ids),captions=[dict(sample_id=s,words=r['quality']['words'],text=r['raw_caption']) for s,r in zip(batch,results)]),flush=True)
    except Exception as e:
        seal(root/('failure_'+str(time.time_ns())+'.json'),dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
        raise
    finally:
        if model is not None:model.release()
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
