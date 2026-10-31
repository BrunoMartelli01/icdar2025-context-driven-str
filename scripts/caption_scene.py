"""Generate a caption with an explicitly selected, qualified approximate profile."""
from __future__ import annotations

import argparse
import gc
import hashlib
import os
from pathlib import Path
import traceback
import time

import numpy as np
from PIL import Image

from caption.qwen_scene import LocalQwenScene
from scripts.run_paper_caption_comparison import read, sha, seal


def main():
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['approximate'])
    p.add_argument('--config',required=True,type=Path);p.add_argument('--image',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--cache-root',type=Path)
    args=p.parse_args();model=None
    try:
        c=read(args.config)
        if c['mode']!='approximate' or 'selected_profile' not in c:
            raise ValueError('An explicitly selected approximate caption profile is required')
        profile=c['profiles'][c['selected_profile']]
        if profile['caption_model']!='qwen':raise ValueError('This qualified entry point requires the selected Qwen caption ablation')
        asset=c['assets']['qwen'];cache=args.cache_root or Path(c['hf_cache'])
        snapshot=cache/('models--'+asset['model_id'].replace('/','--'))/'snapshots'/asset['revision']
        for f,h in asset['weight_sha256'].items():
            if sha(snapshot/f)!=h:raise ValueError('Checkpoint SHA256 mismatch: '+f)
        rgb=np.array(Image.open(args.image).convert('RGB'))
        import torch
        torch.set_num_threads(c['cpu_threads']);torch.manual_seed(c['seed'])
        torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=LocalQwenScene(snapshot,c['device'],profile['min_pixels'],profile['max_pixels'],
            profile.get('placement','single_device'),profile.get('gpu_weight_budget'),profile.get('cpu_weight_budget'))
        result=model.generate([rgb],profile,c['caption_target'])[0]
        seal(args.output,dict(mode='approximate',claim_label='approximate reproduction',
            caption_model_is_paper_blip2=False,config_sha256=sha(args.config),model_asset=asset,
            source_sha256={f:sha(f) for f in ['caption/qwen_scene.py','scripts/caption_scene.py']},
            image_path=str(args.image.resolve()),image_sha256=sha(args.image),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            original_shape=list(rgb.shape),hardware=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__),result=result))
        print(result['raw_caption'])
    except Exception as e:
        failure=args.output.with_name(args.output.name+'.failure_'+str(time.time_ns())+'.json')
        seal(failure,dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc()))
        raise
    finally:
        if model is not None:model.release()
        gc.collect()


if __name__=='__main__':main()
