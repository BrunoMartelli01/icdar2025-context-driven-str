"""Replay the complete prediction-only corrected U-Net on authenticated source RGB."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from recognition.corrected_unet_transfer import CorrectedUNetTransfer

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())

def setup(c):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';torch.set_num_threads(c['cpu_threads']);torch.set_num_interop_threads(1)
    np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--trace',type=Path,required=True);args=p.parse_args();c=read(args.config)
    if c['mode']!='approximate' or args.output.exists():raise ValueError('New explicit approximate qualification required')
    setup(c)
    for path,digest in c['input_sha256'].items():
        if sha(path)!=digest:raise ValueError('Changed scientific source: '+path)
    plan=read(args.source/'resolved_configuration.json');rows=read(args.source/'test/corrected.json')
    trace=read(args.trace)
    samples=sorted(plan['records']['test'],key=lambda r:r['sample_id'])[:3]
    torch.cuda.reset_peak_memory_stats();tick=time.monotonic();runtime=CorrectedUNetTransfer(c);checks=[]
    for s in samples:
        sid=s['sample_id']
        if sha(s['image_path'])!=s['image_sha256']:raise ValueError('Changed source RGB')
        rgb=np.asarray(Image.open(s['image_path']).convert('RGB'))
        predicted,probabilities,details=runtime.predict(rgb);expected=rows[sid]
        with np.load(trace[sid]['maps_path'],allow_pickle=False) as z:reference=z['probabilities']
        discrete=lambda rs:[{k:r[k] for k in ['candidate_id','raw_text','bbox','crop_bbox','crop_rgb_sha256','original_raw_text','inherited_box_and_confidence']} for r in rs]
        same=discrete(predicted)==discrete(expected);error=max((abs(p[k]-e[k]) for p,e in zip(predicted,expected) for k in ['confidence','detector_score']),default=0.)
        check=dict(sample_id=sid,source_image_sha256=s['image_sha256'],same_discrete_outputs=same,
            expected_words=len(expected),actual_words=len(predicted),max_score_error=error,
            max_mask_error=float(np.max(np.abs(probabilities-reference))),mask_bitwise_equal=bool(np.array_equal(probabilities,reference)),
            predicted=predicted,expected=expected)
        checks.append(check);print(sid,'same',same,'words',len(predicted),'mask_error',check['max_mask_error'],'score_error',error,flush=True)
        runtime.clear_image_cache()
    report=dict(status='PASS' if all(v['same_discrete_outputs'] and v['mask_bitwise_equal'] and v['max_score_error']<1e-5 for v in checks) else 'FAIL',
        mode='approximate',claim_label='smoke-test result',checks=checks,source_sha256=sha(args.source/'resolved_configuration.json'),
        config_sha256=sha(args.config),trace_sha256=sha(args.trace),qualification_code_sha256=sha(__file__),code_sha256=sha(Path(__file__).parents[1]/'recognition/corrected_unet_transfer.py'),
        runtime_counts=runtime.counters,model_set='Frozen U-Net, PARSeq, pre-selector, pointwise selector, GNN, RGB verifier, text editor',
        device=torch.cuda.get_device_name(),precision='float32',unet_batch_size=1,parseq_batch_size=c['parseq_batch_size'],
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),oom=False,seconds=time.monotonic()-tick)
    runtime.close()
    with args.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    if report['status']!='PASS':raise RuntimeError('Complete corrected U-Net transfer replay failed; do not substitute a lighter expert')

if __name__=='__main__':main()
