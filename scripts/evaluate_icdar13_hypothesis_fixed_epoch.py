"""Predeclared equal-epoch mixed-data control; never chooses a TEST winner."""
import argparse
from pathlib import Path
import time
import traceback
import torch

from scripts.icdar_budget_data import read,sha
from scripts.run_icdar13_hypothesis_tests import model,recognizer,evaluate_net
from scripts.run_yolo_f1_quality_study import seal


def main(args):
    repo=Path(__file__).resolve().parents[1]
    if args.mode!='approximate' or args.config.resolve().parent!=repo/'configs/approximate':raise ValueError('APPROXIMATE configuration required')
    c=read(args.config);root=Path(c['run_root']);plan=read(root/'resolved_configuration.json')
    if c['mode']!=args.mode or sha(root/'resolved_configuration.json')!=c['parent_plan_sha256']:raise ValueError('Changed control contract')
    if plan['smoke'] or not (root/'router_results.json').exists():raise ValueError('Complete main study first')
    out=root/'mixed_fixed_epoch_control';out.mkdir(exist_ok=False)
    seal(out/'resolved_configuration.json',dict(config=c,config_sha256=sha(args.config),producer_sha256=sha(__file__),mode='approximate',claim_label='approximate reproduction'))
    try:
        torch.set_num_threads(plan['config']['cpu_threads']);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        path=root/c['checkpoint'];state=torch.load(path,map_location='cpu',weights_only=False)
        if state['epoch']!=c['epochs'] or state['fit_ids']!=plan['partitions']['fit']:raise ValueError('Wrong fixed-epoch checkpoint or fitting population')
        done=read(root/'mixed_totaltext/complete.json');reuse=done['selected_epoch']==c['epochs']
        if reuse:
            chosen=torch.load(root/'mixed_totaltext/best.pt',map_location='cpu',weights_only=False)
            if set(state['model_state_dict'])!=set(chosen['model_state_dict']) or any(not torch.equal(v,chosen['model_state_dict'][k]) for k,v in state['model_state_dict'].items()):raise ValueError('Equal-epoch selected/final states differ')
        results={}
        if not reuse:
            if torch.cuda.get_device_name(0)!=plan['hardware']['gpu'] or torch.cuda.mem_get_info()[0]<c['minimum_free_gpu_bytes']:raise RuntimeError('Changed or insufficient GPU hardware')
            net=model(plan['light'],path);ocr=recognizer(plan['light'])
            try:
                for split in c['evaluation_splits']:
                    results[split]=dict(counts=evaluate_net(net,ocr,plan,plan['partitions'][split],out/split),source=str(out/split))
            finally:ocr.release()
        else:
            for split in c['evaluation_splits']:
                folder=root/'mixed_totaltext'/split;results[split]=dict(counts=read(folder/'metrics.json')['counts'],source=str(folder))
        seal(out/'results.json',dict(status='COMPLETE',mode='approximate',claim_label='approximate reproduction',epoch=c['epochs'],checkpoint_sha256=sha(path),reused_selected_epoch=bool(reuse),test_used_for_selection=False,results=results,actual_new_deepsolo_calls=0,
            deepsolo_images=0,total_images=len(plan['partitions']['test']),deepsolo_percentage=0.))
        print(results,flush=True)
    except BaseException as exc:
        seal(out/('failure_'+str(time.time_ns())+'.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode',choices=['approximate'],required=True);p.add_argument('--config',type=Path,required=True);main(p.parse_args())
