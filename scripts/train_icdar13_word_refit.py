"""Train isolated IC13 full-word/kernel controls; all selection remains on TRAIN validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,Subset

from scripts.icdar_budget_data import read,sha,key,annotation,predictions
from scripts.run_yolo_f1_quality_study import seal
from segmentation.icdar13_word_refit import WordRefitUNet,WordDataset,word_loss,padded_rows
from recognition.icdar_budget_router import lexical_counts
from recognition.f1_quality_study import aggregate
from recognition.unet_token_routing import unet_proposals,selected_rows
from recognition.parseq import ParseqApproxConfig,ParseqApproxRecognizer,PARSEQ_ARCHIVE_SOURCE
from scripts.crop_clip_runtime import array_sha

REPO=Path(__file__).resolve().parents[1]


def save_checkpoint(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)


def recognizer(plan):
    p=plan['paths'];return ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(p['parseq_source']),checkpoint=Path(p['parseq_checkpoint']),
        runtime_source_archive=Path(p['parseq_archive']),runtime_source_type=PARSEQ_ARCHIVE_SOURCE,device=plan['config']['device']))


def read_rows(probability,r,settings,ocr,batch_size):
    candidates,labels,audit=unet_proposals(probability,r['image_wh'],settings)
    candidates=padded_rows(candidates,r['image_wh'],settings['padding'])
    with Image.open(r['image_path']) as p:rgb=np.ascontiguousarray(p.convert('RGB'))
    rows=[]
    for offset in range(0,len(candidates),batch_size):
        group=candidates[offset:offset+batch_size];crops=[np.ascontiguousarray(rgb[s['bbox'][1]:s['bbox'][3],s['bbox'][0]:s['bbox'][2]]) for s in group]
        outputs=ocr.recognize_candidate_batch(crops)
        if len(outputs)!=len(group):raise RuntimeError('Incomplete PARSeq crop response')
        rows.extend(dict(s,**v.to_dict(),candidate_id=r['sample_id']+':'+str(s['instance_id']),crop_rgb_sha256=array_sha(crop)) for s,v,crop in zip(group,outputs,crops))
    return selected_rows(rows,settings),rows,audit


def evaluate(net,plan,records,directory,settings):
    c=plan['config'];dataset=WordDataset(records,c,augment=False);ocr=recognizer(plan);counts=[];net.eval()
    try:
        for i,r in enumerate(records):
            with torch.inference_mode():prob=net(dataset[i]['image'][None].to(c['device']))[0].sigmoid().cpu().numpy()
            rows,raw,audit=read_rows(prob,r,settings,ocr,c['parseq_batch_size']);gt,valid=annotation(r)
            counted=lexical_counts(predictions(rows,r,valid),gt);counts.append(counted)
            mp=directory/'maps'/(key(r['sample_id'])+'.npz');mp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(mp,probabilities=prob)
            seal(directory/'samples'/(key(r['sample_id'])+'.json'),dict(rows=rows,raw_rows=raw,image_sha256=r['image_sha256'],maps_path=str(mp),maps_sha256=sha(mp),
                counts=counted,expansion=audit,ground_truth_used_for_prediction=False))
    finally:ocr.release();torch.cuda.empty_cache()
    value=aggregate(counts);seal(directory/'metrics.json',dict(metric='lexical_multiset_micro_casefold_alnum_native256_ignore',aggregate=value,
        selection_scope='TRAIN model-selection images only',deepsolo_images=0,total_images=len(records),deepsolo_percentage=0.))
    return value


def prepare(args):
    if args.mode!='approximate' or args.config.resolve().parent!=REPO/'configs/approximate' or args.paths.resolve().parent!=REPO/'configs/approximate':raise ValueError('Explicit approximate namespace required')
    c=read(args.config);p=read(args.paths);targets=read(Path(p['targets_root'])/'manifest.json')
    if targets['config']!=c or c['mode']!='approximate' or p['mode']!='approximate':raise ValueError('Changed target/scientific mode contract')
    for file,digest in c['pinned_inputs'].items():
        if sha(file)!=digest:raise ValueError('Pinned input changed: '+file)
    root=Path(p['output_root'])/('smoke' if args.smoke else 'full');parts=dict(targets['partitions'])
    if args.smoke:parts={k:v[:(16 if k=='fit' else 4)] for k,v in parts.items()}
    folders=['models','segmentation','scripts','recognition','localize','caption','scoring','pipeline','research']
    code={str(f.relative_to(REPO)):sha(f) for folder in folders for f in (REPO/folder).rglob('*.py')}
    plan=dict(mode='approximate',claim_label='approximate reproduction',config=c,paths=p,records=targets['records'],partitions=parts,
        result_class='smoke-test result' if args.smoke else 'approximate reproduction',targets_manifest_sha256=sha(Path(p['targets_root'])/'manifest.json'),
        config_sha256=sha(args.config),paths_sha256=sha(args.paths),code_sha256=code,
        hardware=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version()),
        smoke=args.smoke,test_used_for_selection=False,source_expert='Complete strict Text+Kernel warm start; no head reset')
    seal(root/'resolved_configuration.json',plan)
    for file,digest in code.items():
        target=root/'code_snapshot'/file;target.parent.mkdir(parents=True,exist_ok=True)
        if sha(REPO/file)!=digest:raise ValueError('Source changed while freezing')
        if target.exists():
            if sha(target)!=digest:raise ValueError('Changed frozen source')
        else:shutil.copyfile(REPO/file,target)
    shutil.copytree(REPO/'configs',root/'code_snapshot/configs',dirs_exist_ok=True)
    return root


def train_branch(plan,root,partial):
    c=plan['config'];t=c['training'];device=c['device'];p=plan['paths'];name='partial' if partial else 'frozen';out=root/name
    if (out/'training.json').exists():raise ValueError('Training already completed; preserve results')
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    net=WordRefitUNet(partial_encoder=partial,skip_floor=c['skip_floor'],initial_gap_bias=c['initial_gap_bias'])
    source=torch.load(p['tt_checkpoint'],map_location='cpu',weights_only=False)
    if source['run_contract']!=read(p['tt_contract']):raise ValueError('Text+Kernel source contract changed')
    net.load_state_dict(source['model_state_dict'],strict=True);del source;net.to(device)
    optimizer=torch.optim.Adam([dict(params=[v for k,v in net.named_parameters() if v.requires_grad and not k.startswith('encoder.')],lr=t['decoder_learning_rate'],initial_lr=t['decoder_learning_rate'])]+
        ([dict(params=[v for k,v in net.named_parameters() if v.requires_grad and k.startswith('encoder.')],lr=t['encoder_learning_rate'],initial_lr=t['encoder_learning_rate'])] if partial else []),weight_decay=t['weight_decay'])
    lookup={r['sample_id']:r for r in plan['records']};fit=[lookup[s] for s in plan['partitions']['fit']];validation=[lookup[s] for s in plan['partitions']['selection']]
    dataset=WordDataset(fit,c,augment=True);history=[];best=None;best_epoch=None;start_epoch=0
    if (out/'last.pt').exists():
        state=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if state['plan_sha256']!=sha(root/'resolved_configuration.json') or state['partial_encoder']!=partial:raise ValueError('Changed resume contract')
        net.load_state_dict(state['model_state_dict'],strict=True);optimizer.load_state_dict(state['optimizer_state_dict'])
        history=state['history'];best=state['best'];best_epoch=state['best_epoch'];start_epoch=state['epoch']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        best=evaluate(net,plan,validation,out/'validation/epoch_000',c['inference']);best_epoch=0
        save_checkpoint(out/'best.pt',dict(model_state_dict=net.state_dict(),epoch=0,validation=best,plan_sha256=sha(root/'resolved_configuration.json'),partial_encoder=partial))
    epochs=1 if plan['smoke'] else t['epochs'];start=time.monotonic()
    for epoch in range(start_epoch+1,epochs+1):
        dataset.transform.set_epoch(epoch-1);generator=torch.Generator().manual_seed(c['seed']+epoch)
        loader=DataLoader(dataset,batch_size=t['physical_batch_size'],shuffle=True,generator=generator,num_workers=t['num_workers'],pin_memory=True)
        fraction=(epoch-1)/max(1,t['epochs']-1);factor=t['minimum_learning_rate_ratio']+(1-t['minimum_learning_rate_ratio'])*.5*(1+math.cos(math.pi*fraction))
        for g in optimizer.param_groups:g['lr']=g['initial_lr']*factor
        net.train();optimizer.zero_grad(set_to_none=True);total=0.;seen=[];window=0;count_window=0;steps=0;torch.cuda.reset_peak_memory_stats()
        def update():
            nonlocal window,count_window,steps
            for g in optimizer.param_groups:
                for v in g['params']:
                    if v.grad is not None:v.grad.div_(count_window)
            torch.nn.utils.clip_grad_norm_([v for v in net.parameters() if v.requires_grad],t['gradient_clip_norm'],error_if_nonfinite=True)
            optimizer.step();optimizer.zero_grad(set_to_none=True);window=count_window=0;steps+=1
        for batch in loader:
            logits=net(batch['image'].to(device));loss=word_loss(logits,batch,c)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-word/kernel loss')
            count=len(batch['sample_id']);(loss*count).backward();window+=1;count_window+=count;total+=float(loss.detach())*count;seen+=batch['sample_id']
            if window==t['gradient_accumulation']:update()
        if window:update()
        if sorted(seen)!=sorted(plan['partitions']['fit']):raise ValueError('Incomplete fitting population')
        row=dict(epoch=epoch,loss=total/len(seen),samples=len(seen),steps=steps,sample_order_sha256=hashlib.sha256(('\n'.join(seen)+'\n').encode()).hexdigest(),
            gpu_allocated=torch.cuda.max_memory_allocated(),gpu_reserved=torch.cuda.max_memory_reserved())
        if epoch%c['validation_interval']==0 or epoch==epochs:
            value=evaluate(net,plan,validation,out/('validation/epoch_'+str(epoch).zfill(3)),c['inference']);row['validation']=value
            if value['f1']>best['f1']:
                best=value;best_epoch=epoch;save_checkpoint(out/'best.pt',dict(model_state_dict=net.state_dict(),epoch=epoch,validation=best,plan_sha256=sha(root/'resolved_configuration.json'),partial_encoder=partial))
        history.append(row)
        save_checkpoint(out/'last.pt',dict(model_state_dict=net.state_dict(),optimizer_state_dict=optimizer.state_dict(),epoch=epoch,partial_encoder=partial,
            plan_sha256=sha(root/'resolved_configuration.json'),history=history,best=best,best_epoch=best_epoch,python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
        (out/'epochs.json').write_text(json.dumps(history,indent=2)+'\n')
        print(json.dumps(dict(branch=name,epoch=epoch,loss=row['loss'],best_epoch=best_epoch,selection_f1=best['f1'],seconds=time.monotonic()-start)),flush=True)
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False);net.load_state_dict(saved['model_state_dict'],strict=True);del saved
    replay=evaluate(net,plan,validation,out/'selected_validation_replay',c['inference'])
    if replay!=best:raise ValueError('Selected checkpoint replay changed')
    seal(out/'training.json',dict(status='SUCCESS',training_complete=not plan['smoke'],result_class=plan['result_class'],epochs=epochs,fit_ids=plan['partitions']['fit'],
        selection_ids=plan['partitions']['selection'],best_epoch=best_epoch,best_validation=best,checkpoint_sha256=sha(out/'best.pt'),
        plan_sha256=sha(root/'resolved_configuration.json'),seconds=time.monotonic()-start,maximum_gpu_allocated=max(r['gpu_allocated'] for r in history),maximum_gpu_reserved=max(r['gpu_reserved'] for r in history)))
    del net,optimizer;torch.cuda.empty_cache()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--mode');p.add_argument('--config',type=Path);p.add_argument('--paths',type=Path);p.add_argument('--run-root',type=Path)
    p.add_argument('--smoke',action='store_true');p.add_argument('--prepare-only',action='store_true');a=p.parse_args();root=a.run_root.resolve() if a.run_root else prepare(a)
    if a.prepare_only:return
    if REPO!=(root/'code_snapshot').resolve():
        subprocess.run([sys.executable,'-u','-m','scripts.train_icdar13_word_refit','--run-root',str(root)],cwd=root/'code_snapshot',env={**os.environ,'PYTHONPATH':str(root/'code_snapshot')},check=True);return
    plan=read(root/'resolved_configuration.json');c=plan['config'];torch.set_num_threads(c['cpu_threads']);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    try:
        for file,digest in c['pinned_inputs'].items():
            if sha(file)!=digest:raise ValueError('Pinned source changed')
        if torch.cuda.mem_get_info()[0]<c['minimum_free_gpu_bytes']:raise RuntimeError('Insufficient free GPU memory')
        for partial in c['partial_encoder_candidates']:train_branch(plan,root,partial)
        seal(root/'training_completion.json',dict(status='SUCCESS',result_class=plan['result_class'],plan_sha256=sha(root/'resolved_configuration.json')))
    except BaseException as error:
        seal(root/('failure_'+str(time.time_ns())+'.json'),dict(error=str(error),traceback=traceback.format_exc(),out_of_memory=isinstance(error,torch.cuda.OutOfMemoryError)));raise


if __name__=='__main__':main()
