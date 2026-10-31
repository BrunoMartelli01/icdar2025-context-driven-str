"""Conditional decoder-heldout supervision for the common trained pipeline.

Train the same initial U-Net recipe on each fold's native FIT subset for the
global validation-selected epoch count. No fold label selects its own weights.
Final inference weights are retained after redundant optimizer checkpoints are
removed only following independent state and export audits.
"""

import argparse
import copy
import hashlib
import os
from pathlib import Path
import random
import shutil
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

from scripts.run_uniform_unet_training import read,write,make_model
from segmentation.uniform_unet_fusion import file_hash


def check_fold(plan):
    fit=plan['partitions']['fit'];held=plan['partitions']['decoder_heldout']
    original=plan['fold_contract']['original_fit_ids'];val=plan['partitions']['selection']
    if len(fit)!=len(set(fit)) or len(held)!=len(set(held)) or set(fit)&set(held) or set(fit)|set(held)!=set(original):
        raise ValueError('Fold must partition every original native FIT image exactly')
    lookup={r['sample_id']:r for r in plan['train_records']+plan['test_records']}
    a={lookup[s]['rgb_sha256'] for s in fit+val+plan['partitions']['test']}
    if a&{lookup[s]['rgb_sha256'] for s in held} or set(held)&set(val):
        raise ValueError('Decoder-heldout RGB/ID overlaps fitting, validation or TEST')
    if plan['config']['training']['epochs']!=plan['fold_contract']['fixed_epochs']:
        raise ValueError('Globally frozen fold epoch budget changed')


def train(root):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from scripts.run_icdar13_dense_instances import runtime
    from scripts.train_icdar13_word_refit import save_checkpoint
    from segmentation.icdar13_word_refit import WordDataset
    from segmentation.icdar13_matched_training import matched_word_loss
    plan=read(root/'resolved_configuration.json');check_fold(plan);c=plan['config'];t=c['training']
    if (root/'training_complete.json').exists():return
    for path,digest in plan['artifact_sha256'].items():
        if file_hash(path)!=digest:raise ValueError('Changed fold neural/source artifact: '+path)
    lookup={r['sample_id']:r for r in plan['train_records']}
    for sid in plan['partitions']['fit']:
        for field in ('image','mask','validity','separator','owners'):
            if file_hash(lookup[sid][field+'_path'])!=lookup[sid][field+'_sha256']:raise ValueError('Changed fold fitting input: '+sid+':'+field)
    if shutil.disk_usage(root).free<c['minimum_free_disk_bytes']:raise RuntimeError('Insufficient disk reserve for resumable decoder training')
    runtime(c);net,receipt=make_model(plan);write(root/'initialization.json',receipt)
    optimizer=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=t['learning_rate'],weight_decay=t['weight_decay'])
    lookup={r['sample_id']:r for r in plan['train_records']};ids=plan['partitions']['fit']
    dataset=WordDataset([lookup[s] for s in ids],c,augment=True);history=[]
    if (root/'last.pt').exists():
        state=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        if state['plan_sha256']!=file_hash(root/'resolved_configuration.json'):raise ValueError('Changed fold resume contract')
        net.load_state_dict(state['model_state_dict'],strict=True);optimizer.load_state_dict(state['optimizer_state_dict']);history=state['history']
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);del state
    try:
        for epoch in range(len(history)+1,t['epochs']+1):
            if shutil.disk_usage(root).free<c['minimum_free_disk_bytes']:raise RuntimeError('Fold disk reserve exhausted; preserve the last completed epoch')
            dataset.transform.set_epoch(epoch-1)
            loader=DataLoader(dataset,batch_size=t['physical_batch_size'],shuffle=True,generator=torch.Generator().manual_seed(c['seed']+epoch),num_workers=t['num_workers'],pin_memory=True)
            net.train();optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
            seen=[];total=0.;windows=window_images=steps=0;started=time.monotonic()
            def update():
                nonlocal windows,window_images,steps
                for parameter in net.parameters():
                    if parameter.grad is not None:parameter.grad.div_(window_images)
                torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],t['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True);windows=window_images=0;steps+=1
            for batch in loader:
                local={k:v.to(c['device']) if torch.is_tensor(v) else v for k,v in batch.items()};logits=net(local['image'])
                loss=torch.stack([matched_word_loss(logits[i:i+1],{k:v[i:i+1] for k,v in local.items() if torch.is_tensor(v)},c,'totaltext_bce_dice') for i in range(len(batch['sample_id']))]).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite decoder-fold loss')
                n=len(batch['sample_id']);(loss*n).backward();total+=float(loss.detach())*n;windows+=1;window_images+=n;seen.extend(batch['sample_id'])
                if windows==t['gradient_accumulation']:update()
            if windows:update()
            if sorted(seen)!=sorted(ids):raise ValueError('Incomplete decoder-fold epoch')
            history.append(dict(epoch=epoch,loss=total/len(seen),images=len(seen),steps=steps,
                order_sha256=hashlib.sha256(('\n'.join(seen)+'\n').encode()).hexdigest(),
                peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),seconds=time.monotonic()-started))
            save_checkpoint(root/'last.pt',dict(model_state_dict=net.state_dict(),optimizer_state_dict=optimizer.state_dict(),history=history,
                plan_sha256=file_hash(root/'resolved_configuration.json'),python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
            write(root/'epochs.json',history)
            print(plan['dataset'],'FOLD',plan['fold_contract']['fold'],'EPOCH',epoch,'loss',history[-1]['loss'],flush=True)
        save_checkpoint(root/'final.pt',dict(model_state_dict=net.state_dict(),history=history,
            plan_sha256=file_hash(root/'resolved_configuration.json'),epoch=t['epochs']))
        write(root/'training_complete.json',dict(status='COMPLETE',fixed_epochs=t['epochs'],checkpoint_sha256=file_hash(root/'final.pt'),
            optimizer_checkpoint_sha256=file_hash(root/'last.pt'),gpu=torch.cuda.get_device_name(0),precision='float32',
            image_hw=c['image_hw'],physical_batch_size=t['physical_batch_size'],gradient_accumulation=t['gradient_accumulation'],
            upstream_encoder_exposure=True,decoder_fit_excluded=True,out_of_memory=False,validation_used_to_choose_fold_weights=False))
    except BaseException as exc:
        write(root/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError)));raise
    finally:
        del net;torch.cuda.empty_cache()


def audit(root):
    import math
    import torch
    plan=read(root/'resolved_configuration.json');check_fold(plan);c=plan['config'];torch.set_num_threads(c['cpu_threads'])
    complete=read(root/'training_complete.json');state=torch.load(root/'final.pt',map_location='cpu',weights_only=False)
    assert file_hash(root/'final.pt')==complete['checkpoint_sha256'] and state['plan_sha256']==file_hash(root/'resolved_configuration.json')
    cpu=copy.deepcopy(plan);cpu['config']['device']='cpu';net,receipt=make_model(cpu)
    assert receipt==read(root/'initialization.json');initial=net.state_dict();learned=state['model_state_dict']
    frozen=[k for k in initial if k.startswith('encoder.')]
    assert all(torch.equal(initial[k],learned[k]) for k in frozen)
    assert not torch.equal(initial['up0.block.conv1.weight'],learned['up0.block.conv1.weight'])
    assert all(not torch.equal(initial['head.weight'][i],learned['head.weight'][i]) for i in [0,1])
    net.load_state_dict(learned,strict=True);history=read(root/'epochs.json');assert history==state['history'] and len(history)==complete['fixed_epochs']
    for epoch,row in enumerate(history,1):
        ids=plan['partitions']['fit'];t=c['training'];assert row['epoch']==epoch and row['images']==len(ids)
        assert row['steps']==math.ceil(math.ceil(len(ids)/t['physical_batch_size'])/t['gradient_accumulation'])
        gen=torch.Generator().manual_seed(c['seed']+epoch);torch.empty((),dtype=torch.int64).random_(generator=gen)
        order=[ids[i] for i in torch.randperm(len(ids),generator=gen).tolist()]
        assert hashlib.sha256(('\n'.join(order)+'\n').encode()).hexdigest()==row['order_sha256']
    if (root/'last.pt').exists():
        last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(last['model_state_dict'][k],learned[k]) for k in learned)
    write(root/'independent_training_audit.json',dict(status='PASS',epochs=len(history),frozen_encoder_tensors=len(frozen),both_heads_learned=True,
        checkpoint_sha256=complete['checkpoint_sha256'],fold=plan['fold_contract'],source_sha256=file_hash(__file__)))
    print('FOLD TRAINING AUDIT PASS',root,flush=True)


def export(root):
    import numpy as np
    import torch
    from PIL import Image
    from scripts.run_icdar13_dense_instances import runtime
    from scripts.train_icdar13_word_refit import recognizer,read_rows
    from scripts.export_uniform_unet import rows_with_sources
    from segmentation.icdar13_word_refit import WordRefitUNet
    from segmentation.icdar2015_improvements import make_transform
    from scripts.icdar_budget_data import key
    plan=read(root/'resolved_configuration.json');check_fold(plan);c=plan['config'];complete=read(root/'training_complete.json')
    if read(root/'independent_training_audit.json')['status']!='PASS':raise ValueError('Audited decoder weights required')
    if file_hash(root/'final.pt')!=complete['checkpoint_sha256']:raise ValueError('Changed audited decoder checkpoint')
    out=root/'heldout'
    if out.exists():raise ValueError('Preserve existing heldout export')
    out.mkdir();runtime(c);net=WordRefitUNet(partial_encoder=False,skip_floor=c['skip_floor'],initial_gap_bias=c['initial_gap_bias'])
    state=torch.load(root/'final.pt',map_location='cpu',weights_only=False);net.load_state_dict(state['model_state_dict'],strict=True);del state
    net=net.to(c['device']).eval();ocr=recognizer(plan);transform=make_transform(c,c['image_hw'],augment=False)
    policy=read(plan['fold_contract']['global_emission_policy'])['selected']['policy'];settings=dict(policy,minimum_transcript_probability=0.)
    lookup={r['sample_id']:r for r in plan['train_records']};index={};torch.cuda.reset_peak_memory_stats();started=time.monotonic();crops=0
    try:
        for sid in plan['partitions']['decoder_heldout']:
            r=lookup[sid]
            if file_hash(r['image_path'])!=r['image_sha256']:raise ValueError('Changed heldout RGB')
            with Image.open(r['image_path']) as image:
                rgb=image.convert('RGB');x,_=transform(rgb,Image.new('L',rgb.size,0),0)
            with torch.inference_mode():prob=net(x[None].to(c['device']))[0].sigmoid().cpu().numpy()
            _,raw,expansion=read_rows(prob,r,settings,ocr,c['parseq_batch_size']);raw,rows=rows_with_sources(raw,policy['minimum_transcript_probability'],r['image_wh']);crops+=len(raw)
            mp=out/'maps'/(key(sid)+'.npz');mp.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(mp,probabilities=prob)
            path=out/'samples'/(key(sid)+'.json');write(path,dict(raw_rows=raw,rows=rows,image_wh=r['image_wh'],rgb_sha256=r['rgb_sha256'],image_sha256=r['image_sha256'],
                maps_sha256=file_hash(mp),maps_path=str(mp),decoder_fit_excluded=True,upstream_encoder_exposure=True,ground_truth_used=False,
                checkpoint_sha256=complete['checkpoint_sha256'],fold=plan['fold_contract']['fold'],expansion=expansion))
            index[sid]=dict(path=str(path),sha256=file_hash(path))
        write(out/'index.json',index);write(out/'complete.json',dict(status='COMPLETE',images=len(index),actual_parseq_crop_inferences=crops,
            actual_new_deepsolo_calls=0,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
            seconds=time.monotonic()-started,ground_truth_used=False,checkpoint_sha256=complete['checkpoint_sha256']))
    except BaseException as exc:
        write(out/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError)));raise
    finally:
        ocr.release();del net;torch.cuda.empty_cache()


def audit_export(root):
    import numpy as np
    from PIL import Image
    from scripts.crop_clip_runtime import array_sha
    from scripts.icdar_budget_data import key
    plan=read(root/'resolved_configuration.json');check_fold(plan);out=root/'heldout';index=read(out/'index.json');complete=read(out/'complete.json')
    assert complete['status']=='COMPLETE' and set(index)==set(plan['partitions']['decoder_heldout'])
    lookup={r['sample_id']:r for r in plan['train_records']};crops=0
    for sid,entry in index.items():
        assert file_hash(entry['path'])==entry['sha256'];sample=read(entry['path']);r=lookup[sid]
        assert not sample['ground_truth_used'] and sample['decoder_fit_excluded'] and sample['image_sha256']==r['image_sha256']
        assert sample['checkpoint_sha256']==complete['checkpoint_sha256'] and file_hash(sample['maps_path'])==sample['maps_sha256']
        with Image.open(r['image_path']) as image:rgb=np.asarray(image.convert('RGB'))
        for row in sample['raw_rows']:
            a,b,c,d=row['bbox'];assert array_sha(np.ascontiguousarray(rgb[b:d,a:c]))==row['crop_rgb_sha256'];crops+=1
    write(out/'independent_audit.json',dict(status='PASS',images=len(index),crop_hashes_checked=crops,
        index_sha256=file_hash(out/'index.json'),checkpoint_sha256=complete['checkpoint_sha256'],source_sha256=file_hash(__file__)))
    print('HELDOUT EXPORT AUDIT PASS',root,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--stage',choices=['train','audit','export','audit-export'],required=True);a=p.parse_args()
    {'train':train,'audit':audit,'export':export,'audit-export':audit_export}[a.stage](a.run_root)
