"""Generate/audit matching two-view conditional decoder-heldout observations."""

import argparse
import fcntl
import gc
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch

from scripts.run_uniform_unet_training import read, write
from scripts.uniform_decoder_folds import check_fold
from scripts.icdar_budget_data import key
from scripts.export_uniform_unet import rows_with_sources
from scripts.train_icdar13_word_refit import recognizer, read_rows
from scripts.run_icdar13_dense_instances import runtime
from scripts.crop_clip_runtime import array_sha
from segmentation.uniform_unet_fusion import checked_mode, file_hash
from segmentation.icdar13_word_refit import WordRefitUNet
from segmentation.icdar2015_improvements import make_transform
from recognition.uniform_encoder_context import raw_context
from recognition.uniform_multiscale_words import augment, iou


def prepare(path):
    c=read(path);checked_mode(path,c)
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed qualified bank input: '+name)
    u=read(c['utility_config'])
    for ds in c['dataset_order']:
        vp=read(c['source_validation_profiles'][ds])
        nn=read(Path(u['training_root'])/ds/'resolved_configuration.json')
        if (vp['word_augmentation']!=c['frontend_policy']['word_augmentation'] or
            nn['config']['image_hw']!=[c['frontend_policy']['native_canvas']]*2 or
            read(Path(vp['scaled_export'])/'complete.json')['input_hw']!=[c['frontend_policy']['secondary_canvas']]*2):
            raise ValueError('Matching-bank policy differs from the selected validation frontend')
    root=Path(c['output_root']);root.mkdir(parents=True,exist_ok=False)
    write(root/'resolved_configuration.json',c);repo=Path(__file__).resolve().parents[1];manifest={}
    for package in c['snapshot_packages']:
        for src in (repo/package).rglob('*.py'):
            relative=src.relative_to(repo);dest=root/'source_snapshot'/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(src,dest);manifest[str(relative)]=file_hash(dest)
    write(root/'source_manifest.json',manifest)
    write(root/'frontend_policy.json',c['frontend_policy'])
    write(root/'contract.json',dict(config_sha256=file_hash(path),source_manifest_sha256=file_hash(root/'source_manifest.json'),
        frontend_policy_sha256=file_hash(root/'frontend_policy.json'),ground_truth_used_for_prediction=False,test_used=False))


def forward(net,record,c,canvas):
    if file_hash(record['image_path'])!=record['image_sha256']:raise ValueError('Changed RGB')
    with Image.open(record['image_path']) as image:
        rgb=image.convert('RGB');x,_=make_transform(c,[canvas,canvas],augment=False)(rgb,Image.new('L',rgb.size,0),0)
    capture=[]
    hook=net.encoder.layer4.register_forward_hook(lambda m,a,v:capture.append(torch.cat((v.mean((2,3)),v.amax((2,3))),1)[0].detach()))
    try:
        with torch.inference_mode():prob=net(x[None].to(c['device']))[0].sigmoid().cpu().numpy()
    finally:hook.remove()
    if len(capture)!=1 or prob.shape!=(2,canvas//2,canvas//2):raise ValueError('Unexpected actual neural outputs')
    return prob,capture[0].cpu().numpy()


def independent_rows(native,large,policy):
    rows=[dict(r) for r in native]
    candidates=[(i,r) for i,r in enumerate(large) if r['sequence_confidence_native']>=policy['minimum_sequence_probability'] and any(x.isalnum() for x in r['raw_text'])]
    for i,row in sorted(candidates,key=lambda v:(-v[1]['sequence_confidence_native'],tuple(v[1]['bbox']),v[1]['raw_text'],v[0])):
        if len(rows)>=policy['maximum_candidates']:break
        if not any(iou(row['bbox'],r['bbox'])>=policy['maximum_overlap'] for r in rows):
            rows.append(dict(row,multiscale_added=True,scale_source_row=i,source_row=len(rows)))
    return rows


def fold(c,ds,source,original,out,ids,encoder,context):
    plan=read(source/'resolved_configuration.json');check_fold(plan);cfg=plan['config']
    trained=read(source/'training_complete.json');assert read(source/'independent_training_audit.json')['status']=='PASS'
    assert file_hash(source/'final.pt')==trained['checkpoint_sha256']
    runtime(cfg);net=WordRefitUNet(partial_encoder=False,skip_floor=cfg['skip_floor'],initial_gap_bias=cfg['initial_gap_bias'])
    state=torch.load(source/'final.pt',map_location='cpu',weights_only=False)
    assert state['plan_sha256']==file_hash(source/'resolved_configuration.json')
    assert all(torch.equal(state['model_state_dict'][k],v) for k,v in encoder.items())
    net.load_state_dict(state['model_state_dict'],strict=True);del state
    net=net.to(cfg['device']).eval();ocr=recognizer(plan);lookup={r['sample_id']:r for r in plan['train_records']}
    policy=read(plan['fold_contract']['global_emission_policy'])['selected']['policy']
    settings=dict(policy,native_size=c['frontend_policy']['secondary_canvas']//2,minimum_transcript_probability=0.)
    out.mkdir(parents=True,exist_ok=True);held=out/'heldout';held.mkdir(exist_ok=True)
    write(out/'source_neural_contract.json',dict(source_fold=str(source),plan_sha256=file_hash(source/'resolved_configuration.json'),
        checkpoint_sha256=trained['checkpoint_sha256'],frontend_policy_sha256=c['_frontend_sha'],
        selected_ids=ids,source_decoder_heldout_ids=plan['partitions']['decoder_heldout']))
    part=held/'partial_index.json';index=read(part) if part.exists() else {}
    if not set(index).issubset(ids):raise ValueError('Unexpected partial sample population')
    torch.cuda.reset_peak_memory_stats();started=time.monotonic();native_probes=0;new_images=0;new_crops=0
    try:
        for sid in ids[:c['neural_audit_images']]:
            r=lookup[sid];entry=original[sid];assert file_hash(entry['path'])==entry['sha256'];sample=read(entry['path'])
            prob,vector=forward(net,r,cfg,c['frontend_policy']['native_canvas'])
            with np.load(sample['maps_path'],allow_pickle=False) as z:np.testing.assert_array_equal(prob,z['probabilities'])
            np.testing.assert_array_equal(vector,context[sid]);native_probes+=1
        for sid in ids:
            if shutil.disk_usage(held).free<c['minimum_free_disk_bytes']:raise RuntimeError('Bank storage reserve exhausted; completed records remain indexed')
            r=lookup[sid];entry=original[sid]
            if file_hash(entry['path'])!=entry['sha256']:raise ValueError('Changed original native observation')
            native=read(entry['path'])
            assert native['checkpoint_sha256']==trained['checkpoint_sha256'] and native['decoder_fit_excluded'] and not native['ground_truth_used']
            assert file_hash(native['maps_path'])==native['maps_sha256']
            if sid in index:
                saved=read(index[sid]['path']);assert file_hash(index[sid]['path'])==index[sid]['sha256']
                assert saved['native_sample_sha256']==entry['sha256'] and file_hash(saved['scale_maps_path'])==saved['scale_maps_sha256']
                continue
            p=held/'samples'/(key(sid)+'.json')
            if p.exists():raise ValueError('Preserve unindexed existing sample: '+str(p))
            probability,vector=forward(net,r,cfg,c['frontend_policy']['secondary_canvas'])
            _,raw,expansion=read_rows(probability,r,settings,ocr,cfg['parseq_batch_size'])
            raw,_=rows_with_sources(raw,0.,r['image_wh']);combined=augment(native['raw_rows'],raw,**c['frontend_policy']['word_augmentation'])
            mp=held/'scale_maps'/(key(sid)+'.npz');mp.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(mp,probabilities=probability,encoder_context=vector)
            write(p,dict(raw_rows=combined,rows=combined,scale_raw_rows=raw,scale_expansion=expansion,
                image_wh=r['image_wh'],image_sha256=r['image_sha256'],rgb_sha256=r['rgb_sha256'],
                maps_path=native['maps_path'],maps_sha256=native['maps_sha256'],scale_maps_path=str(mp),scale_maps_sha256=file_hash(mp),
                native_sample_path=entry['path'],native_sample_sha256=entry['sha256'],ground_truth_used=False,decoder_fit_excluded=True,
                upstream_encoder_exposure=True,fold=plan['fold_contract']['fold'],checkpoint_sha256=trained['checkpoint_sha256'],
                frontend_policy_sha256=c['_frontend_sha']))
            index[sid]=dict(path=str(p),sha256=file_hash(p));write(part,index);new_images+=1;new_crops+=len(raw)
        assert set(index)==set(ids)
        crops=0;added=0
        for sid in ids:
            entry=index[sid];sample=read(entry['path']);assert file_hash(entry['path'])==entry['sha256']
            native=read(sample['native_sample_path']);assert file_hash(sample['native_sample_path'])==sample['native_sample_sha256']
            assert sample['raw_rows']==independent_rows(native['raw_rows'],sample['scale_raw_rows'],c['frontend_policy']['word_augmentation'])
            assert sample['raw_rows'][:len(native['raw_rows'])]==native['raw_rows']
            assert file_hash(sample['maps_path'])==sample['maps_sha256'] and file_hash(sample['scale_maps_path'])==sample['scale_maps_sha256']
            with np.load(sample['scale_maps_path'],allow_pickle=False) as z:
                size=c['frontend_policy']['secondary_canvas']//2
                assert z['probabilities'].shape==(2,size,size) and np.isfinite(z['probabilities']).all()
                assert z['encoder_context'].shape==(4096,) and np.isfinite(z['encoder_context']).all()
            with Image.open(lookup[sid]['image_path']) as im:rgb=np.asarray(im.convert('RGB'))
            for row in sample['scale_raw_rows']:
                a,b,x,y=row['bbox'];assert array_sha(np.ascontiguousarray(rgb[b:y,a:x]))==row['crop_rgb_sha256'];crops+=1
            added+=len(sample['raw_rows'])-len(native['raw_rows'])
        replays=0
        for sid in ids[:c['neural_audit_images']]:
            sample=read(index[sid]['path']);prob,vector=forward(net,lookup[sid],cfg,c['frontend_policy']['secondary_canvas'])
            with np.load(sample['scale_maps_path'],allow_pickle=False) as z:
                np.testing.assert_array_equal(prob,z['probabilities']);np.testing.assert_array_equal(vector,z['encoder_context'])
            replays+=1
        write(held/'index.json',index)
        write(held/'complete.json',dict(status='COMPLETE',mode='approximate',claim_label='approximate reproduction',images=len(ids),
            actual_new_scaled_unet_images=new_images,actual_native_probe_images=native_probes,actual_independent_scaled_replay_images=replays,
            actual_new_parseq_crop_inferences=new_crops,actual_new_deepsolo_calls=0,checkpoint_sha256=trained['checkpoint_sha256'],
            peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),out_of_memory=False,
            seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(0),precision='float32',unet_batch_size=1,parseq_batch_size=cfg['parseq_batch_size'],
            native_hw=[c['frontend_policy']['native_canvas']//2]*2,secondary_input_hw=[c['frontend_policy']['secondary_canvas']]*2,
            secondary_logits_hw=[c['frontend_policy']['secondary_canvas']//2]*2,
            model_set='One unchanged frozen-encoder word U-Net, genuine PARSeq; no DeepSolo forward',frontend_policy_sha256=c['_frontend_sha']))
        write(held/'independent_audit.json',dict(status='PASS',images=len(ids),index_sha256=file_hash(held/'index.json'),
            source_fold=str(source),source_fold_export_audit_sha256=file_hash(source/'heldout/independent_audit.json'),
            checkpoint_sha256=trained['checkpoint_sha256'],decoder_fit_excluded=True,crop_hashes_checked=crops,added_candidates=added,
            native_neural_probes=native_probes,independent_scaled_neural_replays=replays,all_encoder_tensors_equal_native=True,
            frontend_policy_sha256=c['_frontend_sha'],source_sha256=file_hash(__file__),
            scope='Exact native neural/context parity, independent second-scale forward replays, all RGB crop hashes and word admission; no independent OCR model/refit. Conditional upstream-exposed decoder folds only.'))
        print('MATCHED FOLD PASS',ds,out.parent.name,out.name,'images',len(ids),'added',added,flush=True)
    except BaseException as exc:
        write(held/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),out_of_memory=isinstance(exc,torch.cuda.OutOfMemoryError)))
        raise
    finally:
        ocr.release();del net,ocr;gc.collect();torch.cuda.empty_cache()


def run(path):
    c=read(path);checked_mode(path,c);root=Path(c['output_root']);contract=read(root/'contract.json')
    assert contract['config_sha256']==file_hash(path)
    for name,digest in read(root/'source_manifest.json').items():
        assert file_hash(root/'source_snapshot'/name)==digest
    for name,digest in c['pinned_inputs'].items():
        if file_hash(name)!=digest:raise ValueError('Changed pinned bank source: '+name)
    c['_frontend_sha']=contract['frontend_policy_sha256'];u=read(c['utility_config'])
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(root/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),config_sha256=file_hash(path)))
        for role in ('smoke','full'):
            for ds in c['dataset_order']:
                native_root=Path(u['training_root'])/ds;plan=read(native_root/'resolved_configuration.json')
                lookup={r['sample_id']:r for r in plan['train_records']};original=read(Path(u['fit_export_overrides'][ds])/'index.json')
                old_audit=read(Path(u['fit_export_overrides'][ds])/'independent_audit.json')
                state=torch.load(native_root/'best.pt',map_location='cpu',weights_only=False)['model_state_dict']
                encoder={k:v for k,v in state.items() if k.startswith('encoder.')};del state
                context=raw_context(u,{sid:lookup[sid] for sid in plan['partitions']['fit']})
                combined={};folds=[]
                for old in old_audit['folds'][:1 if role=='smoke' else c['decoder_folds']]:
                    source=Path(old['root']);p=read(source/'resolved_configuration.json')
                    ids=p['partitions']['decoder_heldout'][:c['smoke_images']] if role=='smoke' else p['partitions']['decoder_heldout']
                    out=root/role/ds/('fold_'+str(old['fold']))
                    while True:
                        free=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip()
                        if len(free.splitlines())!=1:raise ValueError('Single GPU required')
                        if int(free.split(',')[-2])*1024**2>=c['minimum_free_gpu_bytes']:break
                        write(root/'status.json',dict(status='WAITING_GPU',pid=os.getpid(),observed=free));time.sleep(20)
                    write(root/'status.json',dict(status='GENERATING_AND_AUDITING',pid=os.getpid(),role=role,dataset=ds,fold=old['fold'],gpu=free))
                    if not (out/'heldout/independent_audit.json').exists():fold(c,ds,source,original,out,ids,encoder,context)
                    audit=read(out/'heldout/independent_audit.json');assert audit['status']=='PASS' and audit['frontend_policy_sha256']==c['_frontend_sha']
                    index=read(out/'heldout/index.json');assert set(index)==set(ids) and not set(index)&set(combined)
                    assert audit['index_sha256']==file_hash(out/'heldout/index.json');combined.update(index)
                    folds.append(dict(fold=old['fold'],root=str(out),audit_sha256=file_hash(out/'heldout/independent_audit.json')))
                expected=set(plan['partitions']['fit']) if role=='full' else set(ids)
                assert set(combined)==expected
                bank=root/role/ds/'bank';write(bank/'index.json',combined)
                write(bank/'independent_audit.json',dict(status='PASS',images=len(combined),index_sha256=file_hash(bank/'index.json'),
                    source_training_plan_sha256=file_hash(native_root/'resolved_configuration.json'),decoder_fit_excluded=True,
                    upstream_encoder_exposure=True,folds=folds,frontend_policy_sha256=c['_frontend_sha'],smoke=role=='smoke',
                    scope='Matching native/1024 word additions from all declared conditional decoder-fold weights. No TEST or fresh pretraining-unseen OOF claim.'))
                print('MATCHED BANK PASS',role,ds,len(combined),flush=True)
        write(root/'status.json',dict(status='ALL_MATCHED_BANKS_AUDITED',pid=os.getpid(),test_used=False,actual_new_deepsolo_calls=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--stage',choices=['prepare','run'],required=True);a=p.parse_args()
    try:prepare(a.config) if a.stage=='prepare' else run(a.config)
    except BaseException as exc:
        root=Path(read(a.config)['output_root'])
        if root.exists():write(root/f'failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
