"""New Kernel U-Net, fresh decoder/head, local ImageNet encoder only."""
import argparse
from dataclasses import asdict
import fcntl
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import traceback

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from models.kernel_only_fresh import FreshKernelAGUNet
from scripts import train_kernel_only as T
from scripts import evaluate_kernel_only_padding as P

REPO=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=REPO/'configs/approximate/kernel_only_fresh_v1.json'
FILES=tuple(sorted(set(T.FILES+('scripts/train_kernel_only_fresh.py','models/kernel_only_fresh.py'))))


def read_config(path,mode):
    path=Path(path).resolve()
    if mode!='approximate' or REPO/'configs/approximate' not in path.parents:raise ValueError('Explicit approximate configuration required')
    cfg=json.loads(path.read_text())
    if (cfg.get('schema_version')!=1 or cfg.get('reproduction_mode')!=mode
        or cfg.get('claim_label')!='approximate reproduction' or cfg.get('experiment_kind')!='fresh_unet_kernel_supervision'
        or cfg.get('initialization')!='fresh_decoder_head_imagenet_encoder_only'
        or cfg.get('segmentation_checkpoint_initialization') is not False or 'source_checkpoint_sha256' in cfg
        or cfg.get('supervision')!='kernel_only' or cfg.get('encoder')!='frozen_including_BN'
        or cfg.get('precision')!='float32' or cfg.get('allow_tf32') is not False
        or cfg.get('deterministic_algorithms') is not True
        or cfg.get('selection')!='macro_kernel_IoU_then_macro_kernel_F1_then_earliest_include_epoch0'):
        raise ValueError('Fresh initialization and exclusive Kernel supervision are required')
    protocol_path=(REPO/cfg['protocol_config']).resolve()
    if REPO/'configs/approximate' not in protocol_path.parents or T.sha256_file(protocol_path)!=cfg['protocol_config_sha256']:raise ValueError('Protocol configuration changed')
    padding,source,preprocessing=P.read_config(protocol_path,mode)
    if cfg['seed']!=source['seed'] or cfg['batch_size']!=source['batch_size']:raise ValueError('Frozen seed/batch changed')
    if any(type(cfg[k]) is not int or cfg[k]<minimum for k,minimum in [('epochs',1),('num_workers',0)]):raise ValueError('Invalid epochs/workers')
    if set(cfg['loss'])!={'bce_weight','dice_weight','smooth'} or not np.isfinite(list(cfg['loss'].values())).all() or min(cfg['loss'].values())<=0:raise ValueError('Invalid Kernel loss')
    T.training_config(cfg)
    return cfg,padding,source,preprocessing


def build_fresh_model(cfg,source,encoder_checkpoint):
    if T.sha256_file(encoder_checkpoint)!=cfg['encoder_checkpoint_sha256']:raise ValueError('ImageNet encoder checksum mismatch')
    model=FreshKernelAGUNet(skip_floor=source['skip_floor'],encoder_pretrained=False)
    state=torch.load(encoder_checkpoint,map_location='cpu',weights_only=True)
    initialization=model.initialize_encoder(state);del state
    h=sha256()
    for name,value in model.state_dict().items():
        if not name.startswith('encoder.'):
            h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    initialization.update(encoder_checkpoint_sha256=cfg['encoder_checkpoint_sha256'],decoder_head_sha256=h.hexdigest())
    return model,initialization


def run(args):
    cfg,padding,source,preprocessing=read_config(args.config,args.mode)
    output=args.output_root.resolve();reference=args.reference_run.resolve();word_root=args.word_root.resolve();artifacts=args.parseq_artifacts.resolve();encoder=args.encoder_checkpoint.resolve()
    if output==REPO or REPO in output.parents or any(output==r or r in output.parents for r in (reference,word_root)):raise ValueError('Separate external output required')
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():raise RuntimeError('Fresh Kernel training requires CUDA; no CPU fallback')
    device=torch.device(args.device);device=torch.device('cuda',torch.cuda.current_device() if device.index is None else device.index)
    if T.sha256_file(reference/'results.json')!=cfg['reference_results_sha256']:raise ValueError('Reference results changed')
    if T.sha256_file(artifacts/'parseq-bb5792a6.pt')!=T.PARSEQ_CHECKPOINT_SHA256 or T.sha256_file(artifacts/('parseq-'+T.PARSEQ_RUNTIME_REVISION+'.tar.gz'))!=T.PARSEQ_RUNTIME_ARCHIVE_SHA256:raise ValueError('Qualified PARSeq artifacts changed')
    manifest_path=word_root/source['train_manifest']
    if T.sha256_file(manifest_path)!=source['word_manifest_sha256']:raise ValueError('GT manifest changed')
    manifest=T.load_segmentation_manifest(manifest_path,require_checksums=True)
    train_ids,val_ids,split=T._split_indices(manifest,cfg['seed'],source['validation_count'])
    if manifest.dataset!='total_text' or manifest.split_role!='train' or len(train_ids)!=1130 or len(val_ids)!=125 or split!=source['validation_ids_sha256']:raise ValueError('GT population/split changed')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');T.seed_everything(cfg['seed'],deterministic_algorithms=True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;cv2.setNumThreads(1)
    model,initialization=build_fresh_model(cfg,source,encoder)
    contract={'resolved_config':cfg,'padding_config':padding,'source_config':source,'resolved_preprocessing':preprocessing,
        'initialization':initialization,'reference_use':'metrics and OCR cache only; no segmentation weights',
        'validation_ids_sha256':split,'active_loss_components':['kernel'],'training_config':asdict(T.training_config(cfg)),
        'runtime_inputs':{'reference_run':str(reference),'word_root':str(word_root),'parseq_artifacts':str(artifacts),'encoder_checkpoint':str(encoder)},
        'device':str(device),'gpu_name':torch.cuda.get_device_name(device),'backend':P.source_runner.backend_state(),
        'source_sha256':{f:T.sha256_file(REPO/f) for f in FILES},'environment':T.runtime_environment()}
    contract=json.loads(json.dumps(contract));output.mkdir(parents=True,exist_ok=True)
    with (output/'runner.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=output/'run_contract.json'
        if path.exists() and json.loads(path.read_text())!=contract:raise ValueError('Fresh run contract changed; preserve outputs')
        T._json(path,contract)
        initial_path=output/'initial_state.pt'
        if not initial_path.exists():
            if (output/'last.pt').exists() or (output/'best.pt').exists():raise ValueError('Training outputs lack original fresh initialization')
            torch.save({'schema_version':1,'reproduction_mode':'approximate','claim_label':'approximate reproduction',
                'architecture':model.architecture_identity,'model_state_dict':model.state_dict(),'run_contract':contract,
                'initialization':initialization},initial_path)
        def reset_initial():
            T.load_checkpoint(initial_path,model=model,expected_architecture=model.architecture_identity,expected_run_contract=contract)
        reset_initial()
        transforms=[T._transform(preprocessing,'letterbox',train=v) for v in (True,False)]
        datasets=[T.WordKernelDataset(T.SegmentationDataset(manifest,word_root,transform=t),distance_fraction=source['targets']['distance_fraction']) for t in transforms]
        sampler=T.EpochShuffleSampler(len(train_ids),seed=cfg['seed']);options={'num_workers':cfg['num_workers'],'pin_memory':True,'timeout':source['loader_timeout_seconds'] if cfg['num_workers'] else 0}
        train_loader=DataLoader(Subset(datasets[0],train_ids),batch_size=cfg['batch_size'],sampler=sampler,**options)
        val_loader=DataLoader(Subset(datasets[1],val_ids),batch_size=cfg['batch_size'],shuffle=False,**options)
        T._prepare_cuda_memory_tracking(device);model.to(device)
        def optimizer():return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],betas=tuple(cfg['adam_betas']),eps=cfg['adam_epsilon'],weight_decay=cfg['weight_decay'])
        if not (output/'smoke.json').exists():
            opt=optimizer();before=model.head.weight.detach().clone()
            smoke=T.train_kernel_epoch(model,[next(iter(train_loader))],opt,device,cfg)
            update=float((model.head.weight.detach()-before).abs().max());del opt,before
            if smoke['steps']!=1 or update<=0:raise RuntimeError('Fresh Kernel CUDA smoke did not update its head')
            T._json(output/'smoke.json',{'status':'PASS','result_class':'smoke-test result','train':smoke,'head_update_linf':update,
                'initialization':initialization,'memory':T.memory(device,source),'run_contract':contract})
            reset_initial()
        T.seed_everything(cfg['seed'],deterministic_algorithms=True)
        opt=optimizer()
        trained=T.train_loop(output,model,opt,train_loader,val_loader,transforms[0],sampler,cfg,source,contract,device,args.stop_after_epoch)
        del opt
        if not trained['training_complete']:return trained
        T.load_checkpoint(output/'best.pt',model=model,expected_architecture=model.architecture_identity,expected_run_contract=contract,expected_training_config=T.training_config(cfg))
        metrics=T.evaluate_kernel(model,val_loader,device,source,export_dir=output/'validation_exports')
        T._json(output/'validation.json',{'aggregate':metrics,'checkpoint':trained['checkpoint'],'run_contract':contract,'result_class':'approximate reproduction'})
        del model;gc.collect();torch.cuda.empty_cache()
        result=T.padding_ocr(output,reference,manifest,val_ids,word_root,padding,source,artifacts,device)
        result.update(reproduction_mode='approximate',run_contract=contract,initialization=initialization,
            trained_checkpoint=trained['checkpoint'],completed_epochs=cfg['epochs'],training_memory=trained['memory'])
        T._json(output/'results.json',result)
        report=output/'comparison.md';report.write_text(report.read_text().replace('Kernel-only supervised fine-tuning','Fresh U-Net trained on Kernel GT',1))
        T._json(output/'suite_success.json',{'status':'SUCCESS','completed_epochs':cfg['epochs'],'results_sha256':T.sha256_file(output/'results.json'),'result_class':'approximate reproduction'})
        return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['approximate'],required=True)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--device',required=True)
    for name in ('reference-run','word-root','parseq-artifacts','output-root','encoder-checkpoint'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--stop-after-epoch',type=int);return p


if __name__=='__main__':
    args=parser().parse_args()
    try:run(args)
    except Exception as error:
        args.output_root.mkdir(parents=True,exist_ok=True)
        T._json(args.output_root/'failure.json',{'status':'FAILED','error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc(),
            'oom':isinstance(error,torch.cuda.OutOfMemoryError),'device':args.device,'precision':'float32',
            'model_set':'new one-output Kernel U-Net, later PARSeq','segmentation_input_shape':[8,3,512,512],'native_shape':[8,1,256,256],
            'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None,
            'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else None})
        raise
