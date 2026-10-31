"""Versioned APPROXIMATE multi-scale word-slot training and paired OCR battery.

No dependency installation. No STRICT lookup. Source checkpoints, data roles,
cache bytes, all branch configurations and inference budgets remain auditable.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from models.word_slot_unet import WordSlotUNet, sampling_grid, deterministic_sample
from recognition.parseq import ParseqApproxConfig, ParseqApproxRecognizer
from recognition.word_slot_teachers import load_ccd, load_ccd_glyph, load_mae, character_vectors
from segmentation.word_slot_data import (read, write, sha, sample_key, build_index, load_rgb,
                                         image_tensor, target_for, canvas_to_original,
                                         original_to_canvas)
from segmentation.word_slot_objectives import (instance_loss, make_denoising, denoising_loss,
                                               strip_regularization, pcgrad_two,
                                               character_contrastive, pairwise_iou)
from segmentation.totaltext_word_kernel import expand_word_kernels, instance_proposals
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from recognition.word_slot_evaluation import measure as word_measure
from recognition.scene_text_v1_evaluation import box_iou

REPO=Path(__file__).resolve().parents[1]
_CACHE_VERIFIED={}
_CACHE_AUDITS={}


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def source_hashes():
    paths=['scripts/run_word_slot_methods.py','models/word_slot_unet.py',
           'models/unet_adv.py','models/totaltext_word_kernel.py',
           'segmentation/word_slot_objectives.py','segmentation/word_slot_data.py',
           'recognition/word_slot_teachers.py','recognition/parseq.py']
    return {name:sha(REPO/name) for name in paths}


def setup(config):
    runtime=config['runtime']
    supported={'device':'cuda:0','precision':'float32','feature_cache_dtype':'float32','rgb_grid_device':'cpu',
               'deterministic_algorithms':True,'allow_tf32':False}
    if any(runtime.get(key)!=value for key,value in supported.items()):
        raise ValueError('Unqualified word-slot runtime: requires CUDA0, FP32 priors/models, CPU grids, deterministic algorithms and TF32 disabled')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=runtime['cublas_workspace_config']:
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG before launching Python')
    torch.set_num_threads(runtime['cpu_threads']);torch.set_num_interop_threads(1)
    seed=config['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(runtime['deterministic_algorithms'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=runtime['allow_tf32'];torch.backends.cudnn.allow_tf32=runtime['allow_tf32']
    if not torch.cuda.is_available():raise RuntimeError('This execution contract requires CUDA')
    free,total=torch.cuda.mem_get_info()
    if free<runtime['minimum_free_gpu_bytes']:raise RuntimeError('Insufficient free GPU memory; no precision fallback allowed')
    return {'gpu':torch.cuda.get_device_name(),'free_gpu_bytes':free,'total_gpu_bytes':total,
            'torch':torch.__version__,'cuda':torch.version.cuda,'python':platform.python_version(),
            'nvidia_smi':subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free','--format=csv,noheader'],
                                       check=True,capture_output=True,text=True).stdout.strip(),**runtime}


def memory():
    torch.cuda.synchronize()
    return {'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'oom':False}


def make_model(config, arm=None):
    options=dict(config['architecture'])
    if arm is not None:options['mask_initialization']=config['arms'][arm].get('mask_initialization',False)
    return WordSlotUNet(**options)


def load_prior(model,config):
    path=config['paths']['source_checkpoint']
    if sha(path)!=config['input_sha256']['source_checkpoint']:raise ValueError('Prior checkpoint changed')
    # Qualified local checkpoint also stores optimizer and NumPy RNG state.
    # Its exact SHA256 is checked before this explicitly trusted local load.
    data=torch.load(path,map_location='cpu',weights_only=False)
    if data['run_contract']!=read(config['paths']['source_contract']):raise ValueError('Prior training contract mismatch')
    return model.initialize_prior(data['model_state_dict'])


def save_delta(path,model,config,**metadata):
    state={k:v.detach().cpu() for k,v in model.state_dict().items() if k not in model.base_names}
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial');torch.save({'state':state,'prior_sha256':config['input_sha256']['source_checkpoint'],**metadata},temp);temp.replace(path)


def load_delta(model,path,config):
    data=torch.load(path,map_location='cpu',weights_only=True)
    if data['prior_sha256']!=config['input_sha256']['source_checkpoint']:raise ValueError('Wrong delta prior')
    own=model.state_dict();expected=set(own)-set(model.base_names)
    if set(data['state'])!=expected:raise ValueError('Missing or extra learned decoder tensors')
    own.update(data['state']);model.load_state_dict(own,strict=True)
    return data


def prior_cache_path(root,record):return root/'prior_cache'/record['phase']/(sample_key(record['sample_id'])+'.pt')


def checked_cache(root,record):
    path=prior_cache_path(root,record);root_key=str(root)
    if root_key not in _CACHE_AUDITS:_CACHE_AUDITS[root_key]=read(root/'cache_audit.json')['hashes']
    stat=path.stat();signature=(stat.st_size,stat.st_mtime_ns)
    if _CACHE_VERIFIED.get(str(path))!=signature:
        expected=_CACHE_AUDITS[root_key][str(path.relative_to(root))]
        if sha(path)!=expected:raise ValueError('FP32 feature cache bytes changed: '+str(path))
        _CACHE_VERIFIED[str(path)]=signature
    item=torch.load(path,map_location='cpu',weights_only=True)
    if item['image_sha256']!=record['image_sha256']:raise ValueError('Wrong cached sample')
    if any(t.dtype!=torch.float32 for t in item['pyramid']):raise ValueError('Feature precision changed')
    return item


def cache_load(root,record,device):
    item=checked_cache(root,record)
    return tuple(t.to(device) for t in item['pyramid']),{k:v.to(device) for k,v in item['target'].items() if isinstance(v,torch.Tensor)}


def prepare(root,config,hardware,limit):
    index,inputs=build_index(config,limit)
    plan={'config':config,'config_digest':digest(config),'index':index,'dataset_sha256':inputs,
          'runtime':hardware,'mode':'approximate','claim_label':'approximate reproduction',
          'result_class':'smoke-test result' if limit else 'approximate reproduction; one-seed fixed-budget research screen',
          'selection_uses_test':False,'test_historically_explored':True,
          'code_sha256_at_prepare':source_hashes()}
    path=root/'protocol.json'
    if path.exists():
        old=read(path)
        for key in ('config_digest','index','dataset_sha256'):
            if old[key]!=plan[key]:raise ValueError('Existing run input contract changed: '+key)
    else:write(path,plan)
    model=make_model(config);initialization=load_prior(model,config);model.eval().to(config['runtime']['device'])
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();hashes={};empty=[]
    for phase,records in index.items():
        for i,record in enumerate(records,1):
            path=prior_cache_path(root,record)
            if path.exists():
                cached=torch.load(path,map_location='cpu',weights_only=True)
                if cached['image_sha256']!=record['image_sha256'] or cached['prior_sha256']!=config['input_sha256']['source_checkpoint']:
                    raise ValueError('Changed existing feature cache')
            else:
                rgb=load_rgb(record);pyramid,pixel=model.prior_pyramid(image_tensor(rgb).to(config['runtime']['device']))
                target=target_for(record,config['training']['mask_size']) if phase!='test' else {}
                prob=pixel[0].sigmoid().cpu().numpy();initial=config['initialization']
                canvas=np.asarray(letterbox_canvas_mask(tuple(record['image_wh']),native_size=prob.shape[-1]),bool)
                labels,_=expand_word_kernels((prob[0]>=initial['text_threshold'])&canvas,
                                              (prob[1]>=initial['kernel_threshold'])&canvas)
                proposals=instance_proposals(labels)
                target['prior_boxes']=torch.tensor([p['bbox_native'] for p in proposals[:initial['maximum_predicted_seeds']]],dtype=torch.float32).reshape(-1,4)/prob.shape[-1]
                cached={'pyramid':tuple(t.cpu() for t in pyramid),'pixel_logits':pixel.cpu(),
                        'target':target,'image_sha256':record['image_sha256'],'prior_sha256':config['input_sha256']['source_checkpoint']}
                path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.partial');torch.save(cached,temp);temp.replace(path)
            hashes[str(path.relative_to(root))]=sha(path)
            if phase!='test' and len(cached['target']['empty_mask_indices']):
                empty.append({'phase':phase,'sample_id':record['sample_id'],
                              'indices':cached['target']['empty_mask_indices'].tolist(),
                              'treatment':'all original GT words retained in bbox losses and metric denominators'})
            if i%50==0 or i==len(records):print('CACHE',phase,i,len(records),flush=True)
    # Checkpoint-compatible ordinary pixels are also a useful independent control.
    audit={'status':'PASS','initialization':initialization,'hashes':hashes,'empty_resized_masks':empty,
           'seconds':time.perf_counter()-tick,'cache_dtype':'float32','base_parameters_frozen':True,
           'models':['qualified frozen pixel U-Net'],'batch_size':1,'input_shape':[1,3,512,512],**memory()}
    write(root/'cache_audit.json',audit)
    del model;gc.collect();torch.cuda.empty_cache()


def reader(config):
    p=config['paths'];r=ParseqApproxRecognizer(ParseqApproxConfig(Path(p['parseq_source']),Path(p['parseq_checkpoint']),
                     device=config['runtime']['device'],runtime_source_type='github_commit_archive',runtime_source_archive=Path(p['parseq_archive'])))
    for parameter in r.model.parameters():parameter.requires_grad_(False)
    return r


def trainable(model):return [p for p in model.parameters() if p.requires_grad]


def target_device(item,device):return {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in item.items()}


def teacher_preprocessing(kind):
    if kind not in ('mae','ccd','glyph'):raise ValueError('Unknown crop teacher')
    if kind=='mae':
        return {'normalization_mean':[.5,.5,.5],'normalization_std':[.5,.5,.5],
                'pil_interpolation':'bicubic','crop_size_hw':[32,128]}
    return {'normalization_mean':[.485,.456,.406],'normalization_std':[.229,.224,.225],
            'pil_interpolation':'bilinear','crop_size_hw':[32,128]}


def validate_teacher_entry(entry,kind,checkpoint_sha,image_sha):
    if entry['teacher_sha256']!=checkpoint_sha or entry['image_sha256']!=image_sha:
        raise ValueError('Crop teacher cache checkpoint/RGB changed')
    if kind=='mae':
        expected=teacher_preprocessing(kind);saved=entry.get('input_preprocessing',{})
        if any(saved.get(key)!=value for key,value in expected.items()):
            raise ValueError('Unqualified MAE pretraining targets: run qualify_word_slot_mae_teacher before initializing its branch, or use a fresh output directory')


def matched_crop_indices(pair,record,max_crops,generator,tokenizer):
    left,right=pair;valid=[]
    for i,j in zip(left.tolist(),right.tolist()):
        text=record['ground_truth'][j]['text']
        if 0<len(text)<=25 and all(c in tokenizer._stoi for c in text):valid.append((i,j,text))
    valid.sort(key=lambda value:value[1])
    if len(valid)>max_crops:
        order=torch.randperm(len(valid),generator=generator)[:max_crops].tolist();valid=[valid[i] for i in order]
    return valid


def normalized_edit(a,b):
    a,b=a.casefold(),b.casefold();row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]/max(len(a),len(b),1)


def ocr_objective(output,record,pair,recognizer,options,training,generator):
    choices=matched_crop_indices(pair,record,training['ocr_crops_per_image'],generator,recognizer.model.tokenizer)
    if not choices:return output['boxes'].sum()*0,{'ocr_crops':0,'unsupported_or_long_words':len(pair[0])}
    ids=torch.tensor([i for i,_,_ in choices],device=output['boxes'].device)
    boxes=canvas_to_original(output['boxes'][ids],record['image_wh'])
    scale=boxes.new_tensor(record['image_wh']*2);boxes=boxes/scale
    offsets=output['strip_offsets'][ids] if options.get('strip') else None
    grid=sampling_grid(boxes,offsets,height=32,width=128,strength=training['strip_strength'])
    rgb=TF.to_tensor(load_rgb(record))
    crops=deterministic_sample(rgb,grid,output['boxes'].device)*2-1
    system=recognizer.model;model=system.model;tokenizer=system.tokenizer
    labels=tokenizer.encode([text for _,_,text in choices],device=crops.device)
    memory_tokens=model.encode(crops)
    inp,out=labels[:,:-1],labels[:,1:]
    length=inp.shape[1]
    mask=torch.triu(torch.ones(length,length,dtype=torch.bool,device=crops.device),1)
    hidden=model.decode(inp,memory_tokens,tgt_mask=mask,tgt_padding_mask=inp==tokenizer.pad_id,tgt_query_mask=mask)
    logits=model.head(hidden)
    per_token=F.cross_entropy(logits.flatten(0,1),out.flatten(),ignore_index=tokenizer.pad_id,reduction='none').reshape(len(choices),-1)
    per_word=per_token.sum(1)/(out!=tokenizer.pad_id).sum(1).clamp_min(1)
    allocation=None
    if options.get('mcl'):
        with torch.no_grad():
            decoded,_=tokenizer.decode(system(crops.detach()).softmax(-1))
        deep=output['deep_predictions'];allocation=[]
        for (_,j,text),hypothesis in zip(choices,decoded):
            candidates=[p['text'] for p in deep if p.get('bbox') is not None and box_iou(p['bbox'],record['ground_truth'][j]['bbox'])>=.5]
            deep_risk=min((normalized_edit(candidate,text) for candidate in candidates),default=1.)
            allocation.append(normalized_edit(hypothesis,text)<=deep_risk)
        # Both experts are allocated using the SAME bounded word edit risk.
        # A nonwinning student receives a predeclared exploration weight.
        weights=per_word.new_tensor([1. if win else training['mcl_exploration'] for win in allocation])
        loss=(per_word*weights).mean()
    elif options.get('specialist'):
        weights=output['specialist_weights'][[j for _,j,_ in choices]]
        loss=(per_word*weights).mean()
    else:loss=per_word.mean()
    loss=training['ocr_weight']*loss
    if options.get('strip'):loss=loss+training['strip_regularization']*strip_regularization(grid,training['minimum_grid_step'])
    return loss,{'ocr_crops':len(choices),'ocr_ce':float(per_word.mean().detach()),
                 'mcl_student_wins':sum(allocation) if allocation is not None else 0,
                 'unsupported_or_long_words':len(pair[0])-len(matched_crop_indices(pair,record,len(pair[0]),generator,tokenizer))}


def character_objective(model,output,record,pair,root,options,training):
    kind=options.get('teacher','ccd')
    path=root/'teacher_cache'/kind/(sample_key(record['sample_id'])+'.pt')
    data=torch.load(path,map_location='cpu',weights_only=True)
    ids,gt=pair
    if not len(ids):return output['boxes'].sum()*0,{}
    boxes=output['boxes'][ids]
    grid=sampling_grid(boxes,None,height=8,width=32,strength=0)
    sampled=deterministic_sample(output['features'][0],grid,boxes.device)
    crop_features=sampled
    sampled=model.character_projection(sampled)
    lengths=[len(record['ground_truth'][j]['text']) for j in gt.tolist()]
    lengths=[min(n,25) for n in lengths]
    student=character_vectors(sampled,lengths)
    teacher=torch.cat([data['characters'][j][:n] for j,n in zip(gt.tolist(),lengths)]).to(boxes.device)
    labels=torch.tensor([ord(c) for j,n in zip(gt.tolist(),lengths) for c in record['ground_truth'][j]['text'][:n]],device=boxes.device)
    loss=output['boxes'].sum()*0;stats={}
    if options.get('ccd'):
        contrastive=character_contrastive(student,teacher,labels,training['character_temperature'])
        loss=loss+training['character_weight']*contrastive;stats['contrastive']=float(contrastive.detach())
    if options.get('glyph'):
        # SIGA-inspired self-guidance: frozen crop-teacher affinity guides the
        # spatial distribution of character-compatible U-Net features. This
        # is an adaptation, not the unavailable original SIGA implementation.
        glyph_data=torch.load(root/'teacher_cache'/'glyph'/(sample_key(record['sample_id'])+'.pt'),map_location='cpu',weights_only=True)
        truth=glyph_data['glyph'][gt.cpu()].to(boxes.device)
        prediction=model.glyph_head(crop_features)
        guidance=F.binary_cross_entropy_with_logits(prediction,truth)
        loss=loss+training['glyph_weight']*guidance;stats['character_guidance']=float(guidance.detach())
    return loss,stats


def teacher_cache(root,config):
    index=read(root/'protocol.json')['index'];device=config['runtime']['device'];audits={}
    previous_audits=read(root/'teacher_cache_audit.json') if (root/'teacher_cache_audit.json').exists() else {}
    for kind in ('ccd','mae','glyph'):
        preprocessing=teacher_preprocessing(kind)
        segmentation=None
        if kind=='ccd':teacher,audit=load_ccd(config['ccd'],device)
        elif kind=='mae':teacher,audit=load_mae(config['mae'],device)
        else:teacher,segmentation,audit=load_ccd_glyph(config['ccd'],device)
        write(root/(kind+'_loading_audit.json'),audit)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter();count=0
        with torch.no_grad():
            for phase in ('fit','calibration'):
                for i,record in enumerate(index[phase],1):
                    path=root/'teacher_cache'/kind/(sample_key(record['sample_id'])+'.pt')
                    if path.exists():
                        validate_teacher_entry(torch.load(path,map_location='cpu',weights_only=True),kind,
                              config['mae' if kind=='mae' else 'ccd']['checkpoint_sha256'],record['image_sha256'])
                        continue
                    rgb=load_rgb(record);crops=[]
                    for word in record['ground_truth']:
                        interpolation=Image.Resampling.BICUBIC if preprocessing['pil_interpolation']=='bicubic' else Image.Resampling.BILINEAR
                        crop=TF.to_tensor(rgb.crop(tuple(map(int,word['bbox']))).resize((128,32),interpolation))
                        crops.append(TF.normalize(crop,preprocessing['normalization_mean'],preprocessing['normalization_std']))
                    values=[];glyphs=[]
                    for offset in range(0,len(crops),16):
                        batch=torch.stack(crops[offset:offset+16]).to(device)
                        if kind=='mae':tokens=teacher(batch)[:,1:]
                        else:tokens,intermediate=teacher(batch)
                        if kind=='glyph':glyphs.append(F.avg_pool2d(segmentation(intermediate).softmax(1)[:,1:2],4).cpu())
                        values.extend(tokens.cpu().split(1))
                    characters=[character_vectors(token,[min(len(word['text']),25)]) for token,word in zip(values,record['ground_truth'])]
                    entry={'characters':characters,'teacher_sha256':config['mae' if kind=='mae' else 'ccd']['checkpoint_sha256'],
                           'image_sha256':record['image_sha256'],'alignment':'uniform horizontal character bins, approximate',
                           'input_preprocessing':preprocessing}
                    if kind=='glyph':entry['glyph']=torch.cat(glyphs)
                    path.parent.mkdir(parents=True,exist_ok=True);torch.save(entry,path)
                    count+=len(crops)
                    if i%100==0:print('TEACHER',kind,phase,i,flush=True)
        audits[kind]=previous_audits[kind] if count==0 and kind in previous_audits else {'status':'PASS','new_crops':count,'seconds':time.perf_counter()-start,
           'models':[kind+' frozen crop teacher'],'precision':'float32','batch_size':16,'input_shape':[16,3,32,128],**memory()}
        audits[kind]['input_preprocessing']=preprocessing
        del teacher,segmentation;gc.collect();torch.cuda.empty_cache()
    write(root/'teacher_cache_audit.json',audits)


def prepare_deep(root,config):
    from recognition.clip_router_evaluation import deepsolo_predictions
    index=read(root/'protocol.json')['index'];source=read(config['paths']['deepsolo_results'])
    old={phase:{r['sample_id']:r for r in values['per_image']} for phase,values in source['splits'].items()}
    threshold=read(config['paths']['deepsolo_plan'])['config']['score_threshold']
    receipts={}
    for phase,records in index.items():
        for record in records:
            sid=record['sample_id'];path=root/'deepsolo'/phase/(sample_key(sid)+'.json')
            if phase in ('evaluation','test'):
                original=old['test' if phase=='test' else 'train'][sid];raw=Path(original['raw_response_path'])
                if sha(raw)!=original['raw_response_sha256']:raise ValueError('Changed qualified DeepSolo response')
                shard=read(raw)
                if shard['contract']['image_sha256']!=record['image_sha256']:raise ValueError('DeepSolo split/RGB identity mismatch')
                response=shard['response'];receipts[str(raw)]=sha(raw)
            else:
                raw=root/'deep_training_raw'/phase/(sample_key(sid)+'.json');data=read(raw)
                if data['image_sha256']!=record['image_sha256']:raise ValueError('Wrong exported DeepSolo RGB')
                response={'detections':data['detections']};receipts[str(raw)]=sha(raw)
            predictions=deepsolo_predictions(response,record['image_wh'],record['orientation'],np.ones((128,128),bool),threshold)
            # Same care-only/no-ignore protocol for every expert; the historical
            # DeepSolo result using polygon ignores remains a separate number.
            for p in predictions:
                p['ignored']=False;p['confidence']=p['score'];p['presence']=1.
            metric=word_measure(predictions,record['ground_truth'],config['evaluation']['iou_threshold'])
            write(path,{'predictions':predictions,'metric':metric,'source_path':str(raw),'source_sha256':sha(raw),
                        'image_sha256':record['image_sha256']})
    write(root/'deep_conversion_audit.json',{'status':'PASS','source_sha256':receipts,
          'protocol':'care GT, all predictions; exact casefold+original bbox IoU0.5; no GT ignore masking'})


def train(root,config,arm,warmup=False,smoke=False):
    protocol=read(root/'protocol.json');records=protocol['index']['fit'];device=config['runtime']['device']
    options={} if warmup else config['arms'][arm];training=config['training']
    directory=root/('warmup' if warmup else 'branches/'+arm);directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return
    # Identical random initialization for every arm. Shared warmup is frozen on disk.
    torch.manual_seed(config['seed']);torch.cuda.manual_seed_all(config['seed'])
    model=make_model(config,None if warmup else arm);load_prior(model,config)
    if not warmup:load_delta(model,root/'warmup'/'last.pt',config)
    model.to(device).train();parameters=trainable(model)
    optimizer=torch.optim.AdamW(parameters,lr=training['learning_rate'],weight_decay=training['weight_decay'])
    recognizer=reader(config) if options.get('ocr') else None
    epochs=1 if smoke else training['warmup_epochs' if warmup else 'branch_epochs']
    completed=0;history=[]
    # Resume state is local trusted data, with a pinned prior/config contract.
    resume=directory/'resume.pt'
    if resume.exists():
        data=torch.load(resume,map_location='cpu',weights_only=False)
        if data['config_digest']!=digest(config) or data['code_sha256']!=source_hashes():raise ValueError('Training resume contract changed')
        load_delta(model,directory/'last.pt',config);optimizer.load_state_dict(data['optimizer'])
        completed=data['epoch'];history=data['history']
    generator=torch.Generator();tick=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    for epoch in range(completed+1,epochs+1):
        generator.manual_seed(config['seed']+epoch)
        order=torch.randperm(len(records),generator=generator).tolist();optimizer.zero_grad(set_to_none=True)
        totals={};conflicts=[];epoch_tick=time.perf_counter()
        for step,position in enumerate(order,1):
            record=records[position];pyramid,target=cache_load(root,record,device)
            if len(target['boxes'])>model.slots:
                raise ValueError('More care words than independent slots: raise slots before training, never drop GT')
            dn=make_denoising(target,groups=training['denoising_groups'],noise=training['denoising_noise'],
                             background_queries=training['denoising_background_queries'],generator=generator) if options.get('denoising') else None
            dense=training['dense_groups'] if options.get('dense') else 1
            dense_offsets=None
            if dense>1:
                dense_generator=torch.Generator().manual_seed(config['seed']+epoch*1000033+position*1013)
                dense_offsets=((torch.rand((dense-1,model.slots,4),generator=dense_generator)*2-1)*training['dense_jitter']).to(device)
            output=model.decode_pyramid(pyramid,denoising=dn,dense_groups=dense,prior_boxes=target['prior_boxes'],dense_offsets=dense_offsets)
            masks=F.interpolate(output['masks'][None],target['masks'].shape[-2:],mode='nearest')[0]
            weights=None
            if options.get('specialist') or options.get('mcl'):
                d=read(root/'deepsolo'/record['phase']/(sample_key(record['sample_id'])+'.json'))
                correct=set(d['metric']['matched_gt_indices'])
                output['deep_predictions']=d['predictions']
                if options.get('specialist'):
                    weights=target['boxes'].new_tensor([1 if j in correct else training['specialist_failure_weight'] for j in range(len(target['boxes']))])
                    output['specialist_weights']=weights
            loss,pair,stats=instance_loss(output['boxes'],output['presence'],masks,target,
                       no_object_weight=training['no_object_weight'],mask_weight=training['mask_weight'],word_weights=weights,mal=options.get('dense',False))
            if dn is not None:
                loss=loss+training['denoising_weight']*denoising_loss(output,target,dn,
                            no_object_weight=training['no_object_weight'],mask_weight=training['mask_weight'])
            if dense>1:
                for group in range(1,dense):
                    section=slice(group*model.slots,(group+1)*model.slots)
                    dense_masks=F.interpolate(output['all_masks'][None,section],masks.shape[-2:],mode='nearest')[0]
                    extra=instance_loss(output['all_boxes'][section],output['all_presence'][section],dense_masks,target,
                        no_object_weight=training['no_object_weight'],mask_weight=training['mask_weight'],mal=True)[0]
                    loss=loss+training['dense_weight']*extra/(dense-1)
            auxiliary=loss.new_zeros(())
            if options.get('ccd') or options.get('glyph'):
                extra,character_stats=character_objective(model,output,record,pair,root,options,training)
                auxiliary=auxiliary+extra;stats.update(character_stats)
            if recognizer is not None:
                crop_generator=torch.Generator().manual_seed(config['seed']+epoch*1000003+position*1009)
                extra,ocr_stats=ocr_objective(output,record,pair,recognizer,options,training,crop_generator)
                auxiliary=auxiliary+extra;stats.update(ocr_stats)
            if not torch.isfinite(loss+auxiliary):raise FloatingPointError('Nonfinite training loss: '+record['sample_id'])
            accumulation=training['gradient_accumulation'];group_size=min(accumulation,len(order)-((step-1)//accumulation)*accumulation)
            if options.get('pcgrad') and auxiliary.requires_grad:
                previous=[None if p.grad is None else p.grad.clone() for p in parameters]
                evidence=pcgrad_two(loss/group_size,auxiliary/group_size,parameters);conflicts.append(evidence)
                for p,old in zip(parameters,previous):
                    if old is not None:p.grad=old if p.grad is None else p.grad+old
            else:((loss+auxiliary)/group_size).backward()
            if step%accumulation==0 or step==len(order):
                norm=torch.nn.utils.clip_grad_norm_(parameters,training['gradient_clip_norm'],error_if_nonfinite=True)
                optimizer.step();optimizer.zero_grad(set_to_none=True)
            stats['loss']=float(loss.detach());stats['auxiliary']=float(auxiliary.detach())
            for key,value in stats.items():totals[key]=totals.get(key,0)+float(value)
            if step%200==0:print('TRAIN',arm if not warmup else 'warmup',epoch,step,len(order),'loss',round(totals['loss']/step,4),flush=True)
        row={'epoch':epoch,'images':len(order),'seconds':time.perf_counter()-epoch_tick,
             'mean':{key:value/len(order) for key,value in totals.items()},
             'gradient_conflicts':sum(bool(x['conflict']) for x in conflicts),
             'gradient_comparisons':len(conflicts),'gradient_cosines':conflicts}
        history.append(row)
        save_delta(directory/'last.pt',model,config,epoch=epoch,arm=arm,code_sha256=source_hashes())
        # Epoch-local ordering and choices have explicit generators; no dropout or
        # stochastic model layer is enabled. Resuming an epoch repeats its full data.
        torch.save({'config_digest':digest(config),'code_sha256':source_hashes(),'epoch':epoch,
                    'optimizer':optimizer.state_dict(),'history':history},resume)
        write(directory/'history.json',history)
        print('EPOCH',arm,epoch,json.dumps({k:v for k,v in row.items() if k!='gradient_cosines'}),flush=True)
    write(directory/'complete.json',{'status':'SUCCESS','arm':arm,'epochs':epochs,'samples_per_epoch':len(records),
          'checkpoint_sha256':sha(directory/'last.pt'),'code_sha256':source_hashes(),'seconds':time.perf_counter()-tick,
          'claim_label':'approximate reproduction','result_class':'smoke-test result' if smoke else protocol['result_class'],
          'models':['frozen qualified pixel U-Net prior','trainable four-scale word-slot decoder']+(['frozen full RGB PARSeq'] if recognizer else []),
          'precision':'float32','microbatch_size':1,'gradient_accumulation':training['gradient_accumulation'],
          'cached_input_shapes':[[1,512,32,32],[1,256,64,64],[1,128,128,128],[1,64,256,256]],**memory()})
    del optimizer,model,recognizer;gc.collect();torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate']);parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--stage',required=True,choices=['prepare','teacher','deep','warmup','train'])
    parser.add_argument('--arm');parser.add_argument('--smoke',action='store_true');parser.add_argument('--limit-images',type=int)
    args=parser.parse_args();config=read(args.config);root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if REPO/'configs/approximate' not in args.config.resolve().parents or config['reproduction_mode']!=args.mode or config['experiment_kind']!='word_slot_methods_v1':
        raise ValueError('Use explicit isolated APPROXIMATE word-slot configuration')
    for key,expected in config['input_sha256'].items():
        if sha(config['paths'][key])!=expected:raise ValueError('Pinned input changed: '+key)
    try:
        hardware=setup(config);write(root/('runtime_'+args.stage+'.json'),hardware)
        if args.stage=='prepare':prepare(root,config,hardware,args.limit_images)
        else:
            if read(root/'protocol.json')['config_digest']!=digest(config):raise ValueError('Run configuration changed')
            if args.stage=='teacher':teacher_cache(root,config)
            elif args.stage=='deep':prepare_deep(root,config)
            elif args.stage=='warmup':train(root,config,'warmup',warmup=True,smoke=args.smoke)
            else:
                arms=[args.arm] if args.arm else list(config['arms'])
                if any(arm not in config['arms'] for arm in arms):raise ValueError('Unknown branch')
                for arm in arms:train(root,config,arm,smoke=args.smoke)
    except Exception as error:
        write(root/('failure_'+args.stage+'_'+str(os.getpid())+'.json'),{'status':'FAILED','type':type(error).__name__,
              'message':str(error),'traceback':traceback.format_exc(),'oom':isinstance(error,torch.cuda.OutOfMemoryError)})
        raise


if __name__=='__main__':main()
