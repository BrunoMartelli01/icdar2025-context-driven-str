"""Five controlled APPROXIMATE pilots; explicit artifacts, fit-only updates and word F1."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from localize.teacher_student_tests import (CandidateScorer, ema_update, confidence_consistency,
    token_distillation, occlude_words, grouped_set_loss, compatible_selection, teacher_matches)
from localize.occlusion_four_tests import (decode_instance_labels, instance_pool, features, row,
    ocr_features, iou, bind_baseline)
from models.adaspot_gap import AdaSpotGapAGUNet
from scripts.run_mask_crop_budget import read, write, setup, utc, CropOCR, measure, aggregate
from scripts.run_occlusion_four_tests import (load_rgb, path_for, labels_for, score_record,
    selected, preselect, changes, digest)
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from segmentation.data import load_segmentation_manifest, SegmentationDataset, file_sha256
from segmentation.totaltext_word_kernel import WordKernelDataset
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from segmentation.word_multitask import masked_bce_dice
from segmentation.training import seed_everything

REPO = Path(__file__).resolve().parents[1]


def announce(out, stage, **values):
    write(out/'status.json', dict(status='SUCCESS' if stage=='REPORT_COMPLETE' else 'RUNNING', phase=stage, utc=utc(), pid=os.getpid(), **values))
    print(stage, json.dumps(values), flush=True)


def peaks(models, batch, shape):
    torch.cuda.synchronize()
    return dict(models=models, precision='float32', batch_size=batch, input_shape=shape,
                gpu=torch.cuda.get_device_name(0), allocated=torch.cuda.max_memory_allocated(),
                reserved=torch.cuda.max_memory_reserved(), oom=False)


def release():
    gc.collect(); torch.cuda.empty_cache()


def context(args):
    cp = args.config.resolve()
    c = read(cp)
    if args.mode != 'approximate' or REPO/'configs/approximate' not in cp.parents or c['reproduction_mode'] != args.mode or c['experiment_kind'] != 'teacher_student_tests_v1':
        raise ValueError('Explicit isolated APPROXIMATE configuration required')
    if c['metric_iou_threshold'] != .5:
        raise ValueError('Qualified shared metric requires explicit IoU0.5')
    for key, value in c['input_sha256'].items():
        if file_sha256(Path(c['paths'][key])) != value:
            raise ValueError('Input drift: '+key)
    source = Path(c['source_run'])
    if file_sha256(source/'results.json') != c['source_results_sha256'] or read(source/'results.json')['status'] != 'SUCCESS':
        raise ValueError('Completed qualified source run required')
    if read(source/'independent_audit.json')['status'] != 'PASS':
        raise ValueError('Source matcher audit failed')
    protocol = read(source/'protocol.json')
    # Reusing neural outputs requires matching the source implementation, not merely the weights.
    for name in ('recognition/parseq.py', 'models/adaspot_gap.py', 'models/unet_word_research.py',
                 'models/totaltext_word_kernel.py', 'models/unet_adv.py', 'localize/occlusion_four_tests.py'):
        if file_sha256(REPO/name) != protocol['source_code_sha256'][name]:
            raise ValueError('Frozen implementation changed: '+name)
    hardware = setup(c)
    index = protocol['index']
    if args.smoke:
        index = {phase: entries[:c['smoke'][phase]] for phase, entries in index.items()}
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    code = {str(p.relative_to(REPO)): file_sha256(p) for folder in ('scripts','localize','segmentation','models','recognition') for p in sorted((REPO/folder).rglob('*.py'))}
    identity = dict(config_sha256=file_sha256(cp), source_code_sha256=code, index=index, smoke=args.smoke)
    if (out/'protocol.json').exists():
        if read(out/'protocol.json')['identity'] != identity:
            raise ValueError('Resume contract changed')
    else:
        write(out/'protocol.json',dict(identity=identity, resolved_config=c, hardware=hardware,
              result_class='smoke-test result' if args.smoke else 'approximate reproduction; short-budget exploratory pilot',
              metric=protocol['metric'], historical_test_exploration=True, seed_count=1))
        write(out/'resolved_config.json', c)
    contract = read(c['paths']['source_contract'])
    return c, out, index, contract


def source_records(c, index):
    records = {phase: {r['sample_id']: read(path_for(Path(c['source_run']),phase,r['sample_id'])) for r in entries} for phase, entries in index.items()}
    ocr = {}
    for path in sorted(Path(c['source_run']).glob('ocr_*.jsonl')):
        for line in path.open():
            r = json.loads(line); key = r['crop_array_sha256']
            if key in ocr and ocr[key] != r: raise ValueError('OCR collision')
            ocr[key] = r
    return records, ocr


def frozen_selector(c, kind):
    p = torch.load(Path(c['source_run'])/('selector_'+kind+'.pt'), map_location='cpu', weights_only=True)
    model = torch.nn.Sequential(torch.nn.Linear(p['features'],32),torch.nn.ReLU(),torch.nn.Linear(32,1))
    model.load_state_dict(p['state_dict'],strict=True)
    return model.eval(),p['mean'],p['scale']


def report(records, predictions, readings):
    gt, hashes = labels_for(records)
    values = [{'sample_id': sid, **measure(predictions[sid],gt[sid],readings)} for sid in records]
    return dict(aggregate=aggregate(values),per_image=values,annotation_hashes=hashes)


def baseline_predictions(c, records, ocr, budget):
    pre, post = frozen_selector(c,'pre'), frozen_selector(c,'post')
    policy = read(Path(c['source_run'])/'selection_frozen_before_evaluation.json')['selector']['mk']
    output = {}; available = {}
    cache = type('Readings',(),dict(rows=ocr))()
    for sid,r in records.items():
        before = score_record(r,'mk',cache,pre,False)
        after = score_record(r,'mk',cache,post,True)
        ids = preselect(r,'mk',before,budget); available[sid] = ids
        output[sid] = selected(r,'mk',after,policy,ids)
    return output, available


def selector_stage(c,out,index,contract,smoke):
    directory=out/'selector'; directory.mkdir(exist_ok=False)
    records,ocr=source_records(c,index); fitted=[]; groups=[]; target=[]; xs=[]; offset=0
    gt,_=labels_for(records['fit'])
    for sid,r in records['fit'].items():
        pool=r['arms']['mk']['pool']
        for p in pool:
            xs.append(r['features']['mk'][p['id']]+ocr_features(ocr[p['crop_array_sha256']]))
            matches=[j for j,g in enumerate(gt[sid]) if iou(p['bbox'],g['bbox'])>=.5 and ocr[p['crop_array_sha256']]['raw_text'].casefold()==g['text'].casefold()]
            target.append(float(bool(matches))); groups.append(offset+matches[0] if matches else -1)
        offset+=len(gt[sid])
    x=torch.tensor(xs);y=torch.tensor(target);ids=torch.tensor(groups);mean=x.mean(0);scale=x.std(0).clamp_min(c['selector']['minimum_feature_scale']);x=(x-mean)/scale
    for kind in ('pointwise','set'):
        seed_everything(c['seed'],deterministic_algorithms=True)
        model=CandidateScorer(x.shape[1],c['selector']['hidden'])
        opt=torch.optim.AdamW(model.parameters(),lr=c['selector']['learning_rate'],weight_decay=c['selector']['weight_decay']);history=[]
        for epoch in range(c['selector']['epochs']):
            opt.zero_grad(set_to_none=True);logits=model(x)
            loss=F.binary_cross_entropy_with_logits(logits,y) if kind=='pointwise' else grouped_set_loss(logits,y,ids,offset,c['selector']['duplicate_weight'],c['selector']['coverage_weight'])
            if not loss.isfinite():raise FloatingPointError('Nonfinite set loss')
            loss.backward();opt.step();history.append(float(loss.detach()))
        torch.save(dict(state_dict=model.state_dict(),mean=mean,scale=scale),directory/(kind+'.pt'))
        write(directory/(kind+'_training.json'),dict(loss=history,candidates=len(y),positives=int(y.sum()),fit_ids=list(records['fit'])))
        fitted.append((kind,model.eval()))
    all_predictions={};policies={}
    for phase in ('calibration','evaluation','test'):
        base,available=baseline_predictions(c,records[phase],ocr,c['budget'])
        predictions={'baseline_budget20':base}
        for kind,model in fitted:
            scores={}
            for sid,r in records[phase].items():
                pool=[p for p in r['arms']['mk']['pool'] if p['id'] in available[sid]]
                if pool:
                    values=torch.tensor([r['features']['mk'][p['id']]+ocr_features(ocr[p['crop_array_sha256']]) for p in pool])
                    with torch.no_grad():s=model((values-mean)/scale).sigmoid().tolist()
                    scores[sid]=dict(zip([p['id'] for p in pool],s))
                else:scores[sid]={}
            if phase=='calibration':
                grid=[]
                for threshold in c['selector']['thresholds']:
                    pp={sid:compatible_selection([p for p in r['arms']['mk']['pool'] if p['id'] in available[sid]],scores[sid],threshold,c['selector']['overlap']) for sid,r in records[phase].items()}
                    grid.append(dict(threshold=threshold,aggregate=report(records[phase],pp,ocr)['aggregate']))
                policies[kind]=max(grid,key=lambda g:(g['aggregate']['f1'],g['threshold']))['threshold']
                write(directory/(kind+'_calibration.json'),grid)
            predictions[kind]={sid:compatible_selection([p for p in r['arms']['mk']['pool'] if p['id'] in available[sid]],scores[sid],policies[kind],c['selector']['overlap']) for sid,r in records[phase].items()}
        if phase=='calibration':write(directory/'selection_frozen.json',policies)
        results={name:report(records[phase],pp,ocr) for name,pp in predictions.items()}
        if not smoke and phase in ('evaluation','test'):
            expected=read(Path(c['source_run'])/(phase+'_results.json'))['budget_mk_learned_20']['aggregate']
            if results['baseline_budget20']['aggregate']!=expected:
                raise ValueError('Frozen baseline failed exact count/rate replay')
        write(directory/(phase+'_predictions.json'),predictions);write(directory/(phase+'_results.json'),results)
    write(directory/'complete.json',dict(status='SUCCESS',policies=policies))


def source_model(c,contract):
    model=AdaSpotGapAGUNet('control',skip_floor=contract['resolved_config']['skip_floor'],initial_gap_bias=-2.)
    p=torch.load(c['paths']['source_checkpoint'],map_location='cpu',weights_only=False)
    if p['run_contract']!=contract:raise ValueError('Checkpoint contract changed')
    model.load_state_dict(p['model_state_dict'],strict=True)
    return model.cuda().eval()


def export_model(model,c,contract,entries,directory):
    transform=_transform(contract['resolved_preprocessing'],'letterbox',train=False)
    pp=contract['resolved_config']['postprocessing'];hyp=read(Path(c['source_run'])/'resolved_config.json')['hypotheses']
    output={}
    with torch.no_grad():
        for entry in entries:
            rgb=load_rgb(entry);prob=model(_tensor(rgb,transform).cuda())[0].sigmoid().cpu().numpy()
            canvas=np.asarray(letterbox_canvas_mask((rgb.shape[1],rgb.shape[0]),native_size=prob.shape[1]),bool)
            labels=decode_instance_labels(prob,None,canvas,pp,None,'cut')
            pool=instance_pool(prob[0],labels,rgb,hyp)
            record={**entry,'image_hw':list(rgb.shape[:2]),'arms':{'mk':pool},'features':{'mk':{p['id']:features(p,rgb,prob[0],text_threshold=pp['text_threshold']) for p in pool['pool']}}}
            sid=entry['sample_id'];output[sid]=record
            target=directory/(sid.replace(':','__')+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,probabilities=prob)
    return output


def recognized_export(model,c,contract,entries,directory,ocr):
    records=export_model(model,c,contract,entries,directory/'maps')
    for r in records.values():
        ocr.image(load_rgb(r),r['arms']['mk']['pool']);bind_baseline(r['arms']['mk'])
    return records


def pseudo_stage(c,out,index,contract,smoke):
    directory=out/'pseudo';directory.mkdir(exist_ok=False)
    pc=c['pseudo'];manifest=Path(pc['manifest'])
    if file_sha256(manifest)!=pc['manifest_sha256']:raise ValueError('Pseudo manifest changed')
    m=load_segmentation_manifest(manifest,require_checksums=True)
    if m.split_role!='train' or m.dataset!='icdar2015':raise ValueError('Additional TRAIN only')
    excluded={e['image_sha256'] for entries in index.values() for e in entries}
    # Always exclude full source populations, including in a smoke run.
    source_index=read(Path(c['source_run'])/'protocol.json')['index']
    excluded|={e['image_sha256'] for entries in source_index.values() for e in entries}
    chosen=sorted(m.samples,key=lambda s:digest([c['seed'],s.sample_id]))[:c['smoke']['pseudo'] if smoke else pc['maximum_images']]
    entries=[]
    for s in chosen:
        if s.image_sha256 in excluded:raise ValueError('Additional TRAIN overlaps evaluation/source')
        entries.append(dict(sample_id=s.sample_id,image_path=str(Path(pc['data_root'])/s.image_path),image_sha256=s.image_sha256))
    model=source_model(c,contract);transform=_transform(contract['resolved_preprocessing'],'letterbox',train=False)
    pp=contract['resolved_config']['postprocessing'];hyp=read(Path(c['source_run'])/'resolved_config.json')['hypotheses'];records=[]
    torch.cuda.reset_peak_memory_stats()
    for number,entry in enumerate(entries,1):
        rgb=load_rgb(entry);tensor=_tensor(rgb,transform).cuda()
        altered=(tensor*(1+pc['contrast'])).clamp(-5,5)
        with torch.no_grad():
            clean=model(tensor)[0].sigmoid().cpu().numpy()
            alternate=model(altered.flip(-1))[0].sigmoid().flip(-1).cpu().numpy()
        canvas=np.asarray(letterbox_canvas_mask((rgb.shape[1],rgb.shape[0]),native_size=clean.shape[1]),bool)
        arms=[]
        for prob in (clean,alternate):
            labels=decode_instance_labels(prob,None,canvas,pp,None,'cut')
            arms.append(instance_pool(prob[0],labels,rgb,{**hyp,'maximum_extra':0})['baseline'])
        path=directory/'maps'/(entry['sample_id'].replace(':','__')+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,clean=clean,alternate=alternate,canvas=canvas)
        records.append({**entry,'image_hw':list(rgb.shape[:2]),'arms':arms,'maps_path':str(path),'maps_sha256':file_sha256(path)})
        if number%32==0 or number==len(entries):announce(out,'PSEUDO_MAPS',images=number,total=len(entries))
    write(directory/'segmentation_memory.json',peaks('frozen M/K two-view pseudo teacher',1,[1,3,512,512]))
    del model;release();ocr=CropOCR(c,directory/'ocr.jsonl');torch.cuda.reset_peak_memory_stats()
    for r in records:
        rgb=load_rgb(r)
        for arm in r['arms']:ocr.image(rgb,arm)
        r['accepted']=teacher_matches(*r['arms'],ocr.rows,pc['iou'],pc['minimum_confidence'])
        with np.load(r['maps_path']) as saved:clean=saved['clean'];alternate=saved['alternate'];canvas=saved['canvas']
        support=np.zeros(clean.shape[1:],bool)
        for p in r['accepted']:
            x0,y0,x1,y1=p['bbox_native'];support[y0:y1,x0:x1]=True
        confidence=pc['map_confidence']
        valid=np.broadcast_to(support,clean.shape).copy() & ((clean>=confidence)|(clean<=1-confidence)) & (np.abs(clean-alternate)<=pc['maximum_probability_difference']) & canvas[None]
        target=directory/'targets'/(r['sample_id'].replace(':','__')+'.npz');target.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target,probabilities=clean,valid=valid)
        r['target_path']=str(target);r['target_sha256']=file_sha256(target)
    write(directory/'ocr_memory.json',peaks('qualified PARSeq pseudo-label text agreement',c['runtime']['ocr_batch_size'],[32,3,32,128]));ocr.release()
    count=sum(len(r['accepted']) for r in records)
    write(directory/'manifest.json',records)
    write(directory/'complete.json',dict(status='SUCCESS',images=len(records),accepted_boxes=count,accepted_images=sum(bool(r['accepted']) for r in records),annotations_read=False,minimum_full_boxes=pc['minimum_accepted_boxes']))
    if not smoke and count<pc['minimum_accepted_boxes']:raise ValueError('Too few stable pseudo boxes; inspect manifest before training')


def supervised_loss(logits,batch,loss):
    terms=[]
    for channel,key in enumerate(('mask','kernel')):
        terms.append(masked_bce_dice(logits[:,channel:channel+1],batch[key],batch['validity'],bce_weight=loss['bce_weight'],dice_weight=loss['dice_weight'],smooth=loss['smooth'],pixel_weight=1+(loss['separation_weight']-1)*batch['separation'] if channel==0 else None))
    return sum(terms)


def segmentation_stage(c,out,index,contract,smoke):
    directory=out/'segmentation';directory.mkdir(exist_ok=False)
    records,old_ocr=source_records(c,index);source=Path(c['source_run']);sc=c['segmentation'];base=contract['resolved_config']
    manifest=load_segmentation_manifest(Path(c['paths']['train_manifest']),require_checksums=True)
    transform=_transform(contract['resolved_preprocessing'],'letterbox',train=False)
    dataset=WordKernelDataset(SegmentationDataset(manifest,Path(c['paths']['word_root']),transform=transform),distance_fraction=base['targets']['distance_fraction'])
    by={s.sample_id:i for i,s in enumerate(manifest.samples)};fit=[by[e['sample_id']] for e in index['fit']]
    # Cache deterministic paired targets once; no hidden augmentation RNG in workers.
    batches=[dataset[i] for i in fit]
    pseudo=[r for r in read(out/'pseudo/manifest.json') if r['accepted']]
    ocr=CropOCR(c,directory/'ocr.jsonl');ocr.rows.update(old_ocr)
    all_results={};all_predictions={};calibrations={}
    baseline,_=baseline_predictions(c,records['calibration'],old_ocr,c['budget'])
    initial_f1=report(records['calibration'],baseline,old_ocr)['aggregate']['f1']
    for arm in ('supervised','consistency','pseudo'):
        seed_everything(c['seed'],deterministic_algorithms=True)
        student=source_model(c,contract);teacher=copy.deepcopy(student).requires_grad_(False).eval()
        optimizer=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=sc['learning_rate'],weight_decay=sc['weight_decay'])
        generator=torch.Generator().manual_seed(c['seed']);history=[];best={arm:dict(epoch=0,f1=initial_f1)}
        if arm=='supervised':best['ema_only']=dict(epoch=0,f1=initial_f1)
        branch=directory/arm;branch.mkdir();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for epoch in range(1,(1 if smoke else sc['epochs'])+1):
            student.train();order=torch.randperm(len(batches),generator=generator).tolist();losses=[];consistencies=[]
            for start in range(0,len(order),sc['batch_size']):
                chosen=[batches[i] for i in order[start:start+sc['batch_size']]]
                batch={k:torch.stack([b[k] for b in chosen]).cuda() for k in ('image','mask','kernel','validity','separation')}
                flips=torch.rand((len(chosen),),generator=generator)<.5
                for k in batch:batch[k][flips]=batch[k][flips].flip(-1)
                weak=batch['image'];brightness=float(torch.rand((),generator=generator)*2-1)*sc['brightness'];contrast=1+float(torch.rand((),generator=generator)*2-1)*sc['contrast']
                strong=weak*contrast+brightness
                optimizer.zero_grad(set_to_none=True);logits=student(strong);loss=supervised_loss(logits,batch,base['loss']);extra=loss*0
                if arm=='consistency':
                    with torch.no_grad():targets=teacher(weak).sigmoid()
                    extra=confidence_consistency(logits,targets,batch['validity'],sc['confidence']);loss=loss+sc['consistency_weight']*extra
                if arm=='pseudo' and pseudo:
                    p=pseudo[(start//sc['batch_size']+(epoch-1)*((len(order)+sc['batch_size']-1)//sc['batch_size']))%len(pseudo)]
                    if file_sha256(Path(p['target_path']))!=p['target_sha256']:raise ValueError('Pseudo target changed')
                    rgb=load_rgb(p)
                    with np.load(p['target_path']) as z:targets=torch.tensor(z['probabilities'][None],device='cuda');valid=torch.tensor(z['valid'][None],device='cuda')
                    pred=student(_tensor(rgb,transform).cuda()*contrast+brightness)
                    extra=(pred.sigmoid()-targets).square()[valid].mean() if valid.any() else pred.sum()*0
                    loss=loss+sc['pseudo_weight']*extra
                if not loss.isfinite():raise FloatingPointError('Nonfinite segmentation training')
                loss.backward();optimizer.step();ema_update(teacher,student,sc['ema_decay']);losses.append(float(loss.detach()));consistencies.append(float(extra.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),auxiliary=float(np.mean(consistencies)),updates=len(losses),seconds=time.perf_counter()-started))
            write(branch/'history.json',history)
            # Free optimizer intermediates before sequential neural evaluation.
            optimizer.zero_grad(set_to_none=True);student.eval()
            for name,model in [(arm,student)]+([('ema_only',teacher)] if arm=='supervised' else []):
                exported=recognized_export(model,c,contract,index['calibration'],branch/f'epoch{epoch}_{name}',ocr)
                pred,_=baseline_predictions(c,exported,ocr.rows,c['budget']);metric=report(exported,pred,ocr.rows)
                calibrations[name+':'+str(epoch)]=metric['aggregate']
                if metric['aggregate']['f1']>best[name]['f1']:
                    best[name]=dict(epoch=epoch,f1=metric['aggregate']['f1'])
                    torch.save(dict(state_dict=model.state_dict(),selection=best[name]),directory/(name+'_best.pt'))
                torch.save(dict(state_dict=model.state_dict(),epoch=epoch),directory/(name+'_last.pt'))
                announce(out,'SEGMENTATION',arm=name,epoch=epoch,calibration_f1=metric['aggregate']['f1'],best=best[name])
            ocr.release()
        write(branch/'memory.json',peaks('M/K student + EMA teacher + AdamW; PARSeq co-resident during calibration',sc['batch_size'],[sc['batch_size'],3,512,512]))
        write(branch/'selection_frozen.json',best)
        del optimizer,teacher,student;release()
        for name,choice in best.items():
            for result_name,checkpoint in ((name, directory/(name+'_best.pt') if choice['epoch'] else None),
                                           (name+'_final',directory/(name+'_last.pt'))):
                model=source_model(c,contract)
                if checkpoint is not None:
                    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state_dict'],strict=True)
                for phase in ('evaluation','test'):
                    # A selected epoch0 is explicitly reported alongside final-epoch performance.
                    if checkpoint is None:exported=records[phase]
                    else:exported=recognized_export(model,c,contract,index[phase],directory/result_name/phase,ocr)
                    pred,_=baseline_predictions(c,exported,ocr.rows,c['budget'])
                    key=phase+':'+result_name;all_predictions[key]=pred;all_results[key]=report(exported,pred,ocr.rows)
                    write(directory/(key.replace(':','_')+'_predictions.json'),pred)
                    write(directory/(key.replace(':','_')+'_results.json'),all_results[key]);ocr.release()
                del model;release()
        write(directory/'calibration.json',calibrations)
    write(directory/'complete.json',dict(status='SUCCESS',calibration=calibrations,results={k:v['aggregate'] for k,v in all_results.items()}))


def teacher_forced(model,images,tokens):
    core=model.model;memory=core.encode(images);length=tokens.shape[1]-1
    causal=torch.triu(torch.ones((length,length),dtype=torch.bool,device=images.device),1)
    features=core.decode(tokens[:,:-1],memory,tgt_mask=causal,tgt_padding_mask=tokens[:,:-1]==model.tokenizer.pad_id,tgt_query_mask=causal)
    return core.head(features)


def decode_crops(recognizer,records,predictions,output,batch_size):
    readings={};rows=[]
    for sid,r in records.items():
        rgb=load_rgb(r);pool=predictions[sid]
        crops=[np.ascontiguousarray(rgb[p['bbox'][1]:p['bbox'][3],p['bbox'][0]:p['bbox'][2]]) for p in pool]
        for start in range(0,len(crops),batch_size):
            for p,result in zip(pool[start:start+batch_size],recognizer.recognize_candidate_batch(crops[start:start+batch_size])):
                key=p['crop_array_sha256'];readings[key]=result.to_dict()
    write(output,readings)
    return readings


def recognition_stage(c,out,index,contract,smoke):
    directory=out/'recognition';directory.mkdir(exist_ok=False)
    records,old_ocr=source_records(c,index);rc=c['recognition']
    from recognition.parseq import ParseqApproxRecognizer,ParseqApproxConfig,PARSEQ_ARCHIVE_SOURCE
    def load():return ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(c['paths']['parseq_source']),checkpoint=Path(c['paths']['parseq_checkpoint']),device='cuda:0',runtime_source_type=PARSEQ_ARCHIVE_SOURCE,runtime_source_archive=Path(c['paths']['parseq_archive'])))
    qualified=load();charset=set(qualified.model.tokenizer._stoi);examples=[];omitted=[]
    gt,hashes=labels_for(records['fit'])
    for sid,r in records['fit'].items():
        rgb=load_rgb(r)
        for j,g in enumerate(gt[sid]):
            text=g['text']
            if not text or len(text)>qualified.model.model.max_label_length or any(ch not in charset for ch in text):
                omitted.append(dict(sample_id=sid,word=j,text=text,reason='outside qualified charset/maximum length'));continue
            x0,y0,x1,y1=g['bbox'];crop=rgb[y0:y1,x0:x1]
            examples.append((qualified.transform(Image.fromarray(crop)),text))
    write(directory/'training_population.json',dict(fit_ids=list(records['fit']),words=len(examples),omitted=omitted,annotation_hashes=hashes))
    initial_state=copy.deepcopy(qualified.model.state_dict());qualified.release();del qualified;release()
    base_predictions={phase:baseline_predictions(c,records[phase],old_ocr,c['budget'])[0] for phase in ('calibration','evaluation','test')}
    cal_baseline=report(records['calibration'],base_predictions['calibration'],old_ocr)['aggregate']['f1'];calibration={};results={}
    for arm in ('occlusion_ce','occlusion_distill'):
        seed_everything(c['seed'],deterministic_algorithms=True);recognizer=load();student=recognizer.model
        student.load_state_dict(initial_state,strict=True)
        for name,p in student.named_parameters():p.requires_grad_(not name.startswith('model.encoder.'))
        teacher=copy.deepcopy(student).requires_grad_(False).eval()
        opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=rc['learning_rate'],weight_decay=rc['weight_decay'])
        generator=torch.Generator().manual_seed(c['seed']);best=dict(epoch=0,f1=cal_baseline);history=[];branch=directory/arm;branch.mkdir();torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,(1 if smoke else rc['epochs'])+1):
            student.train();student.model.encoder.eval();order=torch.randperm(len(examples),generator=generator).tolist();losses=[];kls=[]
            for start in range(0,len(order),rc['batch_size']):
                batch=[examples[i] for i in order[start:start+rc['batch_size']]];clean=torch.stack([e[0] for e in batch]).cuda()
                corrupted=occlude_words(clean,generator,rc['occlusion_probability'],rc['occlusion_width_fraction'])
                tokens=student.tokenizer.encode([e[1] for e in batch],device=clean.device);target=tokens[:,1:]
                opt.zero_grad(set_to_none=True);logits=teacher_forced(student,corrupted,tokens)
                loss=F.cross_entropy(logits.flatten(0,1),target.flatten(),ignore_index=student.tokenizer.pad_id);kl=loss*0
                if arm=='occlusion_distill':
                    with torch.no_grad():soft=teacher_forced(teacher,clean,tokens)
                    kl=token_distillation(logits,soft,target,student.tokenizer.pad_id,rc['temperature']);loss=loss+rc['distillation_weight']*kl
                if not loss.isfinite():raise FloatingPointError('Nonfinite recognition training')
                loss.backward();opt.step();losses.append(float(loss.detach()));kls.append(float(kl.detach()))
            student.eval();opt.zero_grad(set_to_none=True)
            readings=decode_crops(recognizer,records['calibration'],base_predictions['calibration'],branch/f'epoch{epoch}_calibration_ocr.json',c['runtime']['ocr_batch_size'])
            metric=report(records['calibration'],base_predictions['calibration'],readings);calibration[arm+':'+str(epoch)]=metric['aggregate']
            if metric['aggregate']['f1']>best['f1']:
                best=dict(epoch=epoch,f1=metric['aggregate']['f1']);torch.save(student.state_dict(),branch/'best.pt')
            torch.save(student.state_dict(),branch/'last.pt')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),distillation=float(np.mean(kls)),updates=len(losses),calibration=metric['aggregate']))
            write(branch/'history.json',history);announce(out,'RECOGNITION',arm=arm,epoch=epoch,calibration_f1=metric['aggregate']['f1'],best=best)
        write(branch/'memory.json',peaks('pretrained PARSeq student + clean teacher, frozen encoders, AdamW decoder',rc['batch_size'],[rc['batch_size'],3,32,128]))
        write(branch/'selection_frozen.json',best)
        del teacher,opt;release()
        for suffix,checkpoint in (('',branch/'best.pt' if best['epoch'] else None),('_final',branch/'last.pt')):
            student.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True) if checkpoint else initial_state,strict=True);student.eval()
            for phase in ('evaluation','test'):
                pp=base_predictions[phase]
                readings=old_ocr if checkpoint is None else decode_crops(recognizer,records[phase],pp,branch/(phase+suffix+'_ocr.json'),c['runtime']['ocr_batch_size'])
                metric=report(records[phase],pp,readings);results[phase+':'+arm+suffix]=metric
                write(branch/(phase+suffix+'_predictions.json'),{sid:[{**p,'corrected_text':readings[p['crop_array_sha256']]['raw_text']} for p in pool] for sid,pool in pp.items()})
                write(branch/(phase+suffix+'_results.json'),metric)
        recognizer.release();del student,recognizer;release()
    write(directory/'calibration.json',calibration)
    write(directory/'complete.json',dict(status='SUCCESS',results={k:v['aggregate'] for k,v in results.items()}))


def student_prepare_stage(c,out,index,contract,smoke):
    directory=out/'student';directory.mkdir(exist_ok=False)
    if file_sha256(Path(c['student']['checkpoint']))!=c['student']['checkpoint_sha256']:
        raise ValueError('Student checkpoint changed')
    records,_=source_records(c,index);gt,hashes=labels_for(records['fit'])
    pseudo=[r for r in read(out/'pseudo/manifest.json') if r['accepted']]
    jobs=[]
    for sid,r in records['fit'].items():
        jobs.append((sid,r,[g['bbox'] for g in gt[sid]]))
    counts={}
    for arm in ('gt_control','distilled'):
        data=directory/arm/'data';images=data/'images';labels=data/'labels';images.mkdir(parents=True);labels.mkdir()
        training=[]
        def add(name,record,boxes):
            rgb=load_rgb(record);h,w=rgb.shape[:2]
            target=images/(name+Path(record['image_path']).suffix);target.symlink_to(Path(record['image_path']))
            values=[]
            for x0,y0,x1,y1 in boxes:
                if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Student label outside image')
                values.append(f'0 {(x0+x1)/(2*w):.9f} {(y0+y1)/(2*h):.9f} {(x1-x0)/w:.9f} {(y1-y0)/h:.9f}')
            (labels/(name+'.txt')).write_text('\n'.join(values)+'\n')
            training.append(str(target))
        for sid,r,boxes in jobs:add('gt_'+sid.replace(':','__'),r,boxes)
        for i,p in enumerate(pseudo):
            if arm=='distilled':add('extra_'+str(i),p,[q['bbox'] for q in p['accepted']])
            else:
                _,r,boxes=jobs[i%len(jobs)];add('extra_'+str(i),r,boxes)
        (data/'train.txt').write_text('\n'.join(training)+'\n')
        # Trainer validation is disabled; its mandatory final internal validation
        # sees only the fit set. Its mAP never selects a scientific checkpoint.
        (data/'dataset.yaml').write_text('path: '+str(data)+'\ntrain: '+str(data/'train.txt')+'\nval: '+str(data/'train.txt')+'\nnames:\n  0: text\n')
        counts[arm]=len(training)
    if len(set(counts.values()))!=1:raise ValueError('Student update budgets differ')
    write(directory/'training_data.json',dict(counts=counts,fit_images=len(jobs),extra_images=len(pseudo),annotation_hashes=hashes,pseudo_manifest_sha256=file_sha256(out/'pseudo/manifest.json'),control_extra='Deterministic fit repeats replace pseudo images to equalize image/update counts; new images are an explicit part of the distillation treatment'))
    write(directory/'inference_index.json',{phase:entries for phase,entries in index.items() if phase!='fit'})
    write(directory/'launch.json',dict(config=c['student'],seed=c['seed'],smoke=smoke,source_checkpoint=c['student']['checkpoint'],source_checkpoint_sha256=c['student']['checkpoint_sha256']))
    write(directory/'prepare_complete.json',dict(status='SUCCESS'))


def student_evaluate_stage(c,out,index,contract,smoke):
    directory=out/'student';records,old_ocr=source_records(c,index);ocr=CropOCR(c,directory/'ocr.jsonl');ocr.rows.update(old_ocr)
    raw={arm:read(directory/arm/'inference.json') for arm in ('initial','gt_control','distilled')}
    policies={};results={};predictions={};torch.cuda.reset_peak_memory_stats()
    # Freeze all three score thresholds before accessing evaluation annotations.
    for arm in raw:
        items={}
        for phase,entries in raw[arm].items():
            items[phase]={}
            for sid,pool in entries.items():
                record=records[phase][sid];rgb=load_rgb(record);h,w=rgb.shape[:2]
                pp=[]
                for j,p in enumerate(pool):
                    box=[max(0,int(np.floor(p['bbox'][0]))),max(0,int(np.floor(p['bbox'][1]))),min(w,int(np.ceil(p['bbox'][2]))),min(h,int(np.ceil(p['bbox'][3])))]
                    if box[0]>=box[2] or box[1]>=box[3]:raise ValueError('Invalid YOLO box')
                    pp.append({**row(box,(h,w),f'yolo:{j}','yolo'),'detector_confidence':p['confidence']})
                ocr.image(rgb,pp);items[phase][sid]=pp
        grid=[]
        for threshold in c['student']['score_thresholds']:
            pp={sid:[p for p in pool if p['detector_confidence']>=threshold] for sid,pool in items['calibration'].items()}
            grid.append(dict(threshold=threshold,aggregate=report(records['calibration'],pp,ocr.rows)['aggregate']))
        policies[arm]=max(grid,key=lambda p:(p['aggregate']['f1'],p['threshold']))['threshold']
        predictions[arm]=items;write(directory/arm/'calibration.json',grid)
    write(directory/'selection_frozen.json',policies)
    for phase in ('evaluation','test'):
        phase_results={}
        for arm,items in predictions.items():
            pp={sid:[p for p in pool if p['detector_confidence']>=policies[arm]] for sid,pool in items[phase].items()}
            metric=report(records[phase],pp,ocr.rows);phase_results[arm]=metric
            write(directory/arm/(phase+'_predictions.json'),pp)
        write(directory/(phase+'_results.json'),phase_results);results[phase]={arm:v['aggregate'] for arm,v in phase_results.items()}
    write(directory/'ocr_used.json',ocr.rows);write(directory/'ocr_memory.json',peaks('qualified PARSeq on YOLO crops',32,[32,3,32,128]));ocr.release()
    write(directory/'complete.json',dict(status='SUCCESS',policies=policies,results=results))


def report_stage(c,out,index,contract,smoke):
    records,old_ocr=source_records(c,index);all_reports={};all_outputs={};all_ocr=dict(old_ocr)
    cache_float_differences=0
    for p in out.rglob('ocr.jsonl'):
        for line in p.open():
            r=json.loads(line);key=r['crop_array_sha256']
            if key in all_ocr and all_ocr[key]!=r:
                if all_ocr[key]['raw_text']!=r['raw_text']:raise ValueError('Qualified OCR transcript conflict')
                cache_float_differences+=1
            else:all_ocr[key]=r
    selectors=read(out/'selector/test_results.json');seg=read(out/'segmentation/complete.json');rec=read(out/'recognition/complete.json')
    for phase in ('evaluation','test'):
        base,_=baseline_predictions(c,records[phase],old_ocr,c['budget']);reports={'baseline_budget20':report(records[phase],base,old_ocr)};outputs={'baseline_budget20':base}
        for name,value in read(out/'selector'/(phase+'_results.json')).items():
            if name=='baseline_budget20':continue
            reports['selector_'+name]=value;outputs['selector_'+name]=read(out/'selector'/(phase+'_predictions.json'))[name]
        for arm in ('supervised','ema_only','consistency','pseudo','supervised_final','ema_only_final','consistency_final','pseudo_final'):
            reports['seg_'+arm]=read(out/'segmentation'/(phase+'_'+arm+'_results.json'));outputs['seg_'+arm]=read(out/'segmentation'/(phase+'_'+arm+'_predictions.json'))
        for arm in ('occlusion_ce','occlusion_distill'):
            reports['rec_'+arm]=read(out/'recognition'/arm/(phase+'_results.json'));outputs['rec_'+arm]=read(out/'recognition'/arm/(phase+'_predictions.json'))
            reports['rec_'+arm+'_final']=read(out/'recognition'/arm/(phase+'_final_results.json'));outputs['rec_'+arm+'_final']=read(out/'recognition'/arm/(phase+'_final_predictions.json'))
        for arm,value in read(out/'student'/(phase+'_results.json')).items():
            reports['yolo_'+arm]=value;outputs['yolo_'+arm]=read(out/'student'/arm/(phase+'_predictions.json'))
        controls={'selector_pointwise':'baseline_budget20','selector_set':'selector_pointwise','seg_supervised':'baseline_budget20','seg_ema_only':'seg_supervised','seg_consistency':'seg_supervised','seg_pseudo':'seg_supervised','rec_occlusion_ce':'baseline_budget20','rec_occlusion_distill':'rec_occlusion_ce','yolo_initial':'baseline_budget20','yolo_gt_control':'yolo_initial','yolo_distilled':'yolo_gt_control'}
        controls.update(seg_supervised_final='baseline_budget20',seg_ema_only_final='seg_supervised_final',seg_consistency_final='seg_supervised_final',seg_pseudo_final='seg_supervised_final',rec_occlusion_ce_final='baseline_budget20',rec_occlusion_distill_final='rec_occlusion_ce_final')
        rng=np.random.default_rng(c['seed']);comparisons={}
        for arm,control in controls.items():
            candidate,base_report=reports[arm],reports[control];delta=(candidate['aggregate']['f1']-base_report['aggregate']['f1'])*100
            paired=changes(candidate,base_report);rows=candidate['per_image'];by={r['sample_id']:r for r in base_report['per_image']}
            if set(by)!=set(records[phase]) or len(rows)!=len(records[phase]):raise ValueError('Metric population changed')
            arrays=np.array([[r['correct'],r['proposals'],r['gt_words'],by[r['sample_id']]['correct'],by[r['sample_id']]['proposals']] for r in rows],dtype=np.float64)
            draws=rng.integers(0,len(rows),(c['bootstrap_draws'],len(rows)));sums=arrays[draws].sum(1)
            bootstrap=200*sums[:,0]/(sums[:,1]+sums[:,2])-200*sums[:,3]/(sums[:,4]+sums[:,2])
            comparisons[arm]=dict(control=control,delta_f1_points=delta,conditional_paired_95_percentile_interval=np.quantile(bootstrap,[.025,.975]).tolist(),**paired,
                images_improved=[r['sample_id'] for r in rows if r['correct']>by[r['sample_id']]['correct']],images_worsened=[r['sample_id'] for r in rows if r['correct']<by[r['sample_id']]['correct']])
        all_reports[phase]=reports;all_outputs[phase]=outputs
        if phase=='test':
            write(out/'test_duplicate_sensitivity.json',{arm:aggregate([r for r in value['per_image'] if r['sample_id'] not in c['known_duplicate_test_ids']]) for arm,value in reports.items()})
        write(out/(phase+'_comparisons.json'),comparisons);write(out/(phase+'_results.json'),reports)
    # Audit every achieved count with a separately implemented graph matcher.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_bipartite_matching
    checks=0
    for phase,reports in all_reports.items():
        gt,_=labels_for(records[phase])
        for arm,metric in reports.items():
            by={r['sample_id']:r for r in metric['per_image']}
            for sid,pool in all_outputs[phase][arm].items():
                adj=np.zeros((len(pool),len(gt[sid])),bool)
                for i,p in enumerate(pool):
                    text=p.get('corrected_text',all_ocr.get(p.get('crop_array_sha256'),{}).get('raw_text'))
                    if text is None:raise ValueError('Audit lacks transcript')
                    for j,g in enumerate(gt[sid]):adj[i,j]=iou(p['bbox'],g['bbox'])>=.5 and text.casefold()==g['text'].casefold()
                count=int((maximum_bipartite_matching(csr_matrix(adj),perm_type='column')>=0).sum()) if adj.size else 0
                if count!=by[sid]['correct']:raise ValueError('Independent matcher disagrees: '+arm+' '+sid)
                checks+=1
    result=dict(status='SUCCESS',claim_label='approximate reproduction',result_class=read(out/'protocol.json')['result_class'],source_results_sha256=c['source_results_sha256'],cohorts={phase:{arm:r['aggregate'] for arm,r in reports.items()} for phase,reports in all_reports.items()},audit_checks=checks)
    write(out/'results.json',result);write(out/'independent_audit.json',dict(status='PASS',image_arm_checks=checks,qualified_cache_confidence_differences=cache_float_differences,results_sha256=file_sha256(out/'results.json')))
    lines=['# Teacher/student short-budget pilots','',result['result_class'],'','Primary metric: micro exact casefold (punctuation retained) + unpadded bbox IoU >= 0.5; all care GT and outputs. Historical TEST is explored.','', '| TEST arm | Correct | Outputs | F1 % | Control | Delta F1 pp | Recovered / damaged |','| --- | ---: | ---: | ---: | --- | ---: | ---: |']
    comp=read(out/'test_comparisons.json')
    for arm,r in all_reports['test'].items():
        a=r['aggregate'];v=comp.get(arm,{})
        lines.append(f"| {arm} | {a['correct']} | {a['proposals']} | {100*a['f1']:.3f} | {v.get('control','—')} | {v.get('delta_f1_points',0):+.3f} | {v.get('recovered',0)} / {v.get('damaged',0)} |")
    lines+=['','Intervals condition on fixed training and calibration, exclude one-seed variation and historical search. Short budgets do not establish convergence. A calibration-selected epoch0 is explicitly the initial model. No strict or complete context/DeepSolo result is claimed.']
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    announce(out,'AUDITED',image_arm_checks=checks)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--mode',required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--stage',choices=('selector','pseudo','segmentation','recognition','student_prepare','student_evaluate','report'),required=True);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();out=args.output
    try:
        c,out,index,contract=context(args);announce(out,args.stage.upper(),smoke=args.smoke)
        globals()[args.stage+'_stage'](c,out,index,contract,args.smoke)
        announce(out,args.stage.upper()+'_COMPLETE')
    except BaseException as error:
        if out.exists():write(out/('failure_'+args.stage+'.json'),dict(status='FAILED',type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),oom=isinstance(error,torch.cuda.OutOfMemoryError),utc=utc()))
        raise


if __name__=='__main__':main()
