"""Three APPROXIMATE frozen-OCR tests; explicit artifacts and TRAIN-only selection."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import time
import traceback

import numpy as np
import torch

from recognition.text_reconstruction import (TextPrior, apply_operations, box_iou,
    generate_operations, insertion_candidate, lexical_changes, lexical_metrics,
    operation_features, reading_order, supervised_target, synthetic_pairs, tokens,
    union_box, validate_repair, structural_oracle, preserves_letters)

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(2**20),b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
    os.replace(temporary,path)


def log(root,phase,**values):
    write(root/'progress.json',dict(phase=phase,pid=os.getpid(),**values))
    print(phase,json.dumps(values),flush=True)


def checked_config(path,mode):
    path=Path(path).resolve()
    if mode!='approximate' or REPO/'configs/approximate' not in path.parents:
        raise ValueError('Explicit isolated APPROXIMATE configuration required')
    c=read(path)
    if c['reproduction_mode']!=mode or c['experiment_kind']!='text_reconstruction_tests_v1':
        raise ValueError('Unknown reconstruction contract')
    for name,value in c['source_sha256'].items():
        if sha(Path(c['paths']['source_run'])/name)!=value:raise ValueError('Source artifact changed: '+name)
    for name,value in c['spell']['files_sha256'].items():
        if sha(Path(c['paths']['spell_snapshot'])/name)!=value:raise ValueError('NeuSpell asset changed: '+name)
    if sha(c['paths']['spell_vocabulary'])!=c['spell']['vocabulary_sha256']:
        raise ValueError('NeuSpell vocabulary changed')
    for name,value in c['denoiser']['files_sha256'].items():
        if sha(Path(c['paths']['byt5_snapshot'])/name)!=value:raise ValueError('ByT5 asset changed: '+name)
    return c


def setup(c):
    rt=c['runtime']
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=rt['cublas_workspace_config']:
        raise RuntimeError('Set the declared CUBLAS_WORKSPACE_CONFIG before execution')
    torch.set_num_threads(rt['cpu_threads']);torch.set_num_interop_threads(1)
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    if rt['device']!='cuda:0' or rt['precision']!='float32' or not torch.cuda.is_available():
        raise RuntimeError('The declared CUDA FP32 runtime is unavailable')
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0]<rt['minimum_free_bytes']:raise RuntimeError('Insufficient free VRAM')
    return dict(gpu=torch.cuda.get_device_name(0),total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','scipy','scikit-learn']},
                precision='float32',device=rt['device'],deterministic_algorithms=True,tf32=False,
                existing_gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True))


def memory(model,batch,shapes,seconds):
    torch.cuda.synchronize()
    return dict(model=model,precision='float32',device='cuda:0',batch_size=batch,input_shapes=sorted(map(list,shapes)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                seconds=seconds,oom=False)


def annotations(sample):
    from scripts.evaluate_totaltext_v1_findcontours_parseq import _gt
    if sha(sample['annotation_path'])!=sample['annotation_sha256']:raise ValueError('Annotation changed')
    return _gt(Path(sample['annotation_path']),tuple(reversed(sample['image_hw'])))


def prepare(root,c,runtime,smoke):
    source=Path(c['paths']['source_run']);protocol=read(source/'protocol.json');manifest=read(source/'final_artifact_manifest.json')
    annotation_hashes={}
    for name in ['train_annotation_hashes.json','evaluation_annotation_hashes.json','test_annotation_hashes.json']:
        annotation_hashes.update(read(source/name))
    cache={}
    with (source/'ocr_0.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);key=row['crop_array_sha256']
            if key in cache and cache[key]!=row:raise ValueError('Conflicting frozen OCR rows')
            cache[key]=row
    data={};input_files={};rgb_roles={}
    for phase,entries in protocol['index'].items():
        entries=entries[:c['smoke_images'][phase]] if smoke else entries
        data[phase]=[]
        for entry in entries:
            relative='exports/'+phase+'/'+entry['sample_id'].replace(':','__')+'.json'
            path=source/relative
            expected=manifest[relative]
            if isinstance(expected,dict):expected=expected['sha256']
            if sha(path)!=expected:raise ValueError('Frozen OCR export changed: '+relative)
            record=read(path);blocks=[]
            for b in record['arms']['mk']['baseline']:
                ocr=cache[b['crop_array_sha256']]
                blocks.append(dict(id=b['id'],bbox=b['bbox'],text=ocr['raw_text'],
                    confidence=ocr['sequence_confidence_native'],minimum_token_probability=ocr['minimum_token_probability'],
                    crop_sha256=b['crop_array_sha256']))
            item=dict(sample_id=record['sample_id'],image_sha256=record['image_sha256'],image_hw=record['image_hw'],
                      annotation_path=record['annotation_path'],annotation_sha256=sha(record['annotation_path']),blocks=reading_order(blocks))
            if annotation_hashes[item['annotation_path']]!=item['annotation_sha256']:
                raise RuntimeError('Annotation no longer matches the frozen OCR source')
            data[phase].append(item);input_files[str(path)]=expected
            rgb_roles.setdefault(item['image_sha256'],[]).append((phase,item['sample_id']))
    crossing=[v for v in rgb_roles.values() if len({r for r,s in v})>1]
    if any(any(r in ['calibration','evaluation'] for r,s in group) for group in crossing):
        raise RuntimeError('RGB-identical images cross corrector fitting/calibration/evaluation roles')
    known=sorted(s for group in crossing for r,s in group if r=='test')
    if not smoke and known!=sorted(c['known_duplicate_test_ids']):raise RuntimeError('Unexpected TRAIN/TEST RGB overlap')
    fit_texts=[w['text'] for s in data['fit'] for w in annotations(s)]
    vocabulary=read(c['paths']['spell_vocabulary'])
    prior=TextPrior(fit_texts,vocabulary['labels'])
    # Candidate construction receives no annotations, ownership IDs or validity masks.
    for phase,samples in data.items():
        for sample in samples:sample['operations']=generate_operations(sample['blocks'],prior,c['structure'])
    write(root/'observations.json',data)
    with (root/'text_prior.pkl').open('wb') as stream:pickle.dump(prior,stream)
    own_files=[REPO/'recognition/text_reconstruction.py',Path(__file__)]
    contract=dict(schema_version=1,reproduction_mode='approximate',claim_label='approximate reproduction',
        result_class='smoke-test result' if smoke else 'exploratory approximate reproduction',resolved_config=c,
        config_sha256=digest(c),source_code_sha256={str(p.relative_to(REPO)):sha(p) for p in own_files},
        inputs_sha256=input_files,observations_sha256=sha(root/'observations.json'),prior_sha256=sha(root/'text_prior.pkl'),
        populations={k:dict(images=len(v),crops=sum(len(s['blocks']) for s in v),operations=sum(len(s['operations']) for s in v)) for k,v in data.items()},
        runtime=runtime,known_rgb_overlap=crossing,fit_transcriptions=len(fit_texts),smoke=smoke,
        selection_uses_evaluation_or_test=False,independent_validation=False,
        limitation='Inherited models/splits and historical repeated exploration; no independent-generalization or strict-paper claim.')
    write(root/'protocol.json',contract);write(root/'resolved_config.json',c)
    log(root,'PREPARED',populations=contract['populations'])
    return data,prior


def load_prepared(root,c,smoke):
    p=read(root/'protocol.json')
    if p['config_sha256']!=digest(c) or p['smoke']!=smoke:raise RuntimeError('Configuration/scope drift')
    for name,value in p['source_code_sha256'].items():
        if sha(REPO/name)!=value:raise RuntimeError('Cannot resume different source code: '+name)
    if sha(root/'observations.json')!=p['observations_sha256'] or sha(root/'text_prior.pkl')!=p['prior_sha256']:
        raise RuntimeError('Prepared inputs changed')
    with (root/'text_prior.pkl').open('rb') as stream:prior=pickle.load(stream)
    return read(root/'observations.json'),prior


def baseline_groups(sample):
    return [dict(id='keep:'+b['id'],parents=[b['id']],texts=[b['text']],source_texts=[b['text']],bbox=b['bbox'],op='keep') for b in sample['blocks']]


def flatten(groups):return [text for group in groups for text in group['texts']]


def completion(root,c,data,prior):
    from transformers import AutoTokenizer
    from recognition.neuspell_correction import load_word_head,NeuSpellWordHead,SpellPolicy
    from recognition.bert_correction import eligible_word
    from recognition.bert_method_selection import MethodPolicy,insertion_selection
    if (root/'completion.json').exists():return read(root/'completion.json')
    cfg=c['spell'];snapshot=Path(c['paths']['spell_snapshot']);torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,use_fast=True,do_lower_case=True)
    model,audit=load_word_head(snapshot,device=c['runtime']['device'],expected_labels=cfg['label_count'],allowed_unused_keys=cfg['allowed_unused_keys'])
    decoder=NeuSpellWordHead(tokenizer,model,read(c['paths']['spell_vocabulary']),device=c['runtime']['device'],
                            batch_size=cfg['batch_size'],max_input_tokens=cfg['max_input_tokens'],saved_top_k=cfg['saved_top_k'])
    eligibility=SpellPolicy(**cfg['eligibility_policy'])
    unique=sorted({b['text'] for samples in data.values() for s in samples for b in s['blocks'] if eligible_word(b['text'],eligibility)})
    neural={};shapes=set()
    for start in range(0,len(unique),cfg['batch_size']*16):
        strings=unique[start:start+cfg['batch_size']*16]
        rows=decoder.predict([dict(original=s,left=[],right=[],caption=None) for s in strings]);shapes.update(decoder.input_shapes)
        neural.update(zip(strings,rows))
        if start%1024==0:log(root,'NEUSPELL',done=min(len(unique),start+len(strings)),total=len(unique))
    policy=MethodPolicy(**cfg['method_policy']);output={}
    for phase,samples in data.items():
        output[phase]={}
        for sample in samples:
            arms={name:[] for name in ['baseline','neuspell_insertions','neuspell_long_insertions']}
            for group in baseline_groups(sample):
                old=group['texts'][0];evidence=neural.get(old,{'status':'protected_word_shape'})
                short=insertion_selection(old,evidence,policy)['text']
                long=old
                if (evidence['status']=='scored' and not evidence['special_prediction'] and not evidence['reference_label_ids']
                    and evidence['probability']>=cfg['minimum_long_probability']
                    and insertion_candidate(old,evidence['prediction'],maximum_insertions=cfg['maximum_long_insertions'],minimum_length=cfg['minimum_source_length'])):
                    from recognition.bert_method_selection import restore_case
                    long=restore_case(evidence['prediction'],old)
                for name,text in [('baseline',old),('neuspell_insertions',short),('neuspell_long_insertions',long)]:
                    arms[name].append({**group,'texts':[text]})
            output[phase][sample['sample_id']]=arms
    write(root/'completion_neural.json',dict(loading_audit=audit,unique_words=neural,
          memory=memory('frozen NeuSpell trained word head',cfg['batch_size'],shapes,time.perf_counter()-tick)))
    write(root/'completion.json',output)
    del decoder,model;gc.collect();torch.cuda.empty_cache()
    return output


def operation_label(sample,operation,words):
    before=[b['text'] for b in sample['blocks']]
    after=flatten(apply_operations(sample['blocks'],[operation],[1.],0.))
    a,b=lexical_metrics(before,words),lexical_metrics(after,words)
    changes=lexical_changes(before,after,words)
    return int(b['f1']>a['f1'] and changes['damaged']==0)


def structure(root,c,data,prior):
    from sklearn.ensemble import HistGradientBoostingClassifier
    if (root/'structure.json').exists():return read(root/'structure.json')
    x,y=[],[];fit_stats=Counter()
    for sample in data['fit']:
        words=annotations(sample)
        for op in sample['operations']:
            x.append(operation_features(op,sample['blocks'],prior));y.append(operation_label(sample,op,words));fit_stats[op['op']]+=1
    if len(set(y))!=2:raise RuntimeError('Structural fitting requires both positive and negative candidates')
    cfg=c['selector']
    model=HistGradientBoostingClassifier(max_iter=cfg['iterations'],max_leaf_nodes=cfg['max_leaf_nodes'],learning_rate=cfg['learning_rate'],
        l2_regularization=cfg['l2_regularization'],min_samples_leaf=cfg['min_samples_leaf'],early_stopping=False,random_state=c['seed'])
    positives=sum(y);weights=np.array([.5/positives if value else .5/(len(y)-positives) for value in y])*len(y)
    model.fit(np.asarray(x,dtype=np.float64),np.asarray(y),sample_weight=weights)
    with (root/'structural_selector.pkl').open('wb') as stream:pickle.dump(model,stream)
    output={}
    for phase,samples in data.items():
        output[phase]={}
        for sample in samples:
            ops=sample['operations'];features=[operation_features(op,sample['blocks'],prior) for op in ops]
            scores=model.predict_proba(np.asarray(features))[:,1].tolist() if features else []
            output[phase][sample['sample_id']]=dict(scores=scores,
                policies={str(t):apply_operations(sample['blocks'],ops,scores,t) for t in cfg['thresholds']})
    calibration=[]
    for threshold in cfg['thresholds']:
        correct=damage=recovered=predictions=gt=0
        for sample in data['calibration']:
            words=annotations(sample);after=flatten(output['calibration'][sample['sample_id']]['policies'][str(threshold)])
            score=lexical_metrics(after,words);change=lexical_changes([b['text'] for b in sample['blocks']],after,words)
            correct+=score['correct'];predictions+=score['predictions'];gt+=score['gt_words'];damage+=change['damaged'];recovered+=change['recovered']
        calibration.append(dict(threshold=threshold,correct=correct,recovered=recovered,damaged=damage,
                                f1=2*correct/(predictions+gt) if predictions+gt else 0.))
    acceptable=[r for r in calibration if r['damaged']<=cfg['calibration_damage_limit']]
    if not acceptable:raise RuntimeError('KEEP structural policy must remain admissible')
    choice=max(acceptable,key=lambda r:(r['correct'],r['f1'],r['threshold']))
    write(root/'structural_calibration.json',dict(candidates=calibration,choice=choice,fit_candidates=len(y),fit_positives=positives,fit_operations=dict(fit_stats)))
    pool=[]
    for sample in data['calibration']:
        words=annotations(sample)
        pool.append(dict(sample_id=sample['sample_id'],baseline=lexical_metrics([b['text'] for b in sample['blocks']],words)['correct'],
                         **structural_oracle(sample['blocks'],sample['operations'],words)))
    write(root/'structural_calibration_pool_oracle.json',dict(diagnostic_only=True,images=pool,
          baseline=sum(r['baseline'] for r in pool),correct=sum(r['correct'] for r in pool)))
    write(root/'structure.json',output)
    log(root,'STRUCTURE_SELECTED',fit_candidates=len(y),positives=positives,choice=choice)
    return output


def training_pairs(data,c):
    pairs=[];rng=random.Random(c['seed']);counts=Counter()
    for sample in data['fit']:
        words=annotations(sample)
        groups=baseline_groups(sample)
        seen={tuple(g['parents']) for g in groups}
        for op in sample['operations']:
            key=tuple(op['parents'])
            if len(key)>1 and key not in seen:
                seen.add(key);groups.append(op)
        for group in groups:
            target=supervised_target(group['bbox'],words,minimum_coverage=c['denoiser']['training_gt_coverage'])
            if target:
                pairs.append((' | '.join(group['source_texts']),target,'real_ocr'))
        for word in words:pairs.extend(synthetic_pairs(word['text'],rng))
        # Synthetic merged word boundaries from neighboring fit annotations only.
        ordered=sorted(words,key=lambda w:((w['bbox'][1]+w['bbox'][3])/2,w['bbox'][0]))
        for left,right in zip(ordered,ordered[1:]):
            if left['text'] and right['text']:
                pairs.append((left['text']+right['text'],left['text']+' '+right['text'],'synthetic_merged_words'))
    unique=list(dict.fromkeys(pairs));rng.shuffle(unique)
    accepted=[]
    for source,target,kind in unique:
        if len(('repair: '+source).encode('utf-8'))+1>c['denoiser']['max_input_bytes'] or len(target.encode('utf-8'))+1>c['denoiser']['max_target_bytes']:
            counts['length_excluded']+=1;continue
        accepted.append(dict(source=source,target=target,kind=kind))
    if len(accepted)>c['denoiser']['maximum_training_pairs']:
        counts['budget_excluded']=len(accepted)-c['denoiser']['maximum_training_pairs'];accepted=accepted[:c['denoiser']['maximum_training_pairs']]
    counts.update(r['kind'] for r in accepted)
    return accepted,dict(counts)


def load_byt5(c):
    from transformers import AutoTokenizer,T5ForConditionalGeneration
    snapshot=Path(c['paths']['byt5_snapshot'])
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    model,info=T5ForConditionalGeneration.from_pretrained(snapshot,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
    if any(info.get(k) for k in ['missing_keys','unexpected_keys','mismatched_keys','error_msgs']):
        raise RuntimeError('ByT5 checkpoint loading failed: '+str(info))
    return tokenizer,model,info


def infer_byt5(model,tokenizer,groups,c):
    model.eval();output=[];shapes=set();cfg=c['denoiser']
    eligible=[]
    for i,group in enumerate(groups):
        source='repair: '+' | '.join(group['source_texts'])
        original=' '.join(group['texts'])
        item={**group,'raw_repair':None,'repair_reason':'input_byte_limit_KEEP'}
        output.append(item)
        protected=cfg['repair_policy'].get('protect_non_alphabetic',False) and any(not word.isalpha() for word in original.split())
        if protected:output[-1]['repair_reason']='protected_original_KEEP'
        if len(source.encode('utf-8'))+1<=cfg['max_input_bytes'] and original and not protected:
            eligible.append((i,source))
    for start in range(0,len(eligible),cfg['inference_batch_size']):
        batch=eligible[start:start+cfg['inference_batch_size']]
        encoded=tokenizer([s for i,s in batch],padding=True,return_tensors='pt',truncation=False)
        shapes.add(tuple(encoded['input_ids'].shape));encoded={k:v.cuda() for k,v in encoded.items()}
        with torch.inference_mode():
            generated=model.generate(**encoded,do_sample=False,num_beams=1,max_new_tokens=cfg['max_target_bytes'],use_cache=True)
        decoded=tokenizer.batch_decode(generated,skip_special_tokens=True)
        for (index,source),text,path in zip(batch,decoded,generated.tolist()):
            group=groups[index]
            ended=tokenizer.eos_token_id in path[1:]
            selected,reason=validate_repair(group['texts'],text,cfg['repair_policy']) if ended else (' '.join(group['texts']),'generation_limit_KEEP')
            output[index].update(texts=selected.split() or [''],raw_repair=text,repair_reason=reason,
                                 generation_ended=ended,generated_ids=path)
    return output,shapes


def score_calibration(generated,samples):
    correct=predictions=gt=recovered=damaged=0
    for sample in samples:
        words=annotations(sample);after=flatten(generated[sample['sample_id']]);before=[b['text'] for b in sample['blocks']]
        metrics=lexical_metrics(after,words);change=lexical_changes(before,after,words)
        correct+=metrics['correct'];predictions+=metrics['predictions'];gt+=metrics['gt_words'];recovered+=change['recovered'];damaged+=change['damaged']
    return dict(correct=correct,predictions=predictions,gt_words=gt,f1=2*correct/(predictions+gt) if predictions+gt else 0.,recovered=recovered,damaged=damaged)


def denoise(root,c,data,structure_output):
    cfg=c['denoiser'];selected=read(root/'structural_calibration.json')['choice']['threshold']
    # Stage-local seed makes --action all and --action denoise equivalent.
    torch.manual_seed(c['seed']);np.random.seed(c['seed']);random.seed(c['seed'])
    tokenizer,model,loading=load_byt5(c);model.cuda();model.config.use_cache=False
    torch.cuda.reset_peak_memory_stats();tick=time.perf_counter();pairs,pair_counts=training_pairs(data,c)
    if read(root/'protocol.json')['smoke']:pairs=pairs[:cfg['smoke_training_pairs']]
    write(root/'denoiser_training_pairs.json',dict(counts=pair_counts,pairs=pairs))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],foreach=False)
    rng=random.Random(c['seed']);history=[];shapes=set();selected_state=None
    best=None;calibrations=[]
    epochs=1 if read(root/'protocol.json')['smoke'] else cfg['epochs']
    for epoch in range(1,epochs+1):
        order=list(range(len(pairs)));rng.shuffle(order);model.train();total_loss=0.;batches=0
        for start in range(0,len(order),cfg['batch_size']):
            chosen=[pairs[i] for i in order[start:start+cfg['batch_size']]]
            encoded=tokenizer(['repair: '+p['source'] for p in chosen],padding=True,return_tensors='pt',truncation=False)
            target=tokenizer([p['target'] for p in chosen],padding=True,return_tensors='pt',truncation=False)
            shapes.add(tuple(encoded['input_ids'].shape));labels=target['input_ids'];labels=labels.masked_fill(target['attention_mask']==0,-100).cuda()
            encoded={k:v.cuda() for k,v in encoded.items()};optimizer.zero_grad(set_to_none=True)
            loss=model(**encoded,labels=labels,use_cache=False).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite denoiser loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();total_loss+=float(loss.detach());batches+=1
            if batches%cfg['log_batches']==0:log(root,'BYT5_TRAIN',epoch=epoch,batches=batches,total_batches=(len(order)+cfg['batch_size']-1)//cfg['batch_size'],loss=total_loss/batches,gradient_norm=float(norm))
        history.append(dict(epoch=epoch,mean_batch_loss=total_loss/batches,updates=batches,pairs=len(pairs)))
        model.config.use_cache=True;generated={}
        for sample in data['calibration']:
            groups=baseline_groups(sample);outputs,seen=infer_byt5(model,tokenizer,groups,c);shapes.update(seen);generated[sample['sample_id']]=outputs
        score=score_calibration(generated,data['calibration']);score['epoch']=epoch;calibrations.append(score)
        write(root/f'denoiser_calibration_epoch{epoch}.json',generated)
        # Pick the strongest measured endpoint; deploy/KEEP eligibility is separate.
        rank=(score['correct'],score['f1'],-score['damaged'],-epoch)
        if best is None or rank>best:
            best=rank;selected_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};selected_epoch=epoch
        write(root/'denoiser_history.json',dict(history=history,calibrations=calibrations,training_counts=pair_counts))
        log(root,'BYT5_EPOCH',**history[-1],calibration=score)
        model.config.use_cache=False
    del optimizer
    model.load_state_dict(selected_state,strict=True);model.config.use_cache=True;model.eval()
    torch.save(dict(model_state_dict=selected_state,config_sha256=digest(c),selected_epoch=selected_epoch),root/'denoiser_selected.pt')
    del selected_state;gc.collect();torch.cuda.empty_cache()
    output={}
    for phase,samples in data.items():
        if phase=='fit':continue
        output[phase]={}
        for number,sample in enumerate(samples,1):
            sid=sample['sample_id'];arms={}
            for name,groups in [('denoiser',baseline_groups(sample)),('structure_denoiser',structure_output[phase][sid]['policies'][str(selected)])]:
                generated,seen=infer_byt5(model,tokenizer,groups,c);shapes.update(seen);arms[name]=generated
                preserved=[]
                for before,after in zip(groups,generated):
                    if preserves_letters(before['texts'],' '.join(after['texts'])):
                        preserved.append(copy.deepcopy(after))
                    else:
                        preserved.append({**copy.deepcopy(after),'texts':before['texts'],'repair_reason':'letter_preservation_KEEP'})
                arms[name+'_preserve']=preserved
            output[phase][sid]=arms
            if number%25==0 or number==len(samples):log(root,'BYT5_INFER_'+phase,images=number,total=len(samples))
        write(root/f'denoiser_{phase}.json',output[phase])
    write(root/'denoiser_memory.json',dict(**memory('ByT5-small full fine-tuning then greedy inference',cfg['batch_size'],shapes,time.perf_counter()-tick),
          inference_batch_size=cfg['inference_batch_size'],loading_audit=loading,parameter_count=sum(p.numel() for p in model.parameters()),
          selected_epoch=selected_epoch,calibrations=calibrations,training_counts=pair_counts))
    del model;gc.collect();torch.cuda.empty_cache()
    return output


def choose_completion_and_denoiser(root,c,data,completion_output,denoiser_output):
    choices={}
    for name in ['neuspell_insertions','neuspell_long_insertions','denoiser','structure_denoiser','denoiser_preserve','structure_denoiser_preserve']:
        source=completion_output if name.startswith('neuspell') else denoiser_output
        generated={s['sample_id']:source['calibration'][s['sample_id']][name] for s in data['calibration']}
        score=score_calibration(generated,data['calibration'])
        baseline=score_calibration({s['sample_id']:baseline_groups(s) for s in data['calibration']},data['calibration'])
        enable=score['correct']>baseline['correct'] and score['damaged']<=c['selector']['calibration_damage_limit']
        choices[name]=dict(enabled=enable,calibration=score)
    write(root/'selection_frozen_before_evaluation.json',dict(choices=choices,structure=read(root/'structural_calibration.json')['choice'],
         source_checkpoint_sha256=c['source_sha256'],denoiser_checkpoint_sha256=sha(root/'denoiser_selected.pt'),
         annotation_transcripts_not_parsed_for_selection=['evaluation','test'],independent_validation=False))
    return choices


def spatial_score(groups,words):
    from scipy.optimize import linear_sum_assignment
    predicted=[(g['bbox'],s) for g in groups for s in g['texts']]
    adjacency=np.zeros((len(predicted),len(words)),dtype=np.int8)
    for i,(box,text) in enumerate(predicted):
        for j,word in enumerate(words):adjacency[i,j]=box_iou(box,word['bbox'])>=.5 and text.casefold()==word['text'].casefold()
    if not adjacency.size:return 0
    row,col=linear_sum_assignment(-adjacency)
    return int(adjacency[row,col].sum())


def evaluate(root,c,data,completion_output,structure_output,denoiser_output):
    selection=read(root/'selection_frozen_before_evaluation.json');threshold=selection['structure']['threshold'];result={}
    for phase in ['evaluation','test']:
        per_image=[]
        for sample in data[phase]:
            sid=sample['sample_id'];words=annotations(sample);base=baseline_groups(sample);before=flatten(base)
            arms={**completion_output[phase][sid],
                  'structure':structure_output[phase][sid]['policies'][str(threshold)],
                  'structure_raw':structure_output[phase][sid]['policies'][str(c['selector']['raw_threshold'])],
                  **denoiser_output[phase][sid]}
            for name in selection['choices']:
                arms[name+'_calibrated']=arms[name] if selection['choices'][name]['enabled'] else base
            row=dict(sample_id=sid,gt_words=len(words),known_rgb_duplicate=sid in c['known_duplicate_test_ids'],arms={})
            for name,groups in arms.items():
                texts=flatten(groups);metric=lexical_metrics(texts,words);change=lexical_changes(before,texts,words)
                row['arms'][name]=dict(**metric,**change,groups=groups)
                # Only unchanged-localization completion arms have interpretable fixed-box OCR metrics.
                if name in ['baseline','neuspell_insertions','neuspell_long_insertions']:
                    row['arms'][name]['fixed_bbox_exact']=spatial_score(groups,words)
            per_image.append(row)
        result[phase]=dict(images=len(per_image),gt_words=sum(r['gt_words'] for r in per_image),arms={})
        for name in per_image[0]['arms']:
            counts={k:sum(r['arms'][name][k] for r in per_image) for k in ['correct','predictions','gt_words','recovered','damaged']}
            tp,n,g=counts['correct'],counts['predictions'],counts['gt_words'];counts.update(precision=tp/n if n else 0.,recall=tp/g if g else 0.,f1=2*tp/(n+g) if n+g else 0.)
            if 'fixed_bbox_exact' in per_image[0]['arms'][name]:counts['fixed_bbox_exact']=sum(r['arms'][name]['fixed_bbox_exact'] for r in per_image)
            # Conditional paired image bootstrap, without fitting/search uncertainty.
            rng=np.random.default_rng(c['seed']);draws=[]
            for _ in range(c['bootstrap_draws']):
                indices=rng.integers(0,len(per_image),len(per_image));a=[per_image[i]['arms'][name] for i in indices];b=[per_image[i]['arms']['baseline'] for i in indices]
                f=lambda rows:2*sum(r['correct'] for r in rows)/sum(r['predictions']+r['gt_words'] for r in rows) if sum(r['predictions']+r['gt_words'] for r in rows) else 0.
                draws.append(f(a)-f(b))
            counts['conditional_f1_delta_interval95']=np.quantile(draws,[.025,.975]).tolist();result[phase]['arms'][name]=counts
        if not read(root/'protocol.json')['smoke']:
            expected=c['expected_populations'][phase]
            if (len(per_image),result[phase]['gt_words'],result[phase]['arms']['baseline']['fixed_bbox_exact'])!=(expected['images'],expected['care_words'],expected['fixed_bbox_exact']):
                raise RuntimeError('Frozen baseline/denominator replay disagrees: '+phase)
        if phase=='test':
            clean=[r for r in per_image if not r['known_rgb_duplicate']]
            result[phase]['without_known_rgb_duplicates']={name:{k:sum(r['arms'][name][k] for r in clean) for k in ['correct','predictions','gt_words','recovered','damaged']} for name in result[phase]['arms']}
        pool=[]
        for sample in data[phase]:
            words=annotations(sample)
            pool.append(dict(sample_id=sample['sample_id'],**structural_oracle(sample['blocks'],sample['operations'],words)))
        result[phase]['bounded_structural_pool_oracle']=dict(correct=sum(r['correct'] for r in pool),diagnostic_only=True,images=pool)
        write(root/(phase+'_per_image.json'),per_image)
        log(root,'EVALUATED_'+phase,summary=result[phase]['arms'])
    write(root/'results.json',dict(reproduction_mode='approximate',claim_label='approximate reproduction',
         result_class='smoke-test result' if read(root/'protocol.json')['smoke'] else 'exploratory approximate reproduction',
         metric='micro per-image Unicode casefold whitespace-token multiset, punctuation retained; empty observations count as predictions; care GT unchanged',
         recovered_damaged='changes in capped per-image matched text multiplicities, not spatial GT identities',
         independent_validation=False,selection=selection,cohorts=result))
    lines=['# Frozen OCR text reconstruction tests','','Approximate reproduction / exploratory component experiments. No U-Net or PARSeq updates.','',
           'Primary scores are lexical, not official polygon spotting or strict paper Table 5. Empty predictions count; all care GT remain.','']
    for phase,cohort in result.items():
        lines += ['## '+phase,'','| Arm | Correct | Outputs | Recovered | Damaged | Precision | Recall | F1 |','|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,row in cohort['arms'].items():lines.append(f"| {name} | {row['correct']} | {row['predictions']} | {row['recovered']} | {row['damaged']} | {row['precision']:.2%} | {row['recall']:.2%} | {row['f1']:.2%} |")
        lines+=['']
    lines+=['Calibration and all model choices were frozen before new evaluation/test annotation reads. The datasets were historically explored; intervals condition on fitted models and do not establish independent generalization.','',
            'Raw arms are diagnostic fixed policies. Calibrated arms may select KEEP; a preserved baseline is not evidence that a corrector works.','',
            'No full BLIP-2/MPNet/DeepSolo pipeline is evaluated or modified.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--action',required=True,choices=['prepare','completion','structure','denoise','evaluate','all'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    if (root/'status.json').exists():raise FileExistsError('Completed run is immutable; choose a fresh output directory')
    try:
        c=checked_config(args.config,args.mode);runtime=setup(c)
        if (root/'protocol.json').exists():data,prior=load_prepared(root,c,args.smoke)
        else:data,prior=prepare(root,c,runtime,args.smoke)
        if args.action=='prepare':return
        if args.action in ['completion','all']:completion(root,c,data,prior)
        if args.action in ['structure','all']:structure(root,c,data,prior)
        if args.action in ['denoise','all']:
            denoise(root,c,data,read(root/'structure.json'))
        if args.action in ['evaluate','all']:
            complete=read(root/'completion.json');struct=read(root/'structure.json')
            denoised={phase:read(root/f'denoiser_{phase}.json') for phase in ['calibration','evaluation','test']}
            choose_completion_and_denoiser(root,c,data,complete,denoised)
            evaluate(root,c,data,complete,struct,denoised)
            write(root/'status.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json')))
    except Exception as error:
        write(root/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
                                      oom=isinstance(error,torch.cuda.OutOfMemoryError)))
        raise


if __name__=='__main__':main()
