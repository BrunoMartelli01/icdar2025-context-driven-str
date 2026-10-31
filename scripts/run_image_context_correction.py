"""Controlled APPROXIMATE image-wide OCR correction battery on frozen observations."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import gc
import json
import math
import os
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

from recognition.image_context_correction import (ListwiseRanker, accept_generated, adjacent_spans,
    aligned_target, apply_choices, block_features, confusions, context_windows, corrupt_text,
    eligible, filter_readings, listwise_loss, operation_features, overlap_fraction,
    pooled_language, select_subset, unambiguous_pairs)
from recognition.contextual_boundary_selection import class_utility, neighbours
from recognition.text_reconstruction import TextPrior, generate_operations, lexical_changes, lexical_metrics, tokens, union_box
from scripts.run_contextual_boundary_selection import evaluate, bootstrap
from scripts.run_text_reconstruction_tests import annotations, digest, load_byt5, read, sha, training_pairs, write
from scripts.run_selective_word_repair_tests import setup, gpu_start, gpu_end

REPO=Path(__file__).resolve().parents[1]
PHASES=('fit','calibration','evaluation','test')


def log(root,stage,**values):
    write(root/'status.json',dict(stage=stage,pid=os.getpid(),**values))
    print(stage,json.dumps(values),flush=True)


def wait_gpu(root,c,label):
    """Queue before model execution; runtime/model/OOM failures are never retried."""
    started=time.monotonic();last_log=-float('inf')
    if not torch.cuda.is_available():raise RuntimeError('Configured CUDA device unavailable')
    while torch.cuda.mem_get_info()[0]<c['runtime']['minimum_free_bytes']:
        elapsed=time.monotonic()-started
        if elapsed>=c['runtime']['maximum_queue_seconds']:raise RuntimeError('GPU memory queue timeout; other jobs preserved')
        if elapsed-last_log>=c['runtime']['queue_log_seconds']:
            log(root,'WAITING_FOR_GPU',label=label,free_bytes=torch.cuda.mem_get_info()[0],waited_seconds=elapsed)
            last_log=elapsed
        time.sleep(c['runtime']['queue_poll_seconds'])
    record=gpu_start(c,label);record['queue_seconds']=time.monotonic()-started
    return record


def checked_config(path,mode):
    path=Path(path).resolve()
    if mode!='approximate' or REPO/'configs/approximate' not in path.parents:
        raise ValueError('Explicit isolated APPROXIMATE configuration required')
    c=read(path)
    if c['reproduction_mode']!=mode or c['experiment_kind']!='image_context_correction_v1':
        raise ValueError('Unknown experiment contract')
    for name,expected in c['input_sha256'].items():
        if sha(name)!=expected: raise ValueError('Scientific input changed: '+name)
    if c['context']['window_tokens']+c['context']['target_reserve_tokens']+3>c['context']['max_tokens']:
        raise ValueError('Context and target budgets do not fit the declared BERT limit')
    return c


def prepare(root,c,smoke):
    if (root/'protocol.json').exists(): raise ValueError('Fresh preparation requires a new output')
    source=Path(c['paths']['boundary_run']); old=read(source/'protocol.json')
    data=read(source/'candidates.json')
    if smoke: data={p:ss[:c['smoke_images'][p]] for p,ss in data.items()}
    for ss in data.values():
        for s in ss:
            if sha(s['annotation_path'])!=s['annotation_sha256']: raise ValueError('Annotation bytes changed')
    folds={s['sample_id']:old['folds'][s['sample_id']] for s in data['fit']}
    duplicates=defaultdict(list)
    for p,ss in data.items():
        for s in ss: duplicates[s['image_sha256']].append((p,s['sample_id']))
    crossing=[v for v in duplicates.values() if len({p for p,_ in v})>1]
    if any(any(p in ('calibration','evaluation') for p,_ in v) for v in crossing): raise ValueError('Cross-role duplicate RGB')
    known=sorted(sid for v in crossing for p,sid in v if p=='test')
    if not smoke and known!=sorted(c['known_duplicate_test_ids']): raise ValueError('Unexpected duplicate RGB')
    critical=['recognition/image_context_correction.py','scripts/run_image_context_correction.py',
        'recognition/text_reconstruction.py','recognition/bert_correction.py',
        'scripts/run_text_reconstruction_tests.py','scripts/run_selective_word_repair_tests.py',
        'scripts/run_contextual_boundary_selection.py']
    write(root/'data.json',data); write(root/'resolved_config.json',c)
    write(root/'protocol.json',dict(claim_label='approximate reproduction',
        result_class='smoke-test result' if smoke else 'approximate reproduction',smoke=smoke,
        config_sha256=digest(c),data_sha256=sha(root/'data.json'),runtime=setup(c),folds=folds,
        code_sha256={p:sha(REPO/p) for p in critical},duplicate_rgb=crossing,
        source_populations={p:dict(images=len(ss),blocks=sum(len(s['blocks']) for s in ss)) for p,ss in data.items()},
        independent_validation=False,strict_pipeline_changed=False))
    log(root,'PREPARED',images={p:len(ss) for p,ss in data.items()})


def load_prepared(root,c,smoke):
    p=read(root/'protocol.json')
    if p['config_sha256']!=digest(c) or p['smoke']!=smoke or p['data_sha256']!=sha(root/'data.json'):
        raise ValueError('Prepared configuration/data drift')
    for path,value in p['code_sha256'].items():
        if sha(REPO/path)!=value: raise ValueError('Resume requires unchanged source: '+path)
    return read(root/'data.json')


def pairs_for(samples,c,recipe,seed,external,smoke):
    cfg=c['generator']; rng=random.Random(seed)
    old=read(Path(c['paths']['text_run'])/'resolved_config.json');old['seed']=seed
    prior=TextPrior([w['text'] for s in samples for w in annotations(s)],external)
    if recipe=='historical':
        ss=copy.deepcopy(samples)
        for s in ss:s['operations']=generate_operations(s['blocks'],prior,old['structure'])
        rows,_=training_pairs(dict(fit=ss),old)
        rows=[dict(**r,source_ids_scope='fold_training_ids') for r in rows]
        rejected=0; empirical=[]
    else:
        rows=[];targets=[]
        for s in samples:
            words=annotations(s)
            targets.extend((w['text'],s['sample_id']) for w in words if all(t.isalpha() for t in w['text'].split()) and w['text'].strip())
            for parents in adjacent_spans(s['blocks'],c['geometry']):
                source=' | '.join(s['blocks'][i]['text'] for i in parents)
                if not all(eligible(s['blocks'][i]['text'],cfg['maximum_source_characters']) for i in parents):continue
                target=aligned_target(union_box([s['blocks'][i]['bbox'] for i in parents]),words,c['alignment'])
                if target and all(w.isalpha() for w in target.split()) and Levenshtein_ratio(source.replace(' | ',''),target.replace(' ',''))<=cfg['maximum_training_relative_edit']:
                    rows.append(dict(source=source,target=target,kind='real_aligned',source_ids=[s['sample_id']]))
                    if len(target.split())>1:targets.append((target,s['sample_id']))
        rows=[r for r in rows if len(('repair: '+r['source']).encode())+1<=cfg['max_input_bytes'] and len(r['target'].encode())+1<=cfg['max_target_bytes']]
        rows,rejected=unambiguous_pairs(rows);empirical=confusions(rows)
        real=list(rows);rng.shuffle(real);rows=real[:cfg['maximum_real_pairs']]
        if not targets: raise ValueError('No clean TRAIN targets')
        attempts=0
        while len(rows)<cfg['training_pairs'] and attempts<cfg['training_pairs']*10:
            target,sid=rng.choice(targets)
            noisy,kind=corrupt_text(target,rng,empirical,recipe,cfg)
            row=dict(source=noisy,target=target,kind='synthetic_'+kind,source_ids=[sid])
            if noisy and len(('repair: '+noisy).encode())+1<=cfg['max_input_bytes'] and len(target.encode())+1<=cfg['max_target_bytes']:
                rows.append(row)
            attempts+=1
    rows=[r for r in rows if r['source'] and len(('repair: '+r['source']).encode())+1<=cfg['max_input_bytes'] and len(r['target'].encode())+1<=cfg['max_target_bytes']]
    rng.shuffle(rows);rows=rows[:cfg['smoke_pairs'] if smoke else cfg['training_pairs']]
    if not smoke and len(rows)!=cfg['training_pairs']:
        raise ValueError('Could not construct the exact declared training-record budget')
    if not rows:raise ValueError('Empty generator training set')
    return rows,dict(kinds=dict(Counter(r['kind'] for r in rows)),conflicting_sources_removed=rejected,empirical=empirical,
        training_ids=[s['sample_id'] for s in samples],recipe=recipe,pairs=len(rows))


def Levenshtein_ratio(a,b):
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a.casefold(),b.casefold())/max(1,len(a),len(b))


def generate_neural(model,tokenizer,samples,c,root,tag):
    cfg=c['generator'];groups=[];output={s['sample_id']:[] for s in samples};shapes=set();model.eval()
    for s in samples:
        for parents in adjacent_spans(s['blocks'],c['geometry']):
            sources=[s['blocks'][i]['text'] for i in parents];source='repair: '+' | '.join(sources)
            if all(eligible(t,cfg['maximum_source_characters']) for t in sources) and len(source.encode())+1<=cfg['max_input_bytes']:
                groups.append((s['sample_id'],list(parents),sources,source))
    for start in range(0,len(groups),cfg['batch_size']):
        group=groups[start:start+cfg['batch_size']]
        encoded=tokenizer([r[3] for r in group],padding=True,return_tensors='pt')
        shapes.add(tuple(encoded['input_ids'].shape));encoded={k:v.cuda() for k,v in encoded.items()}
        with torch.inference_mode():
            prediction=model.generate(**encoded,do_sample=False,num_beams=cfg['beams'],num_return_sequences=cfg['returns'],
                max_new_tokens=cfg['max_target_bytes'],use_cache=True,length_penalty=1.,early_stopping=False)
        texts=tokenizer.batch_decode(prediction,skip_special_tokens=True)
        for j,(sid,parents,sources,_) in enumerate(group):
            values=[]
            for k in range(cfg['returns']):
                index=j*cfg['returns']+k;ended=tokenizer.eos_token_id in prediction[index,1:].tolist()
                text=accept_generated(sources,texts[index],cfg) if ended else None
                values.append(dict(raw=texts[index],ended=ended,accepted=text))
            output[sid].append(dict(parents=parents,values=values))
        if start%(cfg['batch_size']*80)==0:log(root,'GENERATOR_INFERENCE',tag=tag,done=start+len(group),total=len(groups))
    return output,sorted(map(list,shapes))


def generators(root,c,data,smoke):
    cfg=c['generator'];external=read(c['paths']['spell_vocabulary'])['labels'];folds=read(root/'protocol.json')['folds']
    old=read(Path(c['paths']['text_run'])/'resolved_config.json')
    for recipe in cfg['recipes']:
        fit_outputs={}
        for fold in [*range(c['folds']),'final']:
            tag=f'{recipe}_{fold}';work=root/'generators'/tag;work.mkdir(parents=True,exist_ok=True)
            if (work/'receipt.json').exists():
                receipt=read(work/'receipt.json')
                if sha(work/'checkpoint.pt')!=receipt['checkpoint_sha256'] or sha(work/'predictions.json')!=receipt['predictions_sha256']:
                    raise ValueError('Generator checkpoint or predictions changed')
                if fold!='final':fit_outputs.update(read(work/'predictions.json'))
                continue
            if any(work.iterdir()):raise ValueError('Incomplete generator stage requires a fresh output: '+str(work))
            train=[s for s in data['fit'] if fold=='final' or folds[s['sample_id']]!=fold]
            held=[s for s in data['fit'] if folds[s['sample_id']]==fold] if fold!='final' else [s for p in PHASES[1:] for s in data[p]]
            if not train or not held:raise ValueError('Empty generator training/prediction cohort')
            seed=c['seed']+(fold if isinstance(fold,int) else c['folds'])
            pairs,details=pairs_for(train,c,recipe,seed,external,smoke)
            write(work/'pairs.json',dict(**details,records=pairs,prediction_ids=[s['sample_id'] for s in held]))
            torch.manual_seed(seed);rt=wait_gpu(root,c,tag);tick=time.monotonic()
            tokenizer,model,loading=load_byt5(old);model.cuda();model.train();model.config.use_cache=False
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],foreach=False)
            order=list(range(len(pairs)));random.Random(seed).shuffle(order);losses=[];shapes=set()
            for start in range(0,len(order),cfg['batch_size']):
                rows=[pairs[j] for j in order[start:start+cfg['batch_size']]]
                encoded=tokenizer(['repair: '+r['source'] for r in rows],padding=True,return_tensors='pt')
                targets=tokenizer([r['target'] for r in rows],padding=True,return_tensors='pt')
                shapes.add(tuple(encoded['input_ids'].shape));encoded={k:v.cuda() for k,v in encoded.items()}
                labels=targets['input_ids'].masked_fill(targets['attention_mask']==0,-100).cuda()
                optimizer.zero_grad(set_to_none=True);loss=model(**encoded,labels=labels,use_cache=False).loss
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite generator loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
                losses.append(float(loss.detach()))
                if len(losses)%100==0:log(root,'GENERATOR_TRAIN',tag=tag,updates=len(losses),loss=float(np.mean(losses)))
            del optimizer,loss,labels,encoded,targets;gc.collect();torch.cuda.empty_cache();model.config.use_cache=True
            torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},training_ids=details['training_ids'],recipe=recipe),work/'checkpoint.pt')
            prediction,infer_shapes=generate_neural(model,tokenizer,held,c,root,tag)
            write(work/'predictions.json',prediction)
            receipt=dict(**gpu_end(rt,'ByT5-small full fine-tuning and beam inference',cfg['batch_size'],sorted(map(list,shapes)),tick),
                loading=loading,updates=len(losses),mean_training_loss=float(np.mean(losses)),inference_shapes=infer_shapes,
                checkpoint_sha256=sha(work/'checkpoint.pt'),predictions_sha256=sha(work/'predictions.json'),training=details)
            write(work/'receipt.json',receipt)
            if fold!='final':fit_outputs.update(prediction)
            del model;gc.collect();torch.cuda.empty_cache()
        write(root/f'neural_{recipe}_fit.json',fit_outputs)
        final=read(root/'generators'/f'{recipe}_final'/'predictions.json')
        for p in PHASES[1:]:write(root/f'neural_{recipe}_{p}.json',{s['sample_id']:final[s['sample_id']] for s in data[p]})


def build_pool(root,c,data):
    for phase,samples in data.items():
        predictions={r:read(root/f'neural_{r}_{phase}.json') for r in c['generator']['recipes']}
        for s in samples:
            ops=[];seen={}
            def add(parents,text,origin):
                if text==' '.join(s['blocks'][i]['text'] for i in parents):return
                key=(tuple(parents),text.casefold())
                if key in seen:
                    ops[seen[key]]['origins'].append(origin);return
                same=''.join(s['blocks'][i]['text'] for i in parents).casefold()==''.join(text.split()).casefold()
                kind='fragment' if len(parents)>1 else 'boundary' if same else 'character'
                seen[key]=len(ops);ops.append(dict(parents=list(parents),text=text,origins=[origin],kind=kind))
            for i,options in enumerate(s['candidates']):
                for o in options[1:]:add([i],o['text'],'original_boundary_pool')
            for recipe,output in predictions.items():
                for group in output[s['sample_id']]:
                    for v in group['values']:
                        if v['accepted']:add(group['parents'],v['accepted'],recipe)
            for parents in adjacent_spans(s['blocks'],c['geometry']):
                if len(parents)>1:
                    texts=[s['blocks'][i]['text'] for i in parents]
                    if all(eligible(t,c['generator']['maximum_source_characters']) for t in texts):add(parents,''.join(texts),'literal_join')
            s['pool']=ops
    write(root/'pool.json',data)
    log(root,'POOL',counts={p:dict(Counter(o['kind'] for s in ss for o in s['pool'])) for p,ss in data.items()})


def score_language(root,c,data):
    from transformers import AutoTokenizer,AutoModelForMaskedLM
    from recognition.bert_correction import BertWordScorer
    cfg=c['context'];rt=wait_gpu(root,c,'Image-wide BERT counterfactual evidence');tick=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(c['paths']['bert_snapshot'],local_files_only=True,use_fast=True)
    model,loading=AutoModelForMaskedLM.from_pretrained(c['paths']['bert_snapshot'],local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,output_loading_info=True)
    if any(loading[k] for k in ('missing_keys','mismatched_keys','error_msgs')) or set(loading['unexpected_keys'])-set(cfg['allowed_unused_keys']):
        raise ValueError('BERT loading audit failed: '+str(loading))
    model.cuda();scorer=BertWordScorer(tokenizer,model,device='cuda:0',batch_size=cfg['batch_size'],max_tokens=cfg['max_tokens'])
    cache={};counts=Counter()
    # Exact-key reuse of previously verified scores, including original references.
    source=Path(c['paths']['boundary_run'])
    for phase,samples in data.items():
        language=read(source/f'language_{phase}.json')
        for s in samples:
            for i,options in enumerate(s['candidates']):
                evidence=language[s['sample_id']][i]
                if evidence is None:continue
                li,ri=s['contexts'][i];left=tuple(s['blocks'][j]['text'] for j in li);right=tuple(s['blocks'][j]['text'] for j in ri)
                for key,l,r in [('word',(),()),('context',left,right)]:
                    for option,row in zip(options,evidence[key]):cache[(option['text'],l,r,None)]=row
    def scored(texts,left=(),right=(),caption=None):
        keys=[(text,tuple(left),tuple(right),caption) for text in texts];missing=list(dict.fromkeys(k for k in keys if k not in cache))
        if missing:
            rows=scorer.score([k[0] for k in missing],left=left,right=right,caption=caption)
            for key,row in zip(missing,rows):
                counts['inferred_candidates']+=1
                if row['status']=='input_token_limit':raise ValueError('Context window exceeded budget; no silent truncation')
                counts[row['status']]+=1;cache[key]=row
        return [cache[k] for k in keys]
    for phase,samples in data.items():
        target=root/f'language_{phase}.json'
        if target.exists():raise ValueError('Language evidence exists; use a fresh output or a later stage')
        output={}
        for n,s in enumerate(samples):
            sid=s['sample_id'];groups=defaultdict(list)
            for j,op in enumerate(s['pool']):groups[tuple(op['parents'])].append(j)
            records=[None]*len(s['pool']);context_audit=[];original_global=[[] for _ in s['blocks']]
            wrong=samples[(n+1)%len(samples)]
            for parents,indices in groups.items():
                original=' '.join(s['blocks'][i]['text'] for i in parents)
                texts=[original]+[s['pool'][j]['text'] for j in indices]
                window_budget=min(cfg['window_tokens'],cfg['max_tokens']-3-max(len(tokenizer.encode(t,add_special_tokens=False)) for t in texts))
                if window_budget<1:raise ValueError('Target alone exhausts BERT capacity; no silent exclusion')
                li,ri=neighbours(s['blocks'],parents[0],cfg['local'])
                left=[s['blocks'][j]['text'] for j in li if j not in parents];right=[s['blocks'][j]['text'] for j in ri if j not in parents]
                features=[{} for _ in indices]
                for name,l,r in [('word',[],[]),('local',left,right)]:
                    rows=scored(texts,l,r)
                    for j in range(len(indices)):
                        a,b=rows[0],rows[j+1];valid=a['status']==b['status']=='scored'
                        features[j][name]=([a['mean_log_probability'],b['mean_log_probability'],b['mean_log_probability']-a['mean_log_probability'],
                            a['sum_log_probability'],b['sum_log_probability'],b['sum_log_probability']-a['sum_log_probability'],a['token_count'],b['token_count'],1.]
                            if valid else [0.]*8+[0.])
                for name,blocks,excluded,mode in [('ordered',s['blocks'],parents,'ordered'),('inventory',s['blocks'],parents,'inventory'),
                                                ('wrong',wrong['blocks'],[],'inventory')]:
                    # A target anchor is needed by windowing even when all context comes from another image.
                    windows=context_windows(blocks,excluded if excluded else [-1],tokenizer,window_budget,mode,cfg['confidence_floor'])
                    if not windows:windows=[dict(left=[],right=[],caption=None,weight=1.,source_indices=[],token_ids=[])]
                    context_audit.append(dict(parents=list(parents),mode=name,source_sample_id=wrong['sample_id'] if name=='wrong' else sid,window_budget=window_budget,windows=windows))
                    values=[[] for _ in texts];weights=[];valids=[True]*len(texts)
                    for w in windows:
                        rows=scored(texts,w['left'],w['right'],w['caption']);weights.append(w['weight'])
                        for j,r in enumerate(rows):
                            valids[j]&=r['status']=='scored';values[j].append(r['mean_log_probability'] if r['status']=='scored' else None)
                    for j in range(len(indices)):
                        valid=valids[0] and valids[j+1]
                        features[j][name]=pooled_language(values[0],values[j+1],weights)+[1.] if valid else [0.]*9+[0.]
                    if len(parents)==1 and name=='inventory' and valids[0]:
                        original_global[parents[0]]=[float(np.mean(values[0])),float(np.std(values[0])),len(windows),1.]
                for index,f in zip(indices,features):records[index]=f
            # Blocks without alternatives get an explicit unavailable-evidence flag, never a fabricated likelihood.
            original_global=[v or [0.,0.,0.,0.] for v in original_global]
            output[sid]=dict(operations=records,context_audit=context_audit,original_global=original_global)
            if n%20==0:log(root,'IMAGE_CONTEXT',phase=phase,images=n+1,total=len(samples),new_candidates=counts['inferred_candidates'])
        write(target,output)
    write(root/'language_receipt.json',dict(**gpu_end(rt,'Frozen BERT target PLL, all original OCR windows',cfg['batch_size'],
        dict(max_input_tokens_seen=scorer.max_input_tokens_seen),tick),loading=loading,counts=dict(counts),cache_entries=len(cache)))
    del model;gc.collect();torch.cuda.empty_cache()


def features(root,c,data):
    external={s.casefold() for s in read(c['paths']['spell_vocabulary'])['labels']}
    folds=read(root/'protocol.json')['folds'];priors={}
    for fold in [*range(c['folds']),'final']:
        ss=[s for s in data['fit'] if fold=='final' or folds[s['sample_id']]!=fold]
        priors[str(fold)]=TextPrior([w['text'] for s in ss for w in annotations(s)],external)
    for phase,samples in data.items():
        lm=read(root/f'language_{phase}.json');out={}
        for s in samples:
            sid=s['sample_id'];prior=priors[str(folds[sid])] if phase=='fit' else priors['final'];rows=[]
            for op,e in zip(s['pool'],lm[sid]['operations']):
                base=operation_features(s['blocks'],op,prior,external);word=e['word'];local=e['local'];ordered=e['ordered'];inventory=e['inventory'];wrong=e['wrong']
                rows.append(dict(word=base+word,local=base+word+local,ordered=base+word+ordered,
                    inventory=base+word+inventory,global_context=base+word+local+ordered+inventory,wrong_context=base+word+wrong))
            out[sid]=dict(operations=rows,blocks=[block_features(s['blocks'],i,prior,external,lm[sid]['original_global'][i]) for i in range(len(s['blocks']))])
        write(root/f'features_{phase}.json',out)


def operation_label(sample,op,words,c):
    before=[b['text'] for b in sample['blocks']];after,_=apply_choices(sample['blocks'],[op],[1.],0.)
    change=lexical_changes(before,after,words)
    label=2 if change['damaged'] else 1 if change['recovered'] else 0
    target=aligned_target(union_box([sample['blocks'][i]['bbox'] for i in op['parents']]),words,c['alignment'])
    # Clean-label control removes ambiguous target ownership and partial-match "successes".
    clean=target is not None and (label!=1 or tokens([op['text']])==tokens([target]))
    return dict(label=label,clean=clean,aligned_target=target,**change)


def model_specs():
    return [dict(name='boundary_'+f,subset='boundary',feature=f,model='hgb',clean=False,balanced=False)
        for f in ('word','local','ordered','inventory','global_context','wrong_context')]+[
        dict(name='boundary_global_balanced',subset='boundary',feature='global_context',model='hgb',clean=False,balanced=True),
        dict(name='boundary_global_clean',subset='boundary',feature='global_context',model='hgb',clean=True,balanced=False),
        dict(name='boundary_global_listwise',subset='boundary',feature='global_context',model='listwise',clean=False,balanced=False),
        dict(name='character_word',subset='character',feature='word',model='hgb',clean=False,balanced=False),
        dict(name='character_global',subset='character',feature='global_context',model='hgb',clean=False,balanced=False),
        dict(name='fragment_word',subset='fragment',feature='word',model='hgb',clean=False,balanced=False),
        dict(name='fragment_global',subset='fragment',feature='global_context',model='hgb',clean=False,balanced=False),
        dict(name='expanded_global',subset='expanded',feature='global_context',model='hgb',clean=False,balanced=False),
        dict(name='expanded_global_clean',subset='expanded',feature='global_context',model='hgb',clean=True,balanced=False)]


def fit_selector(data,feature_data,labels,spec,c):
    rows=[];ys=[];groups=[]
    for s in data['fit']:
        sid=s['sample_id'];by=defaultdict(list)
        for i in select_subset(s['pool'],spec['subset']):
            label=labels[sid][i]
            if spec['clean'] and not label['clean']:continue
            rows.append(feature_data['fit'][sid]['operations'][i][spec['feature']]);ys.append(label['label']);by[tuple(s['pool'][i]['parents'])].append(len(rows)-1)
        groups.extend(by.values())
    x=np.asarray(rows,dtype=float);y=np.asarray(ys,dtype=int);cfg=c['selector']
    if not len(rows) or len(set(ys))<2:raise ValueError('Selector requires nonempty multi-class supervision')
    if spec['model']=='hgb':
        weight=None
        if spec['balanced']:
            counts=Counter(ys);weight=np.asarray([len(y)/(len(counts)*counts[v]) for v in ys])
        model=HistGradientBoostingClassifier(max_iter=cfg['iterations'],max_leaf_nodes=cfg['max_leaf_nodes'],
            min_samples_leaf=cfg['min_samples_leaf'],learning_rate=cfg['learning_rate'],l2_regularization=cfg['l2_regularization'],
            early_stopping=False,random_state=c['seed']).fit(x,y,sample_weight=weight)
        trained=dict(kind='hgb',model=model)
    else:
        torch.manual_seed(c['seed']);mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);z=torch.tensor((x-mean)/scale,dtype=torch.float32)
        model=ListwiseRanker(x.shape[1],cfg['listwise_hidden']);optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['listwise_learning_rate'],weight_decay=.01)
        rng=random.Random(c['seed']);history=[]
        for epoch in range(cfg['listwise_epochs']):
            order=list(range(len(groups)));rng.shuffle(order);losses=[]
            for start in range(0,len(order),cfg['listwise_batch_groups']):
                selected=[groups[i] for i in order[start:start+cfg['listwise_batch_groups']]];optimizer.zero_grad(set_to_none=True)
                loss=torch.stack([listwise_loss(model(z[g]),y[g].tolist()) for g in selected]).mean()
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite listwise loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            history.append(float(np.mean(losses)))
        model.eval();trained=dict(kind='listwise',model=model,mean=mean,scale=scale,loss=history)
    return trained,dict(rows=len(rows),classes=dict(Counter(ys)),groups=len(groups),dimensions=x.shape[1])


def predict_selector(trained,rows):
    if not rows:return []
    x=np.asarray(rows,dtype=float)
    if trained['kind']=='hgb':
        model=trained['model'];p=model.predict_proba(x)
        return (p[:,list(model.classes_).index(1)] if 1 in model.classes_ else np.zeros(len(x))).__sub__(
            p[:,list(model.classes_).index(2)] if 2 in model.classes_ else np.zeros(len(x))).tolist()
    with torch.inference_mode():return trained['model'](torch.tensor((x-trained['mean'])/trained['scale'],dtype=torch.float32)).tolist()


def select_predictions(samples,score,threshold,subset):
    output={}
    for s in samples:
        indices=select_subset(s['pool'],subset)
        output[s['sample_id']]=apply_choices(s['blocks'],[s['pool'][i] for i in indices],score[s['sample_id']],threshold)[0]
    return output


def calibrate(samples,score,subset,c):
    curve=[]
    for threshold in c['selector']['thresholds']:
        pred=select_predictions(samples,score,threshold,subset)
        curve.append(dict(threshold=threshold,**evaluate(samples,pred)['aggregate']))
    choices={}
    for cap in c['selector']['damage_limits']:
        valid=[r for r in curve if r['damaged']<=cap]
        if not valid:raise ValueError('KEEP must be admissible')
        for objective in ('correct','f1'):
            key=lambda r:((r['correct'],r['f1'],r['threshold']) if objective=='correct' else (r['f1'],r['correct'],r['threshold']))
            choices[f'{objective}_cap{cap}']=max(valid,key=key)
    return dict(curve=curve,choices=choices)


def selectors(root,c,data):
    feature_data={p:read(root/f'features_{p}.json') for p in PHASES};labels={}
    for s in data['fit']:
        words=annotations(s);labels[s['sample_id']]=[operation_label(s,o,words,c) for o in s['pool']]
    write(root/'fit_labels.json',labels);all_scores={};choices={};models={}
    for spec in model_specs():
        name=spec['name'];trained,details=fit_selector(data,feature_data,labels,spec,c);models[name]=trained
        score={}
        for p in PHASES[1:]:
            score[p]={}
            for s in data[p]:
                indices=select_subset(s['pool'],spec['subset'])
                score[p][s['sample_id']]=predict_selector(trained,[feature_data[p][s['sample_id']]['operations'][i][spec['feature']] for i in indices])
        calibrated=calibrate(data['calibration'],score['calibration'],spec['subset'],c)
        choices[name]=dict(spec=spec,training=details,**calibrated);all_scores[name]=score
        log(root,'SELECTOR',name=name,calibration=calibrated['choices']['correct_cap2'])
    # Literal raw generator comparisons isolate matched training recipes without learned selection.
    generator_choices={}
    for recipe in c['generator']['recipes']:
        pred=raw_generator_predictions(root,c,data['calibration'],'calibration',recipe)
        generator_choices[recipe]=evaluate(data['calibration'],pred)['aggregate']
    with (root/'selectors.pkl').open('wb') as stream:pickle.dump(models,stream)
    write(root/'scores.json',all_scores)
    write(root/'selection_frozen.json',dict(models=choices,generator_calibration=generator_choices,
        evaluation_test_used_for_selection=False,scores_sha256=sha(root/'scores.json')))


def raw_generator_predictions(root,c,samples,phase,recipe):
    neural=read(root/f'neural_{recipe}_{phase}.json');out={}
    for s in samples:
        best={tuple(g['parents']):g for g in neural[s['sample_id']] if len(g['parents'])==1}
        out[s['sample_id']]=[]
        for i,b in enumerate(s['blocks']):
            g=best.get((i,));value=g['values'][0]['accepted'] if g else None
            out[s['sample_id']].append(value or b['text'])
    return out


def filters(root,c,data):
    feature_data={p:read(root/f'features_{p}.json') for p in PHASES};x=[];y=[];presence=[]
    for s in data['fit']:
        words=annotations(s);remaining=Counter(w['text'].casefold() for w in words)
        for i,b in enumerate(s['blocks']):
            text=b['text'].casefold();correct=remaining[text]>0
            if correct:remaining[text]-=1
            x.append(feature_data['fit'][s['sample_id']]['blocks'][i]);y.append(int(correct))
            presence.append(int(any(overlap_fraction(b['bbox'],w['bbox'])>=c['alignment']['presence_gt_coverage'] for w in words)))
    cfg=c['selector'];models=[]
    for target in (y,presence):
        if len(set(target))<2:raise ValueError('Both filter classes required')
        models.append(HistGradientBoostingClassifier(max_iter=cfg['iterations'],max_leaf_nodes=cfg['max_leaf_nodes'],
            min_samples_leaf=cfg['min_samples_leaf'],learning_rate=cfg['learning_rate'],l2_regularization=cfg['l2_regularization'],
            early_stopping=False,random_state=c['seed']).fit(np.asarray(x),target))
    scores={}
    for p in PHASES[1:]:
        scores[p]={}
        for s in data[p]:
            rows=np.asarray(feature_data[p][s['sample_id']]['blocks'])
            scores[p][s['sample_id']]=dict(exact=models[0].predict_proba(rows)[:,1].tolist(),presence=models[1].predict_proba(rows)[:,1].tolist()) if len(rows) else dict(exact=[],presence=[])
    selection={}
    for policy in ('exact','presence','duplicate'):
        curve=[]
        for threshold in c['filter']['thresholds']:
            for pt in (c['filter']['presence_thresholds'] if policy=='presence' else [0.]):
                pred=filter_predictions(data['calibration'],scores['calibration'],threshold,policy,pt)
                curve.append(dict(threshold=threshold,presence_threshold=pt,**evaluate(data['calibration'],pred)['aggregate']))
        choices={}
        for cap in c['selector']['damage_limits']:
            valid=[r for r in curve if r['damaged']<=cap]
            choices[f'cap{cap}']=max(valid,key=lambda r:(r['f1'],r['correct'],-r['threshold'],-r['presence_threshold']))
        selection[policy]=dict(curve=curve,choices=choices)
    with (root/'filters.pkl').open('wb') as stream:pickle.dump(models,stream)
    write(root/'filter_scores.json',scores)
    write(root/'filter_selection_frozen.json',dict(selection=selection,evaluation_test_used_for_selection=False,
        fit_exact_labels=dict(Counter(y)),fit_presence_proxy_labels=dict(Counter(presence)),scores_sha256=sha(root/'filter_scores.json')))


def filter_predictions(samples,score,threshold,policy,pt):
    return {s['sample_id']:filter_readings(s['blocks'],score[s['sample_id']]['exact'],threshold,policy,
        score[s['sample_id']]['presence'],pt)[0] for s in samples}


def report(root,c,data):
    frozen=read(root/'selection_frozen.json');scores=read(root/'scores.json')
    fs=read(root/'filter_scores.json');fc=read(root/'filter_selection_frozen.json')
    if frozen['scores_sha256']!=sha(root/'scores.json') or fc['scores_sha256']!=sha(root/'filter_scores.json'):raise ValueError('Score drift after calibration')
    output={};predictions={};comparisons={}
    old=read(Path(c['paths']['boundary_run'])/'predictions.json')
    for p in PHASES[1:]:
        samples=data[p];arms={'baseline':{s['sample_id']:[b['text'] for b in s['blocks']] for s in samples}}
        for name in ('historical_boundary','word_lm_seed20261024_cap2'):
            arms['reference_'+name]={s['sample_id']:old[p][name][s['sample_id']] for s in samples}
        for name,choice in frozen['models'].items():
            for key,value in choice['choices'].items():
                arms[name+'_'+key]=select_predictions(samples,scores[name][p],value['threshold'],choice['spec']['subset'])
            arms[name+'_raw']=select_predictions(samples,scores[name][p],0.,choice['spec']['subset'])
        for recipe in c['generator']['recipes']:arms['generator_'+recipe+'_raw']=raw_generator_predictions(root,c,samples,p,recipe)
        for policy,choice in fc['selection'].items():
            for key,value in choice['choices'].items():
                arms['filter_'+policy+'_'+key]=filter_predictions(samples,fs[p],value['threshold'],policy,value['presence_threshold'])
        # All model, threshold and filter choices are already frozen before this first EVAL/TEST annotation read.
        output[p]={name:evaluate(samples,pred) for name,pred in arms.items()};predictions[p]=arms
        if p in ('evaluation','test'):
            comparisons[p]={name:bootstrap(output[p]['baseline'],value,c['bootstrap_draws'],c['seed']) for name,value in output[p].items() if name!='baseline'}
            for a,b in [('boundary_word','boundary_local'),('boundary_word','boundary_ordered'),('boundary_word','boundary_inventory'),
                        ('boundary_word','boundary_global_context'),('boundary_wrong_context','boundary_global_context'),
                        ('boundary_global_context','boundary_global_balanced'),('boundary_global_context','boundary_global_clean'),
                        ('boundary_global_context','boundary_global_listwise'),('boundary_global_context','character_global'),
                        ('boundary_global_context','fragment_global'),('boundary_global_context','expanded_global')]:
                na=a+'_correct_cap2';nb=b+'_correct_cap2';comparisons[p][nb+'_versus_'+na]=bootstrap(output[p][na],output[p][nb],c['bootstrap_draws'],c['seed'])
        log(root,'REPORT',phase=p,arms=len(arms),baseline=output[p]['baseline']['aggregate'])
    write(root/'predictions.json',predictions)
    result=dict(status='SUCCESS',claim_label='approximate reproduction',metric='micro per-image casefold whitespace-token multiset; punctuation and empty observations retained; all care GT; no IoU or ignored-prediction neutralization',
        protocol_sha256=sha(root/'protocol.json'),selection_sha256=sha(root/'selection_frozen.json'),filter_selection_sha256=sha(root/'filter_selection_frozen.json'),
        results=output,comparisons=comparisons,
        routing={p:dict(images_sent_to_deepsolo=0,total_evaluated_images=len(data[p]),percentage=0.,
            execution='offline correction of saved PARSeq observations; no online DeepSolo inference') for p in PHASES[1:]},limitations=[
            'Historically explored cohorts; source calibration exposure and two known TRAIN/TEST RGB duplicates.',
            'One generator/selector seed; conditional image bootstrap omits training variation and historical selection.',
            'New method is an unvalidated adaptation, not faithful tokenization-paper reproduction or an established novel method.',
            'Presence supervision is care-annotation coverage, not physical text-presence ground truth.',
            'Clean alignment and synthetic empirical noise are assumptions, not manually audited perfect labels.',
            'Window pooling includes all other OCR tokens but does not jointly encode an arbitrarily long image transcript.',
            'Current RGB and recognizer weights unchanged; no new crops or OCR; RTX3090 does not qualify 8GiB.'])
    write(root/'results.json',result)
    lines=['# Image-context correction battery','','Approximate reproduction; explored lexical transcription cohorts.','',
        '| TEST arm | Correct | Outputs | FP | Recovered | Damaged | F1 | DeepSolo images |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    names=['baseline','reference_historical_boundary','reference_word_lm_seed20261024_cap2']+[s['name']+'_correct_cap2' for s in model_specs()]+[
        'generator_'+r+'_raw' for r in c['generator']['recipes']]+['filter_'+p+'_cap2' for p in ('exact','presence','duplicate')]
    for name in names:
        r=output['test'][name]['aggregate'];lines.append(f"| {name} | {r['correct']} | {r['predictions']} | {r['fp']} | {r['recovered']} | {r['damaged']} | {r['f1']*100:.4f}% | 0/{len(data['test'])} (0.00%) |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    write(root/'completion.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json')))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True);parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path);parser.add_argument('--stage',choices=['all','prepare','generators','pool','language','features','selectors','filters','report'],default='all')
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    try:
        c=checked_config(args.config,args.mode)
        if args.stage in ('all','prepare'):prepare(root,c,args.smoke)
        else:setup(c)
        data=load_prepared(root,c,args.smoke)
        if args.stage in ('all','generators'):generators(root,c,data,args.smoke)
        if args.stage in ('all','pool'):build_pool(root,c,data)
        if args.stage in ('all','language'):score_language(root,c,read(root/'pool.json'))
        if args.stage in ('all','features'):features(root,c,read(root/'pool.json'))
        if args.stage in ('all','selectors'):selectors(root,c,read(root/'pool.json'))
        if args.stage in ('all','filters'):filters(root,c,read(root/'pool.json'))
        if args.stage in ('all','report'):report(root,c,read(root/'pool.json'))
        log(root,'STAGE_COMPLETE',completed=args.stage)
    except Exception as exc:
        write(root/'failure.json',dict(status='FAILED',exception=type(exc).__name__,message=str(exc),oom=isinstance(exc,torch.cuda.OutOfMemoryError),
            allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
