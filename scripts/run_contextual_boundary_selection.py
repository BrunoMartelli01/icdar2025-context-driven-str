"""Controlled APPROXIMATE boundary reranking with image-separated OOF proposals."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gc
import json
import os
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
import torch

from recognition.contextual_boundary_selection import (candidates, class_utility,
    edit_label, image_fold, language_features, lexical_features, neighbours,
    projection_profile, select, visual_features)
from recognition.text_reconstruction import TextPrior, generate_operations, lexical_changes, lexical_metrics
from scripts.run_text_reconstruction_tests import (annotations, digest, load_byt5,
    read, sha, training_pairs, write)
from scripts.run_selective_word_repair_tests import setup, gpu_start, gpu_end

REPO = Path(__file__).resolve().parents[1]
PHASES = ('fit', 'calibration', 'evaluation', 'test')
ARMS = ('lexical', 'word_lm', 'context', 'context_visual')


def log(root, stage, **values):
    write(root/'status.json', dict(stage=stage, pid=os.getpid(), **values))
    print(stage, json.dumps(values), flush=True)


def checked_config(path, mode):
    path = Path(path).resolve()
    if mode != 'approximate' or REPO/'configs/approximate' not in path.parents:
        raise ValueError('An explicit isolated APPROXIMATE configuration is required')
    c = read(path)
    if c['reproduction_mode'] != mode or c['experiment_kind'] != 'contextual_boundary_selection_v1':
        raise ValueError('Unknown scientific configuration')
    for path, expected in c['input_sha256'].items():
        if sha(path) != expected:
            raise ValueError('Scientific input changed: '+path)
    return c


def prepare(root, c, smoke):
    data = read(Path(c['paths']['text_run'])/'observations.json')
    if smoke:
        data = {k: v[:c['smoke_images'][k]] for k, v in data.items()}
    for samples in data.values():
        for s in samples:
            if sha(s['annotation_path']) != s['annotation_sha256']:
                raise ValueError('Source annotation identity changed')
    folds = {s['sample_id']: image_fold(s['sample_id'], c['oof']['folds'], c['seed']) for s in data['fit']}
    pixels = {}
    for phase, samples in data.items():
        for s in samples:
            pixels.setdefault(s['image_sha256'], []).append([phase, s['sample_id']])
    duplicates = [v for v in pixels.values() if len({p for p, _ in v}) > 1]
    if any(any(p in ('calibration', 'evaluation') for p, _ in v) for v in duplicates):
        raise ValueError('RGB identity crosses fitting/calibration/evaluation roles')
    if any(len({folds[sid] for phase, sid in v if phase == 'fit'}) > 1 for v in pixels.values()):
        raise ValueError('Identical FIT RGBs cross OOF folds')
    known = sorted(sid for v in duplicates for p, sid in v if p == 'test')
    if not smoke and known != sorted(c['known_duplicate_test_ids']):
        raise ValueError('Unexpected cross-role RGB duplicate')
    critical = ['recognition/contextual_boundary_selection.py', 'scripts/run_contextual_boundary_selection.py',
                'recognition/text_reconstruction.py', 'recognition/bert_correction.py',
                'scripts/run_text_reconstruction_tests.py', 'scripts/run_selective_word_repair_tests.py']
    protocol = dict(resolved_config=c, config_sha256=digest(c), smoke=smoke,
        claim_label='approximate reproduction', result_class='smoke-test result' if smoke else 'approximate reproduction',
        runtime=setup(c), folds=folds, duplicate_rgb=duplicates, independent_validation=False,
        code_sha256={p: sha(REPO/p) for p in critical},
        source_populations={k: dict(images=len(v), blocks=sum(len(s['blocks']) for s in v)) for k,v in data.items()})
    write(root/'data.json', data); protocol['data_sha256'] = sha(root/'data.json')
    write(root/'protocol.json', protocol); log(root, 'PREPARED', populations=protocol['source_populations'])


def load_prepared(root, c, smoke):
    p = read(root/'protocol.json')
    if p['config_sha256'] != digest(c) or p['smoke'] != smoke or sha(root/'data.json') != p['data_sha256']:
        raise ValueError('Prepared configuration or data changed')
    for path, expected in p['code_sha256'].items():
        if sha(REPO/path) != expected:
            raise ValueError('Resume with unchanged code only: '+path)
    return read(root/'data.json')


def original_config(c):
    return copy.deepcopy(read(Path(c['paths']['text_run'])/'resolved_config.json'))


def prior_for(samples, external):
    return TextPrior([w['text'] for s in samples for w in annotations(s)], external)


def neural_predictions(model, tokenizer, samples, c, root, tag):
    output = {s['sample_id']: [[] for _ in s['blocks']] for s in samples}
    eligible = [(s['sample_id'], i, b['text']) for s in samples for i,b in enumerate(s['blocks'])
                if b['text'].isalpha() and c['candidates']['minimum_characters'] <= len(b['text']) <= c['candidates']['maximum_characters']
                and len(('repair: '+b['text']).encode())+1 <= c['candidates']['max_input_bytes']]
    cfg = c['candidates']; model.eval(); shapes = set()
    for start in range(0, len(eligible), cfg['batch_size']):
        group = eligible[start:start+cfg['batch_size']]
        encoded = tokenizer(['repair: '+v for _,_,v in group], padding=True, return_tensors='pt')
        shapes.add(tuple(encoded['input_ids'].shape)); encoded = {k:v.cuda() for k,v in encoded.items()}
        with torch.inference_mode():
            greedy = model.generate(**encoded, do_sample=False, num_beams=1,
                                    max_new_tokens=cfg['max_new_tokens'], use_cache=True)
            beam = model.generate(**encoded, do_sample=False, num_beams=cfg['num_beams'],
                num_return_sequences=cfg['maximum_neural'], length_penalty=1., early_stopping=False,
                max_new_tokens=cfg['max_new_tokens'], use_cache=True)
        gtexts = tokenizer.batch_decode(greedy, skip_special_tokens=True)
        btexts = tokenizer.batch_decode(beam, skip_special_tokens=True)
        for j, (sid,i,_) in enumerate(group):
            texts = [gtexts[j]] if tokenizer.eos_token_id in greedy[j,1:].tolist() else []
            for k in range(cfg['maximum_neural']):
                pos = j*cfg['maximum_neural']+k
                if tokenizer.eos_token_id in beam[pos,1:].tolist():
                    texts.append(btexts[pos])
            output[sid][i] = list(dict.fromkeys(texts))
        if start % (cfg['batch_size']*40) == 0:
            log(root, 'BYT5_PROPOSALS', tag=tag, done=min(start+len(group),len(eligible)), total=len(eligible))
    return output, sorted(map(list, shapes))


def proposals(root, c, data, smoke):
    cfg = c['oof']; old = original_config(c); external = read(c['paths']['spell_vocabulary'])['labels']
    fold_map = read(root/'protocol.json')['folds']; outputs = {}; receipts = []
    for fold in range(cfg['folds']):
        train = [s for s in data['fit'] if fold_map[s['sample_id']] != fold]
        held = [s for s in data['fit'] if fold_map[s['sample_id']] == fold]
        if not train or not held:
            raise ValueError('Empty OOF training or prediction fold')
        old['seed'] = c['seed']+fold
        p = prior_for(train, external)
        # Regenerate operations with the fold-only prior, avoiding indirect FIT label leakage.
        train = copy.deepcopy(train)
        for s in train:
            s['operations'] = generate_operations(s['blocks'], p, old['structure'])
        pairs, counts = training_pairs(dict(fit=train), old)
        pairs = pairs[:96] if smoke else pairs[:cfg['max_training_pairs']]
        write(root/f'oof{fold}_pairs.json', dict(training_ids=[s['sample_id'] for s in train],
             prediction_ids=[s['sample_id'] for s in held], pairs=pairs, counts=counts))
        torch.manual_seed(c['seed']+fold)
        receipt = gpu_start(c, 'ByT5 OOF fold '+str(fold)); tick = time.monotonic()
        tokenizer, model, loading = load_byt5(old); model.cuda(); model.train(); model.config.use_cache=False
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                     weight_decay=cfg['weight_decay'], foreach=False)
        rng = random.Random(c['seed']+fold); history=[]; shapes=set()
        for epoch in range(cfg['epochs']):
            order=list(range(len(pairs))); rng.shuffle(order); losses=[]
            for start in range(0,len(order),cfg['batch_size']):
                rows=[pairs[i] for i in order[start:start+cfg['batch_size']]]
                encoded=tokenizer(['repair: '+p['source'] for p in rows],padding=True,return_tensors='pt')
                target=tokenizer([p['target'] for p in rows],padding=True,return_tensors='pt')
                shapes.add(tuple(encoded['input_ids'].shape))
                labels=target['input_ids'].masked_fill(target['attention_mask']==0,-100).cuda()
                encoded={k:v.cuda() for k,v in encoded.items()}; optimizer.zero_grad(set_to_none=True)
                loss=model(**encoded,labels=labels,use_cache=False).loss
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite OOF generator loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step(); losses.append(float(loss.detach()))
                if len(losses)%cfg['log_batches']==0:
                    log(root,'OOF_TRAIN',fold=fold,updates=len(losses),total=(len(order)+cfg['batch_size']-1)//cfg['batch_size'],loss=float(np.mean(losses)))
            history.append(dict(epoch=epoch+1,updates=len(losses),loss=float(np.mean(losses))))
        del optimizer; gc.collect(); torch.cuda.empty_cache(); model.config.use_cache=True
        checkpoint=root/f'oof{fold}.pt'
        torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                        fold=fold,training_ids=[s['sample_id'] for s in train]),checkpoint)
        prediction, infer_shapes=neural_predictions(model,tokenizer,held,c,root,'oof'+str(fold)); outputs.update(prediction)
        receipts.append(dict(**gpu_end(receipt,'ByT5-small full OOF fine-tuning/beam inference',cfg['batch_size'],
            sorted(map(list,shapes)),tick),fold=fold,loading=loading,history=history,
            inference_shapes=infer_shapes,training_images=len(train),prediction_images=len(held),
            checkpoint_sha256=sha(checkpoint)))
        write(root/f'oof{fold}_predictions.json',prediction); write(root/'generator_memory.json',receipts)
        del model; gc.collect(); torch.cuda.empty_cache()
    write(root/'neural_fit.json',outputs)
    receipt=gpu_start(c,'Frozen historical ByT5 proposals');tick=time.monotonic()
    tokenizer,model,loading=load_byt5(old)
    checkpoint=torch.load(Path(c['paths']['text_run'])/'denoiser_selected.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'],strict=True);del checkpoint
    model.cuda();model.eval();shapes=[]
    for phase in PHASES[1:]:
        prediction,seen=neural_predictions(model,tokenizer,data[phase],c,root,phase);shapes+=seen
        write(root/f'neural_{phase}.json',prediction)
    receipts.append(dict(**gpu_end(receipt,'Frozen historical ByT5-small greedy/beam4',c['candidates']['batch_size'],shapes,tick),loading=loading))
    write(root/'generator_memory.json',receipts);del model;gc.collect();torch.cuda.empty_cache()


def build_candidates(root,c,data):
    external=read(c['paths']['spell_vocabulary'])['labels']; fold_map=read(root/'protocol.json')['folds']
    priors={str(fold):prior_for([s for s in data['fit'] if fold_map[s['sample_id']]!=fold],external)
            for fold in range(c['oof']['folds'])}
    priors['final']=prior_for(data['fit'],external)
    with (root/'priors.pkl').open('wb') as stream:pickle.dump(priors,stream)
    for phase,samples in data.items():
        neural=read(root/f'neural_{phase}.json')
        for s in samples:
            prior=priors[str(fold_map[s['sample_id']])] if phase=='fit' else priors['final']
            s['candidates']=[candidates(b['text'],texts,prior,c['candidates']) for b,texts in zip(s['blocks'],neural[s['sample_id']])]
            s['contexts']=[neighbours(s['blocks'],i,c['context']) for i in range(len(s['blocks']))]
    write(root/'candidates.json',data)
    log(root,'CANDIDATES',counts={k:sum(len(v)-1 for s in ss for v in s['candidates']) for k,ss in data.items()})


def language(root,c,data):
    from transformers import AutoTokenizer,AutoModelForMaskedLM
    from recognition.bert_correction import BertWordScorer
    receipt=gpu_start(c,'Frozen BERT target scoring');tick=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(c['paths']['bert_snapshot'],local_files_only=True,use_fast=True)
    model,loading=AutoModelForMaskedLM.from_pretrained(c['paths']['bert_snapshot'],local_files_only=True,
        use_safetensors=True,torch_dtype=torch.float32,output_loading_info=True)
    if any(loading[k] for k in ('missing_keys','mismatched_keys','error_msgs')) or set(loading['unexpected_keys'])-set(c['context']['allowed_unused_keys']):
        raise ValueError('BERT loading audit failed: '+str(loading))
    model.cuda();scorer=BertWordScorer(tokenizer,model,device='cuda:0',batch_size=c['context']['bert_batch_size'],max_tokens=c['context']['max_tokens'])
    cache={};outputs={};counts=Counter()
    for phase,samples in data.items():
        outputs[phase]={}
        for n,s in enumerate(samples):
            rows=[]
            for i,options in enumerate(s['candidates']):
                if len(options)==1:rows.append(None);continue
                li,ri=s['contexts'][i];left=[s['blocks'][j]['text'] for j in li];right=[s['blocks'][j]['text'] for j in ri]
                values={}
                for kind,l,r in [('word',[],[]),('context',left,right)]:
                    result=[]
                    for candidate in options:
                        key=(candidate['text'],tuple(l),tuple(r))
                        if key not in cache:
                            cache[key]=scorer.score([candidate['text']],left=l,right=r,caption=None)[0]
                            counts['model_scored_candidates']+=1
                        result.append(cache[key])
                    if any(x['status']!='scored' for x in result):
                        raise ValueError('Language candidate could not be scored: '+str(result))
                    values[kind]=result
                rows.append(values)
                counts['with_context' if left or right else 'without_context']+=1
            outputs[phase][s['sample_id']]=rows
            if n%100==0:log(root,'BERT_SCORE',phase=phase,images=n+1,total=len(samples),unique=len(cache))
        write(root/f'language_{phase}.json',outputs[phase])
    write(root/'language_memory.json',dict(**gpu_end(receipt,'BERT-base uncased target-only PLL',c['context']['bert_batch_size'],
          dict(max_tokens_seen=scorer.max_input_tokens_seen),tick),loading=loading,counts=dict(counts)))
    del model;gc.collect();torch.cuda.empty_cache()


def visual(root,c,data):
    restorations={r['sample_id']:r for r in read(c['paths']['restoration_manifest'])['restored']}
    manifest=read(Path(c['paths']['source_run'])/'final_artifact_manifest.json');files={};output={}
    for phase,samples in data.items():
        output[phase]={}
        for s in samples:
            active=[i for i,v in enumerate(s['candidates']) if len(v)>1]
            profiles=[None for _ in s['blocks']]
            if active:
                relative='exports/'+phase+'/'+s['sample_id'].replace(':','__')+'.json'
                path=Path(c['paths']['source_run'])/relative
                if sha(path)!=manifest[relative]:raise ValueError('Original OCR export changed')
                record=read(path);image=Path(record['image_path'])
                if sha(image)!=s['image_sha256']:
                    if s['sample_id'] not in restorations:raise ValueError('Changed original RGB')
                    image=Path(restorations[s['sample_id']]['isolated_copy'])
                if sha(image)!=s['image_sha256']:raise ValueError('RGB restoration identity mismatch')
                files[str(image)]=s['image_sha256']
                with Image.open(image) as im:rgb=np.asarray(im.convert('RGB'))
                for i in active:
                    x1,y1,x2,y2=s['blocks'][i]['bbox']
                    profiles[i]=projection_profile(rgb[y1:y2,x1:x2])
            output[phase][s['sample_id']]=profiles
        write(root/f'visual_{phase}.json',output[phase])
    write(root/'visual_inputs.json',dict(rgb_files=files,alignment='proportional Unicode character position proxy; no new crops/OCR'))


def feature_rows(root,c,data):
    with (root/'priors.pkl').open('rb') as stream:priors=pickle.load(stream)
    external={s.casefold() for s in read(c['paths']['spell_vocabulary'])['labels']}
    fold_map=read(root/'protocol.json')['folds'];output={}
    for phase,samples in data.items():
        language_data=read(root/f'language_{phase}.json');visual_data=read(root/f'visual_{phase}.json');output[phase]={}
        for s in samples:
            p=priors[str(fold_map[s['sample_id']])] if phase=='fit' else priors['final'];rows=[]
            for i,options in enumerate(s['candidates']):
                variants=[]
                for j,option in enumerate(options[1:],1):
                    base=lexical_features(s['blocks'][i],option,p,external)
                    lm=language_data[s['sample_id']][i]
                    word=language_features(lm['word'][0],lm['word'][j]);ctx=language_features(lm['context'][0],lm['context'][j])
                    li,ri=s['contexts'][i]
                    context=ctx+[ctx[2]-word[2],len(li),len(ri)]
                    vis=visual_features(visual_data[s['sample_id']][i],option,c['visual']['relative_boundary_radius'])
                    variants.append(dict(lexical=base,word_lm=base+word,context=base+word+context,
                                         context_visual=base+word+context+vis))
                rows.append(variants)
            output[phase][s['sample_id']]=rows
        write(root/f'features_{phase}.json',output[phase])
    return output


def evaluate(samples,predictions):
    rows=[]
    for s in samples:
        sid=s['sample_id'];words=annotations(s);before=[b['text'] for b in s['blocks']]
        score=lexical_metrics(predictions[sid],words);change=lexical_changes(before,predictions[sid],words)
        rows.append(dict(sample_id=sid,**score,**change))
    total={k:sum(r[k] for r in rows) for k in ['correct','predictions','gt_words','recovered','damaged']}
    t,n,g=(total[k] for k in ['correct','predictions','gt_words'])
    total.update(fp=n-t,fn=g-t,precision=t/n if n else 0.,recall=t/g if g else 0.,f1=2*t/(n+g) if n+g else 0.)
    return dict(aggregate=total,per_image=rows)


def bootstrap(left,right,draws,seed):
    a=np.array([[r[k] for k in ['correct','predictions','gt_words']] for r in left['per_image']])
    b=np.array([[r[k] for k in ['correct','predictions','gt_words']] for r in right['per_image']])
    rng=np.random.default_rng(seed);changes=[]
    for _ in range(draws):
        ids=rng.integers(0,len(a),len(a));x=a[ids].sum(0);y=b[ids].sum(0)
        changes.append(2*y[0]/(y[1]+y[2])-2*x[0]/(x[1]+x[2]))
    return dict(delta_f1=right['aggregate']['f1']-left['aggregate']['f1'],
                conditional_95_interval=np.quantile(changes,[.025,.975]).tolist(),draws=draws)


def selectors(root,c,data,features):
    label_rows={};training=[]
    for s in data['fit']:
        words=annotations(s);rows=[]
        for i,options in enumerate(s['candidates']):
            values=[edit_label(s['blocks'],i,o['text'],words) for o in options[1:]]
            rows.append(values)
            training.extend((features['fit'][s['sample_id']][i][j],v) for j,v in enumerate(values))
        label_rows[s['sample_id']]=rows
    write(root/'fit_labels.json',label_rows)
    y=np.array([v['label'] for _,v in training]);log(root,'SELECTOR_LABELS',classes=dict(Counter(map(int,y))),candidates=len(y))
    if len(set(y))<2 or 1 not in y:raise ValueError('Insufficient training classes')
    predictions={};scores={};calibration={};models={};cfg=c['selector']
    for seed in cfg['seeds']:
        for arm in ARMS:
            name=arm+'_seed'+str(seed)
            x=np.asarray([f[arm] for f,_ in training])
            model=HistGradientBoostingClassifier(max_iter=cfg['iterations'],max_leaf_nodes=cfg['max_leaf_nodes'],
                min_samples_leaf=cfg['min_samples_leaf'],learning_rate=cfg['learning_rate'],
                l2_regularization=cfg['l2_regularization'],random_state=seed,early_stopping=cfg['early_stopping']).fit(x,y)
            models[name]=model;scores[name]={};predictions[name]={};calibration[name]={}
            for phase in PHASES[1:]:
                scores[name][phase]={}
                for s in data[phase]:
                    values=[]
                    for variants in features[phase][s['sample_id']]:
                        probs=model.predict_proba(np.array([v[arm] for v in variants])) if variants else np.zeros((0,len(model.classes_)))
                        values.append(class_utility(probs,model.classes_).tolist())
                    scores[name][phase][s['sample_id']]=values
            for cap in cfg['damage_limits']:
                curve=[]
                for threshold in cfg['thresholds']:
                    pred={s['sample_id']:select(s,scores[name]['calibration'][s['sample_id']],threshold)[0] for s in data['calibration']}
                    curve.append(dict(threshold=threshold,**evaluate(data['calibration'],pred)['aggregate']))
                valid=[r for r in curve if r['damaged']<=cap]
                choice=max(valid,key=lambda r:(r['correct'],r['f1'],r['threshold']))
                calibration[name][str(cap)]=dict(curve=curve,choice=choice)
            log(root,'SELECTOR_CALIBRATED',name=name,choices={k:v['choice'] for k,v in calibration[name].items()})
    # Save all choices before reading EVAL/TEST annotation transcripts for scoring.
    write(root/'selection_frozen.json',dict(calibration=calibration,fit_label_counts=dict(Counter(map(int,y))),
        evaluation_test_used_for_selection=False,feature_dimensions={a:len(training[0][0][a]) for a in ARMS}))
    write(root/'scores.json',scores)
    with (root/'selectors.pkl').open('wb') as stream:pickle.dump(models,stream)
    return scores,calibration


def report(root,c,data,scores,calibration):
    from recognition.text_reconstruction import structural_oracle
    output={};all_predictions={};all_changes={};oracle={}
    for phase in PHASES[1:]:
        pred={s['sample_id']:[b['text'] for b in s['blocks']] for s in data[phase]}
        arms={'baseline':pred};changes={}
        # Historical greedy: a fixed original-generator boundary-only control.
        historical=read(Path(c['paths']['text_run'])/f'denoiser_{phase}.json')
        historical_pred={}
        for s in data[phase]:
            by={g['parents'][0]:g for g in historical[s['sample_id']]['denoiser']}
            from recognition.contextual_boundary_selection import copied_segmentation
            historical_pred[s['sample_id']]=[copied_segmentation(b['text'],' '.join(by[b['id']]['texts']),c['candidates']['maximum_words']) or b['text'] for b in s['blocks']]
        arms['historical_boundary']=historical_pred
        for name,phase_scores in scores.items():
            for cap in c['selector']['damage_limits']:
                key=name+'_cap'+str(cap);threshold=calibration[name][str(cap)]['choice']['threshold'];arms[key]={};changes[key]={}
                for s in data[phase]:
                    chosen,trace=select(s,phase_scores[phase][s['sample_id']],threshold)
                    arms[key][s['sample_id']]=chosen;changes[key][s['sample_id']]=trace
            if name.endswith(str(c['selector']['seeds'][0])):
                key=name+'_raw';arms[key]={};changes[key]={}
                for s in data[phase]:
                    chosen,trace=select(s,phase_scores[phase][s['sample_id']],0.)
                    arms[key][s['sample_id']]=chosen;changes[key][s['sample_id']]=trace
        output[phase]={name:evaluate(data[phase],p) for name,p in arms.items()}
        all_predictions[phase]=arms;all_changes[phase]=changes;oracle[phase]=[]
        for s in data[phase]:
            ops=[dict(id=f'{i}:{j}',parents=[b['id']],texts=o['text'].split(),op='split')
                 for i,(b,options) in enumerate(zip(s['blocks'],s['candidates'])) for j,o in enumerate(options[1:],1)]
            result=structural_oracle(s['blocks'],ops,annotations(s))
            oracle[phase].append(dict(sample_id=s['sample_id'],**result))
        log(root,'REPORT',phase=phase,baseline=output[phase]['baseline']['aggregate'])
    comparisons={}
    for phase in ('evaluation','test'):
        comparisons[phase]={name:bootstrap(output[phase]['baseline'],v,c['bootstrap_draws'],c['seed'])
                            for name,v in output[phase].items() if name!='baseline'}
        seed=c['selector']['seeds'][0]
        for cap in c['selector']['damage_limits']:
            for a,b in [('lexical','word_lm'),('word_lm','context'),('context','context_visual')]:
                na=f'{a}_seed{seed}_cap{cap}';nb=f'{b}_seed{seed}_cap{cap}'
                comparisons[phase][nb+'_versus_'+na]=bootstrap(output[phase][na],output[phase][nb],c['bootstrap_draws'],c['seed'])
    write(root/'predictions.json',all_predictions);write(root/'changes.json',all_changes);write(root/'oracle.json',oracle)
    result=dict(status='SUCCESS',claim_label='approximate reproduction',metric='micro per-image casefold whitespace-token multiset; punctuation retained; all care GT and empty predictions; no IoU or ignore-neutralization',
        protocol_sha256=sha(root/'protocol.json'),selection_sha256=sha(root/'selection_frozen.json'),
        results=output,comparisons=comparisons,oracle={k:sum(r['correct'] for r in v) for k,v in oracle.items()},
        limitations=['Historical explored cohorts, two TRAIN/TEST RGB duplicates and source calibration exposure.',
            'Three selector seeds reuse the same OOF generators and data: not full training variance.',
            'Conditional image bootstrap omits historical search and generator variability.',
            'RGB projection uses proportional positions, not true character alignment.',
            'OOF generator has fewer training images than the unchanged historical final generator.',
            'No official spotting/strict-paper claim; no new crops or OCR.'])
    write(root/'results.json',result)
    lines=['# Contextual boundary selection results','','Approximate reproduction / exploratory lexical transcription study.','',
           '| TEST arm | Correct | Outputs | Recovered | Damaged | F1 |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for name,value in output['test'].items():
        r=value['aggregate'];lines.append(f"| {name} | {r['correct']} | {r['predictions']} | {r['recovered']} | {r['damaged']} | {100*r['f1']:.4f}% |")
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--stage',choices=['all','prepare','proposals','features','selectors','report'],default='all');parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    try:
        c=checked_config(args.config,args.mode)
        if args.stage in ('all','prepare'):
            if (root/'protocol.json').exists():raise ValueError('Use a new output for a fresh experiment')
            prepare(root,c,args.smoke)
        else:setup(c)
        data=load_prepared(root,c,args.smoke)
        if args.stage in ('all','proposals'):proposals(root,c,data,args.smoke);build_candidates(root,c,data)
        if args.stage in ('all','features'):
            data=read(root/'candidates.json');language(root,c,data);visual(root,c,data);feature_rows(root,c,data)
        if args.stage in ('all','selectors'):
            data=read(root/'candidates.json');features={p:read(root/f'features_{p}.json') for p in PHASES}
            scores,calibration=selectors(root,c,data,features)
        if args.stage in ('all','report'):
            data=read(root/'candidates.json');scores=read(root/'scores.json');calibration=read(root/'selection_frozen.json')['calibration']
            report(root,c,data,scores,calibration)
            write(root/'completion.json',dict(status='SUCCESS',results_sha256=sha(root/'results.json')))
        log(root,'STAGE_COMPLETE',completed=args.stage)
    except Exception as exc:
        write(root/'failure.json',dict(status='FAILED',exception=type(exc).__name__,message=str(exc),oom=isinstance(exc,torch.cuda.OutOfMemoryError),
            allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
            reserved=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
