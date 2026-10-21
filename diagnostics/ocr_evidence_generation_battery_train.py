"""Frozen exploratory TRAIN generation battery. CPU prepare/evaluate; manual GPU only."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import statistics
import time

from diagnostics.caption_routing_train import checked, seal_write
from diagnostics.contextual_ocr_correction import verify, word_scores, lexical_counts, runtime_info
from diagnostics.geometry_context_reconstruction_train import load_run as load_source
from diagnostics.lavis_captioner_candidates import digest, sha, write_json


def read(path):
    return json.loads(Path(path).read_text())


def variants(cfg):
    out=[]
    for x in cfg['variants']:
        v={**cfg['base'],**(cfg['rich'] if x.get('template')=='rich' else {}),
           **{k:val for k,val in x.items() if k!='template'}}
        if v['beam'] not in (0,1,3,5,10,20) or v['neighbors'] not in (0,2,4,'all'):
            raise ValueError('Unregistered evidence size')
        out.append(v)
    if len(out)!=40 or len({x['id'] for x in out})!=40: raise ValueError('40 unique frozen conditions required')
    return out


def neighbor_indices(blocks, target, n, metric):
    if metric=='reading_order': cost=lambda i:abs(i-target)
    elif metric=='bbox_center':
        b=blocks[target]['bbox']; x=(b[0]+b[2])/2; y=(b[1]+b[3])/2
        cost=lambda i: ((blocks[i]['bbox'][0]+blocks[i]['bbox'][2])/2-x)**2+((blocks[i]['bbox'][1]+blocks[i]['bbox'][3])/2-y)**2
    else: raise ValueError('Unknown neighbor metric')
    order=sorted((i for i in range(len(blocks)) if i!=target),key=lambda i:(cost(i),i))
    return sorted(order if n=='all' else order[:n])


def caption_view(text, v, ocr, tokenizer):
    if v['caption']=='none': return '',0,0
    original=len(tokenizer.encode(text,add_special_tokens=False))
    if v['caption_mode']=='quoted': text='; '.join(re.findall(r'["“]([^"”]+)["”]',text))
    elif v['caption_mode']=='relevant':
        words=set(re.findall(r'\w+',ocr.casefold())); sentences=re.split(r'(?<=[.!?])\s+',text)
        scores=[len(words & set(re.findall(r'\w+',s.casefold()))) for s in sentences]
        selected=sorted(sorted((i for i,s in enumerate(scores) if s>0),key=lambda i:(-scores[i],i))[:2])
        text=' '.join(sentences[i] for i in selected)
    elif v['caption_mode'] not in ('prefix','duplicate'): raise ValueError('Unknown caption view')
    ids=tokenizer.encode(text,add_special_tokens=False); limit=v['caption_tokens']
    if limit is not None and len(ids)>limit: text=tokenizer.decode(ids[:limit],skip_special_tokens=True)
    if v['caption_mode']=='duplicate' and text: text=text+'\n'+text
    return text,original,len(tokenizer.encode(text,add_special_tokens=False))


def observation(b,v,cfg,include_beam):
    out={'id':b['candidate_id'],'bbox':b['bbox'],'ocr':b['text']}
    digits=cfg['probability_digits']; r=lambda x:round(x,digits) if x is not None else None
    if v['confidence']!='none':
        out['refined_confidence']={k:r(b['confidence'][k]) for k in
                                  ('sequence_confidence_native','mean_token_log_probability','minimum_token_probability')}
    if v['confidence']=='characters':
        out['character_probabilities']=[[c,r(p)] for c,p in zip(b['text'],b['confidence']['token_probabilities'])]
    if v['mask_hint']:
        out['uncertain_character_hint']=''.join(c if p>=cfg['masked_character_probability'] else '?'
            for c,p in zip(b['text'],b['confidence']['token_probabilities']))
    if include_beam and v['beam']:
        # Do not refill ranks after dropping empty hypotheses or substitute new strings.
        ar=b['ar_candidates']; selected=[(i,x) for i,x in enumerate(ar[:v['beam']],1) if x['text'].strip()]
        if v['reverse_beam']: selected=list(reversed(selected))
        out['ar_hypotheses']=[]
        for rank,x in selected:
            h={'rank':rank,'text':x['text']}
            if v['beam_scores']: h.update(logprob=r(x['logprob']),delta_to_ar_best=r(x['logprob']-ar[0]['logprob']))
            out['ar_hypotheses'].append(h)
        if not ar: out['ar_cache_status']='unavailable; retain V1 only'
    return out


def choices_for(target):
    strings=[target['ocr']]; seen={strings[0].casefold()}
    for h in target.get('ar_hypotheses',[]):
        if h['text'].casefold() not in seen: strings.append(h['text']);seen.add(h['text'].casefold())
    return strings


def make_request(sample, target_index, v, cfg, tokenizer, other_caption=None):
    blocks=sample['blocks']; b=blocks[target_index]
    neighbor_ids=neighbor_indices(blocks,target_index,v['neighbors'],v['neighbor_metric'])
    target=observation(b,v,cfg,True)
    neighbors=[observation(blocks[i],v,cfg,v['beam_scope']=='all') for i in neighbor_ids]
    source_caption=other_caption if v['negative_caption'] else sample['captions'].get(v['caption'],'')
    caption,full_n,used_n=caption_view(source_caption or '',v,' '.join([b['text']]+[blocks[i]['text'] for i in neighbor_ids]),tokenizer)
    payload={}
    if v['layout']=='caption_first' and caption: payload['fallible_scene_context']=caption
    payload.update(target=target,other_separate_crops_in_reading_order=neighbors)
    if v['layout']=='caption_last' and caption: payload['fallible_scene_context']=caption
    choices=choices_for(target)
    formats={'plain':'Return ONLY one word or code, no whitespace or explanation. Return <KEEP> to preserve TARGET.',
             'json':'Return ONLY a JSON object with exactly one key: {"text":"word"}. The value must be one word/code or <KEEP>. No explanation.',
             'edit':'Return ONLY {"operation":"KEEP"} or {"operation":"REPLACE","text":"word"}. Replacement must be one word/code, no explanation.',
             'choice':'Return ONLY {"index":N} selecting one of the numbered allowed readings below. Index0 preserves TARGET; invent no new string.'}
    if v['format']=='choice':payload['allowed_readings']=dict(enumerate(choices))
    system=cfg['instruction']+'\n'+formats[v['format']]
    messages=[{'role':'system','content':system}]
    if v['fewshot']:
        for ex in cfg['synthetic_examples']:
            messages.extend([{'role':'user','content':json.dumps({'target':{'ocr':ex['target']},'fallible_scene_context':ex['context']})},
                             {'role':'assistant','content':ex['answer']}])
    messages.append({'role':'user','content':json.dumps(payload,ensure_ascii=False,separators=(',',':'))})
    return {'sample_id':sample['sample_id'],'candidate_id':b['candidate_id'],'original':b['text'],
            'format':v['format'],'choices':choices if v['format']=='choice' else [],'messages':messages,
            'evidence':{'caption_original_tokens':full_n,'caption_used_tokens':used_n,'neighbors':len(neighbors),
                        'beam_target_available':bool(b['ar_candidates']),
                        'ar_hypotheses_shown':sum(len(x.get('ar_hypotheses',[])) for x in [target]+neighbors)}}


def parse_answer(raw, original, fmt, choices=(), ended=True):
    reason=None; answer=raw.strip()
    if not original.strip(): reason='empty_original_keep'
    elif not ended: reason='truncated'
    else:
        if fmt!='plain':
            try: obj=json.loads(answer)
            except (json.JSONDecodeError,TypeError): obj=None
            if not isinstance(obj,dict): reason='invalid_schema'
            elif fmt=='json' and set(obj)=={'text'} and isinstance(obj['text'],str): answer=obj['text']
            elif fmt=='edit' and obj=={'operation':'KEEP'}: answer='<KEEP>'
            elif fmt=='edit' and set(obj)=={'operation','text'} and obj['operation']=='REPLACE' and isinstance(obj['text'],str):answer=obj['text']
            elif fmt=='choice' and set(obj)=={'index'} and type(obj['index']) is int and 0<=obj['index']<len(choices):answer=choices[obj['index']]
            else:reason='invalid_schema'
        if reason is None:
            if answer=='<KEEP>': reason='explicit_keep'
            elif not answer or any(c.isspace() for c in answer): reason='empty_or_multiword'
            elif answer.startswith(('```','{','[')) or answer in ('<ABSTAIN>',):reason='invalid_response'
    return {'status':'keep_fallback' if reason else 'valid','reason':reason,'text':original if reason else answer}


def validate_confidence(row,text):
    if row['raw_text']!=text or row['predicted_length']!=len(text):raise RuntimeError('Frozen OCR mismatch')
    probs=row['token_probabilities']; logs=row['token_log_probabilities']
    if len(probs)!=len(text) or len(logs)!=len(text):raise RuntimeError('Character alignment mismatch')
    if any(not math.isfinite(p) or not 0<=p<=1 for p in probs):raise RuntimeError('Invalid probability')
    if any(not math.isfinite(p) for p in logs):raise RuntimeError('Invalid logprob')
    if not 0<=row['sequence_confidence_native']<=1:raise RuntimeError('Invalid sequence confidence')
    if text:
        if not math.isclose(row['mean_token_log_probability'],sum(logs)/len(logs),abs_tol=1e-6):raise RuntimeError('Mean confidence drift')
        if row['minimum_token_probability']!=min(probs):raise RuntimeError('Min confidence drift')
    return {k:row[k] for k in ('sequence_confidence_native','mean_token_log_probability','minimum_token_probability','token_probabilities')}


def prepare(a):
    cfg=read(a.config)
    if cfg['mode']!='APPROXIMATE' or cfg['scope']!={'split':'TRAIN','images':229,'gt_words':849,'crops':1203} or cfg['training'] or cfg['test_loaded']:
        raise RuntimeError('TRAIN229 only')
    vs=variants(cfg); verify(a.source_root/'results.json',cfg['source_report_sha256'])
    old,samples,old_contract=load_source(a.source_root)
    audit_path=a.audit; verify(audit_path,old['config']['audit_sha256']); audit=read(audit_path)['per_word']
    verify(a.coverage,cfg['coverage_sha256']); coverage=read(a.coverage)
    if coverage['scope']['split']!='TRAIN' or coverage['scope']['test_loaded']:raise RuntimeError('Beam source split mismatch')
    beams={};beam_v1={}
    for row in coverage['details']:
        k=(str(row['sample_id']),row['candidate_id']); ar=row['ar_candidates']
        if k in beams and (beams[k]!=ar or beam_v1[k]!=row['v1_text']):raise RuntimeError('Shared beam mismatch')
        if any(not math.isfinite(x['logprob']) for x in ar):raise RuntimeError('Invalid AR score')
        if any(x['logprob']<y['logprob'] for x,y in zip(ar,ar[1:])):raise RuntimeError('Unsorted beam')
        beams[k]=ar;beam_v1[k]=row['v1_text']
    verify(a.compact_root/'manifest.json',cfg['compact_manifest_sha256']); compact=read(a.compact_root/'manifest.json')
    sources={str(f.resolve()):sha(f) for f in (a.config,a.source_root/'results.json',a.coverage,audit_path,a.compact_root/'manifest.json')}
    manifest_by={str(x['sample_id']):x for x in compact['samples']}
    for s in samples:
        sid=s['sample_id']; rec=manifest_by[sid]
        cf=a.compact_root/rec['candidate_shard']; of=a.compact_root/rec['ocr_shard']
        verify(cf,rec['candidate_shard_sha256']);verify(of,rec['ocr_shard_sha256'])
        c=read(cf);o=read(of)
        if c['sample_id']!=sid or o['sample_id']!=sid or c['image_sha256']!=s['image_sha256']:raise RuntimeError('Image association drift')
        if o['parseq_checkpoint_sha256']!=compact['parseq']['checkpoint_sha256']:raise RuntimeError('Checkpoint drift')
        boxes={x['candidate_id']:x for x in c['candidates']};ocr={x['candidate_id']:x for x in o['results']}
        if set(boxes)!=set(ocr) or set(boxes)!={b['candidate_id'] for b in s['blocks']}:raise RuntimeError('Crop population drift')
        for b in s['blocks']:
            cid=b['candidate_id'];b['id']=cid
            if boxes[cid]['bbox']!=b['bbox']:raise RuntimeError('Bbox drift')
            b['confidence']=validate_confidence(ocr[cid],b['text'])
            if (sid,cid) in beam_v1 and beam_v1[(sid,cid)]!=b['text']:raise RuntimeError('AR/V1 text mismatch')
            b['ar_candidates']=beams.get((sid,cid),[])
        sources[str(cf.resolve())]=sha(cf);sources[str(of.resolve())]=sha(of)
    if len(beams)!=700 or sum(bool(b['ar_candidates']) for s in samples for b in s['blocks'])!=700:raise RuntimeError('Beam coverage drift')
    from transformers import AutoTokenizer
    snapshot=a.hf_cache/('models--'+old['model']['model_id'].replace('/','--'))/'snapshots'/old['model']['revision']
    meta=old['model_metadata'];verify(meta['path'],meta['sha256']); model_source=checked(Path(meta['path']),old['model_source_contract'])
    for name,h in model_source['snapshot_files_sha256'].items():
        if 'token' in name or name.endswith('config.json') or name.endswith('merges.txt') or name.endswith('vocab.json'):verify(snapshot/name,h)
    tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
    prompts={};requests=[];stats={};by_gt=defaultdict(list)
    for w in audit:by_gt[w['sample_id']].append(w)
    baseline_exact=baseline_lex=0
    for si,s in enumerate(samples):
        baseline_exact+=sum(w['exact'] for w in word_scores(s,by_gt[s['sample_id']],{b['id']:b['text'] for b in s['blocks']}))
        baseline_lex+=lexical_counts([b['text'] for b in s['blocks']],[w['gt_text'] for w in by_gt[s['sample_id']]])['matches']
        for v in vs:
            for ti,b in enumerate(s['blocks']):
                req=make_request(s,ti,v,cfg,tokenizer,samples[(si+1)%len(samples)]['captions'].get(v['caption'],''))
                # Content-addressed inference identity excludes variant name and diagnostics.
                identity={k:req[k] for k in ('sample_id','candidate_id','original','format','choices','messages')}
                h=digest(identity)
                if h not in prompts:
                    n=len(tokenizer.apply_chat_template(req['messages'],add_generation_prompt=True))
                    if n>cfg['max_input_tokens']:raise RuntimeError(f'No silent truncation: {v["id"]} {b["id"]} requires{n} input tokens')
                    prompts[h]={**identity,'input_tokens':n}
                requests.append({'variant':v['id'],'sample_id':s['sample_id'],'candidate_id':b['id'],'prompt_sha256':h,'evidence':req['evidence']})
        if (si+1)%50==0:print(f'[prepare CPU] {si+1}/229 images; {len(prompts)} distinct requests',flush=True)
    if (baseline_exact,baseline_lex)!=(502,503) or len(audit)!=849:raise RuntimeError('Baseline/control drift')
    for v in vs:
        rr=[r for r in requests if r['variant']==v['id']]; ns=[prompts[r['prompt_sha256']]['input_tokens'] for r in rr]
        stats[v['id']]={'requests':len(rr),'input_tokens_min':min(ns),'input_tokens_median':statistics.median(ns),'input_tokens_max':max(ns),
                       'caption_tokens_median':statistics.median(r['evidence']['caption_used_tokens'] for r in rr),
                       'ar_hypotheses_median':statistics.median(r['evidence']['ar_hypotheses_shown'] for r in rr)}
    deps=[__file__,str(a.config),'diagnostics/run_ocr_evidence_battery_host.sh','diagnostics/contextual_ocr_correction.py',
          'diagnostics/geometry_context_reconstruction_train.py','diagnostics/caption_routing_train.py','diagnostics/lavis_captioner_candidates.py']
    p={'config':cfg,'resolved_variants':vs,'source_sha256':sources,'source_contract':old_contract,
       'code_sha256':{str(Path(x).resolve()):sha(x) for x in deps},'model':old['model'],'model_metadata':meta,
       'model_source_contract':old['model_source_contract'],'samples_sha256':digest(samples),'labels_sha256':digest(audit),
       'prompts_sha256':digest(prompts),'requests_sha256':digest(requests),'request_stats':stats,
       'logical_requests':len(requests),'unique_prompts':len(prompts),'baseline_spatial_exact':502,'baseline_lexical_matches':503,
       'gt_in_prompts':False,'table5_accuracy':None}
    contract=digest(p)
    seal_write(a.output_dir/'protocol.json',{**p,'contract_sha256':contract})
    for filename,field,value in [('inputs.json','samples',samples),('evaluation-labels.json','labels',audit),
                                 ('requests.json','requests',requests),('prompts.json','prompts',prompts)]:
        seal_write(a.output_dir/filename,{'contract_sha256':contract,field:value})
    examples={v['id']:prompts[next(r['prompt_sha256'] for r in requests if r['variant']==v['id'])]['messages'] for v in vs}
    seal_write(a.output_dir/'prompt-examples.json',{'contract_sha256':contract,'examples':examples})
    print(f'[prepared] 40 variants /{len(requests)} logical requests /{len(prompts)} unique GPU prompts; baseline502 spatial/503 lexical',flush=True)


def load(root):
    p=checked(root/'protocol.json');c=p['contract_sha256']
    if digest({k:v for k,v in p.items() if k not in ('contract_sha256','payload_sha256')})!=c:raise RuntimeError('Protocol drift')
    for f,h in p['code_sha256'].items():verify(f,h)
    data={}
    for file,field in [('inputs.json','samples'),('evaluation-labels.json','labels'),('requests.json','requests'),('prompts.json','prompts')]:
        data[field]=checked(root/file,c)[field]
        if digest(data[field])!=p[field+'_sha256']:raise RuntimeError('Prepared data drift')
    return p,data,c


def generation_path(root,h):return root/'generations'/h[:2]/(h+'.json')


def check_generation(root,h,q,c):
    g=checked(generation_path(root,h),c)
    if g['prompt_sha256']!=h or digest({k:q[k] for k in ('sample_id','candidate_id','original','format','choices','messages')})!=h:
        raise RuntimeError('Cached request drift')
    parsed=parse_answer(g['raw_output'],q['original'],q['format'],q['choices'],g['ended'])
    if any(g[k]!=v for k,v in parsed.items()):raise RuntimeError('Cached parser drift')
    if g['model_metadata_sha256'] is not None:verify(root/'model.json',g['model_metadata_sha256'])
    elif q['original'].strip():raise RuntimeError('Nonempty unscored request')
    return g


def infer(a):
    p,d,c=load(a.output_dir);cfg=p['config'];pending=[]
    for h,q in d['prompts'].items():
        if generation_path(a.output_dir,h).exists():check_generation(a.output_dir,h,q,c)
        else:pending.append((h,q))
    print(f'[resume] {len(d["prompts"])-len(pending)} verified; {len(pending)} pending',flush=True)
    if not pending:return
    spec=p['model'];meta=p['model_metadata'];verify(meta['path'],meta['sha256']);old=checked(Path(meta['path']),p['model_source_contract'])
    snapshot=a.hf_cache/('models--'+spec['model_id'].replace('/','--'))/'snapshots'/spec['revision']
    for name,h in old['snapshot_files_sha256'].items():verify(snapshot/name,h)
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig
    device=torch.device(a.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Explicit manual PAPER GPU stage only')
    torch.cuda.set_device(device);torch.cuda.init()
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Qualified BF16 required')
    torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False
    by_label=defaultdict(list)
    for w in d['labels']:by_label[w['sample_id']].append(w)
    by_sample={s['sample_id']:s for s in d['samples']};aliases=defaultdict(list)
    for r in d['requests']:aliases[r['prompt_sha256']].append(r['variant'])
    model=None
    try:
        print('[GPU load] Qwen only; no PARSeq/BLIP/segmentation inference',flush=True)
        torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
        tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        model,info=AutoModelForCausalLM.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,
            use_safetensors=True,torch_dtype=torch.bfloat16,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):raise RuntimeError(str(info))
        if cfg['max_input_tokens']+cfg['generation']['max_new_tokens']>model.config.max_position_embeddings:raise RuntimeError('Context budget exceeds model')
        model.to(device).eval();torch.cuda.synchronize(device)
        metadata={'contract_sha256':c,'spec':spec,'runtime':runtime_info(torch,device),'batch_size':1,'quantization':None,
                  'loading_info':info,'parameters':sum(x.numel() for x in model.parameters()),'snapshot_files_sha256':old['snapshot_files_sha256']}
        mp=a.output_dir/'model.json'
        if mp.exists():
            prev=checked(mp,c)
            if any(prev[k]!=v for k,v in metadata.items()):raise RuntimeError('Runtime drift on resume')
        else:seal_write(mp,{**metadata,'load_seconds':time.perf_counter()-start,'load_peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                           'load_peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
        eos=model.generation_config.eos_token_id;eos_ids=eos if isinstance(eos,list) else [eos]
        gen=GenerationConfig(**cfg['generation'],eos_token_id=eos,pad_token_id=tokenizer.pad_token_id)
        for ni,(h,q) in enumerate(pending,len(d['prompts'])-len(pending)+1):
            encoded=tokenizer.apply_chat_template(q['messages'],add_generation_prompt=True,return_tensors='pt')
            if encoded.shape[1]!=q['input_tokens']:raise RuntimeError('Tokenizer drift')
            torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter()
            ids=[];raw='';ended=True
            if q['original'].strip():
                with torch.inference_mode():
                    ids=model.generate(input_ids=encoded.to(device),attention_mask=torch.ones_like(encoded).to(device),
                                       generation_config=gen)[0,encoded.shape[1]:].cpu().tolist()
                raw=tokenizer.decode(ids,skip_special_tokens=True);ended=bool(ids) and ids[-1] in eos_ids
            torch.cuda.synchronize(device)
            parsed=parse_answer(raw,q['original'],q['format'],q['choices'],ended)
            seal_write(generation_path(a.output_dir,h),{'contract_sha256':c,'prompt_sha256':h,'raw_output':raw,'ended':ended,**parsed,
                'model_metadata_sha256':sha(mp),'input_tokens':q['input_tokens'],'output_tokens':len(ids),'seconds':time.perf_counter()-start,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)})
            b=next(b for b in by_sample[q['sample_id']]['blocks'] if b['id']==q['candidate_id'])
            gt=[w['gt_text'] for w in by_label[q['sample_id']] if w['available'] and w['crop_bbox']==b['bbox']]
            print(f'[{ni}/{len(d["prompts"])}] {q["candidate_id"]} {",".join(aliases[h])} {parsed["status"]} OCR={q["original"]!r} OUT={parsed["text"]!r} GT(eval only)={gt}',flush=True)
    except Exception as exc:
        write_json(a.output_dir/'last-error.json',{'type':type(exc).__name__,'message':str(exc),'oom':isinstance(exc,torch.cuda.OutOfMemoryError)});raise
    finally:
        if model is not None:del model
        torch.cuda.empty_cache()


def evaluate(a):
    p,d,c=load(a.output_dir);by_labels=defaultdict(list)
    for w in d['labels']:by_labels[w['sample_id']].append(w)
    generations={};hashes={}
    if not a.baseline_only:
        for h,q in d['prompts'].items():
            g=check_generation(a.output_dir,h,q,c);generations[h]=g;hashes[h]=sha(generation_path(a.output_dir,h))
    summary={};details={};image_metrics={}
    vlist=['V1']+([] if a.baseline_only else [v['id'] for v in p['resolved_variants']])
    for v in vlist:
        requests=[r for r in d['requests'] if r['variant']==v];lookup={r['candidate_id']:r for r in requests}
        words=[];ims=[];statuses=Counter();changed=0;timing=[];tokens=[];novel=0;strata=defaultdict(lambda:Counter())
        for s in d['samples']:
            corrected={}
            for b in s['blocks']:
                if v=='V1':text=b['text']
                else:
                    r=lookup[b['id']];g=generations[r['prompt_sha256']];text=g['text'];statuses[g['reason'] or 'valid']+=1
                    timing.append(g['seconds']);tokens.append(g['input_tokens'])
                corrected[b['id']]=text;changed+=text!=b['text']
                novel+=text.casefold() not in {b['text'].casefold(),*(x['text'].casefold() for x in b['ar_candidates'][:20])}
            ws=word_scores(s,by_labels[s['sample_id']],corrected);words.extend(ws)
            blocks={b['id']:b for b in s['blocks']}
            for w in ws:
                has=bool(blocks[w['candidate_id']]['ar_candidates']) if w['candidate_id'] else False
                st=strata['AR_available' if has else 'AR_unavailable'];st['gt_rows']+=1;st['exact']+=w['exact'];st['recovered']+=w['corrected_error'];st['damaged']+=w['damaged_correct']
            ims.append({'sample_id':s['sample_id'],**lexical_counts(list(corrected.values()),[w['gt_text'] for w in by_labels[s['sample_id']]])})
        exact=sum(w['exact'] for w in words);rec=sum(w['corrected_error'] for w in words);dam=sum(w['damaged_correct'] for w in words)
        if len(words)!=849 or exact!=502+rec-dam:raise RuntimeError('Spatial accounting mismatch')
        matches=sum(x['matches'] for x in ims);pred=sum(x['predicted'] for x in ims)
        summary[v]={'gt_words':849,'eligible_gt_words':745,'spatial_exact':exact,'exact_rate_all849':exact/849,
            'fuzzy_ge_0_8':sum(w['fuzzy']>=.8 for w in words),'recovered':rec,'damaged':dam,'net':rec-dam,
            'changed_crops':changed,'lexical_matches':matches,'lexical_recall':matches/849,'lexical_precision':matches/pred if pred else 0,
            'lexical_f1':2*matches/(849+pred),'predicted_tokens':pred,'outputs_outside_V1_and_cached_AR20':novel,
            'statuses':dict(statuses),'median_input_tokens':statistics.median(tokens) if tokens else 0,
            'median_seconds':statistics.median(timing) if timing else 0,'logical_seconds_including_shared_requests':sum(timing),
            'beam_availability_strata':dict(strata)}
        details[v]=words;image_metrics[v]=ims
    if summary['V1']['spatial_exact']!=502 or summary['V1']['lexical_matches']!=503:raise RuntimeError('Baseline changed')
    controls=[v['id'] for v in p['resolved_variants'] if v['negative_caption'] or v['caption_mode']=='duplicate' or v['reverse_beam']]
    report={'contract_sha256':c,'mode':'APPROXIMATE','classification':'PRACTICAL_APPROXIMATION','claim_label':p['config']['claim_label'],
            'scope':p['config']['scope'],'summary':summary,'per_word':details,'per_image_lexical':image_metrics,
            'negative_or_stress_controls':controls,'generation_sha256':hashes,'pipeline_promoted':False,'table5_accuracy':None,
            'actual_unique_inference_seconds':sum(g['seconds'] for g in generations.values()),
            'peak_allocated_bytes':max((g['peak_allocated_bytes'] for g in generations.values()),default=0),
            'peak_reserved_bytes':max((g['peak_reserved_bytes'] for g in generations.values()),default=0),
            'warning':'Exploratory TRAIN selection, not independent validation. Same849 frozen Task2.2-derived care GT, no reannotation. Shared requests are correlated. Beam available only for700 previously GT-associated crops.'}
    path=a.output_dir/('baseline.json' if a.baseline_only else 'results.json');seal_write(path,report)
    print('VARIANT exact recovered damaged changed lexical')
    for v,r in summary.items():print(v,r['spatial_exact'],r['recovered'],r['damaged'],r['changed_crops'],r['lexical_matches'])
    print('JSON',path,'SHA256',sha(path))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ss=ap.add_subparsers(dest='stage',required=True)
    prep=ss.add_parser('prepare')
    for name in ('config','source-root','compact-root','coverage','audit','hf-cache'):prep.add_argument('--'+name,type=Path,required=True)
    inf=ss.add_parser('infer');inf.add_argument('--hf-cache',type=Path,required=True);inf.add_argument('--device',required=True)
    ev=ss.add_parser('evaluate');ev.add_argument('--baseline-only',action='store_true')
    for x in (prep,inf,ev):x.add_argument('--output-dir',type=Path,required=True)
    a=ap.parse_args();{'prepare':prepare,'infer':infer,'evaluate':evaluate}[a.stage](a)


if __name__=='__main__':main()
