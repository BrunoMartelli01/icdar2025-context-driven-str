"""GPU alignment and spatially grounded character consensus, APPROXIMATE only."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.nn import functional as F

from localize.occlusion_four_tests import iou
from localize.word_graph_recovery import HypothesisScorer, hypothesis_adjacency, conflicts
from localize.word_evidence_graphs import consensus_reading
from recognition.word_evidence_alignment import align_batch
from scripts.run_mask_crop_budget import CropOCR
from scripts.run_word_evidence_graphs import (load,log,rgb_for,state_digest,matching_count,
    compatible_add,save_arm)
from scripts.run_selective_word_repair_tests import gpu_start,gpu_end,report
from scripts.run_text_reconstruction_tests import read,write,sha
from scripts.evaluate_cached_totaltext_padding import crop_hash


def neighbors(sample,options,overlap):
    """Observable IoU-supported pairs; no GT selection or string-length filter."""
    result={}
    for p in sample['pool']:
        if p['id'] in {q['id'] for q in sample['baseline_gnn']} or any(conflicts(p,q,overlap) for q in sample['baseline_gnn']):continue
        donor=[q for q in sample['pool'] if q['id']!=p['id'] and q['text'] and
               iou(p['bbox'],q['bbox'])>=options['minimum_pair_iou']]
        if donor:result[p['id']]=donor
    return result


def adjacency_for(pool,c):
    # Only transient graph keys change: physical RGB hashes remain untouched.
    transient=[dict(p,crop_array_sha256=p['id']) for p in pool]
    return hypothesis_adjacency(transient,{p['id']:dict(raw_text=p['text']) for p in pool},
        neighbors=c['relations']['neighbors'],maximum_distance=c['relations']['maximum_distance'])


def character_grams(text,bins):
    vector=np.zeros(bins,np.float32);text='^'+text.casefold()+'$'
    for n in (1,2,3):
        for j in range(len(text)-n+1):
            k=int.from_bytes(hashlib.sha256(text[j:j+n].encode()).digest()[:8],'big')%bins
            vector[k]+=1
    return vector/max(1.,float(np.linalg.norm(vector)))


def alignment_stage(root,c,smoke):
    data=load(root);options=c['consensus'];needed={};pairs={}
    for phase in data:
        pairs[phase]={}
        for s in data[phase]:
            related=neighbors(s,options,c['compatibility_iou']);pairs[phase][s['sample_id']]={k:[q['id'] for q in v] for k,v in related.items()}
            ids=set(related)|{q['id'] for v in related.values() for q in v}
            for p in s['pool']:
                if p['id'] in ids and p['text']:needed.setdefault(p['crop_array_sha256'],(s,p))
    if not needed:raise ValueError('No observable character-consensus pairs; alignment is unqualified')
    memory=gpu_start(c,'frozen PARSeq character alignment');started=time.monotonic()
    ocr=CropOCR(c,root/'alignment_unused_native_ocr.jsonl');recognizer=ocr.recognizer()
    recognizer.model.requires_grad_(False).eval();before=state_digest(recognizer.model)
    with (root/'source_ocr.pkl').open('rb') as stream:readings=pickle.load(stream)
    aligned={};items=list(needed.items());batch_size=c['runtime']['alignment_batch_size']
    for start in range(0,len(items),batch_size):
        batch=items[start:start+batch_size];crops=[];texts=[];boxes=[]
        for key,(s,p) in batch:
            rgb=rgb_for(s);a,b,d,e=p['padded_bbox_original'];crop=np.ascontiguousarray(rgb[b:e,a:d])
            if crop_hash(crop)!=key:raise ValueError('Character source RGB differs')
            crops.append(crop);texts.append(p['text']);boxes.append(p['padded_bbox_original'])
        values=align_batch(recognizer,crops,texts,boxes)
        for (key,(_,p)),row in zip(batch,values):
            confidence=readings[key]['token_probabilities'][:len(p['text'])]
            if len(confidence)!=len(p['text']):raise ValueError('Native probability/character cardinality differs')
            aligned[key]=dict(**row,text=p['text'],confidence=confidence,crop_array_sha256=key)
        if start% (batch_size*10)==0:log(root,'ALIGNMENT',completed=min(start+batch_size,len(items)),total=len(items))
    after=state_digest(recognizer.model)
    if before!=after:raise ValueError('PARSeq changed during read-only alignment')
    write(root/'character_alignments.json',aligned);write(root/'character_pairs.json',pairs)
    write(root/'alignment_qualification.json',dict(status='PASS',pairs=sum(len(v) for phase in pairs.values() for v in phase.values()),
        unique_crops=len(aligned),encoder_decoder_state_before=before,encoder_decoder_state_after=after,
        no_new_native_readings=True,extra_teacher_forcing_passes=len(aligned),
        memory=gpu_end(memory,'Frozen qualified PARSeq; read-only last decoder cross-attention hook',batch_size,[batch_size,3,32,128],started)))
    ocr.release()


def build_consensus_data(data,aligned,c,kind):
    result={};statistics={};opt=c['consensus']
    for phase,samples in data.items():
        result[phase]=[];stats=dict(eligible_parents=0,changed=0,new_strings=0,donated_characters=0,fit_positive=0)
        for s in samples:
            relation=neighbors(s,opt,c['compatibility_iou']);original={p['id']:p for p in s['pool']}
            pool=copy.deepcopy(s['pool']);x=[np.r_[row,np.zeros(12,np.float32)] for row in s['arrays']['features']]
            y=[0.]*len(pool);valid=[False]*len(pool);details=[]
            for p in s['pool']:
                if p['id'] not in relation:continue
                def observation(q):
                    return dict(**aligned[q['crop_array_sha256']],id=q['id'],height=q['padded_bbox_original'][3]-q['padded_bbox_original'][1])
                target=observation(p);donors=[observation(q) for q in relation[p['id']]]
                if kind=='consensus':
                    text,confidence,provenance=consensus_reading(target,donors,
                        maximum_distance=opt['maximum_character_distance'],confidence_margin=opt['confidence_margin'])
                else:
                    winner=max([target]+donors,key=lambda q:(float(np.mean(q['confidence'])),q['id']))
                    text,confidence,provenance=winner['text'],winner['confidence'],[winner['id']]*len(winner['text'])
                parents=list(p.get('parents',[])) or [p['id']]
                q=dict(p,id=kind+':'+p['id'],text=text,parents=parents,experiment_origin=kind,
                       source_proposal_id=p['id'],character_donors=provenance)
                changed=sum(a!=b for a,b in zip(p['text'],text))+abs(len(p['text'])-len(text));novel=text not in [v['text'] for v in s['pool']]
                feature=s['arrays']['features'][[v['id'] for v in s['pool']].index(p['id'])].copy()
                feature[-32:]=character_grams(text,32)
                extra=np.asarray([changed/max(1,len(text)),len(donors)/20,float(np.mean(confidence)),
                    float(np.min(confidence)),float(np.std(confidence)),float(novel),len(text)/25,
                    float(np.mean(target['confidence'])),float(np.min(target['confidence'])),
                    float(np.mean(target['entropy'])),float(np.mean([np.mean(d['entropy']) for d in donors])),1.],np.float32)
                pool.append(q);x.append(np.r_[feature,extra]);y.append(0.);valid.append(True)
                if phase=='fit':
                    y[-1]=float(matching_count(s['baseline_gnn']+[q],s['words'])>matching_count(s['baseline_gnn'],s['words']))
                    stats['fit_positive']+=int(y[-1])
                stats['eligible_parents']+=1;stats['changed']+=bool(changed);stats['new_strings']+=bool(novel)
                stats['donated_characters']+=sum(d is not None for d in provenance)
                details.append(dict(id=q['id'],source=p['text'],text=text,changed_characters=changed,new_to_pool=novel,donors=provenance))
            result[phase].append(dict(sample=s,pool=pool,x=np.asarray(x,np.float32).reshape(-1,236),
                adjacency=adjacency_for(pool,c),target=np.asarray(y,np.float32),valid=np.asarray(valid,bool),details=details))
        statistics[phase]=stats
    return result,statistics


def consensus_stage(root,c,smoke):
    data=load(root);aligned=read(root/'character_alignments.json');g=c['training']
    directory=root/'consensus';directory.mkdir(exist_ok=False)
    for kind in ('whole_reading','consensus'):
        prepared,statistics=build_consensus_data(data,aligned,c,kind);branch=directory/kind;branch.mkdir()
        write(branch/'generation.json',statistics)
        write(branch/'candidates.json',{phase:{z['sample']['sample_id']:z['details'] for z in values} for phase,values in prepared.items()})
        vectors=np.concatenate([z['x'] for z in prepared['fit'] if len(z['x'])])
        mean=torch.tensor(vectors.mean(0),device='cuda');scale=torch.tensor(vectors.std(0),device='cuda').clamp_min(g['minimum_scale'])
        for values in prepared.values():
            for z in values:
                z['tensor']=(torch.tensor(z['x'],device='cuda')-mean)/scale
                z['graph']=torch.tensor(z['adjacency'],device='cuda')
        torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed']);memory=gpu_start(c,kind);started=time.monotonic()
        model=HypothesisScorer(236,g['hidden'],g['layers'],message_passing=True).cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay']);rng=torch.Generator().manual_seed(c['seed'])
        initial=state_digest(model);base=read(root/'calibration_baseline_gnn_results.json')['aggregate']['spatial']
        best=dict(epoch=0,threshold='KEEP',f1=base['f1'],predictions=base['predictions']);history=[]
        def predict(phase,threshold):
            result={};model.eval()
            with torch.inference_mode():
                for z in prepared[phase]:
                    scores=model(z['tensor'],z['graph']).sigmoid().cpu().numpy()
                    scores[~z['valid']]=0.
                    result[z['sample']['sample_id']]=compatible_add(z['sample']['baseline_gnn'],z['pool'],
                        dict(zip([p['id'] for p in z['pool']],scores.tolist())),threshold,c['compatibility_iou'])
            return result
        for epoch in range(1,(1 if smoke else g['epochs'])+1):
            losses=[];model.train()
            for j in torch.randperm(len(prepared['fit']),generator=rng).tolist():
                z=prepared['fit'][j]
                if not z['valid'].any():continue
                valid=torch.tensor(z['valid'],device='cuda');target=torch.tensor(z['target'],device='cuda')
                opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(model(z['tensor'],z['graph'])[valid],target[valid])
                if not loss.isfinite():raise FloatingPointError('Nonfinite consensus loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True)
                opt.step();losses.append(float(loss.detach()))
            if not losses:raise ValueError('No consensus FIT updates; do not replace missing evidence with synthetic pairs')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
            if epoch%g['calibration_every']==0 or smoke or epoch==g['epochs']:
                trials=[]
                for threshold in g['thresholds']:
                    value=report(data['calibration'],predict('calibration',threshold),c)['aggregate']['spatial']
                    trials.append(dict(threshold=threshold,**value))
                    if (value['f1'],-value['predictions'])>(best['f1'],-best['predictions']):
                        best=dict(epoch=epoch,threshold=threshold,f1=value['f1'],predictions=value['predictions'])
                        torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),scale=scale.cpu()),branch/'best.pt')
                write(branch/f'epoch{epoch}_calibration.json',trials)
            write(branch/'history.json',history);log(root,'CONSENSUS_TRAIN',arm=kind,**history[-1],best=best)
        final_digest=state_digest(model)
        if initial==final_digest:raise ValueError('Consensus parameters unchanged')
        torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),scale=scale.cpu()),branch/'last.pt')
        for phase in ('evaluation','test'):save_arm(root,kind+'_final',phase,data[phase],predict(phase,g['raw_threshold']),c)
        if best['epoch']:model.load_state_dict(torch.load(branch/'best.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
        for phase in ('evaluation','test'):save_arm(root,kind,phase,data[phase],predict(phase,best['threshold']),c)
        write(branch/'selection_frozen.json',best)
        write(branch/'qualification.json',dict(parameters=sum(p.numel() for p in model.parameters()),initial_parameter_sha256=initial,
            final_parameter_sha256=final_digest,memory=gpu_end(memory,'Hypothesis GNN + AdamW; cached frozen PARSeq alignments',1,[40,236],started)))
        del model,opt;torch.cuda.empty_cache()
