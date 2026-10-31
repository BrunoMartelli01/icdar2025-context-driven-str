"""A graph acquires at most two visual observations; matched-cost controls."""
from __future__ import annotations
import copy
import json
import pickle
import time

import numpy as np
import torch
from torch.nn import functional as F

from localize.word_evidence_graphs import action_crop
from localize.word_graph_recovery import HypothesisScorer,conflicts
from scripts.word_evidence_consensus import adjacency_for,character_grams
from scripts.run_word_evidence_graphs import (load,log,rgb_for,state_digest,matching_count,save_arm)
from scripts.run_selective_word_repair_tests import gpu_start,gpu_end,report
from scripts.run_text_reconstruction_tests import write
from scripts.run_mask_crop_budget import CropOCR
from scripts.evaluate_cached_totaltext_padding import crop_hash


def actions_for(sample,c):
    rgb=rgb_for(sample);actions=[];prefix=sample['baseline_gnn'];seen=set()
    for index,p in enumerate(sample['pool']):
        if p['id'] in {q['id'] for q in prefix} or any(conflicts(p,q,c['compatibility_iou']) for q in prefix):continue
        for kind in c['active']['actions']:
            crop,box,removed=action_crop(rgb,p,sample['pool'],kind,fraction=c['active']['fraction'],
                overlap=c['compatibility_iou'],isolation_periphery=c['active']['isolation_periphery'])
            key=crop_hash(crop)
            if key==p['crop_array_sha256'] or key in seen:continue
            seen.add(key);parents=list(p.get('parents',[])) or [p['id']]
            q=dict(p,id='action:'+p['id']+':'+kind,parents=parents,padded_bbox_original=box,
                crop_array_sha256=key,source_proposal_id=p['id'],experiment_origin='active',action=kind)
            # Recognition pixels change; localization is the unchanged original box.
            actions.append(dict(proposal=q,crop=crop,parent=p,index=index,removed=removed))
    return actions


class ActionReader:
    def __init__(self,c,root):
        self.c=c;self.cache=CropOCR(c,root/'new_action_ocr.jsonl')
        with (root/'source_ocr.pkl').open('rb') as stream:self.cache.rows.update(pickle.load(stream))
        self.physical_new=0;self.current_digest=None;self.state_audits=[]

    def release(self):
        if self.cache.model is not None:
            after=state_digest(self.cache.model.model)
            if after!=self.current_digest:raise ValueError('Frozen action recognizer changed')
            self.state_audits.append(dict(before=self.current_digest,after=after))
        self.cache.release();self.current_digest=None

    def read(self,actions):
        pending={}
        for a in actions:
            key=a['proposal']['crop_array_sha256']
            if crop_hash(a['crop'])!=key:raise ValueError('Action RGB changed')
            if key not in self.cache.rows:pending.setdefault(key,a['crop'])
        keys=list(pending);size=self.c['runtime']['ocr_batch_size']
        for start in range(0,len(keys),size):
            batch=keys[start:start+size]
            recognizer=self.cache.recognizer()
            if self.current_digest is None:
                recognizer.model.requires_grad_(False).eval();self.current_digest=state_digest(recognizer.model)
            values=recognizer.recognize_candidate_batch([pending[k] for k in batch])
            if len(batch)!=len(values):raise ValueError('Action recognizer cardinality differs')
            with self.cache.output.open('a') as stream:
                for key,value in zip(batch,values):
                    row=dict(**value.to_dict(),crop_array_sha256=key,source='new qualified action RGB inference')
                    self.cache.rows[key]=row;stream.write(json.dumps(row,ensure_ascii=False)+'\n');self.physical_new+=1
        return {a['proposal']['id']:self.cache.rows[a['proposal']['crop_array_sha256']] for a in actions}


def observed_proposal(action,reading):return dict(action['proposal'],text=reading['raw_text'])


def state_features(sample,actions,observed,prefix,c):
    rows=[]
    for a in actions:
        p=a['proposal'];original=a['parent'];row=sample['arrays']['features'][a['index']].copy()
        observation=observed.get(p['id']);text=observation['raw_text'] if observation else original['text']
        if observation:row[-32:]=character_grams(text,32)
        from localize.occlusion_four_tests import iou
        confidence=observation['token_probabilities'] if observation else []
        row[185:192]=[0.,max((iou(p['bbox'],q['bbox']) for q in prefix),default=0.),
            float(any(conflicts(p,q,c['compatibility_iou']) for q in prefix)),
            float(any(q['text'].casefold()==text.casefold() for q in prefix)),len(text)/25,
            float(np.mean(confidence)) if confidence else row[190],float(np.min(confidence)) if confidence else row[191]]
        a0,b0,d0,e0=original['padded_bbox_original'];a1,b1,d1,e1=p['padded_bbox_original'];h=max(1,e0-b0)
        extra=[float(a['proposal']['action']==kind) for kind in c['active']['actions']]
        extra += [np.log(max(1,(d1-a1)*(e1-b1))/max(1,(d0-a0)*(e0-b0))),
            a['removed']/max(1,(d1-a1)*(e1-b1)),(a1+d1-a0-d0)/(2*h),(b1+e1-b0-e0)/(2*h),
            float(observation is not None),float(np.mean(confidence)) if confidence else 0.,
            float(np.min(confidence)) if confidence else 0.,len(observed)/c['active']['query_budget']]
        rows.append(np.r_[row,np.asarray(extra,np.float32)])
    return np.asarray(rows,np.float32).reshape(-1,236)


def action_targets(sample,actions,readings,prefix,c):
    base=matching_count(prefix,sample['words']);targets=[];valid=[]
    for a in actions:
        q=observed_proposal(a,readings[a['proposal']['id']])
        ok=not any(conflicts(q,p,c['compatibility_iou']) for p in prefix)
        targets.append(float(ok and matching_count(prefix+[q],sample['words'])>base));valid.append(ok)
    return np.asarray(targets,np.float32),np.asarray(valid,bool)


def active_stage(root,c,smoke):
    data=load(root);g=c['training'];directory=root/'active';directory.mkdir(exist_ok=False)
    memory=gpu_start(c,'active observation preparation');started=time.monotonic();reader=ActionReader(c,root)
    action_data={};states=[];counts={}
    for phase in data:
        action_data[phase]=[]
        for index,s in enumerate(data[phase]):
            actions=actions_for(s,c)
            adjacency=adjacency_for([a['proposal'] for a in actions],c)
            action_data[phase].append(dict(sample=s,actions=actions,adjacency=adjacency))
            if phase=='fit' and actions:
                readings=reader.read(actions);target,valid=action_targets(s,actions,readings,s['baseline_gnn'],c)
                def add_state(observed,prefix,post=False):
                    y,v=action_targets(s,actions,readings,prefix,c)
                    if not post:v &= np.asarray([a['proposal']['id'] not in observed for a in actions])
                    states.append(dict(x=state_features(s,actions,observed,prefix,c),adjacency=adjacency,target=y,valid=v))
                add_state({},s['baseline_gnn'])
                # FIT-only counterfactual supervision: all action outcomes were
                # genuinely measured. Oracle choices construct TRAIN states only.
                for positive in (True,False):
                    choices=np.where(valid & ((target>0)==positive))[0]
                    if not len(choices):continue
                    j=int(choices[0]);a=actions[j];observed={a['proposal']['id']:readings[a['proposal']['id']]}
                    add_state(observed,s['baseline_gnn'],post=True)
                    if positive:add_state(observed,s['baseline_gnn']+[observed_proposal(a,readings[a['proposal']['id']])])
            if index%100==0:log(root,'ACTION_PREPARE',split=phase,images=index+1,physical_new_ocr=reader.physical_new)
        counts[phase]=dict(images=len(data[phase]),potential_actions=sum(len(z['actions']) for z in action_data[phase]))
    reader.release()
    if not states:raise ValueError('No real FIT action training states')
    write(directory/'generation.json',dict(populations=counts,training_states=len(states),
        positive_targets=sum(int(z['target'][z['valid']].sum()) for z in states),
        eligible_targets=sum(int(z['valid'].sum()) for z in states),training_counterfactuals_use_fit_gt_only=True))
    write(directory/'preparation_memory.json',gpu_end(memory,'Frozen PARSeq RGB action outcomes; CPU cached graph states',c['runtime']['ocr_batch_size'],[c['runtime']['ocr_batch_size'],3,32,128],started))
    vectors=np.concatenate([z['x'] for z in states]);mean=torch.tensor(vectors.mean(0),device='cuda')
    scale=torch.tensor(vectors.std(0),device='cuda').clamp_min(g['minimum_scale'])
    for z in states:
        z['tensor']=(torch.tensor(z['x'],device='cuda')-mean)/scale
        z['graph']=torch.tensor(z['adjacency'],device='cuda')
        z['y']=torch.tensor(z['target'],device='cuda');z['v']=torch.tensor(z['valid'],device='cuda')
    def predict(model,phase,threshold,fixed=False):
        model.eval();predictions={};traces={}
        for z in action_data[phase]:
            s,actions=z['sample'],z['actions'];prefix=copy.deepcopy(s['baseline_gnn']);observed={};trace=[]
            if threshold=='KEEP':predictions[s['sample_id']]=prefix;traces[s['sample_id']]=[];continue
            for step in range(c['active']['query_budget']):
                eligible=[j for j,a in enumerate(actions) if a['proposal']['id'] not in observed and
                    not any(conflicts(a['proposal'],q,c['compatibility_iou']) for q in prefix)]
                if not eligible:break
                x=state_features(s,actions,observed,prefix,c)
                with torch.inference_mode():
                    scores=model((torch.tensor(x,device='cuda')-mean)/scale,torch.tensor(z['adjacency'],device='cuda')).sigmoid().cpu().tolist()
                if fixed:
                    order={kind:i for i,kind in enumerate(c['active']['fixed_action_order'])}
                    j=min(eligible,key=lambda k:(float(s['arrays']['features'][actions[k]['index']][190]),
                        order[actions[k]['proposal']['action']],actions[k]['proposal']['id']))
                else:j=max(eligible,key=lambda k:(scores[k],actions[k]['proposal']['id']))
                a=actions[j];reading=reader.read([a])[a['proposal']['id']];observed[a['proposal']['id']]=reading
                x=state_features(s,actions,observed,prefix,c)
                with torch.inference_mode():
                    probability=float(model((torch.tensor(x,device='cuda')-mean)/scale,
                        torch.tensor(z['adjacency'],device='cuda')).sigmoid()[j])
                accepted=probability>=threshold
                if accepted:prefix.append(observed_proposal(a,reading))
                trace.append(dict(step=step+1,id=a['proposal']['id'],action=a['proposal']['action'],
                    prior_score=scores[j],post_score=probability,text=reading['raw_text'],accepted=accepted,
                    crop_array_sha256=a['proposal']['crop_array_sha256'],bbox=a['proposal']['bbox'],
                    recognition_bbox=a['proposal']['padded_bbox_original']))
            if len(trace)>c['active']['query_budget']:raise ValueError('Observation budget overflow')
            predictions[s['sample_id']]=prefix;traces[s['sample_id']]=trace
        return predictions,traces
    for kind in ('self','gnn'):
        branch=directory/kind;branch.mkdir();torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
        memory=gpu_start(c,'active '+kind);started=time.monotonic()
        model=HypothesisScorer(236,g['hidden'],g['layers'],message_passing=kind=='gnn').cuda()
        opt=torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay']);rng=torch.Generator().manual_seed(c['seed'])
        initial=state_digest(model);base=report(data['calibration'],{s['sample_id']:s['baseline_gnn'] for s in data['calibration']},c)['aggregate']['spatial']
        best=dict(epoch=0,threshold='KEEP',f1=base['f1'],predictions=base['predictions']);history=[]
        for epoch in range(1,(1 if smoke else g['epochs'])+1):
            losses=[];model.train()
            for j in torch.randperm(len(states),generator=rng).tolist():
                z=states[j]
                if not z['v'].any():continue
                opt.zero_grad(set_to_none=True)
                loss=F.binary_cross_entropy_with_logits(model(z['tensor'],z['graph'])[z['v']],z['y'][z['v']])
                if not loss.isfinite():raise FloatingPointError('Nonfinite action utility loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True)
                opt.step();losses.append(float(loss.detach()))
            if not losses:raise ValueError('No action updates')
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
            if epoch%g['calibration_every']==0 or smoke or epoch==g['epochs']:
                trials=[]
                for threshold in g['thresholds']:
                    pp,trace=predict(model,'calibration',threshold);value=report(data['calibration'],pp,c)['aggregate']['spatial']
                    trials.append(dict(threshold=threshold,queries=sum(len(v) for v in trace.values()),**value))
                    if (value['f1'],-value['predictions'])>(best['f1'],-best['predictions']):
                        best=dict(epoch=epoch,threshold=threshold,f1=value['f1'],predictions=value['predictions'])
                        torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),scale=scale.cpu()),branch/'best.pt')
                write(branch/f'epoch{epoch}_calibration.json',trials)
                reader.release()
            write(branch/'history.json',history);log(root,'ACTIVE_TRAIN',arm=kind,**history[-1],best=best)
        final_digest=state_digest(model)
        if final_digest==initial:raise ValueError('Action model unchanged')
        torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),scale=scale.cpu()),branch/'last.pt')
        for phase in ('evaluation','test'):
            pp,trace=predict(model,phase,g['raw_threshold']);save_arm(root,'active_'+kind+'_final',phase,data[phase],pp,c)
            write(branch/(phase+'_final_trace.json'),trace)
        if best['epoch']:model.load_state_dict(torch.load(branch/'best.pt',map_location='cpu',weights_only=True)['state_dict'],strict=True)
        for phase in ('evaluation','test'):
            pp,trace=predict(model,phase,best['threshold']);save_arm(root,'active_'+kind,phase,data[phase],pp,c)
            write(branch/(phase+'_selected_trace.json'),trace)
        write(branch/'selection_frozen.json',best)
        if kind=='gnn':
            # Same trained admission model; isolate query choice at budget <=2.
            checkpoint_epoch=best['epoch'] or history[-1]['epoch']
            for phase in ('evaluation','test'):
                pp,trace=predict(model,phase,g['raw_threshold'])
                save_arm(root,'active_gnn_checkpoint_raw',phase,data[phase],pp,c)
                write(directory/(phase+'_checkpoint_raw_trace.json'),trace)
            fixed_best=dict(threshold='KEEP',f1=base['f1'],predictions=base['predictions'])
            for threshold in g['thresholds']:
                pp,_=predict(model,'calibration',threshold,fixed=True);value=report(data['calibration'],pp,c)['aggregate']['spatial']
                if (value['f1'],-value['predictions'])>(fixed_best['f1'],-fixed_best['predictions']):
                    fixed_best=dict(threshold=threshold,f1=value['f1'],predictions=value['predictions'])
            write(directory/'fixed_selection_frozen.json',dict(**fixed_best,common_model_checkpoint_epoch=checkpoint_epoch,
                comparison='fixed_queries_raw versus active_gnn_checkpoint_raw holds weights and threshold identical'))
            for phase in ('evaluation','test'):
                for suffix,threshold in [('',fixed_best['threshold']),('_raw',g['raw_threshold'])]:
                    pp,trace=predict(model,phase,threshold,fixed=True);save_arm(root,'fixed_queries'+suffix,phase,data[phase],pp,c)
                    write(directory/(phase+'_fixed'+suffix+'_trace.json'),trace)
        reader.release()
        write(branch/'qualification.json',dict(parameters=sum(p.numel() for p in model.parameters()),initial_parameter_sha256=initial,
            final_parameter_sha256=final_digest,memory=gpu_end(memory,'Utility graph/self + AdamW; sequential frozen PARSeq when uncached queries need pixels',1,[80,236],started)))
        del model,opt;torch.cuda.empty_cache()
    write(directory/'cost.json',dict(physical_new_action_inferences=reader.physical_new,
        recognizer_state_audits=reader.state_audits,
        deployment_budget='original <=20 unique RGB crops plus <=2 selected additional observations per image',
        training_counterfactual_cost_is_separate=True,selection_calibration_cost_is_separate=True,
        qualified_recognizer='unchanged pinned PARSeq native AR + refinement1'))
