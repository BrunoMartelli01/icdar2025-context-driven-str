"""Learned versus fixed queries for unread proposals and new tile localizations."""
from __future__ import annotations
import copy
import gc
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from torch.nn import functional as F

from localize.word_graph_recovery import HypothesisScorer
from localize.word_multitask import parseq_word_proposals
from localize.occlusion_four_tests import decode_instance_labels
from models.adaspot_gap import AdaSpotGapAGUNet
from models.grounded_word_recovery import GroundedVerifier
from recognition.grounded_word_recovery import (tile_boxes,map_tile_box,geometry_features,
    decide,operation)
from recognition.text_reconstruction import box_iou
from recognition.selective_word_repair import exact_matching
from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
from scripts.run_grounded_word_recovery import (load,native_features,VisualBank,native_scores,score_rows,
    policies,metric,save_arm,log,PHASES,gpu_start)
from scripts.run_selective_word_repair_tests import gpu_end
from scripts.run_teacher_student_tests import frozen_selector
from scripts.run_occlusion_four_tests import score_record
from scripts.run_text_reconstruction_tests import read,write,sha
from scripts.run_word_evidence_graphs import rgb_for,state_digest,verify
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor


def query_features(sample,actions,visual,prefix,observed,c):
    rows=[]
    for index,action in enumerate(actions):
        rows.append(np.r_[visual[index].mean(0),geometry_features(action['box'],sample['image_hw'],prefix,
            observed=float(action['id'] in observed),kind=float(action['kind']=='tile')),
            float(action['kind']=='retrieval'),float(action['kind']=='tile'),action['prior'],
            observed.get(action['id'],{}).get('score',0.)])
    return np.asarray(rows,np.float32).reshape(-1,c['search']['dimensions'])


def query_adjacency(actions,c):
    matrix=np.eye(len(actions),dtype=np.float32)
    for i,a in enumerate(actions):
        box=a['box'];center=np.asarray([(box[0]+box[2])/2,(box[1]+box[3])/2]);choices=[]
        for j,b in enumerate(actions):
            if i==j:continue
            q=b['box'];other=np.asarray([(q[0]+q[2])/2,(q[1]+q[3])/2])
            distance=float(np.linalg.norm(center-other)/np.sqrt(max(1,box[3]-box[1])*max(1,q[3]-q[1])))
            if distance<=c['search']['maximum_distance']:choices.append((distance,j))
        for _,j in sorted(choices)[:c['search']['neighbors']]:matrix[i,j]=matrix[j,i]=1.
    degree=matrix.sum(1)
    return matrix/np.sqrt(degree[:,None]*degree[None])


class SearchEnvironment:
    def __init__(self,root,c,bank,verifier,pre,source_manifest):
        self.root,self.c,self.bank,self.verifier,self.pre=root,c,bank,verifier,pre
        self.manifest=source_manifest;self.actions_cache={};self.outcomes={}
        self.segmenter=None;self.segmentation_calls=0;self.segmentation_before=None;self.saved_inputs={}
        self.native_source=Path(c['paths']['occlusion_run'])
        self.contract=read(c['paths']['segmentation_contract']);self.mask_directory=root/'tile_masks';self.mask_directory.mkdir()

    def actions(self,phase,sample):
        key=(phase,sample['sample_id'])
        if key in self.actions_cache:return self.actions_cache[key]
        relative='exports/'+phase+'/'+sample['sample_id'].replace(':','__')+'.json'
        expected=self.manifest[relative];expected=expected['sha256'] if isinstance(expected,dict) else expected
        verify(self.native_source/relative,expected);self.saved_inputs[relative]=expected
        record=read(self.native_source/relative)
        if record['image_sha256']!=sample['image_sha256'] or record['image_hw']!=sample['image_hw']:
            raise ValueError('Unread source proposals belong to different RGB/coordinates')
        before=score_record(record,'mk',None,self.pre,False)
        seen={p['crop_array_sha256'] for p in sample['pool']}
        retrieval=[]
        for p in sorted(record['arms']['mk']['pool'],key=lambda p:(-before[p['id']],p['id'])):
            if p['crop_array_sha256'] in seen:continue
            seen.add(p['crop_array_sha256'])
            proposal=dict(p,id='retrieval:'+p['id'],source_proposal_id=p['id'],experiment_origin='new_localization_query')
            proposal.pop('text',None)
            retrieval.append(dict(id=proposal['id'],kind='retrieval',box=p['padded_bbox_original'],proposal=proposal,prior=before[p['id']]))
            if len(retrieval)>=self.c['search']['retrieval_candidates']:break
        tiles=[dict(id='tile:'+str(i),kind='tile',box=list(box),prior=0.)
               for i,box in enumerate(tile_boxes(sample['image_hw'],self.c['search']['tile_fraction']))]
        actions=retrieval+tiles;rgb=rgb_for(sample)
        crops=[np.ascontiguousarray(rgb[a['box'][1]:a['box'][3],a['box'][0]:a['box'][2]]) for a in actions]
        visual=np.asarray(self.bank.encode(crops),np.float32)
        adjacency=query_adjacency(actions,self.c)
        self.actions_cache[key]=(actions,visual,adjacency)
        return self.actions_cache[key]

    def segmentation(self):
        if self.segmenter is None:
            self.segmenter=AdaSpotGapAGUNet('control',skip_floor=self.contract['resolved_config']['skip_floor'],
                initial_gap_bias=self.contract['resolved_config']['initial_gap_bias'])
            checkpoint=torch.load(self.c['paths']['segmentation_checkpoint'],map_location='cpu',weights_only=False)
            if checkpoint['run_contract']!=self.contract:raise ValueError('Qualified segmentation checkpoint/contract differ')
            self.segmenter.load_state_dict(checkpoint['model_state_dict'],strict=True)
            self.segmenter.requires_grad_(False).cuda().eval();self.segmentation_before=state_digest(self.segmenter)
        return self.segmenter

    def tile_proposal(self,phase,sample,action):
        rgb=rgb_for(sample);x,y,u,v=action['box'];crop=np.ascontiguousarray(rgb[y:v,x:u])
        model=self.segmentation();transform=_transform(self.contract['resolved_preprocessing'],'letterbox',train=False)
        with torch.inference_mode():probability=model(_tensor(crop,transform).cuda())[0].sigmoid().cpu().numpy()
        self.segmentation_calls+=1
        canvas=letterbox_canvas_mask((u-x,v-y),native_size=probability.shape[-1])
        labels=decode_instance_labels(probability,None,canvas,self.contract['resolved_config']['postprocessing'],None,'cut')
        local=parseq_word_proposals(labels,(u-x,v-y),padding=0,min_area=self.contract['resolved_config']['postprocessing']['min_area'],max_instances=None)
        candidates=[]
        for p in local:
            box=map_tile_box(p['bbox'],action['box'])
            overlap=max((box_iou(box,q['bbox']) for q in sample['pool']),default=0.)
            if overlap>=self.c['search']['novelty_iou']:continue
            support=labels==p['instance_id'];quality=float(probability[0,support].mean())
            score=quality*np.sqrt(int(support.sum()))*(1-overlap)
            candidates.append((score,-p['instance_id'],box,p))
        folder=self.mask_directory/phase;folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/(sample['sample_id'].replace(':','__')+'_'+action['id'].replace(':','_')+'.npz'),
            probabilities=probability,labels=labels,canvas=canvas,tile=np.asarray(action['box']))
        if not candidates:return None
        score,_,box,p=max(candidates,key=lambda row:row[:2])
        return dict(id='zoom:'+action['id'],bbox=box,padded_bbox_original=box,bbox_native=p['bbox_native'],
            parents=[],method='zoom',experiment_origin='new_localization_query',local_instance_id=p['instance_id'],
            tile_bbox=action['box'],component_area_native=p['foreground_area'],proposal_score=float(score))

    def acquire(self,phase,sample,action):
        key=(phase,sample['sample_id'],action['id'])
        if key in self.outcomes:return self.outcomes[key]
        proposal=copy.deepcopy(action['proposal']) if action['kind']=='retrieval' else self.tile_proposal(phase,sample,action)
        if proposal is None:value=dict(proposal=None,score=0.,kind=action['kind'],outcome='no_novel_geometry')
        else:
            image,context=self.bank.observe(sample,[proposal]);self.bank.read(sample,[proposal])
            scores=score_rows(self.verifier,sample,[proposal],(image,context),self.c)[0]
            value=dict(proposal=proposal,score=float(scores[-1]),geometry_score=float(scores[0]),
                       spelling_score=float(scores[1]),kind=action['kind'],outcome='native_reading')
        self.outcomes[key]=value
        return value


def query_operation(prefix,outcome,scores,policy,c):
    if outcome['proposal'] is None:return copy.deepcopy(prefix)
    proposal=outcome['proposal'];values=dict(scores);values[proposal['id']]=outcome['score']
    return decide(prefix,[proposal],values,**policy,overlap=c['compatibility_iou'],replacements=True)


def search_stage(root,c,smoke):
    data=load(root);features=native_features(root,data);directory=root/'search';directory.mkdir()
    memory=gpu_start(c,'new localization queries');tick=time.monotonic();g=c['search']
    v=c['verifier'];verifier=GroundedVerifier(c['visual_dimensions'],v['hidden'],v['layers'],c['maximum_bytes']).cuda().eval()
    verifier.load_state_dict(torch.load(root/'verifier/best.pt',weights_only=True),strict=True);verifier.requires_grad_(False)
    source_cache=pickle.loads((Path(c['paths']['evidence_run'])/'source_ocr.pkl').read_bytes())
    bank=VisualBank(directory,c,source_cache)
    # Restore only actually measured native/source visual features, keyed by RGB.
    for phase,samples in data.items():
        for s in samples:
            for p,z in zip(s['pool'],features[phase][s['sample_id']]['image']):bank.values[p['crop_array_sha256']]=z
    original_scores={phase:native_scores(root,verifier,data,features,c,phase) for phase in PHASES}
    selected=read(root/'verifier/selection_frozen.json')['replacement']['policy']
    prefix={phase:{s['sample_id']:decide(s['baseline_gnn'],s['pool'],original_scores[phase][s['sample_id']],
        **selected,overlap=c['compatibility_iou'],replacements=True) for s in data[phase]} for phase in PHASES}
    full_manifest=read(Path(c['paths']['occlusion_run'])/'final_artifact_manifest.json')
    pre=frozen_selector(dict(source_run=c['paths']['occlusion_run']),'pre')
    environment=SearchEnvironment(directory,c,bank,verifier,pre,full_manifest)
    states=[];fit_outcomes={};training_counts=dict(states=0,positives=0,labels=0,observed_counts={})
    for number,s in enumerate(data['fit']):
        sid=s['sample_id'];actions,visual,adjacency=environment.actions('fit',s)
        outcomes={a['id']:environment.acquire('fit',s,a) for a in actions}
        fit_outcomes[sid]=outcomes
        # Every single-observation state, without a first-positive/left bias.
        histories=[{}]+[{a['id']:outcomes[a['id']]} for a in actions]
        available=exact_matching(s['pool'],s['words'],c['matching_iou'])['correct']
        for observed in histories:
            current=copy.deepcopy(prefix['fit'][sid]);scores=dict(original_scores['fit'][sid])
            for outcome in observed.values():
                current=query_operation(current,outcome,scores,selected,c)
                if outcome['proposal'] is not None:scores[outcome['proposal']['id']]=outcome['score']
            before=exact_matching(current,s['words'],c['matching_iou'])['correct'];target=[];valid=[]
            for action in actions:
                outcome=outcomes[action['id']];candidate=outcome['proposal']
                change=operation(current,candidate,overlap=c['compatibility_iou']) if candidate else None
                beneficial=False
                if change is not None:
                    after=exact_matching(change[0],s['words'],c['matching_iou'])['correct']
                    novel=exact_matching(s['pool']+[candidate],s['words'],c['matching_iou'])['correct']>available
                    beneficial=after>before and novel
                target.append(float(beneficial));valid.append(action['id'] not in observed)
            states.append(dict(x=query_features(s,actions,visual,current,observed,c),adjacency=adjacency,
                target=np.asarray(target,np.float32),valid=np.asarray(valid,bool)))
            training_counts['states']+=1;training_counts['labels']+=sum(valid)
            training_counts['positives']+=sum(t for t,ok in zip(target,valid) if ok)
            key=str(len(observed));training_counts['observed_counts'][key]=training_counts['observed_counts'].get(key,0)+1
        if number%50==0:log(root,'SEARCH_FIT_COUNTERFACTUALS',images=number+1,segmentation_calls=environment.segmentation_calls,new_OCR=bank.native_new)
    if not states or not training_counts['positives']:raise ValueError('No genuine new FIT word outcomes; no synthetic positives substituted')
    write(directory/'training_data.json',training_counts);write(directory/'fit_outcomes.json',fit_outcomes)
    write(directory/'preparation_memory.json',gpu_end(memory,'Frozen U-Net/PARSeq/grounded verifier; real FIT counterfactuals',1,[1,3,512,512],tick))
    vectors=np.concatenate([z['x'] for z in states]);mean=torch.tensor(vectors.mean(0),device='cuda')
    scale=torch.tensor(vectors.std(0),device='cuda').clamp_min(c['search']['minimum_scale'])
    for z in states:
        z['tensor']=(torch.tensor(z['x'],device='cuda')-mean)/scale;z['graph']=torch.tensor(z['adjacency'],device='cuda')
        z['y']=torch.tensor(z['target'],device='cuda');z['validity']=torch.tensor(z['valid'],device='cuda')
    def predict(model,phase,policy,*,fixed=False):
        model.eval();predictions={};traces={}
        for s in data[phase]:
            sid=s['sample_id'];current=copy.deepcopy(prefix[phase][sid]);observed={};scores=dict(original_scores[phase][sid]);trace=[]
            if policy['threshold']=='KEEP':predictions[sid]=current;traces[sid]=trace;continue
            actions,visual,adjacency=environment.actions(phase,s)
            for step in range(g['query_budget']):
                eligible=[i for i,a in enumerate(actions) if a['id'] not in observed]
                if not eligible:break
                with torch.inference_mode():
                    utility=model((torch.tensor(query_features(s,actions,visual,current,observed,c),device='cuda')-mean)/scale,
                        torch.tensor(adjacency,device='cuda')).sigmoid().cpu().tolist()
                index=min(eligible) if fixed else max(eligible,key=lambda i:(utility[i],actions[i]['id']))
                action=actions[index];outcome=environment.acquire(phase,s,action);observed[action['id']]=outcome
                previous=[p['id'] for p in current];current=query_operation(current,outcome,scores,policy,c)
                if outcome['proposal'] is not None:scores[outcome['proposal']['id']]=outcome['score']
                trace.append(dict(step=step+1,action=action,utility=utility[index],outcome=outcome,
                    changed=previous!=[p['id'] for p in current]))
            predictions[sid]=current;traces[sid]=trace
        return predictions,traces
    for kind in ('self','gnn'):
        branch=directory/kind;branch.mkdir();tick=time.monotonic();memory=gpu_start(c,'search utility '+kind)
        torch.manual_seed(c['seed']);model=HypothesisScorer(g['dimensions'],g['hidden'],g['layers'],message_passing=kind=='gnn').cuda()
        initial=state_digest(model);optimizer=torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay'])
        rng=torch.Generator().manual_seed(c['seed']);best=None;history=[];maximum_training_nodes=0
        for epoch in range(1,(1 if smoke else g['epochs'])+1):
            model.train();losses=[]
            order=torch.randperm(len(states),generator=rng).tolist()
            for start in range(0,len(order),g['state_batch_size']):
                batch=[states[i] for i in order[start:start+g['state_batch_size']] if states[i]['validity'].any()]
                if not batch:continue
                maximum_training_nodes=max(maximum_training_nodes,sum(len(z['y']) for z in batch))
                logits=model(torch.cat([z['tensor'] for z in batch]),torch.block_diag(*[z['graph'] for z in batch]))
                offset=0;per_state=[]
                for z in batch:
                    values=logits[offset:offset+len(z['y'])];valid=z['validity'];offset+=len(z['y'])
                    per_state.append(F.binary_cross_entropy_with_logits(values[valid],z['y'][valid]))
                loss=torch.stack(per_state).mean()
                if not loss.isfinite():raise FloatingPointError('Nonfinite query utility loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
            if epoch%g['calibration_every']==0 or smoke or epoch==g['epochs']:
                trials=[]
                for policy in policies(c):
                    pp,tr=predict(model,'calibration',policy);value=metric(data['calibration'],pp,c)
                    trials.append(dict(policy=policy,queries=sum(map(len,tr.values())),**value))
                    key=(value['f1'],-value['predictions'],policy['threshold']=='KEEP')
                    if best is None or key>best[0]:
                        best=(key,dict(epoch=epoch,policy=policy,metric=value));torch.save(dict(state_dict=model.state_dict(),mean=mean.cpu(),scale=scale.cpu()),branch/'best.pt')
                write(branch/f'epoch{epoch}_calibration.json',trials)
            write(branch/'history.json',history);log(root,'SEARCH_UTILITY_TRAIN',arm=kind,**history[-1],selected=best[1] if best else None)
        if initial==state_digest(model):raise ValueError('Query utility parameters unchanged')
        torch.save(model.state_dict(),branch/'last.pt');write(branch/'selection_frozen.json',best[1])
        for suffix,policy in [('final',dict(threshold=c['policy']['raw_threshold'],margin=c['policy']['raw_margin'])),('selected',best[1]['policy'])]:
            if suffix=='selected':model.load_state_dict(torch.load(branch/'best.pt',weights_only=True)['state_dict'],strict=True)
            for phase in ('evaluation','test'):
                pp,trace=predict(model,phase,policy);save_arm(root,'search_'+kind+'_'+suffix,phase,data[phase],pp,c)
                write(branch/f'{phase}_{suffix}_trace.json',trace)
        if kind=='gnn':
            # Hold utility checkpoint, observations, gate and thresholds identical.
            for phase in ('evaluation','test'):
                for name,fixed in [('search_gnn_matched',False),('search_fixed_matched',True)]:
                    policy=dict(threshold=c['policy']['raw_threshold'],margin=c['policy']['raw_margin'])
                    pp,trace=predict(model,phase,policy,fixed=fixed);save_arm(root,name,phase,data[phase],pp,c)
                    write(directory/f'{phase}_{name}_trace.json',trace)
        write(branch/'qualification.json',dict(initial_sha256=initial,final_sha256=state_digest(model),
            maximum_training_nodes=maximum_training_nodes,
            memory=gpu_end(memory,'Query GNN/self + AdamW; frozen U-Net/PARSeq/verifier when required',
                g['state_batch_size'],[maximum_training_nodes,g['dimensions']],tick)))
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    if environment.segmenter is not None and state_digest(environment.segmenter)!=environment.segmentation_before:
        raise ValueError('Frozen segmentation model changed')
    bank.close();write(directory/'cost.json',dict(actual_segmentation_calls=environment.segmentation_calls,
        actual_new_native_OCR=bank.native_new,extra_encoder_passes=bank.encodings,frozen_parseq=bank.audit,
        segmentation_state_before=environment.segmentation_before,
        segmentation_state_after=state_digest(environment.segmenter) if environment.segmenter is not None else None,
        measured_source_exports=environment.saved_inputs))
    del environment,bank,verifier,features,states;gc.collect();torch.cuda.empty_cache()
