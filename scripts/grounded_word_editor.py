"""Visual versus equal-capacity text-only explicit edit transducers."""
from __future__ import annotations
import copy
import gc
import time
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from models.grounded_word_recovery import GroundedVerifier, VisualEditTransducer
from recognition.grounded_word_recovery import (BOS,PAD,edit_script,apply_edits,pointer_sequence,choose_target,decide)
from scripts.run_grounded_word_recovery import (load,native_features,text_tensor,score_rows,native_scores,
    policies,metric,save_arm,log,gpu_start)
from scripts.run_selective_word_repair_tests import gpu_end
from scripts.run_text_reconstruction_tests import read,write
from scripts.run_word_evidence_graphs import state_digest


def edit_batch(rows,features,c,device):
    source=text_tensor([z['source'] for z in rows],c['maximum_bytes'],device)
    image=torch.tensor(np.stack([features[z['sample_id']]['image'][z['index']] for z in rows]),device=device)
    length=max(len(z['actions']) for z in rows)
    incoming=torch.full((len(rows),length),PAD,dtype=torch.long,device=device)
    target=incoming.clone();cursor=torch.zeros_like(incoming)
    for i,z in enumerate(rows):
        actions=z['actions'];n=len(actions)
        incoming[i,:n]=torch.tensor([BOS]+actions[:-1],device=device)
        target[i,:n]=torch.tensor(actions,device=device);cursor[i,:n]=torch.tensor(pointer_sequence(actions),device=device)
    return source,image,incoming,cursor,target


def edited_candidates(model,verifier,sample,visual,c,kind,prefix):
    by={p['id']:i for i,p in enumerate(sample['pool'])}
    supported=[p for p in prefix if len(p['text'].encode())<=c['maximum_bytes']]
    output=[];details=[];size=c['runtime']['editor_batch_size']
    for start in range(0,len(supported),size):
        original=supported[start:start+size];indices=[by[p['id']] for p in original]
        source=text_tensor([p['text'] for p in original],c['maximum_bytes'],'cuda')
        image=torch.tensor(visual['image'][indices],device='cuda')
        actions=model.generate(source,image,[len(p['text'].encode()) for p in original])
        for p,script,index in zip(original,actions,indices):
            value=apply_edits(p['text'],script,c['maximum_bytes'])
            row=dict(source_id=p['id'],source=p['text'],actions=script,**value)
            details.append(row)
            # Invalid generated UTF8 is an explicit abstention; not an OCR fallback.
            if not value['valid'] or value['text']==p['text']:continue
            q=dict(p,id='edit:'+kind+':'+p['id'],text=value['text'],parents=p.get('parents') or [p['id']],
                   source_proposal_id=p['id'],edit_actions=script,experiment_origin='visual_edit_'+kind)
            output.append((q,index))
    edited=[p for p,i in output]
    if edited:
        indices=[i for p,i in output]
        scores=score_rows(verifier,sample,edited,(visual['image'][indices],visual['context'][indices]),c)[:,-1]
    else:scores=[]
    return edited,dict(zip([p['id'] for p in edited],map(float,scores))),details


def editor_stage(root,c,smoke):
    data=load(root);features=native_features(root,data);directory=root/'editor';directory.mkdir()
    prefixes={phase:read(root/f'{phase}_verifier_replacement_predictions.json') if phase!='calibration' else None
              for phase in ('calibration','evaluation','test')}
    v=c['verifier'];verifier=GroundedVerifier(c['visual_dimensions'],v['hidden'],v['layers'],c['maximum_bytes']).cuda().eval()
    verifier.load_state_dict(torch.load(root/'verifier/best.pt',weights_only=True),strict=True);verifier.requires_grad_(False)
    original_scores={p:native_scores(root,verifier,data,features,c,p) for p in ('calibration','evaluation','test')}
    selection=read(root/'verifier/selection_frozen.json')['replacement']['policy']
    prefixes['calibration']={s['sample_id']:decide(s['baseline_gnn'],s['pool'],original_scores['calibration'][s['sample_id']],
        **selection,overlap=c['compatibility_iou'],replacements=True) for s in data['calibration']}
    training=[];counts=Counter()
    for sample in data['fit']:
        for i,p in enumerate(sample['pool']):
            target=choose_target(p,sample['words'],c['matching_iou'])
            if target is None:counts['excluded_no_valid_geometry']+=1;continue
            if max(len(target['text'].encode()),len(p['text'].encode()))>c['maximum_bytes']:
                counts['excluded_explicit_byte_limit']+=1;continue
            actions=edit_script(p['text'],target['text'])
            if apply_edits(p['text'],actions,c['maximum_bytes']) != dict(valid=True,text=target['text'],consumed=len(p['text'].encode())):
                raise ValueError('FIT edit supervision does not reconstruct exact target')
            training.append(dict(sample_id=sample['sample_id'],index=i,source=p['text'],target=target['text'],actions=actions))
            counts['identity' if p['text']==target['text'] else 'real_OCR_error']+=1
    if not training:raise ValueError('No real FIT word/crop supervision')
    write(directory/'training_data.json',dict(counts=dict(counts),examples=len(training),
        targets='best original-box IoU>=configured threshold, stable GT-order ties; FIT only; exact UTF8'))
    g=c['editor']
    for kind in ('text','visual'):
        branch=directory/kind;branch.mkdir();memory=gpu_start(c,'edit '+kind);tick=time.monotonic()
        torch.manual_seed(c['seed']);model=VisualEditTransducer(c['visual_dimensions'],g['hidden'],g['layers'],c['maximum_bytes'],visual=kind=='visual').cuda()
        initial=state_digest(model);optimizer=torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay'])
        rng=torch.Generator().manual_seed(c['seed']);history=[];best=None
        def generate(phase):
            candidates={};scores={};details={}
            for s in data[phase]:
                sid=s['sample_id'];pp,sc,dd=edited_candidates(model,verifier,s,features[phase][sid],c,kind,prefixes[phase][sid])
                candidates[sid]=pp;scores[sid]=dict(original_scores[phase][sid],**sc);details[sid]=dd
            return candidates,scores,details
        def predictions(phase,proposals,scores,policy):
            return {s['sample_id']:decide(prefixes[phase][s['sample_id']],proposals[s['sample_id']],scores[s['sample_id']],
                **policy,overlap=c['compatibility_iou'],replacements=True) for s in data[phase]}
        for epoch in range(1,(1 if smoke else g['epochs'])+1):
            model.train();losses=[];order=torch.randperm(len(training),generator=rng).tolist()
            for start in range(0,len(order),c['runtime']['editor_batch_size']):
                rows=[training[i] for i in order[start:start+c['runtime']['editor_batch_size']]]
                source,image,incoming,cursor,target=edit_batch(rows,features['fit'],c,'cuda')
                logits=model(source,image,incoming,cursor)
                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),target.reshape(-1),ignore_index=PAD)
                if not loss.isfinite():raise FloatingPointError('Nonfinite edit loss')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
            history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
            if epoch%g['calibration_every']==0 or smoke or epoch==g['epochs']:
                model.eval();pp,sc,details=generate('calibration');trials=[]
                for policy in policies(c):
                    value=metric(data['calibration'],predictions('calibration',pp,sc,policy),c)
                    trials.append(dict(policy=policy,**value));key=(value['f1'],-value['predictions'],policy['threshold']=='KEEP')
                    if best is None or key>best[0]:
                        best=(key,dict(epoch=epoch,policy=policy,metric=value));torch.save(model.state_dict(),branch/'best.pt')
                write(branch/f'epoch{epoch}_calibration.json',trials)
            write(branch/'history.json',history);log(root,'EDIT_TRAIN',arm=kind,**history[-1],selected=best[1] if best else None)
        if initial==state_digest(model):raise ValueError('Editor did not train')
        torch.save(model.state_dict(),branch/'last.pt')
        # Freeze the selected policy before either held-out generation/evaluation.
        write(branch/'selection_frozen.json',best[1])
        for suffix,checkpoint,policy in [('final','last.pt',dict(threshold=c['policy']['raw_threshold'],margin=c['policy']['raw_margin'])),
                                        ('selected','best.pt',best[1]['policy'])]:
            model.load_state_dict(torch.load(branch/checkpoint,weights_only=True),strict=True);model.eval()
            for phase in ('evaluation','test'):
                pp,sc,details=generate(phase)
                write(branch/f'{phase}_{suffix}_edits.json',details)
                save_arm(root,'editor_'+kind+'_'+suffix,phase,data[phase],predictions(phase,pp,sc,policy),c)
        write(branch/'qualification.json',dict(initial_parameter_sha256=initial,final_parameter_sha256=state_digest(model),
            extra_native_OCR_calls=0, extra_ENCODER_calls=0,
            memory=gpu_end(memory,'Byte edit transducer + frozen grounded verifier + AdamW; cached visual sequences',
                c['runtime']['editor_batch_size'],dict(maximum_action_length=2*c['maximum_bytes']+1,
                    maximum_source_length=c['maximum_bytes']+1,visual_columns=c['visual_columns'],visual_channels=c['visual_dimensions']),tick)))
        del model,optimizer;gc.collect();torch.cuda.empty_cache()
    del verifier,features;gc.collect();torch.cuda.empty_cache()
