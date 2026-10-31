"""Three controlled APPROXIMATE recovery experiments; immutable-source execution."""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time
import traceback

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from models.grounded_word_recovery import GroundedVerifier, VisualEditTransducer
from recognition.grounded_word_recovery import (BOS, PAD, KEEP, DELETE, STOP, edit_script,
    apply_edits, pointer_sequence, expanded_box, geometry_features, choose_target, decide)
from recognition.text_reconstruction import box_iou
from scripts.run_word_evidence_graphs import rgb_for, state_digest, verify
from scripts.run_selective_word_repair_tests import setup, gpu_start as _gpu_start, gpu_end, report
from scripts.run_mask_crop_budget import CropOCR
from scripts.run_text_reconstruction_tests import read, write, sha
from scripts.evaluate_cached_totaltext_padding import crop_hash

REPO = Path(__file__).resolve().parents[1]
PHASES = ('fit', 'calibration', 'evaluation', 'test')


def gpu_start(c,label):
    started=time.monotonic();deadline=started+c['runtime']['gpu_wait_seconds']
    while torch.cuda.is_available() and torch.cuda.mem_get_info()[0]<c['runtime']['minimum_free_bytes']:
        if time.monotonic()>=deadline:raise RuntimeError('GPU memory remained below declared minimum; other jobs preserved')
        print('GPU_WAIT',json.dumps(dict(stage=label,free_bytes=torch.cuda.mem_get_info()[0])),flush=True)
        time.sleep(min(c['runtime']['gpu_poll_seconds'],max(0.,deadline-time.monotonic())))
    return dict(**_gpu_start(c,label),resource_wait_seconds=time.monotonic()-started)


def log(root, stage, **values):
    write(root/'status.json', dict(status='RUNNING', stage=stage, pid=os.getpid(), **values))
    print(stage, json.dumps(values), flush=True)


def validate_parameters(c):
    if c['claim_label']!='approximate reproduction' or c['matching_iou']!=.5:
        raise ValueError('Qualified approximate word-matching protocol required')
    if c['runtime']['precision']!='float32' or c['runtime']['device']!='cuda:0':
        raise ValueError('Explicit qualified FP32 CUDA runtime required')
    for name in ('verifier','editor','search'):
        g=c[name]
        if g['hidden']<4 or g['hidden']%4 or min(g['epochs'],g['layers'],g['calibration_every'])<1:
            raise ValueError('Invalid declared architecture/training budget')
        if g['learning_rate']<=0 or g['weight_decay']<0 or g['gradient_clip']<=0:
            raise ValueError('Invalid declared optimizer')
    if c['search']['query_budget']!=2 or c['search']['dimensions']!=c['visual_dimensions']+16:
        raise ValueError('Qualified two-query feature contract required')
    if c['maximum_bytes']<1 or c['context_fraction']<0 or not .5<=c['search']['tile_fraction']<=1:
        raise ValueError('Invalid explicit crop/byte bounds')
    if any(not 0<t<=1 for t in c['policy']['thresholds']) or any(not 0<=m<=1 for m in c['policy']['margins']):
        raise ValueError('Invalid declared word policy')


def load(root):
    with (root/'prepared.pkl').open('rb') as stream: return pickle.load(stream)


def metric(samples, outputs, c):
    return report(samples, outputs, c)['aggregate']['spatial']


def save_arm(root, name, phase, samples, predictions, c):
    reference = {s['sample_id']:s['baseline_gnn'] for s in samples}
    write(root/f'{phase}_{name}_predictions.json', predictions)
    result = report(samples, predictions, c, reference)
    write(root/f'{phase}_{name}_results.json', result)
    return result


def prepare(root, c, smoke):
    root.mkdir(exist_ok=False, parents=True)
    for path, expected in c['input_sha256'].items(): verify(path, expected)
    source = Path(c['paths']['evidence_run']); manifest = read(source/'artifact_manifest.json')['artifacts']
    for relative, entry in manifest.items(): verify(source/relative, entry['sha256'] if isinstance(entry, dict) else entry)
    data = pickle.loads((source/'prepared.pkl').read_bytes())
    if smoke: data = {p:rows[:c['smoke_images'][p]] for p, rows in data.items()}
    evidence = read(source/'resolved_config.json')
    if any(c['paths'][k] != evidence['paths'][k] for k in ('parseq_source','parseq_checkpoint','parseq_archive')):
        raise ValueError('Qualified recognizer identity changed')
    for phase, samples in data.items():
        for sample in samples:
            verify(sample['image_path'], sample['image_sha256']); verify(sample['annotation_path'], sample['annotation_sha256'])
            verify(sample['maps_path'], sample['maps_sha256'])
    (root/'prepared.pkl').write_bytes(pickle.dumps(data))
    write(root/'resolved_config.json', c)
    write(root/'protocol.json', dict(result_class='smoke-test result' if smoke else 'approximate reproduction',
        claim_label='approximate reproduction', mode='approximate', populations={p:len(v) for p,v in data.items()},
        metric='Micro maximum one-to-one original bbox IoU>=0.5 AND exact Unicode casefold; punctuation retained; all care GT/output/empty/duplicate counts; no ignore neutralization',
        seed=c['seed'], independent_validation=False, source_calibration_exposure=True,
        historical_test_exploration=True, known_duplicate_test_ids=c['known_duplicate_test_ids'],
        config_sha256=sha(root/'resolved_config.json'), primary_inputs_verified=len(manifest),
        scientific_decisions=c['scientific_decisions'], source_code_sha256={str(p.relative_to(REPO)):sha(p)
            for folder in ('recognition','models','localize','scripts') for p in sorted((REPO/folder).glob('*.py'))}))
    for phase in ('calibration','evaluation','test'):
        value = save_arm(root, 'baseline_gnn', phase, data[phase], {s['sample_id']:s['baseline_gnn'] for s in data[phase]}, c)
        if phase == 'test' and not smoke and value['aggregate']['spatial']['correct'] != 1201:
            raise ValueError('Reference GNN must reproduce 1201 TEST matches')
    return data


class VisualBank:
    """Read-only frozen PARSeq encoding; native decoding remains unchanged."""
    def __init__(self, root, c, cache):
        self.root, self.c, self.values = root, c, {}
        self.ocr = CropOCR(c, root/'new_native_ocr.jsonl')
        self.ocr.rows.update(cache)
        self.encodings = 0; self.encoder_batches = 0; self.native_new = 0; self.audit = []
        self.before = None

    def recognizer(self):
        recognizer = self.ocr.recognizer()
        recognizer.model.requires_grad_(False).eval()
        if self.before is None: self.before = state_digest(recognizer.model)
        return recognizer

    def encode(self, crops):
        pending = {}
        for crop in crops:
            if crop.ndim != 3 or crop.shape[2] != 3 or min(crop.shape[:2]) == 0:
                raise ValueError('Nonempty RGB crop required')
            key = crop_hash(crop)
            if key not in self.values: pending.setdefault(key, crop)
        items = list(pending.items()); batch = self.c['runtime']['encoding_batch_size']
        recognizer = self.recognizer()
        for start in range(0, len(items), batch):
            rows = items[start:start+batch]
            tensor = torch.stack([recognizer.transform(Image.fromarray(im)) for _, im in rows]).cuda()
            with torch.inference_mode():
                model = recognizer.model.model
                encoded = model.encode(tensor)
                h,w = model.encoder.patch_embed.grid_size
                if encoded.shape[1:] != (h*w, self.c['visual_dimensions']) or w != self.c['visual_columns']:
                    raise ValueError('Pinned PARSeq visual shape changed')
                sequence = encoded.reshape(len(rows), h, w, -1).mean(1).cpu().numpy()
            for (key,_), value in zip(rows,sequence): self.values[key] = value.copy()
            self.encodings += len(rows); self.encoder_batches += 1
        return [self.values[crop_hash(im)] for im in crops]

    def observe(self, sample, proposals):
        rgb = rgb_for(sample); crops=[]; contexts=[]; hashes=[]
        for p in proposals:
            a,b,d,e = p['padded_bbox_original']; crop = np.ascontiguousarray(rgb[b:e,a:d])
            key = crop_hash(crop)
            if 'crop_array_sha256' in p and key != p['crop_array_sha256']: raise ValueError('Proposal/RGB mismatch')
            p['crop_array_sha256'] = key; hashes.append(key); crops.append(crop)
            box = expanded_box(p['bbox'], sample['image_hw'], self.c['context_fraction'])
            x,y,u,v = box; contexts.append(np.ascontiguousarray(rgb[y:v,x:u]))
        return self.encode(crops), self.encode(contexts)

    def read(self, sample, proposals):
        self.recognizer(); rgb = rgb_for(sample)
        for p in proposals:
            a,b,d,e = p['padded_bbox_original']; crop = np.ascontiguousarray(rgb[b:e,a:d]); key = crop_hash(crop)
            if key != p['crop_array_sha256']: raise ValueError('Native OCR crop binding changed')
            if key not in self.ocr.rows:
                result = self.ocr.model.recognize_candidate_batch([crop])[0].to_dict()
                row = dict(result, crop_array_sha256=key)
                self.ocr.rows[key] = row
                with self.ocr.output.open('a') as stream: stream.write(json.dumps(row, ensure_ascii=False)+'\n')
                self.native_new += 1
            p['text'] = self.ocr.rows[key]['raw_text']
        return proposals

    def close(self):
        if self.before is not None:
            after = state_digest(self.ocr.model.model)
            if after != self.before: raise ValueError('Frozen PARSeq tensors changed')
            self.audit.append(dict(before=self.before, after=after))
        self.ocr.release()


def extract(root, c):
    data = load(root); tick = time.monotonic(); memory = gpu_start(c, 'frozen visual evidence')
    cache = pickle.loads((Path(c['paths']['evidence_run'])/'source_ocr.pkl').read_bytes())
    bank = VisualBank(root, c, cache); directory = root/'visual'; directory.mkdir()
    for phase, samples in data.items():
        (directory/phase).mkdir()
        for index,s in enumerate(samples):
            image, context = bank.observe(s, s['pool'])
            np.savez_compressed(directory/phase/(s['sample_id'].replace(':','__')+'.npz'),
                image=np.asarray(image,np.float32), context=np.asarray(context,np.float32),
                ids=np.asarray([p['id'] for p in s['pool']]))
            if index%100 == 0: log(root,'VISUAL_EXTRACTION',split=phase,images=index+1,encodings=bank.encodings)
    bank.close()
    write(root/'visual_qualification.json',dict(frozen_state_audits=bank.audit, extra_encoder_passes=bank.encodings,
        native_ocr_calls=bank.native_new, vertical_pooling='mean over pretrained patch rows; retain all16 columns and384 channels',
        memory=gpu_end(memory,'Frozen PARSeq encoder FP32; no decoder calls',c['runtime']['encoding_batch_size'],
                       [c['runtime']['encoding_batch_size'],3,32,128],tick)))


def native_features(root, data):
    features = {}
    for phase,samples in data.items():
        features[phase]={}
        for s in samples:
            with np.load(root/'visual'/phase/(s['sample_id'].replace(':','__')+'.npz')) as z:
                if z['ids'].tolist() != [p['id'] for p in s['pool']]: raise ValueError('Feature IDs differ')
                features[phase][s['sample_id']] = dict(image=z['image'].copy(), context=z['context'].copy())
    return features


def text_tensor(texts, maximum_bytes, device):
    rows = [[BOS]+list(text.encode('utf-8')) for text in texts]
    if any(len(row)>maximum_bytes+1 for row in rows): raise ValueError('Explicit text byte limit exceeded')
    tensor = torch.full((len(rows),max(map(len,rows))), PAD, dtype=torch.long,device=device)
    for i,row in enumerate(rows): tensor[i,:len(row)] = torch.tensor(row,device=device)
    return tensor


def verifier_tensor(model, samples, proposals, visuals, c):
    if not proposals: return np.empty((0,3),np.float32)
    device = next(model.parameters()).device
    image, context = visuals
    text = text_tensor([p['text'] for p in proposals], c['maximum_bytes'], device)
    geometry = np.stack([geometry_features(p['bbox'],s['image_hw'],[]) for s,p in zip(samples,proposals)])
    return model(torch.tensor(np.asarray(image),device=device),torch.tensor(np.asarray(context),device=device),
                 text,torch.tensor(geometry,device=device))


def score_rows(model, sample, proposals, visuals, c):
    model.eval(); result=[]; size=c['runtime']['head_batch_size']
    with torch.inference_mode():
        for start in range(0,len(proposals),size):
            pp=proposals[start:start+size]
            scores=verifier_tensor(model,[sample]*len(pp),pp,
                (visuals[0][start:start+size],visuals[1][start:start+size]),c).sigmoid().cpu().numpy()
            result.extend(scores.tolist())
    return np.asarray(result,np.float32).reshape(-1,3)


def native_scores(root, model, data, features, c, phase):
    return {s['sample_id']:dict(zip([p['id'] for p in s['pool']],
        score_rows(model,s,s['pool'],(features[phase][s['sample_id']]['image'],features[phase][s['sample_id']]['context']),c)[:,-1].tolist())) for s in data[phase]}


def policies(c):
    return [dict(threshold='KEEP',margin=0.)]+[dict(threshold=t,margin=m) for t in c['policy']['thresholds'] for m in c['policy']['margins']]


def choose_policy(samples, scores, c, *, replacements):
    best=None; trials=[]
    for policy in policies(c):
        predictions={s['sample_id']:decide(s['baseline_gnn'],s['pool'],scores[s['sample_id']],
            **policy,overlap=c['compatibility_iou'],replacements=replacements) for s in samples}
        value=metric(samples,predictions,c); trials.append(dict(policy=policy,**value))
        key=(value['f1'],-value['predictions'],policy['threshold']=='KEEP')
        if best is None or key>best[0]:best=(key,policy,value)
    return best[1],best[2],trials


def verifier_stage(root,c,smoke):
    data=load(root);features=native_features(root,data);directory=root/'verifier';directory.mkdir()
    memory=gpu_start(c,'grounded verifier');tick=time.monotonic();g=c['verifier']
    examples=[];stats=dict(native=0,correct_text_contrasts=0,negative_contrasts=0)
    for sample in data['fit']:
        for index,p in enumerate(sample['pool']):
            target=choose_target(p,sample['words'],c['matching_iou'])
            variants=[p['text']];stats['native']+=1
            if target and len(target['text'].encode())<=c['maximum_bytes']:
                if target['text']!=p['text']: variants.append(target['text']);stats['correct_text_contrasts']+=1
                word=target['text'];negative=(word[:-1] if len(word)>1 else word+'X')
                variants.append(negative);stats['negative_contrasts']+=1
            for text in variants:
                q=dict(p,text=text)
                geometry=float(target is not None)
                joint=float(any(w['text'].casefold()==text.casefold() and
                    box_iou(p['bbox'],w['bbox'])>=c['matching_iou'] for w in sample['words']))
                examples.append(dict(sample=sample,index=index,proposal=q,target=[geometry,joint,joint]))
    write(directory/'training_data.json',dict(examples=len(examples),**stats))
    torch.manual_seed(c['seed']);model=GroundedVerifier(c['visual_dimensions'],g['hidden'],g['layers'],c['maximum_bytes']).cuda()
    initial=state_digest(model);optimizer=torch.optim.AdamW(model.parameters(),lr=g['learning_rate'],weight_decay=g['weight_decay'])
    generator=torch.Generator().manual_seed(c['seed']);history=[];best=None
    for epoch in range(1,(1 if smoke else g['epochs'])+1):
        model.train();losses=[]
        order=torch.randperm(len(examples),generator=generator).tolist()
        for start in range(0,len(order),c['runtime']['head_batch_size']):
            batch=[examples[j] for j in order[start:start+c['runtime']['head_batch_size']]]
            ss=[v['sample'] for v in batch];pp=[v['proposal'] for v in batch]
            image=[features['fit'][s['sample_id']]['image'][v['index']] for s,v in zip(ss,batch)]
            context=[features['fit'][s['sample_id']]['context'][v['index']] for s,v in zip(ss,batch)]
            targets=torch.tensor([v['target'] for v in batch],device='cuda')
            logits=verifier_tensor(model,ss,pp,(image,context),c)
            loss=F.binary_cross_entropy_with_logits(logits[:,0],targets[:,0])+F.binary_cross_entropy_with_logits(logits[:,2],targets[:,2])
            if (targets[:,0]>0).any():loss=loss+g['spelling_weight']*F.binary_cross_entropy_with_logits(logits[targets[:,0]>0,1],targets[targets[:,0]>0,1])
            if not loss.isfinite():raise FloatingPointError('Nonfinite grounding loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),g['gradient_clip'],error_if_nonfinite=True);optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),updates=len(losses)))
        if epoch%g['calibration_every']==0 or smoke or epoch==g['epochs']:
            scores=native_scores(root,model,data,features,c,'calibration')
            policy,value,trials=choose_policy(data['calibration'],scores,c,replacements=True)
            write(directory/f'epoch{epoch}_calibration.json',trials)
            key=(value['f1'],-value['predictions'])
            if best is None or key>best[0]:
                best=(key,dict(epoch=epoch,policy=policy,metric=value));torch.save(model.state_dict(),directory/'best.pt')
        write(directory/'history.json',history);log(root,'VERIFIER_TRAIN',**history[-1],selected=best[1] if best else None)
    if initial==state_digest(model):raise ValueError('Verifier did not train')
    torch.save(model.state_dict(),directory/'last.pt')
    model.load_state_dict(torch.load(directory/'best.pt',weights_only=True),strict=True)
    selection=dict(replacement=best[1]);cal=native_scores(root,model,data,features,c,'calibration')
    selection['addonly']=choose_policy(data['calibration'],cal,c,replacements=False)[0]
    cache=pickle.loads((Path(c['paths']['evidence_run'])/'source_ocr.pkl').read_bytes())
    confidence={phase:{s['sample_id']:{p['id']:float(np.mean(cache[p['crop_array_sha256']]['token_probabilities']))
        if cache[p['crop_array_sha256']]['token_probabilities'] else 0. for p in s['pool']} for s in data[phase]} for phase in ('calibration','evaluation','test')}
    selection['confidence']=choose_policy(data['calibration'],confidence['calibration'],c,replacements=True)[0]
    write(directory/'selection_frozen.json',selection)
    for phase in ('evaluation','test'):
        scores=native_scores(root,model,data,features,c,phase);write(directory/f'{phase}_scores.json',scores)
        for name,policy,replace,values in [('verifier_replacement',best[1]['policy'],True,scores),
            ('verifier_addonly',selection['addonly'],False,scores),('confidence_replacement',selection['confidence'],True,confidence[phase])]:
            predictions={s['sample_id']:decide(s['baseline_gnn'],s['pool'],values[s['sample_id']],**policy,
                overlap=c['compatibility_iou'],replacements=replace) for s in data[phase]}
            save_arm(root,name,phase,data[phase],predictions,c)
    model.load_state_dict(torch.load(directory/'last.pt',weights_only=True),strict=True)
    for phase in ('evaluation','test'):
        scores=native_scores(root,model,data,features,c,phase)
        predictions={s['sample_id']:decide(s['baseline_gnn'],s['pool'],scores[s['sample_id']],
            threshold=c['policy']['raw_threshold'],margin=c['policy']['raw_margin'],overlap=c['compatibility_iou'],replacements=True) for s in data[phase]}
        save_arm(root,'verifier_final',phase,data[phase],predictions,c)
    write(directory/'qualification.json',dict(initial_sha256=initial,final_sha256=state_digest(model),
        memory=gpu_end(memory,'Grounded RGB/context/text verifier + AdamW; cached frozen encoder',c['runtime']['head_batch_size'],
                       [c['runtime']['head_batch_size'],2,c['visual_columns'],c['visual_dimensions']],tick)))
    del model,optimizer,features;gc.collect();torch.cuda.empty_cache()


def run(args):
    path=Path(args.config).resolve()
    if args.mode!='approximate' or REPO/'configs/approximate' not in path.parents:raise ValueError('Explicit isolated APPROXIMATE namespace required')
    c=read(path)
    if c['reproduction_mode']!='approximate' or c['experiment_kind']!='grounded_word_recovery_v1':raise ValueError('Wrong scientific configuration')
    validate_parameters(c)
    runtime=setup(c);root=Path(args.output).resolve()
    if args.stage in ('all','prepare'):prepare(root,c,args.smoke)
    elif read(root/'resolved_config.json')!=c:raise ValueError('Resolved experiment configuration changed')
    if not (root/'runtime.json').exists():write(root/'runtime.json',runtime)
    try:
        if args.stage in ('all','extract'):extract(root,c)
        if args.stage in ('all','verifier'):verifier_stage(root,c,args.smoke)
        if args.stage in ('all','search'):
            from scripts.grounded_word_search import search_stage
            search_stage(root,c,args.smoke)
        if args.stage in ('all','editor'):
            from scripts.grounded_word_editor import editor_stage
            editor_stage(root,c,args.smoke)
        if args.stage in ('all','report'):
            from scripts.analyze_grounded_word_recovery import analyze
            analyze(root,c)
            write(root/'status.json',dict(status='SUCCESS',stage='COMPLETE',pid=os.getpid()))
    except BaseException as error:
        write(root/'failure.json',dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            oom=isinstance(error,torch.cuda.OutOfMemoryError)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',required=True,choices=['approximate'])
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--stage',default='all',choices=['all','prepare','extract','verifier','search','editor','report'])
    parser.add_argument('--smoke',action='store_true');run(parser.parse_args())
