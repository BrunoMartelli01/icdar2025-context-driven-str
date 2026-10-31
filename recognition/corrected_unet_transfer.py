"""Prediction-only deployment of the frozen TotalText corrected U-Net expert.

The model, candidate budget, image preprocessing, graph and edit policies come
from authenticated source configurations. No annotation is an inference input.
"""
import copy
import gc
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from localize.occlusion_four_tests import instance_pool,features,ocr_features,decode_instance_labels
from localize.teacher_student_tests import CandidateScorer,compatible_selection
from localize.word_graph_recovery import HypothesisScorer,addition_features,hypothesis_adjacency,preserve_and_add
from models.adaspot_gap import AdaSpotGapAGUNet
from models.grounded_word_recovery import GroundedVerifier,VisualEditTransducer
from recognition.grounded_word_recovery import decide,expanded_box
from recognition.group_word_selection import reading_confidence
from recognition.parseq import ParseqApproxConfig,ParseqApproxRecognizer,PARSEQ_ARCHIVE_SOURCE
from recognition.unet_corrected_token_interfaces import native_observation
from scripts.evaluate_cached_totaltext_padding import crop_hash
from scripts.evaluate_totaltext_v4_image_frequency_val_parseq import _tensor
from scripts.train_totaltext_v4_image_frequency_ablation import _transform
from scripts.run_grounded_word_recovery import score_rows
from scripts.grounded_word_editor import edited_candidates

read=lambda p:json.loads(Path(p).read_text())


class CorrectedUNetTransfer:
    def __init__(self,config):
        self.c=config;paths=config['paths'];self.paths=paths;self.readings={};self.encoded={}
        self.old=read(Path(paths['corrected_source'])/'resolved_config.json')
        self.graph_config=read(Path(paths['graph_source'])/'resolved_config.json')
        self.occlusion=read(Path(paths['occlusion_source'])/'resolved_config.json')
        self.contract=read(paths['unet_contract']);self.device=config['device']
        self.transform=_transform(self.contract['resolved_preprocessing'],'letterbox',train=False)
        saved=torch.load(paths['unet_checkpoint'],map_location='cpu',weights_only=False)
        if saved['run_contract']!=self.contract:raise ValueError('Qualified U-Net contract mismatch')
        self.unet=AdaSpotGapAGUNet('control',skip_floor=self.contract['resolved_config']['skip_floor'],initial_gap_bias=self.occlusion['initial_gap_bias'])
        self.unet.load_state_dict(saved['model_state_dict'],strict=True);del saved
        self.unet.requires_grad_(False).eval().to(self.device)
        ck=torch.load(Path(paths['occlusion_source'])/'selector_pre.pt',map_location='cpu',weights_only=True)
        self.pre=nn.Sequential(nn.Linear(ck['features'],32),nn.ReLU(),nn.Linear(32,1));self.pre.load_state_dict(ck['state_dict'],strict=True)
        self.pre.eval();self.pre_mean=ck['mean'];self.pre_scale=ck['scale']
        ck=torch.load(Path(paths['teacher_source'])/'selector/pointwise.pt',map_location='cpu',weights_only=True)
        self.filter=CandidateScorer(ck['mean'].numel(),self.graph_config['filter']['hidden']);self.filter.load_state_dict(ck['state_dict'],strict=True)
        self.filter.eval();self.filter_mean=ck['mean'];self.filter_scale=ck['scale']
        ck=torch.load(Path(paths['graph_source'])/'additions/gnn/best.pt',map_location='cpu',weights_only=True)
        g=self.graph_config['additions'];self.graph=HypothesisScorer(ck['mean'].numel(),g['hidden'],g['layers'],message_passing=True)
        self.graph.load_state_dict(ck['state_dict'],strict=True);self.graph.eval();self.graph_mean=ck['mean'];self.graph_scale=ck['scale']
        self.graph_policy=read(Path(paths['graph_source'])/'additions/gnn/selection_frozen.json')
        self.feature_size=read(Path(paths['contour_source'])/'resolved_config.json')['contour']['feature_size']
        v=self.old['verifier'];e=self.old['editor'];source=Path(paths['corrected_source'])
        self.verifier=GroundedVerifier(self.old['visual_dimensions'],v['hidden'],v['layers'],self.old['maximum_bytes']).to(self.device).eval()
        self.verifier.load_state_dict(torch.load(source/'verifier/best.pt',map_location='cpu',weights_only=True),strict=True)
        self.editor=VisualEditTransducer(self.old['visual_dimensions'],e['hidden'],e['layers'],self.old['maximum_bytes'],visual=False).to(self.device).eval()
        self.editor.load_state_dict(torch.load(source/'editor/text/best.pt',map_location='cpu',weights_only=True),strict=True)
        self.vpolicy=read(source/'verifier/selection_frozen.json')['replacement']['policy'];self.epolicy=read(source/'editor/text/selection_frozen.json')['policy']
        self.recognizer=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=Path(paths['parseq_source']),checkpoint=Path(paths['parseq_checkpoint']),
            runtime_source_archive=Path(paths['parseq_archive']),runtime_source_type=PARSEQ_ARCHIVE_SOURCE,device=self.device))
        for m in [self.pre,self.filter,self.graph,self.verifier,self.editor,self.recognizer.model]:m.requires_grad_(False).eval()
        self.counters=dict(native_crops=0,visual_crops=0,images=0)

    def native_read(self,rgb,pool):
        pending={}
        for p in pool:
            a,b,c,d=p['padded_bbox_original'];crop=np.ascontiguousarray(rgb[b:d,a:c]);key=crop_hash(crop)
            if 'crop_array_sha256' in p and p['crop_array_sha256']!=key:raise ValueError('Changed proposal RGB')
            p['crop_array_sha256']=key
            if key not in self.readings:pending[key]=crop
        pairs=list(pending.items());batch=self.c['parseq_batch_size']
        for start in range(0,len(pairs),batch):
            group=pairs[start:start+batch];decoded=self.recognizer.recognize_candidate_batch([im for _,im in group])
            if len(decoded)!=len(group):raise ValueError('Incomplete native PARSeq batch')
            for (key,_),value in zip(group,decoded):self.readings[key]=value.to_dict()
            self.counters['native_crops']+=len(group)
        for p in pool:p['text']=self.readings[p['crop_array_sha256']]['raw_text']

    @torch.inference_mode()
    def encode(self,crops):
        pending={crop_hash(im):im for im in crops if crop_hash(im) not in self.encoded};pairs=list(pending.items())
        size=self.old['runtime']['encoding_batch_size']
        for start in range(0,len(pairs),size):
            group=pairs[start:start+size];tensor=torch.stack([self.recognizer.transform(Image.fromarray(im)) for _,im in group]).to(self.device)
            model=self.recognizer.model.model;value=model.encode(tensor);h,w=model.encoder.patch_embed.grid_size
            if value.shape[1:]!=(h*w,self.old['visual_dimensions']):raise ValueError('Native PARSeq encoding shape differs')
            value=value.reshape(len(group),h,w,-1).mean(1).cpu().numpy()
            for (key,_),v in zip(group,value):self.encoded[key]=v.copy()
            self.counters['visual_crops']+=len(group)
        return [self.encoded[crop_hash(im)] for im in crops]

    def correct(self,rgb,pool,prefix):
        if not pool:return [],dict(verifier_scores={},edited=[],pool=[])
        crops=[];contexts=[]
        for p in pool:
            a,b,c,d=p['padded_bbox_original'];crops.append(np.ascontiguousarray(rgb[b:d,a:c]))
            a,b,c,d=expanded_box(p['bbox'],rgb.shape[:2],self.old['context_fraction']);contexts.append(np.ascontiguousarray(rgb[b:d,a:c]))
        visual=dict(image=np.asarray(self.encode(crops),np.float32),context=np.asarray(self.encode(contexts),np.float32))
        sample=dict(image_hw=list(rgb.shape[:2]),pool=pool)
        values=score_rows(self.verifier,sample,pool,(visual['image'],visual['context']),self.old)[:,-1]
        scores=dict(zip([p['id'] for p in pool],map(float,values)))
        native=decide(prefix,pool,scores,**self.vpolicy,overlap=self.old['compatibility_iou'],replacements=True)
        pp,extra,details=edited_candidates(self.editor,self.verifier,sample,visual,self.old,'text',native);scores.update(extra)
        final=decide(native,pp,scores,**self.epolicy,overlap=self.old['compatibility_iou'],replacements=True)
        by={p['id']:p for p in pool};outputs=[]
        for p in final:
            parent=by[p.get('source_proposal_id',p['id'])];reading=self.readings[parent['crop_array_sha256']]
            outputs.append(native_observation(p,parent,reading,reading_confidence(reading)['confidence_balanced'],scores[p['id']]))
        return outputs,dict(verifier_scores=scores,edited=details,pool=pool,baseline_ids=[p['id'] for p in prefix])

    @torch.inference_mode()
    def predict(self,rgb):
        e=self.unet.forward_evidence(_tensor(rgb,self.transform).to(self.device));prob=e['words'][0].sigmoid().cpu().numpy()
        field=F.interpolate(torch.cat([e['features'],e['words'].sigmoid()],1),size=(self.feature_size,self.feature_size),mode='bilinear',align_corners=False)[0,:64].cpu().numpy()
        from segmentation.totaltext_image_frequency_augment import letterbox_canvas_mask
        canvas=np.asarray(letterbox_canvas_mask(tuple(rgb.shape[:2][::-1]),native_size=prob.shape[-1]),bool)
        pp=self.contract['resolved_config']['postprocessing'];labels=decode_instance_labels(prob,None,canvas,pp,None,'cut')
        pool=instance_pool(prob[0],labels,rgb,self.occlusion['hypotheses'])['pool']
        f={p['id']:features(p,rgb,prob[0],text_threshold=pp['text_threshold']) for p in pool}
        for p in pool:
            a,b,c,d=p['padded_bbox_original'];p['crop_array_sha256']=crop_hash(np.ascontiguousarray(rgb[b:d,a:c]))
        if not pool:self.counters['images']+=1;return [],prob,dict(pool=[],baseline_ids=[],verifier_scores={},edited=[])
        x=torch.tensor([f[p['id']] for p in pool],dtype=torch.float32)
        values=self.pre((x-self.pre_mean)/self.pre_scale)[:,0].sigmoid().tolist();scores=dict(zip([p['id'] for p in pool],values))
        chosen=set();hashes=set()
        for p in sorted(pool,key=lambda p:(-scores[p['id']],p['id'])):
            key=p['crop_array_sha256']
            if key not in hashes and len(hashes)>=self.graph_config['budget']:continue
            hashes.add(key);chosen.add(p['id'])
        pool=[p for p in pool if p['id'] in chosen];self.native_read(rgb,pool)
        x=torch.tensor([f[p['id']]+ocr_features(self.readings[p['crop_array_sha256']]) for p in pool])
        scores=self.filter((x-self.filter_mean)/self.filter_scale).sigmoid().tolist()
        filt=self.graph_config['filter'];baseline=compatible_selection(pool,dict(zip([p['id'] for p in pool],scores)),filt['threshold'],filt['overlap'])
        visual=[];side=prob.shape[-1]
        for p in pool:
            box=np.asarray(p['bbox_native'],float)*field.shape[1]/side;a,b=np.floor(box[:2]).astype(int);c,d=np.ceil(box[2:]).astype(int)
            patch=field[:,max(0,b):max(b+1,d),max(0,a):max(a+1,c)].reshape(64,-1)
            if not patch.shape[1]:raise ValueError('Empty native U-Net feature crop')
            a,b,c,d=p['padded_bbox_original'];gray=cv2.cvtColor(rgb[b:d,a:c],cv2.COLOR_RGB2GRAY)
            bitmap=cv2.resize(gray,(8,4),interpolation=cv2.INTER_AREA).astype(np.float32).ravel()/255
            visual.append(np.r_[patch.mean(1),patch.std(1),bitmap])
        g=self.graph_config['additions'];x=addition_features(pool,baseline,f,self.readings,visual,text_bins=g['text_bins'],overlap=filt['overlap'])
        adjacency=hypothesis_adjacency(pool,self.readings,neighbors=g['neighbors'],maximum_distance=g['maximum_distance'])
        values=self.graph((torch.from_numpy(x)-self.graph_mean)/self.graph_scale,torch.from_numpy(adjacency)).sigmoid().tolist()
        if self.graph_policy['threshold']!='KEEP':baseline=preserve_and_add(baseline,pool,dict(zip([p['id'] for p in pool],values)),threshold=self.graph_policy['threshold'],overlap=filt['overlap'])
        outputs,trace=self.correct(rgb,pool,baseline);self.counters['images']+=1
        return outputs,prob,trace

    def clear_image_cache(self):
        self.readings.clear();self.encoded.clear()

    def close(self):
        self.recognizer.release()
        for name in ['unet','recognizer','verifier','editor','pre','filter','graph']:setattr(self,name,None)
        self.clear_image_cache();gc.collect();torch.cuda.empty_cache()
