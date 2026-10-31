"""Frozen PARSeq context-order probes and unchanged-pixel BLIP2 context swaps."""
from __future__ import annotations

import gc
import time

import numpy as np
from PIL import Image
import torch

from recognition.parseq import ParseqApproxConfig,ParseqApproxRecognizer
from recognition.grounded_context_runtime import FrozenLocalVerifier
from recognition.creative_context_signals import glyph_patch
from scripts.contextual_lm_verifier_runtime import state_digest
from scripts.crop_clip_runtime import _cuda_peak,array_sha


class FrozenPermutationProbe:
    def __init__(self,paths,settings,device):
        self.settings=settings;self.device=device;torch.cuda.reset_peak_memory_stats(device)
        self.adapter=ParseqApproxRecognizer(ParseqApproxConfig(runtime_source=paths['source'],checkpoint=paths['checkpoint'],
            runtime_source_type='github_commit_archive',runtime_source_archive=paths['archive'],device=device))
        self.wrapper=self.adapter.model;self.model=self.wrapper.model
        self.wrapper.requires_grad_(False).eval();self.before=state_digest(self.wrapper);self.qualification=None
        self.metadata={'provenance':self.adapter._provenance,'precision':'float32','device':device,'batch_size':1,
            'processed_shape':[1,3,*self.wrapper.hparams.img_size],'model_state_sha256_before':self.before,
            'optimizer_steps':0,'context_orders':['forward','reverse_with_null_EOS','odd_even','null_context'],
            'attention_is_not_character_ground_truth':True}

    def word(self,rgb,row):
        text=row['raw_text'];n=len(text)
        if not text: return {'candidate_id':row['candidate_id'],'text':text,'abstention':'empty_prediction'}
        if n>self.model.max_label_length:
            # A native decode without EOS can emit max_label_length+1 symbols.
            # It remains an original prediction; do not truncate it or invent probe scores.
            return {'candidate_id':row['candidate_id'],'text':text,'abstention':'prediction_exceeds_conditional_probe_capacity'}
        a,b,c,d=row['bbox'];crop=rgb[b:d,a:c]
        if array_sha(crop)!=row['crop_rgb_sha256']: raise ValueError('Original PARSeq crop changed')
        tensor=self.adapter.transform(Image.fromarray(crop)).unsqueeze(0).to(self.device)
        tokens=self.wrapper.tokenizer.encode([text],device=self.device)
        perm=[list(range(n+2)),[0,n+1,*range(n,0,-1)], [0,*range(1,n+1,2),*range(2,n+1,2),n+1]]
        distributions=[];attention=[];expected=None
        with torch.inference_mode():
            memory=self.model.encode(tensor)
            for order in perm:
                cm,qm=self.wrapper.generate_attn_masks(torch.tensor(order,device=self.device))
                captured=[]
                handle=self.model.decoder.layers[-1].cross_attn.register_forward_hook(lambda module,args,out:captured.append(out[1].detach()))
                try:
                    logits=self.model.head(self.model.decode(tokens[:,:-1],memory,cm,tgt_query_mask=qm))
                finally: handle.remove()
                if len(captured)!=1 or captured[0] is None: raise RuntimeError('PARSeq query cross-attention capture differs')
                if self.qualification is None:
                    plain=self.model.head(self.model.decode(tokens[:,:-1],memory,cm,tgt_query_mask=qm))
                    error=float((plain-logits).abs().max())
                    if not torch.equal(plain,logits): raise RuntimeError('Read-only attention capture altered logits')
                    self.qualification={'status':'PASS','maximum_logit_difference':error,'native_generate_attn_masks':True}
                distributions.append(logits.softmax(-1)[0].cpu().numpy());attention.append(captured[0][0].cpu().numpy())
            bos=tokens[:,:1];query=self.model.pos_queries[:,:n+1]
            logits=self.model.head(self.model.decode(bos,memory,tgt_query=query))
            distributions.append(logits.softmax(-1)[0].cpu().numpy())
        probs=np.asarray(distributions);target=tokens[0,1:].cpu().numpy()
        picked=np.take_along_axis(probs,target[None,:,None],axis=2)[...,0]
        if not np.isfinite(probs).all(): raise RuntimeError('Nonfinite permutation distributions')
        # First (forward) context supplies an attention-guided spatial proxy.
        grid=tuple(self.model.encoder.patch_embed.grid_size);weights=attention[0][:n]
        if weights.shape[1]!=grid[0]*grid[1]: raise ValueError('Attention/image patch grid mismatch')
        horizontal=weights.reshape(n,*grid).sum(axis=1);horizontal/=horizontal.sum(axis=1,keepdims=True)
        centers=(horizontal*(np.arange(grid[1])+.5)).sum(axis=1)/grid[1]
        monotonic=bool(n==1 or np.all(np.diff(centers)>self.settings['minimum_center_gap']))
        characters=[]
        if monotonic:
            boundaries=np.r_[0.,(centers[1:]+centers[:-1])/2,1.]
            for i,char in enumerate(text):
                x1=max(0,min(crop.shape[1]-1,int(np.floor(boundaries[i]*crop.shape[1]))))
                x2=max(x1+1,min(crop.shape[1],int(np.ceil(boundaries[i+1]*crop.shape[1]))))
                box=[a+x1,b,a+x2,d]
                characters.append({'position':i,'character':char,'probability':float(picked[0,i]),'bbox':box,
                    'attention_center':float(centers[i]),'vector':glyph_patch(rgb,box,self.settings['glyph_shape'])})
        return {'candidate_id':row['candidate_id'],'text':text,'target_probabilities':picked.tolist(),
            'argmax_ids':probs.argmax(axis=-1).tolist(),'target_ids':target.tolist(),'attention_centers':centers.tolist(),
            'attention_monotonic':monotonic,'characters':characters,'crop_rgb_sha256':array_sha(crop),
            'ground_truth_used':False,'original_word_output_changed':False}

    def infer(self,rgb,rows):
        started=time.monotonic();words=[self.word(rgb,r) for r in rows]
        return {'words':words,'seconds':time.monotonic()-started,'ground_truth_used':False,'base_outputs_changed':False}

    def finish(self):
        after=state_digest(self.wrapper)
        if after!=self.before or self.qualification is None: raise RuntimeError('Unqualified or mutated PARSeq probe')
        return {**self.metadata,'model_state_sha256_after':after,'attention_capture_qualification':self.qualification,
            'cuda_peak':_cuda_peak(self.device)}

    def close(self):
        self.adapter=None;self.model=None;self.wrapper=None;gc.collect();torch.cuda.empty_cache()


class FrozenContextSwap(FrozenLocalVerifier):
    """Same verified exact-word likelihood path; only surrounding pixels change."""
    def infer(self,views,job,question):
        return self.verify(views,job,question)
