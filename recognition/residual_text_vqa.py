"""Frozen BLIP2 residual-text verification, an APPROXIMATE routing experiment."""
from __future__ import annotations

import gc
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageFilter
import torch

from scripts.contextual_lm_verifier_runtime import answer_log_probabilities, state_digest
from scripts.crop_clip_runtime import _cuda_peak, array_sha


def residual_views(rgb, boxes, *, mask_value, blur_fraction):
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or not 0<=mask_value<=255 or blur_fraction<=0:
        raise ValueError('Explicit original RGB/mask/blur parameters required')
    hard=rgb.copy();blurred=rgb.copy();height,width=rgb.shape[:2];union=np.zeros((height,width),bool)
    # Every blurred crop comes from the same original, independent of box order.
    for box in boxes:
        if len(box)!=4 or any(type(v)is not int for v in box):raise ValueError('Integer exclusive original boxes required')
        x1,y1,x2,y2=box
        if not 0<=x1<x2<=width or not 0<=y1<y2<=height:raise ValueError('Out-of-image residual box')
        union[y1:y2,x1:x2]=True
    # Global blur avoids box-order-dependent repeated filtering at overlaps.
    radius=max(1.,blur_fraction*min(width,height))
    blurred_all=np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=radius)))
    hard[union]=mask_value;blurred[union]=blurred_all[union]
    return {'original':rgb.copy(),'hard_mask':hard,'blur_mask':blurred}, {
        'masked_pixels':int(union.sum()),'image_pixels':int(union.size),
        'mask_fraction':float(union.mean()),'gaussian_blur_radius':radius,
        'mask_value':mask_value,'boxes':boxes,'views_rgb_sha256':{
            'original':array_sha(rgb),'hard_mask':array_sha(hard),'blur_mask':array_sha(blurred)},
        'ground_truth_used':False,'masking_may_cover_unrecognized_text_inside_boxes':True}


class FrozenResidualVQA:
    """Exact staged BLIP2 forward with reused image query embeddings.

    Before scoring, compare staged Yes/No logits to the native model forward on
    the first image/prompt. No generation, gradients, truncation, offload or AMP.
    """
    def __init__(self, snapshot, *, device, plain_question, context_template):
        from transformers import AutoProcessor, Blip2ForConditionalGeneration
        started=time.monotonic();self.device=device
        self.processor=AutoProcessor.from_pretrained(str(snapshot),local_files_only=True)
        torch.cuda.reset_peak_memory_stats(device)
        self.model,loading=Blip2ForConditionalGeneration.from_pretrained(str(snapshot),
            local_files_only=True,torch_dtype=torch.float32,low_cpu_mem_usage=True,output_loading_info=True,
            use_safetensors=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError('BLIP2 checkpoint mismatch: '+str(loading))
        if self.model.config.use_decoder_only_language_model or getattr(self.model.config,'image_token_index',None)is not None:
            raise ValueError('Only pinned legacy BLIP2 FLAN-T5 conditional-query path is qualified')
        self.model.requires_grad_(False).eval().to(device)
        self.before=state_digest(self.model);self.plain_question=plain_question;self.context_template=context_template
        answer=self.processor.tokenizer(['Yes','No'],add_special_tokens=True,truncation=False)['input_ids']
        self.labels=torch.full((2,max(map(len,answer))),-100,dtype=torch.long,device=device)
        for i,ids in enumerate(answer):self.labels[i,:len(ids)]=torch.tensor(ids,device=device)
        self.provenance={'snapshot':str(snapshot),'revision':Path(snapshot).name,'loading_info':loading,
            'device':device,'precision':'float32','model_state_sha256_before':self.before,'optimizer_steps':0,
            'models_trained':False,'answer_ids':answer,'answer_score':'sum_logprob_with_EOS; normalized over Yes/No',
            'vision_batch_size':1,'language_batch_size':2,'query_tokens':self.model.config.num_query_tokens,
            'native_forward_check':None,'load_seconds':time.monotonic()-started,'cuda_peak':_cuda_peak(device)}
        self.blank_queries=None;self.blank_results={};self.maximum_tokens=0

    def queries(self,rgb):
        pixel=self.processor.image_processor(images=Image.fromarray(rgb),return_tensors='pt')['pixel_values'].to(self.device)
        with torch.inference_mode():
            emb=self.model.vision_model(pixel_values=pixel,return_dict=True).last_hidden_state
            attention=torch.ones(emb.shape[:-1],dtype=torch.long,device=self.device)
            query=self.model.qformer(query_embeds=self.model.query_tokens.expand(emb.shape[0],-1,-1),
                encoder_hidden_states=emb,encoder_attention_mask=attention,return_dict=True).last_hidden_state
            projected=self.model.language_projection(query)
        if not torch.isfinite(projected).all():raise RuntimeError('Nonfinite BLIP2 visual queries')
        return projected,pixel

    def answer(self,queries,prompt):
        encoded=self.processor.tokenizer(prompt,add_special_tokens=True,return_tensors='pt',truncation=False)
        tokens=encoded['input_ids'].shape[1];self.maximum_tokens=max(self.maximum_tokens,tokens)
        if tokens+self.model.config.num_query_tokens>512:
            raise ValueError('BLIP2 question exceeds512 total language inputs; refusing truncation')
        ids=encoded['input_ids'].to(self.device);attention=encoded['attention_mask'].to(self.device)
        with torch.inference_mode():
            text=self.model.language_model.get_input_embeddings()(ids)
            combined=torch.cat([queries,text],dim=1).repeat_interleave(2,dim=0)
            mask=torch.cat([torch.ones(queries.shape[:-1],dtype=torch.long,device=self.device),attention],dim=1).repeat_interleave(2,dim=0)
            logits=self.model.language_model(inputs_embeds=combined,attention_mask=mask,labels=self.labels,
                use_cache=False,return_dict=True).logits
            logp=answer_log_probabilities(logits,self.labels)
            probability=float(torch.softmax(logp,dim=0)[0])
        return {'yes_probability':probability,'yes_logprob':float(logp[0]),'no_logprob':float(logp[1]),
            'prompt':prompt,'tokens':tokens},logits,encoded

    def score(self,views,caption):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('BLIP2 must remain frozen/eval')
        prompts={'plain':self.plain_question,'context':self.context_template.format(caption=caption)}
        results={}
        if self.blank_queries is None:
            self.blank_queries,_=self.queries(np.full((224,224,3),128,np.uint8))
        for name,rgb in views.items():
            query,pixel=self.queries(rgb);results[name]={}
            for condition,prompt in prompts.items():
                result,logits,encoded=self.answer(query,prompt);results[name][condition]=result
                if self.provenance['native_forward_check']is None:
                    with torch.inference_mode():
                        native=self.model(pixel_values=pixel.repeat_interleave(2,dim=0),
                            input_ids=encoded['input_ids'].to(self.device).repeat_interleave(2,dim=0),
                            attention_mask=encoded['attention_mask'].to(self.device).repeat_interleave(2,dim=0),
                            labels=self.labels,return_dict=True).logits
                    error=float((native-logits).abs().max())
                    if not torch.allclose(native,logits,rtol=2e-4,atol=2e-4):
                        raise RuntimeError('Staged BLIP2 differs from native forward: '+str(error))
                    self.provenance['native_forward_check']={'status':'PASS','max_absolute_logit_error':error,
                        'rtol':2e-4,'atol':2e-4,'source':'transformers.Blip2ForConditionalGeneration.forward'}
        results['blank']={}
        for condition,prompt in prompts.items():
            if prompt not in self.blank_results:self.blank_results[prompt]=self.answer(self.blank_queries,prompt)[0]
            results['blank'][condition]=self.blank_results[prompt]
        return results

    def finish(self):
        after=state_digest(self.model)
        if after!=self.before:raise RuntimeError('BLIP2 state changed')
        return {**self.provenance,'model_state_sha256_after':after,'cuda_peak':_cuda_peak(self.device),
                'maximum_question_tokens':self.maximum_tokens}

    def close(self):
        self.model=None;self.blank_queries=None;gc.collect();torch.cuda.empty_cache()
