"""Pinned local Florence-2 native detailed-caption ablation; no OCR task."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from scoring.paper_image_router import caption_quality


class LocalFlorenceScene:
    def __init__(self, snapshot, device):
        self.snapshot=snapshot;self.device=device;self.model=None;self.processor=None

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        if self.model is not None:return
        self.processor=AutoProcessor.from_pretrained(self.snapshot,local_files_only=True,trust_remote_code=True)
        self.model,info=AutoModelForCausalLM.from_pretrained(self.snapshot,local_files_only=True,
            trust_remote_code=True,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Florence native checkpoint incompatibility: '+str(info))
        self.model.to(self.device).eval()

    def generate(self, rgbs, profile, target):
        import torch
        task=profile['task']
        if (not rgbs or task not in ('<DETAILED_CAPTION>','<MORE_DETAILED_CAPTION>')
                or any(r.ndim!=3 or r.shape[2]!=3 or r.dtype!=np.uint8 for r in rgbs)):
            raise ValueError('Explicit native caption-only task and RGB batch required')
        torch.cuda.init();torch.cuda.reset_peak_memory_stats(self.device);started=time.monotonic()
        self.load();load_seconds=time.monotonic()-started
        inputs=self.processor(images=[Image.fromarray(r) for r in rgbs],text=[task]*len(rgbs),return_tensors='pt')
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        torch.cuda.synchronize(self.device);generation_started=time.monotonic()
        with torch.inference_mode():outputs=self.model.generate(**inputs,**profile['generation'])
        torch.cuda.synchronize(self.device);seconds=time.monotonic()-generation_started
        raw=self.processor.batch_decode(outputs,skip_special_tokens=False)
        decoded=self.processor.batch_decode(outputs,skip_special_tokens=True)
        if len(raw)!=len(rgbs):raise RuntimeError('Florence caption output cardinality mismatch')
        results=[]
        for text,clean,rgb in zip(raw,decoded,rgbs):
            caption=self.processor.post_process_generation(clean,task=task,image_size=(rgb.shape[1],rgb.shape[0]))[task]
            if not isinstance(caption,str) or not caption.strip():raise RuntimeError('Empty native Florence caption; no substitution')
            native=dict(raw_caption=caption.strip(),raw_native_sequence=text,generated_token_ids=outputs[len(results)].tolist(),generation_config=profile['generation'],
                native_task=task,device=self.device,dtype='float32',placement='single_device',quantization='none',
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),load_time_seconds=load_seconds,
                generation_time_seconds=seconds,latency_scope='shared native batch; not individual-image latency')
            results.append(dict(raw_caption=caption.strip(),answers=[dict(prompt=task,result=native)],
                profile=profile,quality=caption_quality(caption.strip(),target['min_words'],target['max_words']),
                caption_family='Florence-2 detailed-caption ablation',checkpoint_qualification='approximate reproduction',
                native_batch_size=len(rgbs),image_only_captioning=True,parseq_in_prompt=False,
                ground_truth_in_caption=False,text_postprocessing='native caption task decoding only; no OCR appended'))
        return results

    def release(self):
        self.model=None;self.processor=None
