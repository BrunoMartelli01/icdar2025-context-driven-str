"""Native pinned Qwen2-VL caption-quality ablation, separate from paper BLIP-2."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from scoring.paper_image_router import caption_quality


class LocalQwenScene:
    def __init__(self,snapshot,device,min_pixels,max_pixels,placement='single_device',gpu_weight_budget=None,cpu_weight_budget=None):
        self.snapshot=snapshot;self.device=device;self.min_pixels=min_pixels;self.max_pixels=max_pixels
        self.placement=placement;self.gpu_weight_budget=gpu_weight_budget;self.cpu_weight_budget=cpu_weight_budget
        self.model=None;self.processor=None

    def load(self):
        import torch
        from transformers import AutoProcessor,Qwen2VLForConditionalGeneration
        if self.model is not None:return
        self.processor=AutoProcessor.from_pretrained(self.snapshot,local_files_only=True,
            min_pixels=self.min_pixels,max_pixels=self.max_pixels)
        options=dict(local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation='sdpa',low_cpu_mem_usage=True,output_loading_info=True)
        if self.placement=='cpu_offload':
            if self.device!='cuda:0' or not self.gpu_weight_budget or not self.cpu_weight_budget:
                raise ValueError('Explicit CUDA/CPU weight budgets are required for offload')
            options.update(device_map='auto',max_memory={0:self.gpu_weight_budget,'cpu':self.cpu_weight_budget})
        elif self.placement!='single_device':raise ValueError('Unknown Qwen device placement')
        self.model,info=Qwen2VLForConditionalGeneration.from_pretrained(self.snapshot,**options)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('Qwen2-VL native checkpoint incompatibility: '+str(info))
        if self.placement=='single_device':self.model.to(self.device)
        else:
            devices=list(self.model.hf_device_map.values())
            if 'cpu' not in devices or not any(isinstance(v,int) or str(v).startswith('cuda') for v in devices) or 'disk' in devices:
                raise ValueError('Expected explicit CUDA/CPU dispatch without disk offload')
        self.model.eval()

    def generate(self,rgbs,profile,target):
        import torch
        if (len(rgbs)!=1 or any(r.ndim!=3 or r.shape[2]!=3 or r.dtype!=np.uint8 for r in rgbs)
                or not profile['prompt']):
            raise ValueError('Qualified Qwen2-VL captioning requires one RGB image and explicit prompt')
        torch.cuda.init();torch.cuda.reset_peak_memory_stats(self.device);start=time.monotonic()
        self.load();load_seconds=time.monotonic()-start
        messages=[dict(role='user',content=[dict(type='image'),dict(type='text',text=profile['prompt'])])]
        chat=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[chat],images=[Image.fromarray(rgbs[0])],padding=True,return_tensors='pt').to(self.device)
        torch.cuda.synchronize(self.device);began=time.monotonic()
        with torch.inference_mode():output=self.model.generate(**inputs,**profile['generation'])
        torch.cuda.synchronize(self.device);seconds=time.monotonic()-began
        generated=output[:,inputs['input_ids'].shape[1]:]
        text=self.processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
        if not text:raise RuntimeError('Empty native Qwen2-VL caption; no substitution')
        native=dict(raw_caption=text,generated_token_ids=generated[0].tolist(),generation_config=profile['generation'],
            prompt=profile['prompt'],native_chat_prompt=chat,image_grid_thw=inputs['image_grid_thw'].tolist(),
            device=self.device,dtype='bfloat16',placement=self.placement,quantization='none',
            hf_device_map={str(k):str(v) for k,v in getattr(self.model,'hf_device_map',{}).items()},
            peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
            load_time_seconds=load_seconds,generation_time_seconds=seconds,latency_scope='native single-image caption call',
            min_pixels=self.min_pixels,max_pixels=self.max_pixels,attention_implementation='sdpa')
        return [dict(raw_caption=text,answers=[dict(prompt=profile['prompt'],result=native)],profile=profile,
            quality=caption_quality(text,target['min_words'],target['max_words']),caption_family='Qwen2-VL-7B caption ablation',
            checkpoint_qualification='approximate reproduction',native_batch_size=1,image_only_captioning=True,
            parseq_in_prompt=False,ground_truth_in_caption=False,text_postprocessing='native generated suffix decoding; special tokens only removed')]

    def release(self):
        self.model=None;self.processor=None
