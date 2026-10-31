"""Native Qwen caption batching with explicit padding and shared timing."""
from __future__ import annotations

import time

import numpy as np
from PIL import Image

from caption.qwen_scene import LocalQwenScene
from scoring.paper_image_router import caption_quality


class LocalQwenBatchScene(LocalQwenScene):
    def generate(self,rgbs,profile,target):
        import torch
        if not 1<=len(rgbs)<=3 or any(r.ndim!=3 or r.shape[2]!=3 or r.dtype!=np.uint8 for r in rgbs):
            raise ValueError('Qualified caption batch requires one to three RGB images')
        torch.cuda.init();torch.cuda.reset_peak_memory_stats(self.device);began=time.monotonic()
        self.load();load_seconds=time.monotonic()-began
        self.processor.tokenizer.padding_side='left'
        chats=[self.processor.apply_chat_template([dict(role='user',content=[dict(type='image'),dict(type='text',text=profile['prompt'])])],tokenize=False,add_generation_prompt=True) for _ in rgbs]
        inputs=self.processor(text=chats,images=[Image.fromarray(r) for r in rgbs],padding=True,return_tensors='pt').to(self.device)
        torch.cuda.synchronize(self.device);started=time.monotonic()
        with torch.inference_mode():output=self.model.generate(**inputs,**profile['generation'])
        torch.cuda.synchronize(self.device);seconds=time.monotonic()-started
        generated=output[:,inputs['input_ids'].shape[1]:]
        texts=self.processor.batch_decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        if len(texts)!=len(rgbs) or any(not text.strip() for text in texts):raise RuntimeError('Empty/incorrect-cardinality native batch output')
        grids=inputs['image_grid_thw'].tolist()
        if len(grids)!=len(rgbs):raise RuntimeError('Expected one native image grid per caption')
        results=[]
        for i,text in enumerate(texts):
            text=text.strip()
            native=dict(raw_caption=text,generated_token_ids=generated[i].tolist(),generation_config=profile['generation'],
                prompt=profile['prompt'],native_chat_prompt=chats[i],image_grid_thw=[grids[i]],batch_image_grids=grids,
                device=self.device,dtype='bfloat16',placement=self.placement,quantization='none',
                hf_device_map={str(k):str(v) for k,v in getattr(self.model,'hf_device_map',{}).items()},
                peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                load_time_seconds=load_seconds,generation_time_seconds=seconds,latency_scope='shared native caption batch; do not sum repeated per-image times',
                min_pixels=self.min_pixels,max_pixels=self.max_pixels,attention_implementation='sdpa',padding_side='left')
            results.append(dict(raw_caption=text,answers=[dict(prompt=profile['prompt'],result=native)],profile=profile,
                quality=caption_quality(text,target['min_words'],target['max_words']),caption_family='Qwen2-VL-7B caption ablation',
                checkpoint_qualification='approximate reproduction',native_batch_size=len(rgbs),image_only_captioning=True,
                parseq_in_prompt=False,ground_truth_in_caption=False,text_postprocessing='native generated suffix decoding; special tokens only removed'))
        return results
