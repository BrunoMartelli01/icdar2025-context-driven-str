"""Local-only BLIP-2 caption profiles motivated by paper Figure 4.

No PARSeq strings or labels enter prompts. Multi-question assembly is an explicit
approximation; raw answers and the assembled paragraph are both preserved.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass,field
import re

from caption.blip2_captioner import BLIP2Captioner,BLIP2ApproxConfig,canonical_torch_device


@dataclass(frozen=True)
class InstructionCaptionConfig(BLIP2ApproxConfig):
    """A separate ablation, never accepted as the BLIP2 paper checkpoint."""
    model_id: str=field()
    revision: str=field()

    def __post_init__(self):
        if self.model_id!='Salesforce/instructblip-flan-t5-xl' or not re.fullmatch('[0-9a-f]{40}',self.revision):
            raise ValueError('Explicit pinned instruction-caption ablation required')
        device=canonical_torch_device(self.device)
        if self.quantization!='none' or self.placement=='cpu_offload' and device.type!='cuda':
            raise ValueError('Unsupported instruction-caption placement/quantization')


class LocalPaperBLIP2(BLIP2Captioner):
    def __init__(self,config,snapshot,max_memory):
        super().__init__(config);self.snapshot=snapshot;self.max_memory=max_memory

    def _load(self):
        if self._model is not None:return
        import torch
        from transformers import AutoProcessor,Blip2ForConditionalGeneration,InstructBlipForConditionalGeneration
        self._processor=AutoProcessor.from_pretrained(self.snapshot,local_files_only=True)
        options={'torch_dtype':getattr(torch,self.config.dtype),'low_cpu_mem_usage':True,
                 'local_files_only':True,'output_loading_info':True}
        if self.config.placement=='cpu_offload':
            if self.config.device!='cuda:0':raise ValueError('Explicit single CUDA device required')
            options['device_map']='auto';options['max_memory']={0:self.max_memory['gpu'],
                'cpu':self.max_memory['cpu']}
        cls=InstructBlipForConditionalGeneration if isinstance(self.config,InstructionCaptionConfig) else Blip2ForConditionalGeneration
        self._model,info=cls.from_pretrained(self.snapshot,**options)
        if any(info.values()):raise ValueError('BLIP2 checkpoint key mismatch: '+str(info))
        if self.config.placement=='single_device':self._model.to(self.device)
        self._model.eval()


def generate_profile(captioner,rgb,profile,target):
    from dataclasses import replace
    from scoring.paper_image_router import caption_quality
    if (rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.dtype!=np.uint8
            or not profile['prompts'] or profile['assembly'] not in ('single','paragraph_join')
            or profile['assembly']=='single' and len(profile['prompts'])!=1):
        raise ValueError('Explicit image/prompt/assembly profile required')
    answers=[];config=captioner.config
    try:
        for prompt in profile['prompts']:
            captioner.config=replace(config,prompt=prompt,generation=profile['generation'])
            answer=captioner.caption(rgb).to_dict()
            if not answer['success'] or not answer['raw_caption'].strip():
                raise RuntimeError('BLIP2 returned an empty/failed caption; no substitution')
            answers.append({'prompt':prompt,'result':answer})
    finally:
        captioner.config=config
    texts=[a['result']['raw_caption'] for a in answers]
    # Add punctuation only; no authored scene facts, OCR appending or length padding.
    text=texts[0] if profile['assembly']=='single' else ' '.join(
        t if t[-1] in '.!?' else t+'.' for t in texts)
    return {'raw_caption':text,'answers':answers,'profile':profile,
        'quality':caption_quality(text,target['min_words'],target['max_words']),
        'image_only_captioning':True,'parseq_in_prompt':False,'ground_truth_in_caption':False,
        'text_postprocessing':'single: none; structured: join answers with punctuation only',
        'caption_family':'InstructBLIP ablation' if isinstance(config,InstructionCaptionConfig) else 'BLIP-2',
        'checkpoint_qualification':'approximate reproduction'}
