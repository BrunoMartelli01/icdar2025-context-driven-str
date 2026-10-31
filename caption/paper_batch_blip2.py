"""Native batched BLIP-2 decoding with explicit shared-batch memory accounting."""
from __future__ import annotations

from dataclasses import replace
import time

import numpy as np
from PIL import Image

from caption.blip2_captioner import CaptionResult, cpu_peak_rss_bytes
from scoring.paper_image_router import caption_quality


def generate_batch(captioner, rgbs, profile, target):
    import torch
    if (not rgbs or profile['assembly'] != 'single' or len(profile['prompts']) != 1
            or any(rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8 for rgb in rgbs)):
        raise ValueError('Explicit nonempty RGB batch and single native prompt required')
    if captioner.config.placement != 'single_device' or captioner.config.device != 'cuda:0':
        raise ValueError('Batched qualification requires explicit single CUDA device')
    config = captioner.config
    captioner.config = replace(config, prompt=profile['prompts'][0], generation=profile['generation'])
    torch.cuda.init();torch.cuda.reset_peak_memory_stats(captioner.device)
    started = time.monotonic(); load_started = started
    try:
        captioner._load(); load_time = time.monotonic()-load_started
        inputs = captioner._processor(images=[Image.fromarray(rgb) for rgb in rgbs],
            text=[profile['prompts'][0]]*len(rgbs),return_tensors='pt')
        inputs = inputs.to(device=captioner.device,dtype=getattr(torch,config.dtype))
        torch.cuda.synchronize(captioner.device);generation_started=time.monotonic()
        with torch.inference_mode():outputs=captioner._model.generate(**inputs,**profile['generation'])
        torch.cuda.synchronize(captioner.device);generation_time=time.monotonic()-generation_started
        texts=captioner._processor.batch_decode(outputs,skip_special_tokens=True)
        if len(texts)!=len(rgbs) or any(not text.strip() for text in texts):
            raise RuntimeError('Empty/incorrect-cardinality native caption batch; no substitution')
        elapsed=time.monotonic()-started
        from caption.blip2_captioner import stable_hash
        results=[]
        for text in texts:
            text=text.strip()
            native=CaptionResult(raw_caption=text,model_id=config.model_id,revision=config.revision,
                prompt_hash=stable_hash(profile['prompts'][0]),generation_config=dict(profile['generation']),
                generation_config_hash=stable_hash(profile['generation']),dtype=config.dtype,
                device=str(captioner.device),placement=config.placement,quantization=config.quantization,
                latency_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(captioner.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(captioner.device),load_time_seconds=load_time,
                generation_time_seconds=generation_time,total_latency_seconds=elapsed,
                cpu_peak_rss_bytes=cpu_peak_rss_bytes(),device_placement_summary=captioner._device_placement_summary(),
                hardware_deviation=f'HARDWARE_DEVIATION: {config.dtype} BLIP-2, batch {len(rgbs)}',
                latency_scope='shared batch; repeated per image, do not sum as individual call times').to_dict()
            results.append(dict(raw_caption=text,answers=[dict(prompt=profile['prompts'][0],result=native)],
                profile=profile,quality=caption_quality(text,target['min_words'],target['max_words']),
                image_only_captioning=True,parseq_in_prompt=False,ground_truth_in_caption=False,
                text_postprocessing='single: none',caption_family='BLIP-2',checkpoint_qualification='approximate reproduction',
                native_batch_size=len(rgbs)))
        return results
    finally:captioner.config=config
