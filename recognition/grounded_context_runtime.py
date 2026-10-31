"""Offline frozen Florence inventory and local BLIP2 visual evidence runtimes."""
from __future__ import annotations

import gc
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from recognition.residual_text_vqa import FrozenResidualVQA
from scripts.contextual_lm_verifier_runtime import state_digest
from scripts.crop_clip_runtime import _cuda_peak, array_sha


def florence_parser_input(processor, tokens, decoder_start_id):
    """Exclude only the generated decoder-start token; preserve raw data elsewhere.

    The pinned OCR parser removes BOS but leaves decoder-start EOS attached to
    the first lexical label. Removing its known token position avoids changing
    punctuation or stripping marker-like strings inside actual generated text.
    """
    if tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 2 or int(tokens[0,0]) != decoder_start_id:
        raise ValueError('Florence generation must begin with its explicit decoder-start token')
    return processor.batch_decode(tokens[:,1:], skip_special_tokens=False)[0]


class FrozenInventory:
    def __init__(self, snapshot, settings, device):
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.settings = settings; self.device = device
        torch.cuda.reset_peak_memory_stats(device)
        self.processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=True)
        self.model, loading = AutoModelForCausalLM.from_pretrained(str(snapshot), local_files_only=True,
            trust_remote_code=True, torch_dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise RuntimeError('Florence checkpoint mismatch: '+str(loading))
        self.model.requires_grad_(False).eval().to(device)
        # Florence.generate delegates to language_model.generate; the outer
        # generation configuration has eos_token_id=None in this pinned model.
        eos = self.model.language_model.generation_config.eos_token_id
        self.eos_ids = [eos] if isinstance(eos, int) else eos
        if not self.eos_ids or not all(isinstance(x,int) for x in self.eos_ids):
            raise ValueError('Pinned Florence decoder must specify EOS token IDs')
        self.before = state_digest(self.model)
        self.provenance = {'snapshot': str(snapshot), 'revision': Path(snapshot).name, 'loading_info': loading,
            'precision': 'float32', 'device': device, 'batch_size': 1, 'model_state_sha256_before': self.before,
            'tasks': settings['tasks'], 'generation': settings['generation'], 'optimizer_steps': 0,
            'visual_embedding': 'L2-normalized mean of original-image _encode_image tokens; no OCR input',
            'processed_shapes': [], 'image_only_inventory': True}
        self.provenance['decoder_eos_token_ids'] = self.eos_ids
        self.decoder_start_id = self.model.language_model.generation_config.decoder_start_token_id
        if not isinstance(self.decoder_start_id,int): raise ValueError('Explicit Florence decoder-start ID required')
        self.provenance['decoder_start_token_id'] = self.decoder_start_id

    def infer(self, rgb):
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Inventory model must stay frozen/eval')
        result = {}; pixels = None; started = time.monotonic()
        for task in self.settings['tasks']:
            inputs = self.processor(text=task, images=Image.fromarray(rgb), return_tensors='pt')
            inputs = {k:v.to(self.device) for k,v in inputs.items()}; pixels = inputs['pixel_values']
            shape = list(pixels.shape)
            if shape not in self.provenance['processed_shapes']: self.provenance['processed_shapes'].append(shape)
            with torch.inference_mode():
                tokens = self.model.generate(**inputs, **self.settings['generation'])
            raw = self.processor.batch_decode(tokens, skip_special_tokens=False)[0]
            parser_input = florence_parser_input(self.processor,tokens,self.decoder_start_id)
            parsed = self.processor.post_process_generation(parser_input, task=task, image_size=(rgb.shape[1],rgb.shape[0]))
            if task not in parsed or not isinstance(parsed[task], dict): raise ValueError('Unexpected Florence task output')
            result[task] = {'raw': raw, 'parser_input': parser_input, 'parsed': parsed[task], 'token_ids': tokens[0].tolist(),
                'ended_with_eos': int(tokens[0,-1]) in self.eos_ids, 'generated_tokens': int(tokens.shape[1]-1)}
        with torch.inference_mode():
            hidden = self.model._encode_image(pixels)
            vector = torch.nn.functional.normalize(hidden.float().mean(dim=1), dim=-1)[0]
        if not torch.isfinite(vector).all(): raise RuntimeError('Nonfinite frozen visual representation')
        return {'tasks': result, 'embedding': vector.cpu().tolist(), 'rgb_sha256': array_sha(rgb),
            'seconds': time.monotonic()-started, 'ground_truth_used': False, 'ocr_predictions_given_to_model': False}

    def finish(self):
        after = state_digest(self.model)
        if after != self.before: raise RuntimeError('Inventory model state changed')
        return {**self.provenance, 'model_state_sha256_after': after, 'cuda_peak': _cuda_peak(self.device)}

    def close(self):
        self.model = None; gc.collect(); torch.cuda.empty_cache()


class FrozenLocalVerifier(FrozenResidualVQA):
    """Exact-word question on a line crop, same crop with target erased, and blank."""
    def verify(self, views, job, template):
        result = {}; started = time.monotonic()
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError('Local verifier must stay frozen/eval')
        for name, rgb in views.items():
            query, pixel = self.queries(rgb); result[name] = []
            for word in job['alternatives']:
                # Quoted data is never treated as a prompt instruction by our code.
                # A model may still be susceptible to text inside the question.
                import json
                prompt = template.format(word=json.dumps(word, ensure_ascii=False))
                answer, logits, encoded = self.answer(query, prompt)
                result[name].append(answer)
                if self.provenance['native_forward_check'] is None:
                    with torch.inference_mode():
                        native = self.model(pixel_values=pixel.repeat_interleave(2,dim=0),
                            input_ids=encoded['input_ids'].to(self.device).repeat_interleave(2,dim=0),
                            attention_mask=encoded['attention_mask'].to(self.device).repeat_interleave(2,dim=0),
                            labels=self.labels, return_dict=True).logits
                    error = float((native-logits).abs().max())
                    if not torch.allclose(native, logits, rtol=2e-4, atol=2e-4):
                        raise RuntimeError('Local staged verifier differs from native forward: '+str(error))
                    self.provenance['native_forward_check'] = {'status': 'PASS', 'max_absolute_logit_error': error,
                        'rtol': 2e-4, 'atol': 2e-4, 'source': 'transformers.Blip2ForConditionalGeneration.forward'}
        return {**job, 'answers': result, 'views_rgb_sha256': {k:array_sha(v) for k,v in views.items()},
            'seconds': time.monotonic()-started, 'ground_truth_used': False,
            'scores_are_calibrated_correctness_probabilities': False}
