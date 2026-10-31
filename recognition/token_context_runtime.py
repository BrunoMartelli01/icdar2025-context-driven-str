"""Frozen local CLIP dense tokens and BLIP2 transcript likelihood, FP32 only."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from recognition.token_context_router import context_views, patch_geometry, erased_views, transcript_chunks, likelihood_features
from recognition.residual_text_vqa import FrozenResidualVQA
from scripts.contextual_lm_verifier_runtime import state_digest, answer_log_probabilities
from scripts.crop_clip_runtime import _cuda_peak, array_sha


class FrozenDenseCLIP:
    def __init__(self, snapshot, settings, device):
        from transformers import CLIPModel, CLIPProcessor
        self.settings, self.device = settings, device; torch.cuda.reset_peak_memory_stats(device)
        self.model, loading = CLIPModel.from_pretrained(str(snapshot), local_files_only=True,
            use_safetensors=False, output_loading_info=True, torch_dtype=torch.float32)
        if any(loading.values()):
            raise RuntimeError('CLIP checkpoint incompatibility: '+str(loading))
        self.processor = CLIPProcessor.from_pretrained(str(snapshot), local_files_only=True)
        self.model.requires_grad_(False).eval().to(device); self.before = state_digest(self.model)
        self.grid = self.model.config.vision_config.image_size//self.model.config.vision_config.patch_size
        self.dimension = self.model.config.projection_dim; self.qualification = None
        if self.dimension != settings['clip_dim']:
            raise ValueError('Pinned CLIP projection dimension differs')
        self.metadata = {'snapshot': str(snapshot), 'loading_info': loading, 'device': device, 'precision': 'float32',
            'model_state_sha256_before': self.before, 'native_image_size': self.model.config.vision_config.image_size,
            'patch_size': self.model.config.vision_config.patch_size, 'patch_grid': self.grid, 'optimizer_steps': 0,
            'context_preprocessing': 'original raster/tile, symmetric square padding, then native processor',
            'patch_feature_definition': 'vision last_hidden_state, shared post_layernorm and visual_projection; not a pretrained patch correctness score'}

    @torch.inference_mode()
    def images(self, images, patches=False):
        if not images:
            return np.zeros((0, self.dimension), np.float32)
        result = []
        for start in range(0, len(images), self.settings['image_batch_size']):
            batch = images[start:start+self.settings['image_batch_size']]
            pixels = self.processor(images=[Image.fromarray(v) for v in batch], return_tensors='pt')['pixel_values'].to(self.device)
            values = self.model.vision_model(pixel_values=pixels, return_dict=True)
            pooled = F.normalize(self.model.visual_projection(values.pooler_output), dim=-1)
            if self.qualification is None:
                native = F.normalize(self.model.get_image_features(pixel_values=pixels), dim=-1)
                difference = float((native-pooled).abs().max())
                if not torch.allclose(native, pooled, rtol=self.settings['native_tolerance'], atol=self.settings['native_tolerance']):
                    raise RuntimeError('CLIP pooled projection differs from native forward')
                self.qualification = {'status': 'PASS', 'max_absolute_error': difference}
            value = (F.normalize(self.model.visual_projection(self.model.vision_model.post_layernorm(values.last_hidden_state[:, 1:])), dim=-1)
                     if patches else pooled)
            result.append(value.cpu().numpy())
        return np.concatenate(result).astype(np.float32)

    @torch.inference_mode()
    def text(self, words):
        result = np.zeros((len(words), self.dimension), np.float32)
        available = [i for i, w in enumerate(words) if w.strip()]
        for start in range(0, len(available), self.settings['text_batch_size']):
            indices = available[start:start+self.settings['text_batch_size']]
            texts = [self.settings['word_template'].format(word=json.dumps(words[i], ensure_ascii=False)) for i in indices]
            tokens = self.processor.tokenizer(texts, padding=True, truncation=False, return_tensors='pt')
            if tokens['input_ids'].shape[1] > self.model.config.text_config.max_position_embeddings:
                raise ValueError('One original OCR reading exceeds CLIP text capacity; no truncation')
            features = F.normalize(self.model.get_text_features(**{k: v.to(self.device) for k, v in tokens.items()}), dim=-1)
            result[indices] = features.cpu().numpy()
        return result

    def infer(self, rgb, rows):
        start = time.monotonic(); views = context_views(rgb, self.settings['tile_grid'], self.settings['padding_rgb'])
        patches = self.images([v['rgb'] for v in views], patches=True); geometry = []; valid = []; masses = []
        for v in views:
            g, m, a = patch_geometry(v, (rgb.shape[1], rgb.shape[0]), self.grid)
            geometry.append(g); valid.append(m); masses.append(a)
        crops = [np.ascontiguousarray(rgb[r['bbox'][1]:r['bbox'][3], r['bbox'][0]:r['bbox'][2]]) for r in rows]
        for row, crop in zip(rows, crops):
            if array_sha(crop) != row['crop_rgb_sha256']:
                raise ValueError('Frozen original word-crop pixels changed')
        values = {'patches': patches.reshape(-1, self.dimension), 'patch_geometry': np.concatenate(geometry),
            'patch_valid': np.concatenate(valid), 'patch_mass': np.concatenate(masses),
            'word_text': self.text([r['raw_text'] for r in rows]), 'word_image': self.images(crops)}
        return values, {'seconds': time.monotonic()-start, 'views': [{k: v[k] for k in ('bbox', 'pad_xy', 'side')} for v in views],
            'view_rgb_sha256': [array_sha(v['rgb']) for v in views], 'original_crop_hashes_checked': len(crops),
            'processed_shape': [min(len(views), self.settings['image_batch_size']), 3, self.model.config.vision_config.image_size, self.model.config.vision_config.image_size],
            'ground_truth_used': False}

    def finish(self):
        after = state_digest(self.model)
        if self.qualification is None or after != self.before:
            raise RuntimeError('Unqualified or modified CLIP extractor')
        return {**self.metadata, 'model_state_sha256_after': after, 'native_forward_check': self.qualification,
                'cuda_peak': _cuda_peak(self.device)}

    def close(self):
        self.model = None; gc.collect(); torch.cuda.empty_cache()


class FrozenTranscriptBLIP(FrozenResidualVQA):
    def __init__(self, snapshot, settings, device):
        self.settings = settings
        super().__init__(snapshot, device=device, plain_question=settings['prompt'], context_template='{caption}')
        self.provenance.update(task='actual OCR transcript teacher-forced likelihood; includes EOS',
            answer_score='sum logprob and mean per original target token, plus original-minus-erased contrast',
            language_batch_size=settings['language_batch_size'], probabilities_are_calibrated=False)
        self.provenance.pop('answer_ids')
        self.labels = None
        self.last_input_shapes = {}
        self.encoded_cache = {}; self.raw_query_dimension = None

    @torch.inference_mode()
    def visual(self, rgb):
        pixel = self.processor.image_processor(images=Image.fromarray(rgb), return_tensors='pt')['pixel_values'].to(self.device)
        self.last_input_shapes['vision'] = list(pixel.shape)
        emb = self.model.vision_model(pixel_values=pixel, return_dict=True).last_hidden_state
        query = self.model.qformer(query_embeds=self.model.query_tokens.expand(len(emb), -1, -1),
            encoder_hidden_states=emb, encoder_attention_mask=torch.ones(emb.shape[:-1], dtype=torch.long, device=self.device),
            return_dict=True).last_hidden_state
        self.raw_query_dimension = query.shape[-1]
        if query.shape[1:] != (self.settings['query_count'], self.settings['query_dim']):
            raise ValueError('BLIP2 query tensor differs from declared schema')
        return self.model.language_projection(query), pixel, query[0].cpu().numpy()

    @torch.inference_mode()
    def sequence(self, queries, pixel, texts):
        if not texts:
            return []
        prompt = self.settings['prompt']
        formatted = self.processor(images=Image.new('RGB', (self.settings['image_size'], self.settings['image_size'])),
            text=prompt, return_tensors='pt', truncation=False)
        ids = formatted['input_ids'].to(self.device); attention = formatted['attention_mask'].to(self.device)
        text = self.model.language_model.get_input_embeddings()(ids)
        index = getattr(self.model.config, 'image_token_index', None)
        if index is None:
            combined = torch.cat([queries, text], 1)
            mask = torch.cat([torch.ones(queries.shape[:2], dtype=torch.long, device=self.device), attention], 1)
        else:
            special = ids.eq(index)
            if int(special.sum()) != queries.shape[1]:
                raise RuntimeError('Native processor lost visual query slots')
            combined = text.masked_scatter(special[..., None].expand_as(text), queries.to(text.dtype)); mask = attention
        self.maximum_tokens = max(self.maximum_tokens, combined.shape[1])
        self.last_input_shapes['encoder'] = list(combined.shape)
        if combined.shape[1] > self.settings['maximum_input_tokens']:
            raise ValueError('BLIP prompt including visual queries exceeds explicit context budget')
        encoded = self.model.language_model.encoder(inputs_embeds=combined, attention_mask=mask, return_dict=True).last_hidden_state
        results = []
        for offset in range(0, len(texts), self.settings['language_batch_size']):
            group = texts[offset:offset+self.settings['language_batch_size']]
            tokenized = self.processor.tokenizer(group, padding=True, truncation=False, add_special_tokens=True, return_tensors='pt')
            labels = tokenized['input_ids'].to(self.device).masked_fill(tokenized['attention_mask'].to(self.device).eq(0), -100)
            if labels.shape[1] > self.settings['maximum_target_tokens']:
                raise ValueError('BLIP target exceeds configured budget; no truncation')
            self.last_input_shapes['target'] = list(labels.shape)
            logits = self.model.language_model(encoder_outputs=(encoded.repeat_interleave(len(group), 0),),
                attention_mask=mask.repeat_interleave(len(group), 0), labels=labels, use_cache=False, return_dict=True).logits
            totals = answer_log_probabilities(logits, labels)
            if self.provenance['native_forward_check'] is None:
                native = self.model(pixel_values=pixel.repeat_interleave(len(group), 0),
                    input_ids=ids.repeat_interleave(len(group), 0), attention_mask=attention.repeat_interleave(len(group), 0),
                    labels=labels, return_dict=True).logits
                error = float((native-logits).abs().max()); tolerance = self.settings['native_tolerance']
                if not torch.allclose(native, logits, rtol=tolerance, atol=tolerance):
                    raise RuntimeError('BLIP actual-transcript staged logits differ from native forward: '+str(error))
                self.provenance['native_forward_check'] = {'status': 'PASS', 'maximum_logit_difference': error, 'tolerance': tolerance}
            probs = logits.log_softmax(-1).gather(-1, labels.masked_fill(labels.eq(-100), 0)[..., None]).squeeze(-1)
            for i, target in enumerate(group):
                count = int(labels[i].ne(-100).sum()); target_ids = labels[i, :count].tolist()
                if target_ids[-1] != self.processor.tokenizer.eos_token_id:
                    raise ValueError('Actual transcript target must retain native EOS')
                results.append({'text': target, 'target_ids': target_ids, 'token_logprobs': probs[i, :count].tolist(),
                    'sum_logprob': float(totals[i]), 'mean_logprob': float(totals[i]/count), 'eos_logprob': float(probs[i, count-1])})
        return results

    def infer(self, rgb, rows):
        start = time.monotonic(); erased, local = erased_views(rgb, rows, self.settings['erasure_rgb'], self.settings['local_margin_height'])
        query, pixel, raw_queries = self.visual(rgb)
        eq, ep, _ = self.visual(erased)
        tokenize = lambda s: self.processor.tokenizer(s, add_special_tokens=True, truncation=False)['input_ids']
        chunks = transcript_chunks(rows, tokenize, self.settings['maximum_target_tokens'])
        texts = [r['text'] for r in chunks]
        original = self.sequence(query, pixel, texts); removed = self.sequence(eq, ep, texts)
        def contrast(a, b, extra):
            return {'original': a, 'erased': b, 'visual_gain': a['mean_logprob']-b['mean_logprob'],
                'eos_visual_gain': a['eos_logprob']-b['eos_logprob'], 'original_logprob_per_token': a['mean_logprob'], **extra}
        global_scores = [contrast(x, y, {'row_indices': c['row_indices']}) for x, y, c in zip(original, removed, chunks)]
        word_scores = []
        for row, view in zip(rows, local):
            if not row['raw_text'].strip():
                continue
            q, p, _ = self.visual(view['original']); hq, hp, _ = self.visual(view['erased'])
            original = self.sequence(q, p, [row['raw_text']])[0]; removed = self.sequence(hq, hp, [row['raw_text']])[0]
            word_scores.append(contrast(original, removed, {'candidate_id': row['candidate_id'], 'context_box': view['bbox'],
                'original_rgb_sha256': array_sha(view['original']), 'erased_rgb_sha256': array_sha(view['erased'])}))
        values = likelihood_features(global_scores, word_scores)
        return raw_queries.astype(np.float32), values, {'global_scores': global_scores, 'word_scores': word_scores,
            'global_original_rgb_sha256': array_sha(rgb), 'global_erased_rgb_sha256': array_sha(erased),
            'last_input_shapes': self.last_input_shapes.copy(), 'vision_batch_size': 1,
            'language_batch_size': self.settings['language_batch_size'], 'precision': 'float32',
            'seconds': time.monotonic()-start, 'ground_truth_used': False, 'original_word_outputs_changed': False}
