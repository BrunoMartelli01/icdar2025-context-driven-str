"""Explicit preprocessing for the isolated, literature-motivated CLIP experiment.

SigLIP2 lowercases text before its Gemma tokenizer (paper section 2.1). DFN's
published OpenCLIP config specifies square resize, normalization and bicubic.
No guessed processor, mutable model revision or non-strict checkpoint loading.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor

from scripts.run_clip_router_prompt_ablation import Encoder as FrozenEncoder


class LowercaseTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer=tokenizer

    def __call__(self, text, **kwargs):
        if isinstance(text,str):
            text=text.lower()
        elif isinstance(text,list) and all(isinstance(t,str) for t in text):
            text=[t.lower() for t in text]
        else:
            raise ValueError('Expected raw strings for SigLIP2 tokenizer')
        return self.tokenizer(text,**kwargs)


class PublishedSquashProcessor:
    """Reconstruct only the explicitly supported native DFN processor."""
    def __init__(self, native_config, image_size):
        cfg=native_config['preprocess_cfg']
        if (cfg['resize_mode']!='squash' or cfg['interpolation']!='bicubic'
                or native_config['model_cfg']['vision_cfg']['image_size']!=image_size
                or len(cfg['mean'])!=3 or len(cfg['std'])!=3
                or not np.isfinite(cfg['mean']).all() or not np.isfinite(cfg['std']).all()
                or any(v<=0 for v in cfg['std'])):
            raise ValueError('Unsupported published processor; refusing a guessed fallback')
        self.transform=Compose([Resize((image_size,image_size),InterpolationMode.BICUBIC),
                                ToTensor(),Normalize(cfg['mean'],cfg['std'])])

    def __call__(self, *, images, return_tensors, input_data_format):
        if return_tensors!='pt' or input_data_format!='channels_last':
            raise ValueError('Expected explicit RGB tensor processing')
        if not images or any(not isinstance(i,Image.Image) or i.mode!='RGB' for i in images):
            raise ValueError('Expected nonempty PIL RGB images')
        return {'pixel_values':torch.stack([self.transform(i) for i in images])}


def validate_native_architecture(native, config, spec, tokenizer):
    """Check the HF conversion against the publisher's native dimensions/activation."""
    model=native['model_cfg'];vision=model['vision_cfg'];text=model['text_cfg']
    pairs=[(model['embed_dim'],config.projection_dim),
        (vision['image_size'],config.vision_config.image_size),
        (vision['width'],config.vision_config.hidden_size),
        (vision['layers'],config.vision_config.num_hidden_layers),
        (vision['patch_size'],config.vision_config.patch_size),
        (text['context_length'],config.text_config.max_position_embeddings),
        (text['width'],config.text_config.hidden_size),
        (text['heads'],config.text_config.num_attention_heads),
        (text['layers'],config.text_config.num_hidden_layers)]
    if any(a!=b for a,b in pairs):
        raise ValueError('Published native/HF model dimensions disagree')
    expected='quick_gelu' if model.get('quick_gelu',False) else 'gelu'
    if config.text_config.hidden_act!=expected or config.vision_config.hidden_act!=expected:
        raise ValueError('Published native/HF activation disagrees')
    head_width=vision.get('head_width',spec['native_vision_head_width'])
    if (head_width!=spec['native_vision_head_width'] or vision['width']%head_width
            or vision['width']//head_width!=config.vision_config.num_attention_heads):
        raise ValueError('Published native/HF vision attention heads disagree')
    native_vocab=text['vocab_size'];hf_text=config.text_config
    if spec['native_vocab_policy']=='identical':
        if hf_text.vocab_size!=native_vocab:
            raise ValueError('Published native/HF vocabulary disagrees')
    elif spec['native_vocab_policy']=='publisher_extra_unused_pad_row':
        # The pinned DFN HF conversion has one extra embedding row. The published
        # tokenizer still pads with EOS, so that row must never enter a sequence.
        if (hf_text.vocab_size!=native_vocab+1 or hf_text.pad_token_id!=native_vocab
                or hf_text.eos_token_id!=native_vocab-1):
            raise ValueError('Unexpected published extra-padding conversion')
    else:
        raise ValueError('Explicit native vocabulary policy required')
    if (len(tokenizer)!=native_vocab or tokenizer.bos_token_id!=native_vocab-2
            or tokenizer.eos_token_id!=native_vocab-1 or tokenizer.pad_token_id!=native_vocab-1):
        raise ValueError('Published CLIP tokenizer special IDs/vocabulary disagree')
    # HF's legacy eos=2 branch pools at the first largest token ID; the modern
    # branch explicitly pools at EOS. Both must select the publisher's EOS.
    if hf_text.eos_token_id not in (2,tokenizer.eos_token_id):
        raise ValueError('Model would pool at a different EOS token')
    probe=tokenizer(['text','a longer written word'],return_tensors='pt',padding=True,
                    add_special_tokens=True,truncation=False)
    if 'attention_mask' not in probe:
        raise ValueError('Published CLIP padding must provide an attention mask')
    for ids,mask in zip(probe['input_ids'],probe['attention_mask']):
        active=ids[mask.bool()]
        if (not len(active) or active[0]!=tokenizer.bos_token_id
                or active[-1]!=tokenizer.eos_token_id or int(ids.max())>=native_vocab
                or torch.any(ids[~mask.bool()]!=tokenizer.pad_token_id)):
            raise ValueError('Published CLIP tokenizer sequence/padding disagrees')


class LiteratureEncoder(FrozenEncoder):
    def __init__(self, identity, device, batch_size):
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
        directory=Path(identity['local_path']);self.device=device;self.batch_size=batch_size
        self.model,info=AutoModel.from_pretrained(directory,local_files_only=True,
            use_safetensors=identity['weight_file'].endswith('safetensors'),output_loading_info=True)
        if any(info.values()):
            raise ValueError('Checkpoint key mismatch: '+str(info))
        # Validate all metadata before moving a potentially large model to CUDA.
        self.backend=identity['backend'];self.limit=self.model.config.text_config.max_position_embeddings
        self.image_size=self.model.config.vision_config.image_size
        if self.image_size!=identity['image_size']:
            raise ValueError('Published/model input resolution differs')
        native=None
        self.tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True,use_fast=True)
        if identity.get('native_config'):
            native=json.loads((directory/identity['native_config']).read_text())
            validate_native_architecture(native,self.model.config,identity,self.tokenizer)
        if identity['text_preprocessing']=='lowercase_before_tokenizer':
            if self.backend!='siglip':raise ValueError('Lowercase policy reserved for SigLIP2')
            if (self.tokenizer.padding_side!='right' or self.tokenizer.add_bos_token
                    or not self.tokenizer.add_eos_token):
                raise ValueError('SigLIP2 requires published right padding, EOS and no BOS')
            self.tokenizer=LowercaseTokenizer(self.tokenizer)
        elif identity['text_preprocessing']!='native_tokenizer':
            raise ValueError('Unsupported text preprocessing')
        if identity['image_preprocessing']=='published_native_squash':
            if native is None:raise ValueError('Native processor evidence required')
            self.processor=PublishedSquashProcessor(native,self.image_size)
        elif identity['image_preprocessing']=='published_hf_processor':
            self.processor=AutoImageProcessor.from_pretrained(directory,local_files_only=True)
        else:
            raise ValueError('Unsupported image preprocessing')
        self.model=self.model.eval().float().to(device);self.cache={}
