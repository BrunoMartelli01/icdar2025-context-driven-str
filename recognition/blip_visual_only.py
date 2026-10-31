"""Explicit BLIP2 visual/QFormer subset; the language model is never constructed."""
import json
from pathlib import Path

import torch
from torch import nn


class BLIPVisualOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.blip_2.modeling_blip_2 import Blip2VisionModel, Blip2QFormerModel
        self.vision_model = Blip2VisionModel(config.vision_config)
        self.qformer = Blip2QFormerModel(config.qformer_config)
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))

    @torch.inference_mode()
    def forward(self, pixels):
        visual = self.vision_model(pixel_values=pixels, return_dict=True).last_hidden_state
        return self.qformer(query_embeds=self.query_tokens.expand(len(visual), -1, -1),
            encoder_hidden_states=visual, encoder_attention_mask=torch.ones(visual.shape[:-1], dtype=torch.long,
            device=visual.device), return_dict=True).last_hidden_state


def load_visual_only(snapshot):
    """Select exact pinned native keys, require a complete strict match and FP32."""
    from safetensors import safe_open
    from transformers import Blip2Config, BlipImageProcessor
    snapshot = Path(snapshot)
    config = Blip2Config.from_pretrained(snapshot, local_files_only=True)
    with torch.device('meta'): model = BLIPVisualOnly(config)
    index = json.loads((snapshot/'model.safetensors.index.json').read_text())['weight_map']
    wanted = {k:v for k,v in index.items() if k.startswith(('vision_model.', 'qformer.')) or k=='query_tokens'}
    if set(wanted) != set(model.state_dict()):
        raise ValueError('Visual-only key set does not match native checkpoint exactly')
    state = {}
    for filename in sorted(set(wanted.values())):
        with safe_open(snapshot/filename, framework='pt', device='cpu') as source:
            for key in sorted(k for k,v in wanted.items() if v==filename):
                tensor = source.get_tensor(key)
                if tensor.dtype != torch.float32: raise ValueError('No implicit precision conversion is permitted')
                state[key] = tensor
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    return model, BlipImageProcessor.from_pretrained(snapshot, local_files_only=True)
