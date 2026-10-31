import json

import pytest
import torch

from recognition.blip_visual_only import BLIPVisualOnly, load_visual_only


def tiny_snapshot(tmp_path, drop_key=False):
    from safetensors.torch import save_file
    from transformers import Blip2Config, BlipImageProcessor
    config=Blip2Config(vision_config=dict(hidden_size=8,intermediate_size=16,num_hidden_layers=1,
        num_attention_heads=2,image_size=8,patch_size=4),qformer_config=dict(hidden_size=8,
        intermediate_size=16,num_hidden_layers=2,num_attention_heads=2,encoder_hidden_size=8),num_query_tokens=2)
    config.save_pretrained(tmp_path);BlipImageProcessor(size=dict(height=8,width=8)).save_pretrained(tmp_path)
    torch.manual_seed(12);native=BLIPVisualOnly(config).eval();state=native.state_dict()
    if drop_key:state.pop(next(iter(state)))
    state['language_model.unused']=torch.ones(2)
    save_file(state,tmp_path/'model-00001-of-00001.safetensors')
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={k:'model-00001-of-00001.safetensors' for k in state})))
    return native


def test_strict_subset_exact_output_without_language_model(tmp_path):
    native=tiny_snapshot(tmp_path);thin,_=load_visual_only(tmp_path)
    pixels=torch.randn(1,3,8,8)
    assert torch.equal(native(pixels),thin(pixels))
    assert not hasattr(thin,'language_model') and not hasattr(thin,'language_projection')
    assert all(not p.requires_grad for p in thin.parameters())


def test_incomplete_native_subset_fails(tmp_path):
    tiny_snapshot(tmp_path,drop_key=True)
    with pytest.raises(ValueError,match='key set'):load_visual_only(tmp_path)
