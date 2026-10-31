"""Independent native-token/RGB/T1-reuse audit for the Qwen router ablation."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.run_paper_caption_comparison import cache_path, digest, read, seal, sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    plan = read(args.root / 'resolved_configuration.json'); c = plan['config']; profile = c['caption_profile']
    assert plan['mode'] == 'approximate' and plan['caption_model_is_paper_blip2'] is False
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(plan['assets']['qwen']['snapshot'], local_files_only=True)
    captions = 0; reused = 0; words = []; peaks = dict(allocated=0, reserved=0)
    for sid, sample in plan['samples'].items():
        rgb = np.array(Image.open(sample['image_path']).convert('RGB'))
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == sample['rgb_sha256']
        wrapper = read(cache_path(args.root, 'captions', sid)); payload = wrapper['payload']; result = payload['result']
        assert wrapper['payload_sha256'] == digest(payload)
        assert result['caption_family'] == 'Qwen2-VL-7B caption ablation' and result['image_only_captioning']
        assert not result['ground_truth_in_caption'] and not result['parseq_in_prompt']
        assert result['profile'] == profile and len(result['answers']) == 1
        native = result['answers'][0]['result']
        assert native['generation_config'] == profile['generation']
        assert native['prompt'] == profile['prompt'] and native['dtype'] == 'bfloat16'
        assert native['placement'] == profile['placement'] and native['quantization'] == 'none'
        assert native['attention_implementation'] == 'sdpa' and native['padding_side'] == 'left'
        assert 0 < len(native['generated_token_ids']) <= profile['generation']['max_new_tokens']
        text = processor.tokenizer.decode(native['generated_token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        assert text == result['raw_caption'] == native['raw_caption'] and text
        chat = processor.apply_chat_template([dict(role='user', content=[dict(type='image'), dict(type='text', text=profile['prompt'])])], tokenize=False, add_generation_prompt=True)
        grid = native['image_grid_thw']; assert len(grid) == 1
        tokens = np.prod(grid[0]) / 4
        assert profile['min_pixels'] / 28**2 <= tokens <= profile['max_pixels'] / 28**2
        # Native Qwen processing expands its mutable batch text list in place.
        # The saved chat therefore contains one image marker per visual token.
        assert tokens == int(tokens)
        expanded_chat = chat.replace('<|image_pad|>', '<|image_pad|>' * int(tokens))
        assert expanded_chat == native['native_chat_prompt']
        group = payload['shared_batch_ids']; index = group.index(sid)
        assert native['batch_image_grids'][index] == grid[0]
        assert len(group) == payload['batch_size'] == result['native_batch_size']
        assert len(text.split()) == result['quality']['words']
        for name, key in [('allocated', 'peak_allocated_bytes'), ('reserved', 'peak_reserved_bytes')]:
            assert native[key] == payload[key] > 0
            peaks[name] = max(peaks[name], native[key])
        words.append(len(text.split())); captions += 1
        old = read(cache_path(args.root, 't1', sid)); origin = old['reuse_provenance']; source = Path(origin['source'])
        assert origin['neural_inference_rerun'] is False and sha(source) == origin['source_sha256']
        original = read(source); assert old['payload'] == original['payload']
        assert original['payload_sha256'] == digest(original['payload'])
        assert original['contract']['sample'] == sample
        reused += 1
    seal(args.output, dict(status='PASS', claim_label='approximate reproduction',
        captions_with_native_token_replay=captions, independently_checked_RGBs=captions,
        unchanged_T1_sources_checked=reused, caption_model_is_paper_blip2=False,
        minimum_words=min(words), maximum_words=max(words), mean_words=sum(words)/len(words),
        gpu_peaks=peaks, auditor_sha256=sha(__file__),
        resolved_configuration_sha256=sha(args.root/'resolved_configuration.json'),
        scope='Native caption, image and reuse provenance; run the independent routing/count/semantic auditor separately'))
    print(args.output)


if __name__ == '__main__': main()
