"""Explicit approximate long scene captions with raw native answers preserved."""
from __future__ import annotations

from caption.paper_batch_blip2 import generate_batch
from caption.paper_style_blip2 import InstructionCaptionConfig
from scoring.paper_image_router import caption_quality


def detailed_profile(captioner, rgbs, profile, target):
    """Generate one paragraph or join image-only native aspect descriptions.

    No OCR, annotation, hand-authored scene fact, rewriting or length filler is
    appended. Model-family checks prevent InstructBLIP being labelled BLIP-2.
    """
    family = 'instructblip' if isinstance(captioner.config, InstructionCaptionConfig) else 'blip2'
    if (profile['caption_model'] != family or not profile['prompts']
            or profile['assembly'] not in ('single', 'paragraph_join')
            or profile['assembly'] == 'single' and len(profile['prompts']) != 1):
        raise ValueError('Explicit matching model family and native prompt assembly required')
    answers = [[] for _ in rgbs]
    for prompt in profile['prompts']:
        native_profile = dict(profile, assembly='single', prompts=[prompt])
        outputs = generate_batch(captioner, rgbs, native_profile, target)
        if len(outputs) != len(rgbs):
            raise RuntimeError('Native caption cardinality mismatch')
        for collected, output in zip(answers, outputs):
            collected.extend(output['answers'])
    results = []
    for native in answers:
        texts = [a['result']['raw_caption'] for a in native]
        text = texts[0] if profile['assembly'] == 'single' else ' '.join(
            t if t[-1] in '.!?' else t + '.' for t in texts)
        results.append(dict(raw_caption=text, answers=native, profile=profile,
            quality=caption_quality(text, target['min_words'], target['max_words']),
            image_only_captioning=True, parseq_in_prompt=False, ground_truth_in_caption=False,
            caption_family='InstructBLIP ablation' if family == 'instructblip' else 'BLIP-2',
            checkpoint_qualification='approximate reproduction', native_batch_size=len(rgbs),
            text_postprocessing='single: none; multi-aspect: join raw native answers with terminal punctuation only'))
    return results
