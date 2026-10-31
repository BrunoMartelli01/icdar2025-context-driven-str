"""Descriptive, GT-blind caption review against the exact Figure 4 scene."""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import statistics


def read(p):return json.loads(Path(p).read_text())


def repetition(text):
    words=[''.join(c for c in w.casefold() if c.isalnum()) for w in text.split()]
    triples=[tuple(words[i:i+3]) for i in range(max(0,len(words)-2))]
    return 1-len(set(triples))/len(triples) if triples else 0.


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args();plan=read(args.root/'resolved_configuration.json');protocol=read(args.protocol)
    captions={};groups={}
    for path in (args.root/'cache/captions').glob('*.json'):
        d=read(path);sid=d['contract']['sample']['sample_id'];captions[sid]=d['payload']['result']['raw_caption']
    if set(captions)!=set(plan['samples']):raise ValueError('Full caption population is required')
    for name,case in plan['cases'].items():
        if not name.endswith('__unet'):continue
        for split,ids in case['partitions'].items():
            n=[len(captions[s].split()) for s in ids];r=[repetition(captions[s]) for s in ids]
            groups[name.split('__')[0]+'__'+split]=dict(images=len(ids),minimum_words=min(n),maximum_words=max(n),
                mean_words=statistics.mean(n),median_words=statistics.median(n),
                words_40_to_80=sum(40<=v<=80 for v in n),
                repeated_trigrams_gt20pct=sum(v>.2 for v in r),mean_repeated_trigram_fraction=statistics.mean(r))
    ids=[protocol['matched_scene_id']]+[s for v in protocol['sample_ids'].values() for s in v]
    sample={s:dict(raw_caption=captions[s],words=len(captions[s].split()),repeated_trigram_fraction=repetition(captions[s]),
                  image_path=plan['samples'][s]['image_path']) for s in ids}
    # Semantic similarity on the same scene is descriptive, not factual verification.
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import torch
    torch.set_num_threads(plan['config']['cpu_threads'])
    model=SentenceTransformer(plan['assets']['mpnet']['snapshot'],device='cpu',local_files_only=True,
        model_kwargs={'torch_dtype':torch.float32});model.max_seq_length=384
    vectors=model.encode([protocol['paper_caption'],captions[protocol['matched_scene_id']]],
        convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=False).astype(np.float64)
    similarity=float(np.dot(*vectors)/(np.linalg.norm(vectors[0])*np.linalg.norm(vectors[1])))
    reference=dict(protocol,generated_same_scene=sample[protocol['matched_scene_id']],
        same_scene_mpnet_cosine=similarity,cosine_is_not_factual_or_style_verification=True,
        paper_caption_repeated_trigram_fraction=repetition(protocol['paper_caption']))
    from transformers import AutoTokenizer
    native_tokenizer=AutoTokenizer.from_pretrained(plan['assets']['blip2']['snapshot'],local_files_only=True)
    reference['paper_caption_tokens_with_candidate_blip2_tokenizer']=len(native_tokenizer(protocol['paper_caption'])['input_ids'])
    reference['token_length_scope']='Candidate tokenizer only; the author checkpoint/tokenizer remains unresolved.'
    result=dict(claim_label='approximate reproduction',caption_images=len(captions),groups=groups,
        reference=reference,samples=sample,human_factual_review='See caption_visual_review.json; automatic word/repetition counts do not verify factual grounding.',
        reviewer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output_root.mkdir(parents=True,exist_ok=False)
    with (args.output_root/'caption_review.json').open('x') as out:json.dump(result,out,indent=2,sort_keys=True)
    pieces=['<!doctype html><meta charset="utf-8"><title>Caption review</title>',
            '<style>body{font:17px system-ui;max-width:1100px;margin:40px auto}img{max-width:360px;max-height:400px}article{display:grid;grid-template-columns:380px 1fr;gap:24px;margin:30px 0;padding:20px;border:1px solid #aaa}p{line-height:1.5}</style>',
            '<h1>BLIP-2 captions: Figure 4 and GT-blind CAL sample</h1>',
            '<p>Approximate reproduction. Raw generated text, unchanged. Caption review does not select a router or optimize TEST results.</p>',
            '<h2>Paper Figure 4: 72 words</h2><p>'+html.escape(protocol['paper_caption'])+'</p>',
            '<p>Same-scene MPNet cosine: '+f'{similarity:.4f}'+'. Semantic overlap does not establish factual correctness or style equivalence.</p>']
    for sid,v in sample.items():
        encoded=base64.b64encode(Path(v['image_path']).read_bytes()).decode()
        pieces.append('<article><img src="data:image/jpeg;base64,'+encoded+'"><div><h2>'+html.escape(sid)+'</h2><p>'+html.escape(v['raw_caption'])+'</p><p>'+str(v['words'])+' words; repeated trigram fraction '+f"{v['repeated_trigram_fraction']:.3f}"+'</p></div></article>')
    (args.output_root/'caption_review.html').write_text('\n'.join(pieces))
    print(json.dumps(dict(caption_images=len(captions),same_scene_mpnet_cosine=similarity,reference_words=72,
                         generated_words=reference['generated_same_scene']['words'],groups=groups)))


if __name__=='__main__':main()
