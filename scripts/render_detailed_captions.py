"""Standalone original-versus-improved caption review with unmodified text."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import statistics

from scripts.review_paper_captions import repetition


def read(p):return json.loads(Path(p).read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--profile',required=True);p.add_argument('--baseline-root',type=Path,required=True)
    p.add_argument('--qualification-root',type=Path);p.add_argument('--paper-root',type=Path)
    p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
    records=[]
    for role in ('development','qualification','paper_example'):
        source_root=(args.qualification_root or args.root) if role=='qualification' else (args.paper_root or args.root) if role=='paper_example' else args.root
        for path in sorted((source_root/args.profile/role).glob('*.json')):
            d=read(path);sid=d['sample']['sample_id'];base=args.baseline_root/'cache/captions'/(hashlib.sha256(sid.encode()).hexdigest()+'.json')
            old=read(base)['payload']['result']['raw_caption'];new=d['result']['raw_caption']
            records.append(dict(sample_id=sid,role=role,image_path=d['sample']['image_path'],baseline=old,improved=new,
                baseline_words=len(old.split()),improved_words=len(new.split()),baseline_repetition=repetition(old),improved_repetition=repetition(new),
                gpu_weight_budget=d['profile'].get('gpu_weight_budget'),
                placement=d['result']['answers'][0]['result'].get('placement'),
                native_batch_size=d['result'].get('native_batch_size'),
                generated_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),generated_source=str(path)))
    if not records:raise ValueError('No generated captions to render')
    groups={}
    protocol=read(read(args.root/'plan_development.json')['config']['sample_protocol'])
    for role in ('development','qualification','paper_example'):
        rows=[r for r in records if r['role']==role]
        planned=(len(protocol['roles']['paper_example']) if role=='paper_example' else
            sum(len(v[role]) for v in protocol['roles'].values() if isinstance(v,dict)))
        if rows:groups[role]=dict(images=len(rows),planned_images=planned,complete=len(rows)==planned,
            baseline_mean_words=statistics.mean(r['baseline_words'] for r in rows),
            improved_mean_words=statistics.mean(r['improved_words'] for r in rows),
            baseline_mean_repetition=statistics.mean(r['baseline_repetition'] for r in rows),
            improved_mean_repetition=statistics.mean(r['improved_repetition'] for r in rows))
    args.output_root.mkdir(parents=True,exist_ok=False)
    result=dict(claim_label='approximate reproduction',selected_profile=args.profile,groups=groups,records=records,
        scope='Caption-quality review only; no new recognition/router F1 is claimed')
    (args.output_root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    sections=['<!doctype html><meta charset="utf-8"><title>Detailed caption comparison</title>',
        '<style>body{font:17px system-ui;max-width:1250px;margin:40px auto;line-height:1.5}article{border-top:2px solid #aaa;padding:25px 0;display:grid;grid-template-columns:300px 1fr 1fr;gap:25px}img{max-width:300px;max-height:400px}h2{grid-column:1/-1}.old{color:#555}</style>',
        '<h1>Detailed captions: native outputs and original baseline</h1>',
        '<p>Explicit Qwen2-VL model ablation. Captions are unchanged native generated text. Selection uses development images; CAL and paper example are separate quality checks. This report contains no new recognition F1.</p>']
    sections.append('<p>Coverage: '+html.escape('; '.join(f"{role}: {g['images']}/{g['planned_images']}" for role,g in groups.items()))+'.</p>')
    for r in sorted(records,key=lambda r:(r['role']!='paper_example',r['role'],r['sample_id'])):
        data=base64.b64encode(Path(r['image_path']).read_bytes()).decode()
        sections.append('<article><h2>'+html.escape(r['sample_id'])+' — '+r['role']+'</h2><img src="data:image/jpeg;base64,'+data+'"><div class="old"><strong>Original BLIP-2: '+str(r['baseline_words'])+' words</strong><p>'+html.escape(r['baseline'])+'</p></div><div><strong>Selected caption: '+str(r['improved_words'])+' words</strong><p>'+html.escape(r['improved'])+'</p></div></article>')
    (args.output_root/'comparison.html').write_text('\n'.join(sections))
    print(json.dumps(groups))


if __name__=='__main__':main()
