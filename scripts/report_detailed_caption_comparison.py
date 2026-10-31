"""Present complete Qwen/BLIP2/trained-router results without selecting on TEST."""
from __future__ import annotations

import argparse
import base64
import html
from pathlib import Path
import statistics

from scripts.review_paper_captions import repetition
from scripts.run_paper_caption_comparison import cache_path, load_cache, read, seal, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--baseline-root',type=Path,required=True);p.add_argument('--sample-protocol',type=Path,required=True)
    p.add_argument('--paper-reference',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);args=p.parse_args()
    plan=read(args.root/'resolved_configuration.json');new=read(args.root/'results.json')
    old=read(args.baseline_root/'results.json')
    baseline_plan=read(args.baseline_root/'resolved_configuration.json')
    plans={args.root:plan,args.baseline_root:baseline_plan}
    def caption(root,sid):
        payload=load_cache(root,plans[root],'captions',plans[root]['samples'][sid])
        if payload is None:raise FileNotFoundError('Missing sealed caption: '+sid)
        return payload
    audits={name:read(args.root/name) for name in ('independent_audit.json','native_audit.json')}
    assert all(a['status']=='PASS' for a in audits.values())
    assert new['result_class']=='approximate reproduction' and not new['caption_model_is_paper_blip2']
    assert new['metric']==old['metric']
    before={(r['case'],r['method']):r for r in old['table']};after={(r['case'],r['method']):r for r in new['table']}
    rows=[];controls=[];quality={}
    for name,case in sorted(plan['cases'].items()):
        assert before[name,'trained_lightweight']==after[name,'trained_lightweight']
        trained=after[name,'trained_lightweight'];b=before[name,'paper_selected__matched_budget']
        q=after[name,'paper_selected__matched_budget'];t=after[name,'joined_t3__matched_budget']
        assert len({(r['deepsolo_images'],r['total_images']) for r in [trained,b,q,t]})==1
        rows.append(dict(case=name,blip2_t1_t3_f1=b['f1_percent'],qwen_t1_t3_f1=q['f1_percent'],
            qwen_t3_only_f1=t['f1_percent'],trained_f1=trained['f1_percent'],
            qwen_minus_blip2_pp=q['f1_percent']-b['f1_percent'],
            deepsolo_images=q['deepsolo_images'],total_images=q['total_images'],deepsolo_percentage=q['deepsolo_percentage']))
        controls.extend(after[name,method] for method in ('paper_selected__tau0.8','joined_t3__tau0.8','all_deepsolo','all_light'))
        dataset=name.split('__')[0]
        if dataset not in quality:
            stats={}
            for label,root in [('blip2',args.baseline_root),('qwen',args.root)]:
                texts=[caption(root,sid)['result']['raw_caption'] for sid in case['partitions']['test']]
                lengths=[len(text.split()) for text in texts];repeats=[repetition(text) for text in texts]
                stats[label]=dict(images=len(texts),mean_words=statistics.mean(lengths),median_words=statistics.median(lengths),
                    minimum_words=min(lengths),maximum_words=max(lengths),mean_repeated_trigram_fraction=statistics.mean(repeats),
                    repeated_trigram_fraction_above_0_2=sum(x>.2 for x in repeats))
            quality[dataset]=stats
    protocol=read(args.sample_protocol);review=[]
    for role in ('development','qualification','paper_example'):
        ids=protocol['roles']['paper_example'] if role=='paper_example' else [s for v in protocol['roles'].values() if isinstance(v,dict) for s in v[role]]
        for sid in ids:
            path=cache_path(args.root,'captions',sid)
            if not path.exists():raise FileNotFoundError('Missing planned caption review sample: '+sid)
            d=read(path);assert d['payload']==caption(args.root,sid)
            b=caption(args.baseline_root,sid)
            review.append(dict(sample_id=sid,role=role,image_path=plan['samples'][sid]['image_path'],
                baseline=b['result']['raw_caption'],caption=d['payload']['result']['raw_caption'],
                actual_batch_ids=d['payload']['shared_batch_ids'],source=str(path),source_sha256=sha(path)))
    reference=read(args.paper_reference);figure=next(r for r in review if r['role']=='paper_example')
    assert reference['matched_scene_id']==figure['sample_id']
    import torch
    from sentence_transformers import SentenceTransformer
    from scoring.semantic import MPNetApproxConfig,MPNetSemanticSimilarity
    torch.set_num_threads(plan['config']['cpu_threads'])
    model=SentenceTransformer(plan['assets']['mpnet']['snapshot'],device='cpu',local_files_only=True,
        model_kwargs={'torch_dtype':torch.float32})
    semantic=MPNetSemanticSimilarity(MPNetApproxConfig(device='cpu',precision='float32'),model=model)
    old_similarity,new_similarity=semantic.compare_candidates(figure['baseline'],figure['caption'],reference['paper_caption'])
    reference_comparison=dict(paper_words=reference['paper_caption_words'],full_run_words=len(figure['caption'].split()),
        old_blip2_cosine=old_similarity,new_qwen_cosine=new_similarity,
        semantic_model=plan['assets']['mpnet']['model_id'],semantic_revision=plan['assets']['mpnet']['revision'],
        scope='Text semantic overlap only; not factuality or unknown-author equivalence')
    args.output_root.mkdir(parents=True,exist_ok=False)
    summary=dict(status='COMPLETE_WITH_LIMITATIONS',claim_label='approximate reproduction',metric=new['metric'],
        caption_model_is_paper_blip2=False,matched_budget_rows=rows,literal_and_expert_controls=controls,
        caption_quality_test_statistics=quality,native_caption_review=review,
        paper_figure4_comparison=reference_comparison,
        total_unique_test_images=sum(v['qwen']['images'] for v in quality.values()),
        total_generated_captions=len(plan['samples']),new_deepsolo_neural_inferences=0,
        logical_cached_routing=True,online_latency_measured=False,source_sha256={
            str(args.root/'results.json'):sha(args.root/'results.json'),str(args.baseline_root/'results.json'):sha(args.baseline_root/'results.json'),
            **{str(args.root/n):sha(args.root/n) for n in audits},str(args.sample_protocol):sha(args.sample_protocol),
            str(args.paper_reference):sha(args.paper_reference)},
        scope='Frozen development-selected caption model/profile ablation; complete previously explored TEST sets; no strict-author or population-superiority claim')
    seal(args.output_root/'summary.json',summary)
    lines=['# Detailed caption router comparison','',
        'Approximate reproduction. Qwen2-VL-7B is an explicit caption ablation, not the unknown paper BLIP2 configuration. Common29/10 experts, lexical corrections, T1 readings, trained-router decisions and DeepSolo outputs are frozen.','',
        'Metric: '+new['metric']+'. Every image remains in the denominator. Router counts refer to logical cached decisions; no new DeepSolo neural inference or online end-to-end latency measurement.','',
        '| Dataset/expert | BLIP2 T1/T3 F1 % | Qwen T1/T3 F1 % | Qwen T3-only F1 % | Trained router F1 % | DeepSolo images/total (%), each gate |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for r in rows:lines.append(f"| {r['case']} | {r['blip2_t1_t3_f1']:.4f} | {r['qwen_t1_t3_f1']:.4f} | {r['qwen_t3_only_f1']:.4f} | {r['trained_f1']:.4f} | {r['deepsolo_images']}/{r['total_images']} ({r['deepsolo_percentage']:.2f}%) |")
    lines+=['','The exact matched budget ranks an entire batch with RGB-hash ties and no TEST labels; it changes the paper threshold policy. CAL-only transferred scalar cuts and complete TP/FP/FN remain in the original results JSON. These point differences do not establish statistical or population superiority.','',
        '| Dataset/expert | Literal paper tau0.8 F1 % | DeepSolo images/total (%) |','| --- | ---: | --- |']
    for r in controls:
        if r['method']=='paper_selected__tau0.8':lines.append(f"| {r['case']} | {r['f1_percent']:.4f} | {r['deepsolo_images']}/{r['total_images']} ({r['deepsolo_percentage']:.2f}%) |")
    lines+=['','Caption statistics below measure length/repetition, not factual correctness. All TEST captions are included; sample-level direct visual errors and selection limitations remain documented.','',
        '| Dataset | Captioner | Images | Mean words | Mean repeated-trigram fraction | Fraction >0.20 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for dataset,stats in quality.items():
        for label,s in stats.items():lines.append(f"| {dataset} | {label} | {s['images']} | {s['mean_words']:.2f} | {s['mean_repeated_trigram_fraction']:.5f} | {s['repeated_trigram_fraction_above_0_2']}/{s['images']} |")
    lines+=['','Actual full-run Figure4 caption ('+str(len(figure['caption'].split()))+' words), unchanged native text:','', '> '+figure['caption'],'',
        f"MPNet text overlap with the72-word paper example: old BLIP2 {old_similarity:.6f}, full-run Qwen {new_similarity:.6f}. This is semantic overlap, not a factuality certificate or an identification of the author checkpoint.",'',
        'Both independent audits pass: full native-token/RGB/reused-T1 verification and independent word-count/routing/full-MPNet reencoding. Hardware, raw generation tokens, memory peaks, model/source hashes and every per-image output are retained. Native batch arithmetic changes captions; no scalar/offload equivalence is presumed. Caption factuality errors, the unknown author model and historical TEST exposure limit interpretation.']
    (args.output_root/'summary.md').write_text('\n'.join(lines)+'\n')
    body=['<!doctype html><meta charset="utf-8"><title>Full-run native caption review</title>',
        '<style>body{font:17px system-ui;max-width:1300px;margin:40px auto;line-height:1.5}article{border-top:2px solid #aaa;padding:25px 0;display:grid;grid-template-columns:300px 1fr 1fr;gap:25px}img{max-width:300px;max-height:400px}h2{grid-column:1/-1}</style>',
        '<h1>Actual full-run captions: original BLIP2 and Qwen ablation</h1><p>Unchanged native text. Fixed9 development,9 CAL,1 paper diagnostic images; original Figure4 caption72 words. These examples do not establish population factuality or unknown-author equivalence.</p>']
    for r in sorted(review,key=lambda r:(r['role']!='paper_example',r['role'],r['sample_id'])):
        data=base64.b64encode(Path(r['image_path']).read_bytes()).decode()
        body.append('<article><h2>'+html.escape(r['sample_id'])+' — '+r['role']+'</h2><img src="data:image/jpeg;base64,'+data+'"><div><strong>Original BLIP2</strong><p>'+html.escape(r['baseline'])+'</p></div><div><strong>Full-run Qwen ('+str(len(r['caption'].split()))+' words)</strong><p>'+html.escape(r['caption'])+'</p></div></article>')
    (args.output_root/'caption_review.html').write_text('\n'.join(body))
    print(args.output_root/'summary.md')


if __name__=='__main__':main()
